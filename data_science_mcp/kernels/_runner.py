#!/usr/bin/python
"""Trusted kernel supervisor executed inside the configured outer sandbox.

Untrusted candidate source runs only in a separate candidate interpreter behind
an authenticated broker. This process owns reference execution, elapsed timing,
correctness checks, and the only authenticated result accepted by the host.
"""

from __future__ import annotations

import base64
import binascii
import math
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

import numpy as np

from data_science_mcp.kernels._protocol import (
    AUTH_KEY_BYTES,
    MAX_CONTROL_LINE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_array,
    encode_array,
    encode_line,
    loads_json,
    read_bounded_line,
    require_exact_keys,
    signed_message,
    verify_signed_message,
)
from data_science_mcp.kernels.kernel_tasks import get_kernel_task

_REPEATS = 5
_MAX_CANDIDATE_BYTES = 256 * 1024
_SUPERVISOR_REQUEST_LIMIT_BYTES = 2048
_MAX_WORKER_OUTPUT_BYTES = 32 * 1024 * 1024
_MAX_WORKER_STDERR_BYTES = 256 * 1024
_REQUEST_ID_RE = re.compile(r"[0-9a-f]{32}")
_WORKER_ERROR_RE = re.compile(
    r"(?:compile/exec|runtime) failed: [A-Za-z_][A-Za-z0-9_]{0,63}"
)
_ALLOWED_WORKER_ERRORS = frozenset(
    {
        "candidate execution failed",
        "candidate exited",
        "candidate output limit exceeded",
        "candidate protocol violation",
        "candidate timeout",
        "missing entrypoint",
    }
)
_ALLOWED_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "DYLD_LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
)


class WorkerFailure(RuntimeError):
    """A sanitized failure reported by the trusted candidate broker."""


class _WorkerLineReader:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        terminate: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._terminate = terminate
        self._items: queue.Queue[bytes | None] = queue.Queue()
        self._failure: str | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pending = bytearray()
        total = 0
        try:
            while chunk := self._stream.read(8192):
                total += len(chunk)
                pending.extend(chunk)
                if (
                    total > _MAX_WORKER_OUTPUT_BYTES
                    or len(pending) > MAX_CONTROL_LINE_BYTES
                ):
                    self._failure = "worker output limit exceeded"
                    self._terminate()
                    return
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    self._items.put(bytes(pending[:newline]))
                    del pending[: newline + 1]
            if pending:
                self._failure = "worker protocol violation"
        except (OSError, ValueError):
            self._failure = "worker protocol violation"
        finally:
            self._items.put(None)

    def receive(self, timeout_s: float) -> bytes:
        try:
            item = self._items.get(timeout=timeout_s)
        except queue.Empty as exc:
            self._terminate()
            raise WorkerFailure("candidate timeout") from exc
        if item is None:
            raise WorkerFailure(self._failure or "candidate exited")
        return item

    def close(self) -> None:
        self._thread.join(timeout=1.0)


class _WorkerStderrReader:
    def __init__(self, stream: BinaryIO, *, terminate: Callable[[], None]) -> None:
        self._stream = stream
        self._terminate = terminate
        self._overflow = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        total = 0
        try:
            while chunk := self._stream.read(8192):
                total += len(chunk)
                if total > _MAX_WORKER_STDERR_BYTES:
                    self._overflow = True
                    self._terminate()
                    return
        except (OSError, ValueError):
            return

    def check(self) -> None:
        if self._overflow:
            raise WorkerFailure("candidate output limit exceeded")

    def close(self) -> None:
        self._thread.join(timeout=1.0)


def _worker_environment() -> dict[str, str]:
    environment = {name: os.environ[name] for name in _ALLOWED_ENV if name in os.environ}
    environment.update(
        {
            "HOME": os.devnull,
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _safe_worker_error(value: object) -> str:
    if isinstance(value, str) and (
        value in _ALLOWED_WORKER_ERRORS or _WORKER_ERROR_RE.fullmatch(value)
    ):
        return value
    return "candidate execution failed"


class WorkerClient:
    """Authenticated client for the trusted broker process."""

    def __init__(
        self,
        candidate_path: Path,
        entrypoint: str,
        *,
        call_timeout_s: float,
        cwd: str,
    ) -> None:
        self._key = secrets.token_bytes(AUTH_KEY_BYTES)
        self._call_timeout_s = call_timeout_s
        self._sequence = 0
        script = Path(__file__).with_name("_worker.py").resolve()
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                str(script),
                str(candidate_path),
                entrypoint,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_worker_environment(),
            close_fds=True,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout = _WorkerLineReader(
            self._proc.stdout,
            terminate=self._terminate,
        )
        self._stderr = _WorkerStderrReader(
            self._proc.stderr,
            terminate=self._terminate,
        )
        bootstrap = {
            "call_timeout_s": call_timeout_s,
            "key": base64.b64encode(self._key).decode("ascii"),
            "version": PROTOCOL_VERSION,
        }
        try:
            self._write(encode_line(bootstrap, limit=1024))
            response = self._read_response(0, None)
            if (
                response["status"] != "ready"
                or response["result"] is not None
                or response["error"] is not None
            ):
                if response["status"] == "error":
                    raise WorkerFailure(_safe_worker_error(response["error"]))
                raise WorkerFailure("candidate protocol violation")
        except (ProtocolError, WorkerFailure):
            self.close()
            raise

    def _terminate(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._proc.kill()
        except (OSError, ProcessLookupError):
            return

    def _write(self, payload: bytes) -> None:
        if self._proc.stdin is None or self._proc.poll() is not None:
            raise WorkerFailure("candidate exited")
        try:
            self._proc.stdin.write(payload)
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise WorkerFailure("candidate exited") from exc

    def _read_response(
        self,
        sequence: int,
        nonce: str | None,
    ) -> dict[str, object]:
        raw = self._stdout.receive(self._call_timeout_s)
        self._stderr.check()
        try:
            body = require_exact_keys(
                verify_signed_message(loads_json(raw), self._key),
                {"error", "nonce", "result", "seq", "status", "version"},
            )
        except ProtocolError as exc:
            self._terminate()
            raise WorkerFailure("candidate protocol violation") from exc
        response_version = body["version"]
        response_sequence = body["seq"]
        if (
            isinstance(response_version, bool)
            or not isinstance(response_version, int)
            or response_version != PROTOCOL_VERSION
            or isinstance(response_sequence, bool)
            or not isinstance(response_sequence, int)
            or response_sequence != sequence
            or body["nonce"] != nonce
            or not isinstance(body["status"], str)
            or body["status"] not in {"ready", "ok", "error", "closed"}
        ):
            self._terminate()
            raise WorkerFailure("candidate protocol violation")
        return body

    def call(self, arguments: list[dict[str, object]]) -> dict[str, object]:
        self._sequence += 1
        nonce = secrets.token_hex(16)
        body = {
            "args": arguments,
            "nonce": nonce,
            "op": "call",
            "seq": self._sequence,
            "version": PROTOCOL_VERSION,
        }
        self._write(encode_line(signed_message(body, self._key)))
        response = self._read_response(self._sequence, nonce)
        if (
            response["status"] == "ok"
            and response["result"] is not None
            and response["error"] is None
        ):
            if not isinstance(response["result"], dict):
                raise WorkerFailure("candidate protocol violation")
            return response["result"]
        if response["status"] == "error" and response["result"] is None:
            raise WorkerFailure(_safe_worker_error(response["error"]))
        raise WorkerFailure("candidate protocol violation")

    def close(self) -> None:
        if self._proc.poll() is None:
            self._sequence += 1
            nonce = secrets.token_hex(16)
            body = {
                "args": [],
                "nonce": nonce,
                "op": "close",
                "seq": self._sequence,
                "version": PROTOCOL_VERSION,
            }
            try:
                self._write(encode_line(signed_message(body, self._key)))
                response = self._read_response(self._sequence, nonce)
                if response["status"] != "closed":
                    self._terminate()
            except (ProtocolError, WorkerFailure):
                self._terminate()
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._terminate()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._stdout.close()
        self._stderr.close()

    def __enter__(self) -> WorkerClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _failure(error: str) -> dict[str, object]:
    return {
        "candidate_time": 0.0,
        "error": error,
        "passed": False,
        "reference_time": 0.0,
        "speedup": 0.0,
    }


def _run(candidate_path: Path, task_name: str, call_timeout_s: float) -> dict[str, object]:
    if (
        candidate_path.is_symlink()
        or not candidate_path.is_file()
        or candidate_path.stat().st_size > _MAX_CANDIDATE_BYTES
    ):
        return _failure("candidate execution failed")
    task = get_kernel_task(task_name)
    generator = np.random.default_rng(secrets.randbits(128))
    candidate_best = math.inf
    reference_best = math.inf
    challenges: list[
        tuple[list[dict[str, object]], np.ndarray]
    ] = []

    try:
        # Freeze every reference result and baseline before candidate startup.
        # This prevents a candidate-created background workload from inflating
        # trusted reference timings in the shared outer sandbox.
        for _ in range(task.n_batches):
            for _ in range(_REPEATS):
                arguments = task.make_inputs(generator)
                encoded_arguments = [encode_array(argument) for argument in arguments]
                reference_arguments = tuple(
                    decode_array(argument) for argument in encoded_arguments
                )
                reference_started = time.perf_counter_ns()
                expected = task.reference(*reference_arguments)
                reference_elapsed = max(
                    time.perf_counter_ns() - reference_started,
                    1,
                )
                expected_array = decode_array(
                    encode_array(expected, force_float64=True)
                )
                reference_best = min(
                    reference_best,
                    reference_elapsed / 1_000_000_000,
                )
                challenges.append((encoded_arguments, expected_array))

        with WorkerClient(
            candidate_path,
            task.entrypoint,
            call_timeout_s=call_timeout_s,
            cwd=str(candidate_path.parent),
        ) as worker:
            for encoded_arguments, expected_array in challenges:
                started = time.perf_counter_ns()
                encoded_result = worker.call(encoded_arguments)
                candidate_elapsed = max(time.perf_counter_ns() - started, 1)
                candidate_array = decode_array(encoded_result)
                candidate_best = min(
                    candidate_best,
                    candidate_elapsed / 1_000_000_000,
                )
                if (
                    candidate_array.shape != expected_array.shape
                    or not np.allclose(
                        candidate_array,
                        expected_array,
                        atol=task.atol,
                        rtol=task.rtol,
                    )
                ):
                    return _failure("incorrect output")
    except WorkerFailure as exc:
        return _failure(_safe_worker_error(str(exc)))
    except (MemoryError, OSError, ProtocolError, ValueError):
        return _failure("supervisor failed")

    if (
        not math.isfinite(candidate_best)
        or not math.isfinite(reference_best)
        or candidate_best <= 0
        or reference_best <= 0
    ):
        return _failure("supervisor failed")
    speedup = reference_best / candidate_best
    return {
        "candidate_time": candidate_best,
        "error": None,
        "passed": True,
        "reference_time": reference_best,
        "speedup": speedup,
    }


def _read_supervisor_request() -> tuple[str, bytes, float]:
    request = require_exact_keys(
        loads_json(
            read_bounded_line(
                sys.stdin.buffer,
                limit=_SUPERVISOR_REQUEST_LIMIT_BYTES,
            )
        ),
        {"call_timeout_s", "key", "request_id", "version"},
    )
    request_id = request["request_id"]
    encoded_key = request["key"]
    timeout_s = request["call_timeout_s"]
    version = request["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PROTOCOL_VERSION
        or not isinstance(request_id, str)
        or not _REQUEST_ID_RE.fullmatch(request_id)
        or not isinstance(encoded_key, str)
        or isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not 0.01 <= float(timeout_s) <= 10.0
    ):
        raise ProtocolError("invalid supervisor request")
    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ProtocolError("invalid supervisor authentication key") from exc
    if len(key) != AUTH_KEY_BYTES:
        raise ProtocolError("invalid supervisor authentication key")
    return request_id, key, float(timeout_s)


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    try:
        request_id, key, call_timeout_s = _read_supervisor_request()
    except ProtocolError:
        return 2
    try:
        result = _run(Path(sys.argv[1]), sys.argv[2], call_timeout_s)
    except BaseException:  # noqa: BLE001 - supervisor always emits a signed failure
        result = _failure("supervisor failed")
    body = {
        "request_id": request_id,
        "result": result,
        "version": PROTOCOL_VERSION,
    }
    try:
        sys.stdout.buffer.write(encode_line(signed_message(body, key), limit=64 * 1024))
        sys.stdout.buffer.flush()
    except (OSError, ProtocolError):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trusted broker between the kernel supervisor and an untrusted candidate.

The HMAC key exists only in this broker and the supervisor. Candidate stdout is
bounded, structurally decoded, canonicalized, and signed before it can cross the
trusted IPC boundary.
"""

from __future__ import annotations

import base64
import binascii
import os
import queue
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

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

_BOOTSTRAP_LIMIT_BYTES = 1024
_MAX_CANDIDATE_OUTPUT_BYTES = 16 * 1024 * 1024
_MAX_CANDIDATE_STDERR_BYTES = 256 * 1024
_TYPE_ERROR_RE = re.compile(
    r"(?:compile/exec|runtime) failed: [A-Za-z_][A-Za-z0-9_]{0,63}"
)
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
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


class CandidateFailure(RuntimeError):
    """A bounded, sanitized failure from the untrusted candidate process."""


class _LinePump:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        line_limit: int,
        total_limit: int,
        fail: Callable[[str], None],
    ) -> None:
        self._stream = stream
        self._line_limit = line_limit
        self._total_limit = total_limit
        self._fail = fail
        self._items: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pending = bytearray()
        total = 0
        try:
            while chunk := self._stream.read(8192):
                total += len(chunk)
                pending.extend(chunk)
                if total > self._total_limit or len(pending) > self._line_limit:
                    self._fail("candidate output limit exceeded")
                    return
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(pending[:newline])
                    del pending[: newline + 1]
                    if len(line) > self._line_limit:
                        self._fail("candidate output limit exceeded")
                        return
                    self._items.put(line)
            if pending:
                self._fail("candidate protocol violation")
        except (OSError, ValueError):
            self._fail("candidate protocol violation")
        finally:
            self._items.put(None)

    def receive(self, timeout_s: float) -> bytes:
        try:
            item = self._items.get(timeout=timeout_s)
        except queue.Empty as exc:
            self._fail("candidate timeout")
            raise CandidateFailure("candidate timeout") from exc
        if item is None:
            raise CandidateFailure("candidate exited")
        return item

    def close(self) -> None:
        self._thread.join(timeout=1.0)


class _DrainPump:
    def __init__(
        self,
        stream: BinaryIO,
        *,
        limit: int,
        fail: Callable[[str], None],
    ) -> None:
        self._stream = stream
        self._limit = limit
        self._fail = fail
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        total = 0
        try:
            while chunk := self._stream.read(8192):
                total += len(chunk)
                if total > self._limit:
                    self._fail("candidate output limit exceeded")
                    return
        except (OSError, ValueError):
            return

    def close(self) -> None:
        self._thread.join(timeout=1.0)


def _candidate_environment() -> dict[str, str]:
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


def _parent_death_signal() -> None:
    """Ask Linux to kill the candidate if its broker disappears."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)
    except (AttributeError, OSError):
        return


class CandidateProcess:
    """One persistent, resource-bounded untrusted candidate interpreter."""

    def __init__(self, candidate_path: Path, entrypoint: str, *, cwd: str) -> None:
        script = Path(__file__).with_name("_candidate_worker.py").resolve()
        kwargs: dict[str, object] = {}
        if os.name == "posix":
            kwargs.update(start_new_session=True, preexec_fn=_parent_death_signal)
        elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):  # pragma: no cover
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self._proc = subprocess.Popen(
            [sys.executable, "-I", "-B", str(script), str(candidate_path), entrypoint],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=_candidate_environment(),
            close_fds=True,
            **kwargs,
        )
        self._failure: str | None = None
        self._failure_lock = threading.Lock()
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout = _LinePump(
            self._proc.stdout,
            line_limit=MAX_CONTROL_LINE_BYTES,
            total_limit=_MAX_CANDIDATE_OUTPUT_BYTES,
            fail=self._fail,
        )
        self._stderr = _DrainPump(
            self._proc.stderr,
            limit=_MAX_CANDIDATE_STDERR_BYTES,
            fail=self._fail,
        )

    def _fail(self, error: str) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = error
        self._terminate()

    def _terminate(self) -> None:
        try:
            if os.name == "posix":
                # The session may still contain candidate grandchildren after
                # its original leader exits, so always target the whole group.
                os.killpg(self._proc.pid, signal.SIGKILL)
            elif self._proc.poll() is None:  # pragma: no cover - Windows
                self._proc.kill()
        except (OSError, ProcessLookupError):
            return

    def _current_failure(self) -> str | None:
        with self._failure_lock:
            return self._failure

    def _receive(
        self,
        sequence: int,
        nonce: str | None,
        timeout_s: float,
    ) -> dict[str, object]:
        try:
            raw = self._stdout.receive(timeout_s)
        except CandidateFailure as exc:
            raise CandidateFailure(self._current_failure() or str(exc)) from exc
        failure = self._current_failure()
        if failure is not None:
            raise CandidateFailure(failure)
        try:
            response = require_exact_keys(
                loads_json(raw),
                {"error", "nonce", "result", "seq", "status", "version"},
            )
        except ProtocolError as exc:
            self._fail("candidate protocol violation")
            raise CandidateFailure("candidate protocol violation") from exc
        response_version = response["version"]
        response_sequence = response["seq"]
        if (
            isinstance(response_version, bool)
            or not isinstance(response_version, int)
            or response_version != PROTOCOL_VERSION
            or isinstance(response_sequence, bool)
            or not isinstance(response_sequence, int)
            or response_sequence != sequence
            or response["nonce"] != nonce
            or not isinstance(response["status"], str)
        ):
            self._fail("candidate protocol violation")
            raise CandidateFailure("candidate protocol violation")
        return response

    @staticmethod
    def _candidate_error(value: object) -> str:
        if value == "missing entrypoint":
            return "missing entrypoint"
        if isinstance(value, str) and _TYPE_ERROR_RE.fullmatch(value):
            return value
        return "candidate execution failed"

    def wait_ready(self, timeout_s: float) -> None:
        response = self._receive(0, None, timeout_s)
        if (
            response["status"] == "ready"
            and response["result"] is None
            and response["error"] is None
        ):
            return
        if response["status"] == "error" and response["result"] is None:
            raise CandidateFailure(self._candidate_error(response["error"]))
        self._fail("candidate protocol violation")
        raise CandidateFailure("candidate protocol violation")

    def call(
        self,
        sequence: int,
        nonce: str,
        arguments: list[dict[str, object]],
        timeout_s: float,
    ) -> dict[str, object]:
        if self._proc.poll() is not None or self._proc.stdin is None:
            raise CandidateFailure(self._current_failure() or "candidate exited")
        request = {
            "args": arguments,
            "nonce": nonce,
            "seq": sequence,
            "version": PROTOCOL_VERSION,
        }
        try:
            self._proc.stdin.write(encode_line(request))
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise CandidateFailure("candidate exited") from exc
        response = self._receive(sequence, nonce, timeout_s)
        if (
            response["status"] == "ok"
            and response["result"] is not None
            and response["error"] is None
        ):
            try:
                return encode_array(decode_array(response["result"]))
            except ProtocolError as exc:
                self._fail("candidate protocol violation")
                raise CandidateFailure("candidate protocol violation") from exc
        if response["status"] == "error" and response["result"] is None:
            raise CandidateFailure(self._candidate_error(response["error"]))
        self._fail("candidate protocol violation")
        raise CandidateFailure("candidate protocol violation")

    def close(self) -> None:
        if self._proc.stdin is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
        self._terminate()
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._terminate()
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._stdout.close()
        self._stderr.close()


def _read_bootstrap() -> tuple[bytes, float]:
    request = require_exact_keys(
        loads_json(read_bounded_line(sys.stdin.buffer, limit=_BOOTSTRAP_LIMIT_BYTES)),
        {"call_timeout_s", "key", "version"},
    )
    timeout_s = request["call_timeout_s"]
    encoded_key = request["key"]
    version = request["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PROTOCOL_VERSION
        or isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not 0.01 <= float(timeout_s) <= 10.0
        or not isinstance(encoded_key, str)
    ):
        raise ProtocolError("invalid worker bootstrap")
    try:
        key = base64.b64decode(encoded_key, validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ProtocolError("invalid worker authentication key") from exc
    if len(key) != AUTH_KEY_BYTES:
        raise ProtocolError("invalid worker authentication key")
    return key, float(timeout_s)


def _send_signed(
    key: bytes,
    sequence: int,
    *,
    nonce: str | None,
    status: str,
    result: dict[str, object] | None = None,
    error: str | None = None,
) -> None:
    body = {
        "error": error,
        "nonce": nonce,
        "result": result,
        "seq": sequence,
        "status": status,
        "version": PROTOCOL_VERSION,
    }
    sys.stdout.buffer.write(encode_line(signed_message(body, key)))
    sys.stdout.buffer.flush()


def _read_supervisor_request(key: bytes) -> dict[str, object]:
    message = loads_json(
        read_bounded_line(sys.stdin.buffer, limit=MAX_CONTROL_LINE_BYTES)
    )
    body = require_exact_keys(
        verify_signed_message(message, key),
        {"args", "nonce", "op", "seq", "version"},
    )
    sequence = body["seq"]
    version = body["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PROTOCOL_VERSION
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or not isinstance(body["nonce"], str)
        or not _NONCE_RE.fullmatch(body["nonce"])
        or not isinstance(body["op"], str)
        or body["op"] not in {"call", "close"}
        or not isinstance(body["args"], list)
        or len(body["args"]) > 16
    ):
        raise ProtocolError("invalid supervisor request")
    return body


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    try:
        key, timeout_s = _read_bootstrap()
    except ProtocolError:
        return 2

    candidate: CandidateProcess | None = None
    try:
        candidate = CandidateProcess(Path(sys.argv[1]), sys.argv[2], cwd=os.getcwd())
        try:
            candidate.wait_ready(timeout_s)
        except CandidateFailure as exc:
            _send_signed(key, 0, nonce=None, status="error", error=str(exc))
            return 1
        _send_signed(key, 0, nonce=None, status="ready")

        while True:
            try:
                request = _read_supervisor_request(key)
            except ProtocolError:
                return 2
            sequence = int(request["seq"])
            nonce = str(request["nonce"])
            if request["op"] == "close":
                if request["args"]:
                    return 2
                _send_signed(key, sequence, nonce=nonce, status="closed")
                return 0
            try:
                arguments = [
                    encode_array(decode_array(item)) for item in request["args"]
                ]
                result = candidate.call(sequence, nonce, arguments, timeout_s)
            except (CandidateFailure, ProtocolError) as exc:
                _send_signed(
                    key,
                    sequence,
                    nonce=nonce,
                    status="error",
                    error=str(exc),
                )
                return 1
            _send_signed(key, sequence, nonce=nonce, status="ok", result=result)
    finally:
        if candidate is not None:
            candidate.close()


if __name__ == "__main__":
    raise SystemExit(main())

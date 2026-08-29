#!/usr/bin/python
"""Machine-verifiable kernel reward for the SAI factory (AHE-3.28).

:class:`KernelVerifier` satisfies the ``agent_utilities.harness.sai_task.Verifier``
protocol: given a candidate kernel source string it returns a
:class:`~agent_utilities.harness.sai_task.VerifierResult` whose ``reward`` is the
correctness-gated speedup over the task reference (``reward = speedup`` when the
output is correct, ``0.0`` otherwise) — a real, comparable signal the SAI factory
both optimizes (adaptation-speed curve, AHE-3.27) and distills into training data
(OS-5.34 / AHE-3.25).

Candidate code is default-deny unless an administrator configures an isolated
no-shell runner (custom argv or a networkless, read-only container). A custom
sandbox is an operator trust boundary and must transport the supervisor's
bounded stdin/stdout protocol without exposing it to the candidate process.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import os
import re
import secrets
import signal
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agent_utilities.harness.sai_task import VerifierResult

from data_science_mcp.kernels._protocol import (
    AUTH_KEY_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    encode_line,
    loads_json,
    require_exact_keys,
    verify_signed_message,
)
from data_science_mcp.kernels.kernel_tasks import KernelTask

_MAX_CANDIDATE_BYTES = 256 * 1024
_MAX_SANDBOX_COMMAND_BYTES = 32 * 1024
_MAX_SANDBOX_OUTPUT_BYTES = 128 * 1024
_MAX_SUPERVISOR_REQUEST_BYTES = 2048
_SANDBOX_PLACEHOLDERS = ("{candidate}", "{task}")
_PINNED_IMAGE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,446}@sha256:[0-9a-f]{64}"
)
_CONTAINER_RUNTIMES = frozenset({"docker", "podman"})


class SandboxOutputLimitError(RuntimeError):
    """Raised after terminating a sandbox that exceeded its output allowance."""


def _finite_in_range(value: object, minimum: float, maximum: float) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        rendered = float(value)
    except OverflowError:
        return False
    return math.isfinite(rendered) and minimum <= rendered <= maximum


def _protocol_number(value: object, *, positive: bool = False) -> float:
    """Decode one finite runner-protocol number without bool/string coercion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid runner protocol number")
    try:
        rendered = float(value)
    except OverflowError as exc:
        raise ValueError("Invalid runner protocol number") from exc
    if not math.isfinite(rendered) or rendered < 0 or (positive and rendered == 0):
        raise ValueError("Invalid runner protocol number")
    return rendered


def _safe_runner_error(value: object) -> str | None:
    """Return only the small, path-free error vocabulary emitted by ``_runner``."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 160:
        return "kernel verification failed"
    if value in {
        "candidate execution failed",
        "candidate exited",
        "candidate output limit exceeded",
        "candidate protocol violation",
        "candidate timeout",
        "incorrect output",
        "missing entrypoint",
        "supervisor failed",
    }:
        return value
    if re.fullmatch(
        r"(?:compile/exec|runtime) failed: [A-Za-z_][A-Za-z0-9_]{0,63}", value
    ):
        return value
    return "kernel verification failed"


def _kernel_failure(task_name: str, error: str, **extra: object) -> VerifierResult:
    """The shared ``reward=0.0, passed=False`` shape every ``verify`` rejection uses."""
    return VerifierResult(
        reward=0.0, passed=False, detail={"task": task_name, "error": error, **extra}
    )


def _encode_candidate_source(candidate: object) -> bytes:
    """Encode + size-check candidate source; raises ``ValueError`` on rejection."""
    if not isinstance(candidate, str):
        raise ValueError("candidate encoding rejected")
    try:
        candidate_bytes = candidate.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError("candidate encoding rejected") from exc
    if len(candidate_bytes) > _MAX_CANDIDATE_BYTES:
        raise ValueError("candidate size limit exceeded")
    return candidate_bytes


def _sandbox_child_env() -> dict[str, str]:
    """The minimal, thread-pinned environment the sandbox subprocess launches with."""
    child_env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "XDG_RUNTIME_DIR")
        if key in os.environ
    }
    child_env.update(
        {
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return child_env


@dataclasses.dataclass
class _RunnerVerdict:
    """The decoded, signature-verified result envelope from ``_runner``."""

    passed: bool
    speedup: float
    candidate_time: float
    reference_time: float
    error: object


def _validate_runner_verdict_consistency(verdict: _RunnerVerdict) -> None:
    """Raises ``ValueError`` when the runner's own reported fields are inconsistent."""
    if verdict.passed:
        if verdict.error is not None or not math.isclose(
            verdict.speedup,
            verdict.reference_time / verdict.candidate_time,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("Invalid successful runner result")
    elif (
        verdict.speedup != 0.0
        or verdict.candidate_time != 0.0
        or verdict.reference_time != 0.0
        or verdict.error is None
    ):
        raise ValueError("Invalid failed runner result")


def _decode_runner_envelope(
    raw_line: str, auth_key: bytes, request_id: str
) -> _RunnerVerdict:
    """Verify + decode one signed ``_runner`` result line into a :class:`_RunnerVerdict`.

    Raises ``ProtocolError``/``TypeError``/``ValueError``/``ZeroDivisionError`` on
    any rejection; the caller folds all of those into one "invalid runner protocol".
    """
    envelope = require_exact_keys(
        verify_signed_message(loads_json(raw_line), auth_key),
        {"request_id", "result", "version"},
    )
    envelope_version = envelope["version"]
    if (
        isinstance(envelope_version, bool)
        or not isinstance(envelope_version, int)
        or envelope_version != PROTOCOL_VERSION
        or envelope["request_id"] != request_id
    ):
        raise ValueError("Invalid runner protocol envelope")
    data = require_exact_keys(
        envelope["result"],
        {"candidate_time", "error", "passed", "reference_time", "speedup"},
    )
    if not isinstance(data["passed"], bool):
        raise ValueError("Invalid runner pass flag")
    passed = data["passed"]
    verdict = _RunnerVerdict(
        passed=passed,
        speedup=_protocol_number(data["speedup"], positive=passed),
        candidate_time=_protocol_number(data["candidate_time"], positive=passed),
        reference_time=_protocol_number(data["reference_time"], positive=passed),
        error=data["error"],
    )
    _validate_runner_verdict_consistency(verdict)
    return verdict


def _is_string_argv(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and len(value) <= 128
        and all(isinstance(item, str) and item for item in value)
    )


def _has_unsafe_argv_chars(parsed: tuple[str, ...]) -> bool:
    return any("\x00" in item or "\n" in item or "\r" in item for item in parsed)


def _missing_sandbox_placeholder(rendered: str) -> bool:
    return any(placeholder not in rendered for placeholder in _SANDBOX_PLACEHOLDERS)


def _validate_sandbox_command(value: object) -> tuple[str, ...]:
    """Validate an externally managed sandbox command before interpolation."""
    if not _is_string_argv(value):
        raise ValueError("Kernel sandbox command must be a non-empty argv array")
    parsed = tuple(value)
    if sum(len(item.encode("utf-8")) for item in parsed) > _MAX_SANDBOX_COMMAND_BYTES:
        raise ValueError("Kernel sandbox command exceeds its size limit")
    if _has_unsafe_argv_chars(parsed):
        raise ValueError("Kernel sandbox command contains invalid characters")
    if _missing_sandbox_placeholder("\n".join(parsed)):
        raise ValueError("Kernel sandbox command must include candidate and task placeholders")
    return parsed


def configured_kernel_sandbox_command() -> tuple[str, ...] | None:
    """Load a no-shell sandbox argv from a JSON array environment setting."""
    raw = os.environ.get("DATA_SCIENCE_KERNEL_SANDBOX_COMMAND", "").strip()
    if not raw:
        return None
    if len(raw.encode("utf-8")) > _MAX_SANDBOX_COMMAND_BYTES:
        raise ValueError("Kernel sandbox command exceeds its size limit")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Kernel sandbox command must be a JSON argv array") from exc
    try:
        return _validate_sandbox_command(parsed)
    except ValueError as exc:
        raise ValueError(f"Invalid kernel sandbox JSON argv: {exc}") from exc


def configured_kernel_sandbox_image() -> str | None:
    """Return an immutable container reference or reject unsafe option-shaped input."""
    image = os.environ.get("DATA_SCIENCE_KERNEL_SANDBOX_IMAGE", "").strip()
    if not image:
        return None
    if not _PINNED_IMAGE_RE.fullmatch(image):
        raise ValueError("Kernel sandbox image must be pinned by sha256 digest")
    return image


def kernel_sandbox_configured() -> bool:
    """Return whether candidate execution has an explicit isolation boundary."""
    try:
        return bool(
            configured_kernel_sandbox_command()
            or configured_kernel_sandbox_image()
        )
    except ValueError:
        return False


class _BoundedOutputCapture:
    """Thread-drained stdout/stderr capture with a shared, hard total-byte cap.

    Each stream is drained by its own thread via :meth:`drain`; bytes past the
    cap are silently dropped and ``overflow`` is set instead of ever growing
    ``buffers`` past ``limit``.
    """

    def __init__(self, streams: dict[str, Any], limit: int) -> None:
        self.streams = streams
        self.limit = limit
        self.buffers: dict[str, bytearray] = {name: bytearray() for name in streams}
        self.used = 0
        self.lock = threading.Lock()
        self.overflow = threading.Event()

    def drain(self, name: str, on_overflow: Callable[[], None]) -> None:
        stream = self.streams[name]
        if stream is None:
            return
        try:
            while chunk := stream.read(8192):
                with self.lock:
                    remaining = self.limit - self.used
                    accepted = chunk[: max(remaining, 0)]
                    self.buffers[name].extend(accepted)
                    self.used += len(accepted)
                    if len(accepted) != len(chunk):
                        self.overflow.set()
                if self.overflow.is_set():
                    on_overflow()
                    return
        except (OSError, ValueError):
            return


class KernelVerifier:
    """Verify a kernel candidate by correctness + measured speedup vs the reference."""

    def __init__(
        self,
        task: KernelTask,
        *,
        timeout_s: float = 30.0,
        speedup_cap: float = 50.0,
        sandbox_command: Sequence[str] | None = None,
    ) -> None:
        if not _finite_in_range(timeout_s, 0.1, 300.0):
            raise ValueError("Kernel timeout must be between 0.1 and 300 seconds")
        if not _finite_in_range(speedup_cap, 0.0, 1000.0):
            raise ValueError("Kernel speedup cap must be between 0 and 1000")
        self.task = task
        self.timeout_s = float(timeout_s)
        self.speedup_cap = float(speedup_cap)
        self.sandbox_command = (
            _validate_sandbox_command(sandbox_command)
            if sandbox_command is not None
            else configured_kernel_sandbox_command()
        )

    def _sandbox_argv(self, candidate_path: Path) -> list[str] | None:
        """Build the isolated runner argv without invoking a command shell."""
        candidate_rendered = str(candidate_path)
        if any(char in candidate_rendered for char in "\x00\r\n,"):
            raise ValueError("Kernel candidate path is unsafe for a sandbox mount")
        if self.sandbox_command:
            return [
                token.replace("{candidate}", candidate_rendered).replace(
                    "{task}", self.task.name
                )
                for token in self.sandbox_command
            ]

        image = configured_kernel_sandbox_image()
        if image:
            runtime_name = os.environ.get(
                "DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME", "docker"
            ).strip()
            if runtime_name not in _CONTAINER_RUNTIMES:
                raise ValueError("Kernel container runtime must be docker or podman")
            runtime = shutil.which(runtime_name)
            if not runtime:
                raise ValueError("Configured kernel container runtime is unavailable")
            return [
                runtime,
                "run",
                "--rm",
                "--interactive",
                "--network=none",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--pids-limit=64",
                "--memory=1g",
                "--cpus=1",
                "--env=MKL_NUM_THREADS=1",
                "--env=NUMEXPR_NUM_THREADS=1",
                "--env=OMP_NUM_THREADS=1",
                "--env=OPENBLAS_NUM_THREADS=1",
                "--env=PYTHONDONTWRITEBYTECODE=1",
                "--env=PYTHONNOUSERSITE=1",
                "--ulimit=core=0:0",
                "--ulimit=cpu=60:60",
                "--ulimit=fsize=16777216:16777216",
                "--ulimit=nofile=64:64",
                "--user=65534:65534",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
                "--cidfile",
                str(candidate_path.with_suffix(".cid")),
                "--mount",
                f"type=bind,src={candidate_rendered},dst=/candidate.py,readonly",
                "--entrypoint=python",
                image,
                "-m",
                "data_science_mcp.kernels._runner",
                "/candidate.py",
                self.task.name,
            ]

        return None

    @staticmethod
    def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:  # pragma: no cover - exercised on Windows
                proc.kill()
        except (OSError, ProcessLookupError):
            pass

    def _run_bounded(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        input_bytes: bytes = b"",
    ) -> subprocess.CompletedProcess[str]:
        """Run a sandbox with a combined, streaming stdout/stderr hard limit."""
        if len(input_bytes) > _MAX_SUPERVISOR_REQUEST_BYTES:
            raise ValueError("Kernel supervisor request exceeds its size limit")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=os.name == "posix",
        )
        capture = _BoundedOutputCapture(
            {"stdout": proc.stdout, "stderr": proc.stderr}, _MAX_SANDBOX_OUTPUT_BYTES
        )
        readers = [
            threading.Thread(
                target=capture.drain,
                args=(name, lambda: self._terminate_process(proc)),
                daemon=True,
            )
            for name in capture.streams
        ]
        for reader in readers:
            reader.start()
        self._feed_stdin(proc, input_bytes)
        try:
            returncode = proc.wait(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate_process(proc)
            proc.wait()
            raise
        finally:
            for reader in readers:
                reader.join(timeout=2.0)
            for stream in capture.streams.values():
                if stream is not None:
                    stream.close()

        if capture.overflow.is_set():
            raise SandboxOutputLimitError
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout=capture.buffers["stdout"].decode("utf-8", errors="replace"),
            stderr=capture.buffers["stderr"].decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _feed_stdin(proc: subprocess.Popen[bytes], input_bytes: bytes) -> None:
        if proc.stdin is not None:
            try:
                proc.stdin.write(input_bytes)
                proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            finally:
                proc.stdin.close()

    @staticmethod
    def _cleanup_container(candidate_path: Path, env: dict[str, str]) -> None:
        """Force-remove a timed-out container without trusting cidfile content."""
        cid_path = candidate_path.with_suffix(".cid")
        if not cid_path.is_file() or cid_path.is_symlink():
            return
        try:
            container_id = cid_path.read_text(encoding="ascii")[:129].strip()
        except (OSError, UnicodeError):
            return
        if not re.fullmatch(r"[0-9a-f]{12,128}", container_id):
            return
        runtime_name = os.environ.get(
            "DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME", "docker"
        ).strip()
        if runtime_name not in _CONTAINER_RUNTIMES:
            return
        runtime = shutil.which(runtime_name)
        if not runtime:
            return
        try:
            subprocess.run(
                [runtime, "rm", "--force", container_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
                env=env,
            )
        except (OSError, subprocess.TimeoutExpired):
            return

    def _launch_sandbox(
        self,
        cand_path: Path,
        tmp: str,
        child_env: dict[str, str],
        supervisor_request: bytes,
    ) -> subprocess.CompletedProcess[str] | None:
        """Build the sandbox argv and run it bounded. ``None`` when unconfigured;
        may propagate ``TimeoutExpired`` / ``SandboxOutputLimitError`` / ``OSError``
        / ``ValueError`` from the sandbox launch itself."""
        argv = self._sandbox_argv(cand_path)
        if argv is None:
            return None
        return self._run_bounded(argv, cwd=tmp, env=child_env, input_bytes=supervisor_request)

    def _run_sandboxed_candidate(
        self, candidate_bytes: bytes, auth_key: bytes, request_id: str
    ) -> subprocess.CompletedProcess[str] | VerifierResult:
        """Write the candidate + run it through the sandbox; a ``VerifierResult`` is
        an already-final rejection, anything else is the completed sandbox process."""
        with tempfile.TemporaryDirectory(prefix="sai-kernel-") as tmp:
            cand_path = Path(tmp) / "candidate.py"
            cand_path.write_bytes(candidate_bytes)
            call_timeout_s = min(10.0, max(0.05, self.timeout_s / 3.0))
            supervisor_request = encode_line(
                {
                    "call_timeout_s": call_timeout_s,
                    "key": base64.b64encode(auth_key).decode("ascii"),
                    "request_id": request_id,
                    "version": PROTOCOL_VERSION,
                },
                limit=_MAX_SUPERVISOR_REQUEST_BYTES,
            )
            child_env = _sandbox_child_env()
            try:
                proc = self._launch_sandbox(cand_path, tmp, child_env, supervisor_request)
                if proc is None:
                    return _kernel_failure(
                        self.task.name, "isolated kernel sandbox is not configured"
                    )
            except subprocess.TimeoutExpired:
                self._cleanup_container(cand_path, child_env)
                return _kernel_failure(self.task.name, "timeout", timeout_s=self.timeout_s)
            except SandboxOutputLimitError:
                self._cleanup_container(cand_path, child_env)
                return _kernel_failure(self.task.name, "sandbox output limit exceeded")
            except (OSError, ValueError):
                self._cleanup_container(cand_path, child_env)
                return _kernel_failure(self.task.name, "sandbox launch failed")
        return proc

    def verify(self, candidate: str) -> VerifierResult:
        """Return correctness-gated speedup as the reward for one candidate."""
        try:
            candidate_bytes = _encode_candidate_source(candidate)
        except ValueError as exc:
            return _kernel_failure(self.task.name, str(exc))

        auth_key = secrets.token_bytes(AUTH_KEY_BYTES)
        request_id = secrets.token_hex(16)
        outcome = self._run_sandboxed_candidate(candidate_bytes, auth_key, request_id)
        if isinstance(outcome, VerifierResult):
            return outcome
        proc = outcome

        raw = (proc.stdout or "").strip().splitlines()
        if proc.returncode != 0 or len(raw) != 1:
            return _kernel_failure(self.task.name, "invalid runner protocol")
        try:
            verdict = _decode_runner_envelope(raw[0], auth_key, request_id)
        except (ProtocolError, TypeError, ValueError, ZeroDivisionError):
            return _kernel_failure(self.task.name, "invalid runner protocol")

        reward = min(verdict.speedup, self.speedup_cap) if verdict.passed else 0.0
        return VerifierResult(
            reward=reward,
            passed=verdict.passed,
            detail={
                "task": self.task.name,
                "speedup": verdict.speedup,
                "candidate_time": verdict.candidate_time,
                "reference_time": verdict.reference_time,
                "error": _safe_runner_error(verdict.error),
            },
        )

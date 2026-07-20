"""Untrusted candidate interpreter used by the kernel evaluation broker.

Nothing emitted by this process is trusted. The broker bounds and authenticates
it before the supervisor performs independent correctness and timing checks.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, BinaryIO

from data_science_mcp.kernels._protocol import (
    MAX_CONTROL_LINE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    decode_array,
    encode_array,
    encode_line,
    loads_json,
    read_bounded_line,
    require_exact_keys,
)

_MAX_CANDIDATE_BYTES = 256 * 1024
_TYPE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")
_NONCE_RE = re.compile(r"[0-9a-f]{32}")
_INPUT = sys.stdin.buffer
_OUTPUT = sys.stdout.buffer


def _bounded_type_name(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if _TYPE_NAME_RE.fullmatch(name) else "Exception"


def _apply_resource_limits() -> None:
    """Apply finite host fallbacks; the configured container remains primary."""
    if os.name != "posix":
        return
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX implementations
        return

    limits = (
        (resource.RLIMIT_AS, 2 * 1024 * 1024 * 1024),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_CPU, 60),
        (resource.RLIMIT_FSIZE, 16 * 1024 * 1024),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_NPROC, 64),
    )
    for kind, requested in limits:
        try:
            _soft, hard = resource.getrlimit(kind)
            target = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
            resource.setrlimit(kind, (target, target))
        except (OSError, ValueError):
            continue


def _send(value: object, output: BinaryIO = _OUTPUT) -> None:
    output.write(encode_line(value))
    output.flush()


def _response(
    sequence: int,
    *,
    nonce: str | None,
    status: str,
    result: dict[str, object] | None = None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "error": error,
        "nonce": nonce,
        "result": result,
        "seq": sequence,
        "status": status,
        "version": PROTOCOL_VERSION,
    }


def _load_candidate(candidate_path: Path, entrypoint: str) -> tuple[Any | None, str | None]:
    try:
        if candidate_path.is_symlink() or not candidate_path.is_file():
            raise ValueError("candidate path")
        source = candidate_path.read_bytes()
        if len(source) > _MAX_CANDIDATE_BYTES:
            raise ValueError("candidate size")
        text = source.decode("utf-8")
        namespace: dict[str, Any] = {
            "__file__": "<candidate>",
            "__name__": "__candidate__",
        }
        exec(compile(text, "<candidate>", "exec"), namespace)  # noqa: S102
    except BaseException as exc:  # noqa: BLE001 - untrusted source fails closed
        return None, f"compile/exec failed: {_bounded_type_name(exc)}"
    function = namespace.get(entrypoint)
    if not callable(function):
        return None, "missing entrypoint"
    return function, None


def _decode_request(raw: bytes) -> tuple[int, str, list[Any]]:
    request = require_exact_keys(
        loads_json(raw),
        {"args", "nonce", "seq", "version"},
    )
    sequence = request["seq"]
    version = request["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PROTOCOL_VERSION
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence <= 0
        or not isinstance(request["nonce"], str)
        or not _NONCE_RE.fullmatch(request["nonce"])
        or not isinstance(request["args"], list)
        or len(request["args"]) > 16
    ):
        raise ProtocolError("invalid candidate request")
    return sequence, request["nonce"], [
        decode_array(item) for item in request["args"]
    ]


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    _apply_resource_limits()
    function, load_error = _load_candidate(Path(sys.argv[1]), sys.argv[2])
    if load_error is not None:
        _send(_response(0, nonce=None, status="error", error=load_error))
        return 1
    _send(_response(0, nonce=None, status="ready"))

    while True:
        try:
            raw = read_bounded_line(_INPUT, limit=MAX_CONTROL_LINE_BYTES)
            sequence, nonce, args = _decode_request(raw)
        except ProtocolError:
            return 2
        try:
            output = function(*args)
            encoded = encode_array(output, force_float64=True)
        except BaseException as exc:  # noqa: BLE001 - candidate failures are data
            _send(
                _response(
                    sequence,
                    nonce=nonce,
                    status="error",
                    error=f"runtime failed: {_bounded_type_name(exc)}",
                )
            )
            return 1
        _send(_response(sequence, nonce=nonce, status="ok", result=encoded))


if __name__ == "__main__":
    raise SystemExit(main())

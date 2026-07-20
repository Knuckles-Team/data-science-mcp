"""Strict bounded protocol primitives for isolated kernel evaluation."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from typing import Any, BinaryIO

import numpy as np

PROTOCOL_VERSION = 1
AUTH_KEY_BYTES = 32
MAX_CONTROL_LINE_BYTES = 8 * 1024 * 1024
MAX_ARRAY_BYTES = 4 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 524_288
MAX_ARRAY_DIMENSIONS = 8

_DTYPES = {
    "b1": np.dtype("?"),
    "f4": np.dtype("<f4"),
    "f8": np.dtype("<f8"),
    "i8": np.dtype("<i8"),
    "u8": np.dtype("<u8"),
}


class ProtocolError(ValueError):
    """Raised when a kernel protocol frame violates its trust boundary."""


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate protocol key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise ProtocolError("non-finite protocol number")


def loads_json(raw: bytes | str) -> object:
    """Load strict JSON while rejecting duplicate keys and NaN-like constants."""
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError("invalid protocol JSON") from exc


def canonical_json(value: object) -> bytes:
    """Encode one deterministic JSON value suitable for HMAC authentication."""
    try:
        rendered = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (RecursionError, TypeError, ValueError) as exc:
        raise ProtocolError("protocol value is not JSON serializable") from exc
    return rendered.encode("ascii")


def encode_line(value: object, *, limit: int = MAX_CONTROL_LINE_BYTES) -> bytes:
    """Encode one newline-delimited frame within a hard byte limit."""
    rendered = canonical_json(value) + b"\n"
    if len(rendered) > limit:
        raise ProtocolError("protocol frame exceeds its size limit")
    return rendered


def read_bounded_line(stream: BinaryIO, *, limit: int) -> bytes:
    """Read exactly one newline-terminated frame without unbounded buffering."""
    line = stream.readline(limit + 1)
    if not line:
        raise ProtocolError("protocol stream ended")
    if len(line) > limit or not line.endswith(b"\n"):
        raise ProtocolError("protocol frame exceeds its size limit")
    return line[:-1]


def signed_message(body: dict[str, object], key: bytes) -> dict[str, object]:
    """Return a canonical-body HMAC-SHA256 envelope."""
    if len(key) != AUTH_KEY_BYTES or "mac" in body:
        raise ProtocolError("invalid protocol authentication state")
    mac = hmac.new(key, canonical_json(body), hashlib.sha256).hexdigest()
    return {**body, "mac": mac}


def verify_signed_message(value: object, key: bytes) -> dict[str, object]:
    """Authenticate an envelope and return its body."""
    if len(key) != AUTH_KEY_BYTES or not isinstance(value, dict):
        raise ProtocolError("invalid authenticated protocol envelope")
    mac = value.get("mac")
    if (
        not isinstance(mac, str)
        or len(mac) != 64
        or any(character not in "0123456789abcdef" for character in mac)
    ):
        raise ProtocolError("invalid protocol authentication code")
    body = {name: item for name, item in value.items() if name != "mac"}
    expected = hmac.new(key, canonical_json(body), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        raise ProtocolError("protocol authentication failed")
    return body


def require_exact_keys(value: object, expected: set[str]) -> dict[str, Any]:
    """Require a JSON object with one exact key set."""
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError("invalid protocol object shape")
    return value


def _normalized_dtype(array: np.ndarray, *, force_float64: bool) -> np.dtype[Any]:
    if array.dtype.kind not in {"b", "f", "i", "u"}:
        raise ProtocolError("array dtype is not supported")
    if force_float64:
        return _DTYPES["f8"]
    if array.dtype.kind == "b":
        return _DTYPES["b1"]
    if array.dtype.kind == "f":
        return _DTYPES["f4"] if array.dtype.itemsize <= 4 else _DTYPES["f8"]
    if array.dtype.kind == "i":
        return _DTYPES["i8"]
    if array.dtype.kind == "u":
        return _DTYPES["u8"]
    raise ProtocolError("array dtype is not supported")


def _shape_size(shape: object) -> tuple[tuple[int, ...], int]:
    if not isinstance(shape, list) or len(shape) > MAX_ARRAY_DIMENSIONS:
        raise ProtocolError("array shape is invalid")
    parsed: list[int] = []
    total = 1
    for dimension in shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or not 0 <= dimension <= MAX_ARRAY_ELEMENTS
        ):
            raise ProtocolError("array dimension is invalid")
        parsed.append(dimension)
        if dimension == 0:
            total = 0
        elif total and total > MAX_ARRAY_ELEMENTS // dimension:
            raise ProtocolError("array has too many elements")
        else:
            total *= dimension
    if total > MAX_ARRAY_ELEMENTS:
        raise ProtocolError("array has too many elements")
    return tuple(parsed), total


def encode_array(value: object, *, force_float64: bool = False) -> dict[str, object]:
    """Encode a bounded numeric array as shape metadata plus base64 raw bytes."""
    try:
        source = np.asarray(value)
        dtype = _normalized_dtype(source, force_float64=force_float64)
        array = np.ascontiguousarray(source, dtype=dtype)
    except (MemoryError, TypeError, ValueError) as exc:
        raise ProtocolError("array could not be encoded") from exc
    _, total = _shape_size(list(array.shape))
    if total != array.size or array.nbytes > MAX_ARRAY_BYTES:
        raise ProtocolError("array exceeds its size limit")
    code = next(name for name, item in _DTYPES.items() if item == dtype)
    return {
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "dtype": code,
        "shape": list(array.shape),
    }


def decode_array(value: object) -> np.ndarray:
    """Decode a bounded numeric array after validating allocation metadata."""
    envelope = require_exact_keys(value, {"data", "dtype", "shape"})
    dtype_name = envelope["dtype"]
    data = envelope["data"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPES:
        raise ProtocolError("array dtype is invalid")
    if not isinstance(data, str) or len(data) > 4 * ((MAX_ARRAY_BYTES + 2) // 3):
        raise ProtocolError("array payload exceeds its size limit")
    shape, total = _shape_size(envelope["shape"])
    dtype = _DTYPES[dtype_name]
    expected_bytes = total * dtype.itemsize
    if expected_bytes > MAX_ARRAY_BYTES:
        raise ProtocolError("array payload exceeds its size limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ProtocolError("array payload is not valid base64") from exc
    if len(raw) != expected_bytes:
        raise ProtocolError("array payload length does not match its shape")
    try:
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (MemoryError, ValueError) as exc:
        raise ProtocolError("array payload could not be decoded") from exc

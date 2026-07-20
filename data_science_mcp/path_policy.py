#!/usr/bin/python
"""Filesystem boundary for MCP-triggered data-science operations."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath

from agent_utilities.core import paths

_BUILTIN_DATASETS = {
    "breast_cancer",
    "california",
    "diabetes",
    "digits",
    "iris",
    "wine",
}


def data_root() -> Path:
    """Return the sole root available to MCP-triggered file reads and writes."""
    configured = os.environ.get("DATA_SCIENCE_DATA_ROOT")
    if configured and (
        len(configured) > 4096 or any(char in configured for char in "\x00\r\n")
    ):
        raise ValueError("Configured data root is invalid")
    try:
        root = (
            Path(configured).expanduser()
            if configured
            else paths.data_dir() / "data-science-mcp"
        )
        return root.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Configured data root is invalid") from exc


def resolve_data_path(value: str, *, must_exist: bool = False) -> Path:
    """Resolve ``value`` beneath :func:`data_root`, rejecting traversal/symlinks."""
    raw = str(value or "").strip()
    if (
        not raw
        or len(raw) > 4096
        or "\x00" in raw
        or "\r" in raw
        or "\n" in raw
    ):
        raise ValueError("Invalid data path")
    if PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
        raise ValueError("Data path uses a foreign absolute path")

    try:
        root = data_root()
        root.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as exc:
        raise ValueError("Configured data root is unavailable") from exc
    try:
        candidate = Path(raw).expanduser()
    except (OSError, RuntimeError) as exc:
        raise ValueError("Invalid data path") from exc
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        lexical = Path(os.path.abspath(candidate))
        relative = lexical.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("Data path escapes DATA_SCIENCE_DATA_ROOT") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("Data paths must not traverse symbolic links")
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError("Data path escapes DATA_SCIENCE_DATA_ROOT") from exc
    if resolved == root:
        raise ValueError("Data path must name a file")
    return resolved


def authorized_dataset_source(name: str) -> str | None:
    """Resolve an MCP CSV input; built-in and unknown logical names need no path."""
    if name.lower() in _BUILTIN_DATASETS or not name.lower().endswith(".csv"):
        return None
    return str(resolve_data_path(name, must_exist=True))


__all__ = ["authorized_dataset_source", "data_root", "resolve_data_path"]

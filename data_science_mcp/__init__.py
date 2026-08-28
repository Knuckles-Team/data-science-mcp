#!/usr/bin/env python

import importlib
import inspect
from typing import Any

__all__: list[str] = []

CORE_MODULES: list[str] = ["data_science_mcp.auth"]

OPTIONAL_MODULES = {
    "data_science_mcp.agent_server": "agent",
    "data_science_mcp.mcp_server": "mcp",
}


def _expose_members(module):
    """Expose public classes and functions from a module into globals and __all__."""
    for name, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) or inspect.isfunction(obj)) and not name.startswith(
            "_"
        ):
            globals()[name] = obj
            if name not in __all__:
                __all__.append(name)


# Eagerly import core modules (keeps API wrappers fast & light)
for module_name in CORE_MODULES:
    if module_name:
        module = importlib.import_module(module_name)
        _expose_members(module)

# Dynamic/lazy loading of optional modules (agent_server, mcp_server)
_loaded_optional_modules: dict[str, Any] = {}


def _import_module_safely(module_name: str):
    """Try to import a module and return it, or None if not available."""
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _module_available(keyword: str) -> bool:
    """Whether the ``OPTIONAL_MODULES`` entry whose name contains ``keyword`` imports."""
    key = next((k for k in OPTIONAL_MODULES if keyword in k), None)
    if key is None:
        return False
    return _import_module_safely(key) is not None


def _availability_flag(name: str) -> bool | None:
    """Resolve ``_MCP_AVAILABLE``/``_AGENT_AVAILABLE``; ``None`` for any other name."""
    if name == "_MCP_AVAILABLE":
        return _module_available("mcp_server")
    if name == "_AGENT_AVAILABLE":
        return _module_available("agent_server")
    return None


def _load_optional_module(module_name: str) -> Any:
    """Import + cache one optional module (lazily, once), exposing its public members."""
    if module_name not in _loaded_optional_modules:
        module = _import_module_safely(module_name)
        if module is not None:
            _loaded_optional_modules[module_name] = module
            _expose_members(module)
    return _loaded_optional_modules.get(module_name)


def __getattr__(name: str) -> Any:
    flag = _availability_flag(name)
    if flag is not None:
        return flag

    for module_name in OPTIONAL_MODULES:
        module = _load_optional_module(module_name)
        if module is not None and hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)

"""Serve the hand-written quant kernels one typed tool per action.

agent-connector-sdk 0.3.0 drops every registrar tool that routes on an
``action`` argument once per-operation client tools are registered, which
silently removed the eight quant_* tools (epistemic-graph finance kernels).
They are computations, not MLEngine operations, so no per-operation tool
covers them.

Each registrar runs against a scratch server; every action it implements
becomes ``<tool>_<action>`` with the registrar's own typed parameters (minus
``action``), so no tool takes ``action`` or an untyped ``params_json``.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable, Iterable
from typing import Any

from agent_connector_sdk.config import setting
from agent_connector_sdk.mcp.tool_surface import registered_tools
from fastmcp import FastMCP

#: Actions that only read or compute; everything else may change state.
#: Every quant kernel is a pure computation over its inputs.
_READ_ONLY = re.compile(r".")
_DESTRUCTIVE = re.compile(r"^$")
_COMPARE = re.compile(r"""action\s*==\s*["']([a-z][a-z0-9_]*)["']""")


def tool_actions(tool: Any) -> list[str]:
    """Actions a tool implements: the ``action == "..."`` branches in its body."""
    fn = inspect.unwrap(tool.fn)
    source = inspect.getsource(fn)
    seen: list[str] = []
    for name in _COMPARE.findall(source):
        if name not in seen:
            seen.append(name)
    return seen


def _annotations(action: str) -> dict[str, Any]:
    destructive = bool(_DESTRUCTIVE.match(action))
    read_only = bool(_READ_ONLY.match(action)) and not destructive
    return {
        "title": action.replace("_", " ").title(),
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": read_only,
        "openWorldHint": True,
    }


def _per_action(tool: Any, action: str) -> Callable[..., Any]:
    fn = tool.fn
    signature = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    parameters = [
        p.replace(annotation=hints.get(p.name, p.annotation))
        for p in signature.parameters.values()
        if p.name != "action"
    ]

    async def call(**kwargs: Any) -> Any:
        result = fn(action=action, **kwargs)
        return await result if inspect.isawaitable(result) else result

    call.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=hints.get("return", signature.return_annotation),
    )
    call.__annotations__ = {k: v for k, v in hints.items() if k != "action"}
    call.__name__ = action
    call.__doc__ = fn.__doc__
    return call


def serve_action_tools(
    mcp: Any,
    registry: Iterable[tuple[str, str, Callable[[Any], Any]]],
    *,
    rename: dict[str, str] | None = None,
) -> list[str]:
    """Run each enabled ``(tag, toggle, registrar)``; serve its tools per action.

    ``rename`` maps a registrar tool name to the prefix its per-action tools use.
    Returns the names registered.
    """
    scratch = FastMCP("quant-actions")
    for _tag, toggle, registrar in registry:
        if setting(toggle, True):
            registrar(scratch)
    served: list[str] = []
    for name, tool in registered_tools(scratch).items():
        if "action" not in (tool.parameters or {}).get("properties", {}):
            mcp.add_tool(tool)
            served.append(name)
            continue
        prefix = (rename or {}).get(name, name)
        for action in tool_actions(tool):
            tool_name = f"{prefix}_{action}"
            mcp.tool(
                name=tool_name,
                description=f"{action.replace('_', ' ').capitalize()} ({name}). "
                + (tool.description or ""),
                tags=set(tool.tags or set()) | {"quant"},
                annotations=_annotations(action),
            )(_per_action(tool, action))
            served.append(tool_name)
    return served

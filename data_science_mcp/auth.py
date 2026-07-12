#!/usr/bin/python
# coding: utf-8
"""Client accessor for the MCP tool surface.

data-science-mcp has no external REST API of its own — per ``AGENTS.md``, all ML
compute is delegated to the Rust ``epistemic-graph`` engine via
:class:`data_science_mcp.ml_engine.MLEngine` (a process-local singleton; see
``MLEngine._rust_client``/``.datascience`` namespace). ``mcp_server.py`` wires
``register_tool_surface(..., client_cls=MLEngine, get_client=get_client, ...)`` to
auto-derive the verbose 1:1 tool surface (``data_science_<method>``) from
``MLEngine``'s public methods, then dispatches each call as
``getattr(get_client(), method)(**kwargs)``. ``get_client`` therefore MUST return
an ``MLEngine`` instance so the introspected surface (``client_cls``) matches the
live dispatch target — an earlier generic REST-client placeholder here caused
every verbose tool (e.g. ``data_science_describe_dataset``) to fail with
``'Client' object has no attribute '<method>'``.
"""

from data_science_mcp.ml_engine import MLEngine


def get_client() -> MLEngine:
    """Return the shared :class:`MLEngine` singleton (the tool surface's client)."""
    return MLEngine()

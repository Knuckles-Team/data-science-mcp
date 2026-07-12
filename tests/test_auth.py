"""
Tests for checking the tool-surface client accessor in auth.py.

data-science-mcp has no external REST API of its own: ``register_tool_surface``
in ``mcp_server.py`` is wired with ``client_cls=MLEngine, get_client=get_client``
(see ``AGENTS.md`` — all compute delegates to the Rust epistemic-graph engine via
``MLEngine``). ``get_client()`` MUST therefore resolve to an ``MLEngine`` instance
so the verbose 1:1 tool surface (introspected from ``MLEngine``) dispatches to a
client that actually has those methods. A prior generic REST-client placeholder
here caused every verbose tool call (e.g. ``data_science_describe_dataset``) to
fail with ``'Client' object has no attribute '<method>'``.
"""

import data_science_mcp.auth as auth
from data_science_mcp.ml_engine import MLEngine


def test_get_client_returns_ml_engine():
    """get_client() must return an MLEngine instance, not an unrelated client."""
    client = auth.get_client()
    assert isinstance(client, MLEngine)


def test_get_client_is_singleton():
    """Repeated calls resolve to the same MLEngine singleton (shared dataset/model state)."""
    client1 = auth.get_client()
    client2 = auth.get_client()
    assert client1 is client2
    assert client1 is MLEngine()


def test_get_client_matches_tool_surface_client_cls():
    """get_client()'s type must match register_tool_surface's client_cls=MLEngine.

    This is the exact contract whose violation caused the describe_dataset bug:
    the verbose surface is introspected from client_cls but dispatched through
    get_client() — the two must agree on every public method name.
    """
    client = auth.get_client()
    assert type(client) is MLEngine
    assert hasattr(client, "describe_dataset")
    assert callable(client.describe_dataset)

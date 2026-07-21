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

The ``get_rest_client`` tests below cover the separate, optional TLS-hardened
upstream REST seam (:func:`data_science_mcp.auth.get_rest_client`) — it is not
part of the tool-surface contract and has no live caller today.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.core.exceptions import AuthError
import data_science_mcp.auth as auth
from data_science_mcp.ml_engine import MLEngine


@pytest.fixture(autouse=True)
def clean_rest_client_singleton():
    """Ensure data_science_mcp.auth._rest_client is clean before/after tests."""
    auth._rest_client = None
    yield
    auth._rest_client = None


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


def test_get_rest_client_singleton_uses_tls_profile():
    """The configured endpoint and injected TLS profile build one REST client."""
    profile = MagicMock()
    profile.configure_requests_session.side_effect = lambda session: session
    with patch.dict(os.environ, {"DATA_SCIENCE_MCP_URL": "https://service.invalid"}):
        client1 = auth.get_rest_client(profile)
    assert client1 is not None
    assert client1.base_url == "https://service.invalid"
    profile.configure_requests_session.assert_called_once()

    # Check singleton property
    client2 = auth.get_rest_client(profile)
    assert client1 is client2


def test_get_rest_client_custom_env():
    """Verify get_rest_client handles a custom endpoint and token."""
    profile = MagicMock()
    profile.configure_requests_session.side_effect = lambda session: session
    custom_env = {
        "DATA_SCIENCE_MCP_URL": "https://service.invalid",
        "DATA_SCIENCE_MCP_TOKEN": "my-secret-token",
    }
    with patch.dict(os.environ, custom_env):
        client = auth.get_rest_client(profile)
        assert client.base_url == "https://service.invalid"
        assert client.session.headers["Authorization"] == "Bearer my-secret-token"


def test_get_rest_client_auth_error():
    """Verify RuntimeError is raised when AuthError/UnauthorizedError occurs."""
    profile = MagicMock()
    with (
        patch.dict(os.environ, {"DATA_SCIENCE_MCP_URL": "https://service.invalid"}),
        patch("requests.Session") as mock_session_cls,
    ):
        # Make Session creation raise AuthError
        mock_session_cls.side_effect = AuthError("Mocked invalid auth token")

        with pytest.raises(RuntimeError) as exc_info:
            auth.get_rest_client(profile)

        assert "AUTHENTICATION ERROR" in str(exc_info.value)
        assert "Mocked invalid auth token" not in str(exc_info.value)

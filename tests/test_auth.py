"""
Tests for checking authentication functions in auth.py.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.core.exceptions import AuthError
import data_science_mcp.auth as auth


@pytest.fixture(autouse=True)
def clean_auth_singleton():
    """Ensure data_science_mcp.auth._client is clean before/after tests."""
    auth._client = None
    yield
    auth._client = None


def test_get_client_singleton_uses_tls_profile():
    """The configured endpoint and injected TLS profile build one client."""
    profile = MagicMock()
    profile.configure_requests_session.side_effect = lambda session: session
    with patch.dict(os.environ, {"DATA_SCIENCE_MCP_URL": "https://service.invalid"}):
        client1 = auth.get_client(profile)
    assert client1 is not None
    assert client1.base_url == "https://service.invalid"
    profile.configure_requests_session.assert_called_once()

    # Check singleton property
    client2 = auth.get_client(profile)
    assert client1 is client2


def test_get_client_custom_env():
    """Verify get_client handles a custom endpoint and token."""
    profile = MagicMock()
    profile.configure_requests_session.side_effect = lambda session: session
    custom_env = {
        "DATA_SCIENCE_MCP_URL": "https://service.invalid",
        "DATA_SCIENCE_MCP_TOKEN": "my-secret-token",
    }
    with patch.dict(os.environ, custom_env):
        client = auth.get_client(profile)
        assert client.base_url == "https://service.invalid"
        assert client.session.headers["Authorization"] == "Bearer my-secret-token"


def test_get_client_auth_error():
    """Verify RuntimeError is raised when AuthError/UnauthorizedError occurs."""
    profile = MagicMock()
    with (
        patch.dict(os.environ, {"DATA_SCIENCE_MCP_URL": "https://service.invalid"}),
        patch("requests.Session") as mock_session_cls,
    ):
        # Make Session creation raise AuthError
        mock_session_cls.side_effect = AuthError("Mocked invalid auth token")

        with pytest.raises(RuntimeError) as exc_info:
            auth.get_client(profile)

        assert "AUTHENTICATION ERROR" in str(exc_info.value)
        assert "Mocked invalid auth token" not in str(exc_info.value)

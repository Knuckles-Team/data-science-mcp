"""
Tests for checking authentication functions in auth.py.
"""

import os
import pytest
from unittest.mock import patch

from agent_utilities.core.exceptions import AuthError
import data_science_mcp.auth as auth


@pytest.fixture(autouse=True)
def clean_auth_singleton():
    """Ensure data_science_mcp.auth._client is clean before/after tests."""
    auth._client = None
    yield
    auth._client = None


def test_get_client_default():
    """Verify that get_client builds a singleton with default environment vars."""
    client1 = auth.get_client()
    assert client1 is not None
    assert client1.base_url == "http://localhost:8080"

    # Check singleton property
    client2 = auth.get_client()
    assert client1 is client2


def test_get_client_custom_env():
    """Verify get_client handles custom URL, TOKEN, and VERIFY variables."""
    custom_env = {
        "DATA_SCIENCE_MCP_URL": "https://custom-mcp-server:9000",
        "DATA_SCIENCE_MCP_TOKEN": "my-secret-token",
        "DATA_SCIENCE_MCP_VERIFY": "false",
    }
    with patch.dict(os.environ, custom_env):
        client = auth.get_client()
        assert client.base_url == "https://custom-mcp-server:9000"
        assert client.session.headers["Authorization"] == "Bearer my-secret-token"
        assert client.session.verify is False

    # Test verify true alternatives
    auth._client = None
    with patch.dict(os.environ, {"DATA_SCIENCE_MCP_VERIFY": "yes"}):
        client_yes = auth.get_client()
        assert client_yes.session.verify is True

    auth._client = None
    with patch.dict(os.environ, {"DATA_SCIENCE_MCP_VERIFY": "1"}):
        client_1 = auth.get_client()
        assert client_1.session.verify is True


def test_get_client_auth_error():
    """Verify RuntimeError is raised when AuthError/UnauthorizedError occurs."""
    with patch("requests.Session") as mock_session_cls:
        # Make Session creation raise AuthError
        mock_session_cls.side_effect = AuthError("Mocked invalid auth token")

        with pytest.raises(RuntimeError) as exc_info:
            auth.get_client()

        assert "AUTHENTICATION ERROR" in str(exc_info.value)
        assert "Mocked invalid auth token" in str(exc_info.value)

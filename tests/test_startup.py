"""
Tests for checking module initialization, lazy loading, and startup server scripts.
"""

from unittest.mock import MagicMock, patch

# Import data_science_mcp to verify its lazy imports
import pytest












@patch("data_science_mcp.mcp_server.get_mcp_instance")
def test_mcp_server_transports(mock_get_mcp):
    """Verify different transport options and invalid transport in mcp_server.py."""
    from data_science_mcp.mcp_server import mcp_server

    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.host = "localhost"
    mock_args.port = 8000
    mock_args.auth_type = "none"

    # Test streamable-http
    mock_args.transport = "streamable-http"
    mock_get_mcp.return_value = (mock_mcp, mock_args, [], [])
    with patch("sys.stdout", MagicMock()), patch("sys.stderr", MagicMock()):
        mcp_server()
    mock_mcp.run.assert_called_with(
        transport="streamable-http", host="localhost", port=8000
    )

    # Test sse
    mock_args.transport = "sse"
    mock_get_mcp.return_value = (mock_mcp, mock_args, [], [])
    with patch("sys.stdout", MagicMock()), patch("sys.stderr", MagicMock()):
        mcp_server()
    mock_mcp.run.assert_called_with(transport="sse", host="localhost", port=8000)

    # Test invalid transport
    mock_args.transport = "invalid-transport"
    mock_get_mcp.return_value = (mock_mcp, mock_args, [], [])
    with (
        patch("sys.stdout", MagicMock()),
        patch("sys.stderr", MagicMock()),
        pytest.raises(SystemExit) as excinfo,
    ):
        mcp_server()
    assert excinfo.value.code == 1



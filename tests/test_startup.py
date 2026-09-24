"""
Tests for checking module initialization, lazy loading, and startup server scripts.
"""

import runpy
import sys
from unittest.mock import MagicMock, patch

# Import data_science_mcp to verify its lazy imports
import data_science_mcp
import pytest


def test_lazy_loading_and_getattr():
    """Verify dynamic availability flags and lazy attribute lookup in __init__.py."""
    # Test availability flags. agent_server.py is retired (EH-480 policy
    # update), so _AGENT_AVAILABLE is always False now.
    assert data_science_mcp._MCP_AVAILABLE is True
    assert data_science_mcp._AGENT_AVAILABLE is False

    # Test __dir__ contains lazy module and properties
    attrs = dir(data_science_mcp)
    assert "get_mcp_instance" in attrs

    # Test AttributeError on invalid attributes
    with pytest.raises(AttributeError, match="has no attribute"):
        data_science_mcp.non_existent_attribute_name

    # Check lazy sub-module references
    assert data_science_mcp.mcp_server is not None


@patch("data_science_mcp.mcp_server.mcp_server")
def test_main_execution(mock_mcp_server):
    """Verify that running the __main__.py module invokes mcp_server
    (agent_server.py retired, EH-480 policy update)."""
    runpy.run_module("data_science_mcp.__main__", run_name="__main__")
    mock_mcp_server.assert_called_once()


def test_init_file_branches():
    """Verify other branches of data_science_mcp/__init__.py."""
    import data_science_mcp

    # 1. Test _import_module_safely with non-existent module
    res = data_science_mcp._import_module_safely("data_science_mcp.non_existent")
    assert res is None

    # 2. Test __getattr__ dynamic attribute retrieval
    # Retrieve get_mcp_instance which is a function in mcp_server.py
    func = data_science_mcp.get_mcp_instance
    assert callable(func)

    # 3. Test _MCP_AVAILABLE and _AGENT_AVAILABLE when OPTIONAL_MODULES is empty
    with patch.dict(data_science_mcp.OPTIONAL_MODULES, {}, clear=True):
        assert data_science_mcp._MCP_AVAILABLE is False
        assert data_science_mcp._AGENT_AVAILABLE is False

    # 4. Test CORE_MODULES loading
    # Since data_science_mcp.auth is natively in CORE_MODULES, it is loaded during import
    assert "get_client" in data_science_mcp.__all__


@patch("agent_connector_sdk.mcp.server.create_mcp_server")
def test_mcp_server_entrypoint_main(mock_create_server):
    """Verify mcp_server.py __main__ entrypoint execution."""
    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.transport = "sse"
    mock_args.host = "127.0.0.1"
    mock_args.port = 8000
    mock_args.auth_type = "none"
    mock_create_server.return_value = (mock_args, mock_mcp, [])

    # Patch sys.argv to avoid argparse errors and run the module as main
    test_argv = ["mcp_server", "--transport", "sse"]
    with (
        patch.object(sys, "argv", test_argv),
        patch("sys.stdout", MagicMock()),
        patch("sys.stderr", MagicMock()),
    ):
        runpy.run_module("data_science_mcp.mcp_server", run_name="__main__")

    mock_mcp.run.assert_called_once_with(transport="sse", host="127.0.0.1", port=8000)


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

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
    # Test availability flags
    assert data_science_mcp._MCP_AVAILABLE is True
    assert data_science_mcp._AGENT_AVAILABLE is True

    # Test __dir__ contains lazy module and properties
    attrs = dir(data_science_mcp)
    assert "agent_server" in attrs
    assert "get_mcp_instance" in attrs

    # Test AttributeError on invalid attributes
    with pytest.raises(AttributeError, match="has no attribute"):
        data_science_mcp.non_existent_attribute_name

    # Check lazy sub-module references
    assert data_science_mcp.mcp_server is not None
    assert data_science_mcp.agent_server is not None


@patch("agent_utilities.initialize_workspace")
@patch("agent_utilities.load_identity")
@patch("agent_utilities.build_system_prompt_from_workspace")
@patch("agent_utilities.create_agent_server")
def test_agent_server_startup(
    mock_create_server,
    mock_build_system_prompt,
    mock_load_identity,
    mock_init_workspace,
):
    """Test full command line arguments parsing and server bootstrap in agent_server.py."""
    # Mock metadata values
    mock_load_identity.return_value = {
        "name": "Data Science Test Agent",
        "description": "A test data science agent",
        "content": None,
    }
    mock_build_system_prompt.return_value = "Mock system prompt from workspace"

    # Mock command line arguments
    test_argv = [
        "agent_server",
        "--mcp-url",
        "http://localhost:8000",
        "--mcp-config",
        "mcp_config_test.json",
        "--host",
        "127.0.0.1",
        "--port",
        "5000",
        "--provider",
        "openai",
        "--model-id",
        "gpt-4",
        "--base-url",
        "http://api.openai.com",
        "--api-key",
        "test-api-key",
        "--custom-skills-directory",
        "/test/skills",
        "--web",
        "--otel",
        "--otel-endpoint",
        "http://otel",
        "--otel-headers",
        "{}",
        "--otel-public-key",
        "pk",
        "--otel-secret-key",
        "sk",
        "--otel-protocol",
        "grpc",
        "--debug",
    ]

    with patch.object(sys, "argv", test_argv):
        from data_science_mcp.agent_server import agent_server

        agent_server()

    # Verify that initialization was run
    mock_init_workspace.assert_called_once()
    mock_load_identity.assert_called_once()
    mock_build_system_prompt.assert_called_once()

    # Verify create_agent_server args passing
    mock_create_server.assert_called_once_with(
        mcp_url="http://localhost:8000",
        mcp_config="mcp_config_test.json",
        host="127.0.0.1",
        port=5000,
        provider="openai",
        model_id="gpt-4",
        router_model="gpt-4",
        agent_model="gpt-4",
        base_url="http://api.openai.com",
        api_key="test-api-key",
        custom_skills_directory="/test/skills",
        enable_web_ui=True,
        enable_otel=True,
        otel_endpoint="http://otel",
        otel_headers="{}",
        otel_public_key="pk",
        otel_secret_key="sk",
        otel_protocol="grpc",
        debug=True,
    )


@patch("data_science_mcp.agent_server.agent_server")
def test_main_execution(mock_agent_server):
    """Verify that running the __main__.py module invokes the agent server."""
    runpy.run_module("data_science_mcp.__main__", run_name="__main__")
    mock_agent_server.assert_called_once()


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

    # Trigger __getattr__ with an already loaded module for attribute that is NOT in globals
    # DEFAULT_AGENT_NAME is a variable inside agent_server, not class/function, so not in globals.
    # This exercises line 69 (return getattr(module, name))
    val = data_science_mcp.DEFAULT_AGENT_NAME
    assert val is None or isinstance(val, str)

    # 3. Test _MCP_AVAILABLE and _AGENT_AVAILABLE when OPTIONAL_MODULES is empty
    with patch.dict(data_science_mcp.OPTIONAL_MODULES, {}, clear=True):
        assert data_science_mcp._MCP_AVAILABLE is False
        assert data_science_mcp._AGENT_AVAILABLE is False

    # 4. Test CORE_MODULES loading
    # Since data_science_mcp.auth is natively in CORE_MODULES, it is loaded during import
    assert "get_client" in data_science_mcp.__all__


@patch("agent_utilities.mcp.server_factory.create_mcp_server")
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


@patch("agent_utilities.create_agent_server")
@patch("agent_utilities.load_identity")
@patch("agent_utilities.initialize_workspace")
def test_agent_server_entrypoint_main(
    mock_init_workspace, mock_load_identity, mock_create_agent_server
):
    """Verify agent_server.py __main__ entrypoint execution."""
    mock_load_identity.return_value = {
        "name": "Data Science Test Agent",
        "description": "A test data science agent",
        "content": None,
    }
    with patch.object(sys, "argv", ["agent_server"]):
        runpy.run_module("data_science_mcp.agent_server", run_name="__main__")
    mock_create_agent_server.assert_called_once()

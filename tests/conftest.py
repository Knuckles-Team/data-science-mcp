"""Shared test fixtures for Data Science Mcp."""

import importlib.util

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("DATA_URL", "https://test.example.com")
    monkeypatch.setenv("DATA_TOKEN", "test-token-12345")
    monkeypatch.setenv("DATA_SSL_VERIFY", "False")


@pytest.fixture
def require_engine():
    """Skip a test unless the epistemic-graph compute engine is reachable.

    Compute (fit/predict/evaluate/cross-validate) runs entirely in the engine;
    point EPISTEMIC_GRAPH_SOCKET / EPISTEMIC_GRAPH_TCP at a running server to
    exercise these integration tests in CI.
    """
    from data_science_mcp.ml_engine import MLEngine, _UNPROBED

    MLEngine._rust_client_cache = _UNPROBED  # re-probe with current env
    if MLEngine._rust_client() is None:
        pytest.skip("epistemic-graph engine not reachable")


@pytest.fixture
def require_sklearn():
    """Skip a test unless scikit-learn is installed (optional `[datasets]` extra,
    used only by the built-in sample-dataset loaders)."""
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("scikit-learn not installed (install data-science-mcp[datasets])")

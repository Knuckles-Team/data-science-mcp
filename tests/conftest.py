"""Shared test fixtures for Data Science Mcp."""

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("DATA_URL", "https://test.example.com")
    monkeypatch.setenv("DATA_TOKEN", "test-token-12345")
    monkeypatch.setenv("DATA_SSL_VERIFY", "False")

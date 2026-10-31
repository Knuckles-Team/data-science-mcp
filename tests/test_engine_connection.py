"""Engine connection through the SDK compute adapter (CONN-DATA-SCIENCE-MCP-R001)."""

import pytest

from data_science_mcp import ml_engine
from data_science_mcp.ml_engine import _UNPROBED, ENGINE_SCOPES, MLEngine

pytestmark = pytest.mark.spec("CONN-DATA-SCIENCE-MCP-R001")


@pytest.fixture(autouse=True)
def _fresh_probe():
    MLEngine._rust_client_cache = _UNPROBED
    yield
    MLEngine._rust_client_cache = _UNPROBED


def test_connects_with_a_signed_compute_context(monkeypatch):
    seen = {}

    class _Compute:
        client = object()

    def _connect(*, scopes):
        seen["scopes"] = scopes
        return _Compute()

    monkeypatch.setattr("scion_sdk.compute.engine.connect_compute", _connect)
    assert MLEngine._rust_client() is _Compute.client
    assert seen["scopes"] == ENGINE_SCOPES
    assert "compute:datascience" in ENGINE_SCOPES


def test_unconfigured_engine_is_cached_as_unavailable(monkeypatch, caplog):
    for name in ("EPISTEMIC_GRAPH_ENDPOINT", "EPISTEMIC_GRAPH_AUTH_SECRET_REF"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def _connect(*, scopes):
        calls.append(scopes)
        from scion_sdk.compute.engine import ComputeUnavailableError

        raise ComputeUnavailableError("EPISTEMIC_GRAPH_ENDPOINT is not configured")

    monkeypatch.setattr("scion_sdk.compute.engine.connect_compute", _connect)
    with caplog.at_level("WARNING", logger=ml_engine.logger.name):
        assert MLEngine._rust_client() is None
        assert MLEngine._rust_client() is None
    assert len(calls) == 1
    assert "ComputeUnavailableError" in caplog.text
    assert "EPISTEMIC_GRAPH_ENDPOINT" in ml_engine._ENGINE_REQUIRED_ERR

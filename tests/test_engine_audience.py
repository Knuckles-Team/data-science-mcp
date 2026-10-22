"""Engine request audience comes from the deployment (CONN-DATA-SCIENCE-MCP-R002)."""

import pytest
from agent_connector_sdk.ingest.engine import EngineSettings

pytestmark = pytest.mark.spec("CONN-DATA-SCIENCE-MCP-R002")


def test_signed_context_carries_the_configured_audience(monkeypatch):
    for key, value in {
        "EPISTEMIC_GRAPH_ENDPOINT": "tcp://127.0.0.1:9100",
        "EPISTEMIC_GRAPH_AUTH_SECRET_REF": "env://EG_SECRET",
        "EPISTEMIC_GRAPH_TENANT": "tenant-a",
        "EPISTEMIC_GRAPH_GRAPH": "kg",
        "EPISTEMIC_GRAPH_AUDIENCE": "agent-services",
    }.items():
        monkeypatch.setenv(key, value)
    settings = EngineSettings.from_settings()
    assert settings is not None
    assert settings.verified_context()["audience"] == "agent-services"

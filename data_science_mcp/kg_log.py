"""Best-effort provenance logging through the SDK knowledge-ingest path.

Dataset-version and training-run provenance records are pushed to epistemic-graph
as typed entities with ``agent_connector_sdk.ingest``. Provenance is telemetry:
when no engine is configured or reachable the write is skipped and never blocks
training or data preparation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("data_science_mcp.kg_log")


def _entity_id(payload: dict[str, Any]) -> str:
    kind = str(payload.get("kind", "Record"))
    ident = payload.get("run_name") or "{}:{}".format(
        payload.get("name", ""), payload.get("version", "")
    )
    if not ident or ident == ":":
        ident = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
    return f"data-science:{kind}:{ident}"


def log_provenance(payload: dict[str, Any]) -> bool:
    """Push one provenance payload; returns whether it was committed."""
    try:  # pragma: no cover - requires a live engine
        from agent_connector_sdk.ingest import (
            ChangeSet,
            Entity,
            IngestBinding,
            ingest_changes,
        )

        entity = Entity(
            id=_entity_id(payload),
            node_type=str(payload.get("kind", "Record")),
            properties={k: v for k, v in payload.items() if k != "kind"},
        )
        ingest_changes(
            IngestBinding(connector="data-science-mcp", stream="provenance"),
            ChangeSet(entities=(entity,)),
        )
        return True
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("Operation failed: error_type=%s", type(e).__name__)
        return False

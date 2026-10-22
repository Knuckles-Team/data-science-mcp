# CONN-DATA-SCIENCE-MCP — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Existing system and reuse

The SDK compute adapter `connect_compute` reads the shared engine settings and signs a verified request context. The connector used a direct client call without that context.

## Architecture

`MLEngine._rust_client` in `data_science_mcp/ml_engine.py` calls `connect_compute` with the data-science and finance compute scopes. It caches the client or the failure.

## Interfaces and data model

Configuration moves to `EPISTEMIC_GRAPH_ENDPOINT`, `EPISTEMIC_GRAPH_AUTH_SECRET_REF`, `EPISTEMIC_GRAPH_TENANT` and `EPISTEMIC_GRAPH_GRAPH`. The old socket and TCP variables are removed.

## Live integration path

The fleet gateway lists the tool, reads its annotation and calls it as the gateway identity.

## Quality and release gates

The focused tests run with `pytest -q`. A merge to the default branch publishes the package and the image.

## Risks, alternatives, and decisions

The engine must admit the connector principal for the compute scopes. A refused admission shows as an engine error on the tool result.

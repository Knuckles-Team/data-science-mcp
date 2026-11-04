# CONN-DATA-SCIENCE-MCP — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The SDK compute adapter `connect_compute` reads the shared engine settings and signs a verified request context. The connector used a direct client call without that context. `MLEngine._rust_client` in `data_science_mcp/ml_engine.py` calls `connect_compute` with the data-science and finance compute scopes. It caches the client or the failure.

## Technical Context

- **Language/Version**: Python 3.14 (floor 3.12)
- **Primary Dependencies**: scion-sdk, fastmcp
- **Storage**: N/A; the connector stores nothing
- **Testing**: pytest; the proof level each row names
- **Target Platform**: Linux containers
- **Project Type**: connector (MCP server)
- **Performance Goals**: N/A
- **Constraints**: one tool per API operation; shared capabilities come from the SDK
- **Scale/Scope**: 3 rows

## Constitution Check

- One owner per concept: pass. The connector owns one external system's calls, mappings and tools only.
- Connectors depend on the SDK and are called through arbora: pass.
- No ratchets: pass. The rows add no suppression.

## Project Structure

```text
data_science_mcp/ml_engine.py  existing
data_science_mcp/mcp_server.py  existing
data_science_mcp/trainers/eval_hooks.py  existing
data_science_mcp/mcp/mcp_trainers.py  existing
data_science_mcp/__init__.py  existing
```

## Phase 0: Research

No unknowns. Each row names the code it touches.

## Phase 1: Design

### Interfaces and data model

Configuration moves to `MEDULLA_ENDPOINT`, `MEDULLA_AUTH_SECRET_REF`, `MEDULLA_TENANT` and `MEDULLA_GRAPH`. The old socket and TCP variables are removed.

### Live integration path

The fleet gateway lists the tool, reads its annotation and calls it as the gateway identity.

### Quality and release gates

The focused tests run with `pytest -q`. A merge to the default branch publishes the package and the image.

### Risks, alternatives, and decisions

The engine must admit the connector principal for the compute scopes. A refused admission shows as an engine error on the tool result.

## Phase 2: Task approach

`tasks.md` holds one task line per row, producer first.

## Complexity Tracking

None. The constitution check passed.

## Build contract

The repository pre-commit hooks apply: `ruff-format`, `connector-standard`, `secrets`, `tests-fast`. Each row names its proving test. No suppression and no duplicate implementation.

## Production caller chain

1. `CONN-DATA-SCIENCE-MCP-R001`: The `data-science-mcp` console script runs `scion_sdk.cli:serve`, which loads `data_science_mcp/__init__.py::connector` (api=`MLEngine`); tools such as `fit_model` in `data_science_mcp/mcp_server.py` call `data_science_mcp/ml_engine.py::MLEngine._rust_client` (line 97), which calls `scion_sdk.compute.engine.connect_compute(scopes=ENGINE_SCOPES)`.
2. `CONN-DATA-SCIENCE-MCP-R002`: The `data-science-mcp` console script runs `scion_sdk.cli:serve`; `data_science_mcp/ml_engine.py::MLEngine._rust_client` calls `scion_sdk.compute.engine.connect_compute`, which builds `scion_sdk.ingest.engine.EngineSettings.from_settings` and signs the context through `verified_context()`.
3. `CONN-DATA-SCIENCE-MCP-R003`: (planned) The `data-science-mcp` console script runs `scion_sdk.cli:serve` and loads `data_science_mcp/__init__.py::connector`; the new `upload_teacher_scores` tool will be added through its `tool_registry`, and `data_science_mcp/trainers/eval_hooks.py::grounded_scorer` (line 31) exists today but nothing calls an uploader.

## Cross-repo seams

none. The cited owner rows are library capabilities the connector calls in process; spec.md names them.

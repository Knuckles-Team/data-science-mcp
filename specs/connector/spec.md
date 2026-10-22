# CONN-DATA-SCIENCE-MCP — Engine compute connection

Owner: data-science-mcp. Cross-repo IDs: none.

## Purpose and user stories

A model or quant tool must run its compute in the graph engine. The connector must reach the engine with a signed request context.

## Requirements and acceptance

- FR-001: The connector must connect to the engine through the SDK compute adapter in `data_science_mcp/ml_engine.py`. The connection must sign the data-science and finance compute scopes and cache one probe result.
- FR-002: The connector must sign engine requests with the audience in `EPISTEMIC_GRAPH_AUDIENCE`. The deployment must set it to the audience the engine expects.
- SC-001: the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only.
- SC-002: the signed context carries the configured audience.

## Why a new spec

The `specs/` tree held only the template and the index.
No local spec owned this behavior, so this spec is the first capability spec.

## Scope and interfaces

The connector owns its operations and their MCP tool annotations.
The connector SDK owns tool registration, transport and the shared client contracts.
Code: ../../data_science_mcp/ml_engine.py ../../tests/test_engine_connection.py ../../tests/test_engine_audience.py

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| FR-001 | plan.md#architecture | T-001 | `tests/test_engine_connection.py` |
| FR-002 | plan.md#architecture | T-002 | `tests/test_engine_audience.py` |

## Open questions

- None.

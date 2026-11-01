# CONN-DATA-SCIENCE-MCP — Engine compute connection

Owner: data-science-mcp. Cross-repo IDs: none.

## Purpose and user stories

A model or quant tool must run its compute in the graph engine. The connector must reach the engine with a signed request context.

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

## User Scenarios & Testing

### User Story 1 - Engine compute connection (Priority: P1)

A model or quant tool must run its compute in the graph engine. The connector must reach the engine with a signed request context.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-DATA-SCIENCE-MCP-R001` (signed engine compute connection) is exercised, **Then** `tests/test_engine_connection.py` must show: the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only. Test: `tests/test_engine_connection.py::test_connects_with_a_signed_compute_context`.
2. **Given** the connector is configured, **When** the behaviour of row `CONN-DATA-SCIENCE-MCP-R002` (engine audience from the deployment) is exercised, **Then** `tests/test_engine_audience.py` must show: the signed context carries the configured audience. Test: `tests/test_engine_audience.py::test_signed_context_carries_the_configured_audience`.
3. **Given** the connector is configured, **When** the behaviour of row `CONN-DATA-SCIENCE-MCP-R003` (served tool uploads teacher scores) is exercised, **Then** served tool uploads teacher scores.

### Edge Cases

- If `CONN-DATA-SCIENCE-MCP-R001` is not met, its proving test fails: signed engine compute connection.
- If `CONN-DATA-SCIENCE-MCP-R002` is not met, its proving test fails: engine audience from the deployment.
- If `CONN-DATA-SCIENCE-MCP-R003` is not met, its proving test fails: served tool uploads teacher scores.

## Requirements

- **FR-001**: Signed engine compute connection. Rows: `CONN-DATA-SCIENCE-MCP-R001`.
- **FR-002**: Engine audience from the deployment. Rows: `CONN-DATA-SCIENCE-MCP-R002`.
- **FR-003**: Served tool uploads teacher scores. Rows: `CONN-DATA-SCIENCE-MCP-R003`.

## Success Criteria

- **SC-001**: `tests/test_engine_connection.py` must show: the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only. Rows: `CONN-DATA-SCIENCE-MCP-R001`.
- **SC-002**: `tests/test_engine_audience.py` must show: the signed context carries the configured audience. Rows: `CONN-DATA-SCIENCE-MCP-R002`.
- **SC-003**: Served tool uploads teacher scores. Rows: `CONN-DATA-SCIENCE-MCP-R003`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-DATA-SCIENCE-MCP-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |
| FR-002 | `CONN-DATA-SCIENCE-MCP-R002` | [plan.md](plan.md) | T-002 | [test-spec.md](test-spec.md) |
| FR-003 | `CONN-DATA-SCIENCE-MCP-R003` | [plan.md](plan.md) | T-003 | [test-spec.md](test-spec.md) |

# CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS — Operations the data-platform vocabulary needs

Owner: data-science-mcp. Cross-repo IDs: `MEDULLA-DATAPLAT-VOCAB-R009`.

## Purpose and user stories

The medulla data-platform vocabulary answers competency questions from connector facts. One question needs an operation this connector lacks. This spec owns that operation.

## Why a new spec

The existing specs in `specs/` cover packaged content and identity, not tool operations. None owns a new connector operation.

## Scope and interfaces

Owned: the one operation named in the row, touching `data_science_mcp/mcp_server.py`. Non-goal: anything in the medulla vocabulary itself.

## Traceability

| Requirement | Design section | Test ID | Evidence |
|---|---|---|---|
| CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001 | plan.md | T-001 |  |

## Open questions

- None.

## User Scenarios & Testing

### User Story 1 - Operations the data-platform vocabulary needs (Priority: P1)

The medulla data-platform vocabulary answers competency questions from connector facts. One question needs an operation this connector lacks. This spec owns that operation.

**Acceptance Scenarios**:

1. **Given** the connector is configured, **When** the behaviour of row `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001` (run-registry read) is exercised, **Then** fixture runs list returns metrics and a baseline id. Planned test: `tests/test_mcp_tools.py::test_run_registry_read_returns_metrics_and_baseline`.

### Edge Cases

- If `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001` is not met, its proving test fails: run-registry read.

## Requirements

- **FR-001**: Run-registry read. Rows: `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001`.

## Success Criteria

- **SC-001**: Fixture runs list returns metrics and a baseline id. Planned test: `tests/test_mcp_tools.py::test_run_registry_read_returns_metrics_and_baseline`. Rows: `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001`.

## Assumptions

- The external system is reachable with operator-supplied credentials.
- The SDK and engine contracts the rows cite are at the floors declared in `pyproject.toml`.

## Traceability

| Label | Rows | Design section | Test ID | Evidence |
|---|---|---|---|---|
| FR-001 | `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001` | [plan.md](plan.md) | T-001 | [test-spec.md](test-spec.md) |

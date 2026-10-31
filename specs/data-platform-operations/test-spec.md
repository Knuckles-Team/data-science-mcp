# CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001 | Contract | A fixture backend | fixture runs list returns metrics and a baseline id. |  |

## Negative and boundary cases

A backend without the capability returns a typed unsupported result.

## Quality and release proof

Run the focused test with `uv run pytest` on the new test file.

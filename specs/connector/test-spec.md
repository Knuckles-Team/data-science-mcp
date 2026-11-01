# CONN-DATA-SCIENCE-MCP — Test specification

Governing spec: [spec.md](spec.md).

| Test ID | Requirement | Level | Setup and input | Expected observation | Evidence |
|---|---|---|---|---|---|
| T-001 | FR-001 | Unit | Mocked HTTP | the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only. | `tests/test_engine_connection.py` |
| T-002 | FR-002 | Unit | Engine settings from the environment | the signed context carries the configured audience. | `tests/test_engine_audience.py` |

## Negative and boundary cases

The tests cover the error path that each row names.

## Quality and release proof

Run `pytest -q`. The served proof is the fleet acceptance check through the gateway.

## Rows awaiting bound tests
- `CONN-DATA-SCIENCE-MCP-R003`: planned proof `tests/test_distilled_scores.py::test_tool_uploads_median_scores_with_bounds`. Expected: A recording uploader receives the median 16-bit fixed-point score with minimum and maximum bounds per document.. Refusal: An unknown scorer_id is refused with SCORER_UNKNOWN and nothing is uploaded..

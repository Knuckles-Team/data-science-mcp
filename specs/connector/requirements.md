# CONN-DATA-SCIENCE-MCP — Requirements

Governing spec: [spec.md](spec.md).

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-DATA-SCIENCE-MCP-R001` | **Signed engine compute connection.** The connector must connect to the engine through the SDK compute adapter in `data_science_mcp/ml_engine.py`. The connection must sign the data-science and finance compute scopes and cache one probe result. | `tests/test_engine_connection.py` must show: the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only. |
| `CONN-DATA-SCIENCE-MCP-R002` | **Engine audience from the deployment.** The connector must sign engine requests with the audience in `EPISTEMIC_GRAPH_AUDIENCE`. The deployment must set it to the audience the engine expects. | `tests/test_engine_audience.py` must show: the signed context carries the configured audience. |
| `CONN-DATA-SCIENCE-MCP-R003` | **Served tool uploads teacher scores.** Subsystem: served tool. Tool `upload_teacher_scores(scorer_id, scorer_revision, documents)` in `data_science_mcp`. For each document it runs the named scorer from `data_science_mcp/trainers/eval_hooks.py` (`Scorer`, default `grounded_scorer`) `DISTILL_SAMPLES` times, default 3, range [1, 9]. The score is the median in 16-bit fixed point. The fidelity bounds are the minimum and maximum. A deterministic scorer gives equal bounds. It uploads through `DistilledScoreUploader` of `SCION-CONNECTOR-CONTROL-R905`. It needs the actor-required flag of `SCION-CONNECTOR-CONTROL-R902.1`. Refusal: `SCORER_UNKNOWN`. Depends on: SCION-CONNECTOR-CONTROL-R905. | Test: `tests/test_distilled_scores.py::test_tool_uploads_median_scores_with_bounds` (L3) calls the served tool with a recording uploader. |

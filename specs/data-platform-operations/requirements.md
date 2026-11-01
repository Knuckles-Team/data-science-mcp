# CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001` | **Run-registry read.** Returns training runs with their dataset, validation metrics and baseline run. Touches `data_science_mcp/mcp_server.py`. Moved from `MEDULLA-DATAPLAT-VOCAB-R009.5` (medulla). Caller chain: (planned) The `data-science-mcp` console script runs `scion_sdk.cli:serve` and loads `data_science_mcp/__init__.py::connector`; the new read tool will register through `tool_registry` next to `register_model_evolution_tools` in `data_science_mcp/mcp_server.py`, and run records today live only in `data_science_mcp/tracking.py::RunTracker`. Shape: Add a read-only tool in `data_science_mcp/mcp_server.py` returning run records from `RunTracker.provenance` and `records` (dataset_version, summary metrics, parent_run as baseline); a persisted run store does not exist, so the fixture supplies the runs. | fixture runs list returns metrics and a baseline id. Planned test: `tests/test_mcp_tools.py::test_run_registry_read_returns_metrics_and_baseline`. |

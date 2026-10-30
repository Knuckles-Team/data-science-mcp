# CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS — Requirements

Governing spec: [spec.md](spec.md).

Each row is one requirement. A row names the file it touches and the test that proves it. The generator sets each row's state in `status.json`.

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-DATA-SCIENCE-MCP-DATA-PLATFORM-OPERATIONS-R001` | **Run-registry read.** Returns training runs with their dataset, validation metrics and baseline run. Touches `data_science_mcp/mcp_server.py`. Moved from `EG-DATAPLAT-VOCAB-R009.5` (medulla). | fixture runs list returns metrics and a baseline id. |

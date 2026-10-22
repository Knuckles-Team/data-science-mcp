# CONN-DATA-SCIENCE-MCP — Requirements

Governing spec: [spec.md](spec.md).

| ID | Requirement | Acceptance |
|---|---|---|
| `CONN-DATA-SCIENCE-MCP-R001` | **Signed engine compute connection.** The connector must connect to the engine through the SDK compute adapter in `data_science_mcp/ml_engine.py`. The connection must sign the data-science and finance compute scopes and cache one probe result. | `tests/test_engine_connection.py` must show: the adapter receives the compute scopes, the client is returned, and an unconfigured engine is probed once and logged by error type only. |

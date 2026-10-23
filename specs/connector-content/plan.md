# CONN-DATA-SCIENCE-MCP-CONNECTOR-CONTENT — Design and implementation plan

Governing spec: [spec.md](spec.md).

## Architecture

`pyproject.toml` declares `data-science-mcp` as the connector identity. `create_mcp_server` resolves the packaged content of `data_science_mcp` from it.

# data-science-mcp — Concept Overview

> **Category**: Intelligence | **Ecosystem Role**: MCP Server + A2A Agent
> Built on [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) — the unified AGI Harness.

## Description

Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AHE-3.8).

## Enterprise Readiness

All agents in the ecosystem inherit enterprise-grade infrastructure from `agent-utilities`:

| Feature | Status | Source |
|:--------|:-------|:-------|
| **JWT/OIDC Authentication** | ✅ Built-in | `agent-utilities[auth]` — Authlib JWKS + API key middleware |
| **OpenTelemetry Instrumentation** | ✅ Built-in | `agent-utilities[logfire]` — OTLP export, FastAPI auto-instrumentation |
| **HashiCorp Vault Integration** | ✅ Built-in | `agent-utilities[vault]` — `secret://`, `env://`, `vault://` URI schemes |
| **Audit Logging** | ✅ Built-in | Append-only compliance trail with 30+ action types (CONCEPT:OS-5.4) |
| **Token Usage Analytics** | ✅ Built-in | 4-bucket tracking with budget alerting (CONCEPT:OS-5.4) |
| **Prompt Injection Defense** | ✅ Built-in | 25+ pattern scanner + jailbreak taxonomy (CONCEPT:OS-5.1) |
| **Guardrail Engine** | ✅ Built-in | Input/output interception with block/redact/warn (CONCEPT:OS-5.3) |
| **Action Execution Pipeline** | ✅ Built-in | Token, cost, duration, and node transition limits Dry-run / commit / rollback phases (CONCEPT:ORCH-1.4) |
| **Resource Scheduling** | ✅ Built-in | Priority queuing + preemption limits (CONCEPT:OS-5.2) |
| **Session Concurrency** | ✅ Built-in | Enqueue/reject/interrupt/rollback (CONCEPT:OS-5.3) |

## LLM Trainer

Beyond engine-backed classical ML, this project is the **agent-driven LLM trainer**
for the ecosystem — it can **create, pretrain from random init, and fine-tune** models:
robust SFT/DPO/GRPO with precision/accumulation/clipping/scheduling/checkpoint-resume
and FSDP+DeepSpeed scale-out (`CONCEPT:ML-001/005`), a corpus curation engine
(`CONCEPT:ML-002`), pretraining from scratch with a trained BPE tokenizer
(`CONCEPT:ML-003`), MLflow + KG tracking (`CONCEPT:ML-004`), and an agent workflow that
runs the whole loop (`CONCEPT:ML-007`). See **[Installation](installation.md)** for the
capability→dependency matrix and **[Model Training](training.md)** for the recipes.

## Concept Registry

This project implements or inherits the following ecosystem concepts (full
`CONCEPT:DSCI-*` + `CONCEPT:ML-*` registry in **[Concepts](concepts.md)**):

| Concept ID | Description | Source |
|:-----------|:------------|:-------|
| ML-001 … ML-007 | **LLM trainer** — hardening, curation, pretrain, tracking, scale-out, eval, agent workflow | this project (cross-repo) |
| DSCI-004 | Model Training Operations (in-house training substrate) | this project |
| AHE-3.1 | Training Substrate (reward / distillation) | `agent-utilities` (bridge) |
| AHE-3.8 | **Agent-Interpretable Model Evolver** | `agent-utilities` (inherited) |
| AHE-3.16 | **LLM-Graded Interpretability Tests** | `agent-utilities` (inherited) |
| ECO-4.1 | MCP & Universal Skills | `agent-utilities` (inherited) |
| KG-2.17 | **Model Display Optimization** | `agent-utilities` (inherited) |

> 📖 **Full Registry**: See the [agent-utilities concept index](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/overview.md) for the complete 5-Pillar concept registry.

## Architecture

This project follows the standardized agent-package pattern:

```
data-science-mcp/
├── data_science_mcp/        # Source code
│   ├── __init__.py
│   ├── agent_server.py      # Entry point (create_graph_agent_server)
│   ├── api_client.py        # REST/GraphQL API wrapper
│   └── mcp_server.py        # FastMCP tool definitions
├── tests/                   # Test suite
├── docs/                    # Documentation
├── pyproject.toml           # Package metadata
├── mcp_config.json          # MCP server configuration
├── main_agent.json          # Agent identity & system prompt
└── Dockerfile               # Container deployment
```

## MCP Configuration

### stdio Mode
```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uv",
      "args": ["run", "--with", "data-science-mcp", "data-mcp"],
      "env": {}
    }
  }
}
```

### Streamable HTTP Mode
```bash
data-mcp --transport streamable-http --port 8001
```

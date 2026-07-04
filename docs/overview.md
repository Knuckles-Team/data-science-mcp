# data-science-mcp — Concept Overview

> **Category**: Intelligence | **Ecosystem Role**: MCP Server + A2A Agent
> Built on [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) — the unified AGI Harness.

## Description

Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AU-AHE.harness.self-improvement-overview).

## Enterprise Readiness

All agents in the ecosystem inherit enterprise-grade infrastructure from `agent-utilities`:

| Feature | Status | Source |
|:--------|:-------|:-------|
| **JWT/OIDC Authentication** | ✅ Built-in | `agent-utilities[auth]` — Authlib JWKS + API key middleware |
| **OpenTelemetry Instrumentation** | ✅ Built-in | `agent-utilities[logfire]` — OTLP export, FastAPI auto-instrumentation |
| **HashiCorp Vault Integration** | ✅ Built-in | `agent-utilities[vault]` — `secret://`, `env://`, `vault://` URI schemes |
| **Audit Logging** | ✅ Built-in | Append-only compliance trail with 30+ action types (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Token Usage Analytics** | ✅ Built-in | 4-bucket tracking with budget alerting (CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox) |
| **Prompt Injection Defense** | ✅ Built-in | 25+ pattern scanner + jailbreak taxonomy (CONCEPT:AU-OS.config.secrets-authentication) |
| **Guardrail Engine** | ✅ Built-in | Input/output interception with block/redact/warn (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |
| **Action Execution Pipeline** | ✅ Built-in | Token, cost, duration, and node transition limits Dry-run / commit / rollback phases (CONCEPT:AU-ORCH.adapter.kg-graph-materialization) |
| **Resource Scheduling** | ✅ Built-in | Priority queuing + preemption limits (CONCEPT:AU-OS.state.cognitive-scheduler-preemption) |
| **Session Concurrency** | ✅ Built-in | Enqueue/reject/interrupt/rollback (CONCEPT:AU-OS.governance.reactive-multi-axis-budget) |

## LLM Trainer

Beyond engine-backed classical ML, this project is the **agent-driven LLM trainer**
for the ecosystem — it can **create, pretrain from random init, and fine-tune** models:
robust SFT/DPO/GRPO with precision/accumulation/clipping/scheduling/checkpoint-resume
and FSDP+DeepSpeed scale-out (`CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer/005`), a corpus curation engine
(`CONCEPT:DS-AHE.trainer.data-engine`), pretraining from scratch with a trained BPE tokenizer
(`CONCEPT:DS-AHE.trainer.concept-2`), MLflow + KG tracking (`CONCEPT:DS-AHE.trainer.concept-3`), and an agent workflow that
runs the whole loop (`CONCEPT:AU-AHE.trainer.concept-2`). See **[Installation](installation.md)** for the
capability→dependency matrix and **[Model Training](training.md)** for the recipes.

## Concept Registry

This project implements or inherits the following ecosystem concepts (full
`CONCEPT:DSCI-*` + `CONCEPT:ML-*` registry in **[Concepts](concepts.md)**):

| Concept ID | Description | Source |
|:-----------|:------------|:-------|
| AU-AHE.trainer.high-caliber-llm-trainer … AU-AHE.trainer.concept-2 | **LLM trainer** — hardening, curation, pretrain, tracking, scale-out, eval, agent workflow | this project (cross-repo) |
| DS-ECO.mcp.model-training | Model Training Operations (in-house training substrate) | this project |
| AHE-3.1 | Training Substrate (reward / distillation) | `agent-utilities` (bridge) |
| AU-AHE.harness.self-improvement-overview | **Agent-Interpretable Model Evolver** | `agent-utilities` (inherited) |
| AU-AHE.harness.width-diverse-best-k | **LLM-Graded Interpretability Tests** | `agent-utilities` (inherited) |
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

# data-science-mcp

Model training, evaluation, and evolution tools for agentic ML workflows — an
**MCP Server + A2A Agent** for the agent-utilities ecosystem, with compute delegated
to the Rust **epistemic-graph** engine.

!!! info "Official documentation"
    This site is the canonical reference for `data-science-mcp`, maintained alongside
    every release.

[![PyPI](https://img.shields.io/pypi/v/data-science-mcp)](https://pypi.org/project/data-science-mcp/)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
[![License](https://img.shields.io/pypi/l/data-science-mcp)](https://github.com/Knuckles-Team/data-science-mcp/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/Knuckles-Team/data-science-mcp)

## Overview

`data-science-mcp` exposes typed, deterministic MCP tools for the full model
lifecycle — load datasets, fit and cross-validate estimators, rank models on a
Pareto frontier, generate interpretability suites, and fine-tune open-weight models
with SFT / DPO / GRPO. It provides:

- **`MLEngine`** — a stateful façade over the `epistemic-graph` Rust compute engine
  (`fit`, `predict`, `evaluate`, `cross_validate`, `load_dataset`, `split_dataset`).
  All ML compute runs in the engine over its MessagePack/UDS protocol; there is no
  scikit-learn compute path.
- **Action-routed MCP tools** across five domains — model training, model evolution,
  interpretability, data management, and quantitative finance — each independently
  togglable to control LLM context.
- **An in-house training substrate** (Wave C): a deterministic SFT/DPO/GRPO corpus and
  reward engine plus torch/PEFT gradient trainers, CPU-smoke-tested on a toy model.
- **A bundled Pydantic-AI A2A agent** that wraps the tool surface for the Agent
  Control Protocol and the Agent Web UI.

## Explore the documentation

<div class="grid cards" markdown>

- :material-rocket-launch: **[Installation](installation.md)** — pip, source, extras, and the prebuilt Docker image.
- :material-server-network: **[Deployment](deployment.md)** — run the MCP server and A2A agent, Docker Compose, Caddy + Technitium.
- :material-console: **[Usage](usage.md)** — the MCP tools, the `MLEngine` Python API, and the CLI.
- :material-sitemap: **[Overview](overview.md)** — ecosystem role, enterprise readiness, and the concept registry.
- :material-school: **[Model Training](training.md)** — the SFT/DPO/GRPO corpus, reward engine, and gradient trainers.
- :material-tag-multiple: **[Concepts](concepts.md)** — the `CONCEPT:DSCI-*` registry.

</div>

## Quick start

```bash
pip install "data-science-mcp[all]"
data-science-mcp                 # stdio MCP server (default transport)
```

Run it as a network server:

```bash
data-science-mcp --transport streamable-http --host 0.0.0.0 --port 8000
```

See **[Installation](installation.md)** and **[Deployment](deployment.md)** for the
full matrix (PyPI extras, Docker image, all transports, the A2A agent, reverse proxy,
DNS).

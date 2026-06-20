# Data Science Mcp
## CLI or API | MCP | Agent

![PyPI - Version](https://img.shields.io/pypi/v/data-science-mcp)
![MCP Server](https://badge.mcpx.dev?type=server 'MCP Server')
![PyPI - Downloads](https://img.shields.io/pypi/dd/data-science-mcp)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/data-science-mcp)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/data-science-mcp)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/data-science-mcp)
![PyPI - License](https://img.shields.io/pypi/l/data-science-mcp)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/data-science-mcp)
![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/data-science-mcp)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/data-science-mcp)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/data-science-mcp)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/data-science-mcp)
![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/data-science-mcp)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/data-science-mcp)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/data-science-mcp)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/data-science-mcp)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/data-science-mcp)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/data-science-mcp)

*Version: 0.27.0*

> **Documentation** — Installation, deployment, usage across the MCP, Python API, and
> CLI interfaces, and the in-house model-training substrate are maintained in the
> [official documentation](https://knuckles-team.github.io/data-science-mcp/).

---

## Overview

**Data Science Mcp** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AHE-3.8)..

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.
- **In-House Model Training (Wave C):** A deterministic SFT/DPO/GRPO corpus + reward engine plus torch/PEFT gradient trainers (LoRA/QLoRA, TIES adapter merge, vLLM rollouts, checkpoint→reliability-suite eval hooks). Loss/optimizer kernels are CPU-smoke-tested on a toy model; real fine-tunes run on the GB10. Install with `pip install data-science-mcp[training]`. MCP tools: `build_training_dataset`, `compose_reward`, `train_sft`, `train_dpo`, `train_grpo`, `merge_adapters_ties`. See **[docs/training.md](docs/training.md)**.

---

## CLI or API

This agent wraps the Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AHE-3.8). API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated — do not edit (synced by the `mcp-readme-table` pre-commit hook)._

<!-- MCP-TOOLS-TABLE:START -->

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `build_training_dataset` | `MODEL-TRAININGTOOL` | Build an SFT/DPO/GRPO training corpus from traces (CONCEPT:AHE-3.1). |
| `compose_reward` | `MODEL-TRAININGTOOL` | Composite, conditionally-gated reward score (CONCEPT:AHE-3.1). |
| `cross_validate` | `MODEL-TRAININGTOOL` | Perform k-fold cross-validation for a model class. |
| `curate_corpus` | `DATA-ENGINETOOL` | Full curation pass: quality-filter → dedup → decontaminate → lineage. |
| `dataset_lineage` | `DATA-ENGINETOOL` | Record a ``DatasetVersion`` provenance node (CONCEPT:ML-002). |
| `decontaminate_corpus` | `DATA-ENGINETOOL` | Drop training records that leak held-out eval examples (CONCEPT:ML-002). |
| `dedup_corpus` | `DATA-ENGINETOOL` | Remove exact + near-duplicate records (CONCEPT:ML-002). |
| `describe_dataset` | `DATA-MANAGEMENTTOOL` | Get descriptive statistics for a loaded dataset. |
| `ds_specialize_kernel` | `SAI-FACTORYTOOL` | Run a SAI-factory specialization cycle on a compute kernel (CONCEPT:AHE-3.29). |
| `evaluate_model` | `MODEL-TRAININGTOOL` | Evaluate a fitted model on a dataset split. |
| `evolve_model_class` | `MODEL-EVOLUTIONTOOL` | Submit a model to the evolutionary Pareto frontier. |
| `fit_model` | `MODEL-TRAININGTOOL` | Fit a machine learning model on a dataset and return metrics. |
| `generate_interpretability_tests` | `INTERPRETABILITYTOOL` | Generate a structured suite of 6 interpretability test cases for a model. |
| `get_pareto_frontier` | `MODEL-EVOLUTIONTOOL` | Retrieve the current Pareto frontier of model classes. |
| `grade_response` | `INTERPRETABILITYTOOL` | Grade a model interpretability response against reference answer. |
| `load_dataset` | `DATA-MANAGEMENTTOOL` | Load and parse a dataset by name or CSV file path. |
| `merge_adapters_ties` | `MODEL-TRAININGTOOL` | TIES-merge multiple task vectors onto a base (MeMo; CONCEPT:AHE-3.1). |
| `predict` | `MODEL-TRAININGTOOL` | Generate predictions using a fitted model. |
| `prepare_pretrain_data` | `DATA-ENGINETOOL` | Tokenize a corpus into a flat-token HDF5 file for pretraining (CONCEPT:ML-010). |
| `pretrain_model` | `MODEL-TRAININGTOOL` | Pretrain a causal LM **from random init** (CONCEPT:ML-003). |
| `quant_derivatives` | `QUANTTOOL` | SABR stochastic-volatility surface kernels (CONCEPT:KG-2.20j). |
| `quant_forensic` | `QUANTTOOL` | Forensic-accounting report (CONCEPT:KG-2.20g). |
| `quant_market_making` | `QUANTTOOL` | Market-making / HFT quoting kernels (CONCEPT:KG-2.20f). |
| `quant_microstructure` | `QUANTTOOL` | Order-flow / toxicity / self-excitation kernels (CONCEPT:KG-2.20f). |
| `quant_signals` | `QUANTTOOL` | Signal-combination / breadth kernels (CONCEPT:KG-2.20i). |
| `quant_sizing` | `QUANTTOOL` | Position-sizing kernels (CONCEPT:KG-2.20f / KG-2.20i). |
| `quant_statespace` | `QUANTTOOL` | State-space / statistical-arbitrage kernels (CONCEPT:KG-2.20h). |
| `quant_validation` | `QUANTTOOL` | Backtest-validation / calibration kernels (CONCEPT:KG-2.20f / KG-2.20i). |
| `rank_models` | `MODEL-EVOLUTIONTOOL` | Rank all registered fitted models by their test R2 score. |
| `run_interpretability_suite` | `INTERPRETABILITYTOOL` | Run and grade the complete 6-category interpretability audit suite for a model. |
| `split_dataset` | `DATA-MANAGEMENTTOOL` | Split a loaded dataset into train, test, and validation sets. |
| `train_dpo` | `MODEL-TRAININGTOOL` | Preference-optimise on a ``dpo`` corpus (CONCEPT:AHE-3.1). |
| `train_grpo` | `MODEL-TRAININGTOOL` | GRPO on advantage-tagged groups (CONCEPT:AHE-3.1). |
| `train_ppo` | `MODEL-TRAININGTOOL` | Proximal Policy Optimization with GAE + value head (CONCEPT:ML-009). |
| `train_reward` | `MODEL-TRAININGTOOL` | Train a Bradley-Terry reward model on preference pairs (CONCEPT:ML-008). |
| `train_sft` | `MODEL-TRAININGTOOL` | Supervised fine-tune on an ``sft`` corpus (CONCEPT:AHE-3.1). |
| `train_tokenizer` | `MODEL-TRAININGTOOL` | Train a byte-level BPE tokenizer from scratch (CONCEPT:ML-003). |

_37 action-routed tools (default `MCP_TOOL_MODE=condensed`). Each is enabled unless its toggle is set false; set `MCP_TOOL_MODE=verbose` (or `both`) for the 1:1 per-operation surface. Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/mcp.md](docs/mcp.md).

### Dynamic Tool Selection & Visibility

This MCP server supports dynamic toolset selection and visibility filtering at runtime. This allows you to restrict the set of exposed tools in order to prevent blowing up the LLM's context window.

You can configure tool filtering via multiple input channels:

- **CLI Arguments:** Pass `--tools` or `--toolsets` (or their disabled counterparts `--disabled-tools` and `--disabled-toolsets`) during startup.
- **Environment Variables:** Define standard environment variables:
  - `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS`
  - `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS`
- **HTTP SSE Request Headers:** Pass custom headers during transport initialization:
  - `x-mcp-enabled-tools` / `x-mcp-disabled-tools`
  - `x-mcp-enabled-tags` / `x-mcp-disabled-tags`
- **HTTP SSE Request Query Parameters:** Append query parameters directly to your transport connection URL:
  - `?tools=tool1,tool2`
  - `?tags=tag1`

When query strings or parameters are supplied, an LLM-free **Knowledge Graph resolution layer** (using `DynamicToolOrchestrator`) matches query intents against known tool tags, names, or descriptions, with safe fallback and automated 24-hour background cache refreshing.

---

### MCP Configuration Examples

#### stdio Transport (Recommended for local IDEs e.g., Cursor, Claude Desktop)
Configure your IDE's `mcp.json` to launch the MCP server via `uvx`:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "data-science-mcp",
        "data-science-mcp"
      ],
      "env": {
        "DATA_SCIENCE_MCP_URL": "your_data_science_mcp_url_here",
        "DATA_SCIENCE_MCP_TOKEN": "your_data_science_mcp_token_here"
      }
    }
  }
}
```

#### Streamable-HTTP Transport (Recommended for production deployments)
Configure your client's `mcp.json` to launch the Streamable-HTTP server via `uvx` with explicit host and port definition:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "data-science-mcp",
        "data-science-mcp"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "DATA_SCIENCE_MCP_URL": "your_data_science_mcp_url_here",
        "DATA_SCIENCE_MCP_TOKEN": "your_data_science_mcp_token_here"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed remote or local Streamable-HTTP instance:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "url": "http://localhost:8000/data-science-mcp/mcp"
    }
  }
}
```

Deploying the Streamable-HTTP server via Docker:

```bash
docker run -d \
  --name data-science-mcp-mcp \
  -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e DATA_SCIENCE_MCP_URL="your_value" \
  -e DATA_SCIENCE_MCP_TOKEN="your_value" \
  knucklessg1/data-science-mcp:latest
```

---

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`data-science-mcp` can also run as a **local container** (Docker / Podman / `uv`) or be
consumed from a **remote deployment**. The
[Deployment guide](https://knuckles-team.github.io/data-science-mcp/deployment/) has full, copy-paste
`mcp_config.json` for all four transports — **stdio**, **streamable-http**,
**local container / uv**, and **remote URL**:

- **Local container / uv** — launch the server from `mcp_config.json` via `uvx`,
  `docker run`, or `podman run`, or point at a local streamable-http container by `url`.
- **Remote URL** — connect to a server deployed behind Caddy at
  `http://data-science-mcp.arpa/mcp` using the `"url"` key.
<!-- END GENERATED: additional-deployment-options -->

## Agent

This repository features a fully integrated Pydantic AI Graph Agent. It communicates over the **Agent Control Protocol (ACP)** and interacts seamlessly with the **Agent Web UI (AG-UI)** and Terminal interface.

### Running the Agent CLI
To start the interactive command-line agent:

```bash
# Set credentials
export DATA_SCIENCE_MCP_URL="your_value"
export DATA_SCIENCE_MCP_TOKEN="your_value"

# Run the agent server
data-science-agent --provider openai --model-id gpt-4o
```

### Docker Compose Orchestration
The following `docker/agent.compose.yml` configures the Agent, Web UI, and Terminal Interface together:

```yaml
version: '3.8'

services:
  data-science-mcp-mcp:
    image: knucklessg1/data-science-mcp:latest
    container_name: data-science-mcp-mcp
    hostname: data-science-mcp-mcp
    restart: always
    env_file:
      - ../.env
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports:
      - "8000:8000"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  data-science-mcp-agent:
    image: knucklessg1/data-science-mcp:latest
    container_name: data-science-mcp-agent
    hostname: data-science-mcp-agent
    restart: always
    depends_on:
      - data-science-mcp-mcp
    env_file:
      - ../.env
    command: [ "data-science-agent" ]
    environment:
      - PYTHONUNBUFFERED=1
      - HOST=0.0.0.0
      - PORT=9004
      - MCP_URL=http://data-science-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
      - ENABLE_OTEL=True
    ports:
      - "9004:9004"
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9004/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

```

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/agent.md](docs/agent.md).

---

## Security & Governance

Built directly upon the enterprise-ready [`agent-utilities`](https://github.com/Knuckles-Team/agent-utilities) core, standard security parameters are fully supported:

### Access Control & Policy Enforcement
- **Eunomia Policies:** Fine-grained, policy-driven tool authorization. Supports `none`, local `embedded` (`mcp_policies.json`), or centralized `remote` modes.
- **OIDC Token Delegation:** Compliant with RFC 8693 token exchange for flowing authenticating user credentials from Web UI / ACP → Agent → MCP.
- **Scoped Credentials:** Execution context runs restricted to the specific caller identity.

### Runtime Security Grid
| Feature | Functionality | Enablement |
|---------|---------------|------------|
| **Tool Guard** | Sensitivity inspection with human-in-the-loop validation | Enabled by default |
| **Prompt Injection Defense** | Input scanning, repetition monitoring, and recursive loop blocks | Enabled by default |
| **Context Safety Guard** | Stuck-loop detectors and contextual overflow preemptive alerts | Enabled by default |

---

## Installation

Install the Python package locally:

```bash
# Using uv (highly recommended)
uv pip install data-science-mcp[all]

# Using standard pip
python -m pip install data-science-mcp[all]
```

---

## Documentation

The complete documentation is published as the
[official documentation site](https://knuckles-team.github.io/data-science-mcp/) and
is the recommended reference for installation, deployment, and day-to-day operation.

| Page | Contents |
|---|---|
| [Installation](https://knuckles-team.github.io/data-science-mcp/installation/) | pip, source, extras, prebuilt Docker image |
| [Deployment](https://knuckles-team.github.io/data-science-mcp/deployment/) | run the MCP server and A2A agent, Compose, Caddy + Technitium, env config |
| [Usage](https://knuckles-team.github.io/data-science-mcp/usage/) | the MCP tools, the `MLEngine` Python API, the console scripts |
| [Overview](https://knuckles-team.github.io/data-science-mcp/overview/) | ecosystem role, enterprise readiness, concept registry |
| [Model Training](https://knuckles-team.github.io/data-science-mcp/training/) | SFT/DPO/GRPO corpus, reward engine, gradient trainers |
| [Concepts](https://knuckles-team.github.io/data-science-mcp/concepts/) | concept registry (`CONCEPT:DSCI-*`) |

---

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-os-genesis-deploy (generated; do not edit between markers) -->

## Deploy with `agent-os-genesis`

This package can be provisioned for you — skill-guided — by the **`agent-os-genesis`**
universal skill (its *single-package deploy mode*): it picks your install method, seeds
secrets to OpenBao/Vault (or `.env`), trusts your enterprise CA, registers the MCP
server, and verifies it — the same machinery that stands up the whole Agent OS, narrowed
to just this package. Ask your agent to **"deploy `data-science-mcp` with agent-os-genesis"**.

| Install mode | Command |
|------|---------|
| Bare-metal, prod (PyPI) | `uvx data-science-mcp` · or `uv tool install data-science-mcp` |
| Bare-metal, dev (editable) | `uv pip install -e ".[all]"` · or `pip install -e ".[all]"` |
| Container, prod | deploy `knucklessg1/data-science-mcp:latest` via docker-compose / swarm / podman / podman-compose / kubernetes |
| Container, dev (editable) | deploy `docker/compose.dev.yml` (source-mounted at `/src`; edits live on restart) |

Secrets are read-existing + seeded via `vault_sync` — you are only prompted for what's missing.

<!-- END agent-os-genesis-deploy -->

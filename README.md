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

*Version: 1.2.0*

> **Documentation** — Installation, deployment, usage across the MCP, Python API, and
> CLI interfaces, and the in-house model-training substrate are maintained in the
> [official documentation](https://knuckles-team.github.io/data-science-mcp/).

---

## Overview

**Data Science Mcp** is a production-grade Agent and Model Context Protocol (MCP) server designed to interface directly with Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AU-AHE.harness.self-improvement-overview)..

---

## Key Features

- **Consolidated Action-Routed MCP Tools:** Minimizes token overhead and eliminates tool bloat in LLM contexts by grouping methods into optimized, togglable tool modules.
- **Enterprise-Grade Security:** Comprehensive support for Eunomia policies, OIDC token delegation, and granular execution context tracking.
- **Integrated Graph Agent:** Built-in Pydantic AI agent supporting the Agent Control Protocol (ACP) and standard Web interfaces (AG-UI).
- **Native Telemetry & Tracing:** Out-of-the-box OpenTelemetry exports and native Langfuse tracing.
- **In-House Model Training (Wave C):** A deterministic SFT/DPO/GRPO corpus + reward engine plus torch/PEFT gradient trainers (LoRA/QLoRA, TIES adapter merge, vLLM rollouts, checkpoint→reliability-suite eval hooks). Loss/optimizer kernels are CPU-smoke-tested on a toy model; real fine-tunes run on the GB10. Install with `pip install data-science-mcp[training]`. MCP tools: `build_training_dataset`, `compose_reward`, `train_sft`, `train_dpo`, `train_grpo`, `merge_adapters_ties`. See **[docs/training.md](docs/training.md)**.

---

## CLI or API

This agent wraps the Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AU-AHE.harness.self-improvement-overview). API. You can interact with it programmatically or via its integrated execution entrypoints.

Detailed instructions on how to use the underlying API wrappers, extended schema bindings, and developer SDK references are maintained in [docs/index.md](docs/index.md).

---

## MCP

This server utilizes dynamic Action-Routed tools to optimize token overhead and maximize IDE compatibility.

### Available MCP Tools

_Auto-generated — do not edit (synced by the `mcp-readme-table` pre-commit hook)._

<!-- MCP-TOOLS-TABLE:START -->

#### Condensed action-routed tools (default — `MCP_TOOL_MODE=condensed`)

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `build_training_dataset` | `MODEL_TRAININGTOOL` | Build an SFT/DPO/GRPO training corpus from traces (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `compose_reward` | `MODEL_TRAININGTOOL` | Composite, conditionally-gated reward score (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `cross_validate` | `MODEL_TRAININGTOOL` | Perform k-fold cross-validation for a model class. |
| `curate_corpus` | `DATA_ENGINETOOL` | Full curation pass: quality-filter → dedup → decontaminate → lineage. |
| `dataset_lineage` | `DATA_ENGINETOOL` | Record a ``DatasetVersion`` provenance node (CONCEPT:DS-AHE.trainer.data-engine). |
| `decontaminate_corpus` | `DATA_ENGINETOOL` | Drop training records that leak held-out eval examples (CONCEPT:DS-AHE.trainer.data-engine). |
| `dedup_corpus` | `DATA_ENGINETOOL` | Remove exact + near-duplicate records (CONCEPT:DS-AHE.trainer.data-engine). |
| `deep_train_predict` | `MODEL_TRAININGTOOL` | Fit + run one delegated deep/heavy model and return predictions/model as JSON. |
| `describe_dataset` | `DATA_MANAGEMENTTOOL` | Get descriptive statistics for a loaded dataset. |
| `ds_specialize_kernel` | `MODEL_TRAININGTOOL` | Run a SAI-factory specialization cycle on a compute kernel (CONCEPT:AU-AHE.harness.sai-controller). |
| `evaluate_model` | `MODEL_TRAININGTOOL` | Evaluate a fitted model on a dataset split. |
| `evolve_model_class` | `MODEL_EVOLUTIONTOOL` | Submit a model to the evolutionary Pareto frontier. |
| `fit_model` | `MODEL_TRAININGTOOL` | Fit a machine learning model on a dataset and return metrics. |
| `generate_interpretability_tests` | `INTERPRETABILITYTOOL` | Generate a structured suite of 6 interpretability test cases for a model. |
| `get_pareto_frontier` | `MODEL_EVOLUTIONTOOL` | Retrieve the current Pareto frontier of model classes. |
| `grade_response` | `INTERPRETABILITYTOOL` | Grade a model interpretability response against reference answer. |
| `load_dataset` | `DATA_MANAGEMENTTOOL` | Load and parse a dataset by name or CSV file path. |
| `merge_adapters_ties` | `MODEL_TRAININGTOOL` | TIES-merge multiple task vectors onto a base (MeMo; CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `predict` | `MODEL_TRAININGTOOL` | Generate predictions using a fitted model. |
| `prepare_pretrain_data` | `DATA_ENGINETOOL` | Tokenize a corpus into a flat-token HDF5 file for pretraining (CONCEPT:DS-AHE.trainer.data-transformation). |
| `pretrain_model` | `MODEL_TRAININGTOOL` | Pretrain a causal LM **from random init** (CONCEPT:DS-AHE.trainer.concept-2). |
| `quant_derivatives` | `QUANTTOOL` | SABR stochastic-volatility surface kernels (CONCEPT:AU-KG.domains.derivatives). |
| `quant_forensic` | `QUANTTOOL` | Forensic-accounting report (CONCEPT:EG-KG.domains.forensic-accounting-kernels). |
| `quant_market_making` | `QUANTTOOL` | Market-making / HFT quoting kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest). |
| `quant_microstructure` | `QUANTTOOL` | Order-flow / toxicity / self-excitation kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest). |
| `quant_signals` | `QUANTTOOL` | Signal-combination / breadth kernels (CONCEPT:EG-KG.domains.quant-finance). |
| `quant_sizing` | `QUANTTOOL` | Position-sizing kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i). |
| `quant_statespace` | `QUANTTOOL` | State-space / statistical-arbitrage kernels (CONCEPT:EG-KG.domains.state-space-statistical-arbitrage). |
| `quant_validation` | `QUANTTOOL` | Backtest-validation / calibration kernels (CONCEPT:EG-KG.domains.market-microstructure-sizing-backtest / KG-2.20i). |
| `rank_models` | `MODEL_EVOLUTIONTOOL` | Rank all registered fitted models by their test R2 score. |
| `run_interpretability_suite` | `INTERPRETABILITYTOOL` | Run and grade the complete 6-category interpretability audit suite for a model. |
| `split_dataset` | `DATA_MANAGEMENTTOOL` | Split a loaded dataset into train, test, and validation sets. |
| `train_dpo` | `MODEL_TRAININGTOOL` | Preference-optimise on a ``dpo`` corpus (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `train_grpo` | `MODEL_TRAININGTOOL` | GRPO on advantage-tagged groups (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `train_ppo` | `MODEL_TRAININGTOOL` | Proximal Policy Optimization with GAE + value head (CONCEPT:DS-AHE.trainer.per-token-value). |
| `train_reward` | `MODEL_TRAININGTOOL` | Train a Bradley-Terry reward model on preference pairs (CONCEPT:DS-AHE.reward.one-sequence-level-score). |
| `train_sft` | `MODEL_TRAININGTOOL` | Supervised fine-tune on an ``sft`` corpus (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort). |
| `train_tokenizer` | `MODEL_TRAININGTOOL` | Train a byte-level BPE tokenizer from scratch (CONCEPT:DS-AHE.trainer.concept-2). |

#### Verbose 1:1 API-mapped tools (`MCP_TOOL_MODE=verbose` or `both`)

<details>
<summary>9 per-operation tools — one per public API method (click to expand)</summary>

| MCP Tool | Toggle Env Var | Description |
|----------|----------------|-------------|
| `data_science_cross_validate` | `ML_ENGINETOOL` | Run k-fold cross-validation via the engine. |
| `data_science_describe_dataset` | `ML_ENGINETOOL` | Get descriptive statistics for a loaded dataset. |
| `data_science_evaluate` | `ML_ENGINETOOL` | Evaluate a fitted model. |
| `data_science_fit` | `ML_ENGINETOOL` | Fit a model on a dataset via the epistemic-graph engine. |
| `data_science_interpretability_reference` | `ML_ENGINETOOL` | Compute reference answers for the interpretability suite without any |
| `data_science_load_dataset` | `ML_ENGINETOOL` | Load a dataset by name or file path. |
| `data_science_predict` | `ML_ENGINETOOL` | Generate predictions from a fitted model. |
| `data_science_ranked_models` | `ML_ENGINETOOL` | Rank fitted models by stored test R² (backend-agnostic, no recompute). |
| `data_science_split_dataset` | `ML_ENGINETOOL` | Split a dataset into train/test/validation sizes. |

</details>

_38 action-routed tool(s) (default) · 9 verbose 1:1 tool(s). Each is enabled unless its `<DOMAIN>TOOL` toggle is set false; `MCP_TOOL_MODE` selects the surface (`condensed` default · `verbose` 1:1 · `both`). Auto-generated — do not edit._
<!-- MCP-TOOLS-TABLE:END -->

Detailed tool schemas, parameter shapes, and validation constraints are preserved in [docs/usage.md](docs/usage.md).

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

<!-- MCP-CONFIG-EXAMPLES:START -->

> **Install the connector-focused `[mcp]` extra.** Examples use `data-science-mcp[mcp]` to add
> FastMCP / FastAPI through `agent-utilities[mcp]`; the required Agent Utilities core
> still carries `epistemic-graph[full]`. The `[agent-runtime]` extra additionally
> enables model orchestration.

#### stdio Transport (local IDEs — Cursor, Claude Desktop, VS Code)

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "data-science-mcp[mcp]",
        "data-science-mcp"
      ],
      "env": {
        "MCP_TOOL_MODE": "intent",
        "DATA_ENGINETOOL": "True",
        "DATA_MANAGEMENTTOOL": "True",
        "DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME": "docker",
        "DATA_SCIENCE_KERNEL_SANDBOX_IMAGE": "registry.example/image@sha256:<digest>",
        "DEEP_DELEGATETOOL": "True",
        "DSM_NEAR_PAIRS_LOCAL_MAX": "20000",
        "INFERENCE_API_KEY": "EMPTY",
        "INFERENCE_BACKEND": "vllm",
        "INTERPRETABILITYTOOL": "True",
        "KERNEL_SPECIALIZETOOL": "True",
        "MODEL_EVOLUTIONTOOL": "True",
        "MODEL_TRAININGTOOL": "True",
        "QUANTTOOL": "True",
        "TRAINERTOOL": "True",
        "TRAINING_DATATOOL": "True"
      }
    }
  }
}
```

Runtime references require an alias-aware launcher such as GraphOS. Other
launchers must omit those entries and inject the resolved values through their
own runtime secret boundary.

#### Streamable-HTTP Transport (networked / production)

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "data-science-mcp[mcp]",
        "data-science-mcp",
        "--transport",
        "streamable-http",
        "--port",
        "8000"
      ],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "MCP_TOOL_MODE": "intent",
        "DATA_ENGINETOOL": "True",
        "DATA_MANAGEMENTTOOL": "True",
        "DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME": "docker",
        "DATA_SCIENCE_KERNEL_SANDBOX_IMAGE": "registry.example/image@sha256:<digest>",
        "DEEP_DELEGATETOOL": "True",
        "DSM_NEAR_PAIRS_LOCAL_MAX": "20000",
        "INFERENCE_API_KEY": "EMPTY",
        "INFERENCE_BACKEND": "vllm",
        "INTERPRETABILITYTOOL": "True",
        "KERNEL_SPECIALIZETOOL": "True",
        "MODEL_EVOLUTIONTOOL": "True",
        "MODEL_TRAININGTOOL": "True",
        "QUANTTOOL": "True",
        "TRAINERTOOL": "True",
        "TRAINING_DATATOOL": "True"
      }
    }
  }
}
```

Alternatively, connect to a pre-deployed Streamable-HTTP instance by `url`:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "url": "http://localhost:8000/data-science-mcp/mcp"
    }
  }
}
```

Run a reviewed container image as a least-privilege stdio child (no
listener or published port):

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  -e MCP_TOOL_MODE=intent \
  -e DATA_ENGINETOOL=True \
  -e DATA_MANAGEMENTTOOL=True \
  -e DATA_SCIENCE_KERNEL_CONTAINER_RUNTIME=docker \
  -e DATA_SCIENCE_KERNEL_SANDBOX_IMAGE="registry.example/image@sha256:<digest>" \
  -e DEEP_DELEGATETOOL=True \
  -e DSM_NEAR_PAIRS_LOCAL_MAX=20000 \
  -e INFERENCE_API_KEY=EMPTY \
  -e INFERENCE_BACKEND=vllm \
  -e INTERPRETABILITYTOOL=True \
  -e KERNEL_SPECIALIZETOOL=True \
  -e MODEL_EVOLUTIONTOOL=True \
  -e MODEL_TRAININGTOOL=True \
  -e QUANTTOOL=True \
  -e TRAINERTOOL=True \
  -e TRAINING_DATATOOL=True \
  registry.example.invalid/data-science-mcp@sha256:<digest> data-science-mcp
```

For containerized network HTTP, supply an authenticated TLS ingress (or
direct server TLS), exact `MCP_ALLOWED_HOSTS`, and an exact trusted-proxy
CIDR policy through the operator-owned deployment profile. The generator
does not emit an unauthenticated non-loopback listener.

_Auto-generated from the code-read env surface (`MCP_TOOL_MODE` + package vars) — do not edit._
<!-- MCP-CONFIG-EXAMPLES:END -->

<!-- BEGIN GENERATED: additional-deployment-options -->
### Additional Deployment Options

`data-science-mcp` can run as a local stdio process or container, or behind a remote
network boundary. The
[Deployment guide](https://knuckles-team.github.io/data-science-mcp/deployment/) carries
the detailed transport contract.

- **Local container** — launch a reviewed immutable image as a least-privilege
  stdio child with no listener or published port.
- **Remote URL** — connect through an operator-supplied authenticated HTTPS
  ingress. Keep its URL, outbound identity references, trust profile, and exact
  `MCP_ALLOWED_HOSTS` in `AgentConfig`.
<!-- END GENERATED: additional-deployment-options -->

---

## Environment Variables

<!-- ENV-VARS-TABLE:START -->

#### Package environment variables

| Variable | Example | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` |  |
| `PORT` | `8000` |  |
| `TRANSPORT` | `stdio` | options: stdio, streamable-http, sse |
| `ENABLE_OTEL` | `True` |  |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:8080/api/public/otel` |  |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` | `pk-...` |  |
| `OTEL_EXPORTER_OTLP_SECRET_KEY` | `sk-...` |  |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | `http/protobuf` |  |
| `EUNOMIA_TYPE` | `none` | options: none, embedded, remote |
| `EUNOMIA_POLICY_FILE` | `mcp_policies.json` |  |
| `EUNOMIA_REMOTE_URL` | `http://eunomia-server:8000` |  |
| `DATA_SCIENCE_MCP_URL` | Required |  |
| `DATA_SCIENCE_MCP_TOKEN` | — |  |
| `TLS_PROFILE` | — | Named `AgentConfig` transport-security profile; certificate and hostname verification remain mandatory. |
| `TLS_PROFILES_REF` | — | Runtime secret reference for the TLS profile catalog. |
| `INFERENCE_BACKEND` | `vllm` | options: vllm, sglang |
| `INFERENCE_BASE_URL` | — | Base URL of the running inference server, e.g. http://host:30000 |
| `INFERENCE_MODEL` | — | Served model id exposed by the inference server. |
| `INFERENCE_API_KEY` | `EMPTY` | Bearer token for the inference server (default "EMPTY" for local servers). |
| `EPISTEMIC_GRAPH_SOCKET` | — | Unix domain socket path to the epistemic-graph engine. |
| `EPISTEMIC_GRAPH_TCP` | — | TCP host:port for the epistemic-graph engine (takes precedence over the socket). |
| `DSM_NEAR_PAIRS_LOCAL_MAX` | `20000` | Cap on local O(n^2) near-pair fallback before requiring the Rust path (0 disables the cap). |
| `MODEL_TRAININGTOOL` | `True` |  |
| `MODEL_EVOLUTIONTOOL` | `True` |  |
| `INTERPRETABILITYTOOL` | `True` |  |
| `DATA_MANAGEMENTTOOL` | `True` |  |
| `DATA_ENGINETOOL` | `True` |  |
| `QUANTTOOL` | `True` |  |
| `TRAINERTOOL` | `True` | Sub-surfaces of model-training; the code gates these via MODEL_TRAININGTOOL. |
| `TRAINING_DATATOOL` | `True` |  |
| `KERNEL_SPECIALIZETOOL` | `True` |  |
| `DEEP_DELEGATETOOL` | `True` |  |

#### Inherited agent-utilities variables (apply to every connector)

| Variable | Example | Description |
|----------|---------|-------------|
| `MCP_TOOL_MODE` | `condensed` | Tool surface: `condensed` | `verbose` | `both` |
| `MCP_ENABLED_TOOLS` | — | Comma-separated tool allow-list |
| `MCP_DISABLED_TOOLS` | — | Comma-separated tool deny-list |
| `MCP_ENABLED_TAGS` | — | Comma-separated tag allow-list |
| `MCP_DISABLED_TAGS` | — | Comma-separated tag deny-list |
| `MCP_CLIENT_AUTH` | — | Outbound MCP auth (`oidc-client-credentials` for fleet calls) |
| `OIDC_CLIENT_ID` | — | OIDC client id (service-account auth) |
| `OIDC_CLIENT_SECRET` | — | OIDC client secret (service-account auth) |
| `DEBUG` | `False` | Verbose logging |
| `PYTHONUNBUFFERED` | `1` | Unbuffered stdout (recommended in containers) |
| `MCP_URL` | `http://localhost:8000/mcp` | URL of the MCP server the agent connects to |
| `PROVIDER` | `openai` | LLM provider for the agent |
| `MODEL_ID` | `gpt-4o` | Model id for the agent |
| `ENABLE_WEB_UI` | `True` | Serve the AG-UI web interface |

_33 package + 14 inherited variable(s). Auto-generated from `.env.example` + the shared agent-utilities set — do not edit._
<!-- ENV-VARS-TABLE:END -->


Every variable the server reads, grouped by purpose.

### MCP server / transport
| Variable | Description | Default |
|----------|-------------|---------|
| `TRANSPORT` | `stdio`, `streamable-http`, or `sse` | `stdio` |
| `HOST` | Bind host (HTTP transports) | `0.0.0.0` |
| `PORT` | Bind port (HTTP transports) | `8000` |
| `MCP_TOOL_MODE` | Tool surface: `condensed`, `verbose`, or `both` | `condensed` |
| `MCP_ENABLED_TOOLS` / `MCP_DISABLED_TOOLS` | Comma-separated tool allow/deny list | — |
| `MCP_ENABLED_TAGS` / `MCP_DISABLED_TAGS` | Comma-separated tag allow/deny list | — |
| `DEBUG` | Verbose logging | `False` |
| `PYTHONUNBUFFERED` | Unbuffered stdout (recommended in containers) | `1` |

### Connection
| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_SCIENCE_MCP_URL` | Base service URL | Required |
| `DATA_SCIENCE_MCP_TOKEN` | API token | — |

### Training / inference backend (full `[training]` extra)
| Variable | Description | Default |
|----------|-------------|---------|
| `INFERENCE_BACKEND` | Served-model rollout backend (`vllm` or `sglang`) | — |
| `INFERENCE_BASE_URL` | OpenAI-compatible inference server base URL | — |

### Tool toggles
Each action-routed tool can be disabled individually via its toggle env var (set to `false`).
The full list is in the [Available MCP Tools](#available-mcp-tools) table above.
| Variable | Tools |
|----------|-------|
| `MODEL_TRAININGTOOL` | training / fit / eval / corpus / kernel tools |
| `MODEL_EVOLUTIONTOOL` | Pareto-frontier evolution + model ranking |
| `INTERPRETABILITYTOOL` | interpretability test generation + grading suite |
| `DATA_MANAGEMENTTOOL` | dataset load / describe / split |
| `DATA_ENGINETOOL` | corpus curation / dedup / decontaminate / lineage |
| `QUANTTOOL` | quant compute kernels |

### Telemetry & governance
| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_OTEL` | Enable OpenTelemetry export | `True` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP collector endpoint | — |
| `OTEL_EXPORTER_OTLP_PUBLIC_KEY` / `OTEL_EXPORTER_OTLP_SECRET_KEY` | OTLP auth keys | — |
| `OTEL_EXPORTER_OTLP_PROTOCOL` | OTLP protocol (e.g. `http/protobuf`) | — |
| `EUNOMIA_TYPE` | Authorization mode: `none`, `embedded`, `remote` | `none` |
| `EUNOMIA_POLICY_FILE` | Embedded policy file | `mcp_policies.json` |
| `EUNOMIA_REMOTE_URL` | Remote Eunomia server URL | — |

### Agent CLI (full `[agent]` runtime only)
| Variable | Description | Default |
|----------|-------------|---------|
| `MCP_URL` | URL of the MCP server the agent connects to | `http://localhost:8000/mcp` |
| `PROVIDER` | LLM provider (e.g. `openai`) | `openai` |
| `MODEL_ID` | Model id (e.g. `gpt-4o`) | `gpt-4o` |
| `ENABLE_WEB_UI` | Serve the AG-UI web interface | `True` |

See [`.env.example`](.env.example) for a copy-paste starting point.

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
    image: example/data-science-mcp:mcp
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
    image: example/data-science-mcp@sha256:<digest>
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

Detailed graph node architecture explanations, custom skill configurations, and agentic trace guides are available in [docs/deployment.md](docs/deployment.md).

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

Pick the extra that matches what you want to run:

| Extra | Installs | Use when |
|-------|----------|----------|
| `data-science-mcp[mcp]` | Connector-focused MCP server (`agent-utilities[mcp]` — FastMCP/FastAPI + `epistemic-graph[full]`) | You only run the **MCP server** (smallest install / image) |
| `data-science-mcp[agent]` | Agent runtime (`agent-utilities[agent-runtime,logfire]` — model orchestration + `epistemic-graph[full]`) | You run the **integrated agent** |
| `data-science-mcp[all]` | Everything (`mcp` + `agent` + `scikit-learn` sample-dataset loaders) | Development / both surfaces |

Heavy ML extras are opt-in and imported lazily — add them only when needed:
`[training]` (torch/PEFT gradient trainers), `[training-scale]` (DeepSpeed/FlashAttention,
GPU-host only), `[training-fast]` (Liger Triton kernels), `[datasets]` (scikit-learn sample
loaders), `[eval]` (LightEval), `[tracking]` (MLflow). See **[docs/training.md](docs/training.md)**.

```bash
# Connector-focused MCP server (includes the shared graph engine)
uv pip install "data-science-mcp[mcp]"

# Agent runtime (adds model orchestration to the shared graph engine)
uv pip install "data-science-mcp[agent]"

# Everything (development)
uv pip install "data-science-mcp[all]"      # or: python -m pip install "data-science-mcp[all]"
```

### Container images (`:mcp` vs `:agent`)

One multi-stage `docker/Dockerfile` builds two right-sized images, selected by `--target`:

| Image tag | Build target | Contents | Entrypoint |
|-----------|--------------|----------|------------|
| `example/data-science-mcp:mcp` | `--target mcp` | `data-science-mcp[mcp]` — **connector-focused**, includes `epistemic-graph[full]`; no model-orchestration stack | `data-science-mcp` |
| `example/data-science-mcp@sha256:<digest>` | `--target agent` (default) | `data-science-mcp[agent]` — **agent runtime**, model orchestration + `epistemic-graph[full]` | `data-science-agent` |

```bash
docker build --target mcp   -t example/data-science-mcp:mcp    docker/   # connector-focused MCP server
docker build --target agent -t example/data-science-mcp:agent-local docker/   # agent runtime
```

`docker/mcp.compose.yml` runs the connector-focused `:mcp` server; `docker/agent.compose.yml` runs the
agent (`immutable agent digest`) with a co-located `:mcp` sidecar.

### Knowledge-graph database (`epistemic-graph`)

`data-science-mcp` uses the mandatory `epistemic-graph[full]` compute capabilities on top of
the `epistemic-graph[full]` runtime carried by every Agent Utilities install.
The connector-focused `[mcp]` surface uses that engine without enabling model
orchestration; `[agent]` adds the orchestration stack. Local deployments can use the
bundled engine. For production or shared state, configure a dedicated epistemic-graph
service. See the
[epistemic-graph deployment guide](https://knuckles-team.github.io/epistemic-graph/deployment/).

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

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=example&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/example)
![GitHub User's stars](https://img.shields.io/github/stars/example)

---

## Contribute

Contributions are welcome! Please ensure code quality by executing local checks before submitting pull requests:
- Format code using `ruff format .`
- Lint code using `ruff check .`
- Validate type-safety with `mypy .`
- Execute test suites using `pytest`


<!-- BEGIN agent-utilities-deployment (generated; do not edit between markers) -->

## Deploy with `agent-utilities-deployment`

Provision this package with the consolidated **`agent-utilities-deployment`**
workflow. It selects an installed-package, editable-source, or immutable-container
path; records only runtime secret and TLS-profile references in `AgentConfig`; and
runs doctor, registration, policy, observability, and rollback gates. Ask your agent
to **"deploy `data-science-mcp` with agent-utilities-deployment"**.

| Install mode | Command |
|------|---------|
| Installed package | `uv tool install "data-science-mcp[mcp]"`, then run `data-science-mcp` |
| Editable source | `uv pip install -e ".[agent]"`, then run `data-science-mcp` |
| Immutable container | deploy `registry.example.invalid/data-science-mcp@sha256:<digest>` through the operator-selected orchestrator |

The repository embeds no deployment profile, credential value, certificate path, or
environment-specific endpoint. Supply those at runtime through `AgentConfig` and the
configured secret provider.

<!-- END agent-utilities-deployment -->

<!-- GOVERNED-CAPABILITY:START -->
## Governed capability contract

This package ships a compact canonical skill surface with specialist procedures
kept as referenced workflows. The current MCP tools, skill metadata,
`connector_manifest.yml`, ontology, mappings, shapes, fixtures, migrations,
tool-schema fingerprints, and certification metadata form one versioned
capability contract. Validate them together; do not rely on stale tool names or
historical per-task skill wrappers.

Runtime endpoints, credentials, certificate trust, tenant identity, retention,
and observability policy are deployment inputs and are never packaged values.
See [Configuration, trust, and privacy](docs/configuration.md) before enabling a
network transport, connector ingestion, GraphOS delegation, or trace export.
<!-- GOVERNED-CAPABILITY:END -->

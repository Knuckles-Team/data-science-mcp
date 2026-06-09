# Installation

`data-science-mcp` is a standard Python package and a prebuilt container image. Pick
the path that matches how you want to run it.

## Requirements

- **Python 3.10+**.
- The Rust **`epistemic-graph`** compute engine (installed as a dependency via
  `epistemic-graph[datascience]`). All ML compute runs in the engine over its
  MessagePack/UDS protocol — there is no scikit-learn compute path.

## From PyPI (recommended)

```bash
pip install data-science-mcp
```

### Optional extras

The base install pulls the MCP runtime and the compute engine. Install an extra for
what you need:

| Extra | Install | Pulls in |
|---|---|---|
| `agent` | `pip install "data-science-mcp[agent]"` | Pydantic-AI A2A agent + Logfire tracing |
| `datasets` | `pip install "data-science-mcp[datasets]"` | `scikit-learn` for the built-in sample-dataset loaders (iris/diabetes/wine/…) |
| `training` | `pip install "data-science-mcp[training]"` | `torch`, `transformers`, `peft`, `bitsandbytes` for the SFT/DPO/GRPO gradient trainers (GPU-oriented) |
| `all` | `pip install "data-science-mcp[all]"` | The MCP server, the A2A agent, and the sample-dataset loaders |

```bash
# Typical: run the MCP server + the A2A agent + sample datasets
pip install "data-science-mcp[all]"
```

The `training` extra is heavy and GPU-oriented; it is imported lazily, so the package
installs and imports without it.

## From source

```bash
git clone https://github.com/Knuckles-Team/data-science-mcp.git
cd data-science-mcp
pip install -e ".[all]"          # editable install
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv pip install -e ".[all]"
uv run data-science-mcp
```

## Prebuilt Docker image

A multi-stage, slim image is published on every release (entrypoint
`data-science-mcp`):

```bash
docker pull knucklessg1/data-science-mcp:latest

docker run --rm -i \
  knucklessg1/data-science-mcp:latest        # stdio transport (default)
```

For an HTTP server with a published port, see [Deployment](deployment.md).

## Verify the install

```bash
data-science-mcp --help
python -c "import data_science_mcp; print(data_science_mcp.__version__)"
```

## Next steps

- **[Deployment](deployment.md)** — run it as a long-lived MCP server and A2A agent behind Caddy + DNS.
- **[Usage](usage.md)** — call the tools, the `MLEngine` API, and the CLI.
- **[Configuration](deployment.md#configuration-environment)** — every environment variable.

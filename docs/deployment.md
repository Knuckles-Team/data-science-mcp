# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`data-science-mcp` exposes its MCP server (console script `data-science-mcp`) four ways. Pick the row that
matches where the server runs relative to your MCP client, then copy the matching
`mcp_config.json` below. Replace the `<your-…>` placeholders with the values from the **Configuration / Environment Variables** section.

| # | Option | Transport | Where it runs | `mcp_config.json` key |
|---|--------|-----------|---------------|------------------------|
| 1 | stdio | `stdio` | client launches a subprocess | `command` |
| 2 | Streamable-HTTP (local) | `streamable-http` | a local network port | `command` or `url` |
| 3 | Local container / uv | `stdio` or `streamable-http` | Docker / Podman / uv on this host | `command` or `url` |
| 4 | Remote URL | `streamable-http` | a remote host behind Caddy | `url` |

### 1. stdio (local subprocess)

The client launches the server over stdio via `uvx` — best for local IDEs
(Cursor, Claude Desktop, VS Code):

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": ["--from", "data-science-mcp", "data-science-mcp"],
      "env": {
        "DATA_SCIENCE_MCP_URL": "<your-data_science_mcp_url>",
        "DATA_SCIENCE_MCP_TOKEN": "<your-data_science_mcp_token>"
      }
    }
  }
}
```

### 2. Streamable-HTTP (local process)

Run the server as a long-lived HTTP process:

```bash
uvx --from data-science-mcp data-science-mcp --transport streamable-http --host 0.0.0.0 --port 8000
curl -s http://localhost:8000/health        # {"status":"OK"}
```

Then either let the client launch it:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": ["--from", "data-science-mcp", "data-science-mcp", "--transport", "streamable-http", "--port", "8000"],
      "env": {
        "TRANSPORT": "streamable-http",
        "HOST": "0.0.0.0",
        "PORT": "8000",
        "DATA_SCIENCE_MCP_URL": "<your-data_science_mcp_url>",
        "DATA_SCIENCE_MCP_TOKEN": "<your-data_science_mcp_token>"
      }
    }
  }
}
```

…or connect to the already-running process by URL:

```json
{
  "mcpServers": {
    "data-science-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

### 3. Local container / uv

**(a) Launch a container directly from `mcp_config.json`** (stdio over the container —
no ports to manage). Swap `docker` for `podman` for a daemonless runtime:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "TRANSPORT=stdio",
        "-e", "DATA_SCIENCE_MCP_URL=<your-data_science_mcp_url>",
        "-e", "DATA_SCIENCE_MCP_TOKEN=<your-data_science_mcp_token>",
        "knucklessg1/data-science-mcp:latest"
      ]
    }
  }
}
```

**(b) Run a local streamable-http container, then connect by URL:**

```bash
docker run -d --name data-science-mcp -p 8000:8000 \
  -e TRANSPORT=streamable-http \
  -e PORT=8000 \
  -e DATA_SCIENCE_MCP_URL="<your-data_science_mcp_url>" \
  -e DATA_SCIENCE_MCP_TOKEN="<your-data_science_mcp_token>" \
  knucklessg1/data-science-mcp:latest
# or, from a clone of this repo:
docker compose -f docker/mcp.compose.yml up -d
```

```json
{
  "mcpServers": {
    "data-science-mcp": { "url": "http://localhost:8000/mcp" }
  }
}
```

**(c) From a local checkout with `uv`:**

```bash
uv run data-science-mcp --transport streamable-http --port 8000
```

### 4. Remote URL (deployed behind Caddy)

When the server is deployed remotely (e.g. as a Docker service) and published through
Caddy on the internal `*.arpa` zone, connect with the `"url"` key — no local process or
image required:

```json
{
  "mcpServers": {
    "data-science-mcp": { "url": "http://data-science-mcp.arpa/mcp" }
  }
}
```

Caddy reverse-proxies `http://data-science-mcp.arpa` to the container's `:8000`
streamable-http listener; `http://data-science-mcp.arpa/health` returns
`{"status":"OK"}` when the service is live.
<!-- END GENERATED: deployment-options -->

This page covers running `data-science-mcp` as a long-lived server: the transports, a
Docker Compose stack, the bundled A2A agent, putting it behind a Caddy reverse proxy,
and giving it a DNS name with Technitium.

> `data-science-mcp` ships both an **MCP server** (console script `data-science-mcp`)
> and an **A2A agent server** (console script `data-science-agent`). The MCP server is
> a typed, deterministic tool surface; the agent wraps it for the Agent Control
> Protocol and the Agent Web UI.

## Run the MCP server

The transport is selected with `--transport` (or the `TRANSPORT` env var):

=== "stdio (default)"

    ```bash
    data-science-mcp
    ```
    For IDE / desktop MCP clients that launch the server as a subprocess.

=== "streamable-http"

    ```bash
    data-science-mcp --transport streamable-http --host 0.0.0.0 --port 8000
    ```
    A network server with a `/health` endpoint and `/mcp` route.

=== "sse"

    ```bash
    data-science-mcp --transport sse --host 0.0.0.0 --port 8000
    ```

Health check (HTTP transports):

```bash
curl -s http://localhost:8000/health        # {"status":"OK"}
```

## Configuration (environment)

`data-science-mcp` is configured entirely from the environment. The **required**
runtime settings:

| Var | Default | Meaning |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for HTTP transports |
| `PORT` | `8000` | Listen port for HTTP transports |
| `TRANSPORT` | `stdio` | `stdio`, `streamable-http`, or `sse` |
| `MODEL_TRAININGTOOL` | `True` | Register the model-training tool domain |
| `MODEL_EVOLUTIONTOOL` | `True` | Register the model-evolution tool domain |
| `INTERPRETABILITYTOOL` | `True` | Register the interpretability tool domain |
| `DATA_MANAGEMENTTOOL` | `True` | Register the data-management tool domain |
| `QUANTTOOL` | `True` | Register the quantitative-finance tool domain |
| `EPISTEMIC_GRAPH_SOCKET` | — | UDS path to the `epistemic-graph` compute engine |
| `EPISTEMIC_GRAPH_TCP` | — | TCP endpoint to the compute engine (alternative to the socket) |

Telemetry (`ENABLE_OTEL`, `OTEL_EXPORTER_OTLP_*`) and access governance
(`EUNOMIA_TYPE`, `EUNOMIA_POLICY_FILE`, `EUNOMIA_REMOTE_URL`) are optional. The full
set, with defaults, is documented in
[`.env.example`](https://github.com/Knuckles-Team/data-science-mcp/blob/main/.env.example).
Copy it to `.env` and populate only what you use.

## Docker Compose

The repo ships [`docker/mcp.compose.yml`](https://github.com/Knuckles-Team/data-science-mcp/blob/main/docker/mcp.compose.yml).
It reads a sibling `.env` and publishes the HTTP server on `:8000`:

```yaml
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
```

```bash
cp .env.example .env          # then edit as needed
docker compose -f docker/mcp.compose.yml up -d
docker compose -f docker/mcp.compose.yml logs -f
```

## Run the A2A agent

`data-science-mcp` ships a bundled Pydantic-AI agent (console script
`data-science-agent`) that connects to the MCP server and exposes the Agent Control
Protocol plus the Agent Web UI. Install the agent extra and run it:

```bash
pip install "data-science-mcp[agent]"

data-science-agent \
  --provider openai --model-id gpt-4o \
  --host 0.0.0.0 --port 9004 \
  --mcp-url http://localhost:8000/mcp
```

The agent reads `MCP_URL` (the MCP server's `/mcp` route) to discover the tool
surface, and `PROVIDER` / `MODEL_ID` for the backing LLM. The repo ships
[`docker/agent.compose.yml`](https://github.com/Knuckles-Team/data-science-mcp/blob/main/docker/agent.compose.yml),
which runs the MCP server and the agent together — the agent depends on the MCP
service and reaches it by container name on `:9004`:

```yaml
services:
  data-science-mcp-mcp:
    image: knucklessg1/data-science-mcp:latest
    hostname: data-science-mcp-mcp
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports: ["8000:8000"]

  data-science-mcp-agent:
    image: knucklessg1/data-science-mcp:latest
    depends_on: [data-science-mcp-mcp]
    command: ["data-science-agent"]
    environment:
      - HOST=0.0.0.0
      - PORT=9004
      - MCP_URL=http://data-science-mcp-mcp:8000/mcp
      - PROVIDER=${PROVIDER:-openai}
      - MODEL_ID=${MODEL_ID:-gpt-4o}
      - ENABLE_WEB_UI=True
    ports: ["9004:9004"]
```

```bash
docker compose -f docker/agent.compose.yml up -d
```

## Behind a Caddy reverse proxy

Expose the HTTP server on a hostname with automatic TLS. Add to your `Caddyfile`:

```caddy
# Internal (self-signed) — homelab .arpa zone
data-science-mcp.arpa {
    tls internal
    reverse_proxy data-science-mcp-mcp:8000
}
```

```caddy
# Public — automatic Let's Encrypt
data-science-mcp.example.com {
    reverse_proxy data-science-mcp-mcp:8000
}
```

Reload Caddy:

```bash
docker compose -f services/caddy/compose.yml exec caddy caddy reload --config /etc/caddy/Caddyfile
```

## DNS with Technitium

Point the hostname at the host running Caddy. Via the Technitium API:

```bash
curl -s "http://technitium.arpa:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=data-science-mcp.arpa" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=10.0.0.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `data-science-mcp.arpa → <caddy-host-ip>` in the Technitium
web console (`http://technitium.arpa:5380`). The ecosystem
[`technitium-dns-mcp`](https://knuckles-team.github.io/technitium-dns-mcp/) automates
this as a tool.

## Register with an MCP client

Add to your client's `mcp_config.json`:

```json
{
  "mcpServers": {
    "data-science-mcp": {
      "command": "uvx",
      "args": ["--from", "data-science-mcp", "data-science-mcp"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

For a remote HTTP server, point the client at `http://data-science-mcp.arpa/mcp`
instead.

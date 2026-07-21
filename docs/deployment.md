# Deployment

<!-- BEGIN GENERATED: deployment-options -->
## Deployment Options

`data-science-mcp` supports local stdio, a loopback-only development listener, a
least-privilege stdio container, and a remote authenticated HTTPS boundary.
Provider endpoint, credential, selector, identity, and trust material are supplied
at runtime through `AgentConfig`; none is stored in this repository.

### Installed stdio process

```json
{
  "mcpServers": {
    "data-science": {
      "command": "data-science-mcp",
      "args": [],
      "env": {"MCP_TOOL_MODE": "intent"}
    }
  }
}
```

### Loopback development listener

```bash
data-science-mcp --transport streamable-http --host 127.0.0.1 --port 8000
```

Do not expose this listener beyond loopback. Network deployments require direct TLS
or an explicitly trusted TLS-terminating ingress, configured authentication, exact
`MCP_ALLOWED_HOSTS`, and an exact trusted-proxy CIDR policy.

### Least-privilege local container

```bash
docker run -i --rm \
  --read-only \
  --cap-drop=ALL \
  --security-opt=no-new-privileges \
  --pids-limit=256 \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  -e TRANSPORT=stdio \
  registry.example.invalid/data-science-mcp@sha256:<digest> data-science-mcp
```

The operator projects the selected AgentConfig profile into the process at runtime;
the image remains immutable and contains no environment connection profile.

### Remote authenticated HTTPS endpoint

```json
{
  "mcpServers": {
    "data-science": {"url": "https://service.example.invalid/mcp"}
  }
}
```

Store the real remote URL, outbound identity reference, and TLS-profile reference in
`AgentConfig`, not in MCP client JSON or documentation.
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
    image: example/data-science-mcp@sha256:<digest>
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
    image: example/data-science-mcp@sha256:<digest>
    hostname: data-science-mcp-mcp
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=streamable-http
    ports: ["8000:8000"]

  data-science-mcp-agent:
    image: example/data-science-mcp@sha256:<digest>
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
# Internal (self-signed) — homelab .example.invalid zone
data-science-mcp.example.invalid {
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
curl -s "http://technitium.example.invalid:5380/api/zones/records/add" \
  --data-urlencode "token=$TECHNITIUM_DNS_TOKEN" \
  --data-urlencode "domain=data-science-mcp.example.invalid" \
  --data-urlencode "zone=arpa" \
  --data-urlencode "type=A" \
  --data-urlencode "ipAddress=192.0.2.10" \
  --data-urlencode "ttl=3600"
```

…or add an **A record** `data-science-mcp.example.invalid → <caddy-host-ip>` in the Technitium
web console (`http://technitium.example.invalid:5380`). The ecosystem
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

For a remote HTTP server, point the client at `http://data-science-mcp.example.invalid/mcp`
instead.

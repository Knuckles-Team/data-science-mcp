# Data Science MCP - A2A | AG-UI | MCP

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

*Version: 0.2.0*

## Overview

**Data Science MCP MCP Server + A2A Agent**

Data Science MCP Server — Model training, evaluation, and evolution tools for agentic ML workflows. Integrates with agent-utilities IModelEvolver (CONCEPT:AHE-3.15).

This repository is actively maintained - Contributions are welcome!

## MCP

### Using as an MCP Server

The MCP Server can be run in two modes: `stdio` (for local testing) or `http` (for networked access).

#### Environment Variables

*   `DATA_SCIENCE_MCP_URL`: The URL of the target service.
*   `DATA_SCIENCE_MCP_TOKEN`: The API token or access token.

#### Run in stdio mode (default):
```bash
export DATA_SCIENCE_MCP_URL="http://localhost:8080"
export DATA_SCIENCE_MCP_TOKEN="your_token"
data-science-mcp --transport "stdio"
```

#### Run in HTTP mode:
```bash
export DATA_SCIENCE_MCP_URL="http://localhost:8080"
export DATA_SCIENCE_MCP_TOKEN="your_token"
data-science-mcp --transport "http" --host "0.0.0.0" --port "8000"
```

## A2A Agent

### Run A2A Server
```bash
export DATA_SCIENCE_MCP_URL="http://localhost:8080"
export DATA_SCIENCE_MCP_TOKEN="your_token"
data-science-agent --provider openai --model-id gpt-4o --api-key sk-...
```

## Docker

### Build

```bash
docker build -t data-science-mcp .
```

### Run MCP Server

```bash
docker run -d \
  --name data-science-mcp \
  -p 8000:8000 \
  -e TRANSPORT=http \
  -e DATA_SCIENCE_MCP_URL="http://your-service:8080" \
  -e DATA_SCIENCE_MCP_TOKEN="your_token" \
  knucklessg1/data-science-mcp:latest
```

### Deploy with Docker Compose

```yaml
services:
  data-science-mcp:
    image: knucklessg1/data-science-mcp:latest
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - TRANSPORT=http
      - DATA_SCIENCE_MCP_URL=http://your-service:8080
      - DATA_SCIENCE_MCP_TOKEN=your_token
    ports:
      - 8000:8000
```

#### Configure `mcp.json` for AI Integration (e.g. Claude Desktop)

```json
{
  "mcpServers": {
    "data-science": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "data-science-mcp",
        "data-science-mcp"
      ],
      "env": {
        "DATA_SCIENCE_MCP_URL": "http://your-service:8080",
        "DATA_SCIENCE_MCP_TOKEN": "your_token"
      }
    }
  }
}
```

## Install Python Package

```bash
python -m pip install data-science-mcp
```
```bash
uv pip install data-science-mcp
```

## Repository Owners

<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)

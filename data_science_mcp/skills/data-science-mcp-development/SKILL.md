---
name: data-science-mcp-development
skill_type: skill
description: >-
  Develop and validate data-science-mcp within the shared connector and spec-delivery
  contracts.
license: MIT
tags: [data-science, development, mcp, spec-delivery]
metadata:
  author: Knuckles-Team
  version: '0.1.0'
---

# Data Science MCP Development

Use this skill when changing the `data-science-mcp` connector, its documentation, or
its spec-delivery integration. Work from the repository root and read `AGENTS.md`, the
affected module, and the relevant `specs/` status before editing. Keep the connector
focused on MCP and A2A exposure; model computation belongs in the medulla
engine through the existing `MLEngine` and client protocol.

## Steps

1. Find the owning requirement with `grep -rn CONN-DATA-SCIENCE-MCP specs`. Verify the change maps to an existing row before editing.
2. Make the smallest change in the existing connector module, routing ML compute through the engine. Verify with `git diff --stat` that only the intended files changed.
3. Run the narrowest relevant test, for example `uv run --locked pytest -q tests/test_kg_ingest.py`. Verify the run reports no failures.
4. Run the repository hooks with `pre-commit run --all-files`. Verify every hook reports Passed and none was skipped.

## Failure recovery

| Failure | Cause | Recovery |
| --- | --- | --- |
| A hook fails | The change violates formatting, spelling or a repository gate | Read the failing hook output, fix the named file and line, and rerun `pre-commit run --all-files` without skip flags. |
| The focused test fails | The change broke connector behavior | Run `uv run --locked pytest -q <failing test file> -x` to see the first failure and fix that code path. |
| The dashboard inputs look wrong | `.config/spec-dashboard.json` or a workflow file has a syntax error | Validate the file with `python3 -c "import json,sys; json.load(open(sys.argv[1]))" .config/spec-dashboard.json` and fix the reported line. |

## Rapid delivery: the 5-minute contract

For a small, reviewable change:

1. Identify one requirement or maintenance slice and name the exact files it touches.
2. Reuse the existing connector, SDK, central pipelines, and package conventions. Do
   not add a local planner, CI framework, publisher, or parallel transport.
3. Keep spec status generated from the canonical pipeline tools. Run the focused tests
   for the changed behavior, then run `pre-commit run --all-files` from the repo root.
4. Review the complete diff, including workflow and package-data changes, and record
   the source revision plus the commands and results used as evidence.
5. Keep credentials, generated output, caches, and scratch files outside the checkout.
   A Pages claim is complete only after the central reusable workflow has run and the
   exact published `spec-delivery/` source can be checked.

### Change boundaries

- Route ML compute through the existing engine instead of introducing a Python compute
  fallback.
- Extend an existing spec or connector operation before adding a new abstraction.
- Use the central Pages workflow and `.config/spec-dashboard.json` for the spec
  dashboard; keep one documentation publisher for this repository.
- Do not edit runtime or ontology assets for a documentation or delivery workflow
  change unless the requirement explicitly calls for them.

### Validation evidence

For code changes, run the narrowest relevant test first and then the repository hooks.
For delivery changes, validate JSON and YAML syntax and inspect the rendered dashboard
inputs before requesting review. Report failures with the command, revision, and file
or line that needs attention; do not hide a failing check behind a skipped hook.

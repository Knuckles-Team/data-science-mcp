---
name: data-science-mcp-operations
skill_type: skill
description: >-
  Operate data-science-mcp through its governed MCP and Arbora capabilities, including data science llm finetuning, data science model reliability, data science model training. Use when a request requires this provider's read, change, automation, ingestion, troubleshooting, or evidence workflows.
---

# Data Science Mcp Operations

Use the provider's governed MCP tools through Arbora delegation.

## Workflow

1. Establish the verified GraphSession and tenant before discovery or retrieval.
2. Discover the current condensed tool surface; never assume a stale tool name or schema.
3. Prefer read-only inspection first. For changes, present impact and use the provider's
   dry-run or preview mode when available.
4. Execute mutations as fenced WorkItems so retries remain idempotent and auditable.
5. Ingest source data only through the signed connector preset and ChangeEnvelope path.
6. Verify the durable result and its trace/evidence before reporting completion.

## Steps

1. Confirm the verified GraphSession and tenant before any call, by calling `ask` for the session context. Verify the session reports the intended tenant.
2. Discover the current tool surface with `find` for the capability you need, never relying on remembered names. Verify the tool and its parameter schema appear in the result.
3. Inspect read-only first with `ask`, and for a change request the provider's dry-run or preview. Verify the preview states the impact before any execution.
4. Execute an approved change as a fenced WorkItem with `act` and the returned plan reference. Verify the result carries a trace or evidence reference.

## Failure recovery

| Failure | Cause | Recovery |
| --- | --- | --- |
| Discovery returns no matching tool | The capability is not exposed to this tenant or the name was guessed | Call `find` with a natural-language intent for the capability and use only the operation id it returns. |
| Preview is unavailable for a mutation | The provider offers no dry-run mode | Stop and ask the operator for explicit approval, stating the impact, before calling `act`. |
| Result lacks trace or evidence | The WorkItem did not complete or was not durable | Call `ask` for the WorkItem status and report the failure instead of claiming completion. |

## Safety contract

- Never persist credentials, endpoints, raw personal identifiers, hostnames, or local paths.
- Resolve TLS trust and verification from environment/configuration; never hardcode bypasses.
- Treat unknown ACL, tenant, schema, or tool-contract state as a hard failure.
- Require explicit approval for destructive, externally visible, or irreversible actions.
- Keep runtime traces policy-scoped and privacy-sanitized.

## Specialized workflows

Read [the workflow catalog](references/catalog.md) only when the request needs a
provider-specific procedure, parameter map, script, or reference asset.

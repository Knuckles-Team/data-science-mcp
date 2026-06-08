# Concept Registry — data-science-mcp

> **Prefix**: `CONCEPT:DSCI-*`
> **Version**: 0.8.0
> **Bridge**: [`CONCEPT:ECO-4.0`](../../agent-utilities/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:DSCI-001` | Data Management Operations | MCP tool domain `data_management` — Action-routed dynamic tool registration |
| `CONCEPT:DSCI-002` | Interpretability Operations | MCP tool domain `interpretability` — Action-routed dynamic tool registration |
| `CONCEPT:DSCI-003` | Model Evolution Operations | MCP tool domain `model_evolution` — Action-routed dynamic tool registration |
| `CONCEPT:DSCI-004` | Model Training Operations | MCP tool domain `model_training` — Action-routed dynamic tool registration; incl. the in-house training substrate (`training_data` corpus/reward engine + `trainers/` SFT/DPO/GRPO + `peft_manager`/`tokenizer_registry`/`rollout_buffer`, `CONCEPT:AHE-3.1`) |
| `CONCEPT:DSCI-005` | State-Space / Stat-Arb Operations | MCP tool domain `quant_statespace` — Kalman filter/beta/volatility, ADF, OU calibration + thresholds, Markov transition (engine `client.finance.*`, KG-2.20h) |
| `CONCEPT:DSCI-006` | Signal-Combination Operations | MCP tool domain `quant_signals` — order-book imbalance, information ratio, effective independent N, alpha combination, convergence gate; plus `empirical_kelly` (quant_sizing) and `brier_score` (quant_validation) (engine `client.finance.*`, KG-2.20i) |
| `CONCEPT:DSCI-007` | SABR Volatility-Surface Operations | MCP tool domain `quant_derivatives` — Hagan-2002 SABR `implied_vol` / `smile` / `calibrate` (fit α,ρ,ν with β fixed → {alpha,beta,rho,nu,rmse,converged}) delegating to engine `client.finance.sabr_*` (KG-2.20j) |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:ECO-4.0` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:ORCH-1.2` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:OS-5.1` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:OS-5.2` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:OS-5.3` | Guardrail Engine | agent-utilities |
| `CONCEPT:OS-5.4` | Audit Logging | agent-utilities |
| `CONCEPT:KG-2.0` | Knowledge Graph Core | agent-utilities |
| `CONCEPT:AHE-3.1` | Training Substrate (reward decomposition / distillation) | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:ECO-4.0` (Unified Toolkit Ingestion). The `data_science_mcp` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all DSCI-* concepts.

# Concept Registry — data-science-mcp

> **Prefixes**: `CONCEPT:DSCI-*` (this project) · `CONCEPT:ML-*` (cross-repo LLM trainer)
> **Version**: 0.9.0
> **Bridge**: [`CONCEPT:ECO-4.0`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

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

## LLM Trainer Concepts (`CONCEPT:ML-*`)

The high-caliber LLM trainer — create, **pretrain from random init**, and fine-tune
models, driven by AI agents. This is a deliberate **cross-repo family** (it spans
`data-science-mcp` + `agent-utilities` + `universal-skills`), so it uses a repo-neutral
`ML-*` prefix rather than `DSCI-*`. It expands [`CONCEPT:DSCI-004`](#project-specific-concepts)
(Model Training Operations) and bridges [`CONCEPT:AHE-3.1`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/architecture/in_house_training_substrate.md)
(the in-house training substrate). See [Model Training](training.md), [Installation](installation.md),
and the SDD spec at `.specify/specs/llm-model-trainer/`.

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:ML-001` | Trainer Hardening | Shared `trainers/loop.py::run_loop` — precision (fp16 scaler / bf16 autocast), gradient accumulation, clipping, LR scheduling, checkpoint save+resume, metrics. SFT/DPO/GRPO/pretrain all route through it; `TrainConfig` defaults reproduce prior behaviour exactly |
| `CONCEPT:ML-002` | Corpus Curation Engine | `data_engine.py` (+ `mcp/mcp_data_engine.py`) — stream / exact+near dedup / decontaminate / quality-filter / pack / `DatasetVersion` lineage; epistemic-graph HNSW/LSH `find_similar_pairs` accelerates the all-pairs search (local-cosine fallback) |
| `CONCEPT:ML-003` | Pretrain From Random Init | `tokenizer_trainer.py` (BPE) + `trainers/pretrain_trainer.py` (`PretrainSpec`, `AutoConfig`→`from_config`, packed next-token CE; kind=`pretrain`) + `run_pretrain_pipeline` |
| `CONCEPT:ML-004` | Experiment Tracking | `tracking.py::RunTracker` — MLflow / W&B / none + best-effort epistemic-graph `TrainingRun` provenance mirror |
| `CONCEPT:ML-005` | Distributed Scale-Out | `trainers/accelerate_launch.py` + `launch/` — FSDP **and** DeepSpeed ZeRO-3 as first-class peers + `accelerate launch` config/command builder (homelab or cloud) |
| `CONCEPT:ML-006` | Benchmark Evaluation | `trainers/eval_hooks.evaluate_benchmarks` — EleutherAI `lm-eval` scoring alongside the AHE-3.1 reliability suite |
| `CONCEPT:ML-007` | Agent-Driven Training | Personas `data_curator`/`training_engineer`/`eval_judge`/`ml_orchestrator` (agent-utilities) + the `ml/train_model` workflow + `model_training_team` (universal-skills) |
| `CONCEPT:ML-008` | Reward Model | `trainers/reward_trainer.py` (Bradley-Terry pairwise loss `objectives.bradley_terry_loss`) on a scalar head (`trainers/value_head.py`); consumes the preference corpus; tool `train_reward`. The RLHF stage between SFT and PPO |
| `CONCEPT:ML-009` | PPO (actor-critic) | `trainers/ppo_trainer.py` — rollout → reward (verifier or ML-008 reward model) → GAE (`objectives.gae`) + value head → clipped surrogate (`grpo_surrogate`) + value loss (`objectives.value_function_loss`) + KL-to-reference; tool `train_ppo`; `training_pipeline.run_rlhf_pipeline` chains SFT→reward→PPO |
| `CONCEPT:ML-010` | Flat-Token Pretrain Data | `data_engine.prepare_pretrain_data` / `read_token_blocks` — stream (`.jsonl`/`.jsonl.zst`/HF) → tokenize (EOS-sep) → contiguous HDF5/`.npy` token array, batched on the fly (no padding); tool `prepare_pretrain_data` |
| `CONCEPT:ML-011` | Training-Job Dispatch | `training_job_runner.py` runs a trainer as a `JobRunner` on the agent-utilities GPU-slot scheduler (KG-2.65): checkpoint on `should_pause`, auto-resume on backfill; the `train_model` workflow submits to a GPU host or falls back to `accelerate launch` |
| `CONCEPT:ML-012` | Chat + Reasoning Format | `chat_template.py` — learnable role markers + `<think>/<answer>` (ordinary tokens) + answer extractor; `eval_hooks.evaluate_gsm8k` / `gsm8k_reward` verifiable exact-match reward |

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

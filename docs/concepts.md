# Concept Registry — data-science-mcp

> **Prefixes**: `CONCEPT:DSCI-*` (this project) · `CONCEPT:ML-*` (cross-repo LLM trainer)
> **Version**: 0.9.0
> **Bridge**: [`CONCEPT:AU-ECO.messaging.native-backend-abstraction`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/concepts.md) (Unified Toolkit Ingestion)

---

## Project-Specific Concepts

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:DS-ECO.mcp.data-management` | Data Management Operations | MCP tool domain `data_management` — Action-routed dynamic tool registration |
| `CONCEPT:DS-ECO.mcp.interpretability` | Interpretability Operations | MCP tool domain `interpretability` — Action-routed dynamic tool registration |
| `CONCEPT:DS-ECO.mcp.model-evolution` | Model Evolution Operations | MCP tool domain `model_evolution` — Action-routed dynamic tool registration |
| `CONCEPT:DS-ECO.mcp.model-training` | Model Training Operations | MCP tool domain `model_training` — Action-routed dynamic tool registration; incl. the in-house training substrate (`training_data` corpus/reward engine + `trainers/` SFT/DPO/GRPO + `peft_manager`/`tokenizer_registry`/`rollout_buffer`, `CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort`) |
| `CONCEPT:DS-ECO.mcp.quant-statespace` | State-Space / Stat-Arb Operations | MCP tool domain `quant_statespace` — Kalman filter/beta/volatility, ADF, OU calibration + thresholds, Markov transition (engine `client.finance.*`, EG-KG.domains.state-space-statistical-arbitrage) |
| `CONCEPT:DS-ECO.mcp.signal-combination` | Signal-Combination Operations | MCP tool domain `quant_signals` — order-book imbalance, information ratio, effective independent N, alpha combination, convergence gate; plus `empirical_kelly` (quant_sizing) and `brier_score` (quant_validation) (engine `client.finance.*`, EG-KG.domains.quant-finance) |
| `CONCEPT:DS-ECO.mcp.sabr-volatility` | SABR Volatility-Surface Operations | MCP tool domain `quant_derivatives` — Hagan-2002 SABR `implied_vol` / `smile` / `calibrate` (fit α,ρ,ν with β fixed → {alpha,beta,rho,nu,rmse,converged}) delegating to engine `client.finance.sabr_*` (AU-KG.domains.derivatives) |

## LLM Trainer Concepts (`CONCEPT:ML-*`)

The high-caliber LLM trainer — create, **pretrain from random init**, and fine-tune
models, driven by AI agents. This is a deliberate **cross-repo family** (it spans
`data-science-mcp` + `agent-utilities` + `universal-skills`), so it uses a repo-neutral
`ML-*` prefix rather than `DSCI-*`. It expands [`CONCEPT:DS-ECO.mcp.model-training`](#project-specific-concepts)
(Model Training Operations) and bridges [`CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort`](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/architecture/in_house_training_substrate.md)
(the in-house training substrate). See [Model Training](training.md), [Installation](installation.md),
and the SDD spec at `.specify/specs/llm-model-trainer/`.

| Concept ID | Name | Description |
|------------|------|-------------|
| `CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer` | Trainer Hardening | Shared `trainers/loop.py::run_loop` — precision (fp16 scaler / bf16 autocast), gradient accumulation, clipping, LR scheduling, checkpoint save+resume, metrics. SFT/DPO/GRPO/pretrain all route through it; `TrainConfig` defaults reproduce prior behaviour exactly |
| `CONCEPT:DS-AHE.trainer.data-engine` | Corpus Curation Engine | `data_engine.py` (+ `mcp/mcp_data_engine.py`) — stream / exact+near dedup / decontaminate / quality-filter / pack / `DatasetVersion` lineage; epistemic-graph HNSW/LSH `find_similar_pairs` accelerates the all-pairs search (local-cosine fallback) |
| `CONCEPT:DS-AHE.trainer.concept-2` | Pretrain From Random Init | `tokenizer_trainer.py` (BPE) + `trainers/pretrain_trainer.py` (`PretrainSpec`, `AutoConfig`→`from_config`, packed next-token CE; kind=`pretrain`) + `run_pretrain_pipeline` |
| `CONCEPT:DS-AHE.trainer.concept-3` | Experiment Tracking | `tracking.py::RunTracker` — MLflow / W&B / none + best-effort epistemic-graph `TrainingRun` provenance mirror |
| `CONCEPT:DS-AHE.trainer.concept-4` | Distributed Scale-Out | `trainers/accelerate_launch.py` + `launch/` — FSDP **and** DeepSpeed ZeRO-3 as first-class peers + `accelerate launch` config/command builder (homelab or cloud) |
| `CONCEPT:DS-AHE.trainer.concept-5` | Benchmark Evaluation | `trainers/eval_hooks.evaluate_benchmarks` — EleutherAI `lm-eval` scoring alongside the AHE-3.1 reliability suite |
| `CONCEPT:AU-AHE.trainer.concept-2` | Agent-Driven Training | Personas `data_curator`/`training_engineer`/`eval_judge`/`ml_orchestrator` (agent-utilities) + the `ml/train_model` workflow + `model_training_team` (universal-skills) |
| `CONCEPT:DS-AHE.reward.one-sequence-level-score` | Reward Model | `trainers/reward_trainer.py` (Bradley-Terry pairwise loss `objectives.bradley_terry_loss`) on a scalar head (`trainers/value_head.py`); consumes the preference corpus; tool `train_reward`. The RLHF stage between SFT and PPO |
| `CONCEPT:DS-AHE.trainer.per-token-value` | PPO (actor-critic) | `trainers/ppo_trainer.py` — rollout → reward (verifier or DS-AHE.reward.one-sequence-level-score reward model) → GAE (`objectives.gae`) + value head → clipped surrogate (`grpo_surrogate`) + value loss (`objectives.value_function_loss`) + KL-to-reference; tool `train_ppo`; `training_pipeline.run_rlhf_pipeline` chains SFT→reward→PPO |
| `CONCEPT:DS-AHE.trainer.data-transformation` | Flat-Token Pretrain Data | `data_engine.prepare_pretrain_data` / `read_token_blocks` — stream (`.jsonl`/`.jsonl.zst`/HF) → tokenize (EOS-sep) → contiguous HDF5/`.npy` token array, batched on the fly (no padding); tool `prepare_pretrain_data` |
| `CONCEPT:AU-AHE.trainer.join-inference` | Training-Job Dispatch | `training_job_runner.py` runs a trainer as a `JobRunner` on the agent-utilities GPU-slot scheduler (KG-2.65): checkpoint on `should_pause`, auto-resume on backfill; the `train_model` workflow submits to a GPU host or falls back to `accelerate launch` |
| `CONCEPT:DS-AHE.trainer.chat-format` | Chat + Reasoning Format | `chat_template.py` — learnable role markers + `<think>/<answer>` (ordinary tokens) + answer extractor; `eval_hooks.evaluate_gsm8k` / `gsm8k_reward` verifiable exact-match reward |
| `CONCEPT:DS-AHE.optimization.cache-agent-loop` | CacheRL Cached Rollouts + Hybrid Thinking | `cache_agent_loop.py` — three-tier fuzzy tool-call cache (`ThreeTierToolCache` exact→fuzzy→semantic) wrapped by `CacheAgentLoop` so multi-turn RL rollouts serve repeated tool calls from cache (live-execution cost ↓, `calls_saved`/`hit_rate` measure it) + `ThinkingTraceAugmenter` interleaving model "why-this-tool" rationale and per-segment provenance labels; reward half (token-mask + cache-tier-aware reward) in agent-utilities AHE-3.49. Distils arXiv:2606.14179 |

## Cross-Project References (from agent-utilities)

| Concept ID | Name | Origin |
|------------|------|--------|
| `CONCEPT:AU-ECO.messaging.native-backend-abstraction` | Unified Toolkit Ingestion | agent-utilities |
| `CONCEPT:AU-ORCH.adapter.hot-cache-invalidation` | Confidence-Gated Router | agent-utilities |
| `CONCEPT:AU-OS.config.secrets-authentication` | Prompt Injection Defense | agent-utilities |
| `CONCEPT:AU-OS.state.cognitive-scheduler-preemption` | Cognitive Scheduler | agent-utilities |
| `CONCEPT:AU-OS.governance.reactive-multi-axis-budget` | Guardrail Engine | agent-utilities |
| `CONCEPT:AU-OS.governance.wasm-micro-agent-sandbox` | Audit Logging | agent-utilities |
| `CONCEPT:AU-KG.query.object-graph-mapper` | Knowledge Graph Core | agent-utilities |
| `CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort` | Training Substrate (reward decomposition / distillation) | agent-utilities |

## Synergy with agent-utilities

This project integrates with `agent-utilities` via `CONCEPT:AU-ECO.messaging.native-backend-abstraction` (Unified Toolkit Ingestion). The `data_science_mcp` MCP server registers its tools with the agent-utilities FastMCP middleware, enabling automatic discovery, telemetry, and Knowledge Graph ingestion of all DSCI-* concepts.

# Changelog

All notable changes to `data-science-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **RLHF stack + flat-token pretrain data (CONCEPT:ML-008..012)** — fills the from-scratch-through-RLHF
  gaps as native capabilities reusing the `run_loop` spine: ML-008 Bradley-Terry reward model
  (`trainers/reward_trainer.py` on a shared scalar `value_head.py`, tool `train_reward`); ML-009 PPO
  (`trainers/ppo_trainer.py` — rollout-consuming actor-critic with GAE, clipped surrogate, value loss,
  KL-to-reference, tool `train_ppo`; `run_rlhf_pipeline` chains SFT→reward→PPO with eval gates);
  ML-010 flat-token pretrain data (`prepare_pretrain_data`/`read_token_blocks` streaming incl.
  `.jsonl.zst` → contiguous HDF5/`.npy`, tool `prepare_pretrain_data`); ML-012 chat format
  (`chat_template.py` `<think>`/`<answer>` + `evaluate_gsm8k`/`gsm8k_reward` verifiable exact-match).
- **GPU-slot training dispatch + run lineage (CONCEPT:ML-011)** — `training_job_runner.py` wraps a
  trainer as a `GpuSlotScheduler` (KG-2.65) `JobRunner` that checkpoints + yields on preempt and
  resumes from checkpoint on backfill (cooperative `should_pause` hook in `run_loop`). `TrainingRun`
  provenance now carries `dataset_version`/`parent_run` + a PROV-O `was_derived_from` edge so the
  dataset→…→model lineage is queryable.
- **Machine-verifiable GPU/compute-kernel SAI specialization (CONCEPT:AHE-3.28/3.29)** —
  `kernel_tasks.py` (`KernelTask` + fused-softmax/layernorm/matmul suite, device-agnostic numpy
  references) + `kernel_verifier.py` (`KernelVerifier`: correctness-gated speedup reward, isolated
  subprocess with wall timeout, fails closed). `build_kernel_task`/`run_kernel_specialization` drive
  the `SaiFactoryController` closed loop to a faster, correct kernel measured by adaptation speed,
  exposed through the gateway as the `ds_specialize_kernel` MCP tool (plan-first when no inference
  backend is configured).
- **LoRA hot-swap serving + per-task adapter library (SAI weight-arm seam)** —
  `inference/openai_compatible.py` per-request `adapter` swap (one base-model vLLM `--enable-lora`
  server serves N specialists by name, no reload) + `adapter_library.py` (`AdapterLibrary` maps
  `task_signature` → trained LoRA specialist so many specialists coexist on one base, each routed by
  `task:`/`adapter:`/`base:` tags via `pick_for_task`). Closes the gap where adapters were trained
  but never served.
- **Shortcut-resistant search-task corpora (CONCEPT:KG-2.70/2.71/2.72, AHE-3.30)** —
  `data_science_mcp/search_task_corpus.py`: turns agent-utilities' synthesized
  shortcut-resistant search tasks + solver trajectories (FORT-Searcher,
  arXiv:2606.12087) into the in-house corpora, reusing the existing builders
  unchanged. `tasks_to_sft` (gold trajectory → SFT), `trajectories_to_preference_pairs`
  (shortcut trajectory as the DPO `rejected`), and `rollouts_to_grpo` +
  `search_reward` (rollouts rewarded by realized search difficulty — solving cost /
  late answer-hit / low prior-shortcut). Surfaced via new `build_training_dataset`
  kinds `search_sft` / `search_dpo` / `search_grpo`. Where FORT trains SFT-only, this
  also mints the DPO/GRPO corpora the same data affords.
- **Deterministic training-data & reward engine (CONCEPT:AHE-3.1)** — `data_science_mcp/training_data.py`:
  pure-Python builders (`build_sft_examples`, `build_preference_pairs` w/ failure-point anchoring,
  `build_grpo_groups` w/ group-normalized advantages, `filter_by_difficulty`, `score_reward`) reusing the
  agent-utilities reward spine (`agent_utilities.graph.training_signals`), plus a `Trainer` Protocol seam
  for the (GPU/torch, Wave-D) gradient trainers. Exposed as MCP tools `build_training_dataset` /
  `compose_reward` (`data_science_mcp/mcp/mcp_training_data.py`, tag `model-training`). The data-quality
  half of the in-house training substrate (Wave C) — runnable today with no GPU; sources b3-02/b6-01/b6-04.
- **Gradient trainers — SFT / DPO / GRPO scaffold (CONCEPT:AHE-3.1)** — `data_science_mcp/trainers/`:
  torch loss kernels (`objectives.py`: masked CE, sequence log-prob, Bradley-Terry `dpo_loss`,
  group-relative `grpo_surrogate` + token-masked LA-GRPO variant, Schulman-k3 `approx_kl`), a shared
  `TrainerBase` (pure `plan()` + DI-friendly model/tokenizer resolution) and concrete `SftTrainer` /
  `DpoTrainer` / `GrpoTrainer` implementing the `training_data.Trainer` Protocol seam. Supporting modules:
  `peft_manager.py` (`LoraSpec`/`PeftManager` lazy peft/QLoRA + pure-numpy `ties_merge` for MeMo b4-01),
  `tokenizer_registry.py` (special/functional-token injection + embedding-resize plan for ATLAS/SDAR),
  `rollout_buffer.py` (prompt→generation→logprob→reward staging + `VLLMRolloutClient`, GRPO export via the
  shared reward spine), and `trainers/eval_hooks.py` (bridges a checkpoint into the AHE-3.1 reliability
  suite). Exposed as MCP tools `train_sft` / `train_dpo` / `train_grpo` / `merge_adapters_ties`
  (`data_science_mcp/mcp/mcp_trainers.py`, tag `model-training`); plan-by-default, `execute=true` to run.
  torch/peft/bitsandbytes/httpx ship in the new optional `data-science-mcp[training]` extra and are imported
  lazily, so the package still installs/imports without them. Loss kernels + train loops are CPU-smoke-tested
  on a toy model (no GPU/HF download); the GB10 runs the real fine-tunes. Sources b3-02/b4-01/b6-01/b6-04/b7-03.
- **Wave-D fine-tune pipeline + deploy seam (CONCEPT:AHE-3.1)** — `data_science_mcp/training_pipeline.py`:
  `run_sft_pipeline` sequences traces → SFT corpus → plan → train → reliability-eval (`eval_hooks`) → save
  checkpoint → `register_checkpoint`, and `register_checkpoint` binds the trained checkpoint to a model-registry
  role (`ModelDefinition` + unique-tag `RoleSpec`) so `pick_for_role`/`create_model(role=…)` serve it with no
  hot-path edit (idempotent re-deploy). CPU-smoke-tested end-to-end on a toy model + the role-binding deploy seam
  against a real `ModelRegistry` (`tests/test_training_pipeline.py`, 4 tests). GB10 runs the real OpenSeeker SFT;
  see `docs/training.md`. The only GPU-gated step is the fine-tune itself.

### Fixed
- **`data-science-mcp` ↔ `epistemic-graph` seam hardening (B5)** — `_near_pairs_engine` reused the
  cached `MLEngine._rust_client()` singleton instead of opening (and leaking) a fresh
  `SyncEpistemicGraphClient.connect()` — with its own thread/event-loop/socket — on every
  `near_duplicate_pairs` call. The local O(n²) cosine fallback now **warns** (was silent
  `logger.debug`) with the row count and **refuses** above `DSM_NEAR_PAIRS_LOCAL_MAX` (default 20k)
  rather than burning CPU on a missing engine; `use_engine=False` still forces the local path.

## [0.8.0] - 2026-05-22

### Added
- Initial CHANGELOG.md creation
- docs/concepts.md with CONCEPT ID registry

### Changed
- Standardized project structure per agent-packages ecosystem conventions

# Changelog

All notable changes to `data-science-mcp` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Shortcut-resistant search-task corpora (CONCEPT:KG-2.67/2.68/2.69, AHE-3.27)** —
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

## [0.8.0] - 2026-05-22

### Added
- Initial CHANGELOG.md creation
- docs/concepts.md with CONCEPT ID registry

### Changed
- Standardized project structure per agent-packages ecosystem conventions

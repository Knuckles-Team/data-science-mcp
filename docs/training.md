# In-House Model Training (Wave C)

> **CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort** — Training Substrate · **CONCEPT:DS-ECO.mcp.model-training** — Model Training Operations
> Part of the cross-repo [In-House Training Substrate](https://github.com/Knuckles-Team/agent-utilities/blob/main/docs/architecture/in_house_training_substrate.md).

`data-science-mcp` owns the **corpus + gradient-trainer** half of the framework's
self-training stack: it turns execution traces into SFT/DPO/GRPO datasets and
fine-tunes open-weight models against them. Everything except the GPU fine-tune
*runs* is deterministic and CPU-testable; the real runs target the GB10 box.

## Layers

### 1. Deterministic data / reward engine (no GPU)
`data_science_mcp/training_data.py` — pure-Python builders reusing the
agent-utilities reward spine (`agent_utilities.graph.training_signals`):

| Builder | Output | Source paper |
|---|---|---|
| `build_sft_examples` | `{prompt, completion}` | OpenSeeker / MeMo |
| `build_preference_pairs` | `{prompt, chosen, rejected, failure_point}` | MedCausalX |
| `build_grpo_groups` | group-normalized advantages | ATLAS / SDAR |
| `filter_by_difficulty` / `score_reward` | data-quality + composite reward | OpenSeeker / ATLAS |

MCP tools: `build_training_dataset`, `compose_reward` (tag `model-training`).

### 2. Gradient trainers (`data-science-mcp[training]`)
Install the extra (`pip install .[training]` → torch / transformers / peft /
bitsandbytes / httpx). Modules under `data_science_mcp/`:

- **`trainers/objectives.py`** — torch loss kernels: masked cross-entropy, sequence
  log-prob, Bradley-Terry `dpo_loss`, group-relative `grpo_surrogate` (+ token-masked
  LA-GRPO), Schulman-k3 `approx_kl`.
- **`trainers/base.py`** — `TrainConfig` + `TrainerBase`: a pure `plan()` (step/batch
  accounting, no torch) and dependency-injectable model/tokenizer so the loop is
  CPU-smoke-testable on a toy model.
- **`trainers/{sft,dpo,grpo}_trainer.py`** — concrete trainers implementing the
  `training_data.Trainer` Protocol.
- **`peft_manager.py`** — `LoraSpec`/`PeftManager` (LoRA/QLoRA, lazy peft) + pure-numpy
  `ties_merge` (MeMo multi-adapter merge, CPU).
- **`tokenizer_registry.py`** — special/functional-token injection + embedding resize.
- **`rollout_buffer.py`** — prompt→generation→logprob→reward staging with an
  injected `InferenceBackend` + GRPO export.
- **`trainers/eval_hooks.py`** — score a checkpoint with the **AHE-3.1 reliability
  suite** (`agent_utilities.harness.reliability_scorers`).

MCP tools: `train_sft`, `train_dpo`, `train_grpo`, `merge_adapters_ties`. They are
**plan-by-default** — they return the training plan and only run when called with
`options.execute=true` (and torch present).

### 3. Rust performance path
The loss/optimizer kernels also exist in pure Rust in `epistemic-graph`
(`client.datascience.{softmax,cross_entropy,dpo_loss,grpo_surrogate,kl_divergence,adam_step,sgd_step}`),
so a trainer can batch a step over the wire in one round-trip. Same math as
`trainers/objectives.py`; torch is the default, Rust is the optimization.

## Example

```python
from data_science_mcp import training_data as td
from data_science_mcp.trainers import get_trainer, TrainConfig

# 1) Build an SFT corpus from traces (deterministic, no GPU).
examples = td.build_sft_examples(traces)

# 2) Plan the run (pure — no torch needed to inspect it).
trainer = get_trainer("sft", TrainConfig(base_model="Qwen/Qwen2.5-1.5B-Instruct",
                                         epochs=1, batch_size=8))
print(trainer.plan(examples))   # {planned_steps, effective_batch, ...}

# 3) Run it (needs data-science-mcp[training] + a GPU for a real base model).
report = trainer.train(examples)   # {steps, losses, final_loss, ...}
```

## End-to-end pipeline + deploy seam (Wave D)

`data_science_mcp/training_pipeline.py` ties the whole flow into one call:

```
traces → SFT corpus → plan → train → reliability-eval → save checkpoint
       → register as a ModelDefinition bound to a role (goes live)
```

```python
from data_science_mcp.training_pipeline import run_sft_pipeline, DeploymentTarget
from data_science_mcp.trainers import TrainConfig, peft_manager  # peft via [training]
from data_science_mcp.peft_manager import LoraSpec
from agent_utilities.models.model_registry import ModelRegistry

# OpenSeeker SFT on the GB10: Qwen2.5-1.5B + LoRA, served by the local vLLM.
report = run_sft_pipeline(
    TrainConfig(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir="/models/openseeker-sft",
        epochs=1, batch_size=8, lr=2e-4,
        lora=LoraSpec(r=16, alpha=32),          # LoRA; quant_4bit=True for QLoRA
    ),
    traces=openseeker_trajectories,             # ~10k synth trajectories (data engine)
    eval_cases=reliability_eval_cases,          # AHE-3.1 internalization/safety checks
    registry=my_registry,                       # the live ModelRegistry
    deploy=DeploymentTarget(
        role="generator",
        served_model_name="qwen2.5-1.5b-openseeker",
        base_url="http://localhost:8000/v1",    # the running vLLM
    ),
    checkpoint_id="openseeker-sft-v1",
)
```

The pipeline returns a structured report (`data`/`plan`/`train`/`eval`/
`checkpoint`/`deployment`). Omit `registry`/`deploy` for a train-only run; omit
`model`/`tokenizer` to load the real HF base (a toy model can be injected for a CPU
smoke).

### Deploy seam

`register_checkpoint` (called by the pipeline) appends a `ModelDefinition` for the
checkpoint and binds the target role to it via a unique tag, so
`model_registry.pick_for_role(role)` (and therefore `create_model(role=…)`) resolves
to the new model — **no hot-path edit**. Serve the checkpoint through the running
vLLM. Re-deploying the same `checkpoint_id` is idempotent.

### Build-now / run-later

Everything above is CPU-smoke-tested end-to-end on a toy model
(`tests/test_training_pipeline.py`). On the GB10 the only deltas are: pin
Blackwell `torch`/`peft`/`bitsandbytes`/`vllm`, point `base_model` at the real
checkpoint, and run on the GPU. The orchestration, evaluation, and deploy seam are
identical.

## Build-now / run-later

| Layer | Status |
|---|---|
| Data/reward engine | ✅ runnable now (no GPU) |
| Trainers (loss kernels + loops) | ✅ CPU-smoke-tested on a toy model |
| Real fine-tunes | ⛔ GB10 (pin Blackwell torch/peft/bnb/vllm); first run = OpenSeeker SFT |

See [`WAVE_C_INFRA.md`](https://github.com/Knuckles-Team/data-science-mcp/blob/main/.specify/specs/research-evolution-20260606/WAVE_C_INFRA.md)
for per-paper GB10 requirements.

---

# High-Caliber LLM Trainer (CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer…006)

The substrate above is hardened into a full LLM trainer: robust fine-tuning at
scale, a corpus curation engine, **pretraining a model from random init**, and an
**agent-driven** workflow that runs the whole loop. See the SDD spec at
the versioned design references distributed with the corresponding release.

## Robustness & throughput knobs (CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer)

`TrainConfig` gained additive fields (defaults reproduce the original behaviour
exactly). They flow through one shared optimisation loop, `trainers/loop.py::run_loop`:

| Field | Effect |
|---|---|
| `precision` (`fp32`/`fp16`/`bf16`) | AMP autocast (+ `GradScaler` on fp16) |
| `grad_accum` | gradient accumulation → larger effective batch |
| `max_grad_norm` | gradient clipping |
| `warmup_steps` + `lr_scheduler` (`cosine`/`linear`/…) | HF `get_scheduler` (default `constant`) |
| `gradient_checkpointing` | activation checkpointing (memory↓) |
| `attn_impl="flash_attention_2"` | FlashAttention-2 (needs `[training-scale]`) |
| `use_liger` | fused Triton kernels (needs `[training-fast]`) |
| `save_steps` / `save_total_limit` / `resume_from` | periodic checkpoints + resume |
| `distributed` (`fsdp`/`deepspeed`) | sharded multi-GPU (see below) |
| `tracker` (`mlflow`/`wandb`) + `kg_log` | experiment tracking + KG mirror |

## Scale-out: FSDP **and** DeepSpeed (CONCEPT:DS-AHE.trainer.concept-4)

Both are first-class peers via `trainers/accelerate_launch.py` (🤗 Accelerate). For a
real multi-GPU/-node run, generate a launch config and command with
`data_science_mcp.launch`:

```python
from data_science_mcp.launch import (
    fsdp_accelerate_config, deepspeed_zero3_config, write_config, build_launch_command)

write_config(fsdp_accelerate_config(num_processes=8, mixed_precision="bf16"), "fsdp.yaml")
cmd = build_launch_command("data_science_mcp.trainers.pretrain_trainer",
                           distributed="fsdp", num_processes=8, config_file="fsdp.yaml")
# multi-node: same call + num_machines/machine_rank/main_process_ip (homelab OR cloud)
```

`flash-attn` and `deepspeed` are **GPU-host installs** (`[training-scale]`, need the
CUDA toolchain) — not CI/CPU-installable.

## Corpus curation engine (CONCEPT:DS-AHE.trainer.data-engine)

`data_engine.py` — data quality is what separates good models from bad ones:

- `stream_corpus` — list / `.jsonl` / `.txt` / 🤗 `datasets(streaming=True)`.
- `dedup` — exact (content-hash) + near-duplicate (cosine/LSH). The all-pairs search
  is offloaded to the **epistemic-graph** Rust `find_similar_pairs` when a live engine
  is present, with a local-cosine fallback. Embeddings use a deterministic hashing
  vectorizer — no embedding model required.
- `decontaminate` — drop training records that leak held-out eval examples.
- `quality_filter`, `pack_sequences` (packed pretrain windows), `dataset_provenance`
  (a `DatasetVersion` lineage node, optionally mirrored into the KG).

MCP tools (tag `data-engine`): `curate_corpus`, `dedup_corpus`,
`decontaminate_corpus`, `dataset_lineage`.

## Pretrain from random init (CONCEPT:DS-AHE.trainer.concept-2)

For genuinely *new* models (≤~1.5B in a homelab; scale the spec up on cloud GPUs):

```python
from data_science_mcp.training_pipeline import run_pretrain_pipeline
from data_science_mcp.trainers import TrainConfig, PretrainSpec

report = run_pretrain_pipeline(
    TrainConfig(output_dir="/models/tiny-lm", max_seq_len=2048, precision="bf16",
                lr=3e-4, warmup_steps=2000, lr_scheduler="cosine",
                save_steps=1000, distributed="fsdp"),
    corpus=curated_records,                 # {text} records from the data engine
    spec=PretrainSpec(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
                      max_position_embeddings=2048),
    train_tokenizer_first=True,             # BPE tokenizer trained on the corpus
)
```

The model is built with **random weights** (`AutoConfig` → `from_config`, *not*
`from_pretrained`); `tokenizer_trainer.py` trains a byte-level BPE tokenizer first.
MCP tools: `train_tokenizer`, `pretrain_model` (plan-first like the fine-tuners).

## Agent-driven training (CONCEPT:AU-AHE.trainer.concept-2)

The whole loop is exposed as an agent workflow. Four agent personas (prompt JSONs in
`agent-utilities/agent_utilities/prompts/`): `data_curator`, `training_engineer`,
`eval_judge`, `ml_orchestrator`. They are bound into the `model_training_team`
TeamConfig and driven by the `train_model` workflow skill
(`universal-skills/.../workflows/ml/train_model/`):

```
prepare_corpus → curate/dedup/decontaminate → (train_tokenizer) →
  train(sft|pretrain) → eval → gate → align(dpo|grpo) → eval → gate →
  merge_adapters → final_eval → register_model
```

Run it via `graph_orchestrate(action="execute_workflow", name="train_model", task=…)`.

## Benchmark evaluation (CONCEPT:DS-AHE.trainer.concept-5)

Alongside the AHE-3.1 reliability suite, `eval_hooks.evaluate_benchmarks(model_path,
tasks)` scores a checkpoint on Hugging Face LightEval tasks (`[eval]` extra; graceful
no-op when absent).

## Compute reality

"High-caliber from scratch" at frontier scale is a capital problem, not a code one —
nobody pretrains a frontier LLM in a homelab. The realistic targets here are
(a) robust SFT/DPO/GRPO of 1.5B–8B bases (QLoRA fits one 24 GB card; FSDP/DeepSpeed
for full FT) and (b) genuine pretraining of small models (≤~1.5B). Given the homelab
GPU situation, plan for cloud-burst GPUs for anything larger — the launcher targets
both from one command. **All code is CPU-unit-tested on a toy model; GPU runs are
validated on a GPU host, not in CI.**

## Build-now / run-later (LLM trainer)

| Layer | Status |
|---|---|
| Data curation engine | ✅ runnable now (no GPU) |
| Trainer hardening (precision/accum/clip/sched/ckpt/resume) | ✅ CPU-smoke-tested |
| Pretrain-from-scratch loop | ✅ CPU-smoke-tested on a toy model |
| Agent workflow (personas + team + skill) | ✅ compiles + validates |
| FSDP / DeepSpeed multi-GPU, flash-attn, real pretrain | ⛔ GPU host (configs + launcher ready) |

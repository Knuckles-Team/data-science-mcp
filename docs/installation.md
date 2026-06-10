# Installation & Dependencies

`data-science-mcp` installs light by default and pulls heavy ML/GPU dependencies only
through **optional extras**, so you install exactly what a capability needs. The core
package imports and runs (planning, data curation, engine-backed ML) with **no torch
and no GPU**.

## Install

```bash
# Core (engine-backed ML + data curation + plan-only training):
pip install data-science-mcp

# A capability bundle (see the matrix below):
pip install "data-science-mcp[training]"          # run SFT/DPO/GRPO + pretrain
pip install "data-science-mcp[training,tracking]" # + MLflow experiment tracking

# From source (this repo):
pip install -e ".[training]"
```

> **`[all]` does _not_ include the GPU training extras.** It bundles the agent +
> sample-dataset deps (`agent`, `datasets`) but **not** `training` /
> `training-scale` / `training-fast`, because those are heavy and (for
> `training-scale`) need a CUDA toolchain. Install training extras explicitly.

## Extras → dependencies

| Extra | Brings in | Weight |
|---|---|---|
| _(core)_ | `agent-utilities[mcp]`, `epistemic-graph[datascience]`, `polars`, `numpy` | light, no GPU |
| `datasets` | `scikit-learn` | light — sample-dataset loaders only (iris/diabetes/…), **not** compute |
| `training` | `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets` (HF), `tokenizers`, `httpx` | heavy, GPU-oriented (CPU-installable) |
| `training-scale` | `deepspeed`, `flash-attn` | **GPU host only** — needs the CUDA toolchain (`nvcc`); **not** CI/CPU-installable |
| `training-fast` | `liger-kernel` | fused Triton kernels (supported GPU archs) |
| `eval` | `lm-eval` | EleutherAI benchmark harness |
| `tracking` | `mlflow` | experiment-tracking dashboard (self-hostable). `wandb` is supported by `RunTracker` but install it yourself |
| `agent` | `agent-utilities[agent,logfire]` | the bundled Pydantic-AI A2A agent runtime |
| `all` | `agent-utilities[mcp,agent,logfire]`, `epistemic-graph[datascience]`, `scikit-learn` | everything **except** the GPU training extras |

All heavy deps are **lazily imported** — they load only when a capability that needs
them actually executes, so the package installs and imports without them.

## Capability → what you need

| Capability | Tools / API | Needs |
|---|---|---|
| Engine-backed ML (fit/predict/evaluate/cross-validate) | `fit_model`, `predict`, `evaluate_model`, `cross_validate`, `MLEngine` | core + a running **epistemic-graph** engine (`EPISTEMIC_GRAPH_SOCKET`/`_TCP`) |
| Sample datasets (iris/diabetes/…) | `load_dataset` | `[datasets]` (CSV needs nothing extra — uses polars) |
| Quant / finance kernels | `quant_*` | core + epistemic-graph engine |
| **Corpus curation** (dedup/decontaminate/quality-filter/pack/lineage) | `curate_corpus`, `dedup_corpus`, `decontaminate_corpus`, `dataset_lineage` (CONCEPT:ML-002) | **core only** (pure-Python; epistemic-graph HNSW accelerates near-dup search when present) |
| HF `datasets` streaming corpora | `stream_corpus({"hf": …})` | `[training]` |
| **Plan** a training run (steps/effective-batch/params) | `train_sft`/`train_dpo`/`train_grpo`/`pretrain_model`/`train_tokenizer` with `execute=false` | **core only** (no torch) |
| **Run** SFT / DPO / GRPO fine-tunes | same tools with `execute=true` | `[training]` + a GPU |
| LoRA / QLoRA adapters | `LoraSpec(quant_4bit=…)`, `merge_adapters_ties` | `[training]` (QLoRA fits a 1.5B–8B base on one 24 GB card) |
| **Pretrain from random init** (CONCEPT:ML-003) | `pretrain_model`, `train_tokenizer`, `run_pretrain_pipeline` | `[training]` + a GPU (small models ≤~1.5B in a homelab) |
| Multi-GPU **FSDP** | `TrainConfig(distributed="fsdp")` + `launch/` | `[training]` (FSDP ships with accelerate+torch) |
| Multi-GPU **DeepSpeed ZeRO-3** | `TrainConfig(distributed="deepspeed")` | `[training-scale]` (GPU host) |
| FlashAttention-2 | `TrainConfig(attn_impl="flash_attention_2")` | `[training-scale]` (GPU host) |
| Fused kernels | `TrainConfig(use_liger=True)` | `[training-fast]` |
| Benchmark eval (hellaswag/arc/gsm8k…) | `evaluate_benchmarks` (CONCEPT:ML-006) | `[eval]` |
| Experiment tracking | `TrainConfig(tracker="mlflow", kg_log=True)` | `[tracking]` (KG mirror needs the agent-utilities KG facade) |
| A2A agent / Web UI | `data-science-agent` | `[agent]` |

## GPU-host notes

- `training-scale` (`deepspeed`, `flash-attn`) compiles CUDA extensions — install on a
  GPU host with the matching CUDA toolchain, never in CI/CPU images.
- On **Blackwell (GB10, sm_120)** pin CUDA-12.6+/cu12x-enabled `torch`/`peft`/
  `bitsandbytes`/`vllm` builds.
- Real fine-tunes/pretrains need a GPU; everything above the "Run" rows is
  CPU-testable (plan, curate, toy-model smokes). See [Model Training](training.md) for
  the GPU runbook and [Concepts](concepts.md) for the `CONCEPT:ML-*` registry.

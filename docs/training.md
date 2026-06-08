# In-House Model Training (Wave C)

> **CONCEPT:AHE-3.1** — Training Substrate · **CONCEPT:DSCI-004** — Model Training Operations
> Part of the cross-repo [In-House Training Substrate](../../agent-utilities/docs/architecture/in_house_training_substrate.md).

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
- **`rollout_buffer.py`** — prompt→generation→logprob→reward staging with
  `VLLMRolloutClient` (generations served by the running vLLM) + GRPO export.
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

See [`WAVE_C_INFRA.md`](../../../.specify/specs/research-evolution-20260606/WAVE_C_INFRA.md)
for per-paper GB10 requirements.

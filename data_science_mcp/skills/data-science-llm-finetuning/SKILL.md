---
name: data-science-llm-finetuning
description: >-
  LLM gradient fine-tuning and corpus pipeline on the data-science-mcp compute server —
  curate a corpus (dedup/decontaminate/quality), build an SFT/DPO/GRPO dataset from
  traces, then run SFT, DPO, reward-model, PPO, GRPO or from-scratch pretraining, train
  a tokenizer, and TIES-merge PEFT adapters. Use when the agent must post-train or
  pretrain a language model, assemble a preference/RL corpus, or produce/merge LoRA
  adapters. Trainers plan by default and only execute with options.execute=true; the
  heavy path needs the [training] extra + a GPU. Do NOT use for tabular scikit-style ML
  (use data-science-model-training) or interpretability grading
  (use data-science-model-reliability).
license: MIT
tags: [data-science, llm, fine-tuning, sft, dpo, ppo, grpo, corpus, peft, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Data Science — LLM Fine-Tuning

Post-train and pretrain language models over the **data-science-mcp** MCP server. The
canonical flow is **curate → build dataset → train → (merge adapters)**. Trainers return
a *plan* unless you pass `"execute": true` in `options_json`; execution runs gradient
steps in-process and needs the `[training]` extra (torch/transformers/peft/accelerate)
plus a GPU.

## When to use
- Curate raw text into a clean corpus (quality-filter + dedup + decontaminate).
- Convert traces/samples into an SFT, DPO, or GRPO training corpus.
- Run a trainer: `train_sft`, `train_dpo`, `train_reward`, `train_ppo`, `train_grpo`,
  or `pretrain_model` (causal LM from random init).
- Train a byte-level BPE tokenizer, or TIES-merge multiple task-vector adapters onto a base.

## When NOT to use
- Tabular / classical ML (fit_model, cross_validate, Pareto) → `data-science-model-training`.
- Interpretability probes, response grading, reliability suites →
  `data-science-model-reliability`.
- GPU kernel micro-optimization → the `ds_specialize_kernel` tool.

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`data-science-mcp`** MCP server.

| Variable / extra | Required | Notes |
|------------------|----------|-------|
| `data-science-mcp[training]` extra | for execution | torch/transformers/peft/bitsandbytes/accelerate/datasets/tokenizers |
| GPU (CUDA; Blackwell → cu12x builds) | for execution | trainers plan without it; only exec needs it |
| `INFERENCE_BACKEND` / `INFERENCE_BASE_URL` | for served rollouts | PPO verifier-reward rollouts against vLLM/SGLang |

Without the extra, trainers still return a *plan* (`executed: false`), so you can dry-run
the pipeline shape on CPU/CI.

## Tools & actions
| Tool | Purpose |
|------|---------|
| `dedup_corpus` | Exact/near dedup a list of `{text}` records |
| `decontaminate_corpus` | Drop records overlapping held-out eval texts |
| `curate_corpus` | One-shot quality + dedup + decontaminate, with provenance |
| `prepare_pretrain_data` | Tokenize a corpus spec to `.npy`/`.hdf5` (needs `tokenizer`, `execute`) |
| `dataset_lineage` | Record/emit a dataset's provenance chain |
| `build_training_dataset` | Turn traces/samples into `sft`/`dpo`/`grpo` records |
| `compose_reward` | Weighted, conditionally-gated composite reward score |
| `train_sft` | Supervised fine-tune on `{prompt, completion}` |
| `train_dpo` | Preference-optimise on `{prompt, chosen, rejected}` |
| `train_reward` | Bradley-Terry reward model on preference pairs |
| `train_ppo` | Actor-critic PPO on scored rollouts |
| `train_grpo` | GRPO on advantage-tagged groups |
| `pretrain_model` | Causal LM from random init (`spec` architecture knobs) |
| `train_tokenizer` | Byte-level BPE tokenizer from a text corpus |
| `merge_adapters_ties` | TIES-merge task vectors onto a base (`density`, `scaling`) |

### Key parameters
- All record/option args are **JSON strings**: `dataset_json`, `options_json`,
  `records_json`, `items_json`, `corpus_json`.
- `options_json` carries `TrainConfig` fields + `{"execute": bool}`. DPO adds `beta`;
  GRPO/PPO add `clip_eps`, `kl_coef` (PPO also `vf_coef`, `gamma`, `gae_lambda`,
  `reward_source`).
- `pretrain_model` `options.spec`: `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `vocab_size`, `max_position_embeddings`, `model_type`.
- `curate_corpus` `options`: `{name, version, parents, kg_log, ...}` + per-stage knobs.

## Recipes
Curate then SFT (plan-only, no GPU):
```text
curate_corpus(records_json="[{\"text\":\"...\"}]",
              options_json="{\"name\":\"docs\",\"version\":\"v1\"}")
build_training_dataset(kind="sft", items_json="[{...traces...}]")
train_sft(dataset_json="[{\"prompt\":\"Q\",\"completion\":\"A\"}]",
          options_json="{\"execute\": false}")
```
DPO / reward pair from the same preference corpus:
```text
train_dpo(dataset_json="[{\"prompt\":\"Q\",\"chosen\":\"A\",\"rejected\":\"B\"}]",
          options_json="{\"beta\": 0.1, \"execute\": true}")
train_reward(dataset_json="[{\"prompt\":\"Q\",\"chosen\":\"A\",\"rejected\":\"B\"}]",
             options_json="{\"execute\": true}")
```
TIES-merge two task vectors onto a base:
```text
merge_adapters_ties(base_json="{\"w\":[0.0,0.1]}",
                    task_vectors_json="[{\"w\":[0.2,-0.1]},{\"w\":[0.05,0.3]}]",
                    options_json="{\"density\": 0.2, \"scaling\": 1.0}")
```

## Gotchas
- Trainers **plan, not run**, unless `options_json` contains `"execute": true`.
- Record shapes are trainer-specific: SFT=`{prompt, completion}`, DPO/reward=
  `{prompt, chosen, rejected}`, GRPO=`{prompt, samples:[{completion, reward, advantage}]}`,
  PPO=`{prompt, completion, reward?}`. Use `build_training_dataset` to shape them.
- The RLHF order is SFT → reward model → PPO; `train_ppo` with `reward_source=reward_model`
  needs an in-process reward model (prefer the `train_model` workflow for that path);
  the verifier path embeds a scalar `reward` per record.
- `prepare_pretrain_data` requires `options.tokenizer` (HF name or local dir) **and**
  `options.execute=true` to actually write; otherwise it returns a plan.
- Everything is JSON-string in / JSON-string out — parse the returned string.

## Related
- `data-science-model-training` — classical/tabular training + Pareto evolution.
- `data-science-model-reliability` — grade the fine-tuned model's behaviour.
- KG mapping: runs → `:TrainingRun` (`:trainerKind`, `:fineTunes` → `:LanguageModel`),
  corpora → `:Corpus` (`:curatedFrom` → `:Document`), adapters → `:Adapter`,
  tokenizers → `:Tokenizer`, reward models → `:RewardModel`.

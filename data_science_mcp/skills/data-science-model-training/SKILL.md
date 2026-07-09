---
name: data-science-model-training
skill_type: skill
description: >-
  Classical (tabular) ML lifecycle on the data-science-mcp compute server — load a
  dataset, fit a model, cross-validate, predict, evaluate, then rank and evolve model
  classes on a performance-vs-complexity Pareto frontier. Use when the agent must train
  a scikit-style regressor/classifier (LinearRegression, Ridge, RandomForest, …) on a
  named or CSV dataset, tune via k-fold CV, score held-out data, or decide which model
  class to keep. Compute runs in the epistemic-graph Rust engine, not in-process. Do NOT
  use for LLM/gradient fine-tuning or corpus curation (use data-science-llm-finetuning)
  or for interpretability/reliability grading (use data-science-model-reliability).
license: MIT
tags: [data-science, machine-learning, training, cross-validation, pareto, mcp]
metadata:
  author: Genius
  version: '0.1.0'
---
# Data Science — Model Training

Train and rank classical ML models over the **data-science-mcp** MCP server. Datasets
are loaded once by name, then referenced by that name across fit/CV/evaluate. Fitted
models are addressed by the returned `model_id`. Heavy compute executes in the
epistemic-graph Rust engine — these tools are thin, backend-agnostic wrappers.

## When to use
- Fit a tabular regressor/classifier on a built-in sample or a `.csv` file.
- k-fold cross-validate a model class to compare hyperparameters.
- Generate predictions or score a fitted model on a `train`/`test` split.
- Rank all fitted models by held-out R², or push model classes onto the evolutionary
  Pareto frontier (performance vs complexity) and read the non-dominated set.

## When NOT to use
- LLM / gradient fine-tuning (SFT/DPO/GRPO/PPO), corpus curation, tokenizers, adapters
  → `data-science-llm-finetuning`.
- Interpretability probes / response grading / reliability suites →
  `data-science-model-reliability`.
- GPU kernel specialization → the `ds_specialize_kernel` tool (separate surface).

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`data-science-mcp`** MCP server.

| Variable | Required | Notes |
|----------|----------|-------|
| `INFERENCE_BACKEND` / `INFERENCE_BASE_URL` | optional | Only for served-model workflows; classical fit needs neither |
| (built-in datasets) | optional | `california`/`diabetes`/`iris`/`wine`/`breast_cancer`/`digits` need the `[datasets]` extra (scikit-learn); CSV paths do not |

Compute is delegated to the epistemic-graph engine; no local scikit-learn is required
for CSV datasets.

## Tools & actions
| Tool | Purpose |
|------|---------|
| `load_dataset` | Load a built-in sample name or a `.csv` path (with `target_column`) |
| `describe_dataset` | Summary stats / schema of a loaded dataset |
| `split_dataset` | Deterministic train/validation/test partitions |
| `fit_model` | Fit `model_class` on `dataset_name`, return metrics + `model_id` |
| `cross_validate` | k-fold CV for a `model_class` (compare hyperparameters) |
| `predict` | Predict with a fitted `model_id` over a list of feature dicts |
| `evaluate_model` | Score a `model_id` on the `train`/`test` split |
| `rank_models` | Rank all fitted models by stored test R² |
| `evolve_model_class` | Submit `(performance, complexity)` for a class to the Pareto frontier |
| `get_pareto_frontier` | Read the current non-dominated model-class set |

### Key parameters
- `model_class` — class name string, e.g. `LinearRegression`, `Ridge`, `RandomForest`.
- `dataset_name` / `name` — the name you loaded with (or the CSV path).
- `hyperparameters_json` — a **JSON string** (not an object) of model hyperparameters.
- `test_size` (default `0.2`), `n_folds` (default `5`), `random_seed` (default `42`).
- `model_id` — returned by `fit_model`; required by `predict` / `evaluate_model`.

## Recipes
Load a CSV dataset and fit a Ridge model:
```text
load_dataset(name="/data/housing.csv", target_column="price")
fit_model(model_class="Ridge", dataset_name="/data/housing.csv",
          hyperparameters_json="{\"alpha\": 1.0}", test_size=0.2)
```
Cross-validate before committing to hyperparameters:
```text
cross_validate(model_class="RandomForest", dataset_name="iris",
               n_folds=5, hyperparameters_json="{\"n_estimators\": 200}")
```
Rank fitted models, then record a class on the Pareto frontier:
```text
rank_models()
evolve_model_class(model_class="Ridge", base_performance=0.91, complexity=2.0)
get_pareto_frontier()
```

## Gotchas
- `hyperparameters_json` / `inputs_json` are **JSON strings** — serialize them; passing a
  raw object returns an `{"error": ...}` payload.
- Built-in sample datasets (`iris`, `wine`, …) require the optional `[datasets]` extra
  (scikit-learn); CSV files do not.
- `evolve_model_class` frontier semantics: higher `performance` is better, **lower**
  `complexity` is better — don't invert the complexity sign.
- `rank_models` only sees models fitted in the current engine session; fit before ranking.
- `predict`/`evaluate_model` need a `model_id` from a prior `fit_model` in the same run.

## Related
- `data-science-llm-finetuning` — gradient trainers + corpus curation for LLMs.
- `data-science-model-reliability` — interpretability + reliability grading of a fitted model.
- KG mapping: runs → `:TrainingRun`, fitted models → `:ModelArtifact` (`:modelId`,
  `:r2Test`), datasets → shared `:Dataset`, the frontier → `:ParetoFrontier`.

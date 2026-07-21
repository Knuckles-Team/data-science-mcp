# Data Science Model Reliability

Model interpretability and reliability grading on the data-science-mcp compute server — generate a structured suite of behavioural probes for a fitted model (feature attribution, point simulation, sensitivity, counterfactual, confidence calibration, data attribution), grade answers against expected references, and run the full graded suite for a pass-rate. Use when the agent must audit or explain a fitted model_id, score interpretability responses, or produce a reliability report. Do NOT use to train models (use data-science-model-training) or to fine-tune LLMs / curate corpora (use data-science-llm-finetuning).

# Data Science — Model Reliability & Interpretability

Audit a **fitted model** over the **data-science-mcp** MCP server. The flow is
**generate probes → answer → grade → suite pass-rate**. Every tool keys off a
`model_id` produced by `fit_model` (see `data-science-model-training`) in the same
engine session.

## When to use
- Generate a fixed 6-case interpretability suite for a `model_id` (attribution,
  simulation, sensitivity, counterfactual, calibration, data attribution).
- Grade a single response against an `expected` reference for a `test_id`.
- Run the whole suite from a `test_id → answer` map and get an aggregate pass-rate.

## When NOT to use
- Training, cross-validation, ranking, Pareto evolution → `data-science-model-training`.
- LLM fine-tuning, corpus curation, adapters/tokenizers → `data-science-llm-finetuning`.
- Standardized benchmark scoring at scale (LightEval) → the `[eval]` extra directly, not
  this suite (this suite is a lightweight per-model behavioural probe).

## Prerequisites & environment
Connect via the `mcp-client` skill against the **`data-science-mcp`** MCP server. A model
must already be fitted (`fit_model`) so its `model_id` resolves; the probes read the
model's coefficients/feature importances, intercept, `n_train`, and test metrics.

| Requirement | Notes |
|-------------|-------|
| A fitted `model_id` | Grading fails with `{"error": "Model ... not found."}` otherwise |
| No extra deps | Runs on the base install; no GPU/scikit-learn required for CSV-fit models |

## Tools & actions
| Tool | Purpose |
|------|---------|
| `generate_interpretability_tests` | Emit 6 typed test cases (`test_id`, `category`, `question`, `expected_hint`) for a `model_id` |
| `grade_response` | Grade one `response` vs `expected` for a `test_id` → `{passed, score}` |
| `run_interpretability_suite` | Grade a whole `test_id → answer` map → aggregate score/pass-rate |

### Key parameters
- `model_id` — the fitted model to probe (required by generate + suite).
- `test_id` — a case id from `generate_interpretability_tests` (`att_`/`sim_`/`sens_`/
  `cf_`/`conf_`/`data_` prefix + model id).
- `response`, `expected` — the answer and its reference for `grade_response`.
- `answers_json` — a **JSON string** mapping `test_id` → response for the suite.

## Recipes
Full audit of a fitted model:
```text
generate_interpretability_tests(model_id="<model_id>")
# answer each question, then grade the batch:
run_interpretability_suite(model_id="<model_id>",
    answers_json="{\"conf_<model_id>_0\": \"0.91\", \"data_<model_id>_0\": \"120\"}")
```
Grade a single response:
```text
grade_response(test_id="conf_<model_id>_0", response="0.91", expected="0.91")
```

## Gotchas
- Every tool needs a live `model_id` — fit the model first, in the **same** engine
  session; a missing model returns `{"error": "Model <id> not found."}`.
- The suite is always **6 cases** with fixed categories; `test_id`s are derived from the
  model id (`conf_<model_id>_0`, `data_<model_id>_0`, …) — read them from
  `generate_interpretability_tests`, don't guess.
- `answers_json` is a **JSON string** (test_id → response), not an object.
- Grading is reference-match based (`passed`/`score` 1.0/0.0); phrase numeric answers to
  match the `expected` reference (e.g. the R² or `n_train` value), not prose.

## Related
- `data-science-model-training` — produce the `model_id` this skill audits.
- KG mapping: a run → `:InterpretabilitySuite` (`:interpretedBy` a `:ModelArtifact`),
  each case → `:InterpretabilityTest` (`:hasTest`), grades → shared `:OutcomeEvaluation`.

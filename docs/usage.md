# Usage — MCP / Python API / CLI

`data-science-mcp` exposes the same capability three ways: as **MCP tools** an agent
calls, as a **Python API** (`MLEngine`) you import, and as **console scripts** you run
directly. The ecosystem role and the full concept registry are in
[Overview](overview.md).

## As an MCP server

Once [deployed](deployment.md), the server registers action-routed tools grouped into
five independently togglable domains. Each domain is gated by an environment toggle so
you can keep the LLM context lean.

| Domain | Toggle | Representative actions |
|---|---|---|
| Model training | `MODEL_TRAININGTOOL` | fit an estimator, build a training dataset, compose a reward, `train_sft` / `train_dpo` / `train_grpo` |
| Model evolution | `MODEL_EVOLUTIONTOOL` | submit a model to the Pareto frontier, rank fitted models by test R² |
| Interpretability | `INTERPRETABILITYTOOL` | generate a structured suite of interpretability test cases for a model |
| Data management | `DATA_MANAGEMENTTOOL` | load and describe a dataset by name or CSV path, split a dataset |
| Quantitative finance | `QUANTTOOL` | state-space / stat-arb, signal combination, SABR volatility-surface tools |

Example agent prompts that map onto these tools:

- *"Fit a RandomForest on the diabetes dataset and report its metrics"* → model-training fit
- *"Rank every fitted model by test R²"* → model-evolution ranking
- *"Generate the interpretability suite for model `<id>`"* → interpretability tool

## As a Python API

`MLEngine` is a stateful façade over the `epistemic-graph` Rust compute engine.
Fitted models and loaded datasets are held in-memory and referenced by ID for
subsequent operations.

```python
from data_science_mcp.ml_engine import MLEngine

engine = MLEngine()

# Load a dataset (built-in sample names need the [datasets] extra; CSV paths do not)
engine.load_dataset("diabetes")

# Fit an estimator — compute runs in the epistemic-graph engine
result = engine.fit("RandomForest", "diabetes", test_size=0.2)
model_id = result["model_id"]
print(result["metrics"])                 # r2 / rmse on the holdout

# Evaluate and cross-validate
print(engine.evaluate(model_id))
print(engine.cross_validate("Ridge", "diabetes", hyperparameters={"alpha": 1.0}))

# Rank every fitted model by test R²
print(engine.ranked_models())
```

Load a CSV instead of a sample dataset:

```python
engine.load_dataset("/path/to/data.csv", target_column="price")
engine.describe_dataset("/path/to/data.csv")
```

Supported estimators route automatically to the engine: `linearregression`, `ridge`,
`lasso`, `elasticnet`, `decisiontree`, `randomforest`, `gradientboosting`,
`adaboost`, and `svr`. To add a new estimator, implement it in the engine and register
its normalized name — see [Overview](overview.md) and the engine compute guide.

### In-house training substrate

The deterministic SFT/DPO/GRPO corpus and reward engine work with no GPU; the
gradient trainers require the `[training]` extra:

```python
from data_science_mcp import training_data as td
from data_science_mcp.trainers import get_trainer, TrainConfig

examples = td.build_sft_examples(traces)             # deterministic, no GPU
trainer = get_trainer("sft", TrainConfig(base_model="Qwen/Qwen2.5-1.5B-Instruct"))
print(trainer.plan(examples))                        # inspect the plan (pure)
```

See [Model Training](training.md) for the full pipeline and the deploy seam.

## As a CLI

`data-science-mcp` ships two console scripts.

Run the MCP server:

```bash
data-science-mcp --transport streamable-http --host 0.0.0.0 --port 8000
data-science-mcp --help
```

Run the A2A agent (needs the `[agent]` extra and a reachable MCP server):

```bash
data-science-agent \
  --provider openai --model-id gpt-4o \
  --mcp-url http://localhost:8000/mcp
```

See [Deployment](deployment.md) for transports, the agent wiring, reverse-proxy, and
DNS.

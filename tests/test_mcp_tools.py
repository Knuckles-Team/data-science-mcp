"""
Tests for programmatically verifying all registered MCP tools and MLEngine functions.
"""

import sys
import os

# Set dummy sys.argv before importing anything to prevent create_mcp_server parsing issues
sys.argv = ["mcp_server.py"]

import json
import tempfile
from typing import Any
import pytest
from unittest.mock import patch
import numpy as np

from data_science_mcp.ml_engine import MLEngine
from data_science_mcp.mcp_server import get_mcp_instance


@pytest.fixture(autouse=True)
def reset_ml_engine():
    """Ensure a clean MLEngine singleton state for every test."""
    engine = MLEngine()
    engine._datasets.clear()
    engine._models.clear()
    yield
    engine._datasets.clear()
    engine._models.clear()


def test_ml_engine_singleton():
    """Verify that MLEngine enforces a singleton pattern."""
    engine1 = MLEngine()
    engine2 = MLEngine()
    assert engine1 is engine2


def test_ml_engine_load_dataset(require_sklearn):
    """Test loading built-in sample datasets (optional sklearn) and CSV files."""
    engine = MLEngine()

    # Test loading iris (classification)
    res_iris = engine.load_dataset("iris")
    assert "error" not in res_iris
    assert res_iris["name"] == "iris"
    assert res_iris["n_features"] == 4
    assert res_iris["n_samples"] > 0
    assert "target_name" in res_iris

    # Test loading diabetes (regression)
    res_diab = engine.load_dataset("diabetes")
    assert "error" not in res_diab
    assert res_diab["name"] == "diabetes"

    # Test loading breast_cancer
    res_bc = engine.load_dataset("breast_cancer")
    assert "error" not in res_bc

    # Test unknown dataset name
    res_err = engine.load_dataset("unknown_dataset_name")
    assert "error" in res_err
    assert "Unknown dataset" in res_err["error"]

    # Test load dataset using custom CSV (written without pandas)
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
        tmp.write("feat1,feat2,label\n")
        for a, b, c in [
            (1.0, 2.0, 0.5),
            (2.0, 4.0, 1.5),
            (3.0, 6.0, 2.5),
            (4.0, 8.0, 3.5),
            (5.0, 10.0, 4.5),
        ]:
            tmp.write(f"{a},{b},{c}\n")
        tmp_name = tmp.name

    try:
        # Load custom CSV (auto-resolve last column as target)
        res_csv = engine.load_dataset(tmp_name)
        assert "error" not in res_csv
        assert res_csv["shape"] == [5, 2]
        assert res_csv["target_name"] == "label"

        # Load custom CSV with explicit target column
        res_csv_explicit = engine.load_dataset(tmp_name, target_column="feat1")
        assert "error" not in res_csv_explicit
        assert res_csv_explicit["target_name"] == "feat1"
        assert "feat2" in res_csv_explicit["feature_names"]
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def test_ml_engine_fit_predict_evaluate(require_engine, require_sklearn):
    """Test model fitting, predictions, and evaluation via the epistemic-graph engine."""
    engine = MLEngine()

    # Fit a model without loading dataset first (should trigger auto-load)
    res_fit = engine.fit("LinearRegression", "diabetes", test_size=0.2)
    assert "error" not in res_fit
    model_id = res_fit["model_id"]
    assert model_id.startswith("model:")
    assert "rmse_train" in res_fit["metrics"]
    assert "rmse_test" in res_fit["metrics"]

    # Test evaluation split
    eval_test = engine.evaluate(model_id, "diabetes", split="test")
    assert "error" not in eval_test
    assert eval_test["split"] == "test"

    eval_train = engine.evaluate(model_id, "diabetes", split="train")
    assert "error" not in eval_train
    assert eval_train["split"] == "train"

    # Test prediction with valid inputs
    feature_names = res_fit["feature_names"]
    test_inputs = [{f: 0.1 for f in feature_names}, {f: -0.2 for f in feature_names}]
    preds = engine.predict(model_id, test_inputs)
    assert len(preds) == 2
    assert all(isinstance(p, float) for p in preds)

    # Test prediction with invalid model ID
    with pytest.raises(ValueError, match="Unknown model_id"):
        engine.predict("invalid_model_id", test_inputs)

    # Test evaluation with invalid model ID
    eval_err = engine.evaluate("invalid_model_id", "diabetes")
    assert "error" in eval_err

    # Test fit failure with non-existent dataset
    res_fit_err = engine.fit("LinearRegression", "invalid_dataset_name")
    assert "error" in res_fit_err


def test_ml_engine_cross_validate(require_engine, require_sklearn):
    """Test cross-validation via the epistemic-graph engine."""
    engine = MLEngine()

    # Load dataset
    engine.load_dataset("diabetes")

    # Cross validate with Ridge and custom n_folds
    res_cv = engine.cross_validate(
        "Ridge", "diabetes", n_folds=3, hyperparameters={"alpha": 1.0}
    )
    assert "error" not in res_cv
    assert len(res_cv["rmse_per_fold"]) == 3
    assert "rmse_mean" in res_cv

    # Cross validate with invalid model class or dataset
    res_cv_err = engine.cross_validate("Ridge", "invalid_dataset")
    assert "error" in res_cv_err


def test_ml_engine_supported_models():
    """Verify model-name normalization and the supported-model registry."""
    for m in ("LinearRegression", "linear_regression", "Ridge", "RandomForest", "SVR"):
        assert MLEngine._is_supported(m), m
    assert not MLEngine._is_supported("UnknownEstimator")
    assert (
        MLEngine._normalize_model("Random_Forest-Regressor") == "randomforestregressor"
    )


def test_ml_engine_split_dataset(require_sklearn):
    """Verify train/test/validation splitting (engine when up, numpy fallback)."""
    engine = MLEngine()
    engine.load_dataset("iris")

    # Split with validation size
    res_split = engine.split_dataset(
        "iris", test_size=0.2, validation_size=0.1, random_seed=42
    )
    assert res_split["train_size"] > 0
    assert res_split["validation_size"] > 0
    assert res_split["total"] == 150

    # Split without validation size
    res_split_no_val = engine.split_dataset("iris", test_size=0.2, validation_size=0.0)
    assert "validation_size" not in res_split_no_val
    assert res_split_no_val["train_size"] > 0

    # Split non-existent dataset
    res_split_err = engine.split_dataset("non_existent", test_size=0.2)
    assert "error" in res_split_err


def test_ml_engine_errors():
    """Verify MLEngine error branches (engine-only compute; no sklearn fallback)."""
    engine = MLEngine()

    # Unknown dataset name (not a sample name, not a .csv path).
    res = engine.load_dataset("unknown_dataset_name")
    assert "error" in res and "Unknown dataset" in res["error"]

    # Unsupported model class is rejected before any engine call.
    engine._datasets["synthetic"] = {
        "X": np.zeros((5, 2)),
        "y": np.zeros(5),
        "feature_names": ["a", "b"],
        "target_name": "y",
    }
    res_fit = engine.fit("TotallyUnknownModel", "synthetic")
    assert "error" in res_fit and "Unsupported model" in res_fit["error"]

    res_cv = engine.cross_validate("TotallyUnknownModel", "synthetic")
    assert "error" in res_cv and "Unsupported model" in res_cv["error"]

    # Fit against a non-existent dataset surfaces the load error.
    res_fit_err = engine.fit("LinearRegression", "invalid_dataset_name")
    assert "error" in res_fit_err

    # predict/evaluate with an unknown model id.
    with pytest.raises(ValueError, match="Unknown model_id"):
        engine.predict("invalid_model_id", [{"a": 0.0, "b": 0.0}])
    assert "error" in engine.evaluate("invalid_model_id", "synthetic")


@pytest.mark.asyncio
async def test_mcp_data_management_tools(require_engine, require_sklearn):
    """Verify FastMCP load, describe, and split dataset tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()

    # 1. Test load_dataset tool
    load_tool = next(t for t in tools if t.name == "load_dataset")
    res_load = await load_tool.fn(name="iris", ctx=None)
    assert res_load["name"] == "iris"

    # 2. Test describe_dataset tool
    describe_tool = next(t for t in tools if t.name == "describe_dataset")
    res_desc = await describe_tool.fn(name="iris", ctx=None)
    assert "feature_stats" in res_desc
    assert res_desc["name"] == "iris"

    # Test describing non-existent dataset
    res_desc_err = await describe_tool.fn(name="non_existent", ctx=None)
    assert "error" in res_desc_err

    # 3. Test split_dataset tool
    split_tool = next(t for t in tools if t.name == "split_dataset")
    res_split = await split_tool.fn(
        name="iris", test_size=0.2, validation_size=0.1, random_seed=42, ctx=None
    )
    assert "train_size" in res_split


def test_tool_surface_client_matches_client_cls():
    """Regression guard for the describe_dataset tool<->client mismatch.

    ``register_tool_surface`` in ``mcp_server.py`` introspects ``client_cls``
    (``MLEngine``) to build the verbose 1:1 tool surface, but dispatches every
    call through ``getattr(get_client(), method_name)``. If ``get_client()``
    ever returns something other than an ``MLEngine`` instance again, every
    public ``MLEngine`` method surfaced as a tool (``describe_dataset``,
    ``compute_stats``, ...) would fail with
    ``'<Type>' object has no attribute '<method>'`` even though the method is
    real -- exactly the bug this test guards against.
    """
    from data_science_mcp.auth import get_client
    from data_science_mcp.ml_engine import MLEngine

    client = get_client()
    assert isinstance(client, MLEngine)

    public_methods = [
        name
        for name in dir(MLEngine)
        if not name.startswith("_") and callable(getattr(MLEngine, name, None))
    ]
    assert "describe_dataset" in public_methods  # sanity: surface isn't empty
    missing = [m for m in public_methods if not hasattr(client, m)]
    assert not missing, (
        f"get_client() is missing MLEngine methods surfaced as verbose MCP "
        f"tools: {missing}"
    )


@pytest.mark.asyncio
async def test_verbose_describe_dataset_tool_dispatches(
    monkeypatch, require_engine, require_sklearn
):
    """End-to-end regression test for the reported bug: calling the verbose,
    auto-derived ``data_science_describe_dataset`` tool (as a delegated task
    would via the fleet multiplexer's ``load_tools``) must dispatch to a real
    ``MLEngine.describe_dataset`` -- not raise
    ``'Client' object has no attribute 'describe_dataset'``.
    """
    monkeypatch.setenv("MCP_TOOL_MODE", "verbose")
    engine = MLEngine()
    engine.load_dataset("iris", "")

    mcp, _, _, _ = get_mcp_instance()
    result = await mcp.call_tool(
        "data_science_describe_dataset", {"params_json": json.dumps({"name": "iris"})}
    )
    payload = result.structured_content
    assert payload["name"] == "iris"
    assert "feature_stats" in payload


@pytest.mark.asyncio
def _tool(tools, name: str):
    """Look up a registered MCP tool by name."""
    return next(t for t in tools if t.name == name)


async def _assert_fit_model_tool(fit_tool) -> dict:
    """fit_model: happy path returns a model_id; invalid hyperparameters_json rejects
    (covers mcp_server.py line 81-82)."""
    res_fit = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )
    assert "model_id" in res_fit
    res_fit_err = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{invalid",
        test_size=0.2,
        ctx=None,
    )
    assert "error" in res_fit_err
    return res_fit


async def _assert_predict_tool(predict_tool, model_id: str, feature_names: list[str]) -> None:
    """predict: happy path, invalid json, and a nonexistent model ID (covers
    mcp_server.py line 114-115)."""
    inputs_json = json.dumps([{f: 0.1 for f in feature_names}])
    res_pred = await predict_tool.fn(
        model_id=model_id, inputs_json=inputs_json, ctx=None
    )
    assert "predictions" in res_pred
    assert len(res_pred["predictions"]) == 1

    res_pred_err = await predict_tool.fn(
        model_id=model_id, inputs_json="{invalid", ctx=None
    )
    assert "error" in res_pred_err

    res_pred_err2 = await predict_tool.fn(
        model_id="nonexistent_id", inputs_json=inputs_json, ctx=None
    )
    assert "error" in res_pred_err2


async def _assert_evaluate_model_tool(evaluate_tool, model_id: str) -> None:
    res_eval = await evaluate_tool.fn(
        model_id=model_id, dataset_name="iris", split="test", ctx=None
    )
    assert "rmse" in res_eval


async def _assert_cross_validate_tool(cv_tool) -> None:
    """cross_validate: happy path returns per-fold RMSE; invalid json rejects."""
    res_cv = await cv_tool.fn(
        model_class="Ridge",
        dataset_name="iris",
        n_folds=3,
        hyperparameters_json="{}",
        ctx=None,
    )
    assert "rmse_per_fold" in res_cv

    res_cv_err = await cv_tool.fn(
        model_class="Ridge",
        dataset_name="iris",
        hyperparameters_json="{invalid",
        ctx=None,
    )
    assert "error" in res_cv_err


async def test_mcp_model_training_tools(require_engine, require_sklearn):
    """Verify FastMCP fit, predict, evaluate, and cross-validate tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()

    # Load iris dataset first
    await _tool(tools, "load_dataset").fn(name="iris", ctx=None)

    # 1. Test fit_model tool
    res_fit = await _assert_fit_model_tool(_tool(tools, "fit_model"))
    model_id = res_fit["model_id"]

    # 2. Test predict tool
    await _assert_predict_tool(_tool(tools, "predict"), model_id, res_fit["feature_names"])

    # 3. Test evaluate_model tool
    await _assert_evaluate_model_tool(_tool(tools, "evaluate_model"), model_id)

    # 4. Test cross_validate tool
    await _assert_cross_validate_tool(_tool(tools, "cross_validate"))


@pytest.mark.asyncio
async def test_mcp_model_evolution_tools(require_engine, require_sklearn):
    """Verify FastMCP evolve_model_class, rank_models, and get_pareto_frontier tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()

    # Clear global pareto models store for clean execution
    from data_science_mcp.mcp_server import _pareto_models

    _pareto_models.clear()

    evolve_tool = next(t for t in tools if t.name == "evolve_model_class")
    rank_tool = next(t for t in tools if t.name == "rank_models")
    pareto_tool = next(t for t in tools if t.name == "get_pareto_frontier")

    # Fit a model to populate MLEngine models cache
    fit_tool = next(t for t in tools if t.name == "fit_model")
    await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )

    # Test ranking
    res_rank = await rank_tool.fn(ctx=None)
    assert "ranked_models" in res_rank
    assert len(res_rank["ranked_models"]) > 0

    # Submit model 1: Linear (perf=0.7, complexity=0.1)
    res_e1 = await evolve_tool.fn(
        model_class="LinearRegression", base_performance=0.7, complexity=0.1, ctx=None
    )
    assert "pareto_frontier" in res_e1
    assert "LinearRegression" in res_e1["pareto_frontier"]

    # Submit model 2: DecisionTree (perf=0.6, complexity=0.2) -> Dominated by e1
    res_e2 = await evolve_tool.fn(
        model_class="DecisionTree", base_performance=0.6, complexity=0.2, ctx=None
    )
    assert "DecisionTree" not in res_e2["pareto_frontier"]

    # Submit model 3: RandomForest (perf=0.9, complexity=0.5) -> Not dominated
    res_e3 = await evolve_tool.fn(
        model_class="RandomForest", base_performance=0.9, complexity=0.5, ctx=None
    )
    assert "RandomForest" in res_e3["pareto_frontier"]
    assert "LinearRegression" in res_e3["pareto_frontier"]

    # Test get_pareto_frontier
    res_pf = await pareto_tool.fn(ctx=None)
    assert "RandomForest" in res_pf["pareto_frontier"]
    assert "LinearRegression" in res_pf["pareto_frontier"]


@pytest.mark.asyncio
async def _assert_generate_interpretability_tests_tool(gen_tool, model_id: str) -> None:
    res_gen = await gen_tool.fn(model_id=model_id, ctx=None)
    assert "tests" in res_gen
    assert len(res_gen["tests"]) == 6

    res_gen_err = await gen_tool.fn(model_id="non_existent", ctx=None)
    assert "error" in res_gen_err


async def _assert_grade_response_tool(grade_tool) -> None:
    """Exact, float-proximity, substring, and failed-match grading."""
    res_g1 = await grade_tool.fn(
        test_id="test_0",
        response="LinearRegression",
        expected="LinearRegression",
        ctx=None,
    )
    assert res_g1["passed"] is True

    res_g2 = await grade_tool.fn(
        test_id="test_1", response=" 1.23456 ", expected="1.2345", ctx=None
    )
    assert res_g2["passed"] is True

    res_g3 = await grade_tool.fn(
        test_id="test_2",
        response="The feature is sepal length (cm)",
        expected="sepal length (cm)",
        ctx=None,
    )
    assert res_g3["passed"] is True

    res_g4 = await grade_tool.fn(
        test_id="test_3", response="wrong", expected="correct", ctx=None
    )
    assert res_g4["passed"] is False


def _build_interpretability_answer_sets(detailed_results: list[dict]) -> tuple[dict, dict]:
    """Exact reference answers, plus a fallback set exercising substring (attribution)
    and float-proximity (numeric) grading instead of an exact match."""
    answers = {}
    answers_fallback = {}
    for item in detailed_results:
        tid = item["test_id"]
        exp = item["expected"]
        answers[tid] = exp
        if "att" in tid:
            answers_fallback[tid] = f"The feature is {exp}"
        else:
            try:
                val = float(exp)
                answers_fallback[tid] = f" {val + 0.0002} "
            except ValueError:
                answers_fallback[tid] = exp
    return answers, answers_fallback


async def _assert_run_interpretability_suite_scoring(run_suite_tool, model_id: str) -> None:
    """A perfect score with exact reference answers, and again with the
    substring/float-proximity fallback set."""
    res_empty = await run_suite_tool.fn(model_id=model_id, answers_json="{}", ctx=None)
    assert "detailed_results" in res_empty

    answers, answers_fallback = _build_interpretability_answer_sets(
        res_empty["detailed_results"]
    )
    res_suite = await run_suite_tool.fn(
        model_id=model_id, answers_json=json.dumps(answers), ctx=None
    )
    assert res_suite["overall_score"] == 1.0

    res_suite_fb = await run_suite_tool.fn(
        model_id=model_id, answers_json=json.dumps(answers_fallback), ctx=None
    )
    assert res_suite_fb["overall_score"] == 1.0


async def _assert_run_interpretability_suite_nonlinear_model(fit_tool, run_suite_tool) -> None:
    """A DecisionTree has no linear coefficients — exercises the 'unknown'-attribution
    reference path."""
    res_dt = await fit_tool.fn(
        model_class="DecisionTree",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )
    res_suite_dt = await run_suite_tool.fn(
        model_id=res_dt["model_id"], answers_json="{}", ctx=None
    )
    assert "overall_score" in res_suite_dt


async def _assert_run_interpretability_suite_error_paths(run_suite_tool, model_id: str) -> None:
    res_suite_err = await run_suite_tool.fn(
        model_id=model_id, answers_json="{invalid", ctx=None
    )
    assert "error" in res_suite_err

    res_suite_err2 = await run_suite_tool.fn(
        model_id="non_existent", answers_json="{}", ctx=None
    )
    assert "error" in res_suite_err2


async def test_mcp_interpretability_tools(require_engine, require_sklearn):
    """Verify FastMCP interpretability tests suite, grading, and run tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()

    # Load dataset & fit a model to test interpretability suite
    fit_tool = next(t for t in tools if t.name == "fit_model")
    res_fit = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )
    model_id = res_fit["model_id"]

    # 1. Test generate_interpretability_tests tool
    gen_tool = next(t for t in tools if t.name == "generate_interpretability_tests")
    await _assert_generate_interpretability_tests_tool(gen_tool, model_id)

    # 2. Test grade_response tool (float, exact, substring matches)
    grade_tool = next(t for t in tools if t.name == "grade_response")
    await _assert_grade_response_tool(grade_tool)

    # 3. Test run_interpretability_suite tool
    run_suite_tool = next(t for t in tools if t.name == "run_interpretability_suite")
    await _assert_run_interpretability_suite_scoring(run_suite_tool, model_id)

    # 4. Test run_interpretability_suite with a non-linear model (no linear
    # coefficients) — exercises the 'unknown'-attribution reference path.
    await _assert_run_interpretability_suite_nonlinear_model(fit_tool, run_suite_tool)

    # 5. Test run_interpretability_suite's invalid-json and non-existent-model errors
    await _assert_run_interpretability_suite_error_paths(run_suite_tool, model_id)


@pytest.mark.asyncio
class _MockContext:
    def __init__(self):
        self.logged: list[str] = []

    async def info(self, msg: str):
        self.logged.append(msg)


async def _assert_tool_logs(
    tools, name: str, kwargs: dict[str, Any], expected_substring: str, ctx: _MockContext
) -> Any:
    """Call MCP tool ``name`` with ``kwargs`` + ``ctx``; assert ``expected_substring``
    appears in a logged message; returns the tool's result."""
    tool = next(t for t in tools if t.name == name)
    result = await tool.fn(**kwargs, ctx=ctx)
    assert any(expected_substring in m for m in ctx.logged)
    return result


async def test_mcp_context_logging(require_engine, require_sklearn):
    """Verify that passing a mock context invokes info/logging methods in all tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    ctx = _MockContext()

    await _assert_tool_logs(tools, "load_dataset", {"name": "iris"}, "Loading configured dataset", ctx)
    await _assert_tool_logs(tools, "describe_dataset", {"name": "iris"}, "Describing statistics", ctx)
    await _assert_tool_logs(
        tools,
        "split_dataset",
        {"name": "iris", "test_size": 0.2, "validation_size": 0.0, "random_seed": 42},
        "Splitting configured dataset",
        ctx,
    )

    res_fit = await _assert_tool_logs(
        tools,
        "fit_model",
        {
            "model_class": "LinearRegression",
            "dataset_name": "iris",
            "hyperparameters_json": "{}",
            "test_size": 0.2,
        },
        "Fitting model class",
        ctx,
    )
    model_id = res_fit["model_id"]
    inputs_json = json.dumps([{f: 0.1 for f in res_fit["feature_names"]}])

    await _assert_tool_logs(
        tools,
        "predict",
        {"model_id": model_id, "inputs_json": inputs_json},
        "Generating predictions",
        ctx,
    )
    await _assert_tool_logs(
        tools,
        "evaluate_model",
        {"model_id": model_id, "dataset_name": "iris", "split": "test"},
        "Evaluating model",
        ctx,
    )
    await _assert_tool_logs(
        tools,
        "cross_validate",
        {
            "model_class": "Ridge",
            "dataset_name": "iris",
            "n_folds": 3,
            "hyperparameters_json": "{}",
        },
        "Running",
        ctx,
    )
    await _assert_tool_logs(
        tools,
        "evolve_model_class",
        {"model_class": "LinearRegression", "base_performance": 0.7, "complexity": 0.1},
        "Submitting",
        ctx,
    )
    await _assert_tool_logs(tools, "rank_models", {}, "Ranking models", ctx)
    await _assert_tool_logs(tools, "get_pareto_frontier", {}, "Retrieving", ctx)
    await _assert_tool_logs(
        tools,
        "generate_interpretability_tests",
        {"model_id": model_id},
        "Generating interpretability tests",
        ctx,
    )
    await _assert_tool_logs(
        tools,
        "grade_response",
        {"test_id": "t_0", "response": "0.1", "expected": "0.1"},
        "Grading response",
        ctx,
    )
    await _assert_tool_logs(
        tools,
        "run_interpretability_suite",
        {"model_id": model_id, "answers_json": "{}"},
        "Running full interpretability suite",
        ctx,
    )


@pytest.mark.asyncio
async def test_mcp_prompts():
    """Verify prompts defined in FastMCP."""
    mcp, _, _, _ = get_mcp_instance()
    prompts = await mcp.list_prompts()

    # Check model_evolution_workflow
    prompt_ev = next(p for p in prompts if p.name == "model_evolution_workflow")
    res_ev = prompt_ev.fn(dataset="iris")
    assert "iris" in res_ev
    assert "fit_model" in res_ev

    # Check interpretability_audit
    prompt_ia = next(p for p in prompts if p.name == "interpretability_audit")
    res_ia = prompt_ia.fn(model_name="LinearRegression")
    assert "LinearRegression" in res_ia
    assert "Feature Attribution" in res_ia


def test_mcp_server_entrypoint():
    """Verify that calling the mcp_server entrypoint initializes and runs the FastMCP server."""
    from data_science_mcp.mcp_server import mcp_server

    with patch("fastmcp.FastMCP.run") as mock_run:
        mcp_server()
        mock_run.assert_called_once()

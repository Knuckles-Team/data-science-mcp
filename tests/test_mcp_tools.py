"""
Tests for programmatically verifying all registered MCP tools and MLEngine functions.
"""

import sys
import os

# Set dummy sys.argv before importing anything to prevent create_mcp_server parsing issues
sys.argv = ["mcp_server.py"]

import json
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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


def test_ml_engine_load_dataset():
    """Test loading built-in sklearn datasets and custom CSV files."""
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

    # Test load dataset using custom CSV
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as tmp:
        df = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feat2": [2.0, 4.0, 6.0, 8.0, 10.0],
                "label": [0.5, 1.5, 2.5, 3.5, 4.5],
            }
        )
        df.to_csv(tmp.name, index=False)
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


def test_ml_engine_fit_predict_evaluate():
    """Test model fitting, predictions, and evaluation workflows on real scikit-learn models."""
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


def test_ml_engine_cross_validate():
    """Test cross-validation functionality across multiple estimators."""
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


def test_ml_engine_resolve_model():
    """Verify sklearn estimator resolution, fully-qualified name resolution, and defaults."""
    # Test shortcut mapping
    model_lr = MLEngine._resolve_model("linear_regression", {})
    assert model_lr.__class__.__name__ == "LinearRegression"

    # Test fully qualified name
    model_ridge = MLEngine._resolve_model("sklearn.linear_model.Ridge", {"alpha": 2.0})
    assert model_ridge.__class__.__name__ == "Ridge"
    assert model_ridge.alpha == 2.0

    # Test unknown model class (should fallback to LinearRegression)
    model_unknown = MLEngine._resolve_model("UnknownEstimator", {})
    assert model_unknown.__class__.__name__ == "LinearRegression"


def test_ml_engine_split_dataset():
    """Verify train/test/validation splitting functionality."""
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
    """Verify MLEngine exception and import error branches."""
    engine = MLEngine()

    # 1. load_dataset exceptions & import errors
    with patch(
        "sklearn.datasets.load_iris",
        side_effect=ImportError("mocked load_iris import error"),
    ):
        res = engine.load_dataset("iris")
        assert "error" in res
        assert "Missing dependency" in res["error"]

    with patch(
        "sklearn.datasets.load_iris",
        side_effect=ValueError("mocked load_iris value error"),
    ):
        res = engine.load_dataset("iris")
        assert "error" in res
        assert "mocked load_iris value error" in res["error"]

    with patch(
        "pandas.read_csv", side_effect=ImportError("mocked pandas import error")
    ):
        res = engine.load_dataset("nonexistent.csv")
        assert "error" in res
        assert "Missing dependency" in res["error"]

    # 2. Fit exceptions & import errors
    engine.load_dataset("iris")
    with patch(
        "sklearn.model_selection.train_test_split",
        side_effect=ImportError("mocked train_test_split import error"),
    ):
        res = engine.fit("LinearRegression", "iris")
        assert "error" in res
        assert "Missing dependency" in res["error"]

    with patch(
        "sklearn.model_selection.train_test_split",
        side_effect=ValueError("mocked train_test_split value error"),
    ):
        res = engine.fit("LinearRegression", "iris")
        assert "error" in res
        assert "mocked train_test_split value error" in res["error"]

    # 3. Cross validate exceptions & import errors
    with patch(
        "sklearn.model_selection.cross_val_score",
        side_effect=ImportError("mocked cross_val_score import error"),
    ):
        res = engine.cross_validate("LinearRegression", "iris")
        assert "error" in res
        assert "Missing dependency" in res["error"]

    with patch(
        "sklearn.model_selection.cross_val_score",
        side_effect=ValueError("mocked cross_val_score value error"),
    ):
        res = engine.cross_validate("LinearRegression", "iris")
        assert "error" in res
        assert "mocked cross_val_score value error" in res["error"]


@pytest.mark.asyncio
async def test_mcp_data_management_tools():
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


@pytest.mark.asyncio
async def test_mcp_model_training_tools():
    """Verify FastMCP fit, predict, evaluate, and cross-validate tools."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()

    # Load iris dataset first
    load_tool = next(t for t in tools if t.name == "load_dataset")
    await load_tool.fn(name="iris", ctx=None)

    # 1. Test fit_model tool
    fit_tool = next(t for t in tools if t.name == "fit_model")
    res_fit = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )
    assert "model_id" in res_fit
    model_id = res_fit["model_id"]

    # Fit with invalid hyperparameters_json (covers mcp_server.py line 81-82)
    res_fit_err = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{invalid",
        test_size=0.2,
        ctx=None,
    )
    assert "error" in res_fit_err

    # 2. Test predict tool
    predict_tool = next(t for t in tools if t.name == "predict")
    feature_names = res_fit["feature_names"]
    inputs_json = json.dumps([{f: 0.1 for f in feature_names}])
    res_pred = await predict_tool.fn(
        model_id=model_id, inputs_json=inputs_json, ctx=None
    )
    assert "predictions" in res_pred
    assert len(res_pred["predictions"]) == 1

    # Predict with invalid json
    res_pred_err = await predict_tool.fn(
        model_id=model_id, inputs_json="{invalid", ctx=None
    )
    assert "error" in res_pred_err

    # Predict with nonexistent model ID (covers mcp_server.py line 114-115)
    res_pred_err2 = await predict_tool.fn(
        model_id="nonexistent_id", inputs_json=inputs_json, ctx=None
    )
    assert "error" in res_pred_err2

    # 3. Test evaluate_model tool
    evaluate_tool = next(t for t in tools if t.name == "evaluate_model")
    res_eval = await evaluate_tool.fn(
        model_id=model_id, dataset_name="iris", split="test", ctx=None
    )
    assert "rmse" in res_eval

    # 4. Test cross_validate tool
    cv_tool = next(t for t in tools if t.name == "cross_validate")
    res_cv = await cv_tool.fn(
        model_class="Ridge",
        dataset_name="iris",
        n_folds=3,
        hyperparameters_json="{}",
        ctx=None,
    )
    assert "rmse_per_fold" in res_cv

    # CV with invalid json
    res_cv_err = await cv_tool.fn(
        model_class="Ridge",
        dataset_name="iris",
        hyperparameters_json="{invalid",
        ctx=None,
    )
    assert "error" in res_cv_err


@pytest.mark.asyncio
async def test_mcp_model_evolution_tools():
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
async def test_mcp_interpretability_tools():
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
    res_gen = await gen_tool.fn(model_id=model_id, ctx=None)
    assert "tests" in res_gen
    assert len(res_gen["tests"]) == 6

    # Test non-existent model ID
    res_gen_err = await gen_tool.fn(model_id="non_existent", ctx=None)
    assert "error" in res_gen_err

    # 2. Test grade_response tool (float, exact, substring matches)
    grade_tool = next(t for t in tools if t.name == "grade_response")

    # Exact match
    res_g1 = await grade_tool.fn(
        test_id="test_0",
        response="LinearRegression",
        expected="LinearRegression",
        ctx=None,
    )
    assert res_g1["passed"] is True

    # Float proximity match
    res_g2 = await grade_tool.fn(
        test_id="test_1", response=" 1.23456 ", expected="1.2345", ctx=None
    )
    assert res_g2["passed"] is True

    # Substring match
    res_g3 = await grade_tool.fn(
        test_id="test_2",
        response="The feature is sepal length (cm)",
        expected="sepal length (cm)",
        ctx=None,
    )
    assert res_g3["passed"] is True

    # Failed match
    res_g4 = await grade_tool.fn(
        test_id="test_3", response="wrong", expected="correct", ctx=None
    )
    assert res_g4["passed"] is False

    # 3. Test run_interpretability_suite tool
    run_suite_tool = next(t for t in tools if t.name == "run_interpretability_suite")

    # Generate reference answers for test ids by first running with empty answers to get expected values
    res_empty = await run_suite_tool.fn(model_id=model_id, answers_json="{}", ctx=None)
    assert "detailed_results" in res_empty

    answers = {}
    answers_fallback = {}
    for item in res_empty["detailed_results"]:
        tid = item["test_id"]
        exp = item["expected"]
        # Standard correct answer
        answers[tid] = exp
        # Fallback answers: substring for attribution, float proximity for numbers
        if "att" in tid:
            answers_fallback[tid] = f"The feature is {exp}"
        else:
            try:
                val = float(exp)
                answers_fallback[tid] = f" {val + 0.0002} "
            except ValueError:
                answers_fallback[tid] = exp

    answers_json = json.dumps(answers)
    res_suite = await run_suite_tool.fn(
        model_id=model_id, answers_json=answers_json, ctx=None
    )
    assert res_suite["overall_score"] == 1.0

    res_suite_fb = await run_suite_tool.fn(
        model_id=model_id, answers_json=json.dumps(answers_fallback), ctx=None
    )
    assert res_suite_fb["overall_score"] == 1.0

    # 4. Test run_interpretability_suite with a model that has feature_importances_ instead of coef_
    # e.g., DecisionTreeRegressor
    res_dt = await fit_tool.fn(
        model_class="sklearn.tree.DecisionTreeRegressor",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=None,
    )
    dt_model_id = res_dt["model_id"]
    res_suite_dt = await run_suite_tool.fn(
        model_id=dt_model_id, answers_json="{}", ctx=None
    )
    assert "overall_score" in res_suite_dt

    # Test run interpretability suite with invalid json
    res_suite_err = await run_suite_tool.fn(
        model_id=model_id, answers_json="{invalid", ctx=None
    )
    assert "error" in res_suite_err

    # Test run interpretability suite with non-existent model
    res_suite_err2 = await run_suite_tool.fn(
        model_id="non_existent", answers_json="{}", ctx=None
    )
    assert "error" in res_suite_err2


@pytest.mark.asyncio
async def test_mcp_context_logging():
    """Verify that passing a mock context invokes info/logging methods in all tools."""

    class MockContext:
        def __init__(self):
            self.logged = []

        async def info(self, msg: str):
            self.logged.append(msg)

    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    ctx = MockContext()

    # Load dataset tool with context
    load_tool = next(t for t in tools if t.name == "load_dataset")
    await load_tool.fn(name="iris", ctx=ctx)
    assert any("Loading dataset" in m for m in ctx.logged)

    # Describe dataset tool with context
    describe_tool = next(t for t in tools if t.name == "describe_dataset")
    await describe_tool.fn(name="iris", ctx=ctx)
    assert any("Describing statistics" in m for m in ctx.logged)

    # Split dataset tool with context
    split_tool = next(t for t in tools if t.name == "split_dataset")
    await split_tool.fn(
        name="iris", test_size=0.2, validation_size=0.0, random_seed=42, ctx=ctx
    )
    assert any("Splitting dataset" in m for m in ctx.logged)

    # Fit model tool with context
    fit_tool = next(t for t in tools if t.name == "fit_model")
    res_fit = await fit_tool.fn(
        model_class="LinearRegression",
        dataset_name="iris",
        hyperparameters_json="{}",
        test_size=0.2,
        ctx=ctx,
    )
    assert any("Fitting model class" in m for m in ctx.logged)
    model_id = res_fit["model_id"]

    # Predict tool with context
    predict_tool = next(t for t in tools if t.name == "predict")
    feature_names = res_fit["feature_names"]
    inputs_json = json.dumps([{f: 0.1 for f in feature_names}])
    await predict_tool.fn(model_id=model_id, inputs_json=inputs_json, ctx=ctx)
    assert any("Generating predictions" in m for m in ctx.logged)

    # Evaluate model tool with context
    evaluate_tool = next(t for t in tools if t.name == "evaluate_model")
    await evaluate_tool.fn(
        model_id=model_id, dataset_name="iris", split="test", ctx=ctx
    )
    assert any("Evaluating model" in m for m in ctx.logged)

    # Cross validate tool with context
    cv_tool = next(t for t in tools if t.name == "cross_validate")
    await cv_tool.fn(
        model_class="Ridge",
        dataset_name="iris",
        n_folds=3,
        hyperparameters_json="{}",
        ctx=ctx,
    )
    assert any("Running" in m for m in ctx.logged)

    # Evolve model tool with context
    evolve_tool = next(t for t in tools if t.name == "evolve_model_class")
    await evolve_tool.fn(
        model_class="LinearRegression", base_performance=0.7, complexity=0.1, ctx=ctx
    )
    assert any("Submitting" in m for m in ctx.logged)

    # Rank models tool with context
    rank_tool = next(t for t in tools if t.name == "rank_models")
    await rank_tool.fn(ctx=ctx)
    assert any("Ranking models" in m for m in ctx.logged)

    # Get pareto frontier tool with context
    pareto_tool = next(t for t in tools if t.name == "get_pareto_frontier")
    await pareto_tool.fn(ctx=ctx)
    assert any("Retrieving" in m for m in ctx.logged)

    # Generate interpretability tests tool with context
    gen_tool = next(t for t in tools if t.name == "generate_interpretability_tests")
    await gen_tool.fn(model_id=model_id, ctx=ctx)
    assert any("Generating interpretability tests" in m for m in ctx.logged)

    # Grade response tool with context
    grade_tool = next(t for t in tools if t.name == "grade_response")
    await grade_tool.fn(test_id="t_0", response="0.1", expected="0.1", ctx=ctx)
    assert any("Grading response" in m for m in ctx.logged)

    # Run interpretability suite tool with context
    run_suite_tool = next(t for t in tools if t.name == "run_interpretability_suite")
    await run_suite_tool.fn(model_id=model_id, answers_json="{}", ctx=ctx)
    assert any("Running full interpretability suite" in m for m in ctx.logged)


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

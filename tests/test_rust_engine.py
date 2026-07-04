"""Integration tests for the epistemic-graph Rust compute path in MLEngine.

These exercise fit / predict / evaluate / describe / split / cross_validate
routed to the Rust engine (CONCEPT:EG-KG.compute.rust-native-training-loss). The engine is started by the
session-scoped `epistemic_graph_engine` fixture (see conftest.py); the data
fixtures depend on `require_engine`, which skips cleanly if no engine could be
started/reached. Engine reachability is resolved at fixture-setup time (not at
import/collection), so the session engine is always up first.
"""

import numpy as np
import pytest

from data_science_mcp.ml_engine import MLEngine


@pytest.fixture
def engine_with_data(require_engine):
    engine = MLEngine()
    engine._datasets.clear()
    engine._models.clear()
    # Exact linear data: y = 3*x0 + 2*x1 + 5
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(60, 2))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 5.0
    engine._datasets["syn"] = {
        "X": X,
        "y": y,
        "feature_names": ["x0", "x1"],
        "target_name": "y",
    }
    yield engine
    engine._datasets.clear()
    engine._models.clear()


def test_fit_predict_evaluate_rust(engine_with_data):
    res = engine_with_data.fit("LinearRegression", "syn", test_size=0.25)
    assert res["backend"] == "rust"
    assert abs(res["metrics"]["r2_test"] - 1.0) < 1e-6
    assert res["metrics"]["rmse_test"] < 1e-6
    assert res["n_train"] == 45 and res["n_test"] == 15

    mid = res["model_id"]
    preds = engine_with_data.predict(
        mid, [{"x0": 1.0, "x1": 1.0}, {"x0": 0.0, "x1": 0.0}]
    )
    assert abs(preds[0] - 10.0) < 1e-6
    assert abs(preds[1] - 5.0) < 1e-6

    ev = engine_with_data.evaluate(mid, "syn", split="test")
    assert ev["rmse"] < 1e-6
    assert abs(ev["r2"] - 1.0) < 1e-6


def test_describe_rust(engine_with_data):
    desc = engine_with_data.describe_dataset("syn")
    assert desc["backend"] == "rust"
    assert set(desc["feature_stats"]) == {"x0", "x1"}
    assert np.isfinite(desc["target_stats"]["mean"])


def test_split_rust(engine_with_data):
    sp = engine_with_data.split_dataset("syn", test_size=0.2, validation_size=0.1)
    assert sp["backend"] == "rust"
    assert sp["total"] == 60
    assert "validation_size" in sp


def test_cross_validate_rust(engine_with_data):
    cv = engine_with_data.cross_validate("LinearRegression", "syn", n_folds=5)
    assert cv["backend"] == "rust"
    assert len(cv["rmse_per_fold"]) == 5
    assert cv["rmse_mean"] < 1e-6


@pytest.fixture
def engine_nonlinear(require_engine):
    """Nonlinear dataset; the engine's seeded-shuffle split keeps the test fold
    interpolatable for tree/SVR models."""
    engine = MLEngine()
    engine._datasets.clear()
    engine._models.clear()
    rng = np.random.default_rng(7)
    X = rng.uniform(-3, 3, size=(90, 1))
    y = X[:, 0] ** 2
    engine._datasets["quad"] = {
        "X": X,
        "y": y,
        "feature_names": ["x0"],
        "target_name": "y",
    }
    yield engine
    engine._datasets.clear()
    engine._models.clear()


@pytest.mark.parametrize(
    "model,params",
    [
        ("Ridge", {"alpha": 1e-6}),
        ("Lasso", {"alpha": 0.01}),
        ("ElasticNet", {"alpha": 0.01, "l1_ratio": 0.5}),
    ],
)
def test_linear_family_estimators_rust(engine_with_data, model, params):
    res = engine_with_data.fit(model, "syn", hyperparameters=params, test_size=0.25)
    assert res["backend"] == "rust_estimator"
    # Exact-linear data -> near-perfect fit for the linear family.
    assert res["metrics"]["r2_test"] > 0.99
    preds = engine_with_data.predict(res["model_id"], [{"x0": 1.0, "x1": 1.0}])
    assert abs(preds[0] - 10.0) < 0.1


@pytest.mark.parametrize(
    "model,params",
    [
        ("RandomForest", {"n_estimators": 40, "random_state": 1}),
        ("GradientBoosting", {"n_estimators": 60, "learning_rate": 0.1}),
        (
            "SVR",
            {
                "C": 10.0,
                "epsilon": 0.05,
                "gamma": 0.5,
                "kernel": "rbf",
                "max_iter": 3000,
            },
        ),
    ],
)
def test_nonlinear_estimators_rust(engine_nonlinear, model, params):
    res = engine_nonlinear.fit(model, "quad", hyperparameters=params, test_size=0.25)
    assert res["backend"] == "rust_estimator"
    assert res["metrics"]["r2_test"] > 0.7
    mid = res["model_id"]
    ev = engine_nonlinear.evaluate(mid, "quad", split="test")
    assert ev["r2"] > 0.7


def test_ridge_cross_validate_rust(engine_with_data):
    cv = engine_with_data.cross_validate(
        "Ridge", "syn", n_folds=4, hyperparameters={"alpha": 1e-6}
    )
    assert cv["backend"] == "rust_estimator"
    assert len(cv["rmse_per_fold"]) == 4


def test_supported_model_flags():
    for m in (
        "Ridge",
        "Lasso",
        "ElasticNet",
        "RandomForest",
        "GradientBoosting",
        "AdaBoost",
        "SVR",
        "linear_regression",
    ):
        assert MLEngine._is_supported(m), m
    assert not MLEngine._is_supported("SomeUnknownModel")

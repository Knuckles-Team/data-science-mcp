"""CONCEPT:AU-KG.mining.dsm-forecast-delegation — the Phase-6 deep-mining delegation target.

Exercises the training functions in ``data_science_mcp.training.deep_delegate``
and the ``deep_train_predict`` MCP tool they back — the delegate agent-utilities'
``graph_mine_deep`` reaches over MCP for the deep-learning / heavy-Python mining
family (LSTM forecasting, MLP/deep classification, autoencoder anomaly/embedding,
a histogram-gradient-boosting classifier standing in for XGBoost). Skipped when
the ``[training]`` extra (torch) is absent, matching ``test_trading_lstm.py``.
"""

import json

import numpy as np
import pytest

try:
    import torch  # noqa: F401

    from data_science_mcp.training.deep_delegate import DEEP_ALGOS

    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

pytestmark = pytest.mark.skipif(
    not HAS_TRAINING, reason="Training dependencies (torch) missing"
)


def _separated_classes(n: int = 10) -> tuple[list, list]:
    x = [[0.0, 0.0]] * n + [[10.0, 10.0]] * n
    y = [0] * n + [1] * n
    return x, y


def test_mlp_classify_separates_two_clusters():
    x, y = _separated_classes()
    out = DEEP_ALGOS["mlp_classify"](x=x, y=y, epochs=80, seed=0)
    assert out["algo"] == "mlp_classify"
    assert out["classes"] == [0, 1]
    labels = [row["label"] for row in out["rows"]]
    assert labels[:10] == [0] * 10
    assert labels[10:] == [1] * 10


def test_mlp_classify_predicts_on_x_predict():
    x, y = _separated_classes()
    out = DEEP_ALGOS["mlp_classify"](
        x=x, y=y, epochs=80, seed=0, x_predict=[[0.0, 0.0], [10.0, 10.0]]
    )
    assert len(out["rows"]) == 2
    assert out["rows"][0]["label"] == 0
    assert out["rows"][1]["label"] == 1


def test_histgbm_classify_matches_mlp_shape():
    x, y = _separated_classes()
    out = DEEP_ALGOS["histgbm_classify"](x=x, y=y, seed=0)
    assert out["algo"] == "histgbm_classify"
    assert out["classes"] == [0, 1]
    assert len(out["rows"]) == len(x)


def test_autoencoder_anomaly_flags_the_outlier():
    x = [[0.0, 0.0]] * 10 + [[50.0, 50.0]]
    out = DEEP_ALGOS["autoencoder_anomaly"](x=x, epochs=80, seed=0)
    assert out["algo"] == "autoencoder_anomaly"
    assert out["rows"][-1]["is_anomaly"] is True
    assert out["rows"][-1]["anomaly_score"] > out["rows"][0]["anomaly_score"]


def test_autoencoder_embed_returns_bottleneck_vectors():
    x, _ = _separated_classes()
    out = DEEP_ALGOS["autoencoder_embed"](x=x, epochs=60, bottleneck=2, seed=0)
    assert out["dim"] == 2
    assert len(out["rows"]) == len(x)
    assert len(out["rows"][0]["vector"]) == 2


def test_lstm_forecast_extrapolates_a_linear_trend():
    values = [float(i) for i in range(30)]
    out = DEEP_ALGOS["lstm_forecast"](values=values, horizon=3, epochs=150, seed=0)
    assert out["algo"] == "lstm_forecast"
    assert out["horizon"] == 3
    assert len(out["forecast"]) == 3
    # a clean upward linear trend should keep extrapolating upward
    assert out["forecast"][0] > values[-2]
    assert all(
        lo <= f <= up for f, lo, up in zip(out["forecast"], out["lower"], out["upper"])
    )


def test_deep_algos_are_json_safe():
    """Every DEEP_ALGOS result must round-trip through json.dumps (the MCP tool
    wraps it directly), so no numpy scalars/arrays may leak into the output."""
    x, y = _separated_classes(n=3)
    for name, fn in DEEP_ALGOS.items():
        if name == "lstm_forecast":
            out = fn(values=[float(i) for i in range(10)], horizon=2, epochs=20)
        elif name in ("mlp_classify", "histgbm_classify"):
            out = fn(x=x, y=y, epochs=10) if name == "mlp_classify" else fn(x=x, y=y)
        else:
            out = (
                fn(x=x, epochs=20)
                if name == "autoencoder_anomaly"
                else fn(x=x, epochs=20)
            )
        json.dumps(out)  # raises TypeError if anything non-JSON-safe leaked through


def test_seeded_runs_are_reproducible():
    x, y = _separated_classes(n=5)
    out1 = DEEP_ALGOS["mlp_classify"](x=x, y=y, epochs=30, seed=7)
    out2 = DEEP_ALGOS["mlp_classify"](x=x, y=y, epochs=30, seed=7)
    assert out1["rows"] == out2["rows"]


def test_numpy_seed_is_isolated_per_call():
    """A stray np.random call elsewhere shouldn't change a seeded result."""
    x, y = _separated_classes(n=5)
    np.random.seed(123)
    np.random.rand(5)
    out = DEEP_ALGOS["mlp_classify"](x=x, y=y, epochs=30, seed=7)
    np.random.seed(999)
    np.random.rand(5)
    out2 = DEEP_ALGOS["mlp_classify"](x=x, y=y, epochs=30, seed=7)
    assert out["rows"] == out2["rows"]

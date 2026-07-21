"""MCP tool for the AU-KG "deep mining" delegation family (CONCEPT:AU-KG.mining.dsm-forecast-delegation).

agent-utilities' engine core is pure-Rust and never imports torch — the
deep-learning / heavy-Python family the data-mining plan marks out of scope for
the engine (LSTM/RNN sequence forecasting, MLP/deep classifiers, autoencoders, a
histogram-gradient-boosting classifier standing in for XGBoost) is delegated
HERE, over MCP, from agent-utilities' ``graph_mine_deep`` tool
(``agent_utilities/mcp/tools/engine_surface_tools.py``, via the fleet
``call_tool_once`` connector). This module is the single entry point that
delegation reaches: one action-routed tool, ``deep_train_predict``, dispatching
into :mod:`data_science_mcp.training.deep_delegate` (the actual torch/sklearn
compute — thin here, no logic duplicated).

Degrades cleanly when the ``[training]`` extra (torch) is absent: the tool
returns a structured ``{"available": False, ...}`` payload instead of raising,
so a caller without a GPU-capable install gets a clear signal, not a crash.
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP
from pydantic import Field


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def register_deep_delegate_tools(mcp: FastMCP) -> None:
    """Register ``deep_train_predict`` (tag ``deep-delegate``)."""

    @mcp.tool(tags={"deep-delegate"})
    def deep_train_predict(
        algo: str = Field(
            default="mlp_classify",
            description=(
                "Which delegated deep/heavy model to train+run: 'mlp_classify' "
                "(MLP deep classifier), 'autoencoder_anomaly' (reconstruction-error "
                "outlier detection), 'autoencoder_embed' (neural bottleneck "
                "embedding of feature rows — NOT text/word embeddings, those stay "
                "on the remote vLLM embedder), 'lstm_forecast' (LSTM sequence "
                "forecaster — the delegated Prophet/LSTM family; Prophet itself "
                "needs an unvendored Stan toolchain so LSTM is what actually runs), "
                "'histgbm_classify' (histogram gradient-boosting classifier — the "
                "documented xgboost substitute; no separate xgboost dependency is "
                "vendored)."
            ),
        ),
        x_json: str = Field(
            default="[]",
            description="JSON 2-D array of feature rows (mlp_classify / "
            "autoencoder_anomaly / autoencoder_embed / histgbm_classify).",
        ),
        y_json: str = Field(
            default="[]",
            description="JSON 1-D array of integer class labels (mlp_classify / histgbm_classify).",
        ),
        values_json: str = Field(
            default="[]",
            description="JSON 1-D numeric series (lstm_forecast).",
        ),
        x_predict_json: str = Field(
            default="",
            description="Optional JSON 2-D array to predict on instead of x_json "
            "(mlp_classify / histgbm_classify; default is transductive fit+predict on x_json).",
        ),
        params_json: str = Field(
            default="{}",
            description="JSON object of algo-specific kwargs, e.g. "
            '{"epochs":100,"lr":0.05,"hidden":32,"seed":0} (mlp_classify); '
            '{"epochs":150,"bottleneck":2,"threshold":null} (autoencoder_anomaly/embed); '
            '{"horizon":5,"lookback":5,"epochs":200,"hidden":16} (lstm_forecast); '
            '{"seed":0} (histgbm_classify).',
        ),
    ) -> str:
        """Fit + run one delegated deep/heavy model and return predictions/model as JSON.

        This is the single generic "train+predict" entry data-science-mcp lacked
        for the AU-KG mining plan's out-of-scope family (CONCEPT:AU-KG.mining.dsm-forecast-delegation) —
        one tool, five algos, dispatching into :mod:`data_science_mcp.training.deep_delegate`.
        Never raises on a missing torch install: returns ``{"available": false, ...}``.
        """
        algo = (algo or "mlp_classify").strip()
        if not _has_torch() and algo != "histgbm_classify":
            return json.dumps(
                {
                    "algo": algo,
                    "available": False,
                    "error": "torch not installed — install data-science-mcp[training]",
                }
            )
        try:
            from data_science_mcp.training.deep_delegate import DEEP_ALGOS
        except ImportError:
            return json.dumps({"algo": algo, "available": False, "error": "Operation failed"})

        fn = DEEP_ALGOS.get(algo)
        if fn is None:
            return json.dumps(
                {
                    "algo": algo,
                    "available": False,
                    "error": f"unknown algo {algo!r}; choose one of {sorted(DEEP_ALGOS)}",
                }
            )
        try:
            params: dict[str, Any] = json.loads(params_json) if params_json else {}
        except (TypeError, ValueError) as exc:
            return json.dumps({"algo": algo, "error": f"invalid params_json: {type(exc).__name__}"})
        if not isinstance(params, dict):
            return json.dumps(
                {"algo": algo, "error": "params_json must decode to an object"}
            )

        kwargs = dict(params)
        try:
            if algo == "lstm_forecast":
                kwargs["values"] = json.loads(values_json) if values_json else []
            else:
                kwargs["x"] = json.loads(x_json) if x_json else []
                if algo in ("mlp_classify", "histgbm_classify"):
                    kwargs["y"] = json.loads(y_json) if y_json else []
                if algo in ("mlp_classify", "histgbm_classify") and x_predict_json:
                    kwargs["x_predict"] = json.loads(x_predict_json)
        except (TypeError, ValueError) as exc:
            return json.dumps({"algo": algo, "error": f"invalid JSON input: {type(exc).__name__}"})

        try:
            result = fn(**kwargs)
        except TypeError as exc:
            return json.dumps({"algo": algo, "error": f"bad arguments: {type(exc).__name__}"})
        except Exception:  # noqa: BLE001 — surface training errors as data
            return json.dumps({"algo": algo, "error": "Operation failed"})
        return json.dumps({"algo": algo, "available": True, "result": result})

"""Generic torch-backed delegation targets for the AU-KG mining "deep" family.

CONCEPT:AU-KG.mining.dsm-forecast-delegation — agent-utilities' engine core stays
pure-Rust (no torch/GPU deps); the deep-learning / heavy-Python family the
data-mining plan marks out of scope for the engine (LSTM/RNN sequence
forecasting, MLP/deep classifiers, autoencoders, a histogram-gradient-boosting
classifier) is trained/served HERE, reached over MCP by
``agent_utilities.mcp.tools.engine_surface_tools.graph_mine_deep`` (the
``call_tool_once`` delegation adapter). This module is the thin, reusable
implementation those MCP tool bodies call into — small, seeded, CPU-friendly
models; NOT a general training framework.

Every function accepts plain Python (list-of-list / list) inputs and returns
plain JSON-safe dicts, so the MCP layer only has to marshal JSON, never tensors.
torch (+ scikit-learn for the histogram-GBM substitute) are ``[training]``-extra
deps — this module is imported lazily and raises a clear ``ImportError`` with an
install hint when they are absent, matching ``training/trading_lstm.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover - exercised only in training-free installs
    raise ImportError(
        "Deep-delegation dependencies missing. Install data-science-mcp[training]."
    ) from e


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class _MLP(nn.Module):
    """A minimal 2-hidden-layer classifier — the delegated 'MLP/deep classifier'."""

    def __init__(self, n_in: int, n_classes: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class _Autoencoder(nn.Module):
    """A minimal bottleneck autoencoder — reconstruction error → anomaly score,
    bottleneck activations → an embedding."""

    def __init__(self, n_in: int, bottleneck: int = 2, hidden: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Linear(hidden, n_in)
        )

    def forward(self, x: Any) -> tuple[Any, Any]:
        z = self.encoder(x)
        return self.decoder(z), z


class _LSTMForecaster(nn.Module):
    """A minimal single-layer LSTM regressor — the delegated 'LSTM sequence
    forecaster' (the Prophet/LSTM out-of-scope family; Prophet itself needs a
    Stan toolchain we do not vendor, so LSTM is the delegated forecaster here —
    ARIMA/Holt-Winters/STL remain the native classical default, see the plan's
    "out of scope" table)."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: Any) -> Any:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def mlp_classify(
    x: list[list[float]],
    y: list[int],
    epochs: int = 100,
    lr: float = 0.05,
    hidden: int = 32,
    seed: int = 0,
    x_predict: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Fit a small MLP on ``(x, y)`` and predict on ``x_predict`` (default: ``x`` itself,
    a transductive fit+predict matching the engine's own ``classify_fit``/``classify_predict``
    single-pass shape)."""
    _seed_everything(seed)
    x_arr = torch.tensor(np.asarray(x, dtype=np.float32))
    y_arr = np.asarray(y, dtype=np.int64)
    classes = sorted({int(v) for v in y_arr.tolist()})
    remap = {c: i for i, c in enumerate(classes)}
    y_idx = torch.tensor([remap[int(v)] for v in y_arr.tolist()], dtype=torch.long)

    model = _MLP(x_arr.shape[1], len(classes), hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        loss = criterion(model(x_arr), y_idx)
        loss.backward()
        optimizer.step()

    predict_arr = (
        x_arr
        if x_predict is None
        else torch.tensor(np.asarray(x_predict, dtype=np.float32))
    )
    model.eval()
    with torch.no_grad():
        logits = model(predict_arr)
        proba = torch.softmax(logits, dim=1).numpy()
    labels = [classes[i] for i in proba.argmax(axis=1).tolist()]
    rows = [
        {"id": i, "label": label, "proba": float(p.max())}
        for i, (label, p) in enumerate(zip(labels, proba, strict=True))
    ]
    return {"rows": rows, "classes": classes, "algo": "mlp_classify"}


def autoencoder_anomaly(
    x: list[list[float]],
    epochs: int = 150,
    lr: float = 0.02,
    bottleneck: int = 2,
    threshold: float | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a small autoencoder on ``x``; reconstruction error is the anomaly score
    (the delegated 'autoencoder anomaly' family — z-score/MAD/Isolation-Forest/LOF/
    One-Class-SVM stay native in the engine's ``graph_mine action=anomaly``)."""
    _seed_everything(seed)
    x_arr = torch.tensor(np.asarray(x, dtype=np.float32))
    model = _Autoencoder(x_arr.shape[1], bottleneck=bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction="none")
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        recon, _ = model(x_arr)
        loss = criterion(recon, x_arr).mean()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon, _ = model(x_arr)
        errors = criterion(recon, x_arr).mean(dim=1).numpy()
    thr = (
        float(threshold)
        if threshold is not None
        else float(errors.mean() + 2 * errors.std())
    )
    rows = [
        {"id": i, "anomaly_score": float(e), "is_anomaly": bool(e > thr)}
        for i, e in enumerate(errors.tolist())
    ]
    return {"rows": rows, "threshold": thr, "algo": "autoencoder_anomaly"}


def autoencoder_embed(
    x: list[list[float]],
    epochs: int = 150,
    lr: float = 0.02,
    bottleneck: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a small autoencoder on ``x`` and return the bottleneck activations as a
    learned (non-linear, neural) embedding — complements the engine's native linear
    ``reduce action=svd/umap/tsne``. NOT for text/word embeddings: those already
    have a home in the remote vLLM embedder (``core/embedding_utilities``); this is
    for embedding arbitrary numeric feature rows."""
    _seed_everything(seed)
    x_arr = torch.tensor(np.asarray(x, dtype=np.float32))
    model = _Autoencoder(x_arr.shape[1], bottleneck=bottleneck)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        recon, _ = model(x_arr)
        loss = criterion(recon, x_arr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, z = model(x_arr)
    rows = [{"id": i, "vector": vec} for i, vec in enumerate(z.numpy().tolist())]
    return {"rows": rows, "dim": int(bottleneck), "algo": "autoencoder_embed"}


def lstm_forecast(
    values: list[float],
    horizon: int = 5,
    lookback: int = 5,
    epochs: int = 200,
    lr: float = 0.01,
    hidden: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit a tiny single-layer LSTM on a 1-D series and forecast ``horizon`` steps
    ahead autoregressively. Confidence band = train-residual std (Gaussian
    approximation — a coarse but honest uncertainty estimate, not a full
    predictive-distribution model)."""
    _seed_everything(seed)
    series = np.asarray(values, dtype=np.float32)
    lb = min(max(1, lookback), max(1, len(series) - 1))
    mean, std = float(series.mean()), float(series.std() or 1.0)
    norm = (series - mean) / std

    xs, ys = [], []
    for i in range(len(norm) - lb):
        xs.append(norm[i : i + lb])
        ys.append(norm[i + lb])
    x_arr = torch.tensor(np.asarray(xs, dtype=np.float32)).unsqueeze(-1)
    y_arr = torch.tensor(np.asarray(ys, dtype=np.float32)).unsqueeze(-1)

    model = _LSTMForecaster(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        loss = criterion(model(x_arr), y_arr)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_resid = (model(x_arr) - y_arr).numpy().flatten()
    resid_std = float(train_resid.std() or 0.0) * std

    window = list(norm[-lb:])
    forecast: list[float] = []
    with torch.no_grad():
        for _ in range(max(1, horizon)):
            step_in = torch.tensor(np.asarray(window[-lb:], dtype=np.float32)).view(
                1, lb, 1
            )
            next_norm = float(model(step_in).item())
            window.append(next_norm)
            forecast.append(next_norm * std + mean)

    lower = [f - 1.96 * resid_std for f in forecast]
    upper = [f + 1.96 * resid_std for f in forecast]
    return {
        "forecast": forecast,
        "lower": lower,
        "upper": upper,
        "horizon": int(horizon),
        "algo": "lstm_forecast",
    }


def histgbm_classify(
    x: list[list[float]],
    y: list[int],
    seed: int = 0,
    x_predict: list[list[float]] | None = None,
) -> dict[str, Any]:
    """Histogram gradient-boosting classifier — the delegated substitute for
    'XGBoost': scikit-learn's ``HistGradientBoostingClassifier`` is already an
    installed ``[training]``-extra dependency and needs no separate C++ xgboost
    build; it is algorithmically the same family (histogram-binned gradient
    boosting) xgboost popularized. A real ``xgboost`` package is not vendored —
    documented cut, not a silent approximation (see CHANGELOG / phase report)."""
    from sklearn.ensemble import HistGradientBoostingClassifier  # PLC0415

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y)
    classes = sorted({v for v in y_arr.tolist()})
    model = HistGradientBoostingClassifier(random_state=seed)
    model.fit(x_arr, y_arr)

    predict_arr = (
        x_arr if x_predict is None else np.asarray(x_predict, dtype=np.float64)
    )
    proba = model.predict_proba(predict_arr)
    labels = model.predict(predict_arr)
    rows = [
        {"id": i, "label": label, "proba": float(p.max())}
        for i, (label, p) in enumerate(zip(labels.tolist(), proba, strict=True))
    ]
    return {"rows": rows, "classes": classes, "algo": "histgbm_classify"}


#: Dispatch table the MCP tool routes ``algo`` through — one place to see the
#: whole delegated-deep-mining family (CONCEPT:AU-KG.mining.dsm-forecast-delegation).
DEEP_ALGOS: dict[str, Callable[..., dict[str, Any]]] = {
    "mlp_classify": mlp_classify,
    "autoencoder_anomaly": autoencoder_anomaly,
    "autoencoder_embed": autoencoder_embed,
    "lstm_forecast": lstm_forecast,
    "histgbm_classify": histgbm_classify,
}

"""LSTM trading model + walk-forward training (CONCEPT:AU-KG.research.research-pipeline-runner).

Re-homed from ``agent_utilities.domains.finance.models`` /
``agent_utilities.domains.finance.evaluation`` so agent-utilities core stays
torch-free (the serving plane must not import torch — see
agent-utilities/AGENTS.md "Dependency discipline"). Heavy ML training belongs in
data-science-mcp, reached over MCP, not imported into core.

Contents:

* :class:`TradingLSTM` — an ``nn.Module`` LSTM that learns conditional-expectation
  functions for topological market regimes.
* :func:`prepare_sequences` — convert 2D tabular features into 3D LSTM sequences
  (pure numpy; lives here because its only consumers are the LSTM trainer).
* :func:`evaluate_trading_signal` — directional accuracy / Sharpe / max-drawdown
  metrics for a trading signal (numpy + scikit-learn).
* :func:`walk_forward_validation` — rolling-window LSTM training that simulates
  live deployment without lookahead bias (torch).

torch and scikit-learn are optional ``[training]``-extra dependencies; the import
is guarded so the module fails loud with a clear install hint when they are
absent.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
except ImportError as e:  # pragma: no cover - exercised only in training-free installs
    raise ImportError(
        "Trading-LSTM training dependencies missing. "
        "Install data-science-mcp[training]."
    ) from e


class TradingLSTM(nn.Module):
    """
    LSTM architecture for sequential market data processing.
    Learns conditional expectation functions for topological market regimes.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the final time step
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.sigmoid(out)


def prepare_sequences(
    features: np.ndarray, target: np.ndarray, lookback: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """
    Converts 2D tabular features into 3D sequences for LSTM consumption.
    Format: (batch_size, sequence_length, features)
    """
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : (i + lookback)])
        y.append(target[i + lookback])
    return np.array(X), np.array(y)


def evaluate_trading_signal(
    predictions: np.ndarray, actuals: np.ndarray, returns: np.ndarray
) -> tuple[float, float, float]:
    """

    CONCEPT:AU-KG.research.research-pipeline-runner
        Evaluates signal against directional accuracy, ROC-AUC, Sharpe, and Drawdown.
    """
    pred_direction = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(actuals, pred_direction)
    # Handle single class predictions (e.g. all 1s) for ROC-AUC gracefully
    try:
        roc_auc_score(actuals, predictions)
    except ValueError:
        pass

    signal = np.where(pred_direction == 1, 1, -1)

    # Trim returns to match the length of predictions (due to sequence lookback)
    returns_trimmed = returns[-len(predictions) :]
    strategy_returns = signal * returns_trimmed

    # Calculate Sharpe Ratio (annualized, assuming daily)
    std_dev = strategy_returns.std()
    if std_dev == 0:
        sharpe = 0.0
    else:
        sharpe = (strategy_returns.mean() / std_dev) * np.sqrt(252)

    # Calculate Max Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return accuracy, sharpe, max_drawdown


def walk_forward_validation(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    train_size: int,
    test_size: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates live deployment by rolling a training window forward through time.
    Strictly prevents lookahead bias.
    """
    all_predictions = []
    all_actuals: list[float] = []

    for start in range(0, len(features) - train_size - test_size, step):
        train_end = start + train_size
        test_end = train_end + test_size

        X_train = features[start:train_end]
        y_train = target[start:train_end]
        X_test = features[train_end:test_end]
        y_test = target[train_end:test_end]

        # Scale features
        scaler = StandardScaler()
        # reshape for scaling (flattening time series)
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        # rebuild 3D sequence
        X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train, lookback)
        X_test_seq, y_test_seq = prepare_sequences(X_test_scaled, y_test, lookback)

        if len(X_train_seq) == 0 or len(X_test_seq) == 0:
            continue

        model = TradingLSTM(input_size=X_train.shape[-1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()

        # Simple training loop for the walk-forward step
        model.train()
        for epoch in range(10):  # Minimal epochs for WF demonstration
            optimizer.zero_grad()
            preds = model(torch.FloatTensor(X_train_seq)).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            loss = criterion(preds, torch.FloatTensor(y_train_seq))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(torch.FloatTensor(X_test_seq)).squeeze()
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

        all_predictions.extend(preds.numpy())
        all_actuals.extend(y_test_seq)

    return np.array(all_predictions), np.array(all_actuals)

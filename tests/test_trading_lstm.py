"""CONCEPT:AU-KG.research.research-pipeline-runner — torch-backed TradingLSTM training (re-homed from agent-utilities core).

These assertions cover the LSTM architecture, sequence preparation, and the
signal-evaluation metrics that moved here so agent-utilities core stays torch-free
(see agent-utilities/AGENTS.md "Dependency discipline"). Skipped when the
``[training]`` extra (torch + scikit-learn) is absent.
"""

import numpy as np
import pytest

try:
    import torch

    from data_science_mcp.training.trading_lstm import (
        TradingLSTM,
        evaluate_trading_signal,
        prepare_sequences,
    )

    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False

pytestmark = pytest.mark.skipif(
    not HAS_TRAINING, reason="Training dependencies (torch + scikit-learn) missing"
)


def test_trading_lstm_architecture():
    input_size = 5
    seq_length = 10
    batch_size = 32

    model = TradingLSTM(input_size=input_size, hidden_size=16, num_layers=1)

    # Mock input: (batch_size, seq_length, input_size)
    mock_input = torch.randn(batch_size, seq_length, input_size)

    output = model(mock_input)
    # Output should be (batch_size, 1) due to binary classification sigmoid
    assert output.shape == (batch_size, 1)
    # Sigmoid constraint
    assert torch.all((output >= 0) & (output <= 1))


def test_prepare_sequences():
    features = np.random.randn(50, 5)
    target = np.random.randint(0, 2, 50)
    lookback = 10

    X, y = prepare_sequences(features, target, lookback=lookback)

    assert X.shape == (40, 10, 5)
    assert y.shape == (40,)


def test_evaluation_metrics():
    # Mock predictions and actuals
    predictions = np.array([0.9, 0.1, 0.8, 0.2, 0.6])
    actuals = np.array([1, 0, 1, 0, 1])
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.005])

    accuracy, sharpe, max_dd = evaluate_trading_signal(predictions, actuals, returns)

    assert accuracy == 1.0  # Perfect prediction
    assert sharpe > 0  # Positive returns
    assert max_dd <= 0  # Drawdown is zero or negative

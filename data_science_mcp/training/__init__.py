"""Torch-backed ML training that was re-homed out of ``agent-utilities`` core.

agent-utilities core is the serving plane and must install/run torch-free
(dependency-discipline rule in agent-utilities/AGENTS.md). The heavy ML training
code that used to live in ``agent_utilities.domains.finance`` and
``agent_utilities.knowledge_graph.retrieval`` now lives here, where torch is a
first-class (optional ``[training]`` extra) dependency:

* :mod:`data_science_mcp.training.trading_lstm` — the ``TradingLSTM`` (nn.Module),
  ``prepare_sequences``, ``walk_forward_validation`` LSTM trainer, and the
  ``evaluate_trading_signal`` metrics helper (CONCEPT:AU-KG.research.research-pipeline-runner).
* :mod:`data_science_mcp.training.pause_token_trainer` — the trainable
  ``<pause>``-token PauseRec recommender (CONCEPT:AU-KG.retrieval.pauserec-implicit-reasoning-generative).

Everything imports lazily / guards torch so the package still imports without the
``[training]`` extra installed.
"""

from __future__ import annotations

__all__ = [
    "TradingLSTM",
    "prepare_sequences",
    "walk_forward_validation",
    "evaluate_trading_signal",
    "PauseTokenRecommender",
    "TrainResult",
    "is_available",
]


def __getattr__(name: str):
    """Lazily resolve symbols from the torch-backed submodules.

    Keeps ``import data_science_mcp.training`` cheap and torch-free; the torch
    import only happens when a caller actually reaches for one of these symbols.
    """
    if name in {
        "TradingLSTM",
        "prepare_sequences",
        "walk_forward_validation",
        "evaluate_trading_signal",
    }:
        from . import trading_lstm

        return getattr(trading_lstm, name)
    if name in {"PauseTokenRecommender", "TrainResult", "is_available"}:
        from . import pause_token_trainer

        return getattr(pause_token_trainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

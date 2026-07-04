#!/usr/bin/python
"""Scalar reward / value head over a transformer backbone (CONCEPT:DS-AHE.trainer.per-token-value).

A single linear head reading the backbone's last-layer hidden states powers both
the **reward model** (CONCEPT:DS-AHE.reward.one-sequence-level-score — one sequence-level score at the last real
token) and the **PPO value function** (CONCEPT:DS-AHE.trainer.per-token-value — a per-token value
estimate). The backbone is any HF causal LM run with ``output_hidden_states=True``;
the head is one ``nn.Linear(hidden, 1)``.

The ``nn.Module`` subclass is built lazily on first use (``_scalar_head_cls``) so
this module — and therefore the trainers package — imports without the
``data-science-mcp[training]`` extra. Only the reward/PPO trainers, which already
run under that extra, ever construct a head.

Concept: value-head
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from data_science_mcp.trainers.base import _torch


@lru_cache(maxsize=1)
def _scalar_head_cls() -> type:
    """Build (once) the ``ScalarHeadModel`` class against the imported torch."""
    torch = _torch()
    nn = torch.nn

    class ScalarHeadModel(nn.Module):
        """Causal-LM backbone + a scalar linear head (reward score or value)."""

        def __init__(self, backbone: Any, hidden_size: int) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(hidden_size, 1)

        def _hidden(self, input_ids: Any, attention_mask: Any) -> Any:
            out = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return out.hidden_states[-1]  # (B, T, H)

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any | None = None,
            *,
            per_token: bool = False,
        ) -> Any:
            """Return ``(B,)`` last-token scores, or ``(B, T)`` per-token values.

            ``per_token=False`` (reward): pool the head output at each row's last
            real (non-pad) token. ``per_token=True`` (value): return the head output
            at every position.
            """
            hidden = self._hidden(input_ids, attention_mask)
            scores = self.head(hidden).squeeze(-1)  # (B, T)
            if per_token:
                return scores
            if attention_mask is not None:
                idx = attention_mask.long().sum(dim=1) - 1
            else:
                idx = torch.full(
                    (scores.size(0),), scores.size(1) - 1, device=scores.device
                )
            idx = idx.clamp_min(0)
            rows = torch.arange(scores.size(0), device=scores.device)
            return scores[rows, idx]

    return ScalarHeadModel


def build_scalar_head_model(backbone: Any, hidden_size: int) -> Any:
    """Wrap an already-loaded ``backbone`` with a fresh scalar head."""
    return _scalar_head_cls()(backbone, hidden_size)


def attach_scalar_head(base_model: str, lora: Any | None = None) -> Any:
    """Load the HF base (via :class:`PeftManager`) and wrap it with a scalar head.

    Used by the reward-model trainer (CONCEPT:DS-AHE.reward.one-sequence-level-score) and as the default PPO value
    function (CONCEPT:DS-AHE.trainer.per-token-value) when one is not dependency-injected.
    """
    from data_science_mcp.peft_manager import PeftManager  # noqa: PLC0415

    pm = PeftManager(base_model, lora)
    backbone = pm.attach() if lora is not None else pm.load_base()
    hidden = getattr(getattr(backbone, "config", None), "hidden_size", None)
    if hidden is None:  # pragma: no cover - defensive (all HF configs carry it)
        raise RuntimeError(
            "backbone config has no hidden_size; cannot size the scalar head"
        )
    return build_scalar_head_model(backbone, int(hidden))


__all__ = ["build_scalar_head_model", "attach_scalar_head"]

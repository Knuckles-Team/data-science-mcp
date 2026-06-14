#!/usr/bin/python
"""Reward-model trainer + Bradley-Terry kernel (CONCEPT:ML-008).

CPU smoke on a tiny dependency-injected scalar model — no transformers / GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data_science_mcp import training_data as td  # noqa: E402
from data_science_mcp.trainers import TrainConfig, get_trainer  # noqa: E402
from data_science_mcp.trainers import objectives as obj  # noqa: E402

_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789?.!")


class ToyTokenizer:
    def __init__(self) -> None:
        self._vocab = {"<pad>": 0}
        for c in _CHARS:
            self._vocab[c] = len(self._vocab)
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<pad>"

    def __len__(self) -> int:
        return len(self._vocab)

    def __call__(
        self, texts, return_tensors=None, padding=False, truncation=False, max_length=64
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            ([self._vocab.get(ch, 1) for ch in t.lower()][:max_length] or [0])
            for t in texts
        ]
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        attn = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class ToyRewardModel(torch.nn.Module):
    """Embed → mean over hidden → scalar at the last real token (B,)."""

    def __init__(self, vocab: int, hidden: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None):
        h = self.emb(input_ids)  # (B, T, H)
        if attention_mask is not None:
            idx = attention_mask.long().sum(dim=1) - 1
        else:
            idx = torch.full((h.size(0),), h.size(1) - 1)
        idx = idx.clamp_min(0)
        rows = torch.arange(h.size(0))
        return self.head(h[rows, idx]).squeeze(-1)  # (B,)


# --- kernels --------------------------------------------------------------- #
def test_bradley_terry_loss_decreases_with_margin():
    big_gap = obj.bradley_terry_loss(torch.tensor([3.0]), torch.tensor([0.0]))
    small_gap = obj.bradley_terry_loss(torch.tensor([0.2]), torch.tensor([0.0]))
    assert big_gap.item() < small_gap.item()
    assert big_gap.item() > 0.0


def test_bradley_terry_zero_gap_is_log2():
    loss = obj.bradley_terry_loss(torch.tensor([1.0]), torch.tensor([1.0]))
    assert loss.item() == pytest.approx(float(np.log(2)), abs=1e-5)


# --- trainer smoke --------------------------------------------------------- #
def test_reward_trainer_smoke_learns_preference():
    tok = ToyTokenizer()
    model = ToyRewardModel(vocab=len(tok))
    data = td.build_preference_pairs(
        [
            {"prompt": "q ", "chosen": "a clear helpful answer", "rejected": "no"},
            {"prompt": "p ", "chosen": "another good response here", "rejected": "x"},
        ]
    )
    trainer = get_trainer("reward", TrainConfig(lr=0.1, epochs=25, batch_size=2, seed=0))
    report = trainer.train(data, model=model, tokenizer=tok)
    assert report["trainer"] == "reward"
    assert report["steps"] > 0
    assert all(np.isfinite(report["losses"]))
    assert report["final_loss"] < report["losses"][0]
    assert report["pairwise_accuracy"] is not None
    assert 0.0 <= report["pairwise_accuracy"] <= 1.0


def test_reward_trainer_empty_dataset():
    report = get_trainer("reward", TrainConfig()).train([])
    assert report["steps"] == 0


def test_train_reward_tool_plans_without_execute():
    from data_science_mcp.mcp.mcp_trainers import register_trainer_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_trainer_tools(_Recorder())
    assert "train_reward" in captured
    out = json.loads(
        captured["train_reward"](
            json.dumps([{"prompt": "a", "chosen": "b", "rejected": "c"}]),
            json.dumps({"epochs": 1, "batch_size": 1}),
        )
    )
    assert out["kind"] == "reward" and out["executed"] is False

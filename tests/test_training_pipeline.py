#!/usr/bin/python
"""Tests for the Wave-D fine-tune pipeline + deploy seam (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Exercises the whole flow on a toy model (CPU, no GPU/HF download) and the
deploy seam.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data_science_mcp.training_pipeline import (  # noqa: E402
    DeploymentTarget,
    run_sft_pipeline,
)
from data_science_mcp.trainers import TrainConfig  # noqa: E402

_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789?.!")


class ToyTokenizer:
    def __init__(self):
        self._vocab = {"<pad>": 0, **{c: i + 1 for i, c in enumerate(_CHARS)}}
        self.pad_token, self.pad_token_id, self.eos_token = "<pad>", 0, "<pad>"

    def __call__(
        self, texts, return_tensors=None, padding=False, truncation=False, max_length=64
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            [self._vocab.get(c, 1) for c in t.lower()][:max_length] or [0]
            for t in texts
        ]
        w = max(len(r) for r in rows)
        ids = [r + [0] * (w - len(r)) for r in rows]
        attn = [[1] * len(r) + [0] * (w - len(r)) for r in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyModel(torch.nn.Module):
    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return _Out(self.head(self.emb(input_ids)))


# --- end-to-end pipeline on a toy model -------------------------------------
def test_run_sft_pipeline_end_to_end():
    config = TrainConfig(base_model="toy", lr=0.1, epochs=8, batch_size=2, seed=0)
    report = run_sft_pipeline(
        config,
        traces=[
            {"prompt": "2+2=", "completion": "4"},
            {"prompt": "cap=", "completion": "yes"},
        ],
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
        generate_fn=lambda x: "the sky is blue",
        eval_cases=[
            {
                "input": "ground?",
                "context": {
                    "evidence": "the sky is blue",
                    "retrieved": ["the sky is blue"],
                },
            }
        ],
        deploy=DeploymentTarget(role="generator", served_model_name="toy-served"),
        checkpoint_id="toy-ckpt",
    )
    # data → plan → train
    assert report["data"]["examples"] == 2
    assert report["plan"]["planned_steps"] > 0
    assert report["train"]["steps"] > 0
    assert report["train"]["final_loss"] < report["train"]["losses"][0]
    # eval ran through the reliability suite
    assert report["eval"]["cases"] == 1
    # deployed + live via role resolution
    assert report["deployment"]["role"] == "generator"


def test_run_sft_pipeline_plan_only_without_deploy():
    config = TrainConfig(base_model="toy", lr=0.1, epochs=2, batch_size=2)
    report = run_sft_pipeline(
        config,
        examples=[{"prompt": "a", "completion": "b"}],
        model=ToyModel(),
        tokenizer=ToyTokenizer(),
    )
    assert "deployment" not in report and "eval" not in report
    assert report["train"]["steps"] > 0

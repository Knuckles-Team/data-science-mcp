#!/usr/bin/python
"""Tests for the from-random-init pretraining path (CONCEPT:DS-AHE.trainer.concept-2).

Pure planning + a CPU smoke of the pretrain loop on the toy model (packed
sequences, next-token CE through the shared run_loop). The real random-init build
(``AutoConfig``→``from_config``) and BPE tokenizer training are gated behind the
``transformers``/``tokenizers`` extras and skip cleanly when absent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data_science_mcp.tokenizer_trainer import (  # noqa: E402
    TokenizerSpec,
    plan_tokenizer,
)
from data_science_mcp.trainers import (  # noqa: E402
    PretrainSpec,
    PretrainTrainer,
    TrainConfig,
    get_trainer,
)
from test_trainers import _toy  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure planning                                                                 #
# --------------------------------------------------------------------------- #
def test_pretrain_registered_in_factory():
    t = get_trainer("pretrain", TrainConfig())
    assert isinstance(t, PretrainTrainer)
    assert t.kind == "pretrain"


def test_pretrain_spec_param_count_scales():
    small = PretrainSpec(hidden_size=256, num_hidden_layers=4)
    big = PretrainSpec(hidden_size=1024, num_hidden_layers=24)
    assert big.approx_params() > small.approx_params() > 0


def test_plan_tokenizer_is_pure():
    plan = plan_tokenizer(TokenizerSpec(vocab_size=8000, extra_tokens=("<tool>",)))
    assert plan.vocab_size == 8000
    assert "<tool>" in plan.special_tokens
    assert plan.algorithm == "byte-level-bpe"


# --------------------------------------------------------------------------- #
# CPU smoke: pretrain loop over packed sequences on the toy model              #
# --------------------------------------------------------------------------- #
def test_pretrain_trainer_smoke_reduces_loss():
    model, tok = _toy()
    corpus = [{"text": "the quick brown fox jumps over the lazy dog"}] * 12
    cfg = TrainConfig(lr=0.1, epochs=12, batch_size=2, max_seq_len=8, seed=0)
    report = PretrainTrainer(cfg).train(corpus, model=model, tokenizer=tok)
    assert report["blocks"] > 0
    assert report["steps"] > 0
    assert all(np.isfinite(report["losses"]))
    assert report["final_loss"] < report["losses"][0]
    assert report["approx_params"] > 0


def test_pretrain_pipeline_toy_end_to_end():
    from data_science_mcp.training_pipeline import run_pretrain_pipeline

    model, tok = _toy()
    corpus = ["alpha beta gamma delta", "epsilon zeta eta theta"] * 6
    cfg = TrainConfig(lr=0.1, epochs=6, batch_size=2, max_seq_len=6)
    report = run_pretrain_pipeline(cfg, corpus=corpus, model=model, tokenizer=tok)
    assert report["data"]["records"] == 12
    assert "plan" in report and "train" in report
    assert report["train"]["steps"] > 0


# --------------------------------------------------------------------------- #
# MCP tools                                                                      #
# --------------------------------------------------------------------------- #
def _capture_tools():
    from data_science_mcp.mcp.mcp_trainers import register_trainer_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_trainer_tools(_Recorder())
    return captured


def test_pretrain_model_tool_plans_without_execute():
    tools = _capture_tools()
    assert "pretrain_model" in tools
    out = json.loads(
        tools["pretrain_model"](
            json.dumps([{"text": "a b c"}, {"text": "d e f"}]),
            json.dumps(
                {
                    "epochs": 1,
                    "batch_size": 1,
                    "spec": {"hidden_size": 128, "num_hidden_layers": 2},
                }
            ),
        )
    )
    assert out["kind"] == "pretrain" and out["executed"] is False
    assert out["plan"]["approx_params"] > 0


def test_train_tokenizer_tool_plans_without_execute():
    tools = _capture_tools()
    assert "train_tokenizer" in tools
    out = json.loads(
        tools["train_tokenizer"](
            json.dumps(["hello world", "foo bar baz"]),
            json.dumps({"vocab_size": 4000, "extra_tokens": ["<act>"]}),
        )
    )
    assert out["executed"] is False
    assert out["plan"]["vocab_size"] == 4000
    assert "<act>" in out["plan"]["special_tokens"]


# --------------------------------------------------------------------------- #
# Heavy path (skips without the extra)                                          #
# --------------------------------------------------------------------------- #
def test_train_tokenizer_real_bpe(tmp_path):
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from data_science_mcp.tokenizer_trainer import train_tokenizer

    corpus = ["the cat sat on the mat"] * 50 + ["dogs run fast in the park"] * 50
    tok = train_tokenizer(
        corpus, spec=TokenizerSpec(vocab_size=300), output_dir=str(tmp_path)
    )
    assert len(tok) > 0
    assert (tmp_path / "tokenizer.json").exists() or (
        tmp_path / "tokenizer_config.json"
    ).exists()

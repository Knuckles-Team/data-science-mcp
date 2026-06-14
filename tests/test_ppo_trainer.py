#!/usr/bin/python
"""PPO trainer + GAE/value/whiten kernels + chat-format verifier (CONCEPT:ML-009/ML-012).

CPU smoke on tiny dependency-injected policy/value models — no transformers / GPU.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from data_science_mcp.trainers import TrainConfig, get_trainer  # noqa: E402
from data_science_mcp.trainers import objectives as obj  # noqa: E402

_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789?.!=+<>/")


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


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyPolicy(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return _Out(self.head(self.emb(input_ids)))


class ToyValue(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask=None, *, per_token: bool = False):
        v = self.head(self.emb(input_ids)).squeeze(-1)  # (B, T)
        return v if per_token else v[:, -1]


# --- kernels --------------------------------------------------------------- #
def test_gae_returns_equal_adv_plus_values():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    adv, returns = obj.gae(rewards, values, gamma=1.0, lam=1.0)
    assert adv.shape == values.shape
    assert torch.allclose(returns, adv + values)


def test_gae_terminal_reward_propagates_undiscounted():
    # γ=λ=1, V=0 → every advantage equals the cumulative future reward (here 1.0).
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    adv, _ = obj.gae(rewards, values, gamma=1.0, lam=1.0)
    assert torch.allclose(adv, torch.ones(1, 3))


def test_whiten_zero_mean_unit_var():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = obj.whiten(x)
    assert w.mean().item() == pytest.approx(0.0, abs=1e-5)
    assert w.std(unbiased=False).item() == pytest.approx(1.0, abs=1e-4)


def test_value_function_loss_nonneg_and_zero_at_match():
    v = torch.tensor([[1.0, 2.0]])
    assert obj.value_function_loss(v, v.clone()).item() == pytest.approx(0.0, abs=1e-7)
    assert obj.value_function_loss(v, v + 1.0).item() > 0.0


def test_token_logprob_shape():
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    lp = obj.token_logprob(logits, labels)
    assert lp.shape == (2, 4)
    assert (lp <= 0).all()


# --- chat template + GSM8K verifier (CONCEPT:ML-012) ----------------------- #
def test_chat_template_extract_answer():
    from data_science_mcp.chat_template import extract_answer, render_think_answer

    text = render_think_answer("2+2 is 4", "4")
    assert extract_answer(text) == "4"
    assert extract_answer("the result is 42") == "42"  # fallback to last number


def test_gsm8k_reward_exact_match():
    from data_science_mcp.chat_template import render_think_answer
    from data_science_mcp.trainers.eval_hooks import evaluate_gsm8k, gsm8k_reward

    good = render_think_answer("steps", "18")
    assert gsm8k_reward("q", good, "18") == 1.0
    assert gsm8k_reward("q", render_think_answer("s", "7"), "18") == 0.0
    out = evaluate_gsm8k(lambda q: good, [{"question": "q", "answer": "... #### 18"}])
    assert out["accuracy"] == 1.0


# --- trainer smoke --------------------------------------------------------- #
def _ppo_setup():
    tok = ToyTokenizer()
    return tok, ToyPolicy(len(tok)), ToyValue(len(tok))


def test_ppo_trainer_smoke_verifier_reward():
    tok, policy, value = _ppo_setup()
    data = [
        {"prompt": "2+2= ", "completion": "<answer>4</answer>", "reward": 1.0},
        {"prompt": "3+1= ", "completion": "<answer>4</answer>", "reward": 1.0},
        {"prompt": "9+9= ", "completion": "<answer>2</answer>", "reward": 0.0},
    ]
    trainer = get_trainer(
        "ppo", TrainConfig(lr=0.05, epochs=2, batch_size=2, kl_coef=0.1, seed=0)
    )
    report = trainer.train(data, model=policy, tokenizer=tok, value_model=value)
    assert report["trainer"] == "ppo"
    assert report["steps"] > 0
    assert all(np.isfinite(report["losses"]))
    assert report["mean_reward"] == pytest.approx((1.0 + 1.0 + 0.0) / 3, abs=1e-6)
    assert report["mean_kl"] is not None and report["mean_kl"] >= 0.0


def test_ppo_trainer_with_reward_fn_verifier():
    from data_science_mcp.trainers.eval_hooks import gsm8k_reward

    tok, policy, value = _ppo_setup()
    data = [{"prompt": "2+2= ", "completion": "<answer>4</answer>"}]
    trainer = get_trainer("ppo", TrainConfig(lr=0.05, epochs=1, reward_source="verifier"))
    report = trainer.train(
        data,
        model=policy,
        tokenizer=tok,
        value_model=value,
        reward_fn=lambda p, c: gsm8k_reward(p, c, "4"),
    )
    assert report["mean_reward"] == pytest.approx(1.0, abs=1e-6)


def test_ppo_trainer_empty_dataset():
    assert get_trainer("ppo", TrainConfig()).train([])["steps"] == 0


def test_run_rlhf_pipeline_reward_then_ppo():
    """SFT-free RLHF: reward model (in place) → PPO scored by it (CONCEPT:ML-009)."""
    from data_science_mcp.training_pipeline import run_rlhf_pipeline

    tok, policy, value = _ppo_setup()
    # reward model = embed → scalar at last real token (B,)
    reward_model = ToyValue(len(tok))
    _orig = reward_model.forward
    reward_model.forward = lambda input_ids, attention_mask=None: _orig(  # type: ignore[assignment]
        input_ids, attention_mask, per_token=False
    )
    report = run_rlhf_pipeline(
        TrainConfig(lr=0.05, epochs=2, batch_size=2, reward_source="reward_model"),
        preference_pairs=[
            {"prompt": "q ", "chosen": "good detailed answer", "rejected": "no"}
        ],
        ppo_dataset=[{"prompt": "2+2= ", "completion": "<answer>4</answer>"}],
        model=policy,
        tokenizer=tok,
        value_model=value,
        reward_model=reward_model,
        gsm8k_cases=[{"question": "2+2=", "answer": "#### 4"}],
        generate_fn=lambda q: "<answer>4</answer>",
    )
    assert "reward" in report["stages"] and "ppo" in report["stages"]
    assert report["stages"]["reward"]["steps"] > 0
    assert report["stages"]["ppo"]["steps"] > 0
    assert report["gsm8k"]["accuracy"] == 1.0


def test_train_ppo_tool_plans_without_execute():
    from data_science_mcp.mcp.mcp_trainers import register_trainer_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_trainer_tools(_Recorder())
    assert "train_ppo" in captured
    out = json.loads(
        captured["train_ppo"](
            json.dumps([{"prompt": "a", "completion": "b", "reward": 1.0}]),
            json.dumps({"epochs": 1, "batch_size": 1}),
        )
    )
    assert out["kind"] == "ppo" and out["executed"] is False

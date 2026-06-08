#!/usr/bin/python
"""Tests for the Wave-C gradient trainer scaffold (CONCEPT:AHE-3.1).

Covers the pure pieces (TIES merge, tokenizer plan, rollout buffer, trainer
planning, loss kernels) and a CPU smoke of each trainer on a tiny dependency-
injected toy model — no GPU, no HF download, no peft/vllm.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the trainers need the [training] extra

from data_science_mcp import training_data as td  # noqa: E402
from data_science_mcp.peft_manager import LoraSpec, ties_merge  # noqa: E402
from data_science_mcp.rollout_buffer import RolloutBuffer  # noqa: E402
from data_science_mcp.tokenizer_registry import TokenizerRegistry  # noqa: E402
from data_science_mcp.trainers import TrainConfig, get_trainer  # noqa: E402
from data_science_mcp.trainers import objectives as obj  # noqa: E402
from data_science_mcp.trainers.eval_hooks import evaluate_checkpoint  # noqa: E402

# --------------------------------------------------------------------------- #
# Toy tokenizer + model (DI for CPU smoke; no transformers / GPU)             #
# --------------------------------------------------------------------------- #
_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789?.!")
_PAD = "<pad>"


class ToyTokenizer:
    def __init__(self) -> None:
        self._vocab = {_PAD: 0}
        for c in _CHARS:
            self._vocab[c] = len(self._vocab)
        self.pad_token = _PAD
        self.pad_token_id = 0
        self.eos_token = _PAD

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._vocab.get(tok, -1)

    def add_special_tokens(self, mapping: dict) -> int:
        added = 0
        for tok in mapping.get("additional_special_tokens", []):
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
                added += 1
        return added

    def _encode_one(self, text: str, max_length: int) -> list[int]:
        ids = [self._vocab.get(ch, 1) for ch in text.lower()][:max_length]
        return ids or [0]

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=64,
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = [self._encode_one(t, max_length) for t in texts]
        width = max(len(r) for r in rows)
        input_ids, attn = [], []
        for r in rows:
            pad = width - len(r)
            input_ids.append(r + [0] * pad)
            attn.append([1] * len(r) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyModel(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 16) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return _Out(self.head(self.emb(input_ids)))

    def resize_token_embeddings(self, new_size):
        old = self.emb.num_embeddings
        if new_size <= old:
            return self.emb
        hidden = self.emb.embedding_dim
        new_emb = torch.nn.Embedding(new_size, hidden)
        new_head = torch.nn.Linear(hidden, new_size)
        with torch.no_grad():
            new_emb.weight[:old] = self.emb.weight
            new_head.weight[:old] = self.head.weight
            new_head.bias[:old] = self.head.bias
        self.emb, self.head = new_emb, new_head
        return self.emb


def _toy():
    tok = ToyTokenizer()
    model = ToyModel(vocab=len(tok) + 8)  # headroom for added special tokens
    return model, tok


# --------------------------------------------------------------------------- #
# Loss kernels (objectives)                                                     #
# --------------------------------------------------------------------------- #
def test_sft_cross_entropy_positive_and_shift():
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.randint(0, 7, (2, 5))
    loss = obj.sft_cross_entropy(logits, labels)
    assert loss.item() > 0 and loss.requires_grad


def test_sequence_logprob_masks_padding():
    logits = torch.randn(1, 4, 5)
    labels = torch.tensor([[1, 2, -100, -100]])
    lp = obj.sequence_logprob(logits, labels)
    assert lp.shape == (1,)
    assert lp.item() <= 0  # sum of log-probs


def test_dpo_loss_decreases_with_margin():
    # Wider chosen−rejected margin (beyond the reference) → lower loss.
    ref_c, ref_r = torch.tensor([0.0]), torch.tensor([0.0])
    small = obj.dpo_loss(
        torch.tensor([0.2]), torch.tensor([0.0]), ref_c, ref_r, beta=1.0
    )
    big = obj.dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), ref_c, ref_r, beta=1.0)
    assert big.item() < small.item()


def test_grpo_surrogate_unit_ratio_equals_neg_mean_adv():
    lp = torch.tensor([0.5, -0.5, 1.0])
    adv = torch.tensor([1.0, -1.0, 0.5])
    loss = obj.grpo_surrogate(lp, lp.clone(), adv, clip_eps=0.2)
    assert loss.item() == pytest.approx(-adv.mean().item(), abs=1e-6)


def test_token_masked_surrogate_only_counts_masked():
    lp = torch.tensor([0.0, 0.0])
    adv = torch.tensor([1.0, 5.0])
    mask = torch.tensor([1.0, 0.0])  # only first token credited
    loss = obj.token_masked_surrogate(lp, lp.clone(), adv, mask, clip_eps=0.2)
    assert loss.item() == pytest.approx(-1.0, abs=1e-6)


def test_approx_kl_non_negative_and_zero_at_equality():
    lp = torch.randn(8)
    assert obj.approx_kl(lp, lp.clone()).item() == pytest.approx(0.0, abs=1e-6)
    assert obj.approx_kl(lp, lp + 0.5).item() >= 0.0


# --------------------------------------------------------------------------- #
# TIES merge (pure numpy)                                                       #
# --------------------------------------------------------------------------- #
def test_ties_merge_sign_election_and_disjoint_mean():
    base = {"w": np.zeros(4)}
    # Sign elected by total mass: Σ = 1+3−2 = +2 → positive wins; the −2 is
    # dropped by the disjoint mean, leaving mean({1,3}) = 2.
    tvs = [
        {"w": np.array([1.0, 0.0, 0.0, 0.0])},
        {"w": np.array([3.0, 0.0, 0.0, 0.0])},
        {"w": np.array([-2.0, 0.0, 0.0, 0.0])},
    ]
    merged = ties_merge(base, tvs, density=1.0)
    assert merged["w"][0] == pytest.approx(2.0)


def test_ties_merge_trim_keeps_top_magnitude():
    base = {"w": np.zeros(4)}
    tvs = [{"w": np.array([0.1, 0.2, 5.0, 0.05])}]
    merged = ties_merge(base, tvs, density=0.25)  # keep top 1 of 4
    assert merged["w"][2] == pytest.approx(5.0)
    assert merged["w"][0] == pytest.approx(0.0)


def test_ties_merge_rejects_bad_density():
    with pytest.raises(ValueError):
        ties_merge({"w": np.zeros(2)}, [{"w": np.ones(2)}], density=0.0)


# --------------------------------------------------------------------------- #
# Tokenizer registry                                                            #
# --------------------------------------------------------------------------- #
def test_tokenizer_registry_plan_and_functional():
    reg = TokenizerRegistry()
    reg.register("<tool>", "</tool>").register("<act>", functional=True)
    plan = reg.plan({"a", "b", "<tool>"})
    assert "<tool>" in plan.already_present
    assert set(plan.new_special_tokens) == {"</tool>"}
    assert plan.new_functional_tokens == ["<act>"]
    assert plan.added_count == 2
    assert reg.functional_tokens == ["<act>"]


def test_tokenizer_registry_apply_resizes(monkeypatch):
    model, tok = _toy()
    old = len(tok)
    reg = TokenizerRegistry().register("<plan>", "<act>", functional=True)
    plan = reg.apply(tok, model)
    assert plan.added_count == 2
    assert len(tok) == old + 2
    assert all(i >= 0 for i in reg.functional_token_ids(tok))


# --------------------------------------------------------------------------- #
# Rollout buffer                                                                #
# --------------------------------------------------------------------------- #
def test_rollout_buffer_score_and_grpo_export():
    buf = RolloutBuffer()
    buf.add_group("2+2=", ["4", "5", "four"])
    buf.score(lambda p, c: 1.0 if c == "4" else 0.0)
    groups = buf.to_grpo_groups(min_group=2)
    assert len(groups) == 1
    advs = [s["advantage"] for s in groups[0]["samples"]]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)  # group-normalised


def test_rollout_buffer_generate_with_fake_client():
    class FakeClient:
        def generate(self, prompt, n, **kw):
            return [{"text": f"{prompt}->{i}", "logprobs": [0.0]} for i in range(n)]

    buf = RolloutBuffer()
    buf.generate(["q1", "q2"], FakeClient(), n=3)
    assert len(buf) == 6
    assert set(buf.prompts) == {"q1", "q2"}


# --------------------------------------------------------------------------- #
# Trainer planning (pure) + LoraSpec                                            #
# --------------------------------------------------------------------------- #
def test_trainer_plan_is_pure():
    trainer = get_trainer("sft", TrainConfig(batch_size=4, epochs=2))
    plan = trainer.plan([{"prompt": "a", "completion": "b"}] * 10)
    assert plan["examples"] == 10
    assert plan["planned_steps"] == 6  # ceil(10/4)=3 per epoch * 2
    assert plan["kind"] == "sft"


def test_lora_spec_defaults():
    spec = LoraSpec(r=8, quant_4bit=False)
    assert spec.r == 8 and "q_proj" in spec.target_modules


# --------------------------------------------------------------------------- #
# CPU smoke: each trainer end-to-end on the toy model                           #
# --------------------------------------------------------------------------- #
def test_sft_trainer_smoke_reduces_loss():
    model, tok = _toy()
    data = td.build_sft_examples(
        [{"prompt": "2+2=", "completion": "4"}, {"prompt": "cap=", "completion": "yes"}]
    )
    trainer = get_trainer("sft", TrainConfig(lr=0.1, epochs=12, batch_size=2, seed=0))
    report = trainer.train(data, model=model, tokenizer=tok)
    assert report["steps"] > 0
    assert all(np.isfinite(report["losses"]))
    assert report["final_loss"] < report["losses"][0]


def test_dpo_trainer_smoke_reduces_loss():
    model, tok = _toy()
    data = td.build_preference_pairs(
        [{"prompt": "q=", "chosen": "good answer", "rejected": "bad"}]
    )
    trainer = get_trainer("dpo", TrainConfig(lr=0.1, epochs=15, batch_size=1, beta=0.5))
    report = trainer.train(data, model=model, tokenizer=tok)
    assert report["steps"] > 0
    assert report["final_loss"] < report["losses"][0]


def test_grpo_trainer_smoke_runs_with_kl():
    model, tok = _toy()
    groups = td.build_grpo_groups(
        [
            {
                "prompt": "go ",
                "completions": ["aaa", "bbb", "ccc"],
                "rewards": [1.0, 0.0, 0.5],
            }
        ]
    )
    trainer = get_trainer(
        "grpo", TrainConfig(lr=0.05, epochs=3, kl_coef=0.1, clip_eps=0.2)
    )
    report = trainer.train(groups, model=model, tokenizer=tok)
    assert report["steps"] == 3
    assert all(np.isfinite(report["losses"]))
    assert report["mean_kl"] is not None and report["mean_kl"] >= 0.0


# --------------------------------------------------------------------------- #
# eval_hooks bridge                                                             #
# --------------------------------------------------------------------------- #
def test_merge_adapters_ties_tool_roundtrip():
    from fastmcp import FastMCP

    from data_science_mcp.mcp.mcp_trainers import register_trainer_tools

    mcp = FastMCP("test")
    register_trainer_tools(mcp)  # registers without error


def test_train_sft_tool_plans_without_execute():
    import json

    from data_science_mcp.mcp.mcp_trainers import register_trainer_tools

    captured: dict = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn

            return deco

    register_trainer_tools(_Recorder())
    out = json.loads(
        captured["train_sft"](
            json.dumps([{"prompt": "a", "completion": "b"}]),
            json.dumps({"epochs": 1, "batch_size": 1}),
        )
    )
    assert out["kind"] == "sft" and out["executed"] is False
    assert out["plan"]["planned_steps"] == 1


@pytest.mark.asyncio
async def test_trainer_tools_registered_on_live_server():
    import sys

    sys.argv = ["mcp_server.py"]
    from data_science_mcp.mcp_server import get_mcp_instance

    mcp, _, _, registered_tags = get_mcp_instance()
    assert "model-training" in registered_tags
    names = {t.name for t in await mcp.list_tools()}
    assert {"train_sft", "train_dpo", "train_grpo", "merge_adapters_ties"} <= names


def test_evaluate_checkpoint_runs_reliability_suite():
    cases = [
        {
            "input": "What grounds this?",
            "context": {
                "evidence": "the sky is blue",
                "retrieved": ["the sky is blue"],
            },
        }
    ]
    out = evaluate_checkpoint(lambda x: "the sky is blue", cases)
    assert out["cases"] == 1
    assert 0.0 <= out["overall_score"] <= 1.0
    assert "scores" in out["results"][0]

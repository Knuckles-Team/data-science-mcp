"""Tests for the deterministic training-data & reward engine (CONCEPT:AHE-3.1).

Pure-Python builders (no engine/GPU) + MCP tool registration.
"""

import pytest

from data_science_mcp import training_data as td

TRAINING_DATA_TOOLS = {"build_training_dataset", "compose_reward"}


# --- SFT --------------------------------------------------------------------


def test_build_sft_examples_normalizes_and_drops_empty():
    traces = [
        {"prompt": "q1", "completion": "a1"},
        {"input": "q2", "output": "a2"},  # alt keys
        {"prompt": "", "completion": "x"},  # dropped (empty prompt)
    ]
    out = td.build_sft_examples(traces)
    assert out == [
        {"prompt": "q1", "completion": "a1"},
        {"prompt": "q2", "completion": "a2"},
    ]


# --- DPO --------------------------------------------------------------------


def test_build_preference_pairs_explicit():
    pairs = td.build_preference_pairs(
        [{"prompt": "p", "chosen": "good", "rejected": "bad"}]
    )
    assert pairs[0]["chosen"] == "good" and pairs[0]["rejected"] == "bad"
    assert pairs[0]["failure_point"] is None


def test_build_preference_pairs_error_attributed():
    pairs = td.build_preference_pairs(
        [
            {
                "prompt": "p",
                "gold": "corrected",
                "biased": "wrong",
                "step_failed": [False, False, True],
            }
        ]
    )
    assert pairs[0]["chosen"] == "corrected"
    assert pairs[0]["rejected"] == "wrong"
    assert pairs[0]["failure_point"] == 2  # first divergence index


# --- GRPO -------------------------------------------------------------------


def test_build_grpo_groups_attaches_advantages():
    groups = td.build_grpo_groups(
        [{"prompt": "p", "completions": ["a", "b", "c"], "rewards": [1.0, 2.0, 3.0]}]
    )
    samples = groups[0]["samples"]
    assert [s["completion"] for s in samples] == ["a", "b", "c"]
    assert samples[1]["advantage"] == 0.0  # mean reward → 0 advantage
    assert samples[0]["advantage"] < 0 < samples[2]["advantage"]


def test_build_grpo_skips_mismatched_lengths():
    assert (
        td.build_grpo_groups([{"prompt": "p", "completions": ["a"], "rewards": []}])
        == []
    )


# --- filter / reward / dispatch --------------------------------------------


def test_filter_by_difficulty():
    trajs = [{"step_count": 1}, {"step_count": 4}]
    assert td.filter_by_difficulty(trajs, min_steps=3) == [{"step_count": 4}]


def test_score_reward_conditional():
    r = td.score_reward(
        {"acc": 1.0, "func": 1.0}, {"acc": 1.0, "func": 1.0}, {"func": False}
    )
    assert r == pytest.approx(1.0)


def test_build_dataset_dispatch_and_unknown():
    assert td.build_dataset("sft", [{"prompt": "p", "completion": "c"}]) == [
        {"prompt": "p", "completion": "c"}
    ]
    with pytest.raises(ValueError):
        td.build_dataset("bogus", [])


def test_trainer_protocol_is_structural_interface():
    from data_science_mcp.training_data import Trainer

    class DummyTrainer:
        name = "dummy"

        def train(self, dataset, **kwargs):
            return {"trained": len(dataset)}

    assert isinstance(DummyTrainer(), Trainer)  # structural conformance


# --- MCP tool registration --------------------------------------------------


@pytest.mark.asyncio
async def test_training_data_tools_registered():
    import sys

    sys.argv = ["mcp_server.py"]  # avoid create_mcp_server parsing the pytest CLI
    from data_science_mcp.mcp_server import get_mcp_instance

    mcp, _, _, registered_tags = get_mcp_instance()
    assert "model-training" in registered_tags
    names = {t.name for t in await mcp.list_tools()}
    assert TRAINING_DATA_TOOLS <= names


def test_register_training_data_tools_on_fresh_server():
    from fastmcp import FastMCP

    from data_science_mcp.mcp.mcp_training_data import register_training_data_tools

    mcp = FastMCP("test")
    register_training_data_tools(mcp)  # registers without error

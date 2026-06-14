#!/usr/bin/python
"""Tests for shortcut-resistant search-task corpora (KG-2.67/2.68/2.69, AHE-3.27)."""

from __future__ import annotations

from data_science_mcp.search_task_corpus import (
    rollouts_to_grpo,
    search_reward,
    tasks_to_sft,
    trajectories_to_preference_pairs,
)

ANS = ["Ada Botanist"]


def _search(observation: str = "") -> dict:
    return {"kind": "search", "observation": observation, "model_text": ""}


def _model(text: str = "") -> dict:
    return {"kind": "model", "observation": "", "model_text": text}


def _hard_traj(correct: bool = True) -> dict:
    return {
        "steps": [_search(), _search(), _search(), _search("Ada Botanist")],
        "answer_aliases": ANS,
        "correct": correct,
    }


def _shortcut_traj() -> dict:
    return {
        "steps": [_model("the answer is Ada Botanist"), _search("Ada Botanist record")],
        "answer_aliases": ANS,
        "correct": True,
    }


def test_tasks_to_sft_round_trips_through_builder():
    items = [
        {"question": "Identify the botanist...", "trajectory": "search; read; answer"},
        {"task": {"question": "Q2"}, "trajectory": {"steps": ["s"]}},
        {"question": "", "trajectory": ""},  # dropped by build_sft_examples
    ]
    out = tasks_to_sft(items)
    assert len(out) == 2
    assert out[0] == {
        "prompt": "Identify the botanist...",
        "completion": "search; read; answer",
    }
    assert all(set(r) == {"prompt", "completion"} for r in out)


def test_trajectories_to_preference_pairs_marks_failure_point():
    items = [
        {
            "question": "Q",
            "gold": "long grounded search",
            "shortcut": "named answer early",
            "step_failed": [False, True],
        }
    ]
    pairs = trajectories_to_preference_pairs(items)
    assert len(pairs) == 1
    assert pairs[0]["chosen"] == "long grounded search"
    assert pairs[0]["rejected"] == "named answer early"
    assert pairs[0]["failure_point"] == 1  # first divergence step


def test_search_reward_prefers_search_heavy_over_shortcut():
    hard = search_reward(_hard_traj(correct=True))
    shortcut = search_reward(_shortcut_traj())
    wrong = search_reward(_hard_traj(correct=False))
    assert hard > shortcut  # late-hitting, non-prior-bound beats answer-first
    assert wrong == 0.0  # incorrect answer earns nothing (bonuses gated off)
    assert shortcut < hard


def test_rollouts_to_grpo_attaches_advantages():
    items = [
        {
            "question": "Q",
            "completions": ["good rollout", "shortcut rollout"],
            "trajectories": [_hard_traj(correct=True), _shortcut_traj()],
        }
    ]
    groups = rollouts_to_grpo(items)
    assert len(groups) == 1
    samples = groups[0]["samples"]
    assert len(samples) == 2
    # the search-heavy rollout gets the higher (positive) advantage
    good = next(s for s in samples if s["completion"] == "good rollout")
    short = next(s for s in samples if s["completion"] == "shortcut rollout")
    assert good["advantage"] > short["advantage"]

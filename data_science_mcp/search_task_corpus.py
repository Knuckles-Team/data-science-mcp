#!/usr/bin/python
"""Shortcut-resistant search tasks → SFT / DPO / GRPO corpora.

The trainer-side glue for agent-utilities' search-task synthesizer
(CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72) and realized-difficulty signatures (CONCEPT:AU-AHE.reward.search-task-corpus),
distilled from FORT-Searcher (arXiv:2606.12087). It turns synthesized
shortcut-resistant tasks and the trajectories a solver produces on them into the
three in-house corpora — reusing the existing builders in
:mod:`data_science_mcp.training_data` unchanged.

Where FORT trains **SFT-only**, this also mints the DPO and GRPO corpora the same
data affords:

* :func:`tasks_to_sft` — gold trajectory as the completion (SFT).
* :func:`trajectories_to_preference_pairs` — a *shortcut* trajectory is the perfect
  DPO ``rejected`` against the gold ``chosen`` route (the edge over SFT-only).
* :func:`rollouts_to_grpo` — reward each rollout by its realized search difficulty
  (FORT signatures), so the policy is pushed toward long, late-hitting,
  non-prior-bound search behavior.

Concept: search-task-corpus
"""

from __future__ import annotations

import json
from typing import Any

from agent_utilities.graph.training_signals import (
    answer_hit_time,
    prior_shortcut_rate,
    solving_cost,
)

from data_science_mcp import training_data as td

# Reference scales that normalize raw signatures into a 0..1 reward band. A
# trajectory at/above the reference solving cost / hit time earns full credit.
_REF_COST = 6.0
_REF_HIT = 4.0

_DEFAULT_REWARD_WEIGHTS = {
    "correct": 1.0,
    "search_depth": 0.5,
    "late_hit": 0.5,
    "prior_penalty": -1.0,
}


def _question(item: dict[str, Any]) -> str:
    if item.get("question"):
        return str(item["question"])
    task = item.get("task") or {}
    return str(task.get("question", ""))


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)


def tasks_to_sft(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Synthesized task + gold trajectory → SFT ``{prompt, completion}`` records.

    Each item is ``{question | task, trajectory}``; ``trajectory`` is the gold
    search trajectory (string or JSON-serializable). Delegates to
    :func:`training_data.build_sft_examples` (empties dropped there).
    """
    raw = [
        {"prompt": _question(it), "completion": _as_text(it.get("trajectory"))}
        for it in items
    ]
    return td.build_sft_examples(raw)


def trajectories_to_preference_pairs(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Gold vs shortcut trajectory → DPO ``{prompt, chosen, rejected, failure_point}``.

    Each item is ``{question | task, gold, shortcut, step_failed?}``. The shortcut
    trajectory (one that solved the task via a cheap identifying route) becomes the
    ``rejected`` sample DPO suppresses; ``step_failed`` (per-step divergence of the
    shortcut route) anchors the failure point. Delegates to
    :func:`training_data.build_preference_pairs`.
    """
    raw = [
        {
            "prompt": _question(it),
            "chosen": _as_text(it.get("gold")),
            "rejected": _as_text(it.get("shortcut")),
            "step_failed": it.get("step_failed"),
        }
        for it in items
    ]
    return td.build_preference_pairs(raw)


def search_reward(
    trajectory: dict[str, Any],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    """Reward one rollout by realized search difficulty (FORT signatures).

    Rewards correctness, then — gated on a correct answer — a deep, late-hitting
    trajectory and penalizes prior-knowledge binding. A correct-but-shortcut
    rollout (answer named before evidence, or solved in one step) scores far below
    a correct search-heavy one. Reuses :func:`training_data.score_reward`
    (the composite reward spine).
    """
    weights = weights or _DEFAULT_REWARD_WEIGHTS
    correct = bool(trajectory.get("correct", trajectory.get("success", False)))
    cost = solving_cost([trajectory])
    hit = answer_hit_time(trajectory)
    prior = prior_shortcut_rate([trajectory])
    components = {
        "correct": 1.0 if correct else 0.0,
        "search_depth": min(1.0, cost / _REF_COST),
        "late_hit": min(1.0, hit / _REF_HIT),
        "prior_penalty": prior,
    }
    # search bonuses only count when the answer is actually correct.
    conditions = {"search_depth": correct, "late_hit": correct}
    return td.score_reward(components, weights, conditions)


def rollouts_to_grpo(
    items: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Per-task rollouts → GRPO groups rewarded by realized search difficulty.

    Each item is ``{question | task, completions:[str], trajectories:[traj]}`` with
    one trajectory per completion. Each rollout is scored by :func:`search_reward`,
    then handed to :func:`training_data.build_grpo_groups` for group-normalized
    advantages.
    """
    groups: list[dict[str, Any]] = []
    for it in items:
        completions = list(it.get("completions", []))
        trajectories = list(it.get("trajectories", []))
        if not completions or len(trajectories) != len(completions):
            continue
        rewards = [search_reward(t, weights=weights) for t in trajectories]
        groups.append(
            {
                "prompt": _question(it),
                "completions": completions,
                "rewards": rewards,
            }
        )
    return td.build_grpo_groups(groups)

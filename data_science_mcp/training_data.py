#!/usr/bin/python
"""Deterministic training-data & reward engine (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

The data-quality half of the in-house training substrate (Wave C). Pure Python —
no torch, no GPU — it turns execution traces into the SFT / DPO / GRPO corpora and
reward signals every training-gated research paper consumes, reusing the
agent-utilities reward spine (`agent_utilities.graph.training_signals`).

Building this now makes the gradient trainers a thin follow-up: a :class:`Trainer`
implementation (torch/PEFT, GPU) consumes these corpora unchanged. See the
research-evolution STRATEGY (Wave C) — the data engine is fully deterministic and
runnable today; only the optimizer step is GPU-gated.

Builders:
* :func:`build_sft_examples`     — trace → ``{prompt, completion}`` (OpenSeeker/MeMo SFT).
* :func:`build_preference_pairs` — gold/biased → ``{prompt, chosen, rejected, failure_point}`` (MedCausalX DPO).
* :func:`build_grpo_groups`      — completions+rewards → group-normalized advantages (ATLAS/SDAR GRPO).
* :func:`filter_by_difficulty`   — drop trajectories below a tool-call/step floor (OpenSeeker data quality).
* :func:`score_reward`           — composite, conditionally-gated reward (ATLAS func-reward).

Concept: training-data
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent_utilities.graph.training_signals import (
    batch_normalized_advantage,
    composite_reward,
    difficulty_floor_filter,
    failure_point,
)


def build_sft_examples(traces: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Trace dicts → SFT ``{prompt, completion}`` records (empties dropped)."""
    out: list[dict[str, str]] = []
    for t in traces:
        prompt = str(t.get("prompt") or t.get("input") or "").strip()
        completion = str(t.get("completion") or t.get("output") or "").strip()
        if prompt and completion:
            out.append({"prompt": prompt, "completion": completion})
    return out


def build_preference_pairs(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build DPO ``{prompt, chosen, rejected, failure_point}`` pairs.

    Accepts explicit ``{prompt, chosen, rejected}`` or error-attributed
    ``{prompt, gold, biased, step_failed:[bool]}`` (the failure-point anchors
    where the biased trajectory first diverged — MedCausalX Stage-I).
    """
    pairs: list[dict[str, Any]] = []
    for s in samples:
        prompt = str(s.get("prompt", "")).strip()
        chosen = str(s.get("chosen") or s.get("gold") or "").strip()
        rejected = str(s.get("rejected") or s.get("biased") or "").strip()
        if not (prompt and chosen and rejected):
            continue
        step_failed = s.get("step_failed")
        fp = failure_point(list(step_failed)) if step_failed else None
        pairs.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "failure_point": fp,
            }
        )
    return pairs


def build_grpo_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build GRPO groups with per-completion group-normalized advantages.

    Each input group is ``{prompt, completions:[...], rewards:[...]}``; the output
    attaches the advantage ``(r−μ)/σ`` to each completion.
    """
    out: list[dict[str, Any]] = []
    for g in groups:
        prompt = str(g.get("prompt", ""))
        comps = list(g.get("completions", []))
        rewards = [float(r) for r in g.get("rewards", [])]
        if not comps or len(rewards) != len(comps):
            continue
        adv = batch_normalized_advantage(rewards)
        out.append(
            {
                "prompt": prompt,
                "samples": [
                    {"completion": comps[i], "reward": rewards[i], "advantage": adv[i]}
                    for i in range(len(comps))
                ],
            }
        )
    return out


def filter_by_difficulty(
    trajectories: list[Any], *, min_steps: int = 2, count_key: str = "step_count"
) -> list[Any]:
    """Keep only trajectories at/above a difficulty floor (data-quality engine)."""
    return difficulty_floor_filter(
        trajectories, min_count=min_steps, count_key=count_key
    )


def score_reward(
    components: dict[str, float],
    weights: dict[str, float],
    conditions: dict[str, bool] | None = None,
) -> float:
    """Composite, conditionally-gated reward (delegates to the reward spine)."""
    return composite_reward(components, weights, conditions=conditions)


_BUILDERS = {
    "sft": build_sft_examples,
    "dpo": build_preference_pairs,
    "grpo": build_grpo_groups,
}


def build_dataset(kind: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dispatch a dataset build by ``kind`` (``sft``|``dpo``|``grpo``)."""
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"unknown dataset kind: {kind!r}")
    return builder(items)


@runtime_checkable
class Trainer(Protocol):
    """Seam for a gradient trainer (Wave D, torch/PEFT, GPU).

    A concrete SFT / DPO / GRPO trainer consumes the corpora produced above and
    returns a training report; registering its checkpoint via the model registry
    role-resolution seam makes it live with no hot-path change. Kept as an
    interface here so the deterministic data engine ships without heavy deps.
    """

    name: str

    def train(self, dataset: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Train on a built dataset and return a report (loss/metrics/checkpoint)."""
        ...

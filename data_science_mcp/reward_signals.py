"""Deterministic reward and dataset-prep signals for the training-data pipeline.

Pure, model-free functions that turn execution traces into reward and dataset
signals: group-normalized advantage, failure-point attribution, composite reward,
difficulty-floor filtering and search-trajectory signatures.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

_TRAINABLE_SOURCES = frozenset({"model", "action"})
_RETRIEVAL_KINDS = frozenset({"search", "retrieval", "tool", "read", "browse"})


def _normalize_advantage(
    rewards: list[float], eps: float, length_unbiased: bool
) -> list[float]:
    """Normalize one group of rewards to advantages. See public wrapper."""
    n = len(rewards)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(rewards) / n
    centered = [r - mean for r in rewards]
    if length_unbiased:
        # Dr.GRPO (arXiv:2503.20783): omit the ``/σ`` term. Dividing by the
        # per-group std injects a question-difficulty bias (easy questions with
        # small σ get inflated advantages) that, coupled with token-length loss
        # normalization, over-rewards short responses. The centered advantage
        # ``(r − μ)`` is the length-/difficulty-unbiased signal.
        return [round(c, 6) for c in centered]
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * n
    return [round(c / std, 6) for c in centered]


def batch_normalized_advantage(
    rewards: list[float],
    *,
    eps: float = 1e-8,
    length_unbiased: bool = False,
    mode: str = "group",
    group_ids: list[Any] | None = None,
) -> list[float]:
    """Group-normalized advantage ``A_i = (r_i − μ) / σ`` (GRPO core).

    Returns zeros for a degenerate group (≤1 sample or ~zero variance), so a
    collapsed batch produces no spurious gradient signal.

    - ``mode="group"`` with ``group_ids`` normalizes *within* each group id (GRPO's
      per-prompt grouping); without ``group_ids`` the whole list is one group.
    - ``mode="global"`` normalizes across the entire list regardless of ``group_ids``
      (REINFORCE++ global normalization, arXiv:2501.03262).
    - ``length_unbiased=True`` applies the Dr.GRPO correction (arXiv:2503.20783):
      the centered ``(r − μ)`` advantage without the difficulty-biased ``/σ`` term.

    Defaults reproduce the original GRPO behaviour exactly (group, σ-normalized).
    """
    n = len(rewards)
    if n == 0:
        return []
    if mode == "group" and group_ids is not None:
        if len(group_ids) != n:
            raise ValueError("group_ids must be the same length as rewards")
        out = [0.0] * n
        buckets: dict[Any, list[int]] = defaultdict(list)
        for i, g in enumerate(group_ids):
            buckets[g].append(i)
        for idxs in buckets.values():
            adv = _normalize_advantage([rewards[i] for i in idxs], eps, length_unbiased)
            for i, a in zip(idxs, adv, strict=True):
                out[i] = a
        return out
    return _normalize_advantage(rewards, eps, length_unbiased)


def failure_point(step_failed: list[bool]) -> int | None:
    """First-divergence index: index of the first failed step, else ``None``.

    ``step_failed[i]`` is True when step ``i`` diverged (e.g. outcome INCORRECT).
    This is the anchor for error-attributed preference pairs (winner = corrected
    trajectory from ``t_fail`` onward).
    """
    for i, failed in enumerate(step_failed):
        if failed:
            return i
    return None


def composite_reward(
    components: dict[str, float],
    weights: dict[str, float],
    *,
    conditions: dict[str, bool] | None = None,
) -> float:
    """Weighted sum of reward components with optional per-component gating.

    A component whose ``conditions[name]`` is ``False`` contributes 0 (e.g. a
    functional-token reward granted only if the token was used AND the answer is
    correct). Components default to "on" when unconditioned.
    """
    conditions = conditions or {}
    total = 0.0
    for name, value in components.items():
        if conditions.get(name, True):
            total += weights.get(name, 0.0) * value
    return round(total, 6)


def difficulty_floor_filter(
    items: list[Any],
    *,
    min_count: int = 1,
    count_key: str = "step_count",
) -> list[Any]:
    """Keep only trajectories at/above a difficulty floor (min steps/tool-calls).

    ``items`` may be dicts (read ``count_key``) or objects (read the attribute).
    The data-quality filter that makes a synthesized SFT corpus hard enough to be
    worth training on.
    """
    out: list[Any] = []
    for it in items:
        count = (
            it.get(count_key, 0) if isinstance(it, dict) else getattr(it, count_key, 0)
        )
        if int(count or 0) >= min_count:
            out.append(it)
    return out


def _is_retrieval(step: dict[str, Any]) -> bool:
    if step.get("is_retrieval"):
        return True
    return str(step.get("kind", "")).lower() in _RETRIEVAL_KINDS


def _alias_in(text: Any, aliases: list[str]) -> bool:
    if not text:
        return False
    low = str(text).lower()
    return any(a and a.lower() in low for a in aliases)


def _hit_times(
    steps: list[dict[str, Any]], aliases: list[str]
) -> tuple[int | None, int | None]:
    """1-based positions of the first tool-observation hit and first model-text hit.

    ``T_tool`` is the first step whose tool ``observation`` contains the gold
    answer (or an alias); ``T_model`` is the first step whose ``model_text``
    mentions it. Positions are on the shared step timeline so they are directly
    comparable (FORT eqs 16/18). ``None`` means the answer never surfaced there.
    """
    t_tool: int | None = None
    t_model: int | None = None
    for i, step in enumerate(steps, start=1):
        if (
            t_tool is None
            and _is_retrieval(step)
            and _alias_in(step.get("observation"), aliases)
        ):
            t_tool = i
        if t_model is None and _alias_in(step.get("model_text"), aliases):
            t_model = i
        if t_tool is not None and t_model is not None:
            break
    return t_tool, t_model


def _retrieval_count(steps: list[dict[str, Any]]) -> int:
    return sum(1 for s in steps if _is_retrieval(s))


def solving_cost(trajectories: list[dict[str, Any]]) -> float:
    """Realized solving cost ``Ω̂`` — mean retrieval calls per trajectory (FORT eq 15).

    Counts only retrieval steps (the cost of interest), not model-only turns.
    A high ``Ω̂`` means the solver issued many searches — but on its own does not
    prove the answer was hard to find (see :func:`mean_answer_hit_time`).
    """
    if not trajectories:
        return 0.0
    return round(
        sum(_retrieval_count(t.get("steps", [])) for t in trajectories)
        / len(trajectories),
        6,
    )


def answer_hit_time(trajectory: dict[str, Any]) -> int:
    """Answer hit time ``T_hit = min(T_tool, T_model)`` for one trajectory (FORT eq 16).

    The earliest step at which the gold answer becomes available — either in a
    tool observation or in the model's own text. A *late* hit time is the
    observable signature of suppressed cheap identifying routes. Returns
    ``len(steps) + 1`` (a beyond-the-end sentinel) when the answer never surfaces.
    """
    steps = trajectory.get("steps", [])
    aliases = list(trajectory.get("answer_aliases", []))
    t_tool, t_model = _hit_times(steps, aliases)
    cands = [t for t in (t_tool, t_model) if t is not None]
    return min(cands) if cands else len(steps) + 1


def prior_shortcut_rate(trajectories: list[dict[str, Any]]) -> float:
    """Prior-shortcut rate ``p̂_prior`` (FORT eq 18) — fraction with ``T_model < T_tool``.

    A trajectory is prior-bound when the model names the answer *before* any
    retrieved observation anchors it — a conservative proxy for solver-side
    prior-knowledge binding. Lower is better training data.
    """
    if not trajectories:
        return 0.0
    bound = 0
    for t in trajectories:
        t_tool, t_model = _hit_times(
            t.get("steps", []), list(t.get("answer_aliases", []))
        )
        if t_model is not None and (t_tool is None or t_model < t_tool):
            bound += 1
    return round(bound / len(trajectories), 6)

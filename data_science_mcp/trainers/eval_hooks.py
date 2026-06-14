#!/usr/bin/python
"""Checkpoint → reliability-suite evaluation bridge (CONCEPT:AHE-3.1).

After a trainer produces a checkpoint, we want the same regression gate the rest
of the framework uses — the **AHE-3.1 reliability suite**
(:func:`agent_utilities.harness.reliability_scorers.build_reliability_suite`:
faithfulness, safety, tool-necessity, deception, citation, Brier-skill, retrieval
recall, trap-injection) — applied to the checkpoint's outputs. This closes the
loop the papers care about: did fine-tuning *internalise* the behaviour without
regressing grounding/safety (SDAR internalisation, ATLAS overhead, MedCausalX
robustness)?

The checkpoint is abstracted as a ``generate_fn(input_text) -> output_text`` so
this works for an HF model, a vLLM endpoint, or an injected fake in tests — no GPU
required to exercise the bridge.

Concept: eval-hooks
"""

from __future__ import annotations

from typing import Any, Callable

from agent_utilities.harness.reliability_scorers import build_reliability_suite

GenerateFn = Callable[[str], str]


def evaluate_checkpoint(
    generate_fn: GenerateFn,
    cases: list[dict[str, Any]],
    *,
    scorers: list[Any] | None = None,
) -> dict[str, Any]:
    """Run the reliability suite over a checkpoint's generations.

    Args:
        generate_fn: maps an input prompt to the checkpoint's output text.
        cases: eval cases, each ``{"input": str, "context": {...}?}`` — ``context``
            carries the per-scorer ground truth (evidence, gold set, etc.).
        scorers: optional explicit scorer instances (defaults to the full suite).

    Returns:
        ``{cases, overall_score, pass_rate, results:[...]}``.
    """
    harness = build_reliability_suite(scorers)
    results: list[dict[str, Any]] = []
    for case in cases:
        inp = str(case.get("input", ""))
        ctx = case.get("context")
        out = generate_fn(inp)
        agg = harness.evaluate(inp, out, ctx)
        results.append(
            {
                "input": inp,
                "output": out,
                "overall": agg.overall_score,
                "passed": agg.all_passed,
                "scores": {r.evaluator: r.score for r in agg.results},
            }
        )
    n = len(results) or 1
    return {
        "cases": len(results),
        "overall_score": sum(r["overall"] for r in results) / n,
        "pass_rate": sum(1 for r in results if r["passed"]) / n,
        "results": results,
    }


class ReliabilityEvalHook:
    """Reusable post-training hook that scores a checkpoint on fixed eval cases."""

    def __init__(
        self, cases: list[dict[str, Any]], *, scorers: list[Any] | None = None
    ) -> None:
        self.cases = cases
        self.scorers = scorers

    def __call__(self, generate_fn: GenerateFn) -> dict[str, Any]:
        return evaluate_checkpoint(generate_fn, self.cases, scorers=self.scorers)


def evaluate_benchmarks(
    model_path: str,
    tasks: list[str],
    *,
    limit: int | None = None,
    batch_size: int | str = "auto",
    device: str | None = None,
) -> dict[str, Any]:
    """Score a checkpoint on standard benchmarks via ``lm-eval`` (CONCEPT:ML-006).

    A complement to the AHE-3.1 reliability suite: where that measures grounding /
    safety regressions, this runs community benchmarks (e.g. ``hellaswag``,
    ``arc_easy``, ``gsm8k``) through EleutherAI's ``lm-evaluation-harness``. The
    harness is an optional GPU-host dep (``data-science-mcp[eval]``), imported
    lazily so this module stays light.

    Returns ``{tasks, results:{task: metrics}}`` or ``{error}`` when ``lm-eval`` is
    absent.
    """
    try:
        from lm_eval import simple_evaluate  # noqa: PLC0415
    except ImportError:  # pragma: no cover - without the extra
        return {"error": "lm-eval not installed — install data-science-mcp[eval]"}
    model_args = f"pretrained={model_path}"
    if device:  # pragma: no cover - GPU host
        model_args += f",device={device}"
    out = simple_evaluate(  # pragma: no cover - heavy GPU eval
        model="hf",
        model_args=model_args,
        tasks=tasks,
        limit=limit,
        batch_size=batch_size,
    )
    return {"tasks": tasks, "results": out.get("results", {})}


def gsm8k_reward(prompt: str, completion: str, gold: Any) -> float:
    """Verifiable GSM8K reward: ``1.0`` if the final answer matches ``gold`` else ``0.0``.

    The PPO/GRPO ``reward_source="verifier"`` signal for math reasoning — exact-match
    on the ``<answer>…</answer>`` span (CONCEPT:ML-012 chat format), no reward model
    needed. ``prompt`` is unused but kept for the ``reward_fn(prompt, completion)``
    signature the trainers expect; bind ``gold`` per prompt via a closure.
    """
    from data_science_mcp.chat_template import answer_matches  # noqa: PLC0415

    return 1.0 if answer_matches(completion, gold) else 0.0


def evaluate_gsm8k(
    generate_fn: GenerateFn, cases: list[dict[str, Any]], *, limit: int | None = None
) -> dict[str, Any]:
    """Greedy-decode GSM8K cases and score exact-match accuracy (CONCEPT:ML-012).

    ``cases`` are ``{"question": str, "answer": str}`` (GSM8K ``answer`` carries the
    gold after ``####``; a bare numeric ``answer`` also works). Returns
    ``{cases, accuracy, results:[{question, output, gold, correct}]}`` — the
    per-stage metric the post-training eval table reports.
    """
    from data_science_mcp.chat_template import answer_matches, gsm8k_gold  # noqa: PLC0415

    rows = cases[:limit] if limit is not None else cases
    results: list[dict[str, Any]] = []
    for case in rows:
        q = str(case.get("question", case.get("input", "")))
        gold = gsm8k_gold(str(case.get("answer", ""))) or str(case.get("answer", ""))
        out = generate_fn(q)
        results.append(
            {
                "question": q,
                "output": out,
                "gold": gold,
                "correct": answer_matches(out, gold),
            }
        )
    n = len(results) or 1
    return {
        "cases": len(results),
        "accuracy": sum(1 for r in results if r["correct"]) / n,
        "results": results,
    }


__all__ = [
    "evaluate_checkpoint",
    "evaluate_benchmarks",
    "evaluate_gsm8k",
    "gsm8k_reward",
    "ReliabilityEvalHook",
    "GenerateFn",
]

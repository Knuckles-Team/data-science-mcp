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


__all__ = ["evaluate_checkpoint", "ReliabilityEvalHook", "GenerateFn"]

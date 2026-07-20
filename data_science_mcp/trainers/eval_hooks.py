#!/usr/bin/python
"""Checkpoint → reliability-suite evaluation bridge (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

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

import re
import tempfile
from pathlib import Path
from typing import Any, Callable

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
    # Keep benchmark-only and launcher imports independent of the optional native
    # numeric kernel. Reliability scoring loads the full harness only when used.
    from agent_utilities.harness.reliability_scorers import (  # noqa: PLC0415
        build_reliability_suite,
    )

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
    revision: str | None = None,
) -> dict[str, Any]:
    """Score a checkpoint on standard benchmarks via LightEval.

    A complement to the AHE-3.1 reliability suite: where that measures grounding /
    safety regressions, this runs community benchmarks (e.g. ``hellaswag``,
    ``arc_easy``, ``gsm8k``) through Hugging Face LightEval. The harness is an
    optional GPU-host dependency (``data-science-mcp[eval]``), imported lazily so
    this module stays light. Remote model IDs must name an immutable commit; local
    checkpoint paths need no revision. Remote code loading and result publishing
    remain disabled.

    Returns ``{tasks, results:{task: metrics}}`` or a bounded ``{error}`` when the
    optional evaluator is absent or the input is unsafe.
    """
    try:
        from lighteval.logging.evaluation_tracker import (  # noqa: PLC0415
            EvaluationTracker,
        )
        from lighteval.models.transformers.transformers_model import (  # noqa: PLC0415
            TransformersModelConfig,
        )
        from lighteval.pipeline import (  # noqa: PLC0415
            ParallelismManager,
            Pipeline,
            PipelineParameters,
        )
    except ImportError:  # pragma: no cover - without the extra
        return {"error": "LightEval not installed — install data-science-mcp[eval]"}

    if not tasks or any(not isinstance(task, str) or not task.strip() for task in tasks):
        return {"error": "at least one non-empty benchmark task is required"}
    if limit is not None and limit < 1:
        return {"error": "limit must be positive"}

    local_checkpoint = Path(model_path).expanduser().is_dir()
    if not local_checkpoint and not (
        revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
    ):
        return {
            "error": "remote model evaluation requires a 40-64 character immutable commit revision"
        }

    configured_batch = None if batch_size == "auto" else batch_size
    if not (configured_batch is None or isinstance(configured_batch, int)):
        return {"error": "batch_size must be a positive integer or 'auto'"}
    if isinstance(configured_batch, int) and configured_batch < 1:
        return {"error": "batch_size must be positive"}

    # LightEval keeps logs/details in the supplied directory. A private temporary
    # directory prevents model prompts or outputs from surviving the evaluation.
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as output_dir:
        tracker = EvaluationTracker(
            output_dir=output_dir,
            save_details=False,
            push_to_hub=False,
            push_to_tensorboard=False,
            use_wandb=False,
        )
        parameters = PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            max_samples=limit,
        )
        model = TransformersModelConfig(
            model_name=str(Path(model_path).expanduser()) if local_checkpoint else model_path,
            revision=revision or "main",  # ignored by Transformers for local paths
            batch_size=configured_batch,
            device=device or "cuda",
            trust_remote_code=False,
        )
        pipeline = Pipeline(  # pragma: no cover - heavyweight evaluator
            tasks=",".join(task.strip() for task in tasks),
            pipeline_parameters=parameters,
            evaluation_tracker=tracker,
            model_config=model,
        )
        pipeline.evaluate()
        output = pipeline.get_results() or {}
    return {"tasks": tasks, "results": output.get("results", {})}


def gsm8k_reward(prompt: str, completion: str, gold: Any) -> float:
    """Verifiable GSM8K reward: ``1.0`` if the final answer matches ``gold`` else ``0.0``.

    The PPO/GRPO ``reward_source="verifier"`` signal for math reasoning — exact-match
    on the ``<answer>…</answer>`` span (CONCEPT:DS-AHE.trainer.chat-format chat format), no reward model
    needed. ``prompt`` is unused but kept for the ``reward_fn(prompt, completion)``
    signature the trainers expect; bind ``gold`` per prompt via a closure.
    """
    from data_science_mcp.chat_template import answer_matches  # noqa: PLC0415

    return 1.0 if answer_matches(completion, gold) else 0.0


def evaluate_gsm8k(
    generate_fn: GenerateFn, cases: list[dict[str, Any]], *, limit: int | None = None
) -> dict[str, Any]:
    """Greedy-decode GSM8K cases and score exact-match accuracy (CONCEPT:DS-AHE.trainer.chat-format).

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

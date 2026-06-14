#!/usr/bin/python
"""End-to-end kernel specialization — the SAI factory's first live task (AHE-3.29).

Wires a registered :class:`~data_science_mcp.kernels.kernel_tasks.KernelTask` into
the ``agent_utilities`` SAI factory: the :class:`KernelVerifier` supplies the
machine-verifiable reward (correctness-gated speedup) and
:class:`SaiFactoryController` runs the closed scaffolding+weights loop to produce a
faster, correct kernel specialist — measured by adaptation speed (AHE-3.27).

This is the concrete live caller of the otherwise task-agnostic controller: the
``generate_fn`` is the candidate-kernel author (an LLM via an inference backend in
production; injectable for tests), and the optional ``weight_arm`` is the
harvest→fine-tune→serve path (Phase 2). Exposed for the data-science MCP surface
so a specialization run is invokable end-to-end.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_utilities.harness.sai_task import SpecializationTask
from agent_utilities.knowledge_graph.research.sai_factory import (
    FactoryResult,
    SaiFactoryController,
    WeightArm,
)

from data_science_mcp.kernels.kernel_tasks import get_kernel_task
from data_science_mcp.kernels.kernel_verifier import KernelVerifier

GenerateFn = Callable[[str], str]


def build_kernel_task(
    task_name: str,
    *,
    target_tau: float = 1.0,
    human_baseline: float | None = None,
    scaffolds: Sequence[str] | None = None,
    timeout_s: float = 30.0,
) -> SpecializationTask:
    """Build a :class:`SpecializationTask` backed by the kernel verifier."""
    kt = get_kernel_task(task_name)
    return SpecializationTask(
        task_id=f"kernel:{kt.name}",
        prompt_corpus=list(scaffolds) if scaffolds else [kt.spec],
        verifier=KernelVerifier(kt, timeout_s=timeout_s),
        target_tau=target_tau,
        human_baseline=human_baseline,
        metadata={"entrypoint": kt.entrypoint, "domain": "compute-kernel"},
    )


def run_kernel_specialization(
    task_name: str,
    generate_fn: GenerateFn,
    *,
    scaffolds: Sequence[str] | None = None,
    rounds: int = 3,
    weight_arm: WeightArm | None = None,
    target_tau: float = 1.0,
    tolerance: float = 0.0,
    timeout_s: float = 30.0,
) -> FactoryResult:
    """Specialize a kernel end-to-end and return the factory result + adaptation curve."""
    task = build_kernel_task(
        task_name, target_tau=target_tau, scaffolds=scaffolds, timeout_s=timeout_s
    )
    controller = SaiFactoryController(
        task,
        generate_fn,
        scaffolds=task.prompt_corpus,
        weight_arm=weight_arm,
        tolerance=tolerance,
    )
    return controller.run(rounds=rounds)

#!/usr/bin/python
"""End-to-end SAI kernel specialization (AHE-3.29) — real verifier + controller.

The genuine live path: SaiFactoryController drives a generator over scaffold
variants, scores each candidate with the real subprocess KernelVerifier, and
records an adaptation curve. Needs the dev ``agent_utilities`` on PYTHONPATH.
"""

from __future__ import annotations

import pytest

pytest.importorskip("agent_utilities.knowledge_graph.research.sai_factory", reason="dev AU")

from data_science_mcp.kernels.specialize import (  # noqa: E402
    build_kernel_task,
    run_kernel_specialization,
)

# A correct but slow softmax (python row loop) → lower speedup.
_SLOW = """
import numpy as np
def fused_softmax(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i]; z = row - row.max(); e = np.exp(z); out[i] = e / e.sum()
    return out
"""

# A correct, vectorized softmax → higher speedup.
_FAST = """
import numpy as np
def fused_softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
"""

# A wrong candidate → reward 0.
_WRONG = """
def fused_softmax(x):
    return x
"""


def _gen(mapping):
    def gen(scaffold: str) -> str:
        return mapping[scaffold]

    return gen


def test_build_kernel_task_uses_kernel_verifier():
    task = build_kernel_task("fused-softmax", target_tau=0.5)
    assert task.task_id == "kernel:fused-softmax"
    # the verifier scores a correct candidate as passing
    res = task.score(_FAST)
    assert res.passed is True
    assert res.reward > 0.0


def test_specialization_prefers_faster_correct_kernel():
    mapping = {"naive": _SLOW, "vectorized": _FAST}
    result = run_kernel_specialization(
        "fused-softmax",
        _gen(mapping),
        scaffolds=["naive", "vectorized"],
        rounds=1,
        target_tau=0.01,
    )
    # the vectorized scaffold yields the higher speedup → it wins
    assert result.specialist.scaffold == "vectorized"
    assert result.specialist.reward > 0.0
    assert result.curve.reached(0.01) is True


def test_specialization_ignores_wrong_candidate():
    mapping = {"wrong": _WRONG, "right": _FAST}
    result = run_kernel_specialization(
        "fused-softmax",
        _gen(mapping),
        scaffolds=["wrong", "right"],
        rounds=1,
        target_tau=0.01,
    )
    assert result.specialist.scaffold == "right"  # wrong earns reward 0
    metrics = result.metrics()
    assert metrics["final_specialist_reward"] > 0.0
    assert metrics["promotions"] >= 1

#!/usr/bin/python
"""Tests for the GPU/compute-kernel specialization verifier.

Verifies the machine-checkable reward contract end-to-end on CPU (numpy kernels):
a correct candidate earns correctness-gated speedup; wrong/broken/slow candidates
fail closed.
"""

from __future__ import annotations

import sys


from data_science_mcp.kernels import KernelVerifier, get_kernel_task  # noqa: E402

# A correct, vectorized fused-softmax candidate.
_CORRECT = """
import numpy as np
def fused_softmax(x):
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
"""

# Correct but deliberately slow (python row loop) — should still pass, lower speedup.
_SLOW_CORRECT = """
import numpy as np
def fused_softmax(x):
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i]
        z = row - row.max()
        e = np.exp(z)
        out[i] = e / e.sum()
    return out
"""

_WRONG = """
import numpy as np
def fused_softmax(x):
    return x  # not a softmax
"""

_NO_ENTRYPOINT = """
import numpy as np
def something_else(x):
    return x
"""

_BROKEN = "def fused_softmax(x): this is not valid python ::::"


def _verifier() -> KernelVerifier:
    return KernelVerifier(
        get_kernel_task("fused-softmax"),
        timeout_s=30.0,
        sandbox_command=[
            sys.executable,
            "-m",
            "data_science_mcp.kernels._runner",
            "{candidate}",
            "{task}",
        ],
    )


def test_correct_candidate_passes_with_positive_reward():
    res = _verifier().verify(_CORRECT)
    assert res.passed is True
    assert res.reward > 0.0
    assert res.detail["error"] is None
    assert res.detail["candidate_time"] is not None


def test_wrong_candidate_fails_with_zero_reward():
    res = _verifier().verify(_WRONG)
    assert res.passed is False
    assert res.reward == 0.0
    assert res.detail["error"] == "incorrect output"


def test_missing_entrypoint_fails_closed():
    res = _verifier().verify(_NO_ENTRYPOINT)
    assert res.passed is False
    assert res.reward == 0.0
    assert "missing entrypoint" in res.detail["error"]


def test_non_compiling_candidate_fails_closed():
    res = _verifier().verify(_BROKEN)
    assert res.passed is False
    assert res.reward == 0.0
    assert "compile/exec failed" in res.detail["error"]


def test_reward_cap_is_respected():
    v = KernelVerifier(
        get_kernel_task("fused-softmax"),
        timeout_s=30.0,
        speedup_cap=2.0,
        sandbox_command=[
            sys.executable,
            "-m",
            "data_science_mcp.kernels._runner",
            "{candidate}",
            "{task}",
        ],
    )
    res = v.verify(_CORRECT)
    if res.passed:
        assert res.reward <= 2.0

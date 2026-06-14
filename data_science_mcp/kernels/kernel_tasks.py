#!/usr/bin/python
"""GPU/compute-kernel specialization tasks for the SAI factory (AHE-3.28).

A :class:`KernelTask` is a machine-verifiable kernel-optimization problem: an
``entrypoint`` function name a candidate must define, a correct ``reference``
implementation (used both to compute the expected output and as the timing
baseline), a seeded ``make_inputs`` generator, and a natural-language ``spec``
prompt the specialization agent writes a candidate against.

The reference implementations are plain numpy, so a candidate is verified
*device-agnostically*: a numpy/torch-CPU candidate is fully checkable on CI with
no GPU, while a Triton/CUDA candidate exercises the same correctness+speedup
contract on a GPU host (it simply fails closed — ``passed=False, reward=0`` — when
its backend is unavailable). The reward is correctness-gated speedup vs the
reference, so an agent is rewarded for producing a *correct and faster* kernel —
the first concrete superhuman-able specialization the SAI factory targets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KernelTask:
    """A single machine-verifiable kernel-optimization problem."""

    name: str
    entrypoint: str  # function name the candidate source must define
    reference: Callable[..., Any]  # correct impl: expected output + timing baseline
    make_inputs: Callable[[np.random.Generator], tuple]  # one seeded input tuple
    spec: str  # natural-language prompt describing the kernel to write
    atol: float = 1e-4
    rtol: float = 1e-4
    n_batches: int = 4  # distinct random input batches checked per candidate
    seed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Reference implementations (numpy, correct-by-construction)
# --------------------------------------------------------------------------- #


def _softmax_ref(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _layernorm_ref(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * gamma + beta


def _matmul_ref(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b


# --------------------------------------------------------------------------- #
# Seeded input generators (candidate and reference receive identical inputs)
# --------------------------------------------------------------------------- #


def _softmax_inputs(rng: np.random.Generator) -> tuple:
    return (rng.standard_normal((64, 256)).astype(np.float32),)


def _layernorm_inputs(rng: np.random.Generator) -> tuple:
    x = rng.standard_normal((64, 256)).astype(np.float32)
    gamma = rng.standard_normal((256,)).astype(np.float32)
    beta = rng.standard_normal((256,)).astype(np.float32)
    return (x, gamma, beta)


def _matmul_inputs(rng: np.random.Generator) -> tuple:
    a = rng.standard_normal((128, 128)).astype(np.float32)
    b = rng.standard_normal((128, 128)).astype(np.float32)
    return (a, b)


FUSED_SOFTMAX = KernelTask(
    name="fused-softmax",
    entrypoint="fused_softmax",
    reference=_softmax_ref,
    make_inputs=_softmax_inputs,
    spec=(
        "Define `fused_softmax(x)` returning the row-wise softmax of a 2-D float32 "
        "array `x` (softmax over the last axis), numerically stable (subtract the "
        "row max). Make it as fast as possible while matching the reference within "
        "atol=1e-4."
    ),
)

LAYERNORM = KernelTask(
    name="layernorm",
    entrypoint="layernorm",
    reference=_layernorm_ref,
    make_inputs=_layernorm_inputs,
    spec=(
        "Define `layernorm(x, gamma, beta)` computing layer normalization over the "
        "last axis of a 2-D float32 array `x` with eps=1e-5, scaled by `gamma` and "
        "shifted by `beta`. Match the reference within atol=1e-4; maximize speed."
    ),
)

MATMUL = KernelTask(
    name="matmul",
    entrypoint="matmul",
    reference=_matmul_ref,
    make_inputs=_matmul_inputs,
    spec=(
        "Define `matmul(a, b)` returning the matrix product of two 2-D float32 "
        "arrays. Match the reference within atol=1e-4; maximize throughput."
    ),
)

KERNEL_TASKS: dict[str, KernelTask] = {t.name: t for t in (FUSED_SOFTMAX, LAYERNORM, MATMUL)}


def get_kernel_task(name: str) -> KernelTask:
    """Look up a registered kernel task by name (raises ``KeyError`` if unknown)."""
    return KERNEL_TASKS[name]

#!/usr/bin/python
"""Kernel-optimization specialization tasks + verifier for the SAI factory.

AHE-3.28 — the first concrete machine-verifiable specialization domain:
correctness-gated, speedup-rewarded compute kernels. See :mod:`kernel_tasks`
(the task suite) and :mod:`kernel_verifier` (the ``Verifier``-protocol reward).
"""

from .kernel_tasks import KERNEL_TASKS, KernelTask, get_kernel_task
from .kernel_verifier import KernelVerifier

__all__ = ["KERNEL_TASKS", "KernelTask", "KernelVerifier", "get_kernel_task"]

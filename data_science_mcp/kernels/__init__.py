#!/usr/bin/python
"""Kernel-optimization specialization tasks + verifier .

Machine-verifiable specialization domain:
correctness-gated, speedup-rewarded compute kernels. See :mod:`kernel_tasks`
(the task suite) and :mod:`kernel_verifier` (the sandboxed reward).
"""

from .kernel_tasks import KERNEL_TASKS, KernelTask, get_kernel_task
from .kernel_verifier import KernelVerifier, VerifierResult

__all__ = ["KERNEL_TASKS", "KernelTask", "KernelVerifier", "VerifierResult", "get_kernel_task"]

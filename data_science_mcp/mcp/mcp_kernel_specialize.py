#!/usr/bin/python
"""MCP tool exposing the SAI factory's kernel-specialization track (AHE-3.29).

Makes the compute-kernel specialization domain reachable through the gateway
(``dsm__*`` via the multiplexer): given a kernel task, author candidate kernels
with the configured inference backend, score them with the machine-verifiable
``KernelVerifier`` (correctness-gated speedup), and run the SAI factory's closed
loop — returning adaptation-speed metrics. When no inference backend is configured
it returns the task spec + a ``requires`` note instead of failing (plan-first,
mirroring the trainer tools).
"""

from __future__ import annotations

import re
from typing import Any

from fastmcp import FastMCP
from pydantic import Field


def _extract_code(text: str) -> str:
    """Pull a python code block from an LLM completion (or return the raw text)."""
    m = re.search(r"```(?:python)?\n(.*?)```", text or "", re.DOTALL)
    return m.group(1) if m else (text or "")


def register_kernel_specialize_tools(mcp: FastMCP) -> None:
    """Register the SAI-factory kernel specialization tool (tag ``sai-factory``)."""

    @mcp.tool(tags={"sai-factory"})
    async def ds_specialize_kernel(
        task_name: str = Field(
            default="fused-softmax",
            description="Kernel task: 'fused-softmax' | 'layernorm' | 'matmul'.",
        ),
        rounds: int = Field(
            default=3, description="Factory rounds (scaffold-search iterations)."
        ),
        target_tau: float = Field(
            default=1.0,
            description="Reward target (speedup) defining 'specialized enough'.",
        ),
    ) -> dict[str, Any]:
        """Run a SAI-factory specialization cycle on a compute kernel (CONCEPT:AHE-3.29).

        Authors candidate kernels with the configured inference backend, scores each
        by correctness-gated speedup (``KernelVerifier``), and runs the closed loop;
        returns adaptation-speed metrics (time-to-target, sample-complexity). Returns
        a plan + ``requires`` note when no inference backend is configured.
        """
        from data_science_mcp.inference import (
            create_inference_backend,
            inference_backend_configured,
        )
        from data_science_mcp.kernels import KERNEL_TASKS, get_kernel_task
        from data_science_mcp.kernels.specialize import run_kernel_specialization

        if task_name not in KERNEL_TASKS:
            return {
                "status": "error",
                "error": f"unknown kernel task '{task_name}'",
                "available": list(KERNEL_TASKS),
            }
        kt = get_kernel_task(task_name)
        if not inference_backend_configured():
            return {
                "status": "plan",
                "task": task_name,
                "entrypoint": kt.entrypoint,
                "spec": kt.spec,
                "requires": "set INFERENCE_BASE_URL (a vLLM/SGLang server) to author candidate kernels",
            }

        backend = create_inference_backend()

        def generate(scaffold: str) -> str:
            out = backend.generate(
                f"{scaffold}\n\nReturn ONLY a Python code block defining the function.",
                n=1,
                max_tokens=1024,
                temperature=0.7,
            )
            return _extract_code(out[0]["text"]) if out else ""

        result = run_kernel_specialization(
            task_name, generate, rounds=rounds, target_tau=target_tau
        )
        return {"status": "ok", **result.metrics()}

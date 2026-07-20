#!/usr/bin/python
"""Tests for the gateway-reachable kernel-specialization MCP tool (AHE-3.29)."""

from __future__ import annotations

import json
import sys

import pytest

pytest.importorskip(
    "agent_utilities.knowledge_graph.research.sai_factory", reason="dev agent-utilities"
)

from data_science_mcp.mcp.mcp_kernel_specialize import (  # noqa: E402
    register_kernel_specialize_tools,
)


class _CaptureMCP:
    def __init__(self) -> None:
        self.tools: dict = {}

    def tool(self, *_a, **_k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn

        return deco


def _tool():
    m = _CaptureMCP()
    register_kernel_specialize_tools(m)
    return m.tools["ds_specialize_kernel"]


@pytest.mark.asyncio
async def test_plan_mode_without_inference(monkeypatch):
    import data_science_mcp.inference as inf

    monkeypatch.setattr(inf, "inference_backend_configured", lambda: False)
    out = await _tool()(task_name="fused-softmax")
    assert out["status"] == "plan"
    assert "spec" in out and out["entrypoint"] == "fused_softmax"


@pytest.mark.asyncio
async def test_unknown_task_errors():
    out = await _tool()(task_name="does-not-exist")
    assert out["status"] == "error"
    assert "fused-softmax" in out["available"]


@pytest.mark.asyncio
async def test_run_with_mocked_backend(monkeypatch):
    import data_science_mcp.inference as inf

    _CORRECT = (
        "```python\n"
        "import numpy as np\n"
        "def fused_softmax(x):\n"
        "    z = x - x.max(axis=-1, keepdims=True)\n"
        "    e = np.exp(z)\n"
        "    return e / e.sum(axis=-1, keepdims=True)\n"
        "```"
    )

    class _Backend:
        def generate(self, _prompt, **_k):
            return [{"text": _CORRECT, "logprobs": []}]

    monkeypatch.setattr(inf, "inference_backend_configured", lambda: True)
    monkeypatch.setattr(inf, "create_inference_backend", lambda *a, **k: _Backend())
    monkeypatch.setenv(
        "DATA_SCIENCE_KERNEL_SANDBOX_COMMAND",
        json.dumps(
            [
                sys.executable,
                "-m",
                "data_science_mcp.kernels._runner",
                "{candidate}",
                "{task}",
            ]
        ),
    )

    out = await _tool()(task_name="fused-softmax", rounds=1, target_tau=0.01)
    assert out["status"] == "ok"
    assert out["final_specialist_reward"] > 0.0

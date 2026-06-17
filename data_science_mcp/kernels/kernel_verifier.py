#!/usr/bin/python
"""Machine-verifiable kernel reward for the SAI factory (AHE-3.28).

:class:`KernelVerifier` satisfies the ``agent_utilities.harness.sai_task.Verifier``
protocol: given a candidate kernel source string it returns a
:class:`~agent_utilities.harness.sai_task.VerifierResult` whose ``reward`` is the
correctness-gated speedup over the task reference (``reward = speedup`` when the
output is correct, ``0.0`` otherwise) — a real, comparable signal the SAI factory
both optimizes (adaptation-speed curve, AHE-3.27) and distills into training data
(OS-5.34 / AHE-3.25).

Candidate code is executed in an **isolated subprocess** (``_runner``) with a wall
timeout, so a miscompiling, crashing, or hanging candidate fails closed instead of
taking down the verifier. This is the Phase-1 isolation tier; the RLM sandbox
(ORCH-1.38: monty/wasm/docker) is the escalation path for untrusted-at-scale runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from agent_utilities.harness.sai_task import VerifierResult

from data_science_mcp.kernels.kernel_tasks import KernelTask


class KernelVerifier:
    """Verify a kernel candidate by correctness + measured speedup vs the reference."""

    def __init__(
        self,
        task: KernelTask,
        *,
        timeout_s: float = 30.0,
        speedup_cap: float = 50.0,
        python_exe: str | None = None,
    ) -> None:
        self.task = task
        self.timeout_s = timeout_s
        self.speedup_cap = speedup_cap
        self.python_exe = python_exe or sys.executable

    def verify(self, candidate: str) -> VerifierResult:
        """Return correctness-gated speedup as the reward for one candidate."""
        with tempfile.TemporaryDirectory(prefix="sai-kernel-") as tmp:
            cand_path = Path(tmp) / "candidate.py"
            cand_path.write_text(candidate or "", encoding="utf-8")
            try:
                proc = subprocess.run(
                    [self.python_exe, "-m", "data_science_mcp.kernels._runner",
                     str(cand_path), self.task.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return VerifierResult(
                    reward=0.0, passed=False,
                    detail={"task": self.task.name, "error": "timeout", "timeout_s": self.timeout_s},
                )

        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            return VerifierResult(
                reward=0.0, passed=False,
                detail={"task": self.task.name, "error": "no runner output",
                        "stderr": (proc.stderr or "")[-500:]},
            )
        try:
            data = json.loads(raw[-1])
        except (ValueError, IndexError):
            return VerifierResult(
                reward=0.0, passed=False,
                detail={"task": self.task.name, "error": "unparsable runner output",
                        "stdout": raw[-1][:500]},
            )

        passed = bool(data.get("passed"))
        speedup = float(data.get("speedup") or 0.0)
        reward = min(speedup, self.speedup_cap) if passed else 0.0
        return VerifierResult(
            reward=reward,
            passed=passed,
            detail={
                "task": self.task.name,
                "speedup": speedup,
                "candidate_time": data.get("candidate_time"),
                "reference_time": data.get("reference_time"),
                "error": data.get("error"),
            },
        )

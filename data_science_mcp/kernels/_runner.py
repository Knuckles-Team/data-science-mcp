#!/usr/bin/python
"""Isolated subprocess harness for verifying one kernel candidate.

AHE-3.28 — runs untrusted candidate kernel source in a *separate process*
(so a crash, hang, or compile error cannot take down the verifier), checks
correctness against the task reference over several seeded input batches, times
candidate vs reference, and emits a single JSON line on stdout:

    {"passed": bool, "speedup": float, "candidate_time": float,
     "reference_time": float, "error": str|null}

Invoked as ``python -m data_science_mcp.kernels._runner <candidate_path> <task_name>``.
All heavy/optional kernel backends (triton, torch) are imported only if the
*candidate* imports them; this harness itself needs only numpy + the task module.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from data_science_mcp.kernels.kernel_tasks import get_kernel_task

_REPEATS = 5


def _time_call(fn: Any, args: tuple, repeats: int) -> tuple[Any, float]:
    """Return (last_output, best_seconds) over ``repeats`` timed calls."""
    out = None
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return out, best


def _run(candidate_path: str, task_name: str) -> dict[str, Any]:
    task = get_kernel_task(task_name)

    src = open(candidate_path, encoding="utf-8").read()  # noqa: SIM115 — short-lived
    namespace: dict[str, Any] = {}
    try:
        exec(compile(src, "<candidate>", "exec"), namespace)  # noqa: S102 — sandboxed subprocess
    except Exception as exc:  # noqa: BLE001 — any failure ⇒ non-compiling candidate
        return {"passed": False, "speedup": 0.0, "candidate_time": 0.0,
                "reference_time": 0.0, "error": f"compile/exec failed: {exc!r}"}

    fn = namespace.get(task.entrypoint)
    if not callable(fn):
        return {"passed": False, "speedup": 0.0, "candidate_time": 0.0,
                "reference_time": 0.0, "error": f"missing entrypoint {task.entrypoint!r}"}

    rng = np.random.default_rng(task.seed)
    cand_best = float("inf")
    ref_best = float("inf")
    try:
        for _ in range(task.n_batches):
            args = task.make_inputs(rng)
            expected, ref_dt = _time_call(task.reference, args, _REPEATS)
            got, cand_dt = _time_call(fn, args, _REPEATS)
            got_arr = np.asarray(got, dtype=np.float64)
            exp_arr = np.asarray(expected, dtype=np.float64)
            if got_arr.shape != exp_arr.shape or not np.allclose(
                got_arr, exp_arr, atol=task.atol, rtol=task.rtol
            ):
                return {"passed": False, "speedup": 0.0, "candidate_time": cand_dt,
                        "reference_time": ref_dt, "error": "incorrect output"}
            cand_best = min(cand_best, cand_dt)
            ref_best = min(ref_best, ref_dt)
    except Exception as exc:  # noqa: BLE001 — runtime failure ⇒ candidate fails closed
        return {"passed": False, "speedup": 0.0, "candidate_time": 0.0,
                "reference_time": 0.0, "error": f"runtime failed: {exc!r}"}

    speedup = (ref_best / cand_best) if cand_best > 0 else 0.0
    return {"passed": True, "speedup": speedup, "candidate_time": cand_best,
            "reference_time": ref_best, "error": None}


def main() -> int:
    if len(sys.argv) != 3:
        print(json.dumps({"passed": False, "speedup": 0.0, "error": "usage: _runner <path> <task>"}))
        return 2
    result = _run(sys.argv[1], sys.argv[2])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

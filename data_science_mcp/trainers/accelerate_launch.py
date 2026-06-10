#!/usr/bin/python
"""Distributed/precision preparation via 🤗 Accelerate (CONCEPT:ML-005).

A thin seam between the trainers and `accelerate.Accelerator`, supporting **FSDP**
and **DeepSpeed ZeRO-3 as equal first-class peers** (user-selected). It is a no-op
when ``config.distributed == "none"`` (the default, and the only path the CPU/toy
tests take), so `accelerate` is never imported unless a real distributed run asks
for it.

When engaged it builds an ``Accelerator`` with the right plugin + mixed precision
and prepares ``(model, optimizer)``; the shared :func:`run_loop` then routes
``backward``/``step``/clipping through the accelerator instead of a ``GradScaler``,
keeping a single correct code path for both single-GPU AMP and multi-GPU sharding.

The multi-process *launch* (how many ranks/nodes) is owned by
:mod:`data_science_mcp.launch` (``accelerate launch`` / ``torchrun`` configs); this
module only prepares the objects inside an already-launched process.

Concept: accelerate-launch
"""

from __future__ import annotations

from typing import Any


def wants_accelerate(config: Any) -> bool:
    """True when the config requests a distributed/sharded backend."""
    return (getattr(config, "distributed", "none") or "none").lower() != "none"


def _mixed_precision(config: Any) -> str:
    prec = (getattr(config, "precision", "fp32") or "fp32").lower()
    return {"fp16": "fp16", "bf16": "bf16"}.get(prec, "no")


def build_accelerator(config: Any) -> Any:
    """Construct an ``Accelerator`` with the FSDP or DeepSpeed plugin from config.

    Raises ``RuntimeError`` (not ``ImportError``) with an actionable hint when the
    scale extra is missing, so the failure is legible in an agent transcript.
    """
    distributed = (getattr(config, "distributed", "none") or "none").lower()
    try:
        from accelerate import Accelerator  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - without the extra
        raise RuntimeError(
            "accelerate is required for distributed training; install "
            "`data-science-mcp[training]`"
        ) from e

    mp = _mixed_precision(config)
    kwargs: dict[str, Any] = {
        "mixed_precision": mp,
        "gradient_accumulation_steps": max(1, getattr(config, "grad_accum", 1)),
    }

    if distributed == "fsdp":
        try:
            from accelerate import FullyShardedDataParallelPlugin  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("accelerate too old for FSDP plugin") from e
        kwargs["fsdp_plugin"] = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            cpu_offload=bool(getattr(config, "cpu_offload", False)),
        )
    elif distributed == "deepspeed":
        try:
            from accelerate.utils import DeepSpeedPlugin  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - needs deepspeed
            raise RuntimeError(
                "deepspeed is required for ZeRO; install "
                "`data-science-mcp[training-scale]`"
            ) from e
        kwargs["deepspeed_plugin"] = DeepSpeedPlugin(
            zero_stage=int(getattr(config, "zero_stage", 3)),
            offload_optimizer_device="cpu"
            if getattr(config, "cpu_offload", False)
            else "none",
            gradient_accumulation_steps=max(1, getattr(config, "grad_accum", 1)),
        )
    else:  # pragma: no cover - guarded by wants_accelerate upstream
        raise ValueError(f"unknown distributed backend: {distributed!r}")

    return Accelerator(**kwargs)


def prepare(config: Any, model: Any, optimizer: Any) -> tuple[Any, Any, Any]:
    """Return ``(accelerator | None, model, optimizer)``.

    ``None`` accelerator means the plain single-device path (toy/CPU/single-GPU AMP
    via the run-loop's ``GradScaler``); otherwise the model/optimizer are wrapped.
    """
    if not wants_accelerate(config):
        return None, model, optimizer
    acc = build_accelerator(config)
    model, optimizer = acc.prepare(model, optimizer)
    return acc, model, optimizer


__all__ = ["wants_accelerate", "build_accelerator", "prepare"]

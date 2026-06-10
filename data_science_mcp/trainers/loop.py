#!/usr/bin/python
"""Shared optimisation loop for every trainer (CONCEPT:ML-001).

The SFT/DPO/GRPO (and pretrain) trainers differ only in their per-step **loss**;
the surrounding machinery — gradient accumulation, mixed precision, gradient
clipping, LR scheduling, checkpoint save/resume, metric emission — is identical.
:func:`run_loop` owns all of it once, so each trainer is reduced to a ``compute_loss``
closure plus an item iterator.

Two execution paths, selected automatically:

* **plain** (default / CPU / single-GPU AMP) — a ``GradScaler`` handles fp16; bf16
  and fp32 run with the scaler disabled. This is the path the toy-model tests take.
* **accelerate** — when an ``accelerator`` is passed (FSDP/DeepSpeed), backward,
  gradient clipping and the optimizer step are routed through it instead.

**Backward-compatibility contract:** with the default config (``precision="fp32"``,
``grad_accum=1``, ``lr_scheduler="constant"``, ``max_grad_norm=None``,
``save_steps=0``, no accelerator, no tracker) this loop is behaviourally identical
to the original hand-rolled per-batch loops — one optimizer step per item, the raw
per-item loss recorded — so existing smoke tests are unaffected.

Concept: train-loop
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Iterable

from data_science_mcp.trainers.base import _torch


def _grad_scaler(enabled: bool) -> Any:
    """Construct a GradScaler across torch 2.4→2.6 API variants."""
    torch = _torch()
    try:  # torch>=2.4 unified API
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _cur_lr(optimizer: Any) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except (IndexError, KeyError, TypeError):  # pragma: no cover - defensive
        return 0.0


def _trainable(model: Any) -> list[Any]:
    return [p for p in model.parameters() if p.requires_grad]


def build_scheduler(optimizer: Any, config: Any, num_training_steps: int) -> Any | None:
    """LR scheduler from config, or ``None`` for the constant-LR default.

    ``lr_scheduler="constant"`` (default) returns ``None`` → fixed ``config.lr``,
    matching the original trainers. Other names use HF ``get_scheduler`` (cosine,
    linear, …) with ``warmup_steps``.
    """
    name = (getattr(config, "lr_scheduler", "constant") or "constant").lower()
    warmup = int(getattr(config, "warmup_steps", 0) or 0)
    if name in ("", "constant") and warmup == 0:
        return None
    try:
        from transformers import get_scheduler  # noqa: PLC0415
    except ImportError:  # pragma: no cover - without the extra
        return None
    return get_scheduler(
        name if name not in ("", "constant") else "constant",
        optimizer=optimizer,
        num_warmup_steps=warmup,
        num_training_steps=max(1, num_training_steps),
    )


# --- checkpointing ---------------------------------------------------------- #
def _save_checkpoint(
    model: Any, optimizer: Any, scheduler: Any, config: Any, step: int
) -> str | None:
    """Write a resumable checkpoint dir; return its path (or ``None`` if no dir).

    Saves a generic ``model_state.pt`` (works for any ``nn.Module`` incl. the toy
    model) and, when available, an HF ``save_pretrained`` snapshot, plus optimizer/
    scheduler state and a ``training_state.json`` carrying the step counter.
    """
    torch = _torch()
    out = getattr(config, "output_dir", "") or ""
    if not out:
        return None
    ckpt = os.path.join(out, f"checkpoint-{step}")
    os.makedirs(ckpt, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt, "model_state.pt"))
    if hasattr(model, "save_pretrained"):
        try:  # pragma: no cover - HF models only
            model.save_pretrained(ckpt)
        except Exception:
            pass
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(ckpt, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(ckpt, "scheduler.pt"))
    with open(os.path.join(ckpt, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step}, f)
    return ckpt


def _enforce_limit(checkpoints: list[str], limit: int) -> None:
    """Keep only the newest ``limit`` checkpoint dirs (best-effort delete)."""
    import shutil  # noqa: PLC0415

    if limit and len(checkpoints) > limit:
        for stale in checkpoints[:-limit]:
            try:  # pragma: no cover - filesystem
                shutil.rmtree(stale, ignore_errors=True)
            except Exception:
                pass
        del checkpoints[:-limit]


def maybe_resume(model: Any, optimizer: Any, scheduler: Any, config: Any) -> int:
    """Restore weights/optimizer/scheduler/step from ``config.resume_from``.

    Returns the step to resume *from* (0 when not resuming). Loads the generic
    ``model_state.pt`` so it works for the toy model as well as HF checkpoints.
    """
    torch = _torch()
    path = getattr(config, "resume_from", None)
    if not path or not os.path.isdir(path):
        return 0
    msf = os.path.join(path, "model_state.pt")
    if os.path.isfile(msf):
        try:
            model.load_state_dict(torch.load(msf, map_location="cpu"))
        except Exception:  # pragma: no cover - shape drift
            pass
    osf = os.path.join(path, "optimizer.pt")
    if optimizer is not None and os.path.isfile(osf):
        try:
            optimizer.load_state_dict(torch.load(osf, map_location="cpu"))
        except Exception:  # pragma: no cover
            pass
    ssf = os.path.join(path, "scheduler.pt")
    if scheduler is not None and os.path.isfile(ssf):
        try:
            scheduler.load_state_dict(torch.load(ssf, map_location="cpu"))
        except Exception:  # pragma: no cover
            pass
    tsf = os.path.join(path, "training_state.json")
    if os.path.isfile(tsf):
        try:
            with open(tsf, encoding="utf-8") as f:
                return int(json.load(f).get("step", 0))
        except Exception:  # pragma: no cover
            return 0
    return 0


# --- the loop --------------------------------------------------------------- #
def run_loop(
    *,
    config: Any,
    model: Any,
    optimizer: Any,
    device: Any,
    epoch_items: Callable[[], Iterable[Any]],
    compute_loss: Callable[[Any], Any],
    scheduler: Any | None = None,
    accelerator: Any | None = None,
    tracker: Any | None = None,
    total_steps: int | None = None,
) -> dict[str, Any]:
    """Drive training; return ``{steps, losses, checkpoints, final_lr, resumed_from_step}``.

    Args:
        config: a ``TrainConfig`` (precision/grad_accum/clip/save/resume/max_steps).
        model/optimizer/device: prepared by the caller (incl. ``.to(device)``/``train()``).
        epoch_items: returns a **fresh** iterable of work items per epoch (batches or
            GRPO groups). Generators are fine — it is re-invoked each epoch.
        compute_loss: ``item -> scalar loss tensor`` (runs the forward; wrapped in
            autocast here for the plain AMP path). Side-channel metrics (e.g. KL) are
            recorded by the closure itself.
        scheduler/accelerator/tracker: optional; see module docstring.
        total_steps: planned optimizer steps (for tracker context only).
    """
    torch = _torch()
    accum = max(1, int(getattr(config, "grad_accum", 1) or 1))
    precision = (getattr(config, "precision", "fp32") or "fp32").lower()
    max_grad_norm = getattr(config, "max_grad_norm", None)
    save_steps = int(getattr(config, "save_steps", 0) or 0)
    save_limit = int(getattr(config, "save_total_limit", 0) or 0)
    max_steps = getattr(config, "max_steps", None)

    use_amp = precision in ("fp16", "bf16") and getattr(device, "type", "cpu") == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    use_scaler = precision == "fp16" and accelerator is None and use_amp
    scaler = _grad_scaler(enabled=True) if use_scaler else None

    start_step = maybe_resume(model, optimizer, scheduler, config)
    losses: list[float] = []
    checkpoints: list[str] = []
    step = 0
    micro = 0
    stop = False
    optimizer.zero_grad(set_to_none=True)

    def _forward(item: Any) -> Any:
        if accelerator is None and use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                return compute_loss(item)
        return compute_loss(item)

    def _optim_step() -> None:
        if accelerator is not None:
            if max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(_trainable(model), max_grad_norm)
            optimizer.step()
        elif use_scaler:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(_trainable(model), max_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    for _epoch in range(max(1, int(getattr(config, "epochs", 1) or 1))):
        for item in epoch_items():
            loss = _forward(item)
            raw = float(loss.detach())
            losses.append(raw)
            scaled = loss / accum
            if accelerator is not None:
                accelerator.backward(scaled)
            elif use_scaler:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            micro += 1
            if micro % accum == 0:
                _optim_step()
                step += 1
                if tracker is not None:
                    tracker.log_metrics(
                        {"loss": raw, "lr": _cur_lr(optimizer)}, step=start_step + step
                    )
                if save_steps and step % save_steps == 0:
                    ck = _save_checkpoint(
                        model, optimizer, scheduler, config, start_step + step
                    )
                    if ck:
                        checkpoints.append(ck)
                        _enforce_limit(checkpoints, save_limit)
                if max_steps is not None and step >= max_steps:
                    stop = True
                    break
        if stop:
            break

    # Flush a trailing partial accumulation window so no gradient work is lost.
    if micro % accum != 0:
        _optim_step()
        step += 1

    return {
        "steps": step,
        "losses": losses,
        "checkpoints": checkpoints,
        "final_lr": _cur_lr(optimizer),
        "resumed_from_step": start_step,
    }


__all__ = ["run_loop", "build_scheduler", "maybe_resume"]

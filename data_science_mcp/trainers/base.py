#!/usr/bin/python
"""Trainer base + shared config (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

:class:`TrainerBase` is the common spine for the SFT/DPO/GRPO trainers. It holds
the :class:`TrainConfig`, exposes a **pure** :meth:`plan` (step/batch accounting
with no torch, so it is callable and testable without the heavy extra), and the
shared plumbing the concrete trainers reuse: device selection, mini-batching,
model/tokenizer resolution (real HF path via :class:`PeftManager`, or
dependency-injected toy objects for CPU smoke tests), and padded encoding.

Each concrete trainer implements the :class:`data_science_mcp.training_data.Trainer`
Protocol (``name`` + ``train``); registering the resulting checkpoint via the model
registry role seam makes it live with no hot-path change.

Concept: trainer-base
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

from data_science_mcp.peft_manager import LoraSpec, PeftManager


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as e:  # pragma: no cover - without the extra
        raise RuntimeError(
            "torch is required to train; install `data-science-mcp[training]`"
        ) from e


@dataclass
class TrainConfig:
    """Hyper-parameters shared by every trainer.

    Defaults reproduce the original single-GPU, constant-LR, fp32 behaviour exactly
    (so existing smoke tests are unaffected); the robustness/scale features below
    activate only when set away from their defaults.
    """

    base_model: str = "toy"
    model_revision: str | None = None
    output_dir: str = ""
    epochs: int = 1
    lr: float = 2e-4
    batch_size: int = 4
    max_steps: int | None = None
    max_seq_len: int = 512
    grad_accum: int = 1
    seed: int = 0
    device: str | None = None
    lora: LoraSpec | None = None
    # DPO
    beta: float = 0.1
    # GRPO / PPO (clip_eps + kl_coef are shared: the clipped-surrogate ε and the
    # KL-to-reference penalty coefficient)
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    group_size: int = 4
    # PPO (CONCEPT:DS-AHE.trainer.per-token-value)
    vf_coef: float = 0.5  # value-loss weight in the PPO total
    gamma: float = 1.0  # GAE discount (1.0 = undiscounted, typical for LM RL)
    gae_lambda: float = 0.95  # GAE λ
    value_clip: float | None = None  # PPO value-clipping range (None = unclipped MSE)
    reward_source: str = "verifier"  # verifier | reward_model (PPO reward signal)
    # --- robustness / throughput (CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer) ---------------------------
    precision: str = "fp32"  # fp32 | fp16 | bf16 → AMP autocast (+ GradScaler on fp16)
    gradient_checkpointing: bool = False
    max_grad_norm: float | None = None  # gradient clipping
    warmup_steps: int = 0
    lr_scheduler: str = (
        "constant"  # constant | linear | cosine | ... (HF get_scheduler)
    )
    resume_from: str | None = None  # path to a checkpoint-N dir
    save_steps: int = 0  # 0 = save only at end; >0 = periodic checkpoints
    save_total_limit: int = 0  # 0 = keep all; >0 = keep newest N
    attn_impl: str | None = None  # e.g. "flash_attention_2"
    use_liger: bool = False  # fuse kernels via liger-kernel when available
    pack_sequences: bool = False  # concat+split for pretrain throughput
    # --- scale-out (CONCEPT:DS-AHE.trainer.concept-4) -----------------------------------------
    distributed: str = "none"  # none | fsdp | deepspeed  (equal first-class peers)
    cpu_offload: bool = False
    zero_stage: int = 3  # DeepSpeed ZeRO stage when distributed="deepspeed"
    # --- tracking (CONCEPT:DS-AHE.trainer.concept-3) ------------------------------------------
    tracker: str = "none"  # none | mlflow | wandb
    run_name: str | None = None
    kg_log: bool = False  # mirror the run into the epistemic-graph as a TrainingRun
    # provenance lineage (PROV-O): the curated DatasetVersion this run consumed and
    # the upstream run it continues (e.g. PPO ← reward ← sft ← pretrain). Emitted on
    # the TrainingRun node as `was_derived_from` so the dataset→…→model chain is
    # queryable via /sparql + OWL transitive closure (CONCEPT:AU-AHE.trainer.join-inference lineage).
    dataset_version: str | None = None
    parent_run: str | None = None


class TrainerBase(ABC):
    """Shared trainer machinery; subclasses set ``name``/``kind`` and ``train``."""

    name: str = "base"
    kind: str = "sft"  # dataset kind this trainer consumes

    def __init__(self, config: TrainConfig | None = None) -> None:
        self.config = config or TrainConfig()

    # --- pure planning (no torch) ------------------------------------------
    def plan(self, dataset: list[dict[str, Any]]) -> dict[str, Any]:
        """Step/batch accounting for ``dataset`` — pure, no torch required."""
        n = len(dataset)
        bs = max(1, self.config.batch_size)
        steps_per_epoch = math.ceil(n / bs) if n else 0
        total = steps_per_epoch * max(1, self.config.epochs)
        if self.config.max_steps is not None:
            total = min(total, self.config.max_steps)
        return {
            "trainer": self.name,
            "kind": self.kind,
            "examples": n,
            "batch_size": bs,
            "grad_accum": self.config.grad_accum,
            "effective_batch": bs * max(1, self.config.grad_accum),
            "epochs": self.config.epochs,
            "planned_steps": total,
            "base_model": self.config.base_model,
            "lora": self.config.lora is not None,
            "lr": self.config.lr,
        }

    # --- shared plumbing ----------------------------------------------------
    def _device(self) -> Any:
        torch = _torch()
        if self.config.device:
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve(self, model: Any | None, tokenizer: Any | None) -> tuple[Any, Any]:
        """Return ``(model, tokenizer)`` — injected if given, else loaded via HF."""
        if model is not None and tokenizer is not None:
            return model, tokenizer
        try:
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "transformers is required to load a tokenizer; install "
                "`data-science-mcp[training]`"
            ) from e
        from data_science_mcp.hf_security import require_pinned_revision

        revision = require_pinned_revision(
            self.config.base_model, self.config.model_revision
        )
        pm = PeftManager(
            self.config.base_model,
            self.config.lora,
            revision=revision,
        )
        loaded = pm.attach() if self.config.lora is not None else pm.load_base()
        tok = tokenizer or AutoTokenizer.from_pretrained(
            self.config.base_model,
            revision=revision,
            trust_remote_code=False,
        )
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        return (model or loaded), tok

    def _batches(
        self, items: list[Any], batch_size: int | None = None
    ) -> Iterator[list[Any]]:
        bs = batch_size or self.config.batch_size
        for i in range(0, len(items), bs):
            yield items[i : i + bs]

    def _encode(self, tokenizer: Any, texts: list[str]) -> Any:
        """Padded/truncated batch encode → BatchEncoding-like with tensors."""
        return tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_len,
        )

    def _optimizer(self, model: Any, optimizer: Any | None) -> Any:
        torch = _torch()
        if optimizer is not None:
            return optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.config.lr)

    # --- robustness / scale hooks (no-ops at default config) ----------------
    def _enable_runtime(self, model: Any) -> None:
        """Apply gradient checkpointing when requested (HF models only)."""
        if self.config.gradient_checkpointing and hasattr(
            model, "gradient_checkpointing_enable"
        ):
            if getattr(model, "config", None) is not None:
                model.config.use_cache = False  # required with checkpointing
            model.gradient_checkpointing_enable()

    def _scheduler(self, optimizer: Any, total_steps: int) -> Any | None:
        """Build the LR scheduler (``None`` for the constant-LR default)."""
        from data_science_mcp.trainers.loop import build_scheduler  # noqa: PLC0415

        return build_scheduler(optimizer, self.config, total_steps)

    def _prepare(self, model: Any, optimizer: Any) -> tuple[Any, Any, Any]:
        """Prepare for FSDP/DeepSpeed; returns ``(accelerator|None, model, optimizer)``."""
        from data_science_mcp.trainers.accelerate_launch import prepare  # noqa: PLC0415

        return prepare(self.config, model, optimizer)

    def _tracker(self, params: dict[str, Any] | None = None) -> Any:
        """Build + start the experiment tracker (no-op at ``tracker='none'``)."""
        from data_science_mcp.tracking import RunTracker  # noqa: PLC0415

        return RunTracker.from_config(self.config, params=params).start()

    def _total_opt_steps(self, items_per_epoch: int) -> int:
        """Planned optimizer updates (accounts for grad accumulation + max_steps)."""
        accum = max(1, self.config.grad_accum)
        per = math.ceil(items_per_epoch / accum) if items_per_epoch else 0
        total = per * max(1, self.config.epochs)
        if self.config.max_steps is not None:
            total = min(total, self.config.max_steps)
        return max(1, total)

    @abstractmethod
    def train(self, dataset: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Run the optimisation loop over ``dataset``; concrete trainers implement."""
        ...


__all__ = ["TrainConfig", "TrainerBase"]

#!/usr/bin/python
"""Trainer base + shared config (CONCEPT:AHE-3.1).

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
    """Hyper-parameters shared by every trainer."""

    base_model: str = "toy"
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
    # GRPO
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    group_size: int = 4


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
        pm = PeftManager(self.config.base_model, self.config.lora)
        loaded = pm.attach() if self.config.lora is not None else pm.load_base()
        tok = tokenizer or AutoTokenizer.from_pretrained(self.config.base_model)
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

    @abstractmethod
    def train(self, dataset: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Run the optimisation loop over ``dataset``; concrete trainers implement."""
        ...


__all__ = ["TrainConfig", "TrainerBase"]

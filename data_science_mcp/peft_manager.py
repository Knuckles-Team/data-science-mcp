#!/usr/bin/python
"""LoRA/QLoRA adapter management + TIES adapter merge (CONCEPT:AHE-3.1).

Adapter-lifecycle half of the Wave-C training substrate. Two concerns:

1. :class:`LoraSpec` / :class:`PeftManager` — build a PEFT ``LoraConfig`` (and an
   optional bitsandbytes 4-bit / QLoRA quant config), load an HF base model,
   attach the adapter, and save / merge it. ``peft`` + ``bitsandbytes`` are heavy,
   GPU-oriented deps so they are imported lazily — installing
   ``data-science-mcp[training]`` provides them; the module itself imports with
   only the std-lib + numpy.

2. :func:`ties_merge` — the **MeMo** (b4-01) multi-adapter merge: TRIM low-magnitude
   deltas, ELECT a consensus sign, then DISJOINT-MEAN the agreeing deltas. This is
   pure numpy (CPU, no torch) so it is fully unit-testable today, and runs on CPU
   in production too (merging is cheap relative to training).

Concept: peft-manager
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LoraSpec:
    """Declarative LoRA / QLoRA configuration (pure data — no peft import)."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    quant_4bit: bool = False  # QLoRA: load base in nf4 4-bit

    def to_peft_config(self) -> Any:
        """Build a ``peft.LoraConfig`` (lazy import)."""
        try:
            from peft import LoraConfig  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "peft is required to attach LoRA adapters; install "
                "`data-science-mcp[training]`"
            ) from e
        return LoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.target_modules),
            bias=self.bias,
            task_type=self.task_type,
        )

    def bnb_config(self) -> Any | None:
        """Build a 4-bit ``BitsAndBytesConfig`` for QLoRA, or ``None`` (lazy)."""
        if not self.quant_4bit:
            return None
        try:
            import torch  # noqa: PLC0415
            from transformers import BitsAndBytesConfig  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "transformers + bitsandbytes are required for QLoRA; install "
                "`data-science-mcp[training]`"
            ) from e
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )


class PeftManager:
    """Load an HF base model and manage its LoRA/QLoRA adapter (lazy heavy deps)."""

    def __init__(
        self,
        base_model: str,
        spec: LoraSpec | None = None,
        *,
        device_map: str | None = "auto",
    ) -> None:
        self.base_model = base_model
        self.spec = spec or LoraSpec()
        self.device_map = device_map
        self.model: Any = None

    def load_base(self) -> Any:
        """Load the HF causal-LM base (4-bit if the spec requests QLoRA)."""
        try:
            from transformers import AutoModelForCausalLM  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "transformers is required to load a base model; install "
                "`data-science-mcp[training]`"
            ) from e
        kwargs: dict[str, Any] = {}
        bnb = self.spec.bnb_config()
        if bnb is not None:
            kwargs["quantization_config"] = bnb
        if self.device_map is not None:
            kwargs["device_map"] = self.device_map
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)
        return self.model

    def attach(self, model: Any | None = None) -> Any:
        """Wrap ``model`` (or the loaded base) with the LoRA adapter."""
        try:
            from peft import get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "peft is required to attach LoRA adapters; install "
                "`data-science-mcp[training]`"
            ) from e
        base = model or self.model
        if base is None:
            base = self.load_base()
        if self.spec.quant_4bit:
            base = prepare_model_for_kbit_training(base)
        self.model = get_peft_model(base, self.spec.to_peft_config())
        return self.model

    def save(self, path: str) -> str:
        """Save the adapter (or model) to ``path``."""
        if self.model is None:
            raise RuntimeError("no model loaded; call attach()/load_base() first")
        self.model.save_pretrained(path)
        return path

    def merge_and_save(self, path: str) -> str:
        """Merge the LoRA weights into the base and save the full model."""
        if self.model is None:
            raise RuntimeError("no model loaded; call attach() first")
        merged = self.model.merge_and_unload()
        merged.save_pretrained(path)
        self.model = merged
        return path


def ties_merge(
    base: dict[str, np.ndarray],
    task_vectors: list[dict[str, np.ndarray]],
    *,
    density: float = 0.2,
    scaling: float = 1.0,
) -> dict[str, np.ndarray]:
    """TIES merge of multiple task vectors onto a base (MeMo, b4-01).

    Each ``task_vector`` is a per-parameter **delta** (fine-tuned − base). For each
    parameter the merge: (1) **trims** each delta to its top-``density`` fraction by
    magnitude (the rest zeroed), (2) **elects** the consensus sign from the summed
    trimmed deltas, (3) **disjoint-means** only the deltas agreeing with that sign,
    and adds ``scaling ×`` the result back onto ``base``. Pure numpy — no torch/GPU.

    Args:
        base: the shared base parameters.
        task_vectors: per-task delta dicts (keys ⊆ ``base``).
        density: fraction of each delta to keep when trimming (0–1].
        scaling: multiplier on the merged delta before adding to base.

    Returns:
        Merged parameter dict (same keys as ``base``).
    """
    if not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    merged: dict[str, np.ndarray] = {}
    for key, base_param in base.items():
        deltas = [tv[key] for tv in task_vectors if key in tv]
        if not deltas:
            merged[key] = np.array(base_param, copy=True)
            continue
        trimmed = [_trim(d, density) for d in deltas]
        stack = np.stack(trimmed, axis=0)
        elected_sign = np.sign(stack.sum(axis=0))
        # Keep only entries whose sign matches the elected sign (disjoint set).
        agree = (np.sign(stack) == elected_sign) & (elected_sign != 0)
        agree_count = agree.sum(axis=0)
        summed = np.where(agree, stack, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            disjoint_mean = np.where(agree_count > 0, summed / agree_count, 0.0)
        merged[key] = base_param + scaling * disjoint_mean
    return merged


def _trim(delta: np.ndarray, density: float) -> np.ndarray:
    """Zero all but the top-``density`` fraction of entries by absolute magnitude."""
    flat = np.asarray(delta, dtype=float).ravel()
    n = flat.size
    keep = max(1, int(round(n * density)))
    if keep >= n:
        return np.asarray(delta, dtype=float)
    threshold = np.partition(np.abs(flat), n - keep)[n - keep]
    out = np.where(np.abs(flat) >= threshold, flat, 0.0)
    return out.reshape(np.asarray(delta).shape)


__all__ = ["LoraSpec", "PeftManager", "ties_merge"]

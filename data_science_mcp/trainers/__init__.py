#!/usr/bin/python
"""Gradient trainers for the in-house training substrate (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Concrete SFT / DPO / GRPO trainers (torch/PEFT) that consume the deterministic
corpora from :mod:`data_science_mcp.training_data` and implement the
:class:`data_science_mcp.training_data.Trainer` Protocol seam. The shared loss
kernels live in :mod:`.objectives`; adapter lifecycle in
:mod:`data_science_mcp.peft_manager`; checkpoint evaluation in :mod:`.eval_hooks`.

torch (+ optional peft/bitsandbytes/vllm) ship in the ``data-science-mcp[training]``
extra. This package imports without those heavy deps — they are pulled in lazily
only when a trainer's ``train()`` actually runs — so the registry and pure planning
(:meth:`TrainerBase.plan`) are always importable.

Concept: trainers
"""

from __future__ import annotations

from data_science_mcp.trainers.base import TrainConfig, TrainerBase
from data_science_mcp.trainers.dpo_trainer import DpoTrainer, build_dpo_trainer
from data_science_mcp.trainers.grpo_trainer import GrpoTrainer, build_grpo_trainer
from data_science_mcp.trainers.ppo_trainer import PpoTrainer, build_ppo_trainer
from data_science_mcp.trainers.pretrain_trainer import (
    PretrainSpec,
    PretrainTrainer,
    build_pretrain_trainer,
)
from data_science_mcp.trainers.reward_trainer import (
    RewardTrainer,
    build_reward_trainer,
)
from data_science_mcp.trainers.sft_trainer import SftTrainer, build_sft_trainer

TRAINERS: dict[str, type[TrainerBase]] = {
    "sft": SftTrainer,
    "dpo": DpoTrainer,
    "grpo": GrpoTrainer,
    "reward": RewardTrainer,
    "ppo": PpoTrainer,
    "pretrain": PretrainTrainer,
}


def get_trainer(kind: str, config: TrainConfig | None = None) -> TrainerBase:
    """Instantiate a trainer by ``kind`` (``sft``|``dpo``|``grpo``|``reward``|``ppo``|``pretrain``)."""
    cls = TRAINERS.get(kind)
    if cls is None:
        raise ValueError(f"unknown trainer kind: {kind!r} (have {sorted(TRAINERS)})")
    return cls(config)


__all__ = [
    "TrainConfig",
    "TrainerBase",
    "SftTrainer",
    "DpoTrainer",
    "GrpoTrainer",
    "RewardTrainer",
    "PpoTrainer",
    "PretrainTrainer",
    "PretrainSpec",
    "build_sft_trainer",
    "build_dpo_trainer",
    "build_grpo_trainer",
    "build_reward_trainer",
    "build_ppo_trainer",
    "build_pretrain_trainer",
    "TRAINERS",
    "get_trainer",
]

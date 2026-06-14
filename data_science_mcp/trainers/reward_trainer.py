#!/usr/bin/python
"""Reward-model trainer — Bradley-Terry pairwise scoring (CONCEPT:ML-008).

The missing RLHF stage between SFT and PPO: train a **scalar reward head** on an
SFT/base backbone so a preferred response scores higher than its rejected partner.
Consumes the same ``{prompt, chosen, rejected}`` corpus as DPO
(:func:`data_science_mcp.training_data.build_preference_pairs`) and optimises the
pairwise loss :func:`data_science_mcp.trainers.objectives.bradley_terry_loss`.

The reward head/backbone is :func:`data_science_mcp.trainers.value_head.attach_scalar_head`
(last-real-token pooling); for a CPU smoke test inject a toy module mapping
``(input_ids, attention_mask) → (batch,)`` plus a tokenizer. The trained reward
model then scores PPO rollouts (``reward_source="reward_model"``).

Concept: reward-trainer
"""

from __future__ import annotations

from typing import Any

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import bradley_terry_loss


class RewardTrainer(TrainerBase):
    """Reward model via the Bradley-Terry pairwise objective."""

    name = "reward"
    kind = "dpo"  # consumes the chosen/rejected (preference) corpus

    def _resolve_reward(self, model: Any | None, tokenizer: Any | None) -> tuple[Any, Any]:
        """Return ``(reward_model, tokenizer)`` — injected if given, else HF + head."""
        if model is not None and tokenizer is not None:
            return model, tokenizer
        from transformers import AutoTokenizer  # noqa: PLC0415

        from data_science_mcp.trainers.value_head import (  # noqa: PLC0415
            attach_scalar_head,
        )

        rm = attach_scalar_head(self.config.base_model, self.config.lora)
        tok = tokenizer or AutoTokenizer.from_pretrained(self.config.base_model)
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        return rm, tok

    def train(
        self,
        dataset: list[dict[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        optimizer: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Train the reward head on ``{prompt, chosen, rejected}`` pairs."""
        import math  # noqa: PLC0415

        from data_science_mcp.trainers.loop import run_loop  # noqa: PLC0415

        torch = _torch()
        pairs = [p for p in dataset if p.get("chosen") and p.get("rejected")]
        if not pairs:
            return {"trainer": self.name, "steps": 0, "examples": 0, "losses": []}
        torch.manual_seed(self.config.seed)
        model, tokenizer = self._resolve_reward(model, tokenizer)
        device = self._device()
        model.to(device)
        model.train()
        self._enable_runtime(model)
        opt = self._optimizer(model, optimizer)
        accel, model, opt = self._prepare(model, opt)
        total = self._total_opt_steps(
            math.ceil(len(pairs) / max(1, self.config.batch_size))
        )
        sched = self._scheduler(opt, total)
        tracker = self._tracker(
            {
                "trainer": self.name,
                "base_model": self.config.base_model,
                "lr": self.config.lr,
                "epochs": self.config.epochs,
                "precision": self.config.precision,
                "lora": self.config.lora is not None,
            }
        )
        accs: list[float] = []

        def _score(texts: list[str]) -> Any:
            enc = self._encode(tokenizer, texts)
            return model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )

        def compute_loss(batch: list[dict[str, Any]]) -> Any:
            chosen = [f"{ex['prompt']}{ex['chosen']}" for ex in batch]
            rejected = [f"{ex['prompt']}{ex['rejected']}" for ex in batch]
            sc = _score(chosen)
            sr = _score(rejected)
            accs.append(float((sc > sr).float().mean().detach()))
            return bradley_terry_loss(sc, sr)

        out = run_loop(
            config=self.config,
            model=model,
            optimizer=opt,
            device=device,
            epoch_items=lambda: self._batches(pairs),
            compute_loss=compute_loss,
            scheduler=sched,
            accelerator=accel,
            tracker=tracker,
            total_steps=total,
        )
        losses = out["losses"]
        mean_acc = (sum(accs) / len(accs)) if accs else None
        report = {
            "trainer": self.name,
            "kind": self.kind,
            "examples": len(pairs),
            "steps": out["steps"],
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "pairwise_accuracy": mean_acc,
            "base_model": self.config.base_model,
            "lora": self.config.lora is not None,
            "checkpoints": out["checkpoints"],
            "resumed_from_step": out["resumed_from_step"],
        }
        tracker.end(
            {
                "final_loss": report["final_loss"],
                "pairwise_accuracy": mean_acc,
                "steps": out["steps"],
            }
        )
        return report


def build_reward_trainer(config: TrainConfig | None = None) -> RewardTrainer:
    return RewardTrainer(config)


__all__ = ["RewardTrainer", "build_reward_trainer"]

#!/usr/bin/python
"""Supervised fine-tuning trainer (CONCEPT:AHE-3.1).

Consumes the ``sft`` corpus from
:func:`data_science_mcp.training_data.build_sft_examples`
(``{prompt, completion}`` records) and runs a standard next-token cross-entropy
loop (full-sequence LM objective over ``prompt + completion``; padding masked) via
:func:`data_science_mcp.trainers.objectives.sft_cross_entropy`. First training
target on GB10 is **OpenSeeker** (b3-02) — Qwen2.5-1.5B LoRA.

Model/tokenizer are dependency-injected for a CPU smoke test on a toy model;
the live path loads the HF base (+ optional LoRA) through :class:`PeftManager`.

Concept: sft-trainer
"""

from __future__ import annotations

from typing import Any

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import sft_cross_entropy


class SftTrainer(TrainerBase):
    """SFT via masked next-token cross-entropy."""

    name = "sft"
    kind = "sft"

    def train(
        self,
        dataset: list[dict[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        optimizer: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Fine-tune on ``{prompt, completion}`` records; return a training report."""
        import math  # noqa: PLC0415

        from data_science_mcp.trainers.loop import run_loop  # noqa: PLC0415

        torch = _torch()
        if not dataset:
            return {"trainer": self.name, "steps": 0, "examples": 0, "losses": []}
        torch.manual_seed(self.config.seed)
        model, tokenizer = self._resolve(model, tokenizer)
        device = self._device()
        model.to(device)
        model.train()
        self._enable_runtime(model)
        opt = self._optimizer(model, optimizer)
        accel, model, opt = self._prepare(model, opt)
        total = self._total_opt_steps(
            math.ceil(len(dataset) / max(1, self.config.batch_size))
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
        pad_id = getattr(tokenizer, "pad_token_id", None)

        def compute_loss(batch: list[dict[str, Any]]) -> Any:
            texts = [
                f"{ex.get('prompt', '')}{ex.get('completion', '')}" for ex in batch
            ]
            enc = self._encode(tokenizer, texts)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn == 0] = -100
            if pad_id is not None:
                labels[labels == pad_id] = -100
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            return sft_cross_entropy(logits, labels)

        out = run_loop(
            config=self.config,
            model=model,
            optimizer=opt,
            device=device,
            epoch_items=lambda: self._batches(dataset),
            compute_loss=compute_loss,
            scheduler=sched,
            accelerator=accel,
            tracker=tracker,
            total_steps=total,
        )
        losses = out["losses"]
        report = {
            "trainer": self.name,
            "kind": self.kind,
            "examples": len(dataset),
            "steps": out["steps"],
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "base_model": self.config.base_model,
            "lora": self.config.lora is not None,
            "checkpoints": out["checkpoints"],
            "resumed_from_step": out["resumed_from_step"],
        }
        tracker.end({"final_loss": report["final_loss"], "steps": out["steps"]})
        return report


def build_sft_trainer(config: TrainConfig | None = None) -> SftTrainer:
    return SftTrainer(config)


__all__ = ["SftTrainer", "build_sft_trainer"]

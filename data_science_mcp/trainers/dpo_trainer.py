#!/usr/bin/python
"""Direct Preference Optimization trainer (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Consumes the ``dpo`` corpus from
:func:`data_science_mcp.training_data.build_preference_pairs`
(``{prompt, chosen, rejected, failure_point}``) and optimises the Bradley-Terry
preference objective (:func:`data_science_mcp.trainers.objectives.dpo_loss`)
against a **frozen reference** copy of the policy. Target paper: **MedCausalX**
(b6-01) Stage-I preference alignment from failure-point-anchored pairs.

Because ``chosen`` and ``rejected`` share the prompt prefix, the prompt token
log-probs cancel in the log-ratio, so a full-sequence log-prob
(:func:`objectives.sequence_logprob`, padding masked) is equivalent to a
response-only one — no prompt-length bookkeeping needed.

Concept: dpo-trainer
"""

from __future__ import annotations

import copy
from typing import Any

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import dpo_loss, sequence_logprob


class DpoTrainer(TrainerBase):
    """DPO against a frozen reference model."""

    name = "dpo"
    kind = "dpo"

    def _logps(self, model: Any, enc: Any, device: Any) -> Any:
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attn == 0] = -100
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        return sequence_logprob(logits, labels)

    def train(
        self,
        dataset: list[dict[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        ref_model: Any | None = None,
        optimizer: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Optimise the DPO objective; return a training report."""
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
        # Frozen reference = snapshot of the policy at start (SFT checkpoint), taken
        # before any distributed wrapping so it stays a plain on-device module.
        ref = ref_model if ref_model is not None else copy.deepcopy(model)
        ref.to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
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
                "beta": self.config.beta,
                "lr": self.config.lr,
                "epochs": self.config.epochs,
                "precision": self.config.precision,
            }
        )

        def compute_loss(batch: list[dict[str, Any]]) -> Any:
            chosen = [f"{ex['prompt']}{ex['chosen']}" for ex in batch]
            rejected = [f"{ex['prompt']}{ex['rejected']}" for ex in batch]
            enc_c = self._encode(tokenizer, chosen)
            enc_r = self._encode(tokenizer, rejected)
            pol_c = self._logps(model, enc_c, device)
            pol_r = self._logps(model, enc_r, device)
            with torch.no_grad():
                ref_c = self._logps(ref, enc_c, device)
                ref_r = self._logps(ref, enc_r, device)
            return dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=self.config.beta)

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
            "beta": self.config.beta,
            "base_model": self.config.base_model,
            "checkpoints": out["checkpoints"],
            "resumed_from_step": out["resumed_from_step"],
        }
        tracker.end({"final_loss": report["final_loss"], "steps": out["steps"]})
        return report


def build_dpo_trainer(config: TrainConfig | None = None) -> DpoTrainer:
    return DpoTrainer(config)


__all__ = ["DpoTrainer", "build_dpo_trainer"]

#!/usr/bin/python
"""Group-Relative Policy Optimization trainer (CONCEPT:AHE-3.1).

Consumes GRPO groups — ``{prompt, samples:[{completion, reward, advantage}]}`` —
produced by :func:`data_science_mcp.training_data.build_grpo_groups` or
:meth:`data_science_mcp.rollout_buffer.RolloutBuffer.to_grpo_groups`, and applies
the clipped policy-gradient surrogate
(:func:`data_science_mcp.trainers.objectives.grpo_surrogate`) with group-normalised
advantages, plus an optional KL-to-reference penalty
(:func:`objectives.approx_kl`). Target papers: **ATLAS** (b6-04) and **SDAR**
(b7-03); the heavy rollout generation is served by vLLM via
:class:`~data_science_mcp.rollout_buffer.VLLMRolloutClient`.

Sequence-level advantages: each sampled completion gets one advantage, applied to
its summed response log-prob. ``old_logprob`` defaults to the detached current
log-prob (single on-policy step); pass stored generation log-probs for multi-epoch
PPO-style reuse.

Concept: grpo-trainer
"""

from __future__ import annotations

import copy
from typing import Any

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import (
    approx_kl,
    grpo_surrogate,
    sequence_logprob,
)


class GrpoTrainer(TrainerBase):
    """GRPO with group-normalised advantages and optional KL penalty."""

    name = "grpo"
    kind = "grpo"

    def _seq_logps(
        self, model: Any, texts: list[str], tokenizer: Any, device: Any
    ) -> Any:
        enc = self._encode(tokenizer, texts)
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
        """Optimise the GRPO surrogate over advantage-tagged groups."""
        torch = _torch()
        groups = [g for g in dataset if g.get("samples")]
        if not groups:
            return {"trainer": self.name, "steps": 0, "groups": 0, "losses": []}
        torch.manual_seed(self.config.seed)
        model, tokenizer = self._resolve(model, tokenizer)
        device = self._device()
        model.to(device)
        model.train()
        ref = None
        if self.config.kl_coef > 0:
            ref = ref_model if ref_model is not None else copy.deepcopy(model)
            ref.to(device)
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
        opt = self._optimizer(model, optimizer)

        losses: list[float] = []
        kls: list[float] = []
        step = 0
        for _epoch in range(max(1, self.config.epochs)):
            for group in groups:
                samples = group["samples"]
                prompt = group.get("prompt", "")
                texts = [f"{prompt}{s['completion']}" for s in samples]
                adv = torch.tensor(
                    [float(s["advantage"]) for s in samples],
                    dtype=torch.float32,
                    device=device,
                )
                logp = self._seq_logps(model, texts, tokenizer, device)
                old = logp.detach()
                loss = grpo_surrogate(logp, old, adv, clip_eps=self.config.clip_eps)
                if ref is not None:
                    with torch.no_grad():
                        ref_logp = self._seq_logps(ref, texts, tokenizer, device)
                    kl = approx_kl(logp, ref_logp)
                    loss = loss + self.config.kl_coef * kl
                    kls.append(float(kl.detach()))
                loss.backward()
                opt.step()
                opt.zero_grad()
                losses.append(float(loss.detach()))
                step += 1
                if self.config.max_steps is not None and step >= self.config.max_steps:
                    break
            if self.config.max_steps is not None and step >= self.config.max_steps:
                break

        return {
            "trainer": self.name,
            "kind": self.kind,
            "groups": len(groups),
            "steps": step,
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "mean_kl": (sum(kls) / len(kls)) if kls else None,
            "clip_eps": self.config.clip_eps,
            "kl_coef": self.config.kl_coef,
            "base_model": self.config.base_model,
        }


def build_grpo_trainer(config: TrainConfig | None = None) -> GrpoTrainer:
    return GrpoTrainer(config)


__all__ = ["GrpoTrainer", "build_grpo_trainer"]

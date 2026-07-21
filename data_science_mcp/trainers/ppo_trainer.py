#!/usr/bin/python
"""Proximal Policy Optimization trainer (CONCEPT:DS-AHE.trainer.per-token-value).

The classic RLHF policy-optimisation stage the in-house substrate was missing
(GRPO is the value-free cousin; this is full actor-critic PPO). Per item it:

1. scores the completion's reward — from a **verifier** (``reward_fn``) or a trained
   **reward model** (CONCEPT:DS-AHE.reward.one-sequence-level-score), selected by ``config.reward_source``;
2. computes per-token policy log-probs (:func:`objectives.token_logprob`) and the
   frozen-reference log-probs for a per-token KL penalty;
3. reads per-token values from a value head
   (:func:`data_science_mcp.trainers.value_head`);
4. forms GAE advantages/returns (:func:`objectives.gae`) over the response tokens
   with the terminal reward + per-token KL shaping, then whitens the advantages;
5. minimises the clipped PPO surrogate (:func:`objectives.grpo_surrogate`, the
   shared clipped-ratio kernel) + ``vf_coef`` · value loss
   (:func:`objectives.value_function_loss`).

Rollout *generation* is upstream (reuse
:class:`data_science_mcp.rollout_buffer.RolloutBuffer` against a served vLLM/SGLang
backend); this trainer consumes ``{prompt, completion, reward?}`` records, mirroring
how GRPO consumes pre-built groups. ``old_logprob`` defaults to the detached current
log-prob (single on-policy step); pass stored generation log-probs for multi-epoch
PPO reuse where the clip genuinely binds.

Model / value model / tokenizer are dependency-injected for a CPU smoke test on toy
modules; the live path loads the HF base (+ value head) via :class:`PeftManager`.

Concept: ppo-trainer
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import (
    gae,
    grpo_surrogate,
    token_logprob,
    value_function_loss,
    whiten,
)


class PpoTrainer(TrainerBase):
    """Actor-critic PPO with GAE, value clipping, and a KL-to-reference penalty."""

    name = "ppo"
    kind = "ppo"

    def train(
        self,
        dataset: list[dict[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        value_model: Any | None = None,
        ref_model: Any | None = None,
        reward_model: Any | None = None,
        reward_fn: Callable[[str, str], float] | None = None,
        optimizer: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Optimise the PPO objective over ``{prompt, completion, reward?}`` records."""
        from data_science_mcp.trainers.loop import run_loop  # noqa: PLC0415

        torch = _torch()
        items = [d for d in dataset if d.get("prompt") and d.get("completion")]
        if not items:
            return {"trainer": self.name, "steps": 0, "examples": 0, "losses": []}
        torch.manual_seed(self.config.seed)
        model, tokenizer = self._resolve(model, tokenizer)
        device = self._device()
        model.to(device)
        model.train()
        self._enable_runtime(model)

        # Value function: injected toy module, or a scalar head on a fresh backbone.
        if value_model is None:
            from data_science_mcp.trainers.value_head import (  # noqa: PLC0415
                attach_scalar_head,
            )

            value_model = attach_scalar_head(
                self.config.base_model,
                self.config.lora,
                revision=self.config.model_revision,
            )
        value_model.to(device)
        value_model.train()

        # Frozen reference for the per-token KL penalty (snapshot of the policy).
        ref = ref_model if ref_model is not None else copy.deepcopy(model)
        ref.to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        if reward_model is not None:
            reward_model.to(device)
            reward_model.eval()

        # One optimizer over policy + value params (shared step in run_loop).
        params = [p for p in model.parameters() if p.requires_grad] + [
            p for p in value_model.parameters() if p.requires_grad
        ]
        opt = optimizer if optimizer is not None else torch.optim.AdamW(
            params, lr=self.config.lr
        )
        accel, model, opt = self._prepare(model, opt)
        total = self._total_opt_steps(
            (len(items) + self.config.batch_size - 1) // max(1, self.config.batch_size)
        )
        sched = self._scheduler(opt, total)
        tracker = self._tracker(
            {
                "trainer": self.name,
                "base_model": self.config.base_model,
                "clip_eps": self.config.clip_eps,
                "kl_coef": self.config.kl_coef,
                "vf_coef": self.config.vf_coef,
                "reward_source": self.config.reward_source,
                "lr": self.config.lr,
                "precision": self.config.precision,
            }
        )

        # --- reward per item (no grad) --------------------------------------- #
        def _reward(prompt: str, completion: str, stored: Any) -> float:
            if reward_fn is not None and self.config.reward_source == "verifier":
                return float(reward_fn(prompt, completion))
            if reward_model is not None and self.config.reward_source == "reward_model":
                enc = self._encode(tokenizer, [f"{prompt}{completion}"])
                with torch.no_grad():
                    s = reward_model(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    )
                return float(s.reshape(-1)[0].detach())
            return float(stored if stored is not None else 0.0)

        rewards_seen: list[float] = []
        kls: list[float] = []

        def _item_loss(item: dict[str, Any]) -> Any:
            prompt = str(item["prompt"])
            completion = str(item["completion"])
            # Encode prompt+completion as one sequence; recover the response span from
            # the prompt's own encoded length (an exact prefix for char/BPE tokenizers,
            # approximate at the boundary token for some HF tokenizers — fine for RL).
            enc = self._encode(tokenizer, [f"{prompt}{completion}"])
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            if input_ids.size(1) < 2:
                return None
            enc_p = self._encode(tokenizer, [prompt])
            prompt_len = int(enc_p["attention_mask"].sum().item())
            # Shifted action positions 0..L-1 generate tokens 1..L; a position is a
            # response action when the token it generates falls past the prompt.
            length = input_ids.size(1)
            prompt_len = min(prompt_len, length)
            gen_idx = torch.arange(1, length, device=device)  # token index per action
            resp_mask = (gen_idx >= prompt_len).float().unsqueeze(0)  # (1, L-1)

            logits = model(input_ids=input_ids, attention_mask=attn).logits
            logp = token_logprob(logits, input_ids)  # (1, L-1)
            with torch.no_grad():
                ref_logits = ref(input_ids=input_ids, attention_mask=attn).logits
                ref_logp = token_logprob(ref_logits, input_ids)
            values = value_model(input_ids, attn, per_token=True)[:, :-1]  # (1, L-1)

            r = _reward(prompt, completion, item.get("reward"))
            rewards_seen.append(r)
            # Per-token reward: signed KL penalty everywhere (log π − log π_ref, the
            # standard PPO shaping term), terminal scalar reward at the last response.
            kl_tok = logp.detach() - ref_logp
            token_rewards = -self.config.kl_coef * kl_tok
            last = int(resp_mask[0].nonzero().max().item()) if resp_mask.sum() > 0 else length - 2
            token_rewards[0, last] = token_rewards[0, last] + r
            # Reported KL = non-negative Schulman k3 estimate over response tokens.
            log_r = ref_logp - logp.detach()
            k3 = torch.exp(log_r) - 1.0 - log_r
            kls.append(float((k3 * resp_mask).sum() / resp_mask.sum().clamp_min(1.0)))

            adv, returns = gae(
                token_rewards.detach(),
                values.detach(),
                mask=resp_mask,
                gamma=self.config.gamma,
                lam=self.config.gae_lambda,
            )
            adv = whiten(adv, mask=resp_mask)
            policy_loss = grpo_surrogate(
                logp, logp.detach(), adv, clip_eps=self.config.clip_eps, mask=resp_mask
            )
            v_loss = value_function_loss(
                values, returns, clip=self.config.value_clip, mask=resp_mask
            )
            return policy_loss + self.config.vf_coef * v_loss

        def compute_loss(batch: list[dict[str, Any]]) -> Any:
            losses = [loss for loss in (_item_loss(it) for it in batch) if loss is not None]
            if not losses:
                return torch.zeros((), device=device, requires_grad=True)
            return torch.stack(losses).mean()

        out = run_loop(
            config=self.config,
            model=model,
            optimizer=opt,
            device=device,
            epoch_items=lambda: self._batches(items),
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
            "examples": len(items),
            "steps": out["steps"],
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "mean_reward": (sum(rewards_seen) / len(rewards_seen)) if rewards_seen else None,
            "mean_kl": (sum(kls) / len(kls)) if kls else None,
            "clip_eps": self.config.clip_eps,
            "kl_coef": self.config.kl_coef,
            "reward_source": self.config.reward_source,
            "base_model": self.config.base_model,
            "checkpoints": out["checkpoints"],
            "resumed_from_step": out["resumed_from_step"],
        }
        tracker.end(
            {
                "final_loss": report["final_loss"],
                "mean_reward": report["mean_reward"],
                "steps": out["steps"],
            }
        )
        return report


def build_ppo_trainer(config: TrainConfig | None = None) -> PpoTrainer:
    return PpoTrainer(config)


__all__ = ["PpoTrainer", "build_ppo_trainer"]

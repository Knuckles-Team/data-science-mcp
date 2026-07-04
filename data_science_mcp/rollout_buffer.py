#!/usr/bin/python
"""Rollout buffer + served-model generation client for on-policy RL (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

GRPO/SDAR/ATLAS need *many* sampled completions per prompt, scored with a reward,
then group-normalised into advantages. This module is that staging area:

* :class:`Rollout` / :class:`RolloutBuffer` — accumulate ``prompt → completions →
  per-token logprobs → reward`` groups, score them with any reward callable, and
  export GRPO training groups (delegating advantage normalisation to
  :func:`data_science_mcp.training_data.build_grpo_groups`, the shared reward spine).
* :data:`VLLMRolloutClient` — back-compat alias of
  :class:`data_science_mcp.inference.VLLMBackend`. Rollout generation is served by
  an **already-running** OpenAI-compatible server; swap engines (vLLM ↔ SGLang) by
  passing a different :class:`~data_science_mcp.inference.InferenceBackend` to
  :meth:`RolloutBuffer.generate` (or via ``INFERENCE_BACKEND``).

The buffer is pure Python (CPU, unit-testable with an injected fake client); only
the backend touches the network, and ``httpx`` is imported lazily.

Concept: rollout-buffer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from data_science_mcp.inference import VLLMBackend
from data_science_mcp.training_data import build_grpo_groups


@dataclass
class Rollout:
    """One sampled completion for a prompt."""

    completion: str
    logprobs: list[float] = field(default_factory=list)
    reward: float = 0.0


class RolloutBuffer:
    """Group sampled completions by prompt and export GRPO training groups."""

    def __init__(self) -> None:
        self._groups: dict[str, list[Rollout]] = {}

    def add(
        self,
        prompt: str,
        completion: str,
        *,
        logprobs: list[float] | None = None,
        reward: float = 0.0,
    ) -> None:
        """Add a single rollout to ``prompt``'s group."""
        self._groups.setdefault(prompt, []).append(
            Rollout(completion=completion, logprobs=list(logprobs or []), reward=reward)
        )

    def add_group(
        self,
        prompt: str,
        completions: list[str],
        *,
        logprobs: list[list[float]] | None = None,
    ) -> None:
        """Add a batch of completions for one prompt (rewards scored later)."""
        for i, comp in enumerate(completions):
            lp = logprobs[i] if logprobs and i < len(logprobs) else None
            self.add(prompt, comp, logprobs=lp)

    @property
    def prompts(self) -> list[str]:
        return list(self._groups)

    def __len__(self) -> int:
        return sum(len(v) for v in self._groups.values())

    def score(self, reward_fn: Callable[[str, str], float]) -> None:
        """Score every rollout in place via ``reward_fn(prompt, completion)``."""
        for prompt, rollouts in self._groups.items():
            for r in rollouts:
                r.reward = float(reward_fn(prompt, r.completion))

    def to_grpo_groups(self, *, min_group: int = 2) -> list[dict[str, Any]]:
        """Export ``{prompt, samples:[{completion, reward, advantage}]}`` groups.

        Groups smaller than ``min_group`` are dropped (group-relative advantage is
        undefined for a single sample). Advantage normalisation is delegated to the
        shared :func:`build_grpo_groups` reward spine.
        """
        raw = [
            {
                "prompt": prompt,
                "completions": [r.completion for r in rollouts],
                "rewards": [r.reward for r in rollouts],
            }
            for prompt, rollouts in self._groups.items()
            if len(rollouts) >= min_group
        ]
        return build_grpo_groups(raw)

    def clear(self) -> None:
        self._groups.clear()

    def generate(
        self,
        prompts: list[str],
        client: Any,
        *,
        n: int = 4,
        **gen_kwargs: Any,
    ) -> None:
        """Fill the buffer by sampling ``n`` completions/prompt from ``client``.

        ``client`` is any object with ``generate(prompt, n, **kwargs) ->
        list[{"text": str, "logprobs": list[float]}]`` — any
        :class:`~data_science_mcp.inference.InferenceBackend` (vLLM or SGLang)
        satisfies it, as does a fake in tests.
        """
        for prompt in prompts:
            outs = client.generate(prompt, n=n, **gen_kwargs)
            self.add_group(
                prompt,
                [o["text"] for o in outs],
                logprobs=[o.get("logprobs", []) for o in outs],
            )


# Back-compat alias: the vLLM rollout client is now the general vLLM inference
# backend. Existing callers (`VLLMRolloutClient(base_url, model)`) are unchanged;
# the HTTP/constrained-decoding logic lives in `data_science_mcp.inference`.
VLLMRolloutClient = VLLMBackend


__all__ = ["Rollout", "RolloutBuffer", "VLLMRolloutClient"]

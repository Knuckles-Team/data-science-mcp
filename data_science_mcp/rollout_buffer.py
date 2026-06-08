#!/usr/bin/python
"""Rollout buffer + vLLM generation client for on-policy RL (CONCEPT:AHE-3.1).

GRPO/SDAR/ATLAS need *many* sampled completions per prompt, scored with a reward,
then group-normalised into advantages. This module is that staging area:

* :class:`Rollout` / :class:`RolloutBuffer` — accumulate ``prompt → completions →
  per-token logprobs → reward`` groups, score them with any reward callable, and
  export GRPO training groups (delegating advantage normalisation to
  :func:`data_science_mcp.training_data.build_grpo_groups`, the shared reward spine).
* :class:`VLLMRolloutClient` — generate completions from the **already-running vLLM**
  OpenAI-compatible endpoint (far faster than in-process ``.generate()``), returning
  text + per-token logprobs for the GRPO surrogate.

The buffer is pure Python (CPU, unit-testable with an injected fake client); only
:class:`VLLMRolloutClient` touches the network, and ``httpx`` is imported lazily.

Concept: rollout-buffer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

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
        list[{"text": str, "logprobs": list[float]}]`` (see
        :class:`VLLMRolloutClient`); a fake satisfies it in tests.
        """
        for prompt in prompts:
            outs = client.generate(prompt, n=n, **gen_kwargs)
            self.add_group(
                prompt,
                [o["text"] for o in outs],
                logprobs=[o.get("logprobs", []) for o in outs],
            )


class VLLMRolloutClient:
    """Generate completions from a vLLM OpenAI-compatible ``/v1/completions`` endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 1.0,
        logprobs: int = 1,
    ) -> list[dict[str, Any]]:
        """Sample ``n`` completions with per-token logprobs (lazy ``httpx``)."""
        try:
            import httpx  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "httpx is required for vLLM rollouts; install "
                "`data-science-mcp[training]`"
            ) from e
        resp = httpx.post(
            f"{self.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "prompt": prompt,
                "n": n,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": logprobs,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "text": ch.get("text", ""),
                "logprobs": (ch.get("logprobs") or {}).get("token_logprobs") or [],
            }
            for ch in data.get("choices", [])
        ]


__all__ = ["Rollout", "RolloutBuffer", "VLLMRolloutClient"]

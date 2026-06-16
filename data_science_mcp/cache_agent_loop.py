#!/usr/bin/python
"""CacheRL cached rollouts + hybrid-thinking augmentation — CONCEPT:ML-013.

Distils **CacheRL: Multi-Turn Tool-Calling Agents via Cached Rollouts and Hybrid
Reward** (arXiv:2606.14179). Multi-turn tool-calling RL is dominated by the cost
of *live tool execution* during rollouts: the same tool is called with the same
(or near-identical) arguments thousands of times across a GRPO group. CacheRL's
``CacheAgentLoop`` serves those calls from a three-tier fuzzy cache so rollouts
run at a fraction of the live cost, and a ``ThinkingTraceAugmenter`` teaches the
model *why* it picks each tool — not just which one.

This module is the cache + augmentation half (pure Python, CPU-unit-testable with
an injected fake executor); the reward half — token-level masking of injected
observations and cache-tier-aware reward shaping — lives in the shared reward
spine ``agent_utilities.graph.training_signals`` (concept AHE-3.49) so every
trainer consumes it. Together they let a small model reach high tool-calling
accuracy at ~100x less rollout compute (the paper's headline).

Concept: cache-agent-loop
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "CacheTier",
    "CacheResult",
    "ThreeTierToolCache",
    "CacheAgentLoop",
    "ThinkingTraceAugmenter",
    "augment_trajectory",
]


class CacheTier(str, Enum):
    """Provenance of a tool result, from most to least reliable.

    ``EXACT`` and ``FUZZY``/``SEMANTIC`` are cache hits; ``LIVE`` is a real
    execution (a cache miss). The ordering matters for reward shaping: a lower
    tier carries more risk that a downstream failure is the cache's fault, not the
    model's (consumed by ``cache_tier_aware_reward``).
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    SEMANTIC = "semantic"
    LIVE = "live"


@dataclass
class CacheResult:
    """Outcome of a cache lookup."""

    hit: bool
    tier: CacheTier
    value: Any = None
    similarity: float = 0.0


def _norm_args(args: Any) -> str:
    """Canonical, order-insensitive string form of tool arguments."""
    if isinstance(args, dict):
        return json.dumps(args, sort_keys=True, default=str)
    return json.dumps(args, default=str)


def _arg_tokens(args: Any) -> set[str]:
    """Bag of lowercase tokens over an argument blob (for fuzzy/semantic match)."""
    text = _norm_args(args).lower()
    return {t for t in "".join(c if c.isalnum() else " " for c in text).split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class _Entry:
    tool: str
    tokens: set[str]
    value: Any
    embedding: list[float] | None = None


class ThreeTierToolCache:
    """Exact → fuzzy → semantic cache of tool-call results.

    * **Tier 0 (exact)** — hash of ``(tool, canonical-args)``; O(1).
    * **Tier 1 (fuzzy)** — Jaccard over argument tokens ≥ ``fuzzy_threshold``
      (same tool only). Catches reordered / whitespace / near-identical args.
    * **Tier 2 (semantic)** — cosine over an optional ``embed_fn(args)`` ≥
      ``semantic_threshold``; degrades to a token-overlap proxy when no embedder is
      supplied, so the tier is always available offline.

    ``lookup`` returns the *highest* matching tier. The cache never executes
    anything — it only remembers what an executor returned (see
    :class:`CacheAgentLoop`).
    """

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 0.8,
        semantic_threshold: float = 0.9,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self._embed_fn = embed_fn
        self._exact: dict[str, Any] = {}
        self._entries: list[_Entry] = []

    @staticmethod
    def _exact_key(tool: str, args: Any) -> str:
        return f"{tool}\x00{_norm_args(args)}"

    def put(self, tool: str, args: Any, value: Any) -> None:
        key = self._exact_key(tool, args)
        if key not in self._exact:
            emb = self._embed_fn(_norm_args(args)) if self._embed_fn else None
            self._entries.append(_Entry(tool, _arg_tokens(args), value, emb))
        self._exact[key] = value

    def lookup(self, tool: str, args: Any) -> CacheResult:
        key = self._exact_key(tool, args)
        if key in self._exact:
            return CacheResult(True, CacheTier.EXACT, self._exact[key], 1.0)

        tokens = _arg_tokens(args)
        best_fuzzy = (0.0, None)
        for e in self._entries:
            if e.tool != tool:
                continue
            s = _jaccard(tokens, e.tokens)
            if s > best_fuzzy[0]:
                best_fuzzy = (s, e)
        if best_fuzzy[1] is not None and best_fuzzy[0] >= self.fuzzy_threshold:
            return CacheResult(True, CacheTier.FUZZY, best_fuzzy[1].value, best_fuzzy[0])

        # Tier 2: semantic (embedding cosine, or token-overlap proxy offline).
        query_emb = self._embed_fn(_norm_args(args)) if self._embed_fn else None
        best_sem = (0.0, None)
        for e in self._entries:
            if e.tool != tool:
                continue
            if query_emb is not None and e.embedding is not None:
                s = _cosine(query_emb, e.embedding)
            else:
                s = _jaccard(tokens, e.tokens)  # offline proxy
            if s > best_sem[0]:
                best_sem = (s, e)
        if best_sem[1] is not None and best_sem[0] >= self.semantic_threshold:
            return CacheResult(True, CacheTier.SEMANTIC, best_sem[1].value, best_sem[0])

        return CacheResult(False, CacheTier.LIVE, None, max(best_fuzzy[0], best_sem[0]))

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class CacheAgentLoop:
    """Serve tool calls from the cache, falling back to a live executor on a miss.

    ``live_executor(tool, args) -> observation`` is the only thing that ever
    touches the real environment; everything the cache can answer is answered for
    free. ``stats`` tallies hits per tier and the count of live executions saved —
    the realized rollout-compute reduction (CacheRL's 100x claim made measurable).
    """

    live_executor: Callable[[str, Any], Any]
    cache: ThreeTierToolCache = field(default_factory=ThreeTierToolCache)
    hits: dict[str, int] = field(
        default_factory=lambda: {t.value: 0 for t in CacheTier}
    )
    live_calls: int = 0
    total_calls: int = 0

    def call(self, tool: str, args: Any) -> tuple[Any, CacheTier]:
        self.total_calls += 1
        res = self.cache.lookup(tool, args)
        if res.hit:
            self.hits[res.tier.value] += 1
            return res.value, res.tier
        value = self.live_executor(tool, args)
        self.cache.put(tool, args, value)
        self.live_calls += 1
        self.hits[CacheTier.LIVE.value] += 1
        return value, CacheTier.LIVE

    def calls_saved(self) -> int:
        """Live executions avoided by cache hits this loop."""
        return self.total_calls - self.live_calls

    def hit_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return round(self.calls_saved() / self.total_calls, 6)


# ── Hybrid Thinking Pipeline ──────────────────────────────────────────────────
# Augment a raw tool-calling trajectory with model-generated reasoning ("why this
# tool") before each action, and emit per-segment provenance labels so the trainer
# can mask injected observations from the loss (token_cache_mask, AHE-3.49).

_THOUGHT = "model"  # trainable reasoning tokens the model must learn to produce
_ACTION = "action"  # the tool call the model emits (trainable)
_OBSERVATION = "observation"  # injected tool result (cached or live) — masked


@dataclass
class ThinkingTraceAugmenter:
    """Insert a reasoning trace before each tool call (CacheRL hybrid thinking).

    ``reason_fn(step, context) -> str`` generates the natural-language rationale
    for a step; in tests it is a deterministic fake, in production an LLM. The
    result is a list of ``{"text", "source", "tier"}`` segments where ``source`` is
    ``model`` (thought), ``action`` (tool call), or ``observation`` (tool output).
    The ``source`` labels drive token-level masking downstream — only ``model`` and
    ``action`` tokens earn gradient; injected observations are masked, which is
    what preserves trajectory quality when observations come from the cache.
    """

    reason_fn: Callable[[dict[str, Any], list[dict[str, Any]]], str]

    def augment(self, steps: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        for step in steps:
            thought = self.reason_fn(step, history)
            if thought:
                out.append({"text": str(thought), "source": _THOUGHT, "tier": None})
            tool = step.get("tool", "")
            args = step.get("args", {})
            out.append(
                {
                    "text": f"{tool}({_norm_args(args)})",
                    "source": _ACTION,
                    "tier": None,
                }
            )
            tier = step.get("tier")
            out.append(
                {
                    "text": str(step.get("observation", "")),
                    "source": _OBSERVATION,
                    "tier": tier.value if isinstance(tier, CacheTier) else tier,
                }
            )
            history.append(step)
        return out


def augment_trajectory(
    steps: Iterable[dict[str, Any]],
    reason_fn: Callable[[dict[str, Any], list[dict[str, Any]]], str],
) -> list[dict[str, Any]]:
    """Convenience wrapper around :class:`ThinkingTraceAugmenter`."""
    return ThinkingTraceAugmenter(reason_fn=reason_fn).augment(steps)

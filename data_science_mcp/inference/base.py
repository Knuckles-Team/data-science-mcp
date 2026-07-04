#!/usr/bin/python
"""Inference-backend interface for served-model generation (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

The trainer substrate needs to *talk to a running inference server* in two
places: GRPO rollouts (sample many completions per prompt, with per-token
logprobs) and post-train reliability eval (one completion per case). Both vLLM
and SGLang expose the **same OpenAI-compatible HTTP protocol**, so the backend
is a thin client — there is no embedded engine and no GPU dependency here.

``InferenceBackend`` is the vendor-agnostic seam (mirrors the ``GraphBackend``
ABC in ``agent_utilities.knowledge_graph.backends``). Concrete backends differ
only in how they name engine-specific request params (e.g. constrained-decoding:
vLLM ``guided_json`` vs SGLang ``json_schema``); prefix-cache / RadixAttention is
automatic server-side and needs no client support.

Concept: inference-backend
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class InferenceBackend(ABC):
    """Abstract client for an OpenAI-compatible inference server.

    Concrete backends (``VLLMBackend``, ``SGLangBackend``) implement
    :meth:`generate` against an already-running server. The return shape matches
    the duck-typed contract :class:`RolloutBuffer` already consumes —
    ``[{"text": str, "logprobs": list[float]}, ...]`` — so a backend is a drop-in
    rollout client.
    """

    #: Engine identifier (e.g. ``"vllm"``, ``"sglang"``); also the model-registry provider.
    provider: str = "openai-compatible"
    #: Whether the backend maps JSON-schema/regex/grammar to constrained decoding.
    supports_constrained: bool = False
    #: Whether the backend can target the chat endpoint (``/v1/chat/completions``).
    supports_chat: bool = False

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        n: int = 4,
        max_tokens: int = 512,
        temperature: float = 1.0,
        logprobs: int = 1,
        json_schema: dict[str, Any] | None = None,
        regex: str | None = None,
        grammar: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Sample ``n`` completions for ``prompt`` with per-token logprobs.

        Args:
            prompt: the input text.
            n: number of completions to sample.
            max_tokens / temperature / logprobs: standard sampling controls.
            json_schema / regex / grammar: optional constrained-decoding spec —
                at most one should be set; backends that
                ``supports_constrained`` map it to the engine's param.

        Returns:
            ``[{"text": str, "logprobs": list[float]}, ...]`` (one per completion).
        """
        ...

    def as_generate_fn(self, **gen_kwargs: Any) -> Callable[[str], str]:
        """Adapt the backend to a ``generate_fn(input) -> output`` for eval.

        :func:`data_science_mcp.trainers.eval_hooks.evaluate_checkpoint` wants a
        single-completion ``Callable[[str], str]``; this returns the first
        completion's text (empty string if the server returns nothing). Pass
        sampling overrides (e.g. ``n=1, temperature=0.0``) via ``gen_kwargs``.
        """
        defaults: dict[str, Any] = {"n": 1, "temperature": 0.0, **gen_kwargs}

        def _fn(text: str) -> str:
            outs = self.generate(text, **defaults)
            return outs[0]["text"] if outs else ""

        return _fn


__all__ = ["InferenceBackend"]

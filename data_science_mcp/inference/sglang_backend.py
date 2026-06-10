#!/usr/bin/python
"""SGLang inference backend (CONCEPT:AHE-3.1).

Talks to an already-running SGLang OpenAI-compatible server
(``python -m sglang.launch_server``). SGLang names constrained decoding
``json_schema`` / ``regex`` / ``ebnf``. Its RadixAttention prefix-reuse is
automatic server-side, so pointing at an SGLang server transparently reuses
shared prompt prefixes (system prompts, few-shot blocks) across rollouts with no
client change — the main reason to prefer it for structured/agent workloads.

Concept: inference-backend
"""

from __future__ import annotations

from typing import Any

from data_science_mcp.inference.openai_compatible import OpenAICompatibleBackend


class SGLangBackend(OpenAICompatibleBackend):
    """OpenAI-compatible backend with SGLang ``json_schema``/``regex``/``ebnf``."""

    provider = "sglang"
    supports_constrained = True
    supports_chat = True

    def _constrained_params(
        self,
        json_schema: dict[str, Any] | None,
        regex: str | None,
        grammar: str | None,
    ) -> dict[str, Any]:
        if json_schema is not None:
            return {"json_schema": json_schema}
        if regex is not None:
            return {"regex": regex}
        if grammar is not None:
            return {"ebnf": grammar}
        return {}


__all__ = ["SGLangBackend"]

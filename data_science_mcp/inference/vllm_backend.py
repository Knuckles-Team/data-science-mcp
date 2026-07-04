#!/usr/bin/python
"""vLLM inference backend (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Talks to an already-running vLLM OpenAI-compatible server. vLLM exposes
constrained decoding as ``guided_json`` / ``guided_regex`` / ``guided_grammar``
(xgrammar/outlines under the hood). Prefix caching is server-side
(``--enable-prefix-caching``) and needs no client support.

Concept: inference-backend
"""

from __future__ import annotations

from typing import Any

from data_science_mcp.inference.openai_compatible import OpenAICompatibleBackend


class VLLMBackend(OpenAICompatibleBackend):
    """OpenAI-compatible backend with vLLM ``guided_*`` constrained decoding."""

    provider = "vllm"
    supports_constrained = True
    supports_chat = True

    def _constrained_params(
        self,
        json_schema: dict[str, Any] | None,
        regex: str | None,
        grammar: str | None,
    ) -> dict[str, Any]:
        if json_schema is not None:
            return {"guided_json": json_schema}
        if regex is not None:
            return {"guided_regex": regex}
        if grammar is not None:
            return {"guided_grammar": grammar}
        return {}


__all__ = ["VLLMBackend"]

#!/usr/bin/python
"""Shared OpenAI-compatible HTTP backend (CONCEPT:AHE-3.1).

vLLM and SGLang both serve an OpenAI-compatible ``/v1/completions`` endpoint, so
one HTTP client covers both engines; the concrete backends differ only in how
they name constrained-decoding params (overriding :meth:`_constrained_params`).

``httpx`` is imported lazily so the module stays importable without the
``data-science-mcp[training]`` extra (the rollout path is the only network user).

Concept: inference-backend
"""

from __future__ import annotations

from typing import Any

from data_science_mcp.inference.base import InferenceBackend


class OpenAICompatibleBackend(InferenceBackend):
    """Generate completions from an OpenAI-compatible ``/v1/completions`` server.

    Holds connection config (``base_url``, ``model``, ``api_key``, ``timeout``)
    and issues the HTTP request; constrained-decoding param mapping is delegated
    to :meth:`_constrained_params` so engine subclasses stay tiny.
    """

    provider = "openai-compatible"

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

    # -- engine-specific hooks ------------------------------------------------ #
    def _constrained_params(
        self,
        json_schema: dict[str, Any] | None,
        regex: str | None,
        grammar: str | None,
    ) -> dict[str, Any]:
        """Map a constrained-decoding spec to engine request params.

        The base backend does not support constrained decoding; subclasses that
        do (``VLLMBackend``, ``SGLangBackend``) override this.
        """
        return {}

    # -- generation ----------------------------------------------------------- #
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
        """Sample ``n`` completions with per-token logprobs (lazy ``httpx``)."""
        try:
            import httpx  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "httpx is required for served-model inference; install "
                "`data-science-mcp[training]`"
            ) from e

        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "logprobs": logprobs,
        }
        if json_schema is not None or regex is not None or grammar is not None:
            payload.update(self._constrained_params(json_schema, regex, grammar))
        payload.update(kwargs)

        resp = httpx.post(
            f"{self.base_url}/v1/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
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


__all__ = ["OpenAICompatibleBackend"]

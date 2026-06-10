#!/usr/bin/python
"""Inference-backend factory (CONCEPT:AHE-3.1).

Vendor-agnostic selection of a served-model client (vLLM / SGLang), mirroring
``agent_utilities.knowledge_graph.backends.create_backend``: resolve config from
explicit args first, then environment, then a sensible default. Both engines
speak the OpenAI HTTP protocol, so the runtime dependency is ``httpx`` only —
the inference server itself runs out-of-process.

Selection:
    provider:   ``INFERENCE_BACKEND``   (default ``"vllm"``; ``"sglang"`` to swap)
    base_url:   ``INFERENCE_BASE_URL``   (e.g. ``http://host:30000``)
    model:      ``INFERENCE_MODEL``      (served model id)
    api_key:    ``INFERENCE_API_KEY``    (default ``"EMPTY"`` for local servers)

Concept: inference-backend
"""

from __future__ import annotations

import os

from data_science_mcp.inference.base import InferenceBackend
from data_science_mcp.inference.openai_compatible import OpenAICompatibleBackend
from data_science_mcp.inference.sglang_backend import SGLangBackend
from data_science_mcp.inference.vllm_backend import VLLMBackend

_BACKENDS: dict[str, type[OpenAICompatibleBackend]] = {
    "vllm": VLLMBackend,
    "sglang": SGLangBackend,
}


def create_inference_backend(
    provider: str | None = None,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> InferenceBackend:
    """Create the configured inference backend.

    Resolves each value from the explicit argument, then the matching env var,
    then a default. ``provider`` falls back to ``INFERENCE_BACKEND`` then
    ``"vllm"``; an unknown provider raises ``ValueError``.

    Args:
        provider: ``"vllm"`` or ``"sglang"``. Falls back to ``INFERENCE_BACKEND``.
        base_url: server URL. Falls back to ``INFERENCE_BASE_URL``.
        model: served model id. Falls back to ``INFERENCE_MODEL``.
        api_key: bearer token. Falls back to ``INFERENCE_API_KEY`` then ``"EMPTY"``.
        timeout: HTTP timeout in seconds.

    Returns:
        A configured :class:`InferenceBackend`.
    """
    provider = (
        provider or os.environ.get("INFERENCE_BACKEND") or "vllm"
    ).lower().strip()
    backend_cls = _BACKENDS.get(provider)
    if backend_cls is None:
        raise ValueError(
            f"Unknown inference backend {provider!r}; "
            f"expected one of {sorted(_BACKENDS)}"
        )

    resolved_base_url = base_url or os.environ.get("INFERENCE_BASE_URL")
    if not resolved_base_url:
        raise ValueError(
            "No inference server URL configured; pass base_url= or set "
            "INFERENCE_BASE_URL (e.g. http://localhost:8000)"
        )
    resolved_model = model or os.environ.get("INFERENCE_MODEL") or ""
    resolved_api_key = api_key or os.environ.get("INFERENCE_API_KEY") or "EMPTY"

    return backend_cls(
        resolved_base_url,
        resolved_model,
        api_key=resolved_api_key,
        timeout=timeout,
    )


def inference_backend_configured() -> bool:
    """Whether a served-model endpoint is configured via the environment.

    Lets callers (e.g. the eval pipeline) decide between a backend-backed
    ``generate_fn`` and a no-op fallback without forcing an env var.
    """
    return bool(os.environ.get("INFERENCE_BASE_URL"))


__all__ = [
    "InferenceBackend",
    "OpenAICompatibleBackend",
    "VLLMBackend",
    "SGLangBackend",
    "create_inference_backend",
    "inference_backend_configured",
]

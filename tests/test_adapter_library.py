#!/usr/bin/python
"""Tests for LoRA specialist library + hot-swap serving (AHE-3.31)."""

from __future__ import annotations

import sys
import types


from data_science_mcp.adapter_library import (  # noqa: E402
    AdapterLibrary,
    SpecialistAdapter,
    served_generator,
)
from data_science_mcp.inference.openai_compatible import OpenAICompatibleBackend  # noqa: E402


def test_adapter_library_grows_with_many_specialists():
    lib = AdapterLibrary()
    lib.register("kernel:fused-softmax", "ad-1", "qwen2.5-1.5b")
    lib.register("world-model:latent", "ad-2", "qwen2.5-1.5b")
    assert len(lib) == 2  # N specialists coexist (no role overwrite)
    assert isinstance(lib.resolve("kernel:fused-softmax"), SpecialistAdapter)
    assert lib.resolve("world-model:latent").adapter_id == "ad-2"


def test_resolve_unknown_returns_none():
    assert AdapterLibrary().resolve("nope") is None


class _FakeBackend:
    """Records the adapter passed per call; returns a per-adapter canned text."""

    def __init__(self, responses: dict[str | None, str]) -> None:
        self.responses = responses
        self.adapters_seen: list[str | None] = []

    def generate(self, prompt, *, n=1, adapter=None, **kw):
        self.adapters_seen.append(adapter)
        return [{"text": self.responses.get(adapter, "base"), "logprobs": []}]


def test_served_generator_passes_adapter_to_backend():
    backend = _FakeBackend({"ad-1": "specialist output"})
    gen = served_generator(backend, "ad-1")
    assert gen("some scaffold") == "specialist output"
    assert backend.adapters_seen == ["ad-1"]


def test_backend_adapter_precedence_sets_served_model():
    """adapter arg > default_adapter > base model, asserted on the request payload."""
    captured = {}

    fake_httpx = types.ModuleType("httpx")

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"text": "ok", "logprobs": {"token_logprobs": []}}]}

    def _post(url, *, headers, json, timeout, **transport):
        captured["model"] = json["model"]
        return _Resp()

    fake_httpx.post = _post
    sys.modules["httpx"] = fake_httpx
    try:
        b = OpenAICompatibleBackend("http://x", "base-model", default_adapter="def-ad")
        b.generate("p", n=1)
        assert captured["model"] == "def-ad"  # default_adapter wins over base
        b.generate("p", n=1, adapter="call-ad")
        assert captured["model"] == "call-ad"  # explicit adapter wins over default
        b2 = OpenAICompatibleBackend("http://x", "base-model")
        b2.generate("p", n=1)
        assert captured["model"] == "base-model"  # no adapter → base model
    finally:
        del sys.modules["httpx"]

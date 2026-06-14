#!/usr/bin/python
"""Tests for LoRA specialist library + hot-swap serving (AHE-3.31)."""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("agent_utilities.models.model_registry", reason="dev agent-utilities")

from agent_utilities.models.model_registry import ModelRegistry  # noqa: E402

from data_science_mcp.adapter_library import (  # noqa: E402
    AdapterLibrary,
    SpecialistAdapter,
    served_generator,
)
from data_science_mcp.inference.openai_compatible import OpenAICompatibleBackend  # noqa: E402


def test_adapter_library_grows_and_registers_model_defs():
    registry = ModelRegistry(models=[])
    lib = AdapterLibrary(registry=registry)
    lib.register("kernel:fused-softmax", "ad-1", "qwen2.5-1.5b")
    lib.register("world-model:latent", "ad-2", "qwen2.5-1.5b")
    assert len(lib) == 2  # N specialists coexist (no role overwrite)
    assert isinstance(lib.resolve("kernel:fused-softmax"), SpecialistAdapter)
    ids = {m.id for m in registry.models}
    assert {"ad-1", "ad-2"} <= ids
    # adapter identity + base ride in model_id + tags (registry's extension point)
    ad1 = next(m for m in registry.models if m.id == "ad-1")
    assert ad1.model_id == "ad-1"
    assert "adapter:ad-1" in ad1.tags and "base:qwen2.5-1.5b" in ad1.tags
    assert "task:kernel:fused-softmax" in ad1.tags
    # routable by the existing tag-based picker
    picked = registry.pick_for_task(required_tags=["task:world-model:latent"])
    assert picked.id == "ad-2"


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

    def _post(url, *, headers, json, timeout):
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


def test_weight_arm_via_serving_feeds_the_factory():
    """The factory weight arm trains+serves an adapter and hands back a generator."""
    from agent_utilities.harness.sai_task import SpecializationTask, VerifierResult
    from agent_utilities.knowledge_graph.research.sai_factory import SaiFactoryController

    class _CountVerifier:
        def verify(self, candidate: str) -> VerifierResult:
            n = candidate.split().count("good")
            return VerifierResult(reward=min(n / 3.0, 1.0), passed=n > 0, detail={})

    task = SpecializationTask(
        task_id="kernel:demo",
        prompt_corpus=["weak"],
        verifier=_CountVerifier(),
        target_tau=0.9,
    )
    registry = ModelRegistry(models=[])
    lib = AdapterLibrary(registry=registry)
    backend = _FakeBackend({"sft-1": "good good good"})  # the trained specialist is strong

    def weight_arm(_task, _harvested):
        lib.register(_task.task_id, "sft-1", "qwen2.5-1.5b")
        return ("sft-1", served_generator(backend, "sft-1"))

    base_gen = lambda scaffold: "good"  # base model: weak (reward 1/3)  # noqa: E731
    controller = SaiFactoryController(task, base_gen, scaffolds=["weak"], weight_arm=weight_arm)
    result = controller.run(rounds=1)
    assert result.specialist.adapter_id == "sft-1"
    assert result.specialist.reward == pytest.approx(1.0)
    assert result.rounds[0].arm == "weights"
    assert lib.resolve("kernel:demo") is not None

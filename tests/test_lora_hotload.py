#!/usr/bin/python
"""Tests for the vLLM/SGLang dynamic-LoRA hot-load (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Closes the "trained but not servable without a manual restart" gap: once
``register_checkpoint`` binds a role, :func:`hot_load_adapter` best-effort POSTs
the checkpoint path to the serving engine's dynamic-LoRA endpoint
(``POST /v1/load_lora_adapter``) so it is immediately servable.

Pure-Python / CPU only: no torch, no GPU, no real server — the network layer is
exercised by injecting a fake ``httpx`` module (same pattern as
``test_inference_backends.py``) so the exact request payload can be asserted.
"""

from __future__ import annotations

import sys

import pytest

from data_science_mcp.training_pipeline import (
    DeploymentTarget,
    _deploy_checkpoint,
    _hotload_enabled,
    hot_load_adapter,
)

_HOTLOAD_ENV = "DATA_SCIENCE_MCP_LORA_HOTLOAD"


@pytest.fixture(autouse=True)
def _clean_hotload_env(monkeypatch):
    monkeypatch.delenv(_HOTLOAD_ENV, raising=False)


class _FakeResponse:
    def __init__(self, status_ok: bool = True) -> None:
        self._status_ok = status_ok

    def raise_for_status(self) -> None:
        if not self._status_ok:
            raise RuntimeError("boom: server rejected the adapter")


class _FakeHTTPX:
    def __init__(self, status_ok: bool = True) -> None:
        self.calls: list[dict] = []
        self._status_ok = status_ok

    def post(self, url, *, json=None, timeout=None, **transport):
        self.calls.append(
            {"url": url, "json": json, "timeout": timeout, "transport": transport}
        )
        return _FakeResponse(self._status_ok)


@pytest.fixture
def fake_httpx(monkeypatch):
    fake = _FakeHTTPX()
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


# --------------------------------------------------------------------------- #
# Feature gate                                                                 #
# --------------------------------------------------------------------------- #
def test_hotload_enabled_by_default():
    assert _hotload_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off"])
def test_hotload_disabled_via_env(monkeypatch, value):
    monkeypatch.setenv(_HOTLOAD_ENV, value)
    assert _hotload_enabled() is False


def test_hotload_skips_when_disabled(monkeypatch, fake_httpx):
    monkeypatch.setenv(_HOTLOAD_ENV, "0")
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    result = hot_load_adapter(
        target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1"
    )
    assert result == {"status": "skipped", "detail": "hot-load disabled via env"}
    assert fake_httpx.calls == []  # never even attempted the POST


# --------------------------------------------------------------------------- #
# Reuses the existing serving-endpoint config (DeploymentTarget.base_url)      #
# --------------------------------------------------------------------------- #
def test_hotload_skips_without_base_url(fake_httpx):
    target = DeploymentTarget(role="generator", served_model_name="m")  # no base_url
    result = hot_load_adapter(
        target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1"
    )
    assert result["status"] == "skipped"
    assert "base_url" in result["detail"]
    assert fake_httpx.calls == []


def test_hotload_skips_without_adapter_path(fake_httpx):
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    result = hot_load_adapter(target, adapter_name="ckpt-1", adapter_path="")
    assert result["status"] == "skipped"
    assert "adapter_path" in result["detail"]
    assert fake_httpx.calls == []


# --------------------------------------------------------------------------- #
# Live POST — the concrete /v1/load_lora_adapter payload (vLLM contract)       #
# --------------------------------------------------------------------------- #
def test_hotload_posts_correct_payload_to_vllm(fake_httpx):
    target = DeploymentTarget(
        role="generator",
        served_model_name="qwen2.5-1.5b-openseeker",
        # The OpenAI-SDK-style convention (already ends in /v1, as
        # ``ModelDefinition.base_url`` commonly does) — normalized to the server
        # root before the dynamic-LoRA endpoint is appended (see below).
        base_url="http://vllm:8000/v1",
        provider="vllm",
    )
    result = hot_load_adapter(
        target, adapter_name="ckpt-42", adapter_path="/models/lora/ckpt-42"
    )

    assert result["status"] == "loaded"
    assert result["provider"] == "vllm"
    assert result["adapter_name"] == "ckpt-42"
    assert "adapter_path" not in result
    assert "base_url" not in result

    assert len(fake_httpx.calls) == 1
    call = fake_httpx.calls[0]
    # No double ``/v1/v1`` even though base_url already carried a trailing /v1.
    assert call["url"] == "http://vllm:8000/v1/load_lora_adapter"
    assert call["json"] == {"lora_name": "ckpt-42", "lora_path": "/models/lora/ckpt-42"}


def test_hotload_normalizes_server_root_base_url_without_v1(fake_httpx):
    """The OTHER convention (server root, no ``/v1``) resolves to the SAME URL."""
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    hot_load_adapter(target, adapter_name="ckpt-42", adapter_path="/lora/ckpt-42")
    assert fake_httpx.calls[0]["url"] == "http://vllm:8000/v1/load_lora_adapter"


def test_hotload_posts_to_sglang_equivalent_endpoint(fake_httpx):
    target = DeploymentTarget(
        role="generator",
        served_model_name="m",
        base_url="http://sglang:30000",
        provider="sglang",
    )
    result = hot_load_adapter(
        target, adapter_name="ckpt-7", adapter_path="/models/lora/ckpt-7"
    )
    assert result["status"] == "loaded"
    assert result["provider"] == "sglang"
    assert fake_httpx.calls[0]["url"] == "http://sglang:30000/v1/load_lora_adapter"


def test_hotload_trims_trailing_slash_on_base_url(fake_httpx):
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000/"
    )
    hot_load_adapter(target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1")
    assert fake_httpx.calls[0]["url"] == "http://vllm:8000/v1/load_lora_adapter"


# --------------------------------------------------------------------------- #
# Degrade gracefully — never raises, never fails the training run             #
# --------------------------------------------------------------------------- #
def test_hotload_degrades_when_server_rejects(monkeypatch):
    fake = _FakeHTTPX(status_ok=False)
    monkeypatch.setitem(sys.modules, "httpx", fake)
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    # Must not raise even though the (fake) server call fails.
    result = hot_load_adapter(
        target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1"
    )
    assert result["status"] == "error"
    assert result["detail"] == "Operation failed"
    assert result["provider"] == "vllm"


def test_hotload_degrades_when_httpx_missing(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "httpx", None
    )  # simulates ImportError on `import httpx`
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    result = hot_load_adapter(
        target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1"
    )
    assert result["status"] == "skipped"
    assert "httpx" in result["detail"]


def test_hotload_degrades_on_connection_error(monkeypatch):
    class _RaisingHTTPX:
        def post(self, *a, **k):
            raise ConnectionError("server unreachable")

    monkeypatch.setitem(sys.modules, "httpx", _RaisingHTTPX())
    target = DeploymentTarget(
        role="generator", served_model_name="m", base_url="http://vllm:8000"
    )
    result = hot_load_adapter(
        target, adapter_name="ckpt-1", adapter_path="/lora/ckpt-1"
    )
    assert result["status"] == "error"
    assert result["detail"] == "Operation failed"


# --------------------------------------------------------------------------- #
# Wired into the shared deploy seam (_deploy_checkpoint)                      #
# --------------------------------------------------------------------------- #
def test_deploy_checkpoint_registers_role_and_hotloads(fake_httpx):
    from agent_utilities.models.model_registry import ModelRegistry

    reg = ModelRegistry()
    target = DeploymentTarget(
        role="generator",
        served_model_name="qwen2.5-1.5b-openseeker",
        base_url="http://vllm:8000",
    )
    deployment = _deploy_checkpoint(reg, target, "ckpt-1", "/models/lora/ckpt-1")

    # Role-bind side (existing deploy seam) is unaffected.
    assert deployment["role"] == "generator"
    assert reg.pick_for_role("generator").id == "ckpt-1"
    # Hot-load side rides along in the same report, best-effort.
    assert deployment["hotload"]["status"] == "loaded"
    assert fake_httpx.calls[0]["json"] == {
        "lora_name": "ckpt-1",
        "lora_path": "/models/lora/ckpt-1",
    }


def test_deploy_checkpoint_hotload_skips_without_base_url(fake_httpx):
    from agent_utilities.models.model_registry import ModelRegistry

    reg = ModelRegistry()
    target = DeploymentTarget(role="rlm-coder", served_model_name="m1")  # no base_url
    deployment = _deploy_checkpoint(reg, target, "ckpt-x", "/local/ckpt-x")

    assert reg.pick_for_role("rlm-coder").id == "ckpt-x"  # role bind still happened
    assert deployment["hotload"]["status"] == "skipped"
    assert fake_httpx.calls == []  # never attempted the POST

#!/usr/bin/python
"""Tests for the inference-backend abstraction (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Pure-Python / CPU only: no torch, no GPU, no real server. The network layer is
exercised by injecting a fake ``httpx`` module so we can assert the exact request
payload each engine sends (constrained-decoding param mapping) and the parsed
response shape.
"""

from __future__ import annotations

import sys

import pytest

from data_science_mcp.inference import (
    InferenceBackend,
    SGLangBackend,
    VLLMBackend,
    create_inference_backend,
    inference_backend_configured,
)

# Import the consumers at module load (before any fake-httpx fixture is active)
# so their heavy import chains resolve against the real environment, not the
# stubbed ``httpx`` a test installs.
from data_science_mcp.rollout_buffer import (  # noqa: E402
    RolloutBuffer,
    VLLMRolloutClient,
)
from data_science_mcp.training_pipeline import (  # noqa: E402
    _resolve_eval_generate_fn,
)

_INFERENCE_ENV = (
    "INFERENCE_BACKEND",
    "INFERENCE_BASE_URL",
    "INFERENCE_MODEL",
    "INFERENCE_API_KEY",
)


@pytest.fixture(autouse=True)
def _clean_inference_env(monkeypatch):
    """Each test starts with no inference env so resolution is deterministic."""
    for var in _INFERENCE_ENV:
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# Fake httpx (so `import httpx` inside generate() picks up our recorder)        #
# --------------------------------------------------------------------------- #
class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHTTPX:
    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    def post(self, url, *, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return _FakeResponse(self._payload)


@pytest.fixture
def fake_httpx(monkeypatch):
    payload = {
        "choices": [
            {"text": "hello", "logprobs": {"token_logprobs": [0.1, 0.2]}},
            {"text": "world", "logprobs": {"token_logprobs": [0.3]}},
        ]
    }
    fake = _FakeHTTPX(payload)
    monkeypatch.setitem(sys.modules, "httpx", fake)
    return fake


# --------------------------------------------------------------------------- #
# Factory resolution                                                           #
# --------------------------------------------------------------------------- #
def test_factory_default_provider_is_vllm():
    be = create_inference_backend(base_url="http://x:8000", model="m")
    assert isinstance(be, VLLMBackend)
    assert be.provider == "vllm"


def test_factory_env_selects_sglang(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "sglang")
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://x:30000")
    be = create_inference_backend()
    assert isinstance(be, SGLangBackend)
    assert be.base_url == "http://x:30000"


def test_factory_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "sglang")
    be = create_inference_backend("vllm", base_url="http://x:8000")
    assert isinstance(be, VLLMBackend)


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown inference backend"):
        create_inference_backend("tgi", base_url="http://x:8000")


def test_factory_missing_base_url_raises():
    with pytest.raises(ValueError, match="No inference server URL"):
        create_inference_backend("vllm")


def test_factory_api_key_default_and_override(monkeypatch):
    be = create_inference_backend("vllm", base_url="http://x:8000")
    assert be.api_key == "EMPTY"
    monkeypatch.setenv("INFERENCE_API_KEY", "secret")
    assert create_inference_backend("vllm", base_url="http://x:8000").api_key == "secret"


def test_inference_backend_configured(monkeypatch):
    assert inference_backend_configured() is False
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://x:8000")
    assert inference_backend_configured() is True


# --------------------------------------------------------------------------- #
# Constrained-decoding param mapping (pure, no network)                        #
# --------------------------------------------------------------------------- #
def test_vllm_constrained_param_names():
    be = VLLMBackend("http://x:8000", "m")
    assert be.supports_constrained is True
    assert be._constrained_params({"type": "object"}, None, None) == {
        "guided_json": {"type": "object"}
    }
    assert be._constrained_params(None, "[0-9]+", None) == {"guided_regex": "[0-9]+"}
    assert be._constrained_params(None, None, "root ::= 'a'") == {
        "guided_grammar": "root ::= 'a'"
    }


def test_sglang_constrained_param_names():
    be = SGLangBackend("http://x:30000", "m")
    assert be.supports_constrained is True
    assert be._constrained_params({"type": "object"}, None, None) == {
        "json_schema": {"type": "object"}
    }
    assert be._constrained_params(None, "[0-9]+", None) == {"regex": "[0-9]+"}
    assert be._constrained_params(None, None, "root ::= 'a'") == {
        "ebnf": "root ::= 'a'"
    }


# --------------------------------------------------------------------------- #
# generate(): payload construction + response parsing                          #
# --------------------------------------------------------------------------- #
def test_generate_parses_choices(fake_httpx):
    be = VLLMBackend("http://x:8000/", "m")
    outs = be.generate("hi", n=2)
    assert outs == [
        {"text": "hello", "logprobs": [0.1, 0.2]},
        {"text": "world", "logprobs": [0.3]},
    ]
    sent = fake_httpx.calls[0]
    assert sent["url"] == "http://x:8000/v1/completions"  # trailing slash trimmed
    assert sent["json"]["model"] == "m"
    assert sent["json"]["prompt"] == "hi"
    assert sent["json"]["n"] == 2
    assert sent["headers"]["Authorization"] == "Bearer EMPTY"


def test_vllm_generate_sends_guided_json(fake_httpx):
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    VLLMBackend("http://x:8000", "m").generate("p", json_schema=schema)
    assert fake_httpx.calls[0]["json"]["guided_json"] == schema
    assert "json_schema" not in fake_httpx.calls[0]["json"]


def test_sglang_generate_sends_json_schema(fake_httpx):
    schema = {"type": "object"}
    SGLangBackend("http://x:30000", "m").generate("p", json_schema=schema)
    assert fake_httpx.calls[0]["json"]["json_schema"] == schema
    assert "guided_json" not in fake_httpx.calls[0]["json"]


def test_generate_omits_constrained_params_when_unset(fake_httpx):
    VLLMBackend("http://x:8000", "m").generate("p")
    sent = fake_httpx.calls[0]["json"]
    assert "guided_json" not in sent and "guided_regex" not in sent


# --------------------------------------------------------------------------- #
# as_generate_fn() (eval adapter)                                              #
# --------------------------------------------------------------------------- #
def test_as_generate_fn_returns_first_text(fake_httpx):
    fn = VLLMBackend("http://x:8000", "m").as_generate_fn()
    assert fn("anything") == "hello"
    sent = fake_httpx.calls[0]["json"]
    assert sent["n"] == 1 and sent["temperature"] == 0.0


# --------------------------------------------------------------------------- #
# RolloutBuffer contract parity + back-compat alias                            #
# --------------------------------------------------------------------------- #
def test_rollout_buffer_drains_a_backend(fake_httpx):
    buf = RolloutBuffer()
    buf.generate(["q1", "q2"], SGLangBackend("http://x:30000", "m"), n=2)
    assert len(buf) == 4  # 2 prompts x 2 completions each
    assert set(buf.prompts) == {"q1", "q2"}


def test_vllm_rollout_client_is_backend_alias(fake_httpx):
    assert VLLMRolloutClient is VLLMBackend
    client = VLLMRolloutClient("http://x:8000", "m")
    assert isinstance(client, InferenceBackend)
    assert client.generate("p", n=1)[0]["text"] == "hello"


# --------------------------------------------------------------------------- #
# Eval-path seam: backend-backed when configured, echo otherwise               #
# --------------------------------------------------------------------------- #
def test_resolve_eval_generate_fn_prefers_injected():
    fn = _resolve_eval_generate_fn(lambda s: f"injected:{s}")
    assert fn("q") == "injected:q"


def test_resolve_eval_generate_fn_echoes_without_server():
    fn = _resolve_eval_generate_fn(None)
    assert fn("q") == "q"  # no INFERENCE_BASE_URL -> echo fallback


def test_resolve_eval_generate_fn_uses_backend_when_configured(
    monkeypatch, fake_httpx
):
    monkeypatch.setenv("INFERENCE_BACKEND", "sglang")
    monkeypatch.setenv("INFERENCE_BASE_URL", "http://x:30000")
    monkeypatch.setenv("INFERENCE_MODEL", "m")
    fn = _resolve_eval_generate_fn(None)
    assert fn("q") == "hello"  # served-model response, not echo
    assert fake_httpx.calls[0]["url"] == "http://x:30000/v1/completions"

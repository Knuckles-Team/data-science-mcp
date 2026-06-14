#!/usr/bin/python
"""Chat + reasoning format for from-scratch RLHF (CONCEPT:ML-012).

A dependency-free, pure-string chat template using **learnable role markers** and a
``<think>…</think><answer>…</answer>`` reasoning format. The markers are ordinary
text tokens — no tokenizer special-token surgery — so the same format works for a
model trained from scratch (with our own BPE tokenizer) and for an HF base. Used to
pack SFT targets (supervise only the assistant span) and to frame PPO/GRPO rollouts;
:func:`extract_answer` is the verifier hook for reward (e.g. GSM8K exact-match).

This module is pure Python (no torch) so it is unit-testable on CPU and importable
without the training extra.

Concept: chat-template
"""

from __future__ import annotations

import re
from typing import Any

USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SYSTEM = "<|system|>"
END = "<|end|>"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

#: The marker vocabulary to register as ``extra_tokens`` when training a tokenizer
#: (CONCEPT:ML-003) so they stay single, stable tokens.
ROLE_MARKERS: tuple[str, ...] = (
    SYSTEM,
    USER,
    ASSISTANT,
    END,
    THINK_OPEN,
    THINK_CLOSE,
    ANSWER_OPEN,
    ANSWER_CLOSE,
)

_ROLE_PREFIX = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}


def render_chat(
    messages: list[dict[str, str]], *, add_generation_prompt: bool = False
) -> str:
    """Render ``[{role, content}, …]`` into the marker format.

    With ``add_generation_prompt=True`` the string ends with a trailing
    ``<|assistant|>`` so a model can continue from the assistant turn (rollout /
    inference); otherwise each turn is closed with ``<|end|>``.
    """
    parts: list[str] = []
    for m in messages:
        prefix = _ROLE_PREFIX.get(str(m.get("role", "user")), USER)
        parts.append(f"{prefix}{m.get('content', '')}{END}")
    if add_generation_prompt:
        parts.append(ASSISTANT)
    return "".join(parts)


def render_think_answer(think: str, answer: str) -> str:
    """Wrap a reasoning chain + final answer in the ``<think>/<answer>`` format."""
    return f"{THINK_OPEN}{think}{THINK_CLOSE}{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"


_ANSWER_RE = re.compile(
    re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.DOTALL
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Return the content of the **last** ``<answer>…</answer>`` span, else ``None``.

    The verifier hook: PPO/GRPO reward and GSM8K eval read the model's final answer
    from here. When no ``<answer>`` tag is present, falls back to the last number in
    the text (common for un-templated completions).
    """
    matches = _ANSWER_RE.findall(text or "")
    if matches:
        return matches[-1].strip()
    nums = _NUMBER_RE.findall(text or "")
    return nums[-1] if nums else None


def _normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    val = m.group(0).replace(",", "")
    try:  # canonicalise "42.0" == "42"
        f = float(val)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:  # pragma: no cover - defensive
        return val


def gsm8k_gold(answer_field: str) -> str | None:
    """Extract the gold numeric answer from a GSM8K ``answer`` (the part after ``####``)."""
    tail = (answer_field or "").split("####")[-1]
    return _normalize_number(tail)


def answer_matches(completion: str, gold: Any) -> bool:
    """True when the model's extracted final answer equals ``gold`` numerically."""
    pred = _normalize_number(extract_answer(completion))
    want = _normalize_number(str(gold)) if not isinstance(gold, str) else _normalize_number(gold)
    return pred is not None and want is not None and pred == want


__all__ = [
    "USER",
    "ASSISTANT",
    "SYSTEM",
    "END",
    "THINK_OPEN",
    "THINK_CLOSE",
    "ANSWER_OPEN",
    "ANSWER_CLOSE",
    "ROLE_MARKERS",
    "render_chat",
    "render_think_answer",
    "extract_answer",
    "gsm8k_gold",
    "answer_matches",
]

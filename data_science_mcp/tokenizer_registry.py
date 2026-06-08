#!/usr/bin/python
"""Special / functional token injection + embedding resize (CONCEPT:AHE-3.1).

Several training-gated papers add **new tokens** to the vocabulary before
fine-tuning: ATLAS/MedCausalX/SDAR introduce *functional* tokens (learned action /
tool-call markers that later carry the GRPO reward) and structural special tokens.
Adding tokens requires resizing the model's embedding matrix; this module keeps a
declarative registry of the tokens to inject, computes the resize **plan** (which
tokens are genuinely new vs. already in the vocab) as pure data, and applies it to
a real tokenizer/model when asked.

The plan logic is pure (CPU, no deps) and unit-testable; :meth:`apply` touches a
real HF tokenizer/model only when called.

Concept: tokenizer-registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResizePlan:
    """Result of planning a vocabulary extension."""

    new_special_tokens: list[str] = field(default_factory=list)
    new_functional_tokens: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    old_vocab_size: int = 0

    @property
    def added_count(self) -> int:
        return len(self.new_special_tokens) + len(self.new_functional_tokens)

    @property
    def new_vocab_size(self) -> int:
        return self.old_vocab_size + self.added_count


class TokenizerRegistry:
    """Collect special/functional tokens and apply them to a tokenizer+model."""

    def __init__(self) -> None:
        self._special: list[str] = []
        self._functional: list[str] = []

    def register(self, *tokens: str, functional: bool = False) -> "TokenizerRegistry":
        """Register one or more tokens (deduped, order-preserving). Chainable.

        ``functional=True`` marks reward-bearing action tokens (ATLAS LA-GRPO); they
        are still added to the vocab but tracked separately so the trainer can build
        a functional-token mask.
        """
        bucket = self._functional if functional else self._special
        other = self._special if functional else self._functional
        for tok in tokens:
            if tok and tok not in bucket and tok not in other:
                bucket.append(tok)
        return self

    @property
    def functional_tokens(self) -> list[str]:
        return list(self._functional)

    @property
    def all_tokens(self) -> list[str]:
        return self._special + self._functional

    def plan(self, existing_vocab: set[str] | dict[str, Any]) -> ResizePlan:
        """Compute which registered tokens are new given the current vocab."""
        vocab = set(existing_vocab)
        plan = ResizePlan(old_vocab_size=len(vocab))
        for tok in self._special:
            (plan.already_present if tok in vocab else plan.new_special_tokens).append(
                tok
            )
        for tok in self._functional:
            (
                plan.already_present if tok in vocab else plan.new_functional_tokens
            ).append(tok)
        return plan

    def apply(self, tokenizer: Any, model: Any | None = None) -> ResizePlan:
        """Add the registered tokens to ``tokenizer`` and resize ``model`` embeddings.

        Returns the :class:`ResizePlan` actually applied. Idempotent: tokens already
        in the vocab are skipped, so re-applying adds nothing.
        """
        vocab = tokenizer.get_vocab()
        plan = self.plan(vocab)
        new_tokens = plan.new_special_tokens + plan.new_functional_tokens
        if new_tokens:
            tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
        return plan

    def functional_token_ids(self, tokenizer: Any) -> list[int]:
        """Resolve the functional tokens to their ids in ``tokenizer`` (for masks)."""
        ids: list[int] = []
        for tok in self._functional:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0:
                ids.append(tid)
        return ids


__all__ = ["TokenizerRegistry", "ResizePlan"]

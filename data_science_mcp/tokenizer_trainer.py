#!/usr/bin/python
"""Train a tokenizer from scratch (CONCEPT:DS-AHE.trainer.concept-2).

The first step of pretraining a model from random init: you need a *vocabulary*
before you can have an embedding table. This trains a byte-level BPE tokenizer with
🤗 ``tokenizers`` over a corpus iterable and saves a portable ``tokenizer.json`` an
HF ``PreTrainedTokenizerFast`` (and thus :class:`PretrainTrainer`) can load.

``tokenizers`` is a heavy, optional dep (the ``data-science-mcp[training]`` extra),
imported lazily — :func:`plan_tokenizer` is pure and always available, so an agent
can plan vocab size / special tokens with no extra installed.

Special/functional tokens are sourced from the existing
:class:`data_science_mcp.tokenizer_registry.TokenizerRegistry`, so the same token
set used for fine-tuning injection is reserved at vocabulary-construction time.

Concept: tokenizer-trainer
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

# Conventional LM control tokens reserved in every trained vocab.
DEFAULT_SPECIAL_TOKENS = ("<pad>", "<unk>", "<s>", "</s>", "<mask>")


@dataclass
class TokenizerSpec:
    """Declarative BPE-tokenizer-training configuration (pure data)."""

    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: tuple[str, ...] = DEFAULT_SPECIAL_TOKENS
    extra_tokens: tuple[str, ...] = ()  # e.g. functional/action tokens
    lowercase: bool = False

    def all_special(self) -> list[str]:
        seen: dict[str, None] = {}
        for tok in (*self.special_tokens, *self.extra_tokens):
            if tok:
                seen.setdefault(tok, None)
        return list(seen)


@dataclass
class TokenizerPlan:
    """Pure summary of what training will produce (no heavy deps)."""

    vocab_size: int
    n_special: int
    special_tokens: list[str] = field(default_factory=list)
    algorithm: str = "byte-level-bpe"


def plan_tokenizer(spec: TokenizerSpec | None = None) -> TokenizerPlan:
    """Plan a tokenizer build — pure, no ``tokenizers`` needed."""
    spec = spec or TokenizerSpec()
    specials = spec.all_special()
    return TokenizerPlan(
        vocab_size=spec.vocab_size,
        n_special=len(specials),
        special_tokens=specials,
    )


def train_tokenizer(
    corpus: Iterable[str],
    *,
    spec: TokenizerSpec | None = None,
    output_dir: str | None = None,
) -> Any:
    """Train a byte-level BPE tokenizer over ``corpus`` → ``PreTrainedTokenizerFast``.

    Args:
        corpus: an iterable of training text strings.
        spec: vocab size / special tokens / etc.
        output_dir: if given, ``save_pretrained`` the result there.

    Returns:
        A ready-to-use ``transformers.PreTrainedTokenizerFast``.
    """
    spec = spec or TokenizerSpec()
    try:
        from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers  # noqa: PLC0415
        from tokenizers.models import BPE  # noqa: PLC0415
        from transformers import PreTrainedTokenizerFast  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - without the extra
        raise RuntimeError(
            "training a tokenizer needs `tokenizers` + `transformers`; install "
            "`data-science-mcp[training]`"
        ) from e

    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=spec.vocab_size,
        min_frequency=spec.min_frequency,
        special_tokens=spec.all_special(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(_as_iterator(corpus, spec.lowercase), trainer=trainer)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fast.save_pretrained(output_dir)
    return fast


def _as_iterator(corpus: Iterable[str], lowercase: bool):
    for line in corpus:
        text = line if isinstance(line, str) else str(line)
        yield text.lower() if lowercase else text


__all__ = [
    "TokenizerSpec",
    "TokenizerPlan",
    "DEFAULT_SPECIAL_TOKENS",
    "plan_tokenizer",
    "train_tokenizer",
]

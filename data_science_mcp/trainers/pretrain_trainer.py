#!/usr/bin/python
"""Pretrain a causal LM from random init (CONCEPT:ML-003).

The "from scratch" trainer: it builds a model with **random weights** from an
architecture spec (``AutoConfig`` → ``AutoModelForCausalLM.from_config`` — NOT
``from_pretrained``) and trains a next-token objective over **packed** token
sequences. Targeted at small models (≤~1.5B) that are genuinely pretrainable on
homelab/cloud-burst GPUs; the architecture knobs (:class:`PretrainSpec`) scale up
to whatever compute is available.

It reuses the whole hardened spine: the shared :func:`run_loop` (precision /
accumulation / clipping / LR schedule / long-horizon checkpoint+resume / FSDP /
DeepSpeed) and the same model-registry deploy seam as the fine-tuners.

For a CPU smoke, inject a tiny toy model + tokenizer (as the SFT/DPO/GRPO tests do);
the live path builds the real random-init model when ``transformers`` is present.

Concept: pretrain-trainer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from data_science_mcp.trainers.base import TrainConfig, TrainerBase, _torch
from data_science_mcp.trainers.objectives import sft_cross_entropy


@dataclass
class PretrainSpec:
    """Random-init architecture knobs (a small-LM config by default)."""

    model_type: str = "llama"
    vocab_size: int = 32000
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None  # defaults to 4×hidden when None
    max_position_embeddings: int = 2048

    def build_config(self) -> Any:
        """Build an HF ``AutoConfig`` for the architecture (lazy import)."""
        try:
            from transformers import AutoConfig  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "transformers is required to build a model config; install "
                "`data-science-mcp[training]`"
            ) from e
        inter = self.intermediate_size or 4 * self.hidden_size
        return AutoConfig.for_model(
            self.model_type,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads or self.num_attention_heads,
            intermediate_size=inter,
            max_position_embeddings=self.max_position_embeddings,
        )

    def build_model(self, attn_impl: str | None = None) -> Any:
        """Instantiate a **random-init** causal LM from the config (lazy import)."""
        try:
            from transformers import AutoModelForCausalLM  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover - without the extra
            raise RuntimeError(
                "transformers is required to build a model; install "
                "`data-science-mcp[training]`"
            ) from e
        cfg = self.build_config()
        kwargs: dict[str, Any] = {}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        return AutoModelForCausalLM.from_config(cfg, **kwargs)

    def approx_params(self) -> int:
        """Rough parameter count (embeddings + transformer blocks) — pure."""
        inter = self.intermediate_size or 4 * self.hidden_size
        h, L, V = self.hidden_size, self.num_hidden_layers, self.vocab_size
        per_layer = 4 * h * h + 3 * h * inter  # attn proj + MLP
        return 2 * V * h + L * per_layer  # tied-ish embeddings + blocks


class PretrainTrainer(TrainerBase):
    """Causal-LM pretraining from random init over packed sequences."""

    name = "pretrain"
    kind = "pretrain"

    def __init__(
        self, config: TrainConfig | None = None, spec: PretrainSpec | None = None
    ) -> None:
        super().__init__(config)
        self.spec = spec or PretrainSpec()

    def _resolve(self, model: Any | None, tokenizer: Any | None) -> tuple[Any, Any]:
        """Build the random-init model + load the (trained) tokenizer when not injected."""
        if model is not None and tokenizer is not None:
            return model, tokenizer
        from transformers import AutoTokenizer  # noqa: PLC0415

        tok = tokenizer or AutoTokenizer.from_pretrained(self.config.base_model)
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        # Keep the spec's vocab in sync with the tokenizer that will feed it.
        self.spec.vocab_size = len(tok)
        built = model or self.spec.build_model(attn_impl=self.config.attn_impl)
        return built, tok

    def _block_ids(self, tokenizer: Any, texts: list[str]) -> list[list[int]]:
        """Tokenize each text to a raw id list (no padding) for packing."""
        out: list[list[int]] = []
        for t in texts:
            enc = tokenizer([t], truncation=False)
            ids = enc["input_ids"][0]
            out.append(list(ids.tolist()) if hasattr(ids, "tolist") else list(ids))
        return out

    def train(
        self,
        dataset: list[dict[str, Any]],
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        optimizer: Any | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """Pretrain over ``{text}`` records (packed into ``max_seq_len`` blocks)."""
        from data_science_mcp.data_engine import pack_sequences  # noqa: PLC0415
        from data_science_mcp.trainers.loop import run_loop  # noqa: PLC0415

        torch = _torch()
        if not dataset:
            return {"trainer": self.name, "steps": 0, "examples": 0, "losses": []}
        torch.manual_seed(self.config.seed)
        model, tokenizer = self._resolve(model, tokenizer)
        device = self._device()
        model.to(device)
        model.train()
        self._enable_runtime(model)

        # Pack the corpus into fixed-length blocks (the pretraining throughput trick).
        texts = [str(r.get("text", r.get("prompt", ""))) for r in dataset]
        eos_id = getattr(tokenizer, "eos_token_id", None)
        id_lists = self._block_ids(tokenizer, texts)
        blocks = pack_sequences(
            id_lists, self.config.max_seq_len, eos_id=eos_id, drop_remainder=False
        )
        blocks = [b for b in blocks if len(b) >= 2]  # need ≥2 tokens to form a target
        if not blocks:
            return {
                "trainer": self.name,
                "steps": 0,
                "examples": len(dataset),
                "losses": [],
            }

        opt = self._optimizer(model, optimizer)
        accel, model, opt = self._prepare(model, opt)
        n_batches = math.ceil(len(blocks) / max(1, self.config.batch_size))
        total = self._total_opt_steps(n_batches)
        sched = self._scheduler(opt, total)
        tracker = self._tracker(
            {
                "trainer": self.name,
                "base_model": self.config.base_model,
                "approx_params": self.spec.approx_params(),
                "blocks": len(blocks),
                "max_seq_len": self.config.max_seq_len,
                "lr": self.config.lr,
                "precision": self.config.precision,
            }
        )
        pad_id = getattr(tokenizer, "pad_token_id", None)

        def compute_loss(batch: list[list[int]]) -> Any:
            width = max(len(b) for b in batch)
            pad = pad_id if pad_id is not None else 0
            input_ids = torch.tensor(
                [b + [pad] * (width - len(b)) for b in batch],
                dtype=torch.long,
                device=device,
            )
            attn = torch.tensor(
                [[1] * len(b) + [0] * (width - len(b)) for b in batch],
                dtype=torch.long,
                device=device,
            )
            labels = input_ids.clone()
            labels[attn == 0] = -100
            logits = model(input_ids=input_ids, attention_mask=attn).logits
            return sft_cross_entropy(logits, labels)

        out = run_loop(
            config=self.config,
            model=model,
            optimizer=opt,
            device=device,
            epoch_items=lambda: self._batches(blocks),
            compute_loss=compute_loss,
            scheduler=sched,
            accelerator=accel,
            tracker=tracker,
            total_steps=total,
        )
        losses = out["losses"]
        report = {
            "trainer": self.name,
            "kind": self.kind,
            "examples": len(dataset),
            "blocks": len(blocks),
            "steps": out["steps"],
            "losses": losses,
            "final_loss": losses[-1] if losses else None,
            "approx_params": self.spec.approx_params(),
            "base_model": self.config.base_model,
            "checkpoints": out["checkpoints"],
            "resumed_from_step": out["resumed_from_step"],
        }
        tracker.end({"final_loss": report["final_loss"], "steps": out["steps"]})
        return report


def build_pretrain_trainer(
    config: TrainConfig | None = None, spec: PretrainSpec | None = None
) -> PretrainTrainer:
    return PretrainTrainer(config, spec)


__all__ = ["PretrainSpec", "PretrainTrainer", "build_pretrain_trainer"]

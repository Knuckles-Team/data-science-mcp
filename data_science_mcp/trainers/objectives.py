#!/usr/bin/python
"""Differentiable training objectives — torch loss kernels (CONCEPT:AHE-3.1).

The gradient half of the in-house training substrate (Wave C). Where
:mod:`data_science_mcp.training_data` builds the *corpora* (deterministic, no GPU)
and :mod:`agent_utilities.graph.training_signals` is the pure-Python *reference*
math, this module is the torch implementation the trainers actually backprop
through:

* :func:`sft_cross_entropy` — next-token CE with label masking (OpenSeeker/MeMo SFT).
* :func:`sequence_logprob`   — summed per-token log-prob of a completion under a model.
* :func:`dpo_loss`           — Bradley-Terry preference loss with a frozen reference (MedCausalX DPO).
* :func:`grpo_surrogate`     — group-relative clipped policy-gradient surrogate (ATLAS/SDAR GRPO).
* :func:`token_masked_surrogate` — GRPO surrogate restricted to functional tokens (ATLAS LA-GRPO).
* :func:`approx_kl`          — k3 (Schulman) low-variance KL estimate for the GRPO penalty.

These mirror the C1 Rust ``candle`` kernels named in ``WAVE_C_INFRA.md`` §C1; the
Rust path is a later drop-in for throughput. Every function is a pure tensor
op — unit-testable on toy tensors on CPU, no model or GPU required.

Concept: training-objectives
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep torch out of import time for callers that only need types
    import torch


def _torch():
    try:
        import torch  # noqa: PLC0415

        return torch
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "torch is required for training objectives; install "
            "`data-science-mcp[training]`"
        ) from e


def sft_cross_entropy(
    logits: "torch.Tensor", labels: "torch.Tensor", *, ignore_index: int = -100
) -> "torch.Tensor":
    """Causal next-token cross-entropy.

    ``logits`` is ``(batch, seq, vocab)`` and ``labels`` is ``(batch, seq)``; both
    are shifted so position *t* predicts token *t+1*. Positions set to
    ``ignore_index`` (padding / prompt tokens) do not contribute.
    """
    torch = _torch()
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def sequence_logprob(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
    *,
    ignore_index: int = -100,
) -> "torch.Tensor":
    """Summed log-prob of each sequence's (non-masked) tokens → ``(batch,)``.

    The DPO building block: ``log π(y|x)`` for a completion. Shifts like
    :func:`sft_cross_entropy`; masked positions contribute 0.
    """
    torch = _torch()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != ignore_index
    safe = shift_labels.masked_fill(~mask, 0).unsqueeze(-1)
    token_logp = logp.gather(-1, safe).squeeze(-1)
    return (token_logp * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logp: "torch.Tensor",
    policy_rejected_logp: "torch.Tensor",
    ref_chosen_logp: "torch.Tensor",
    ref_rejected_logp: "torch.Tensor",
    *,
    beta: float = 0.1,
) -> "torch.Tensor":
    """Bradley-Terry DPO loss (mean over the batch).

    ``-log σ(β · ((logπ_c − logπ_r) − (logπ_ref,c − logπ_ref,r)))``. The reference
    log-probs come from a frozen copy of the base model; ``beta`` controls how far
    the policy may drift. Returns a scalar.
    """
    torch = _torch()
    pi_logratio = policy_chosen_logp - policy_rejected_logp
    ref_logratio = ref_chosen_logp - ref_rejected_logp
    return -torch.nn.functional.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()


def grpo_surrogate(
    logprob: "torch.Tensor",
    old_logprob: "torch.Tensor",
    advantage: "torch.Tensor",
    *,
    clip_eps: float = 0.2,
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """PPO/GRPO clipped surrogate (loss to *minimise*, mean over valid tokens).

    ``ratio = exp(logprob − old_logprob)``; the objective is
    ``min(ratio·A, clip(ratio, 1±ε)·A)`` and we return its negation so callers can
    minimise. Advantages are group-normalised upstream
    (:func:`data_science_mcp.training_data.build_grpo_groups`). An optional ``mask``
    (1 for tokens that count) supports token-level GRPO.
    """
    torch = _torch()
    ratio = torch.exp(logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    obj = torch.minimum(unclipped, clipped)
    if mask is not None:
        denom = mask.sum().clamp_min(1.0)
        return -(obj * mask).sum() / denom
    return -obj.mean()


def token_masked_surrogate(
    logprob: "torch.Tensor",
    old_logprob: "torch.Tensor",
    advantage: "torch.Tensor",
    functional_mask: "torch.Tensor",
    *,
    clip_eps: float = 0.2,
) -> "torch.Tensor":
    """ATLAS LA-GRPO: clipped surrogate credited only on *functional* tokens.

    ``functional_mask`` marks the special/functional tokens (tool calls, learned
    action tokens) that should receive the policy-gradient signal; ordinary tokens
    are excluded from the average.
    """
    return grpo_surrogate(
        logprob, old_logprob, advantage, clip_eps=clip_eps, mask=functional_mask
    )


def approx_kl(logprob: "torch.Tensor", ref_logprob: "torch.Tensor") -> "torch.Tensor":
    """Schulman k3 low-variance KL estimate ``E[ (r−1) − log r ]`` (≥0, mean).

    ``r = exp(ref_logprob − logprob)``. Used as the GRPO/SDAR KL-to-reference
    penalty; unbiased and non-negative unlike the naive ``logprob − ref_logprob``.
    """
    torch = _torch()
    log_r = ref_logprob - logprob
    r = torch.exp(log_r)
    return (r - 1.0 - log_r).mean()

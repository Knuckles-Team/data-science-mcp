#!/usr/bin/python
"""Differentiable training objectives — torch loss kernels (CONCEPT:AHE-3.1).

The gradient half of the in-house training substrate (Wave C). Where
:mod:`data_science_mcp.training_data` builds the *corpora* (deterministic, no GPU)
and :mod:`agent_utilities.graph.training_signals` is the pure-Python *reference*
math, this module is the torch implementation the trainers actually backprop
through:

* :func:`sft_cross_entropy` — next-token CE with label masking (OpenSeeker/MeMo SFT).
* :func:`sequence_logprob`   — summed per-token log-prob of a completion under a model.
* :func:`token_logprob`      — per-token log-prob of the realised tokens → ``(batch, seq-1)`` (PPO).
* :func:`dpo_loss`           — Bradley-Terry preference loss with a frozen reference (MedCausalX DPO).
* :func:`bradley_terry_loss` — pairwise reward-model loss ``-logσ(s_chosen − s_rejected)`` (CONCEPT:ML-008).
* :func:`grpo_surrogate`     — group-relative clipped policy-gradient surrogate (ATLAS/SDAR GRPO; reused by PPO).
* :func:`token_masked_surrogate` — GRPO surrogate restricted to functional tokens (ATLAS LA-GRPO).
* :func:`gae`                — generalized advantage estimation over per-token rewards/values (CONCEPT:ML-009 PPO).
* :func:`value_function_loss`— (optionally clipped) value-head MSE regression to GAE returns (CONCEPT:ML-009 PPO).
* :func:`whiten`             — mean-0 / var-1 normalisation of advantages (PPO stability).
* :func:`approx_kl`          — k3 (Schulman) low-variance KL estimate for the GRPO/PPO penalty.

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


def token_logprob(
    logits: "torch.Tensor",
    labels: "torch.Tensor",
) -> "torch.Tensor":
    """Per-token log-prob of the realised next token → ``(batch, seq-1)``.

    The PPO building block: unlike :func:`sequence_logprob` (which sums to one
    scalar per sequence) this keeps the per-position log-prob ``log π(t+1 | ≤t)`` so
    PPO can form per-token ratios and a per-token KL penalty. Shifted so position
    *t* scores token *t+1*; callers mask padding/prompt positions themselves.
    """
    torch = _torch()
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits, dim=-1)
    return logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)


def bradley_terry_loss(
    chosen_scores: "torch.Tensor",
    rejected_scores: "torch.Tensor",
    *,
    margin: float = 0.0,
) -> "torch.Tensor":
    """Pairwise reward-model loss ``-logσ(s_chosen − s_rejected − margin)`` (CONCEPT:ML-008).

    ``chosen_scores`` / ``rejected_scores`` are ``(batch,)`` scalar rewards read off
    the reward head (last real token). Trains the head so a preferred response
    scores higher than its rejected partner; ``margin`` optionally enforces a gap.
    Returns a scalar (mean over the batch).
    """
    torch = _torch()
    return -torch.nn.functional.logsigmoid(
        chosen_scores - rejected_scores - margin
    ).mean()


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


def whiten(x: "torch.Tensor", mask: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Mean-0 / unit-variance normalise ``x`` (over the masked elements).

    PPO whitens advantages before the surrogate to stabilise the policy-gradient
    scale across batches. With a ``mask`` (1 = valid token) only the valid entries
    drive the statistics; invalid entries are still returned (callers re-mask).
    """
    torch = _torch()
    if mask is not None:
        n = mask.sum().clamp_min(1.0)
        mean = (x * mask).sum() / n
        var = (((x - mean) * mask) ** 2).sum() / n
    else:
        mean = x.mean()
        var = x.var(unbiased=False)
    return (x - mean) / torch.sqrt(var + 1e-8)


def gae(
    rewards: "torch.Tensor",
    values: "torch.Tensor",
    *,
    mask: "torch.Tensor | None" = None,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Generalized Advantage Estimation over per-token rewards/values (CONCEPT:ML-009).

    ``rewards`` / ``values`` are ``(batch, seq)`` per-token tensors (``values`` from
    the value head; ``rewards`` typically the terminal reward at the last response
    token plus a per-token KL penalty). Returns ``(advantages, returns)`` of the same
    shape, where ``returns = advantages + values`` are the value-head regression
    targets. Computed with a reverse recursion
    ``δ_t = r_t + γ V_{t+1} − V_t``, ``A_t = δ_t + γλ A_{t+1}``; an optional ``mask``
    (1 = valid token) zeroes bootstrap across padding.
    """
    torch = _torch()
    m = mask if mask is not None else torch.ones_like(values)
    adv = torch.zeros_like(values)
    last = torch.zeros(values.size(0), dtype=values.dtype, device=values.device)
    seq = values.size(1)
    for t in range(seq - 1, -1, -1):
        next_v = values[:, t + 1] * m[:, t + 1] if t + 1 < seq else torch.zeros_like(last)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last * m[:, t]
    returns = adv + values
    return adv, returns


def value_function_loss(
    values: "torch.Tensor",
    returns: "torch.Tensor",
    *,
    old_values: "torch.Tensor | None" = None,
    clip: float | None = None,
    mask: "torch.Tensor | None" = None,
) -> "torch.Tensor":
    """Value-head regression loss to the GAE returns (CONCEPT:ML-009 PPO).

    Mean squared error ``(V − R)²`` over the valid tokens. When ``old_values`` and
    ``clip`` are given, the PPO value-clipping variant is used — the larger of the
    unclipped and clipped (to ``old_values ± clip``) squared errors — which damps
    large value updates. Returns a scalar.
    """
    torch = _torch()
    unclipped = (values - returns) ** 2
    if old_values is not None and clip is not None:
        clipped_v = old_values + torch.clamp(values - old_values, -clip, clip)
        sq = torch.maximum(unclipped, (clipped_v - returns) ** 2)
    else:
        sq = unclipped
    if mask is not None:
        return 0.5 * (sq * mask).sum() / mask.sum().clamp_min(1.0)
    return 0.5 * sq.mean()


def approx_kl(logprob: "torch.Tensor", ref_logprob: "torch.Tensor") -> "torch.Tensor":
    """Schulman k3 low-variance KL estimate ``E[ (r−1) − log r ]`` (≥0, mean).

    ``r = exp(ref_logprob − logprob)``. Used as the GRPO/SDAR KL-to-reference
    penalty; unbiased and non-negative unlike the naive ``logprob − ref_logprob``.
    """
    torch = _torch()
    log_r = ref_logprob - logprob
    r = torch.exp(log_r)
    return (r - 1.0 - log_r).mean()

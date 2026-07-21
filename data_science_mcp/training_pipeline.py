#!/usr/bin/python
"""End-to-end fine-tune pipeline + deploy seam (Wave D, CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Ties the in-house training substrate into one runnable flow:

    traces → SFT corpus → plan → train → reliability-eval → save checkpoint
           → register as a ModelDefinition bound to a role (goes live)

Everything except the GPU fine-tune *run* is deterministic and CPU-testable on a
toy model (dependency-injected). On the GB10 the only changes are: a real base
model id, the ``data-science-mcp[training]`` extra, and a GPU — the orchestration,
evaluation, and deploy seam are identical. First target: **OpenSeeker SFT**
(Qwen2.5-1.5B LoRA).

The deploy seam is the existing model-registry role resolution
(``model_registry.resolve_role ← rlm/roles ← create_model(role=…)``): once a
checkpoint is registered and a role bound to it, every consumer that asks for that
role resolves to the new model with **no hot-path edit**.

Concept: training-pipeline
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from agent_utilities.core.transport_security import ResolvedTLSProfile

from data_science_mcp import training_data as td
from data_science_mcp.trainers import TrainConfig, get_trainer

logger = logging.getLogger(__name__)

# Feature gate for the dynamic-LoRA hot-load (below): set to "0"/"false"/"no"/"off"
# to disable; on by default so a freshly-trained adapter is servable with zero
# manual ``--lora-modules`` restart whenever a serving ``base_url`` is configured.
_HOTLOAD_ENV = "DATA_SCIENCE_MCP_LORA_HOTLOAD"
# vLLM's dynamic-LoRA contract (``POST /v1/load_lora_adapter`` with
# ``{"lora_name", "lora_path"}``); SGLang mirrors the same OpenAI-compatible
# endpoint + payload where its build supports runtime LoRA loading, so one path
# covers both engines (see ``inference/openai_compatible.py``'s ``default_adapter``
# seam, which is how a hot-loaded adapter is then *selected* per-request).
_HOTLOAD_ENDPOINT = "/v1/load_lora_adapter"


def _resolve_eval_generate_fn(
    generate_fn: Callable[[str], str] | None,
) -> Callable[[str], str]:
    """Pick the inference fn for the reliability eval.

    Prefers an injected ``generate_fn``; otherwise, when a served-model endpoint
    is configured (``INFERENCE_BASE_URL``), uses the selected inference backend
    (vLLM/SGLang via ``INFERENCE_BACKEND``) at temperature 0; otherwise falls
    back to a no-op echo so the pipeline still completes on CPU with no server.
    """
    if generate_fn is not None:
        return generate_fn
    from data_science_mcp.inference import (  # noqa: PLC0415
        create_inference_backend,
        inference_backend_configured,
    )

    if inference_backend_configured():
        return create_inference_backend().as_generate_fn()
    return lambda x: x


@dataclass
class DeploymentTarget:
    """How to publish a trained checkpoint into the model registry."""

    role: str  # functional role to bind (e.g. "generator", "rlm-coder")
    served_model_name: str  # the model id the serving endpoint exposes
    base_url: str | None = None  # e.g. the local vLLM/SGLang OpenAI endpoint
    provider: str = "vllm"  # serving engine: "vllm" | "sglang" (see inference/)
    tier: str = "medium"
    tags: list[str] = field(default_factory=list)
    api_key_env: str | None = None


def _hotload_enabled() -> bool:
    """Whether the dynamic-LoRA hot-load is enabled (default on; env opt-out)."""
    return os.getenv(_HOTLOAD_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def hot_load_adapter(
    target: DeploymentTarget,
    *,
    adapter_name: str,
    adapter_path: str,
    tls_profile: ResolvedTLSProfile | None = None,
) -> dict[str, Any]:
    """Best-effort dynamic-LoRA hot-load onto the serving vLLM/SGLang endpoint.

    POSTs ``{lora_name, lora_path}`` to ``target.base_url``'s dynamic-LoRA
    endpoint (``POST /v1/load_lora_adapter`` — vLLM's contract; SGLang mirrors
    it) so a just-trained checkpoint becomes servable immediately with **no
    manual** ``--lora-modules``/server restart: the deploy seam registers the
    role binding (:func:`register_checkpoint`) while this call makes the
    *weights themselves* reachable on the already-running server.

    Reuses the existing serving-endpoint config (``target.base_url`` /
    ``target.provider`` — the SAME :class:`DeploymentTarget` the role-bind deploy
    seam already carries); no new config surface. Feature-gated
    (``DATA_SCIENCE_MCP_LORA_HOTLOAD=0`` disables) and always **best-effort**:
    a disabled gate, missing ``base_url``/``adapter_path``, missing ``httpx``, or
    an unreachable/erroring server all degrade to a logged, structured
    ``{"status": "skipped"|"error", ...}`` — this NEVER raises, so it can never
    fail the training run.
    """
    if not _hotload_enabled():
        return {"status": "skipped", "detail": "hot-load disabled via env"}
    if not target.base_url:
        return {"status": "skipped", "detail": "no serving base_url configured"}
    if not adapter_path:
        return {"status": "skipped", "detail": "no adapter_path to load"}
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return {"status": "skipped", "detail": "httpx not installed"}

    # ``base_url`` conventions vary by consumer: the OpenAI-SDK-style value this
    # SAME DeploymentTarget carries into ModelDefinition often already ends in
    # ``/v1`` (e.g. ``http://host:8000/v1``), while the raw server root does not
    # (``http://host:8000``). Normalize to the server root before appending the
    # dynamic-LoRA endpoint so both conventions resolve to the SAME, correct URL.
    base_url = target.base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    payload = {"lora_name": adapter_name, "lora_path": adapter_path}
    from agent_utilities.core.transport_security import (  # noqa: PLC0415
        resolve_configured_tls_profile,
    )

    profile = tls_profile or resolve_configured_tls_profile("model")
    try:
        resp = httpx.post(
            f"{base_url}{_HOTLOAD_ENDPOINT}",
            json=payload,
            timeout=30.0,
            **profile.httpx_kwargs(),
        )
        resp.raise_for_status()
        return {
            "status": "loaded",
            "provider": target.provider,
            "base_url": base_url,
            "adapter_name": adapter_name,
            "adapter_path": adapter_path,
        }
    except Exception as e:  # noqa: BLE001 — best-effort, never abort the run
        logger.warning(
            "[lora-hotload] dynamic LoRA load failed (%s @ %s%s): %s",
            target.provider,
            base_url,
            _HOTLOAD_ENDPOINT,
            e,
        )
        return {
            "status": "error",
            "provider": target.provider,
            "base_url": base_url,
            "detail": "Operation failed",
        }
    finally:
        if tls_profile is None:
            profile.cleanup()


def register_checkpoint(
    registry: Any,
    *,
    checkpoint_id: str,
    target: DeploymentTarget,
) -> Any:
    """Register a checkpoint as a ``ModelDefinition`` and bind ``target.role`` to it.

    Returns the created ``ModelDefinition``. After this call,
    ``registry.pick_for_role(target.role)`` selects the checkpoint (its role binding
    resolves via ``resolve_role`` → ``pick_for_task`` on a unique tag), so
    ``create_model(role=target.role)`` serves it — no hot-path change.
    """
    from agent_utilities.models.model_registry import ModelDefinition, RoleSpec

    # A unique tag binds the role to *this* checkpoint via tier+tag resolution.
    bind_tag = f"trained:{checkpoint_id}"
    tags = list(dict.fromkeys([*target.tags, "trained", bind_tag]))
    definition = ModelDefinition(
        id=checkpoint_id,
        name=f"Fine-tuned: {checkpoint_id}",
        provider=target.provider,
        model_id=target.served_model_name,
        base_url=target.base_url,
        api_key_env=target.api_key_env,
        tier=target.tier,  # type: ignore[arg-type]
        tags=tags,
    )
    # Drop any prior definition with the same id (idempotent re-deploy).
    registry.models = [m for m in registry.models if m.id != checkpoint_id]
    registry.models.append(definition)
    registry.role_routing[target.role] = RoleSpec(tier=target.tier, tags=[bind_tag])  # type: ignore[arg-type]
    return definition


def _deploy_checkpoint(
    registry: Any, deploy: DeploymentTarget, cid: str, adapter_path: str
) -> dict[str, Any]:
    """Deploy seam shared by all three pipelines: role-bind + best-effort hot-load.

    Registers the checkpoint (:func:`register_checkpoint` — the existing
    role-resolution deploy seam) and then attempts the dynamic-LoRA hot-load
    (:func:`hot_load_adapter`) onto the same ``deploy`` serving endpoint using
    whatever local checkpoint path the run produced. The hot-load is
    best-effort — its ``status`` (``loaded``/``skipped``/``error``) rides along
    in the returned ``deployment`` report and never aborts the run.
    """
    definition = register_checkpoint(registry, checkpoint_id=cid, target=deploy)
    hotload = hot_load_adapter(deploy, adapter_name=cid, adapter_path=adapter_path)
    return {
        "checkpoint_id": cid,
        "role": deploy.role,
        "model_id": definition.model_id,
        "resolved": registry.pick_for_role(deploy.role).model_dump(),
        "hotload": hotload,
    }


def run_sft_pipeline(
    config: TrainConfig,
    *,
    examples: list[dict[str, Any]] | None = None,
    traces: list[dict[str, Any]] | None = None,
    model: Any = None,
    tokenizer: Any = None,
    generate_fn: Callable[[str], str] | None = None,
    eval_cases: list[dict[str, Any]] | None = None,
    registry: Any = None,
    deploy: DeploymentTarget | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Run the SFT fine-tune pipeline end-to-end and (optionally) deploy.

    Args:
        config: trainer hyper-parameters (base_model, epochs, lora, …).
        examples: pre-built ``{prompt, completion}`` records, or…
        traces: raw traces to build the SFT corpus from (``build_sft_examples``).
        model/tokenizer: inject a toy model/tokenizer for a CPU smoke; omit to load
            the HF base via ``PeftManager``.
        generate_fn: post-train inference fn for the reliability eval (defaults to a
            no-op echo when omitted so the pipeline still completes on CPU).
        eval_cases: reliability-suite cases (see ``trainers.eval_hooks``).
        registry: a ``ModelRegistry`` to deploy into (with ``deploy``).
        deploy: where/how to publish the checkpoint (role binding).
        checkpoint_id: stable id for the checkpoint (defaults from ``config``).

    Returns:
        A structured report: ``{data, plan, train, eval?, checkpoint?, deployment?}``.
    """
    report: dict[str, Any] = {}

    # 1) Data
    if examples is None:
        examples = td.build_sft_examples(traces or [])
    report["data"] = {"examples": len(examples)}

    # 2) Plan (pure — no torch needed)
    trainer = get_trainer("sft", config)
    report["plan"] = trainer.plan(examples)

    # 3) Train
    report["train"] = trainer.train(examples, model=model, tokenizer=tokenizer)

    # 4) Reliability evaluation (AHE-3.1) — only when cases are supplied
    if eval_cases:
        from data_science_mcp.trainers.eval_hooks import evaluate_checkpoint

        report["eval"] = evaluate_checkpoint(
            _resolve_eval_generate_fn(generate_fn), eval_cases
        )

    # 5) Save checkpoint (best-effort; real HF/PEFT models expose save_pretrained)
    cid = checkpoint_id or f"sft-{(config.base_model or 'model').replace('/', '-')}"
    if config.output_dir and model is not None and hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(config.output_dir)
            report["checkpoint"] = {"path": config.output_dir, "saved": True, "id": cid}
        except Exception:  # pragma: no cover - defensive
            report["checkpoint"] = {"error": "Operation failed", "id": cid}

    # 6) Deploy seam — register + bind a role (goes live with no hot-path edit),
    #    then best-effort hot-load the adapter onto the serving vLLM/SGLang.
    if registry is not None and deploy is not None:
        adapter_path = str(
            report.get("checkpoint", {}).get("path") or config.output_dir or ""
        )
        report["deployment"] = _deploy_checkpoint(registry, deploy, cid, adapter_path)

    return report


def run_pretrain_pipeline(
    config: TrainConfig,
    *,
    corpus: list[dict[str, Any]] | list[str],
    spec: Any = None,
    tokenizer_spec: Any = None,
    model: Any = None,
    tokenizer: Any = None,
    train_tokenizer_first: bool = False,
    eval_cases: list[dict[str, Any]] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    registry: Any = None,
    deploy: DeploymentTarget | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Pretrain-from-random-init pipeline (CONCEPT:DS-AHE.trainer.concept-2), mirroring the SFT one.

    Flow: (optional) train a tokenizer from the corpus → build the random-init model
    → pretrain over packed sequences → (optional) reliability-eval → save checkpoint
    → register/bind a role (live with no hot-path edit). Everything but the GPU run
    is deterministic and CPU-testable with an injected toy model/tokenizer.

    Args:
        config: trainer hyper-parameters (``max_seq_len`` is the pack/block length).
        corpus: ``{text}`` records (or raw strings) to pretrain on.
        spec: a ``PretrainSpec`` (architecture); defaults to the small-LM preset.
        tokenizer_spec: a ``TokenizerSpec`` used when ``train_tokenizer_first``.
        model/tokenizer: inject a toy pair for a CPU smoke; omit for the live build.
        train_tokenizer_first: train a BPE tokenizer over the corpus before building.
    """
    from data_science_mcp.trainers import PretrainSpec, PretrainTrainer  # noqa: PLC0415

    report: dict[str, Any] = {}
    records = [r if isinstance(r, dict) else {"text": r} for r in corpus]
    report["data"] = {"records": len(records)}

    # 1) (optional) train a tokenizer from the corpus.
    if train_tokenizer_first and tokenizer is None:
        from data_science_mcp.tokenizer_trainer import (  # noqa: PLC0415
            TokenizerSpec,
            train_tokenizer,
        )

        tspec = tokenizer_spec or TokenizerSpec()
        tokenizer = train_tokenizer(
            (str(r.get("text", "")) for r in records),
            spec=tspec,
            output_dir=config.output_dir or None,
        )
        report["tokenizer"] = {"vocab_size": len(tokenizer)}

    # 2) Build trainer (random-init model from spec) + plan.
    trainer = PretrainTrainer(config, spec or PretrainSpec())
    report["plan"] = trainer.plan(records)

    # 3) Pretrain.
    report["train"] = trainer.train(records, model=model, tokenizer=tokenizer)

    # 4) Reliability evaluation (optional).
    if eval_cases:
        from data_science_mcp.trainers.eval_hooks import evaluate_checkpoint  # noqa: PLC0415

        report["eval"] = evaluate_checkpoint(
            _resolve_eval_generate_fn(generate_fn), eval_cases
        )

    # 5) Deploy seam — identical to the SFT pipeline.
    cid = (
        checkpoint_id or f"pretrain-{(config.base_model or 'model').replace('/', '-')}"
    )
    if registry is not None and deploy is not None:
        adapter_path = str(
            report.get("checkpoint", {}).get("path") or config.output_dir or ""
        )
        report["deployment"] = _deploy_checkpoint(registry, deploy, cid, adapter_path)

    return report


def run_rlhf_pipeline(
    config: TrainConfig,
    *,
    preference_pairs: list[dict[str, Any]] | None = None,
    ppo_dataset: list[dict[str, Any]] | None = None,
    sft_examples: list[dict[str, Any]] | None = None,
    run_sft: bool = False,
    model: Any = None,
    tokenizer: Any = None,
    value_model: Any = None,
    reward_model: Any = None,
    reward_fn: Callable[[str, str], float] | None = None,
    eval_cases: list[dict[str, Any]] | None = None,
    gsm8k_cases: list[dict[str, Any]] | None = None,
    generate_fn: Callable[[str], str] | None = None,
    registry: Any = None,
    deploy: DeploymentTarget | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Full RLHF pipeline: (SFT →) reward model → PPO → eval → deploy (CONCEPT:DS-AHE.trainer.per-token-value).

    Chains the new RLHF stages on top of the existing SFT/pretrain pipelines:

        (sft) → train reward model (Bradley-Terry) → PPO (reward model | verifier)
              → reliability + GSM8K eval → register/bind role (live)

    The reward model trained here is the **same object** PPO then scores rollouts
    with (the reward trainer trains the injected/loaded scorer in place), so
    ``reward_source="reward_model"`` needs no extra wiring. For the verifier path
    pass ``reward_fn`` (e.g. :func:`trainers.eval_hooks.gsm8k_reward` bound to a gold)
    and set ``config.reward_source="verifier"``. Everything but the GPU run is
    CPU-testable with injected toy modules.

    Returns ``{stages:{sft?, reward?, ppo}, eval?, gsm8k?, deployment?}``.
    """
    report: dict[str, Any] = {"stages": {}}

    # 1) Optional SFT warm-start (trains the policy in place when injected).
    if run_sft and sft_examples:
        sft = get_trainer("sft", config)
        report["stages"]["sft"] = sft.train(
            sft_examples, model=model, tokenizer=tokenizer
        )

    # 2) Reward model (skip for a pure verifier run with no pairs).
    using_reward_model = config.reward_source == "reward_model"
    if preference_pairs and (using_reward_model or reward_model is not None):
        reward_trainer = get_trainer("reward", config)
        report["stages"]["reward"] = reward_trainer.train(
            preference_pairs, model=reward_model, tokenizer=tokenizer
        )

    # 3) PPO — the reward model trained above (in place) is reused as the scorer.
    ppo = get_trainer("ppo", config)
    report["stages"]["ppo"] = ppo.train(
        ppo_dataset or [],
        model=model,
        tokenizer=tokenizer,
        value_model=value_model,
        reward_model=reward_model,
        reward_fn=reward_fn,
    )

    # 4) Eval gates — reliability suite + (optional) GSM8K accuracy.
    if eval_cases:
        from data_science_mcp.trainers.eval_hooks import evaluate_checkpoint  # noqa: PLC0415

        report["eval"] = evaluate_checkpoint(
            _resolve_eval_generate_fn(generate_fn), eval_cases
        )
    if gsm8k_cases:
        from data_science_mcp.trainers.eval_hooks import evaluate_gsm8k  # noqa: PLC0415

        report["gsm8k"] = evaluate_gsm8k(
            _resolve_eval_generate_fn(generate_fn), gsm8k_cases
        )

    # 5) Deploy seam — identical to the SFT/pretrain pipelines.
    cid = checkpoint_id or f"ppo-{(config.base_model or 'model').replace('/', '-')}"
    if registry is not None and deploy is not None:
        adapter_path = str(
            report.get("checkpoint", {}).get("path") or config.output_dir or ""
        )
        report["deployment"] = _deploy_checkpoint(registry, deploy, cid, adapter_path)

    return report


__all__ = [
    "DeploymentTarget",
    "register_checkpoint",
    "hot_load_adapter",
    "run_sft_pipeline",
    "run_pretrain_pipeline",
    "run_rlhf_pipeline",
]

#!/usr/bin/python
"""End-to-end fine-tune pipeline + deploy seam (Wave D, CONCEPT:AHE-3.1).

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

from dataclasses import dataclass, field
from typing import Any, Callable

from data_science_mcp import training_data as td
from data_science_mcp.trainers import TrainConfig, get_trainer


@dataclass
class DeploymentTarget:
    """How to publish a trained checkpoint into the model registry."""

    role: str  # functional role to bind (e.g. "generator", "rlm-coder")
    served_model_name: str  # the model id the serving endpoint exposes
    base_url: str | None = None  # e.g. the local vLLM OpenAI endpoint
    provider: str = "vllm"
    tier: str = "medium"
    tags: list[str] = field(default_factory=list)
    api_key_env: str | None = None


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

        gen = generate_fn or (lambda x: x)
        report["eval"] = evaluate_checkpoint(gen, eval_cases)

    # 5) Save checkpoint (best-effort; real HF/PEFT models expose save_pretrained)
    cid = checkpoint_id or f"sft-{(config.base_model or 'model').replace('/', '-')}"
    if config.output_dir and model is not None and hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(config.output_dir)
            report["checkpoint"] = {"path": config.output_dir, "id": cid}
        except Exception as e:  # pragma: no cover - defensive
            report["checkpoint"] = {"error": str(e), "id": cid}

    # 6) Deploy seam — register + bind a role (goes live with no hot-path edit)
    if registry is not None and deploy is not None:
        definition = register_checkpoint(registry, checkpoint_id=cid, target=deploy)
        report["deployment"] = {
            "checkpoint_id": cid,
            "role": deploy.role,
            "model_id": definition.model_id,
            "resolved": registry.pick_for_role(deploy.role).model_dump(),
        }

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
    """Pretrain-from-random-init pipeline (CONCEPT:ML-003), mirroring the SFT one.

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

        report["eval"] = evaluate_checkpoint(generate_fn or (lambda x: x), eval_cases)

    # 5) Deploy seam — identical to the SFT pipeline.
    cid = checkpoint_id or f"pretrain-{(config.base_model or 'model').replace('/', '-')}"
    if registry is not None and deploy is not None:
        definition = register_checkpoint(registry, checkpoint_id=cid, target=deploy)
        report["deployment"] = {
            "checkpoint_id": cid,
            "role": deploy.role,
            "model_id": definition.model_id,
            "resolved": registry.pick_for_role(deploy.role).model_dump(),
        }

    return report


__all__ = [
    "DeploymentTarget",
    "register_checkpoint",
    "run_sft_pipeline",
    "run_pretrain_pipeline",
]

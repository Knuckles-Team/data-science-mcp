"""MCP tools for the Wave-C gradient trainers (CONCEPT:AHE-3.1).

Exposes the SFT / DPO / GRPO trainers and the TIES adapter merge as action-routed
MCP tools. The torch/PEFT optimiser step needs the ``data-science-mcp[training]``
extra and (for real fine-tunes) the GB10 GPU, so each tool first returns the pure
training **plan**; with ``execute=True`` it loads the HF base and runs. When the
extra is absent, the tool still returns the plan plus a clear ``requires`` note —
no import-time hard dependency on torch.

These build directly on the deterministic corpora produced by the
``build_training_dataset`` tool (:mod:`data_science_mcp.mcp.mcp_training_data`).
"""

import json
from typing import Any

from fastmcp import FastMCP


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def register_trainer_tools(mcp: FastMCP) -> None:
    """Register the gradient-trainer tools (tag ``model-training``)."""

    def _run(kind: str, dataset: list, options: dict) -> dict[str, Any]:
        from data_science_mcp.trainers import TrainConfig, get_trainer  # PLC0415

        cfg_keys = {
            "base_model",
            "output_dir",
            "epochs",
            "lr",
            "batch_size",
            "max_steps",
            "max_seq_len",
            "grad_accum",
            "seed",
            "device",
            "beta",
            "clip_eps",
            "kl_coef",
            "group_size",
            # PPO (CONCEPT:ML-009)
            "vf_coef",
            "gamma",
            "gae_lambda",
            "value_clip",
            "reward_source",
            # robustness / throughput (CONCEPT:ML-001)
            "precision",
            "gradient_checkpointing",
            "max_grad_norm",
            "warmup_steps",
            "lr_scheduler",
            "resume_from",
            "save_steps",
            "save_total_limit",
            "attn_impl",
            "use_liger",
            "pack_sequences",
            # scale-out (CONCEPT:ML-005)
            "distributed",
            "cpu_offload",
            "zero_stage",
            # tracking (CONCEPT:ML-004)
            "tracker",
            "run_name",
            "kg_log",
        }
        config = TrainConfig(**{k: v for k, v in options.items() if k in cfg_keys})
        trainer = get_trainer(kind, config)
        plan = trainer.plan(dataset)
        if not options.get("execute"):
            return {
                "kind": kind,
                "plan": plan,
                "executed": False,
                "requires": None if _has_torch() else "data-science-mcp[training]",
                "note": "set options.execute=true to run (loads HF base; GPU recommended)",
            }
        if not _has_torch():
            return {
                "kind": kind,
                "plan": plan,
                "executed": False,
                "error": "torch not installed — install data-science-mcp[training]",
            }
        report = trainer.train(dataset)
        return {"kind": kind, "plan": plan, "executed": True, "report": report}

    @mcp.tool(tags={"model-training"})
    def train_sft(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """Supervised fine-tune on an ``sft`` corpus (CONCEPT:AHE-3.1).

        Args:
            dataset_json: JSON list of ``{prompt, completion}`` records (build with
                ``build_training_dataset kind=sft``).
            options_json: JSON ``TrainConfig`` fields + ``{"execute": bool}``.

        Returns:
            JSON ``{kind, plan, executed, report?}`` (or ``{error}``).
        """
        return _dispatch("sft", dataset_json, options_json, _run)

    @mcp.tool(tags={"model-training"})
    def train_dpo(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """Preference-optimise on a ``dpo`` corpus (CONCEPT:AHE-3.1).

        Args:
            dataset_json: JSON list of ``{prompt, chosen, rejected}`` records.
            options_json: JSON ``TrainConfig`` fields (incl. ``beta``) + ``execute``.
        """
        return _dispatch("dpo", dataset_json, options_json, _run)

    @mcp.tool(tags={"model-training"})
    def train_grpo(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """GRPO on advantage-tagged groups (CONCEPT:AHE-3.1).

        Args:
            dataset_json: JSON list of ``{prompt, samples:[{completion, reward,
                advantage}]}`` groups (build with ``build_training_dataset kind=grpo``).
            options_json: JSON ``TrainConfig`` fields (incl. ``clip_eps``, ``kl_coef``)
                + ``execute``.
        """
        return _dispatch("grpo", dataset_json, options_json, _run)

    @mcp.tool(tags={"model-training"})
    def train_reward(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """Train a Bradley-Terry reward model on preference pairs (CONCEPT:ML-008).

        The RLHF stage between SFT and PPO: fits a scalar reward head so a preferred
        response scores higher than its rejected partner. The trained reward model
        then scores PPO rollouts (``train_ppo`` with ``reward_source=reward_model``).

        Args:
            dataset_json: JSON list of ``{prompt, chosen, rejected}`` records (the
                same corpus as ``train_dpo``; build with
                ``build_training_dataset kind=dpo``).
            options_json: JSON ``TrainConfig`` fields + ``{"execute": bool}``.
        """
        return _dispatch("reward", dataset_json, options_json, _run)

    @mcp.tool(tags={"model-training"})
    def train_ppo(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """Proximal Policy Optimization with GAE + value head (CONCEPT:ML-009).

        Full actor-critic RLHF policy optimisation. Consumes scored rollouts; for
        the verifier reward source embed a scalar ``reward`` per record (generate the
        completions upstream with ``RolloutBuffer`` against a served vLLM/SGLang
        backend). A reward-model reward source needs an in-process reward model, so
        prefer the ``train_model`` workflow for the model-scored path.

        Args:
            dataset_json: JSON list of ``{prompt, completion, reward?}`` records.
            options_json: JSON ``TrainConfig`` fields (incl. ``clip_eps``, ``kl_coef``,
                ``vf_coef``, ``gamma``, ``gae_lambda``, ``value_clip``,
                ``reward_source``) + ``{"execute": bool}``.
        """
        return _dispatch("ppo", dataset_json, options_json, _run)

    @mcp.tool(tags={"model-training"})
    def merge_adapters_ties(
        base_json: str, task_vectors_json: str, options_json: str = "{}"
    ) -> str:
        """TIES-merge multiple task vectors onto a base (MeMo; CONCEPT:AHE-3.1).

        Args:
            base_json: JSON ``{param: [floats]}`` base parameters.
            task_vectors_json: JSON list of per-task delta dicts.
            options_json: JSON ``{"density": float, "scaling": float}``.

        Returns:
            JSON ``{params, merged}`` mapping each param to its merged values.
        """
        try:
            import numpy as np  # noqa: PLC0415

            from data_science_mcp.peft_manager import ties_merge  # noqa: PLC0415

            base = {
                k: np.asarray(v, dtype=float) for k, v in json.loads(base_json).items()
            }
            tvs = [
                {k: np.asarray(v, dtype=float) for k, v in tv.items()}
                for tv in json.loads(task_vectors_json)
            ]
            opts = json.loads(options_json or "{}")
            merged = ties_merge(
                base,
                tvs,
                density=float(opts.get("density", 0.2)),
                scaling=float(opts.get("scaling", 1.0)),
            )
            return json.dumps(
                {
                    "params": list(merged),
                    "merged": {k: v.tolist() for k, v in merged.items()},
                }
            )
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json: {e}"})
        except Exception as e:  # pragma: no cover - defensive
            return json.dumps({"error": str(e)})

    @mcp.tool(tags={"model-training"})
    def pretrain_model(dataset_json: str = "[]", options_json: str = "{}") -> str:
        """Pretrain a causal LM **from random init** (CONCEPT:ML-003).

        Args:
            dataset_json: JSON list of ``{text: ...}`` records (curate with
                ``curate_corpus`` first).
            options_json: JSON ``TrainConfig`` fields + a ``spec`` object with
                ``PretrainSpec`` architecture knobs (``hidden_size``,
                ``num_hidden_layers``, ``num_attention_heads``, ``vocab_size``,
                ``max_position_embeddings``, ``model_type``) + ``{"execute": bool}``.

        Returns:
            JSON ``{kind:"pretrain", plan, executed, report?}`` (or ``{error}``).
        """
        return _dispatch("pretrain", dataset_json, options_json, _run_pretrain)

    @mcp.tool(tags={"model-training"})
    def train_tokenizer(corpus_json: str = "[]", options_json: str = "{}") -> str:
        """Train a byte-level BPE tokenizer from scratch (CONCEPT:ML-003).

        Args:
            corpus_json: JSON list of training text strings.
            options_json: JSON ``TokenizerSpec`` fields (``vocab_size``,
                ``min_frequency``, ``special_tokens``, ``extra_tokens``,
                ``lowercase``) + ``output_dir`` + ``{"execute": bool}``.

        Returns:
            JSON ``{plan, executed, output_dir?, vocab_size?}`` (or ``{error}``).
        """
        try:
            from data_science_mcp.tokenizer_trainer import (  # noqa: PLC0415
                TokenizerSpec,
                plan_tokenizer,
                train_tokenizer as _train_tok,
            )

            corpus = json.loads(corpus_json or "[]")
            opts = json.loads(options_json or "{}")
            spec_keys = {
                "vocab_size",
                "min_frequency",
                "special_tokens",
                "extra_tokens",
                "lowercase",
            }
            spec = TokenizerSpec(
                **{
                    k: (tuple(v) if isinstance(v, list) else v)
                    for k, v in opts.items()
                    if k in spec_keys
                }
            )
            plan = plan_tokenizer(spec)
            base = {
                "plan": {
                    "vocab_size": plan.vocab_size,
                    "n_special": plan.n_special,
                    "special_tokens": plan.special_tokens,
                    "algorithm": plan.algorithm,
                }
            }
            if not opts.get("execute"):
                return json.dumps({**base, "executed": False})
            tok = _train_tok(corpus, spec=spec, output_dir=opts.get("output_dir"))
            return json.dumps(
                {
                    **base,
                    "executed": True,
                    "output_dir": opts.get("output_dir"),
                    "vocab_size": len(tok),
                }
            )
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json: {e}"})
        except Exception as e:  # pragma: no cover - defensive
            return json.dumps({"error": str(e)})


def _run_pretrain(kind: str, dataset: list, options: dict) -> dict[str, Any]:
    """Plan-first runner for the random-init pretrain trainer (CONCEPT:ML-003)."""
    from data_science_mcp.trainers import (  # noqa: PLC0415
        PretrainSpec,
        PretrainTrainer,
        TrainConfig,
    )

    cfg_keys = {
        "base_model",
        "output_dir",
        "epochs",
        "lr",
        "batch_size",
        "max_steps",
        "max_seq_len",
        "grad_accum",
        "seed",
        "device",
        "precision",
        "gradient_checkpointing",
        "max_grad_norm",
        "warmup_steps",
        "lr_scheduler",
        "resume_from",
        "save_steps",
        "save_total_limit",
        "attn_impl",
        "use_liger",
        "pack_sequences",
        "distributed",
        "cpu_offload",
        "zero_stage",
        "tracker",
        "run_name",
        "kg_log",
    }
    spec_keys = {
        "model_type",
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "max_position_embeddings",
    }
    config = TrainConfig(**{k: v for k, v in options.items() if k in cfg_keys})
    spec_opts = options.get("spec", {})
    spec = PretrainSpec(**{k: v for k, v in spec_opts.items() if k in spec_keys})
    trainer = PretrainTrainer(config, spec)
    plan = trainer.plan(dataset)
    plan["approx_params"] = spec.approx_params()
    if not options.get("execute"):
        return {
            "kind": kind,
            "plan": plan,
            "executed": False,
            "requires": None if _has_torch() else "data-science-mcp[training]",
            "note": "set options.execute=true to run (builds a random-init model; GPU recommended)",
        }
    if not _has_torch():
        return {
            "kind": kind,
            "plan": plan,
            "executed": False,
            "error": "torch not installed — install data-science-mcp[training]",
        }
    report = trainer.train(dataset)
    return {"kind": kind, "plan": plan, "executed": True, "report": report}


def _dispatch(kind: str, dataset_json: str, options_json: str, runner) -> str:
    try:
        dataset = json.loads(dataset_json or "[]")
        options = json.loads(options_json or "{}")
        if not isinstance(dataset, list):
            return json.dumps({"error": "dataset_json must be a JSON list"})
        return json.dumps(runner(kind, dataset, options))
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"invalid json: {e}"})
    except Exception as e:  # pragma: no cover - defensive
        return json.dumps({"error": str(e)})

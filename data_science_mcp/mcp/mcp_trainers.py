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

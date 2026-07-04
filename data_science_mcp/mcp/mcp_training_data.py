"""MCP tools for the deterministic training-data & reward engine (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

Exposes the pure-Python dataset/reward builders in
:mod:`data_science_mcp.training_data` (the Wave-C data-quality engine) as
action-routed MCP tools. No engine connection or GPU — these transform traces
into SFT/DPO/GRPO corpora and reward signals for downstream (Wave-D) trainers.
"""

import json
from typing import Any

from fastmcp import FastMCP

from data_science_mcp import search_task_corpus as sc
from data_science_mcp import training_data as td


def register_training_data_tools(mcp: FastMCP) -> None:
    """Register the training-data tools (tag ``model-training``)."""

    @mcp.tool(tags={"model-training"})
    def build_training_dataset(
        kind: str, items_json: str = "[]", options_json: str = "{}"
    ) -> str:
        """Build an SFT/DPO/GRPO training corpus from traces (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

        Args:
            kind: ``sft`` | ``dpo`` | ``grpo`` | ``filter_difficulty``, or the
                shortcut-resistant search variants ``search_sft`` |
                ``search_dpo`` | ``search_grpo`` that consume synthesized tasks +
                solver trajectories and reward by realized search difficulty
                (CONCEPT:AU-KG.retrieval.evidence-graph-workspace/2.71/2.72, AHE-3.30).
            items_json: JSON list of input records (traces / samples / groups).
            options_json: JSON object of options (e.g. ``{"min_steps": 3}`` for
                ``filter_difficulty``; ``{"weights": {...}}`` for ``search_grpo``).

        Returns:
            JSON ``{kind, count, records}`` (or ``{error}``).
        """
        try:
            items = json.loads(items_json or "[]")
            options = json.loads(options_json or "{}")
            if not isinstance(items, list):
                return json.dumps({"error": "items_json must be a JSON list"})
            if kind == "filter_difficulty":
                result: Any = td.filter_by_difficulty(
                    items,
                    min_steps=int(options.get("min_steps", 2)),
                    count_key=str(options.get("count_key", "step_count")),
                )
            elif kind in ("sft", "dpo", "grpo"):
                result = td.build_dataset(kind, items)
            elif kind == "search_sft":
                result = sc.tasks_to_sft(items)
            elif kind == "search_dpo":
                result = sc.trajectories_to_preference_pairs(items)
            elif kind == "search_grpo":
                result = sc.rollouts_to_grpo(items, weights=options.get("weights"))
            else:
                return json.dumps({"error": f"unknown kind: {kind}"})
            return json.dumps({"kind": kind, "count": len(result), "records": result})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json: {e}"})
        except Exception as e:  # pragma: no cover - defensive
            return json.dumps({"error": str(e)})

    @mcp.tool(tags={"model-training"})
    def compose_reward(
        components_json: str, weights_json: str, conditions_json: str = "{}"
    ) -> str:
        """Composite, conditionally-gated reward score (CONCEPT:AU-AHE.evaluation.adaptive-reasoning-effort).

        Args:
            components_json: JSON object ``{name: value}``.
            weights_json: JSON object ``{name: weight}``.
            conditions_json: JSON object ``{name: bool}`` — a component gated False
                contributes 0 (e.g. a functional-token reward only when used+correct).

        Returns:
            JSON ``{reward}`` (or ``{error}``).
        """
        try:
            components = json.loads(components_json)
            weights = json.loads(weights_json)
            conditions = json.loads(conditions_json or "{}")
            return json.dumps(
                {"reward": td.score_reward(components, weights, conditions=conditions)}
            )
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"invalid json: {e}"})
        except Exception as e:  # pragma: no cover - defensive
            return json.dumps({"error": str(e)})

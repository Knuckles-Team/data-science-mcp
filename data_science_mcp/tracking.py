#!/usr/bin/python
"""Experiment tracking for the training substrate (CONCEPT:ML-004).

A single :class:`RunTracker` abstraction over the trackers the trainers emit to.
Three backends, all **lazy** and all **best-effort** (telemetry must never crash a
training run):

* ``none``   — no-op (default; keeps CPU tests + the [training] extra clean).
* ``mlflow`` — self-hosted MLflow run + params/metrics (the homelab dashboard).
* ``wandb``  — Weights & Biases run (available, non-default).

Independently, ``kg_log=True`` mirrors the run into the epistemic-graph as a
``TrainingRun`` provenance node via the agent-utilities KG facade, so every run,
its hyper-parameters, and its metric trajectory are queryable as graph nodes
alongside the datasets (:mod:`data_science_mcp.data_engine`) and checkpoints.

The user-selected default for the agent flow is **MLflow + KG mirror**
(``tracker="mlflow"``, ``kg_log=True``); the trainers themselves default to
``none`` so nothing is imported unless asked.

Concept: run-tracker
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("data_science_mcp.tracking")


class RunTracker:
    """Lazy, best-effort experiment tracker (mlflow / wandb / none + KG mirror)."""

    def __init__(
        self,
        backend: str = "none",
        *,
        run_name: str | None = None,
        kg_log: bool = False,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.backend = (backend or "none").lower()
        self.run_name = run_name
        self.kg_log = kg_log
        self.params = dict(params or {})
        # In-memory mirror — always populated, so a run is inspectable/testable
        # even with no external tracker and forms the KG provenance payload.
        self.records: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self._active = False
        self._mlflow: Any = None
        self._wandb: Any = None

    @classmethod
    def from_config(
        cls, config: Any, *, params: dict[str, Any] | None = None
    ) -> "RunTracker":
        """Build a tracker from a ``TrainConfig`` (``tracker``/``run_name``/``kg_log``)."""
        return cls(
            getattr(config, "tracker", "none"),
            run_name=getattr(config, "run_name", None),
            kg_log=getattr(config, "kg_log", False),
            params=params,
        )

    # --- lifecycle ----------------------------------------------------------
    def start(self) -> "RunTracker":
        """Open the run on the selected backend (best-effort)."""
        self._active = True
        if self.backend == "mlflow":
            try:
                import mlflow  # noqa: PLC0415

                self._mlflow = mlflow
                mlflow.start_run(run_name=self.run_name)
                if self.params:
                    mlflow.log_params(_flat(self.params))
            except Exception as e:  # pragma: no cover - external service
                logger.warning("mlflow tracking disabled: %s", e)
                self._mlflow = None
        elif self.backend == "wandb":
            try:
                import wandb  # noqa: PLC0415

                self._wandb = wandb
                wandb.init(name=self.run_name, config=_flat(self.params))
            except Exception as e:  # pragma: no cover - external service
                logger.warning("wandb tracking disabled: %s", e)
                self._wandb = None
        return self

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)
        if self._mlflow is not None:
            try:  # pragma: no cover - external service
                self._mlflow.log_params(_flat(params))
            except Exception:
                pass

    def log_metrics(
        self, metrics: dict[str, float], *, step: int | None = None
    ) -> None:
        """Record a metric point (mirrored in memory; forwarded to the backend)."""
        rec = {"step": step, **{k: _num(v) for k, v in metrics.items()}}
        self.records.append(rec)
        if self._mlflow is not None:
            try:  # pragma: no cover - external service
                self._mlflow.log_metrics(
                    {k: _num(v) for k, v in metrics.items()}, step=step
                )
            except Exception:
                pass
        if self._wandb is not None:
            try:  # pragma: no cover - external service
                self._wandb.log({k: _num(v) for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    def end(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Close the run, mirror to the KG, and return the provenance payload."""
        self.summary = dict(summary or {})
        if self._mlflow is not None:
            try:  # pragma: no cover - external service
                if self.summary:
                    self._mlflow.log_metrics(
                        {k: _num(v) for k, v in self.summary.items() if _is_num(v)}
                    )
                self._mlflow.end_run()
            except Exception:
                pass
        if self._wandb is not None:
            try:  # pragma: no cover - external service
                self._wandb.finish()
            except Exception:
                pass
        payload = self.provenance()
        if self.kg_log:
            _kg_mirror(payload)
        self._active = False
        return payload

    def provenance(self) -> dict[str, Any]:
        """The run as a single provenance record (KG ``TrainingRun`` payload)."""
        return {
            "kind": "TrainingRun",
            "run_name": self.run_name,
            "backend": self.backend,
            "params": _flat(self.params),
            "metrics": self.records,
            "summary": self.summary,
        }


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _flat(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce param values to tracker-friendly scalars/strings."""
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        out[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
    return out


def _kg_mirror(payload: dict[str, Any]) -> None:
    """Best-effort write of a ``TrainingRun`` node to the epistemic-graph KG.

    Tries the agent-utilities KG facade; any failure (no engine, API drift) is
    swallowed — provenance is telemetry, never a training blocker.
    """
    try:  # pragma: no cover - requires a live KG facade/engine
        from agent_utilities.knowledge_graph.facade import (  # noqa: PLC0415
            KnowledgeGraphFacade,
        )

        kg = KnowledgeGraphFacade()
        writer = getattr(kg, "write", None) or getattr(kg, "add_node", None)
        if writer is not None:
            writer(payload)
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug("KG run mirror skipped: %s", e)


__all__ = ["RunTracker"]

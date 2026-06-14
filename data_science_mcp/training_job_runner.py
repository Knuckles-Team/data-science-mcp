#!/usr/bin/python
"""Training-job runner for the GPU-slot scheduler (CONCEPT:ML-011).

Bridges the data-science-mcp trainers to the agent-utilities single-GPU-slot
scheduler (CONCEPT:KG-2.65), so a long training run shares the one GPU slot with
interactive inference: an inference submission **preempts** the training job, which
checkpoints at the next optimizer step and **auto-resumes from that checkpoint**
when the slot frees.

The scheduler is generic — a job runner is ``async (job, scheduler) -> None`` that
checks ``scheduler.should_pause(job_id)`` at safe boundaries. This module wraps a
trainer as such a runner: it runs ``trainer.train(..., should_pause=…)`` (the SFT /
pretrain trainers forward the cooperative-yield check into ``run_loop``) on a worker
thread, and on pause persists the checkpoint dir as the job's resume point.

Submit training as ``kind="training"`` (a preemptible background tier — see
``GpuSlotScheduler.BACKGROUND_KINDS``). For a GPU host the ``train_model`` workflow
submits here; with no scheduler it falls back to a direct ``accelerate launch``.

Concept: training-job-runner
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

#: ``params -> (trainer, dataset, train_kwargs)``. Rebuilds the trainer + corpus for
#: a job (and again on each backfill resume); ``config.resume_from`` is set by the
#: runner before calling ``train`` so weights/optimizer/step continue.
TrainerFactory = Callable[[dict[str, Any]], "tuple[Any, list, dict[str, Any]]"]


def make_training_job_runner(factory: TrainerFactory) -> Any:
    """Build a scheduler ``JobRunner`` that trains via ``factory`` and is preemptible.

    ``factory(job.params)`` returns ``(trainer, dataset, train_kwargs)``. The runner
    forwards a ``should_pause`` closure into ``trainer.train`` (honoured by the
    SFT/pretrain trainers); when the run pauses it records the latest checkpoint dir
    as the job's ``resume_from`` and yields the slot, leaving the scheduler to mark
    the job ``paused`` and backfill it later.
    """

    async def runner(job: Any, scheduler: Any) -> None:
        from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import (  # noqa: PLC0415
            JobState,
        )

        trainer, dataset, kwargs = factory(dict(job.params))
        resume_from = job.checkpoint.get("resume_from")
        if resume_from is not None:
            trainer.config.resume_from = resume_from

        report = await asyncio.to_thread(
            trainer.train,
            dataset,
            should_pause=lambda: scheduler.should_pause(job.job_id),
            **kwargs,
        )

        if report.get("paused"):
            checkpoints = report.get("checkpoints") or []
            await scheduler.checkpoint(
                job.job_id,
                {
                    "resume_from": checkpoints[-1] if checkpoints else resume_from,
                    "steps": report.get("steps", 0),
                },
            )
            return  # cooperative yield → scheduler pauses + backfills this job

        job.checkpoint["report"] = {
            "steps": report.get("steps"),
            "final_loss": report.get("final_loss"),
        }
        job.state = JobState.DONE

    return runner


__all__ = ["make_training_job_runner", "TrainerFactory"]

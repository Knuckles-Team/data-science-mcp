#!/usr/bin/python
"""Training-job runner ↔ GPU-slot scheduler bridge (CONCEPT:AU-AHE.trainer.join-inference).

Drives a real toy SFT trainer through the runner with a fake scheduler that flips
``should_pause`` after a few steps, asserting the run checkpoints + yields, then
resumes from that checkpoint on the next invocation. No GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler",
    reason="GPU-slot scheduler (KG-2.65) not in the installed agent-utilities",
)

from data_science_mcp import training_data as td  # noqa: E402
from data_science_mcp.trainers import TrainConfig, get_trainer  # noqa: E402
from data_science_mcp.training_job_runner import make_training_job_runner  # noqa: E402

_CHARS = list(" abcdefghijklmnopqrstuvwxyz0123456789?.!")


class ToyTokenizer:
    def __init__(self) -> None:
        self._vocab = {"<pad>": 0}
        for c in _CHARS:
            self._vocab[c] = len(self._vocab)
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<pad>"

    def __len__(self) -> int:
        return len(self._vocab)

    def __call__(
        self, texts, return_tensors=None, padding=False, truncation=False, max_length=64
    ):
        if isinstance(texts, str):
            texts = [texts]
        rows = [
            ([self._vocab.get(ch, 1) for ch in t.lower()][:max_length] or [0])
            for t in texts
        ]
        width = max(len(r) for r in rows)
        ids = [r + [0] * (width - len(r)) for r in rows]
        attn = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


class _Out:
    def __init__(self, logits):
        self.logits = logits


class ToyModel(torch.nn.Module):
    def __init__(self, vocab: int, hidden: int = 8) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return _Out(self.head(self.emb(input_ids)))


class _Job:
    def __init__(self) -> None:
        self.job_id = "train-1"
        self.params: dict = {}
        self.checkpoint: dict = {}
        self.state = None


class _FakeScheduler:
    """should_pause flips True after ``pause_after`` optimizer steps; records checkpoints."""

    def __init__(self, pause_after: int | None) -> None:
        self.pause_after = pause_after
        self.calls = 0
        self.saved: dict = {}

    def should_pause(self, job_id: str) -> bool:
        self.calls += 1
        return self.pause_after is not None and self.calls > self.pause_after

    async def checkpoint(self, job_id: str, ck: dict) -> None:
        self.saved = dict(ck)


@pytest.mark.asyncio
async def test_training_job_runner_pauses_then_resumes(tmp_path):
    model, tok = ToyModel(48), ToyTokenizer()
    data = td.build_sft_examples(
        [
            {"prompt": "2+2=", "completion": "4"},
            {"prompt": "cap=", "completion": "yes"},
        ]
    )

    def factory(params):
        trainer = get_trainer(
            "sft",
            TrainConfig(
                lr=0.1, epochs=20, batch_size=1, output_dir=str(tmp_path), seed=0
            ),
        )
        return trainer, data, {"model": model, "tokenizer": tok}

    runner = make_training_job_runner(factory)
    job = _Job()

    # 1) Preempt after 3 steps → must checkpoint + yield (not DONE).
    sched_pause = _FakeScheduler(pause_after=3)
    await runner(job, sched_pause)
    from agent_utilities.knowledge_graph.ingestion.gpu_slot_scheduler import JobState

    assert job.state != JobState.DONE
    assert sched_pause.saved.get("resume_from")  # a checkpoint dir was persisted
    job.checkpoint = sched_pause.saved  # scheduler would persist this on the Job

    # 2) Backfill resume → no pause → runs to completion from the checkpoint.
    sched_done = _FakeScheduler(pause_after=None)
    await runner(job, sched_done)
    assert job.state == JobState.DONE

#!/usr/bin/python
"""Tests for the Phase-0 trainer hardening (CONCEPT:AU-AHE.trainer.high-caliber-llm-trainer/004/005).

Covers the additive robustness/scale features on the shared optimisation loop —
gradient accumulation, gradient clipping, LR scheduling, checkpoint save/resume,
and the experiment tracker — all on CPU with the toy model from ``test_trainers``
(no GPU, no FSDP/DeepSpeed, no external tracker).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from data_science_mcp import training_data as td  # noqa: E402
from data_science_mcp.tracking import RunTracker  # noqa: E402
from data_science_mcp.trainers import TrainConfig, get_trainer  # noqa: E402
from data_science_mcp.trainers.loop import run_loop  # noqa: E402
from test_trainers import _toy  # noqa: E402  (reuse the toy model/tokenizer)


# --------------------------------------------------------------------------- #
# TrainConfig — new fields exist with behaviour-preserving defaults            #
# --------------------------------------------------------------------------- #
def test_trainconfig_new_defaults_preserve_behaviour():
    c = TrainConfig()
    assert c.precision == "fp32"
    assert c.grad_accum == 1
    assert c.lr_scheduler == "constant"
    assert c.max_grad_norm is None
    assert c.save_steps == 0
    assert c.distributed == "none"
    assert c.tracker == "none"
    assert c.kg_log is False


# --------------------------------------------------------------------------- #
# run_loop — gradient accumulation: N items, accum=k → ceil(N/k) optim steps   #
# --------------------------------------------------------------------------- #
class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))


def test_run_loop_grad_accum_step_count():
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = TrainConfig(epochs=1, grad_accum=2)
    out = run_loop(
        config=cfg,
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        epoch_items=lambda: [1.0, 1.0, 1.0, 1.0],
        compute_loss=lambda t: (model.w - t) ** 2,
    )
    assert len(out["losses"]) == 4  # one record per micro-batch
    assert out["steps"] == 2  # 4 items / accum 2 = 2 optimizer updates


def test_run_loop_grad_accum_flushes_trailing_partial():
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = TrainConfig(epochs=1, grad_accum=2)
    out = run_loop(
        config=cfg,
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        epoch_items=lambda: [1.0, 1.0, 1.0],  # odd count → trailing flush
        compute_loss=lambda t: (model.w - t) ** 2,
    )
    assert out["steps"] == 2  # ceil(3/2)


def test_run_loop_grad_clip_runs_and_learns():
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    cfg = TrainConfig(epochs=20, grad_accum=1, max_grad_norm=0.5)
    out = run_loop(
        config=cfg,
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        epoch_items=lambda: [3.0],
        compute_loss=lambda t: (model.w - t) ** 2,
    )
    assert out["losses"][-1] < out["losses"][0]  # clipped but still converges
    assert float(model.w.detach()) == pytest.approx(3.0, abs=0.5)


# --------------------------------------------------------------------------- #
# LR scheduler — non-constant builds a scheduler and varies the LR             #
# --------------------------------------------------------------------------- #
def test_cosine_scheduler_changes_lr():
    pytest.importorskip("transformers")
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = TrainConfig(epochs=10, lr_scheduler="cosine", warmup_steps=2)
    sched = get_trainer("sft", cfg)._scheduler(opt, total_steps=10)
    assert sched is not None
    out = run_loop(
        config=cfg,
        model=model,
        optimizer=opt,
        device=torch.device("cpu"),
        epoch_items=lambda: [1.0],
        compute_loss=lambda t: (model.w - t) ** 2,
        scheduler=sched,
        total_steps=10,
    )
    assert out["final_lr"] < 0.1  # cosine decayed below the base LR


def test_constant_scheduler_is_none():
    cfg = TrainConfig()
    opt = torch.optim.SGD(_Tiny().parameters(), lr=0.1)
    assert get_trainer("sft", cfg)._scheduler(opt, total_steps=5) is None


# --------------------------------------------------------------------------- #
# Checkpoint save + resume (generic, works on the toy model)                   #
# --------------------------------------------------------------------------- #
def test_sft_checkpoint_save_and_resume(tmp_path):
    model, tok = _toy()
    data = td.build_sft_examples([{"prompt": "a", "completion": "b"}] * 4)
    cfg = TrainConfig(
        lr=0.05, epochs=1, batch_size=1, output_dir=str(tmp_path), save_steps=2
    )
    report = get_trainer("sft", cfg).train(data, model=model, tokenizer=tok)
    assert report["steps"] == 4
    assert report["checkpoints"], "expected periodic checkpoints"
    last = report["checkpoints"][-1]
    assert (tmp_path / "checkpoint-4" / "model_state.pt").exists()
    assert (tmp_path / "checkpoint-4" / "training_state.json").exists()

    # Resume from the last checkpoint → step counter is restored.
    model2, tok2 = _toy()
    cfg2 = TrainConfig(lr=0.05, epochs=1, batch_size=1, resume_from=last)
    report2 = get_trainer("sft", cfg2).train(data, model=model2, tokenizer=tok2)
    assert report2["resumed_from_step"] == 4


def test_save_total_limit_keeps_newest(tmp_path):
    model, tok = _toy()
    data = td.build_sft_examples([{"prompt": "a", "completion": "b"}] * 6)
    cfg = TrainConfig(
        lr=0.05,
        epochs=1,
        batch_size=1,
        output_dir=str(tmp_path),
        save_steps=1,
        save_total_limit=2,
    )
    report = get_trainer("sft", cfg).train(data, model=model, tokenizer=tok)
    kept = sorted(p.name for p in tmp_path.glob("checkpoint-*"))
    assert kept == ["checkpoint-5", "checkpoint-6"]  # only newest 2 survive on disk
    # The returned list reflects surviving checkpoints (older ones pruned).
    assert [p.split("/")[-1] for p in report["checkpoints"]] == kept


# --------------------------------------------------------------------------- #
# RunTracker — no-op default still mirrors in memory + builds provenance       #
# --------------------------------------------------------------------------- #
def test_run_tracker_none_is_noop_but_records():
    t = RunTracker("none", run_name="r1", params={"lr": 0.1}).start()
    t.log_metrics({"loss": 0.9}, step=1)
    t.log_metrics({"loss": 0.5}, step=2)
    payload = t.end({"final_loss": 0.5})
    assert payload["kind"] == "TrainingRun"
    assert payload["run_name"] == "r1"
    assert len(payload["metrics"]) == 2
    assert payload["summary"]["final_loss"] == 0.5


def test_tracker_from_config_defaults_off():
    t = RunTracker.from_config(TrainConfig())
    assert t.backend == "none" and t.kg_log is False


def test_tracker_provenance_carries_lineage():
    """TrainingRun emits PROV-O was_derived_from for the dataset→…→run chain."""
    cfg = TrainConfig(
        tracker="none", dataset_version="corpus@v3", parent_run="sft-run-1"
    )
    payload = RunTracker.from_config(cfg, params={"trainer": "ppo"}).start().end({})
    assert payload["dataset_version"] == "corpus@v3"
    assert payload["parent_run"] == "sft-run-1"
    assert set(payload["was_derived_from"]) == {"corpus@v3", "sft-run-1"}

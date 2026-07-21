#!/usr/bin/python
"""Tests for the distributed launcher + benchmark eval (CONCEPT:DS-AHE.trainer.concept-4/006).

Config builders and the ``accelerate launch`` argv are pure/CPU-testable; the
actual multi-GPU run and LightEval scoring happen on a GPU host.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from data_science_mcp.launch import (
    build_launch_command,
    deepspeed_zero3_config,
    fsdp_accelerate_config,
    write_config,
)


def test_fsdp_config_full_shard():
    cfg = fsdp_accelerate_config(num_processes=4, mixed_precision="bf16")
    assert cfg["distributed_type"] == "FSDP"
    assert cfg["num_processes"] == 4
    assert cfg["fsdp_config"]["fsdp_sharding_strategy"] == "FULL_SHARD"


def test_deepspeed_zero3_config_offload_toggles():
    cfg = deepspeed_zero3_config(offload_optimizer=True, offload_params=False)
    assert cfg["zero_optimization"]["stage"] == 3
    assert cfg["zero_optimization"]["offload_optimizer"]["device"] == "cpu"
    assert cfg["zero_optimization"]["offload_param"]["device"] == "none"


def test_write_config_roundtrip(tmp_path):
    p = write_config(deepspeed_zero3_config(), str(tmp_path / "ds.json"))
    import json

    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    assert data["zero_optimization"]["stage"] == 3


def test_build_launch_command_fsdp_and_deepspeed():
    fsdp = build_launch_command(
        "data_science_mcp.trainers.pretrain_trainer",
        distributed="fsdp",
        num_processes=8,
        config_file="/cfg/fsdp.yaml",
    )
    assert fsdp[:2] == ["accelerate", "launch"]
    assert "--use_fsdp" in fsdp
    assert "--config_file" in fsdp and "/cfg/fsdp.yaml" in fsdp
    assert fsdp[-1] == "data_science_mcp.trainers.pretrain_trainer"

    ds = build_launch_command(
        "trainer.mod",
        distributed="deepspeed",
        num_machines=2,
        machine_rank=1,
        main_process_ip="10.0.0.1",
        deepspeed_config_file="/cfg/ds.json",
        script_args=["--epochs", "3"],
    )
    assert "--use_deepspeed" in ds
    assert "--main_process_ip" in ds and "10.0.0.1" in ds
    assert ds[-2:] == ["--epochs", "3"]


def test_build_launch_command_rejects_bad_backend():
    import pytest

    with pytest.raises(ValueError):
        build_launch_command("m", distributed="horovod")


def test_evaluate_benchmarks_without_lighteval_is_graceful():
    from data_science_mcp.trainers.eval_hooks import evaluate_benchmarks

    out = evaluate_benchmarks("some/model", ["hellaswag"])
    # LightEval is not installed in CI → graceful error, never a crash.
    assert "error" in out or "results" in out


def test_evaluate_benchmarks_uses_hardened_lighteval_contract(monkeypatch):
    """Keep the optional LightEval 0.13 boundary CPU-only and fail-safe."""
    calls: dict[str, object] = {}

    class EvaluationTracker:
        def __init__(self, **kwargs):
            calls["tracker"] = kwargs

    class TransformersModelConfig:
        def __init__(self, **kwargs):
            calls["model"] = kwargs

    class PipelineParameters:
        def __init__(self, **kwargs):
            calls["parameters"] = kwargs

    class Pipeline:
        def __init__(self, **kwargs):
            calls["pipeline"] = kwargs

        def evaluate(self):
            calls["evaluated"] = True

        def get_results(self):
            return {"results": {"hellaswag": {"acc": 0.5}}}

    modules = {
        "lighteval": ModuleType("lighteval"),
        "lighteval.logging": ModuleType("lighteval.logging"),
        "lighteval.logging.evaluation_tracker": ModuleType(
            "lighteval.logging.evaluation_tracker"
        ),
        "lighteval.models": ModuleType("lighteval.models"),
        "lighteval.models.transformers": ModuleType("lighteval.models.transformers"),
        "lighteval.models.transformers.transformers_model": ModuleType(
            "lighteval.models.transformers.transformers_model"
        ),
        "lighteval.pipeline": ModuleType("lighteval.pipeline"),
    }
    modules["lighteval.logging.evaluation_tracker"].EvaluationTracker = (
        EvaluationTracker
    )
    modules[
        "lighteval.models.transformers.transformers_model"
    ].TransformersModelConfig = TransformersModelConfig
    modules["lighteval.pipeline"].ParallelismManager = SimpleNamespace(NONE="none")
    modules["lighteval.pipeline"].Pipeline = Pipeline
    modules["lighteval.pipeline"].PipelineParameters = PipelineParameters
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    from data_science_mcp.trainers.eval_hooks import evaluate_benchmarks

    out = evaluate_benchmarks(
        "organization/model",
        ["hellaswag"],
        limit=2,
        batch_size=1,
        device="cpu",
        revision="a" * 40,
    )

    assert out["results"]["hellaswag"]["acc"] == 0.5
    assert calls["evaluated"] is True
    assert calls["tracker"] == {
        "output_dir": calls["tracker"]["output_dir"],
        "save_details": False,
        "push_to_hub": False,
        "push_to_tensorboard": False,
        "use_wandb": False,
    }
    assert calls["model"] == {
        "model_name": "organization/model",
        "revision": "a" * 40,
        "batch_size": 1,
        "device": "cpu",
        "trust_remote_code": False,
    }
    assert calls["parameters"] == {"launcher_type": "none", "max_samples": 2}
    assert calls["pipeline"]["tasks"] == "hellaswag"

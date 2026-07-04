#!/usr/bin/python
"""Distributed launch configs + ``accelerate launch`` command builder (CONCEPT:DS-AHE.trainer.concept-4).

FSDP and DeepSpeed ZeRO-3 are first-class peers. This module emits the matching
launch configuration (an Accelerate YAML for FSDP, a DeepSpeed JSON for ZeRO-3) and
builds the ``accelerate launch`` command for a chosen trainer entry-point, targeting
**both** the homelab and rented cloud GPUs from one launcher (the only difference is
``--num_processes`` / ``--num_machines`` / ``--main_process_ip``).

Config builders are pure dicts (CPU-testable); :func:`write_config` serialises them
(YAML when available, else JSON), and :func:`build_launch_command` assembles the argv
without executing anything — the actual multi-process run happens on a GPU host.

Concept: launch
"""

from __future__ import annotations

import json
import os
from typing import Any


def fsdp_accelerate_config(
    *,
    num_processes: int = 8,
    num_machines: int = 1,
    mixed_precision: str = "bf16",
    transformer_layer_cls: str | None = None,
) -> dict[str, Any]:
    """An Accelerate config for **FSDP full-shard** across ``num_processes`` GPUs."""
    fsdp: dict[str, Any] = {
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_backward_prefetch": "BACKWARD_PRE",
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
        "fsdp_offload_params": False,
        "fsdp_use_orig_params": True,
    }
    if transformer_layer_cls:
        fsdp["fsdp_transformer_layer_cls_to_wrap"] = transformer_layer_cls
    return {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "FSDP",
        "mixed_precision": mixed_precision,
        "num_processes": num_processes,
        "num_machines": num_machines,
        "machine_rank": 0,
        "rdzv_backend": "static",
        "same_network": True,
        "fsdp_config": fsdp,
    }


def deepspeed_zero3_config(
    *, offload_optimizer: bool = False, offload_params: bool = False
) -> dict[str, Any]:
    """A DeepSpeed **ZeRO-3** config JSON (auto-tuned to the runtime batch/precision)."""
    opt_device = "cpu" if offload_optimizer else "none"
    param_device = "cpu" if offload_params else "none"
    return {
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": opt_device, "pin_memory": True},
            "offload_param": {"device": param_device, "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
    }


def write_config(config: dict[str, Any], path: str) -> str:
    """Serialise a config to ``path`` (YAML if PyYAML present, else JSON)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # noqa: PLC0415

            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)
            return path
        except ImportError:  # pragma: no cover - fall back to JSON content
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def build_launch_command(
    trainer_module: str,
    *,
    distributed: str = "fsdp",
    num_processes: int = 8,
    num_machines: int = 1,
    machine_rank: int = 0,
    main_process_ip: str | None = None,
    main_process_port: int = 29500,
    config_file: str | None = None,
    deepspeed_config_file: str | None = None,
    script_args: list[str] | None = None,
) -> list[str]:
    """Build an ``accelerate launch`` argv for ``trainer_module`` (no execution).

    ``distributed`` selects FSDP or DeepSpeed; multi-node is expressed via
    ``num_machines``/``machine_rank``/``main_process_ip`` (same argv for homelab and
    cloud — only those values change).
    """
    if distributed not in ("fsdp", "deepspeed"):
        raise ValueError(f"distributed must be fsdp|deepspeed, got {distributed!r}")
    cmd = ["accelerate", "launch"]
    if config_file:
        cmd += ["--config_file", config_file]
    cmd += [
        "--num_processes",
        str(num_processes),
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        str(machine_rank),
        "--main_process_port",
        str(main_process_port),
    ]
    if main_process_ip:
        cmd += ["--main_process_ip", main_process_ip]
    if distributed == "fsdp":
        cmd += ["--use_fsdp"]
    else:
        cmd += ["--use_deepspeed"]
        if deepspeed_config_file:
            cmd += ["--deepspeed_config_file", deepspeed_config_file]
    cmd += ["-m", trainer_module]
    if script_args:
        cmd += list(script_args)
    return cmd


__all__ = [
    "fsdp_accelerate_config",
    "deepspeed_zero3_config",
    "write_config",
    "build_launch_command",
]

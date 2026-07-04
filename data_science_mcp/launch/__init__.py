"""Multi-GPU / multi-node launch configs + command builder (CONCEPT:DS-AHE.trainer.concept-4)."""

from data_science_mcp.launch.launch import (
    build_launch_command,
    deepspeed_zero3_config,
    fsdp_accelerate_config,
    write_config,
)

__all__ = [
    "fsdp_accelerate_config",
    "deepspeed_zero3_config",
    "write_config",
    "build_launch_command",
]

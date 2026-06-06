"""MCP tool registration modules for data-science-mcp.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from data_science_mcp.mcp.mcp_data_management import register_data_management_tools
from data_science_mcp.mcp.mcp_interpretability import register_interpretability_tools
from data_science_mcp.mcp.mcp_model_evolution import register_model_evolution_tools
from data_science_mcp.mcp.mcp_model_training import register_model_training_tools
from data_science_mcp.mcp.mcp_quant import register_quant_tools

__all__ = [
    "register_data_management_tools",
    "register_interpretability_tools",
    "register_model_evolution_tools",
    "register_model_training_tools",
    "register_quant_tools",
]

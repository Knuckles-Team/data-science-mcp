"""MCP tools for data management operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine
from data_science_mcp.path_policy import authorized_dataset_source


def register_data_management_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"data-management"})
    async def load_dataset(
        name: str = Field(
            description=(
                "Built-in dataset name or CSV path confined to "
                "DATA_SCIENCE_DATA_ROOT"
            )
        ),
        target_column: str = Field(
            default="", description="Name of the target column (for CSV files)"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Load and parse a dataset by name or CSV file path."""
        if ctx:
            await ctx.info("Loading configured dataset...")
        try:
            source_path = authorized_dataset_source(name)
        except ValueError:
            return {"error": "CSV path is outside the configured data root"}
        engine = MLEngine()
        return engine.load_dataset(name, target_column, source_path=source_path)

    @mcp.tool(tags={"data-management"})
    async def describe_dataset(
        name: str = Field(description="Name of the loaded dataset to describe"),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Get descriptive statistics for a loaded dataset."""
        if ctx:
            await ctx.info("Describing statistics for configured dataset...")
        engine = MLEngine()
        return engine.describe_dataset(name)

    @mcp.tool(tags={"data-management"})
    async def split_dataset(
        name: str = Field(description="Name of the loaded dataset to split"),
        test_size: float = Field(
            default=0.2, description="Holdout fraction for testing"
        ),
        validation_size: float = Field(
            default=0.0, description="Holdout fraction for validation (from train set)"
        ),
        random_seed: int = Field(
            default=42, description="Random seed for reproducibility"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Split a loaded dataset into train, test, and validation sets."""
        if ctx:
            await ctx.info("Splitting configured dataset...")
        engine = MLEngine()
        return engine.split_dataset(name, test_size, validation_size, random_seed)

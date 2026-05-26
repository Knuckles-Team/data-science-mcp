"""MCP tools for data management operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine


def register_data_management_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"data-management"})
    async def load_dataset(
        name: str = Field(
            description="Dataset name ('california', 'diabetes', 'iris', 'wine', 'breast_cancer') or path to .csv file"
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
            await ctx.info(f"Loading dataset {name}...")
        engine = MLEngine()
        return engine.load_dataset(name, target_column)

    @mcp.tool(tags={"data-management"})
    async def describe_dataset(
        name: str = Field(description="Name of the loaded dataset to describe"),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Get descriptive statistics for a loaded dataset."""
        if ctx:
            await ctx.info(f"Describing statistics for dataset {name}...")
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
            await ctx.info(f"Splitting dataset {name} with test_size={test_size}...")
        engine = MLEngine()
        return engine.split_dataset(name, test_size, validation_size, random_seed)

"""MCP tools for model evolution operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine
from data_science_mcp.mcp_server import _pareto_models


def register_model_evolution_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"model-evolution"})
    async def evolve_model_class(
        model_class: str = Field(
            description="Model class name (e.g. 'Ridge', 'DecisionTree')"
        ),
        base_performance: float = Field(
            description="Accuracy/performance score (higher is better)"
        ),
        complexity: float = Field(
            description="Complexity of the model class (lower is better)"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Submit a model to the evolutionary Pareto frontier."""
        if ctx:
            await ctx.info(
                f"Submitting {model_class} to evolutionary Pareto frontier..."
            )

        _pareto_models[model_class] = {
            "performance": base_performance,
            "complexity": complexity,
        }

        # Recompute Pareto Frontier
        frontier = {}
        for m1, spec1 in _pareto_models.items():
            dominated = False
            for m2, spec2 in _pareto_models.items():
                if m1 == m2:
                    continue
                # dominated if spec2 is strictly better in one dimension and at least equal in the other
                p1, c1 = spec1["performance"], spec1["complexity"]
                p2, c2 = spec2["performance"], spec2["complexity"]
                if (p2 >= p1 and c2 < c1) or (p2 > p1 and c2 <= c1):
                    dominated = True
                    break
            if not dominated:
                frontier[m1] = spec1

        return {
            "status": "success",
            "message": f"Successfully submitted {model_class} to the Pareto frontier.",
            "pareto_frontier": frontier,
        }

    @mcp.tool(tags={"model-evolution"})
    async def rank_models(
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Rank all registered fitted models by their test R2 score."""
        if ctx:
            await ctx.info("Ranking models by R2 score...")
        engine = MLEngine()
        ranked = []
        for model_id, model_data in engine._models.items():
            model = model_data["model"]
            X_test = model_data["X_test"]
            y_test = model_data["y_test"]
            r2 = float(model.score(X_test, y_test))
            ranked.append(
                {
                    "model_id": model_id,
                    "dataset": model_data["dataset"],
                    "model_str": str(model),
                    "r2_test": round(r2, 6),
                }
            )
        ranked.sort(key=lambda x: x["r2_test"], reverse=True)
        return {"ranked_models": ranked}

    @mcp.tool(tags={"model-evolution"})
    async def get_pareto_frontier(
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Retrieve the current Pareto frontier of model classes."""
        if ctx:
            await ctx.info("Retrieving evolutionary Pareto frontier...")

        frontier = {}
        for m1, spec1 in _pareto_models.items():
            dominated = False
            for m2, spec2 in _pareto_models.items():
                if m1 == m2:
                    continue
                p1, c1 = spec1["performance"], spec1["complexity"]
                p2, c2 = spec2["performance"], spec2["complexity"]
                if (p2 >= p1 and c2 < c1) or (p2 > p1 and c2 <= c1):
                    dominated = True
                    break
            if not dominated:
                frontier[m1] = spec1

        return {"pareto_frontier": frontier}

"""MCP tools for model training operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import json

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine


def register_model_training_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"model-training"})
    async def fit_model(
        model_class: str = Field(
            description="Model class name (e.g., 'LinearRegression', 'Ridge', 'RandomForest')"
        ),
        dataset_name: str = Field(description="Name of the loaded dataset to train on"),
        hyperparameters_json: str = Field(
            default="{}", description="JSON string of hyperparameters for the model"
        ),
        test_size: float = Field(
            default=0.2, description="Fraction of the dataset to hold out for testing"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Fit a machine learning model on a dataset and return metrics."""
        if ctx:
            await ctx.info(
                f"Fitting model class {model_class} on dataset {dataset_name}..."
            )
        try:
            hparams = json.loads(hyperparameters_json)
        except Exception:
            return {"error": "Operation failed"}

        engine = MLEngine()
        return engine.fit(
            model_class=model_class,
            dataset_name=dataset_name,
            hyperparameters=hparams,
            test_size=test_size,
        )

    @mcp.tool(tags={"model-training"})
    async def predict(
        model_id: str = Field(description="ID of a fitted model"),
        inputs_json: str = Field(
            description="JSON string of a list of feature value dicts"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Generate predictions using a fitted model."""
        if ctx:
            await ctx.info(f"Generating predictions with model {model_id}...")
        try:
            inputs = json.loads(inputs_json)
        except Exception:
            return {"error": "Operation failed"}

        engine = MLEngine()
        try:
            preds = engine.predict(model_id, inputs)
            return {"model_id": model_id, "predictions": preds}
        except Exception:
            return {"error": "Operation failed"}

    @mcp.tool(tags={"model-training"})
    async def evaluate_model(
        model_id: str = Field(description="ID of a fitted model"),
        dataset_name: str = Field(description="Name of the dataset to evaluate on"),
        split: str = Field(
            default="test", description="Dataset split to evaluate: 'train' or 'test'"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Evaluate a fitted model on a dataset split."""
        if ctx:
            await ctx.info(
                f"Evaluating model {model_id} on {split} split of {dataset_name}..."
            )
        engine = MLEngine()
        return engine.evaluate(model_id, dataset_name, split)

    @mcp.tool(tags={"model-training"})
    async def cross_validate(
        model_class: str = Field(description="Model class name"),
        dataset_name: str = Field(description="Name of the dataset"),
        n_folds: int = Field(
            default=5, description="Number of folds for cross-validation"
        ),
        hyperparameters_json: str = Field(
            default="{}", description="JSON string of hyperparameters"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Perform k-fold cross-validation for a model class."""
        if ctx:
            await ctx.info(
                f"Running {n_folds}-fold cross validation for {model_class} on {dataset_name}..."
            )
        try:
            hparams = json.loads(hyperparameters_json)
        except Exception:
            return {"error": "Operation failed"}

        engine = MLEngine()
        return engine.cross_validate(model_class, dataset_name, n_folds, hparams)

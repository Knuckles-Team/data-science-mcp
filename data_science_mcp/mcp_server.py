#!/usr/bin/python
# coding: utf-8
"""Data Science MCP Server.

Provides model training, evaluation, and evolution tools for agentic ML
workflows. Designed as the MCP delegation target for agent-utilities'
IModelEvolver and InterpretabilityTestSuite.

Tool tags:
    - model-training: fit_model, predict, evaluate_model, cross_validate
    - model-evolution: evolve_model_class, rank_models, pareto_frontier
    - interpretability: run_interpretability_suite, grade_response, generate_tests
    - data-management: load_dataset, describe_dataset, split_dataset
    - quant: quant_market_making, quant_microstructure, quant_sizing,
      quant_validation, quant_forensic, quant_statespace, quant_signals,
      quant_derivatives (epistemic-graph finance kernels)
"""

from fastmcp import Context, FastMCP
from pydantic import Field
import json
import logging
import os
import sys
from typing import Any

from dotenv import find_dotenv, load_dotenv

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import create_mcp_server
from agent_utilities.base_utilities import get_logger

__version__ = "0.25.0"

# Redirect logging to stderr to prevent MCP stdout corruption
logger = get_logger(name="MCP_Server")
logger.setLevel(logging.INFO)

# ── Environment-variable toggles ─────────────────────────────────────
DEFAULT_MODEL_TRAININGTOOL = to_boolean(os.getenv("MODEL_TRAININGTOOL", "True"))
DEFAULT_MODEL_EVOLUTIONTOOL = to_boolean(os.getenv("MODEL_EVOLUTIONTOOL", "True"))
DEFAULT_INTERPRETABILITYTOOL = to_boolean(
    os.getenv("INTERPRETABILITYTOOL", "True"),
)
DEFAULT_DATA_MANAGEMENTTOOL = to_boolean(
    os.getenv("DATA_MANAGEMENTTOOL", "True"),
)
DEFAULT_DATA_ENGINETOOL = to_boolean(os.getenv("DATA_ENGINETOOL", "True"))
DEFAULT_QUANTTOOL = to_boolean(os.getenv("QUANTTOOL", "True"))

# ── Model Training Tools ─────────────────────────────────────────────

from data_science_mcp.ml_engine import MLEngine  # noqa: E402

_pareto_models = {}  # In-memory store for model classes submitted to Pareto frontier
_graded_responses = {}  # In-memory store for interpretability tests and grades

# Imported after the module-level stores above so the sibling tool modules in
# data_science_mcp.mcp (which back-import _pareto_models/_graded_responses) load
# without a circular import.
from data_science_mcp.mcp.mcp_quant import register_quant_tools  # noqa: E402
from data_science_mcp.mcp.mcp_training_data import (  # noqa: E402
    register_training_data_tools,
)
from data_science_mcp.mcp.mcp_kernel_specialize import (  # noqa: E402
    register_kernel_specialize_tools,
)
from data_science_mcp.mcp.mcp_trainers import (  # noqa: E402
    register_trainer_tools,
)

# mcp_data_engine is imported lazily inside get_mcp_instance (see below) rather than
# at module top: a top-level import here perturbs the package's lazy OPTIONAL_MODULES
# loader (data_science_mcp/__init__.py), whose _import_module_safely swallows the
# transient partial-init ImportError and would leave mcp_server unexposed.


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
        except Exception as e:
            return {"error": f"Invalid hyperparameters_json: {e}"}

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
        except Exception as e:
            return {"error": f"Invalid inputs_json: {e}"}

        engine = MLEngine()
        try:
            preds = engine.predict(model_id, inputs)
            return {"model_id": model_id, "predictions": preds}
        except Exception as e:
            return {"error": str(e)}

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
        except Exception as e:
            return {"error": f"Invalid hyperparameters_json: {e}"}

        engine = MLEngine()
        return engine.cross_validate(model_class, dataset_name, n_folds, hparams)


# ── Model Evolution Tools ────────────────────────────────────────────


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
        return {"ranked_models": engine.ranked_models()}

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


# ── Interpretability Tools ───────────────────────────────────────────


def register_interpretability_tools(mcp: FastMCP) -> None:
    @mcp.tool(tags={"interpretability"})
    async def generate_interpretability_tests(
        model_id: str = Field(description="ID of a fitted model to generate tests for"),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Generate a structured suite of 6 interpretability test cases for a model."""
        if ctx:
            await ctx.info(f"Generating interpretability tests for model {model_id}...")

        engine = MLEngine()
        if model_id not in engine._models:
            return {"error": f"Model {model_id} not found."}

        model_data = engine._models[model_id]
        feature_names = model_data["feature_names"]

        tests = [
            {
                "test_id": f"att_{model_id}_0",
                "category": "Feature Attribution",
                "question": f"Which feature among {feature_names} has the largest absolute coefficient/importance in the model representation?",
                "expected_hint": "Inspect the model coefficients/feature importances.",
            },
            {
                "test_id": f"sim_{model_id}_0",
                "category": "Point Simulation",
                "question": "Predict the target value when all input features are set to 0.0.",
                "expected_hint": "This corresponds to the model intercept / baseline prediction.",
            },
            {
                "test_id": f"sens_{model_id}_0",
                "category": "Sensitivity Analysis",
                "question": f"How does the model prediction change if we increase the first feature '{feature_names[0]}' by 1.0 unit?",
                "expected_hint": "This corresponds to the derivative / coefficient of the first feature.",
            },
            {
                "test_id": f"cf_{model_id}_0",
                "category": "Counterfactual",
                "question": "What is the required value of the features to achieve exactly the baseline/intercept output?",
                "expected_hint": "Set all features to 0.0.",
            },
            {
                "test_id": f"conf_{model_id}_0",
                "category": "Confidence Calibration",
                "question": "What is the R2 score achieved by this model on the test holdout set?",
                "expected_hint": "Inspect the model evaluation metrics.",
            },
            {
                "test_id": f"data_{model_id}_0",
                "category": "Data Attribution",
                "question": "How many training samples were used to fit this model?",
                "expected_hint": "Check the n_train field from model fit details.",
            },
        ]

        return {"model_id": model_id, "tests": tests}

    @mcp.tool(tags={"interpretability"})
    async def grade_response(
        test_id: str = Field(description="ID of the interpretability test"),
        response: str = Field(description="Answer response to grade"),
        expected: str = Field(description="Expected reference answer"),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Grade a model interpretability response against reference answer."""
        if ctx:
            await ctx.info(f"Grading response for test {test_id}...")

        # Simple grading logic: case-insensitive match or float match if float
        passed = False
        r_clean = response.strip().lower()
        e_clean = expected.strip().lower()

        if r_clean == e_clean:
            passed = True
        else:
            # Try parsing as float
            try:
                r_val = float(r_clean)
                e_val = float(e_clean)
                if abs(r_val - e_val) < 1e-4:
                    passed = True
            except ValueError:
                # Substring match fallback
                if e_clean in r_clean or r_clean in e_clean:
                    passed = True

        result = {
            "test_id": test_id,
            "passed": passed,
            "response": response,
            "expected": expected,
            "score": 1.0 if passed else 0.0,
        }
        _graded_responses[test_id] = result
        return result

    @mcp.tool(tags={"interpretability"})
    async def run_interpretability_suite(
        model_id: str = Field(description="ID of the fitted model"),
        answers_json: str = Field(
            description="JSON string mapping test_id to response answers"
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Run and grade the complete 6-category interpretability audit suite for a model."""
        if ctx:
            await ctx.info(
                f"Running full interpretability suite for model {model_id}..."
            )
        try:
            answers = json.loads(answers_json)
        except Exception as e:
            return {"error": f"Invalid answers_json: {e}"}

        engine = MLEngine()
        if model_id not in engine._models:
            return {"error": f"Model {model_id} not found."}

        # Reference answers from the engine-backed model (no sklearn object).
        ref = engine.interpretability_reference(model_id)
        if ref is None:
            return {"error": f"Model {model_id} not found."}

        expected_answers = {
            f"att_{model_id}_0": ref["largest_feature"],
            f"sim_{model_id}_0": str(round(ref["baseline_pred"], 4)),
            f"sens_{model_id}_0": str(round(ref["coef_first"], 4)),
            f"cf_{model_id}_0": "0.0",
            f"conf_{model_id}_0": str(round(ref["r2"], 4)),
            f"data_{model_id}_0": str(ref["n_train"]),
        }

        results = []
        score_sum = 0.0
        for test_id, expected in expected_answers.items():
            ans = answers.get(test_id, "no answer provided")
            # Grade
            passed = False
            r_clean = ans.strip().lower()
            e_clean = expected.strip().lower()
            if r_clean == e_clean:
                passed = True
            else:
                try:
                    r_val = float(r_clean)
                    e_val = float(e_clean)
                    if abs(r_val - e_val) < 1e-3:
                        passed = True
                except ValueError:
                    if e_clean in r_clean or r_clean in e_clean:
                        passed = True

            score = 1.0 if passed else 0.0
            score_sum += score
            results.append(
                {
                    "test_id": test_id,
                    "passed": passed,
                    "response": ans,
                    "expected": expected,
                    "score": score,
                }
            )

        total_tests = len(expected_answers)
        return {
            "model_id": model_id,
            "overall_score": score_sum / total_tests,
            "passed_count": sum(1 for r in results if r["passed"]),
            "failed_count": sum(1 for r in results if not r["passed"]),
            "detailed_results": results,
        }


# ── Data Management Tools ────────────────────────────────────────────


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


# ── Prompts ──────────────────────────────────────────────────────────


def register_prompts(mcp: FastMCP) -> None:
    """Register MCP prompts for guided workflows."""

    @mcp.prompt(
        name="model_evolution_workflow",
        description=(
            "Step-by-step guide for evolving interpretable models using "
            "the autoresearch loop from arXiv:2605.03808."
        ),
    )
    def model_evolution_workflow(dataset: str) -> str:
        return (
            f"You are running the Agentic-iModels autoresearch loop.\n\n"
            f"Dataset: {dataset}\n\n"
            f"Steps:\n"
            f"1. load_dataset('{dataset}') — Load and inspect the data\n"
            f"2. fit_model('LinearRegression', '{dataset}') — Baseline model\n"
            f"3. evaluate_model(model_id, '{dataset}') — Get accuracy metrics\n"
            f"4. generate_interpretability_tests(...) — Create test cases\n"
            f"5. run_interpretability_suite(...) — Grade the model\n"
            f"6. evolve_model_class(...) — Submit to Pareto frontier\n"
            f"7. Repeat steps 2-6 with increasingly complex model classes\n"
            f"8. get_pareto_frontier() — View the final frontier\n"
        )

    @mcp.prompt(
        name="interpretability_audit",
        description=(
            "Guide for auditing a model's interpretability using the "
            "6-category LLM-graded test protocol."
        ),
    )
    def interpretability_audit(model_name: str) -> str:
        return (
            f"You are auditing the interpretability of model '{model_name}'.\n\n"
            f"Test Categories (200 total):\n"
            f"  1. Feature Attribution (32): Which features matter most?\n"
            f"  2. Point Simulation (43): Can you predict for specific inputs?\n"
            f"  3. Sensitivity Analysis (32): How do changes propagate?\n"
            f"  4. Counterfactual (32): What inputs achieve a target output?\n"
            f"  5. Confidence Calibration (32): How certain are predictions?\n"
            f"  6. Data Attribution (29): Which training data influenced this?\n\n"
            f"Process:\n"
            f"1. Use generate_interpretability_tests() to create test cases\n"
            f"2. For each test, read the model's __str__() and answer the query\n"
            f"3. Use grade_response() to check each answer\n"
            f"4. Use run_interpretability_suite() for aggregate scoring\n"
        )


# ── MCP Initialization ──────────────────────────────────────────────


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize and return the Data Science MCP instance."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="Data Science MCP",
        version=__version__,
        instructions=(
            "Data Science MCP Server for model training, evaluation, and "
            "evolution. Integrates with agent-utilities IModelEvolver "
            " for Agentic-iModels workflows."
        ),
    )

    registered_tags = []

    if DEFAULT_MODEL_TRAININGTOOL:
        register_model_training_tools(mcp)
        register_training_data_tools(mcp)
        register_trainer_tools(mcp)
        register_kernel_specialize_tools(mcp)
        registered_tags.append("model-training")

    if DEFAULT_MODEL_EVOLUTIONTOOL:
        register_model_evolution_tools(mcp)
        registered_tags.append("model-evolution")

    if DEFAULT_INTERPRETABILITYTOOL:
        register_interpretability_tools(mcp)
        registered_tags.append("interpretability")

    if DEFAULT_DATA_MANAGEMENTTOOL:
        register_data_management_tools(mcp)
        registered_tags.append("data-management")

    if DEFAULT_DATA_ENGINETOOL:
        from data_science_mcp.mcp.mcp_data_engine import register_data_engine_tools

        register_data_engine_tools(mcp)
        registered_tags.append("data-engine")

    if DEFAULT_QUANTTOOL:
        register_quant_tools(mcp)
        registered_tags.append("quant")

    register_prompts(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    return mcp, args, middlewares, registered_tags


def mcp_server():
    """Entry point for the MCP server."""
    mcp, args, middlewares, registered_tags = get_mcp_instance()

    print(f"Data Science MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Tags: {', '.join(registered_tags)}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error(f"Invalid transport: {args.transport}")
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()

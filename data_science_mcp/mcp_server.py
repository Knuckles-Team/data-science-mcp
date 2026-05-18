#!/usr/bin/python
# coding: utf-8

"""Data Science MCP Server.

Provides model training, evaluation, and evolution tools for agentic ML
workflows. Designed as the MCP delegation target for agent-utilities'
IModelEvolver (CONCEPT:AHE-3.15) and InterpretabilityTestSuite
(CONCEPT:AHE-3.16).

Tool tags:
    - model-training: fit_model, predict, evaluate_model, cross_validate
    - model-evolution: evolve_model_class, rank_models, pareto_frontier
    - interpretability: run_interpretability_suite, grade_response, generate_tests
    - data-management: load_dataset, describe_dataset, split_dataset
"""

import json
import logging
import os
import sys
from typing import Any

from dotenv import find_dotenv, load_dotenv
from fastmcp import FastMCP
from pydantic import Field

from agent_utilities.base_utilities import to_boolean
from agent_utilities.mcp_utilities import create_mcp_server
from agent_utilities.base_utilities import get_logger

__version__ = "0.5.0"

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


# ── Model Training Tools ─────────────────────────────────────────────


def register_model_training_tools(mcp: FastMCP) -> None:
    """Register model training, prediction, and evaluation tools."""

    @mcp.tool(
        name="fit_model",
        description=(
            "Fit a scikit-learn-compatible model on a dataset. Returns fitted "
            "model metadata including class name, parameters, training RMSE, "
            "R², and the model's __str__() representation for LLM readability."
        ),
        tags=["model-training"],
    )
    def fit_model(
        model_class: str = Field(
            description=(
                "Fully qualified model class name or shorthand. "
                "Examples: 'sklearn.linear_model.Ridge', 'EBM', 'LinearRegression'"
            )),
        dataset_name: str = Field(
            description="Name of a registered dataset (e.g., 'boston', 'california')."),
        hyperparameters: str = Field(
            default="{}", description="JSON-encoded dict of hyperparameters for the model."),
        test_size: float = Field(
            default=0.2, description="Fraction of data to hold out for evaluation."),
    ) -> str:
        """Fit a model and return results as JSON."""
        try:
            params = json.loads(hyperparameters)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in hyperparameters"})

        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            result = engine.fit(
                model_class=model_class,
                dataset_name=dataset_name,
                hyperparameters=params,
                test_size=test_size,
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="predict",
        description=(
            "Generate predictions from a fitted model. Accepts JSON-encoded "
            "input features and returns predicted values."
        ),
        tags=["model-training"],
    )
    def predict(
        model_id: str = Field(
            description="ID of a previously fitted model from fit_model."),
        inputs: str = Field(
            description=(
                "JSON-encoded list of dicts, each containing feature values. "
                "Example: '[{\"x0\": 1.5, \"x1\": 2.0}]'"
            )),
    ) -> str:
        """Generate predictions from a fitted model."""
        try:
            input_data = json.loads(inputs)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in inputs"})

        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            predictions = engine.predict(model_id=model_id, inputs=input_data)
            return json.dumps({"predictions": predictions}, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="evaluate_model",
        description=(
            "Evaluate a fitted model on a dataset split. Returns RMSE, MAE, "
            "R², and per-sample prediction errors."
        ),
        tags=["model-training"],
    )
    def evaluate_model(
        model_id: str = Field(
            description="ID of a previously fitted model."),
        dataset_name: str = Field(
            description="Dataset to evaluate on."),
        split: str = Field(
            default="test", description="Data split to use: 'train', 'test', or 'validation'."),
    ) -> str:
        """Evaluate a fitted model."""
        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            metrics = engine.evaluate(
                model_id=model_id,
                dataset_name=dataset_name,
                split=split,
            )
            return json.dumps(metrics, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="cross_validate",
        description=(
            "Run k-fold cross-validation on a model class with a dataset. "
            "Returns per-fold and aggregate metrics."
        ),
        tags=["model-training"],
    )
    def cross_validate(
        model_class: str = Field(description="Model class name."),
        dataset_name: str = Field(description="Dataset name."),
        n_folds: int = Field(default=5, description="Number of CV folds."),
        hyperparameters: str = Field(
            default="{}", description="JSON-encoded hyperparameters."),
    ) -> str:
        """Run cross-validation."""
        try:
            params = json.loads(hyperparameters)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in hyperparameters"})

        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            result = engine.cross_validate(
                model_class=model_class,
                dataset_name=dataset_name,
                n_folds=n_folds,
                hyperparameters=params,
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Model Evolution Tools ────────────────────────────────────────────


def register_model_evolution_tools(mcp: FastMCP) -> None:
    """Register model evolution and Pareto frontier tools."""

    @mcp.tool(
        name="evolve_model_class",
        description=(
            "Submit a new model class candidate to the IModelEvolver. "
            "The model will be evaluated for both predictive accuracy "
            "and LLM interpretability, then added to the Pareto frontier "
            "if non-dominated. Returns the candidate's fitness metrics."
        ),
        tags=["model-evolution"],
    )
    def evolve_model_class(
        model_class_name: str = Field(
            description="Name of the model class (e.g., 'HingeEBM')."),
        source_code: str = Field(
            default="", description="Python source code of the model class."),
        str_output: str = Field(
            default="", description="The model's __str__() output after fitting."),
        rmse_scores: str = Field(
            default="{}", description=(
                "JSON dict of dataset_name→RMSE scores. "
                "Example: '{\"boston\": 3.2, \"california\": 0.8}'"
            )),
        interpretability_score: float = Field(
            default=0.0, description="LLM interpretability pass rate (0.0-1.0)."),
    ) -> str:
        """Register a candidate in the evolutionary loop."""
        try:
            scores = json.loads(rmse_scores)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON in rmse_scores"})

        try:
            from agent_utilities.harness.imodel_evolver import IModelEvolver

            evolver = IModelEvolver()
            candidate = evolver.register_candidate(
                model_class_name=model_class_name,
                source_code=source_code,
                str_output=str_output,
                rmse_scores=scores,
                interpretability_score=interpretability_score,
            )
            return json.dumps({
                "model_class_name": candidate.model_class_name,
                "predictive_rank": candidate.predictive_rank,
                "interpretability_score": candidate.interpretability_score,
                "status": "registered",
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="rank_models",
        description=(
            "Rank all registered model candidates by mean RMSE across "
            "datasets. Returns normalized ranks (0.0=best, 1.0=worst)."
        ),
        tags=["model-evolution"],
    )
    def rank_models() -> str:
        """Rank all registered candidates."""
        try:
            from agent_utilities.harness.imodel_evolver import IModelEvolver

            evolver = IModelEvolver()
            ranked = evolver.rank_models()
            return json.dumps([
                {
                    "model_class_name": c.model_class_name,
                    "predictive_rank": c.predictive_rank,
                    "interpretability_score": c.interpretability_score,
                }
                for c in ranked
            ], indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="get_pareto_frontier",
        description=(
            "Get the current Pareto frontier of accuracy vs interpretability. "
            "Returns all non-dominated models."
        ),
        tags=["model-evolution"],
    )
    def get_pareto_frontier() -> str:
        """Get current Pareto frontier."""
        try:
            from agent_utilities.harness.imodel_evolver import IModelEvolver

            evolver = IModelEvolver()
            frontier = evolver.evolve_round()
            return json.dumps([
                {
                    "model_id": p.model_id if hasattr(p, "model_id") else str(i),
                    "model_class_name": p.model_class_name,
                    "predictive_rank": p.predictive_rank,
                    "interpretability_score": p.interpretability_score,
                }
                for i, p in enumerate(frontier)
            ], indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Interpretability Tools ───────────────────────────────────────────


def register_interpretability_tools(mcp: FastMCP) -> None:
    """Register interpretability testing and grading tools."""

    @mcp.tool(
        name="run_interpretability_suite",
        description=(
            "Run the full 6-category interpretability test suite on a model. "
            "Categories: feature_attribution, point_simulation, sensitivity, "
            "counterfactual, confidence_calibration, data_attribution. "
            "Returns per-category pass rates and aggregate score."
        ),
        tags=["interpretability"],
    )
    def run_interpretability_suite(
        model_str: str = Field(
            description="The model's __str__() output to test against."),
        test_cases_json: str = Field(
            description=(
                "JSON array of test cases. Each: "
                "{category, query, ground_truth, tolerance?}"
            )),
        llm_responses_json: str = Field(
            description=(
                "JSON array of LLM responses, one per test case."
            )),
    ) -> str:
        """Run interpretability test suite."""
        try:
            test_cases_raw = json.loads(test_cases_json)
            llm_responses = json.loads(llm_responses_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"})

        try:
            from agent_utilities.harness.interpretability_tests import (
                InterpretabilityTestCase,
                InterpretabilityTestSuite,
            )
            from agent_utilities.models.imodel import (
                InterpretabilityTestCategory,
            )

            suite = InterpretabilityTestSuite()
            tests = [
                InterpretabilityTestCase(
                    category=InterpretabilityTestCategory(tc["category"]),
                    query=tc["query"],
                    ground_truth=tc["ground_truth"],
                    tolerance=tc.get("tolerance", 0.05),
                )
                for tc in test_cases_raw
            ]
            result = suite.run_suite(model_str, tests, llm_responses)
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="grade_response",
        description=(
            "Grade a single LLM response against a ground truth answer. "
            "Returns pass/fail, reasoning, and reward hacking check."
        ),
        tags=["interpretability"],
    )
    def grade_response(
        llm_response: str = Field(
            description="The LLM's response to grade."),
        ground_truth: str = Field(
            description="The expected correct answer."),
        model_str: str = Field(
            default="", description=(
                "The model's __str__() for reward hacking detection."
            )),
        tolerance: float = Field(
            default=0.05, description="Numerical tolerance for approximate matches."),
    ) -> str:
        """Grade a single response."""
        try:
            from agent_utilities.harness.interpretability_tests import (
                InterpretabilityGrader,
            )

            grader = InterpretabilityGrader()
            passed, reason = grader.grade(
                llm_response, ground_truth, tolerance=tolerance,
            )
            reward_hacking = (
                grader.detect_reward_hacking(model_str, ground_truth)
                if model_str
                else False
            )
            return json.dumps({
                "passed": passed and not reward_hacking,
                "reason": reason,
                "reward_hacking_detected": reward_hacking,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="generate_interpretability_tests",
        description=(
            "Generate interpretability test cases for a model. Produces "
            "feature attribution, point simulation, and sensitivity tests "
            "from model metadata."
        ),
        tags=["interpretability"],
    )
    def generate_interpretability_tests(
        feature_names: str = Field(
            description="JSON array of feature names."),
        coefficients: str = Field(
            default="[]", description="JSON array of coefficient values per feature."),
        inputs: str = Field(
            default="[]", description=(
                "JSON array of input dicts for point simulation tests."
            )),
        outputs: str = Field(
            default="[]", description="JSON array of output values for point simulation."),
    ) -> str:
        """Generate test cases from model metadata."""
        try:
            feat = json.loads(feature_names)
            coefs = json.loads(coefficients)
            inp = json.loads(inputs)
            out = json.loads(outputs)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}"})

        try:
            from agent_utilities.harness.interpretability_tests import (
                InterpretabilityTestSuite,
            )

            suite = InterpretabilityTestSuite()
            tests = []
            if coefs:
                tests.extend(
                    suite.generate_feature_attribution_tests(feat, coefs),
                )
            if inp and out:
                tests.extend(
                    suite.generate_point_simulation_tests(inp, out),
                )

            return json.dumps([
                {
                    "category": t.category.value,
                    "query": t.query,
                    "ground_truth": t.ground_truth,
                    "tolerance": t.tolerance,
                }
                for t in tests
            ], indent=2)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


# ── Data Management Tools ────────────────────────────────────────────


def register_data_management_tools(mcp: FastMCP) -> None:
    """Register dataset loading and manipulation tools."""

    @mcp.tool(
        name="load_dataset",
        description=(
            "Load a dataset by name. Supports scikit-learn built-in datasets "
            "(boston, california, diabetes, iris, wine) and CSV files."
        ),
        tags=["data-management"],
    )
    def load_dataset(
        name: str = Field(
            description=(
                "Dataset name or file path. Built-in: boston, california, "
                "diabetes, iris, wine."
            )),
        target_column: str = Field(
            default="", description="Target column name (for CSV files)."),
    ) -> str:
        """Load a dataset and return summary statistics."""
        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            summary = engine.load_dataset(
                name=name, target_column=target_column,
            )
            return json.dumps(summary, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="describe_dataset",
        description=(
            "Get descriptive statistics for a loaded dataset: shape, "
            "dtypes, mean, std, min, max, correlations."
        ),
        tags=["data-management"],
    )
    def describe_dataset(
        name: str = Field(description="Name of a loaded dataset."),
    ) -> str:
        """Describe a dataset."""
        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            stats = engine.describe_dataset(name=name)
            return json.dumps(stats, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

    @mcp.tool(
        name="split_dataset",
        description=(
            "Split a dataset into train/test/validation sets. Returns "
            "split sizes and indices."
        ),
        tags=["data-management"],
    )
    def split_dataset(
        name: str = Field(description="Dataset name."),
        test_size: float = Field(
            default=0.2, description="Test split fraction."),
        validation_size: float = Field(
            default=0.0, description="Validation split fraction (from train)."),
        random_seed: int = Field(
            default=42, description="Random seed for reproducibility."),
    ) -> str:
        """Split a dataset."""
        try:
            from data_science_mcp.ml_engine import MLEngine

            engine = MLEngine()
            result = engine.split_dataset(
                name=name,
                test_size=test_size,
                validation_size=validation_size,
                random_seed=random_seed,
            )
            return json.dumps(result, indent=2, default=str)
        except Exception as exc:
            return json.dumps({"error": str(exc)})


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
            "(CONCEPT:AHE-3.15) for Agentic-iModels workflows."
        ),
    )

    registered_tags = []

    if DEFAULT_MODEL_TRAININGTOOL:
        register_model_training_tools(mcp)
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

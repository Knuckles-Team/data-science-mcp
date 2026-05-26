"""MCP tools for interpretability operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import json

from fastmcp import Context, FastMCP
from pydantic import Field

from data_science_mcp.ml_engine import MLEngine
from data_science_mcp.mcp_server import _graded_responses


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

        # We can construct the exact reference answers from the MLEngine models!
        model_data = engine._models[model_id]
        model = model_data["model"]
        feature_names = model_data["feature_names"]

        import numpy as np

        # Att reference: largest coefficient
        largest_feature = "unknown"
        if hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            largest_idx = int(np.argmax(coefs))
            largest_feature = feature_names[largest_idx]
        elif hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            largest_idx = int(np.argmax(importances))
            largest_feature = feature_names[largest_idx]

        # Sim reference: baseline prediction
        baseline_pred = float(model.predict(np.zeros((1, len(feature_names))))[0])

        # Sens reference: coefficient of first feature
        coef_first = 0.0
        if hasattr(model, "coef_"):
            coef_first = float(model.coef_[0])
        elif hasattr(model, "feature_importances_"):
            coef_first = float(model.feature_importances_[0])

        # CF reference
        cf_val = "0.0"

        # Conf reference
        r2 = float(model.score(model_data["X_test"], model_data["y_test"]))

        # Data reference
        n_train = len(model_data["X_train"])

        expected_answers = {
            f"att_{model_id}_0": largest_feature,
            f"sim_{model_id}_0": str(round(baseline_pred, 4)),
            f"sens_{model_id}_0": str(round(coef_first, 4)),
            f"cf_{model_id}_0": cf_val,
            f"conf_{model_id}_0": str(round(r2, 4)),
            f"data_{model_id}_0": str(n_train),
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

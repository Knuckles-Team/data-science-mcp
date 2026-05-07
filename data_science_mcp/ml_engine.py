#!/usr/bin/python
"""ML Engine — Core model training and evaluation logic.

This module provides the actual scikit-learn integration for the
Data Science MCP Server. It handles:
    - Dataset loading (sklearn built-in + CSV)
    - Model fitting and prediction
    - Evaluation metrics (RMSE, MAE, R²)
    - Cross-validation
    - Dataset splitting

All operations are stateful: fitted models and loaded datasets are
stored in-memory and referenced by ID for subsequent operations.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class MLEngine:
    """Stateful ML engine for model training and evaluation.

    Manages fitted models and loaded datasets in-memory.
    Designed for delegation from MCP tool calls.
    """

    _instance: MLEngine | None = None
    _datasets: dict[str, dict[str, Any]] = {}
    _models: dict[str, dict[str, Any]] = {}

    def __new__(cls) -> MLEngine:
        """Singleton pattern for cross-tool state sharing."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_dataset(
        self,
        name: str,
        target_column: str = "",
    ) -> dict[str, Any]:
        """Load a dataset by name or file path.

        Args:
            name: Dataset name ('boston', 'california', 'diabetes',
                'iris', 'wine') or CSV file path.
            target_column: Target column name (for CSV files).

        Returns:
            Dict with dataset summary: shape, features, target, dtypes.
        """
        try:
            data: dict[str, Any] = {}

            sklearn_datasets = {
                "california": "fetch_california_housing",
                "diabetes": "load_diabetes",
                "iris": "load_iris",
                "wine": "load_wine",
                "breast_cancer": "load_breast_cancer",
                "digits": "load_digits",
            }

            if name.lower() in sklearn_datasets:
                from sklearn import datasets as sk_datasets

                loader = getattr(sk_datasets, sklearn_datasets[name.lower()])
                bunch = loader()
                data = {
                    "X": bunch.data,
                    "y": bunch.target,
                    "feature_names": list(
                        getattr(bunch, "feature_names", [f"x{i}" for i in range(bunch.data.shape[1])])
                    ),
                    "target_name": getattr(bunch, "target_names", ["target"])[0]
                    if hasattr(bunch, "target_names")
                    else "target",
                    "description": getattr(bunch, "DESCR", "")[:200],
                }
            elif name.endswith(".csv"):
                import pandas as pd

                df = pd.read_csv(name)
                if target_column and target_column in df.columns:
                    y = df[target_column].values
                    X = df.drop(columns=[target_column]).values
                    feature_names = [c for c in df.columns if c != target_column]
                else:
                    # Last column as target
                    y = df.iloc[:, -1].values
                    X = df.iloc[:, :-1].values
                    feature_names = list(df.columns[:-1])
                    target_column = df.columns[-1]

                data = {
                    "X": X,
                    "y": y,
                    "feature_names": feature_names,
                    "target_name": target_column,
                    "description": f"CSV dataset: {name}",
                }
            else:
                return {"error": f"Unknown dataset: {name}. Use sklearn names or .csv path."}

            self._datasets[name] = data

            return {
                "name": name,
                "shape": list(data["X"].shape),
                "n_features": data["X"].shape[1],
                "n_samples": data["X"].shape[0],
                "feature_names": data["feature_names"],
                "target_name": data.get("target_name", "target"),
                "description": data.get("description", ""),
            }

        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}. Install scikit-learn and pandas."}
        except Exception as exc:
            return {"error": str(exc)}

    def fit(
        self,
        model_class: str,
        dataset_name: str,
        hyperparameters: dict[str, Any] | None = None,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Fit a model on a dataset.

        Args:
            model_class: Model class name (e.g., 'LinearRegression', 'Ridge').
            dataset_name: Name of a loaded dataset.
            hyperparameters: Optional hyperparameters dict.
            test_size: Holdout fraction for evaluation.

        Returns:
            Dict with model_id, metrics, and str_representation.
        """
        if dataset_name not in self._datasets:
            # Try loading it
            result = self.load_dataset(dataset_name)
            if "error" in result:
                return result

        data = self._datasets[dataset_name]
        params = hyperparameters or {}

        try:
            import numpy as np
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                data["X"], data["y"], test_size=test_size, random_state=42,
            )

            # Resolve model class
            model = self._resolve_model(model_class, params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            rmse_train = float(np.sqrt(np.mean((y_train - y_pred_train) ** 2)))
            rmse_test = float(np.sqrt(np.mean((y_test - y_pred_test) ** 2)))
            r2_test = float(model.score(X_test, y_test))

            # Generate model ID
            model_id = f"model:{uuid.uuid4().hex[:8]}"
            str_repr = str(model)

            self._models[model_id] = {
                "model": model,
                "dataset": dataset_name,
                "feature_names": data["feature_names"],
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }

            return {
                "model_id": model_id,
                "model_class": model_class,
                "str_representation": str_repr,
                "metrics": {
                    "rmse_train": round(rmse_train, 6),
                    "rmse_test": round(rmse_test, 6),
                    "r2_test": round(r2_test, 6),
                },
                "n_train": len(X_train),
                "n_test": len(X_test),
                "hyperparameters": params,
                "feature_names": data["feature_names"],
            }

        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}"}
        except Exception as exc:
            return {"error": str(exc)}

    def predict(
        self,
        model_id: str,
        inputs: list[dict[str, Any]],
    ) -> list[float]:
        """Generate predictions from a fitted model.

        Args:
            model_id: ID from a previous fit() call.
            inputs: List of feature value dicts.

        Returns:
            List of predicted values.
        """
        if model_id not in self._models:
            raise ValueError(f"Unknown model_id: {model_id}")

        import numpy as np

        model_data = self._models[model_id]
        model = model_data["model"]
        feature_names = model_data["feature_names"]

        # Convert input dicts to numpy array
        X = np.array([
            [row.get(f, 0.0) for f in feature_names]
            for row in inputs
        ])

        predictions = model.predict(X)
        return [float(p) for p in predictions]

    def evaluate(
        self,
        model_id: str,
        dataset_name: str,
        split: str = "test",
    ) -> dict[str, Any]:
        """Evaluate a fitted model.

        Args:
            model_id: ID from a previous fit() call.
            dataset_name: Dataset to evaluate on.
            split: 'train' or 'test'.

        Returns:
            Dict with RMSE, MAE, R², and per-sample errors.
        """
        if model_id not in self._models:
            return {"error": f"Unknown model_id: {model_id}"}

        import numpy as np

        model_data = self._models[model_id]
        model = model_data["model"]

        if split == "train":
            X, y = model_data["X_train"], model_data["y_train"]
        else:
            X, y = model_data["X_test"], model_data["y_test"]

        y_pred = model.predict(X)
        errors = y - y_pred

        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))
        r2 = float(model.score(X, y))

        return {
            "model_id": model_id,
            "split": split,
            "n_samples": len(y),
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
        }

    def cross_validate(
        self,
        model_class: str,
        dataset_name: str,
        n_folds: int = 5,
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run k-fold cross-validation.

        Args:
            model_class: Model class name.
            dataset_name: Dataset name.
            n_folds: Number of CV folds.
            hyperparameters: Optional hyperparameters.

        Returns:
            Dict with per-fold and aggregate metrics.
        """
        if dataset_name not in self._datasets:
            result = self.load_dataset(dataset_name)
            if "error" in result:
                return result

        data = self._datasets[dataset_name]
        params = hyperparameters or {}

        try:
            import numpy as np
            from sklearn.model_selection import cross_val_score

            model = self._resolve_model(model_class, params)

            scores = cross_val_score(
                model, data["X"], data["y"],
                cv=n_folds, scoring="neg_root_mean_squared_error",
            )

            rmse_scores = [-s for s in scores]

            return {
                "model_class": model_class,
                "dataset": dataset_name,
                "n_folds": n_folds,
                "rmse_per_fold": [round(s, 6) for s in rmse_scores],
                "rmse_mean": round(float(np.mean(rmse_scores)), 6),
                "rmse_std": round(float(np.std(rmse_scores)), 6),
            }

        except ImportError as exc:
            return {"error": f"Missing dependency: {exc}"}
        except Exception as exc:
            return {"error": str(exc)}

    def describe_dataset(self, name: str) -> dict[str, Any]:
        """Get descriptive statistics for a loaded dataset.

        Args:
            name: Name of a loaded dataset.

        Returns:
            Dict with shape, dtypes, summary statistics.
        """
        if name not in self._datasets:
            return {"error": f"Dataset '{name}' not loaded. Use load_dataset() first."}

        import numpy as np

        data = self._datasets[name]
        X = data["X"]
        y = data["y"]

        return {
            "name": name,
            "shape": list(X.shape),
            "feature_names": data["feature_names"],
            "feature_stats": {
                fname: {
                    "mean": round(float(np.mean(X[:, i])), 4),
                    "std": round(float(np.std(X[:, i])), 4),
                    "min": round(float(np.min(X[:, i])), 4),
                    "max": round(float(np.max(X[:, i])), 4),
                }
                for i, fname in enumerate(data["feature_names"])
            },
            "target_stats": {
                "mean": round(float(np.mean(y)), 4),
                "std": round(float(np.std(y)), 4),
                "min": round(float(np.min(y)), 4),
                "max": round(float(np.max(y)), 4),
            },
        }

    def split_dataset(
        self,
        name: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Split a dataset into train/test/validation sets.

        Args:
            name: Dataset name.
            test_size: Test split fraction.
            validation_size: Validation split fraction (from train).
            random_seed: Random seed.

        Returns:
            Dict with split sizes.
        """
        if name not in self._datasets:
            return {"error": f"Dataset '{name}' not loaded."}

        from sklearn.model_selection import train_test_split

        data = self._datasets[name]
        X, y = data["X"], data["y"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed,
        )

        result: dict[str, Any] = {
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

        if validation_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=validation_size,
                random_state=random_seed,
            )
            result["train_size"] = len(X_train)
            result["validation_size"] = len(X_val)

        result["total"] = len(X)
        return result

    @staticmethod
    def _resolve_model(model_class: str, params: dict[str, Any]) -> Any:
        """Resolve a model class name to an sklearn estimator instance.

        Args:
            model_class: Model class name or shorthand.
            params: Hyperparameters dict.

        Returns:
            Fitted sklearn estimator instance.
        """
        from sklearn import linear_model, tree, ensemble, svm

        # Shorthand mapping
        shortcuts: dict[str, Any] = {
            "linearregression": linear_model.LinearRegression,
            "ridge": linear_model.Ridge,
            "lasso": linear_model.Lasso,
            "elasticnet": linear_model.ElasticNet,
            "decisiontree": tree.DecisionTreeRegressor,
            "randomforest": ensemble.RandomForestRegressor,
            "gradientboosting": ensemble.GradientBoostingRegressor,
            "svr": svm.SVR,
            "adaboost": ensemble.AdaBoostRegressor,
        }

        key = model_class.lower().replace("_", "").replace("-", "")
        if key in shortcuts:
            return shortcuts[key](**params)

        # Try fully qualified name
        parts = model_class.rsplit(".", 1)
        if len(parts) == 2:
            import importlib

            module = importlib.import_module(parts[0])
            cls = getattr(module, parts[1])
            return cls(**params)

        # Default to LinearRegression
        logger.warning(
            "Unknown model class '%s', defaulting to LinearRegression",
            model_class,
        )
        return linear_model.LinearRegression(**params)

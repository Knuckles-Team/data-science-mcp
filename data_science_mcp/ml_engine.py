#!/usr/bin/python
"""ML Engine — Core model training and evaluation logic.

Compute (fit / predict / evaluate / cross-validation / stats / split) runs in the
Rust ``epistemic-graph`` engine over its MessagePack/UDS protocol — there is no
scikit-learn compute path. The engine is a required dependency for model
operations; scikit-learn is optional and used only by the built-in sample-dataset
loaders (``data-science-mcp[datasets]``). CSV loading prefers polars, with a
# stdlib-csv + numpy fallback when polars is unavailable.

All operations are stateful: fitted models and loaded datasets are stored
in-memory and referenced by ID for subsequent operations.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Models the engine can fit. Keys are normalized (lowercase, alphanumerics only).
# OLS uses a lightweight local-predict path; the rest fit/predict entirely in the
# engine via the estimator protocol.
_RUST_LINEAR_MODELS = {"linearregression"}
_RUST_ESTIMATOR_MODELS = {
    "ridge",
    "lasso",
    "elasticnet",
    "decisiontree",
    "decisiontreeregressor",
    "randomforest",
    "randomforestregressor",
    "gradientboosting",
    "gradientboostingregressor",
    "adaboost",
    "adaboostregressor",
    "svr",
}
_SUPPORTED_MODELS = _RUST_LINEAR_MODELS | _RUST_ESTIMATOR_MODELS

# Hyperparameter names the engine's EstimatorParams understands. Anything else is
# dropped (the engine applies scikit-learn-compatible defaults).
_RUST_PARAM_KEYS = {
    "alpha",
    "l1_ratio",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "n_estimators",
    "learning_rate",
    "max_features",
    "subsample",
    "random_state",
    "C",
    "epsilon",
    "gamma",
    "kernel",
    "max_iter",
    "tol",
}

_ENGINE_REQUIRED_ERR = (
    "epistemic-graph engine unavailable. Set EPISTEMIC_GRAPH_SOCKET or "
    "EPISTEMIC_GRAPH_TCP and ensure the engine is running."
)

# Sentinel so a cached None ("engine not reachable") is distinct from "unprobed".
_UNPROBED = object()


class MLEngine:
    """Stateful ML engine for model training and evaluation.

    Manages fitted models and loaded datasets in-memory. Compute is delegated to
    the Rust epistemic-graph engine.
    """

    _instance: MLEngine | None = None
    _datasets: dict[str, dict[str, Any]] = {}
    _models: dict[str, dict[str, Any]] = {}
    _rust_client_cache: Any = _UNPROBED

    def __new__(cls) -> MLEngine:
        """Singleton pattern for cross-tool state sharing."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ── Rust epistemic-graph engine integration (CONCEPT:EG-KG.compute.rust-native-training-loss) ──────
    @classmethod
    def _rust_client(cls) -> Any:
        """Return a cached SyncEpistemicGraphClient, or None if unreachable.

        Connects when an endpoint is configured via ``EPISTEMIC_GRAPH_SOCKET`` /
        ``GRAPH_SERVICE_SOCKET`` (UDS) or ``EPISTEMIC_GRAPH_TCP`` (host:port), and
        otherwise tries the client's own default socket. The result (including a
        failed probe) is cached so we never re-probe a dead endpoint per call.
        Compute methods surface a clear error when this returns None.
        """
        if cls._rust_client_cache is not _UNPROBED:
            return cls._rust_client_cache

        client = None
        primary_socket = os.environ.get("EPISTEMIC_GRAPH_SOCKET")
        legacy_socket = os.environ.get("GRAPH_SERVICE_SOCKET")
        socket_path = primary_socket or legacy_socket
        tcp_addr = os.environ.get("EPISTEMIC_GRAPH_TCP")
        try:
            from epistemic_graph.client import SyncEpistemicGraphClient

            if tcp_addr:
                client = SyncEpistemicGraphClient.connect(tcp_addr=tcp_addr)
            elif socket_path:
                client = SyncEpistemicGraphClient.connect(socket_path=socket_path)
            else:
                client = SyncEpistemicGraphClient.connect()
            logger.info("epistemic-graph engine connected; Rust compute enabled")
        except Exception as exc:  # noqa: BLE001 — surfaced by compute methods
            logger.warning("epistemic-graph engine unavailable: %s", exc)
            client = None
        cls._rust_client_cache = client
        return client

    @staticmethod
    def _normalize_model(model_class: str) -> str:
        return "".join(c for c in model_class.lower() if c.isalnum())

    @classmethod
    def _is_supported(cls, model_class: str) -> bool:
        return cls._normalize_model(model_class) in _SUPPORTED_MODELS

    @staticmethod
    def _map_rust_params(params: dict[str, Any]) -> dict[str, Any]:
        """Forward only EstimatorParams-compatible hyperparameters to the engine."""
        return {k: v for k, v in params.items() if k in _RUST_PARAM_KEYS}

    @staticmethod
    def _linear_predict(X: Any, coef: list[float], intercept: float) -> Any:
        """y = X @ coef + intercept (small matvec — kept local, not over the wire)."""
        import numpy as np

        return np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float) + intercept

    @staticmethod
    def _rmse(y_true: Any, y_pred: Any) -> float:
        import numpy as np

        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        return float(np.sqrt(np.mean((yt - yp) ** 2)))

    @staticmethod
    def _r2(y_true: Any, y_pred: Any) -> float:
        import numpy as np

        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # ── Backend-agnostic model introspection (replaces sklearn attrs) ──
    @staticmethod
    def _linear_coefficients(model_data: dict[str, Any]) -> list[float] | None:
        """Linear coefficients for linear-family models, else None (tree/ensemble/
        SVR expose no linear coefficients)."""
        if model_data.get("backend") == "rust":
            return model_data.get("coefficients")
        if model_data.get("backend") == "rust_estimator":
            blob = model_data.get("model_blob", {})
            if blob.get("kind") == "Linear":
                return blob.get("model", {}).get("coefficients")
        return None

    def interpretability_reference(self, model_id: str) -> dict[str, Any] | None:
        """Compute reference answers for the interpretability suite without any
        scikit-learn model object: attribution (largest coefficient), baseline
        prediction, first-feature coefficient, test R², and train size."""
        if model_id not in self._models:
            return None

        import numpy as np

        md = self._models[model_id]
        feature_names = md["feature_names"]
        coefs = self._linear_coefficients(md)

        largest_feature = "unknown"
        coef_first = 0.0
        if coefs:
            arr = np.abs(np.asarray(coefs, dtype=float))
            largest_feature = feature_names[int(np.argmax(arr))]
            coef_first = float(coefs[0])

        baseline_pred = float(
            self.predict(model_id, [{f: 0.0 for f in feature_names}])[0]
        )

        ev = self.evaluate(model_id, md.get("dataset", ""), "test")
        r2 = float(ev["r2"]) if "error" not in ev else float(md.get("r2_test", 0.0))

        return {
            "feature_names": feature_names,
            "largest_feature": largest_feature,
            "baseline_pred": baseline_pred,
            "coef_first": coef_first,
            "r2": r2,
            "n_train": len(md["X_train"]),
        }

    def ranked_models(self) -> list[dict[str, Any]]:
        """Rank fitted models by stored test R² (backend-agnostic, no recompute)."""
        ranked = [
            {
                "model_id": mid,
                "dataset": md.get("dataset", ""),
                "model_str": md.get("str_representation", ""),
                "r2_test": md.get("r2_test", 0.0),
            }
            for mid, md in self._models.items()
        ]
        ranked.sort(key=lambda x: x["r2_test"], reverse=True)
        return ranked

    def load_dataset(
        self,
        name: str,
        target_column: str = "",
    ) -> dict[str, Any]:
        """Load a dataset by name or file path.

        Args:
            name: Built-in sample name ('california', 'diabetes', 'iris', 'wine',
                'breast_cancer', 'digits') — requires the optional ``[datasets]``
                extra (scikit-learn) — or a ``.csv`` file path.
            target_column: Target column name (for CSV files).

        Returns:
            Dict with dataset summary: shape, features, target, description.
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
                try:
                    from sklearn import datasets as sk_datasets
                except ImportError:
                    return {
                        "error": (
                            "Built-in sample datasets require scikit-learn. Install "
                            "'data-science-mcp[datasets]' or pass a .csv file path."
                        )
                    }

                loader = getattr(sk_datasets, sklearn_datasets[name.lower()])
                bunch = loader()
                data = {
                    "X": bunch.data,
                    "y": bunch.target,
                    "feature_names": list(
                        getattr(
                            bunch,
                            "feature_names",
                            [f"x{i}" for i in range(bunch.data.shape[1])],
                        )
                    ),
                    "target_name": getattr(bunch, "target_names", ["target"])[0]
                    if hasattr(bunch, "target_names")
                    else "target",
                    "description": getattr(bunch, "DESCR", "")[:200],
                }
            elif name.endswith(".csv"):
                # Fast path: polars when available; otherwise stdlib csv + numpy,
                # so numeric CSV loading still works without the polars wheel
                # (which core-dumps on CPUs lacking its SIMD baseline).
                import numpy as np

                try:
                    import polars as pl

                    df = pl.read_csv(name)
                    columns = list(df.columns)
                    matrix = df.to_numpy()
                except ImportError:
                    import csv as _csv

                    with open(name, newline="") as fh:
                        reader = _csv.reader(fh)
                        columns = next(reader)
                        rows = [[float(v) for v in row] for row in reader if row]
                    matrix = np.asarray(rows, dtype=float)

                if target_column and target_column in columns:
                    t_idx = columns.index(target_column)
                else:
                    # Last column as target
                    t_idx = len(columns) - 1
                    target_column = columns[t_idx]
                y = matrix[:, t_idx]
                X = np.delete(matrix, t_idx, axis=1)
                feature_names = [c for i, c in enumerate(columns) if i != t_idx]

                data = {
                    "X": X,
                    "y": y,
                    "feature_names": feature_names,
                    "target_name": target_column,
                    "description": f"CSV dataset: {name}",
                }
            else:
                return {
                    "error": f"Unknown dataset: {name}. Use a sample name or .csv path."
                }

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
            return {"error": f"Missing dependency: {exc}."}
        except Exception as exc:
            return {"error": str(exc)}

    def fit(
        self,
        model_class: str,
        dataset_name: str,
        hyperparameters: dict[str, Any] | None = None,
        test_size: float = 0.2,
    ) -> dict[str, Any]:
        """Fit a model on a dataset via the epistemic-graph engine.

        Args:
            model_class: Model name (e.g., 'LinearRegression', 'Ridge',
                'RandomForest', 'SVR').
            dataset_name: Name of a loaded dataset (auto-loaded if a known name).
            hyperparameters: Optional hyperparameters dict.
            test_size: Holdout fraction for evaluation.

        Returns:
            Dict with model_id, metrics, and metadata.
        """
        if dataset_name not in self._datasets:
            result = self.load_dataset(dataset_name)
            if "error" in result:
                return result

        data = self._datasets[dataset_name]
        params = hyperparameters or {}

        if not self._is_supported(model_class):
            return {
                "error": (
                    f"Unsupported model '{model_class}'. Supported: "
                    f"{sorted(_SUPPORTED_MODELS)}"
                )
            }

        client = self._rust_client()
        if client is None:
            return {"error": _ENGINE_REQUIRED_ERR}

        key = self._normalize_model(model_class)
        if key in _RUST_LINEAR_MODELS and not params:
            result = self._fit_rust(client, model_class, dataset_name, data, test_size)
        else:
            result = self._fit_rust_estimator(
                client, model_class, key, dataset_name, data, params, test_size
            )
        if result is None:
            return {"error": "Engine fit failed; see server logs."}
        return result

    def _fit_rust(
        self,
        client: Any,
        model_class: str,
        dataset_name: str,
        data: dict[str, Any],
        test_size: float,
    ) -> dict[str, Any] | None:
        """Fit OLS via the engine. Arrays are shipped whole (one round-trip each)."""
        try:
            import numpy as np

            x_list = np.asarray(data["X"], dtype=float).tolist()
            y_list = np.asarray(data["y"], dtype=float).ravel().tolist()

            split = client.datascience.train_test_split(
                x_list, y_list, test_ratio=test_size
            )
            x_train, x_test = split["x_train"], split["x_test"]
            y_train, y_test = split["y_train"], split["y_test"]

            reg = client.datascience.linear_regression(x_train, y_train)
            coef, intercept = reg["coefficients"], reg["intercept"]

            rmse_train = self._rmse(
                y_train, self._linear_predict(x_train, coef, intercept)
            )
            rmse_test = self._rmse(
                y_test, self._linear_predict(x_test, coef, intercept)
            )
            r2_test = self._r2(y_test, self._linear_predict(x_test, coef, intercept))

            model_id = f"model:{uuid.uuid4().hex[:8]}"
            str_repr = (
                f"RustLinearRegression(coef={len(coef)} features, "
                f"intercept={intercept:.6f})"
            )
            self._models[model_id] = {
                "backend": "rust",
                "coefficients": coef,
                "intercept": intercept,
                "dataset": dataset_name,
                "feature_names": data["feature_names"],
                "model_class": model_class,
                "str_representation": str_repr,
                "r2_test": round(r2_test, 6),
                "X_train": x_train,
                "X_test": x_test,
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
                "n_train": len(x_train),
                "n_test": len(x_test),
                "hyperparameters": {},
                "feature_names": data["feature_names"],
                "backend": "rust",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Rust OLS fit failed: %s", exc)
            return None

    def _fit_rust_estimator(
        self,
        client: Any,
        model_class: str,
        key: str,
        dataset_name: str,
        data: dict[str, Any],
        params: dict[str, Any],
        test_size: float,
    ) -> dict[str, Any] | None:
        """Fit a linear/tree/ensemble/SVR model via the engine. fit and predict
        both run in the engine; the serialized model blob is stored for reuse."""
        try:
            import numpy as np

            x_list = np.asarray(data["X"], dtype=float).tolist()
            y_list = np.asarray(data["y"], dtype=float).ravel().tolist()

            split = client.datascience.train_test_split(
                x_list, y_list, test_ratio=test_size
            )
            x_train, x_test = split["x_train"], split["x_test"]
            y_train, y_test = split["y_train"], split["y_test"]

            blob = client.datascience.fit_estimator(
                key, x_train, y_train, self._map_rust_params(params)
            )
            yhat_train = client.datascience.predict_estimator(blob, x_train)
            yhat_test = client.datascience.predict_estimator(blob, x_test)

            rmse_train = self._rmse(y_train, yhat_train)
            rmse_test = self._rmse(y_test, yhat_test)
            r2_test = self._r2(y_test, yhat_test)

            model_id = f"model:{uuid.uuid4().hex[:8]}"
            str_repr = f"Rust::{blob.get('kind', key)}"
            self._models[model_id] = {
                "backend": "rust_estimator",
                "model_blob": blob,
                "estimator": key,
                "dataset": dataset_name,
                "feature_names": data["feature_names"],
                "model_class": model_class,
                "str_representation": str_repr,
                "r2_test": round(r2_test, 6),
                "X_train": x_train,
                "X_test": x_test,
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
                "n_train": len(x_train),
                "n_test": len(x_test),
                "hyperparameters": params,
                "feature_names": data["feature_names"],
                "backend": "rust_estimator",
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Rust estimator fit failed: %s", exc)
            return None

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
        feature_names = model_data["feature_names"]
        X = np.array([[row.get(f, 0.0) for f in feature_names] for row in inputs])

        backend = model_data.get("backend")
        if backend == "rust":
            preds = self._linear_predict(
                X, model_data["coefficients"], model_data["intercept"]
            )
            return [float(p) for p in np.asarray(preds).ravel()]
        if backend == "rust_estimator":
            client = self._rust_client()
            if client is None:
                raise ValueError(_ENGINE_REQUIRED_ERR)
            preds = client.datascience.predict_estimator(
                model_data["model_blob"], X.tolist()
            )
            return [float(p) for p in preds]

        raise ValueError(f"Model {model_id} has no usable backend")

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
            Dict with RMSE, MAE, R².
        """
        if model_id not in self._models:
            return {"error": f"Unknown model_id: {model_id}"}

        import numpy as np

        model_data = self._models[model_id]

        if split == "train":
            X, y = model_data["X_train"], model_data["y_train"]
        else:
            X, y = model_data["X_test"], model_data["y_test"]

        backend = model_data.get("backend")
        if backend == "rust":
            y_pred = np.asarray(
                self._linear_predict(
                    X, model_data["coefficients"], model_data["intercept"]
                )
            ).ravel()
        elif backend == "rust_estimator":
            client = self._rust_client()
            if client is None:
                return {"error": _ENGINE_REQUIRED_ERR}
            y_pred = np.asarray(
                client.datascience.predict_estimator(
                    model_data["model_blob"], np.asarray(X, dtype=float).tolist()
                )
            ).ravel()
        else:
            return {"error": f"Model {model_id} has no usable backend"}

        y_arr = np.asarray(y, dtype=float).ravel()
        errors = y_arr - y_pred
        rmse = float(np.sqrt(np.mean(errors**2)))
        mae = float(np.mean(np.abs(errors)))
        r2 = self._r2(y_arr, y_pred)

        return {
            "model_id": model_id,
            "split": split,
            "n_samples": len(y_arr),
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
        """Run k-fold cross-validation via the engine.

        Uses contiguous folds to match scikit-learn's ``KFold(shuffle=False)``
        (the default for ``cv=int``).

        Returns:
            Dict with per-fold and aggregate RMSE.
        """
        if dataset_name not in self._datasets:
            result = self.load_dataset(dataset_name)
            if "error" in result:
                return result

        data = self._datasets[dataset_name]
        params = hyperparameters or {}

        if not self._is_supported(model_class):
            return {
                "error": (
                    f"Unsupported model '{model_class}'. Supported: "
                    f"{sorted(_SUPPORTED_MODELS)}"
                )
            }

        client = self._rust_client()
        if client is None:
            return {"error": _ENGINE_REQUIRED_ERR}

        key = self._normalize_model(model_class)
        if key in _RUST_LINEAR_MODELS and not params:
            result = self._cross_validate_rust(
                client, model_class, dataset_name, data, n_folds
            )
        else:
            result = self._cross_validate_rust_estimator(
                client, model_class, key, dataset_name, data, n_folds, params
            )
        if result is None:
            return {"error": "Engine cross-validation failed; see server logs."}
        return result

    def _cross_validate_rust(
        self,
        client: Any,
        model_class: str,
        dataset_name: str,
        data: dict[str, Any],
        n_folds: int,
    ) -> dict[str, Any] | None:
        """OLS k-fold CV via the engine (contiguous folds)."""
        try:
            import numpy as np

            X = np.asarray(data["X"], dtype=float)
            y = np.asarray(data["y"], dtype=float).ravel()
            n = len(X)
            if n_folds < 2 or n_folds > n:
                return None

            fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
            fold_sizes[: n % n_folds] += 1

            rmse_scores: list[float] = []
            current = 0
            for fs in fold_sizes:
                test_idx = list(range(current, current + int(fs)))
                train_idx = list(range(0, current)) + list(range(current + int(fs), n))
                current += int(fs)

                reg = client.datascience.linear_regression(
                    X[train_idx].tolist(), y[train_idx].tolist()
                )
                y_pred = self._linear_predict(
                    X[test_idx], reg["coefficients"], reg["intercept"]
                )
                rmse_scores.append(self._rmse(y[test_idx], y_pred))

            return self._cv_summary(
                model_class, dataset_name, n_folds, rmse_scores, "rust"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rust OLS cross-validate failed: %s", exc)
            return None

    def _cross_validate_rust_estimator(
        self,
        client: Any,
        model_class: str,
        key: str,
        dataset_name: str,
        data: dict[str, Any],
        n_folds: int,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Tree/ensemble/SVR k-fold CV via the engine (contiguous folds)."""
        try:
            import numpy as np

            X = np.asarray(data["X"], dtype=float)
            y = np.asarray(data["y"], dtype=float).ravel()
            n = len(X)
            if n_folds < 2 or n_folds > n:
                return None

            mapped = self._map_rust_params(params)
            fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
            fold_sizes[: n % n_folds] += 1

            rmse_scores: list[float] = []
            current = 0
            for fs in fold_sizes:
                test_idx = list(range(current, current + int(fs)))
                train_idx = list(range(0, current)) + list(range(current + int(fs), n))
                current += int(fs)

                blob = client.datascience.fit_estimator(
                    key, X[train_idx].tolist(), y[train_idx].tolist(), mapped
                )
                y_pred = client.datascience.predict_estimator(
                    blob, X[test_idx].tolist()
                )
                rmse_scores.append(self._rmse(y[test_idx], y_pred))

            return self._cv_summary(
                model_class, dataset_name, n_folds, rmse_scores, "rust_estimator"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rust estimator cross-validate failed: %s", exc)
            return None

    @staticmethod
    def _cv_summary(
        model_class: str,
        dataset_name: str,
        n_folds: int,
        rmse_scores: list[float],
        backend: str,
    ) -> dict[str, Any]:
        import numpy as np

        return {
            "model_class": model_class,
            "dataset": dataset_name,
            "n_folds": n_folds,
            "rmse_per_fold": [round(s, 6) for s in rmse_scores],
            "rmse_mean": round(float(np.mean(rmse_scores)), 6),
            "rmse_std": round(float(np.std(rmse_scores)), 6),
            "backend": backend,
        }

    def describe_dataset(self, name: str) -> dict[str, Any]:
        """Get descriptive statistics for a loaded dataset.

        Prefers the engine's ``compute_stats`` (one round-trip for features, one
        for the target); falls back to numpy if the engine is unavailable.
        """
        if name not in self._datasets:
            return {"error": f"Dataset '{name}' not loaded. Use load_dataset() first."}

        import numpy as np

        data = self._datasets[name]
        X = data["X"]
        y = data["y"]

        client = self._rust_client()
        if client is not None:
            rust_desc = self._describe_rust(client, name, data, X, y)
            if rust_desc is not None:
                return rust_desc

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

    def _describe_rust(
        self, client: Any, name: str, data: dict[str, Any], X: Any, y: Any
    ) -> dict[str, Any] | None:
        """Descriptive stats via the engine. Returns None on failure. Note: the
        engine reports sample std (ddof=1); the numpy fallback uses population
        std (ddof=0)."""
        try:
            import numpy as np

            stats = client.datascience.compute_stats(
                np.asarray(X, dtype=float).tolist()
            )
            tstats = client.datascience.compute_stats(
                [[float(v)] for v in np.asarray(y, dtype=float).ravel()]
            )
            return {
                "name": name,
                "shape": list(X.shape),
                "feature_names": data["feature_names"],
                "feature_stats": {
                    fname: {
                        "mean": round(stats["means"][i], 4),
                        "std": round(stats["stds"][i], 4),
                        "min": round(stats["mins"][i], 4),
                        "max": round(stats["maxs"][i], 4),
                    }
                    for i, fname in enumerate(data["feature_names"])
                },
                "target_stats": {
                    "mean": round(tstats["means"][0], 4),
                    "std": round(tstats["stds"][0], 4),
                    "min": round(tstats["mins"][0], 4),
                    "max": round(tstats["maxs"][0], 4),
                },
                "backend": "rust",
            }
        except Exception as exc:  # noqa: BLE001 — degrade to numpy
            logger.warning("Rust describe failed, falling back to numpy: %s", exc)
            return None

    def split_dataset(
        self,
        name: str,
        test_size: float = 0.2,
        validation_size: float = 0.0,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Split a dataset into train/test/validation sizes.

        Prefers the engine's seeded-shuffle split; falls back to a numpy shuffle
        split if the engine is unavailable (no scikit-learn dependency).
        """
        if name not in self._datasets:
            return {"error": f"Dataset '{name}' not loaded."}

        data = self._datasets[name]
        X, y = data["X"], data["y"]

        client = self._rust_client()
        if client is not None:
            rust_split = self._split_rust(
                client, X, y, test_size, validation_size, random_seed
            )
            if rust_split is not None:
                return rust_split

        return self._split_numpy(X, y, test_size, validation_size, random_seed)

    def _split_rust(
        self,
        client: Any,
        X: Any,
        y: Any,
        test_size: float,
        validation_size: float,
        random_seed: int,
    ) -> dict[str, Any] | None:
        """Train/test(/validation) split via the engine (seeded shuffle)."""
        try:
            import numpy as np

            x_list = np.asarray(X, dtype=float).tolist()
            y_list = np.asarray(y, dtype=float).ravel().tolist()
            total = len(x_list)

            split = client.datascience.train_test_split(
                x_list, y_list, test_ratio=test_size, shuffle=True, seed=random_seed
            )
            x_train = split["x_train"]
            result: dict[str, Any] = {
                "train_size": len(x_train),
                "test_size": len(split["x_test"]),
            }

            if validation_size > 0:
                sub = client.datascience.train_test_split(
                    x_train,
                    split["y_train"],
                    test_ratio=validation_size,
                    shuffle=True,
                    seed=random_seed,
                )
                result["train_size"] = len(sub["x_train"])
                result["validation_size"] = len(sub["x_test"])

            result["total"] = total
            result["backend"] = "rust"
            return result
        except Exception as exc:  # noqa: BLE001 — degrade to numpy
            logger.warning("Rust split failed, falling back to numpy: %s", exc)
            return None

    @staticmethod
    def _split_numpy(
        X: Any,
        y: Any,
        test_size: float,
        validation_size: float,
        random_seed: int,
    ) -> dict[str, Any]:
        """Pure-numpy shuffle split (no scikit-learn)."""

        n = len(X)
        n_test = int(round(n * test_size))
        n_train = n - n_test
        result: dict[str, Any] = {"train_size": n_train, "test_size": n_test}

        if validation_size > 0:
            n_val = int(round(n_train * validation_size))
            result["train_size"] = n_train - n_val
            result["validation_size"] = n_val

        result["total"] = n
        result["backend"] = "numpy"
        return result

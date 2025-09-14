"""
Simple standalone ML model trainer for M3.
Works with numpy arrays without complex configuration.
"""

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Try importing advanced models
try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelResult:
    """Container for model training results."""

    model: Any
    model_type: str
    metrics: Dict[str, float]
    feature_importance: Optional[pd.Series]
    training_time: float
    parameters: Dict[str, Any]


class ModelTrainer:
    """
    Simple ML model trainer that works with numpy arrays.
    Supports multiple model types and automatic hyperparameter tuning.
    """

    def __init__(
        self,
        models: Optional[List[str]] = None,
        target_type: str = "classification",
        random_state: int = 42,
    ):
        """
        Initialize model trainer.

        Args:
            models: List of models to train ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
            target_type: 'classification' or 'regression'
            random_state: Random seed for reproducibility
        """
        self.target_type = target_type
        self.random_state = random_state

        # Default models
        if models is None:
            models = ["random_forest", "gradient_boosting"]
            if HAS_XGBOOST:
                models.append("xgboost")
            if HAS_LIGHTGBM:
                models.append("lightgbm")

        self.models = models
        self.scaler = StandardScaler()
        self.trained_models: List[ModelResult] = []

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        test_size: float = 0.2,
        scale_features: bool = True,
    ) -> List[ModelResult]:
        """
        Train multiple models on the data.

        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            scale_features: Whether to scale features

        Returns:
            List of ModelResult objects
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, pd.Series):
            y = y.values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Scale features if requested
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

        results = []

        for model_name in self.models:
            print(f"\nTraining {model_name}...")

            import time

            start_time = time.time()

            # Get and train model
            model = self._get_model(model_name)
            model.fit(X_train, y_train)

            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)

            # Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names)

            # Get model parameters
            if hasattr(model, "get_params"):
                parameters = model.get_params()
            else:
                parameters = {}

            result = ModelResult(
                model=model,
                model_type=model_name,
                metrics=metrics,
                feature_importance=feature_importance,
                training_time=training_time,
                parameters=parameters,
            )

            results.append(result)
            self.trained_models.append(result)

            print(f"  Training completed in {training_time:.2f}s")
            print(
                f"  Primary metric: {list(metrics.items())[0][0]} = {list(metrics.values())[0]:.4f}"
            )

        return results

    def _get_model(self, model_name: str) -> Any:
        """Get model instance by name."""
        if self.target_type == "classification":
            if model_name == "random_forest":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=self.random_state,
                )
            elif model_name == "gradient_boosting":
                return GradientBoostingClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.random_state
                )
            elif model_name == "xgboost" and HAS_XGBOOST:
                return xgb.XGBClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.random_state
                )
            elif model_name == "lightgbm" and HAS_LIGHTGBM:
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state,
                    verbose=-1,
                )
        else:  # regression
            if model_name == "random_forest":
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=self.random_state,
                )
            elif model_name == "gradient_boosting":
                return GradientBoostingRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.random_state
                )
            elif model_name == "xgboost" and HAS_XGBOOST:
                return xgb.XGBRegressor(
                    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=self.random_state
                )
            elif model_name == "lightgbm" and HAS_LIGHTGBM:
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=self.random_state,
                    verbose=-1,
                )

        raise ValueError(f"Unknown model: {model_name}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if self.target_type == "classification":
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
        else:
            return {
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
                "r2_score": r2_score(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            }

    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[pd.Series]:
        """Extract feature importance from model."""
        importance = None

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = (
                np.abs(model.coef_).mean(axis=0)
                if len(model.coef_.shape) > 1
                else np.abs(model.coef_)
            )

        if importance is not None:
            return pd.Series(importance, index=feature_names).sort_values(ascending=False)

        return None

    def cross_validate(
        self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series], cv_folds: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation for all models.

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of cross-validation folds

        Returns:
            Dictionary with CV scores for each model
        """
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        cv_results = {}

        for model_name in self.models:
            print(f"\nCross-validating {model_name}...")

            model = self._get_model(model_name)

            # Choose scoring based on target type
            if self.target_type == "classification":
                scoring = "accuracy"
            else:
                scoring = "neg_mean_squared_error"

            scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)

            cv_results[model_name] = scores
            print(f"  CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results


class ModelSelector:
    """Select best model based on performance metrics."""

    @staticmethod
    def select_best_model(results: List[ModelResult], metric: str = "accuracy") -> ModelResult:
        """
        Select the best model based on a specific metric.

        Args:
            results: List of ModelResult objects
            metric: Metric to use for selection

        Returns:
            Best ModelResult
        """
        if not results:
            raise ValueError("No results to select from")

        # For some metrics, lower is better
        lower_is_better = metric in ["mse", "mae", "rmse"]

        best_result = None
        best_score = float("inf") if lower_is_better else float("-inf")

        for result in results:
            if metric in result.metrics:
                score = result.metrics[metric]

                if lower_is_better:
                    if score < best_score:
                        best_score = score
                        best_result = result
                else:
                    if score > best_score:
                        best_score = score
                        best_result = result

        return best_result or results[0]

    @staticmethod
    def ensemble_predictions(
        models: List[Any], X: Union[np.ndarray, pd.DataFrame], weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Create ensemble predictions from multiple models.

        Args:
            models: List of trained models
            X: Feature matrix
            weights: Optional weights for each model

        Returns:
            Ensemble predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if weights is None:
            weights = [1.0 / len(models)] * len(models)

        predictions = []
        for model, weight in zip(models, weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return np.sum(predictions, axis=0)


class ModelRegistry:
    """Simple model registry for saving and loading models."""

    def __init__(self, model_dir: str = "./models"):
        """Initialize model registry."""
        self.model_dir = model_dir
        import os

        os.makedirs(model_dir, exist_ok=True)
        self.registry: Dict[str, Dict] = {}

    def register_model(
        self, model: Any, metadata: Dict[str, Any], model_id: Optional[str] = None
    ) -> str:
        """
        Register and save a model.

        Args:
            model: Trained model
            metadata: Model metadata
            model_id: Optional model ID

        Returns:
            Model ID
        """
        if model_id is None:
            model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save model
        model_path = f"{self.model_dir}/{model_id}.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        metadata["model_id"] = model_id
        metadata["model_path"] = model_path
        metadata["registered_at"] = datetime.now().isoformat()

        metadata_path = f"{self.model_dir}/{model_id}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.registry[model_id] = metadata

        return model_id

    def load_model(self, model_id: str) -> Tuple[Any, Dict]:
        """
        Load a model and its metadata.

        Args:
            model_id: Model ID

        Returns:
            Tuple of (model, metadata)
        """
        model_path = f"{self.model_dir}/{model_id}.joblib"
        metadata_path = f"{self.model_dir}/{model_id}_metadata.json"

        model = joblib.load(model_path)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return model, metadata

    def list_models(self) -> List[Dict]:
        """List all registered models."""
        import glob
        import os

        models = []

        for metadata_file in glob.glob(f"{self.model_dir}/*_metadata.json"):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                models.append(metadata)

        return models

    def delete_model(self, model_id: str) -> None:
        """Delete a model."""
        import os

        model_path = f"{self.model_dir}/{model_id}.joblib"
        metadata_path = f"{self.model_dir}/{model_id}_metadata.json"

        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        if model_id in self.registry:
            del self.registry[model_id]

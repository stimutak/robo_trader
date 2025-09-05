"""Model selection and validation framework for RoboTrader.

This module provides automated model selection based on performance metrics
and cross-validation results.
"""

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from .model_trainer import ModelTrainer, ModelType, PredictionType

logger = structlog.get_logger(__name__)


class ModelSelector:
    """Automated model selection and validation."""

    def __init__(
        self,
        model_trainer: ModelTrainer,
        model_dir: Optional[Path] = None,
        validation_window: int = 30,  # days
        min_performance_score: float = 0.6,
    ):
        """Initialize model selector.

        Args:
            model_trainer: Model training pipeline
            model_dir: Directory containing trained models
            validation_window: Days of data for validation
            min_performance_score: Minimum acceptable performance
        """
        self.model_trainer = model_trainer
        self.model_dir = model_dir or Path("models")
        self.validation_window = validation_window
        self.min_performance_score = min_performance_score

        self.available_models: Dict[str, Dict] = {}
        self.model_performance: Dict[str, float] = {}
        self.selected_model: Optional[Dict] = None

    async def load_available_models(self) -> List[Dict]:
        """Load all available trained models.

        Returns:
            List of model information dictionaries
        """
        models = []

        # First check for the improved model
        improved_model_path = Path("trained_models/improved_model.pkl")
        if improved_model_path.exists():
            try:
                with open(improved_model_path, "rb") as f:
                    # Security: Only load trusted model files from our own system
                    model_info = pickle.load(f)  # Security: Trusted file from our system

                # Add as priority model
                models.append(model_info)
                self.available_models["improved_model"] = model_info

                logger.info(
                    "Loaded improved model with confidence filtering",
                    filename=improved_model_path.name,
                    test_score=model_info.get("metrics", {}).get("test_score"),
                    confidence_threshold=model_info.get("confidence_threshold", 0.6),
                )
            except Exception as e:
                logger.error(f"Failed to load improved model: {e}")

        # Then load other models
        for model_file in self.model_dir.glob("*.pkl"):
            try:
                with open(model_file, "rb") as f:
                    # Security: Only load trusted model files from our own system
                    model_info = pickle.load(f)  # Security: Trusted file from our system

                # Load metadata
                metadata_file = model_file.with_suffix(".json")
                if metadata_file.exists():
                    import json

                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    model_info.update(metadata)

                models.append(model_info)
                self.available_models[model_file.stem] = model_info

                logger.info(
                    "Loaded model",
                    filename=model_file.name,
                    model_type=model_info.get("model_type"),
                    test_score=model_info.get("metrics", {}).get("test_score"),
                )

            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")

        return models

    async def select_best_model(
        self,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        selection_criteria: str = "test_score",
        require_recent: bool = True,
        max_age_days: int = 7,
    ) -> Optional[Dict]:
        """Select best model based on performance criteria.

        Args:
            validation_data: Optional validation features and target
            selection_criteria: Metric to use for selection
            require_recent: Whether to require recently trained models
            max_age_days: Maximum age of model in days

        Returns:
            Best model information or None
        """
        if not self.available_models:
            await self.load_available_models()

        if not self.available_models:
            logger.warning("No models available for selection")
            return None

        # Filter by age if required
        candidates = []
        for name, model_info in self.available_models.items():
            if require_recent:
                trained_at = datetime.fromisoformat(model_info.get("trained_at", "2000-01-01"))
                if datetime.now() - trained_at > timedelta(days=max_age_days):
                    logger.debug(f"Skipping old model: {name}")
                    continue

            # Check minimum performance
            score = model_info.get("metrics", {}).get(selection_criteria, 0)
            if score < self.min_performance_score:
                logger.debug(f"Skipping low-performance model: {name}", score=score)
                continue

            candidates.append((name, model_info))

        if not candidates:
            logger.warning("No candidate models meet criteria")
            return None

        # Validate on new data if provided
        if validation_data is not None:
            features, target = validation_data

            for name, model_info in candidates:
                val_metrics = await self.model_trainer.validate_model(model_info, features, target)

                # Update performance tracking
                self.model_performance[name] = val_metrics["validation_score"]
                model_info["validation_metrics"] = val_metrics

        # Select best model
        if validation_data is not None:
            # Use validation performance
            best_name = max(self.model_performance, key=self.model_performance.get)
            best_model = self.available_models[best_name]
        else:
            # Use test performance
            best_name, best_model = max(
                candidates, key=lambda x: x[1].get("metrics", {}).get(selection_criteria, 0)
            )

        self.selected_model = best_model

        logger.info(
            "Selected best model",
            model_name=best_name,
            model_type=best_model.get("model_type"),
            score=best_model.get("metrics", {}).get(selection_criteria),
            validation_score=self.model_performance.get(best_name),
        )

        return best_model

    async def compare_models(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_types: List[ModelType] = None,
        prediction_type: PredictionType = PredictionType.REGRESSION,
    ) -> pd.DataFrame:
        """Compare performance of different model types.

        Args:
            features: Feature matrix
            target: Target variable
            model_types: List of model types to compare
            prediction_type: Type of prediction task

        Returns:
            DataFrame with comparison results
        """
        if model_types is None:
            model_types = [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]

        results = []

        for model_type in model_types:
            logger.info(f"Training {model_type.value} for comparison")

            model_info = await self.model_trainer.train_model(
                features,
                target,
                model_type,
                prediction_type,
                tune_hyperparams=False,  # Quick training for comparison
            )

            result = {
                "model_type": model_type.value,
                "train_score": model_info["metrics"].get("train_score"),
                "test_score": model_info["metrics"].get("test_score"),
                "train_time": model_info.get("train_time", 0),
            }

            if prediction_type == PredictionType.REGRESSION:
                result["test_mse"] = model_info["metrics"].get("test_mse")
                result["test_mae"] = model_info["metrics"].get("test_mae")
                result["direction_accuracy"] = model_info["metrics"].get("test_direction_accuracy")
            else:
                result["test_accuracy"] = model_info["metrics"].get("test_accuracy")
                result["test_precision"] = model_info["metrics"].get("test_precision")
                result["test_recall"] = model_info["metrics"].get("test_recall")
                result["test_f1"] = model_info["metrics"].get("test_f1")

            results.append(result)

        comparison_df = pd.DataFrame(results)

        # Add ranking
        comparison_df["rank"] = (
            comparison_df["test_score"].rank(ascending=False, method="min").astype(int)
        )

        comparison_df = comparison_df.sort_values("rank")

        logger.info(
            "Model comparison complete",
            best_model=comparison_df.iloc[0]["model_type"],
            best_score=comparison_df.iloc[0]["test_score"],
        )

        return comparison_df

    async def perform_walk_forward_selection(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        model_types: List[ModelType] = None,
        window_size: int = 252,  # Trading days in a year
        step_size: int = 21,  # Trading days in a month
        min_train_size: int = 126,  # 6 months minimum
    ) -> Dict[str, Any]:
        """Perform walk-forward model selection.

        Args:
            data: Complete dataset
            feature_columns: List of feature column names
            target_column: Name of target column
            model_types: Model types to evaluate
            window_size: Size of training window
            step_size: Step size for walk-forward
            min_train_size: Minimum training size

        Returns:
            Dictionary with walk-forward results
        """
        if model_types is None:
            model_types = [ModelType.XGBOOST]  # Default to XGBoost for speed

        results = []
        selected_models = []

        # Create walk-forward windows
        for start_idx in range(min_train_size, len(data) - step_size, step_size):
            end_idx = min(start_idx + window_size, len(data) - step_size)
            test_end_idx = min(end_idx + step_size, len(data))

            # Split data
            train_data = data.iloc[start_idx:end_idx]
            test_data = data.iloc[end_idx:test_end_idx]

            # Skip if test data is empty or too small
            if len(test_data) < 2:
                logger.warning(
                    f"Skipping window due to insufficient test data: {len(test_data)} samples"
                )
                continue

            X_train = train_data[feature_columns]
            y_train = train_data[target_column]
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            # Train and evaluate models
            window_results = []
            for model_type in model_types:
                model_info = await self.model_trainer.train_model(
                    X_train,
                    y_train,
                    model_type,
                    tune_hyperparams=False,
                    test_size=0,  # Use all data for training
                )

                # Validate on test window
                val_metrics = await self.model_trainer.validate_model(model_info, X_test, y_test)

                window_results.append(
                    {
                        "model_type": model_type.value,
                        "window_start": train_data.index[0],
                        "window_end": train_data.index[-1],
                        "test_start": test_data.index[0],
                        "test_end": test_data.index[-1],
                        "validation_score": val_metrics["validation_score"],
                        **val_metrics,
                    }
                )

            # Select best model for this window
            best_result = max(window_results, key=lambda x: x["validation_score"])

            results.extend(window_results)
            selected_models.append(best_result)

            logger.info(
                "Walk-forward window complete",
                window=f"{best_result['window_start']} to {best_result['window_end']}",
                best_model=best_result["model_type"],
                score=best_result["validation_score"],
            )

        # Analyze results
        results_df = pd.DataFrame(results)
        selected_df = pd.DataFrame(selected_models)

        # Model selection frequency
        model_selection_freq = selected_df["model_type"].value_counts()

        # Average performance by model type
        avg_performance = results_df.groupby("model_type")["validation_score"].agg(
            ["mean", "std", "min", "max"]
        )

        # Overall best model
        overall_best = avg_performance["mean"].idxmax()

        logger.info(
            "Walk-forward selection complete",
            n_windows=len(selected_models),
            overall_best=overall_best,
            avg_score=avg_performance.loc[overall_best, "mean"],
        )

        return {
            "results": results_df,
            "selected_models": selected_df,
            "model_selection_frequency": model_selection_freq.to_dict(),
            "average_performance": avg_performance,
            "overall_best_model": overall_best,
            "best_model_avg_score": avg_performance.loc[overall_best, "mean"],
        }

    def get_model_confidence(self, model_info: Dict) -> float:
        """Calculate confidence score for a model.

        Args:
            model_info: Model information dictionary

        Returns:
            Confidence score between 0 and 1
        """
        confidence_factors = []

        # Test score
        test_score = model_info.get("metrics", {}).get("test_score", 0)
        confidence_factors.append(min(test_score, 1.0))

        # Validation score if available
        if "validation_metrics" in model_info:
            val_score = model_info["validation_metrics"].get("validation_score", 0)
            confidence_factors.append(min(val_score, 1.0))

        # Training samples (more is better, cap at 10000)
        n_samples = model_info.get("n_train_samples", 0)
        sample_confidence = min(n_samples / 10000, 1.0)
        confidence_factors.append(sample_confidence)

        # Model age (newer is better)
        trained_at = datetime.fromisoformat(model_info.get("trained_at", "2000-01-01"))
        age_days = (datetime.now() - trained_at).days
        age_confidence = max(1.0 - (age_days / 30), 0)  # Decay over 30 days
        confidence_factors.append(age_confidence)

        # Direction accuracy for regression models
        if model_info.get("prediction_type") == "regression":
            dir_acc = model_info.get("metrics", {}).get("test_direction_accuracy", 0.5)
            confidence_factors.append(dir_acc)

        # Calculate weighted average
        confidence = np.mean(confidence_factors)

        return confidence

    async def auto_retrain(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        current_model: Optional[Dict] = None,
        performance_threshold: float = 0.7,
        force_retrain: bool = False,
    ) -> Tuple[bool, Optional[Dict]]:
        """Automatically retrain model if performance degrades.

        Args:
            features: Current feature data
            target: Current target data
            current_model: Currently deployed model
            performance_threshold: Threshold for retraining
            force_retrain: Force retraining regardless of performance

        Returns:
            Tuple of (retrained, new_model_info)
        """
        should_retrain = force_retrain

        if current_model and not force_retrain:
            # Validate current model
            val_metrics = await self.model_trainer.validate_model(current_model, features, target)

            current_score = val_metrics["validation_score"]

            logger.info(
                "Current model validation", score=current_score, threshold=performance_threshold
            )

            if current_score < performance_threshold:
                should_retrain = True
                logger.warning(
                    "Model performance below threshold, retraining",
                    current_score=current_score,
                    threshold=performance_threshold,
                )

        if should_retrain:
            # Retrain model
            model_type = ModelType(
                current_model["model_type"] if current_model else ModelType.XGBOOST.value
            )

            new_model = await self.model_trainer.train_model(
                features, target, model_type, tune_hyperparams=True
            )

            logger.info(
                "Model retrained",
                new_score=new_model["metrics"]["test_score"],
                improvement=new_model["metrics"]["test_score"]
                - (current_score if current_model else 0),
            )

            return True, new_model

        return False, None

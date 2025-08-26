"""ML Model Training Pipeline for RoboTrader.

This module implements automated training for Random Forest, XGBoost, 
and Neural Network models with hyperparameter tuning and validation.
"""

import asyncio
import json
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    accuracy_score, 
    precision_score, 
    recall_score,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import structlog

from ..config import Config
from ..features.feature_pipeline import FeaturePipeline

logger = structlog.get_logger(__name__)


class ModelType(Enum):
    """Available model types for training."""
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


class PredictionType(Enum):
    """Type of prediction task."""
    REGRESSION = "regression"  # Predict returns
    CLASSIFICATION = "classification"  # Predict direction (up/down)
    MULTI_CLASS = "multi_class"  # Predict regime (trending/ranging/volatile)


class ModelTrainer:
    """Automated ML model training pipeline."""
    
    def __init__(
        self,
        config: Config,
        model_dir: Optional[Path] = None,
        feature_pipeline: Optional[FeaturePipeline] = None
    ):
        """Initialize model trainer.
        
        Args:
            config: Trading configuration
            model_dir: Directory to save trained models
            feature_pipeline: Feature engineering pipeline
        """
        self.config = config
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        self.feature_pipeline = feature_pipeline or FeaturePipeline(config)
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.performance_history: List[Dict] = []
        
    async def train_model(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_type: ModelType,
        prediction_type: PredictionType = PredictionType.REGRESSION,
        hyperparams: Optional[Dict] = None,
        tune_hyperparams: bool = True,
        n_splits: int = 5,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """Train a single model with hyperparameter tuning.
        
        Args:
            features: Feature matrix
            target: Target variable
            model_type: Type of model to train
            prediction_type: Type of prediction task
            hyperparams: Custom hyperparameters (optional)
            tune_hyperparams: Whether to perform hyperparameter tuning
            n_splits: Number of time series splits for CV
            test_size: Fraction of data for test set
            
        Returns:
            Dictionary with trained model and metrics
        """
        logger.info(
            "Training model",
            model_type=model_type.value,
            prediction_type=prediction_type.value,
            n_samples=len(features),
            n_features=features.shape[1]
        )
        
        # Split data
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get model and parameters
        model, param_grid = self._get_model_and_params(
            model_type, 
            prediction_type,
            hyperparams
        )
        
        # Hyperparameter tuning
        if tune_hyperparams and param_grid:
            logger.info("Starting hyperparameter tuning")
            best_model = await self._tune_hyperparameters(
                model,
                param_grid,
                X_train_scaled,
                y_train,
                n_splits
            )
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
        
        metrics = self._calculate_metrics(
            y_train, train_pred,
            y_test, test_pred,
            prediction_type
        )
        
        # Feature importance
        feature_importance = self._get_feature_importance(
            best_model, 
            features.columns.tolist(),
            model_type
        )
        
        # Save model
        model_info = {
            "model": best_model,
            "scaler": self.scaler,
            "model_type": model_type.value,
            "prediction_type": prediction_type.value,
            "features": features.columns.tolist(),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "trained_at": datetime.now().isoformat(),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test)
        }
        
        model_filename = await self._save_model(model_info)
        model_info["filename"] = model_filename
        
        # Track performance
        self.performance_history.append({
            "timestamp": datetime.now(),
            "model_type": model_type.value,
            "metrics": metrics
        })
        
        logger.info(
            "Model training complete",
            model_type=model_type.value,
            train_score=metrics.get("train_score", 0),
            test_score=metrics.get("test_score", 0),
            filename=model_filename
        )
        
        return model_info
    
    def _get_model_and_params(
        self,
        model_type: ModelType,
        prediction_type: PredictionType,
        custom_params: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """Get model instance and hyperparameter grid.
        
        Args:
            model_type: Type of model
            prediction_type: Type of prediction task
            custom_params: Custom hyperparameters
            
        Returns:
            Model instance and parameter grid for tuning
        """
        if model_type == ModelType.RANDOM_FOREST:
            if prediction_type == PredictionType.REGRESSION:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    **(custom_params or {})
                )
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            else:
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    **(custom_params or {})
                )
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": ["balanced", None]
                }
                
        elif model_type == ModelType.XGBOOST:
            if prediction_type == PredictionType.REGRESSION:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    **(custom_params or {})
                )
            else:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                    **(custom_params or {})
                )
            
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9]
            }
            
        elif model_type == ModelType.LIGHTGBM:
            if prediction_type == PredictionType.REGRESSION:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **(custom_params or {})
                )
            else:
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    **(custom_params or {})
                )
            
            param_grid = {
                "n_estimators": [100, 200, 300],
                "num_leaves": [31, 50, 100],
                "learning_rate": [0.01, 0.1, 0.3],
                "feature_fraction": [0.7, 0.8, 0.9],
                "bagging_fraction": [0.7, 0.8, 0.9]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model, param_grid if not custom_params else {}
    
    async def _tune_hyperparameters(
        self,
        model: Any,
        param_grid: Dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_splits: int
    ) -> Any:
        """Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid for search
            X_train: Training features
            y_train: Training target
            n_splits: Number of time series splits
            
        Returns:
            Best model after tuning
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring="neg_mean_squared_error" if hasattr(model, "predict") else "accuracy",
            n_jobs=-1,
            verbose=0
        )
        
        # Run grid search in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, grid_search.fit, X_train, y_train)
        
        logger.info(
            "Hyperparameter tuning complete",
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_
        )
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(
        self,
        y_train: np.ndarray,
        train_pred: np.ndarray,
        y_test: np.ndarray,
        test_pred: np.ndarray,
        prediction_type: PredictionType
    ) -> Dict[str, float]:
        """Calculate model performance metrics.
        
        Args:
            y_train: Training target
            train_pred: Training predictions
            y_test: Test target
            test_pred: Test predictions
            prediction_type: Type of prediction task
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if prediction_type == PredictionType.REGRESSION:
            # Regression metrics
            metrics["train_mse"] = mean_squared_error(y_train, train_pred)
            metrics["test_mse"] = mean_squared_error(y_test, test_pred)
            metrics["train_mae"] = mean_absolute_error(y_train, train_pred)
            metrics["test_mae"] = mean_absolute_error(y_test, test_pred)
            metrics["train_score"] = 1 - metrics["train_mse"]
            metrics["test_score"] = 1 - metrics["test_mse"]
            
            # Direction accuracy (for returns)
            train_dir_acc = np.mean(np.sign(train_pred) == np.sign(y_train))
            test_dir_acc = np.mean(np.sign(test_pred) == np.sign(y_test))
            metrics["train_direction_accuracy"] = train_dir_acc
            metrics["test_direction_accuracy"] = test_dir_acc
            
        else:
            # Classification metrics
            metrics["train_accuracy"] = accuracy_score(y_train, train_pred)
            metrics["test_accuracy"] = accuracy_score(y_test, test_pred)
            metrics["train_score"] = metrics["train_accuracy"]
            metrics["test_score"] = metrics["test_accuracy"]
            
            # Additional classification metrics
            avg_method = "binary" if prediction_type == PredictionType.CLASSIFICATION else "macro"
            metrics["test_precision"] = precision_score(y_test, test_pred, average=avg_method, zero_division=0)
            metrics["test_recall"] = recall_score(y_test, test_pred, average=avg_method, zero_division=0)
            metrics["test_f1"] = f1_score(y_test, test_pred, average=avg_method, zero_division=0)
        
        return metrics
    
    def _get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        model_type: ModelType
    ) -> Dict[str, float]:
        """Extract feature importance from trained model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            model_type: Type of model
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_dict = {}
        
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
        
        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    async def _save_model(self, model_info: Dict) -> str:
        """Save trained model to disk.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Filename of saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_info['model_type']}_{timestamp}.pkl"
        filepath = self.model_dir / filename
        
        # Save model
        with open(filepath, "wb") as f:
            pickle.dump(model_info, f)
        
        # Save metadata
        metadata_file = filepath.with_suffix(".json")
        metadata = {
            k: v for k, v in model_info.items()
            if k not in ["model", "scaler"]
        }
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved to {filepath}")
        return filename
    
    async def train_ensemble(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        model_types: List[ModelType] = None,
        prediction_type: PredictionType = PredictionType.REGRESSION,
        **kwargs
    ) -> Dict[str, Any]:
        """Train ensemble of multiple models.
        
        Args:
            features: Feature matrix
            target: Target variable
            model_types: List of model types to include
            prediction_type: Type of prediction task
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary with ensemble model information
        """
        if model_types is None:
            model_types = [
                ModelType.RANDOM_FOREST,
                ModelType.XGBOOST,
                ModelType.LIGHTGBM
            ]
        
        logger.info(
            "Training ensemble model",
            n_models=len(model_types),
            prediction_type=prediction_type.value
        )
        
        ensemble_models = []
        
        # Train individual models
        for model_type in model_types:
            model_info = await self.train_model(
                features,
                target,
                model_type,
                prediction_type,
                **kwargs
            )
            ensemble_models.append(model_info)
        
        # Create ensemble predictions (simple averaging for now)
        split_idx = int(len(features) * (1 - kwargs.get("test_size", 0.2)))
        X_test = features[split_idx:]
        y_test = target[split_idx:]
        
        X_test_scaled = self.scaler.transform(X_test)
        
        # Collect predictions
        predictions = []
        for model_info in ensemble_models:
            model = model_info["model"]
            pred = model.predict(X_test_scaled)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Calculate ensemble metrics
        if prediction_type == PredictionType.REGRESSION:
            ensemble_mse = mean_squared_error(y_test, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            ensemble_score = 1 - ensemble_mse
            direction_acc = np.mean(np.sign(ensemble_pred) == np.sign(y_test))
            
            ensemble_metrics = {
                "test_mse": ensemble_mse,
                "test_mae": ensemble_mae,
                "test_score": ensemble_score,
                "test_direction_accuracy": direction_acc
            }
        else:
            # For classification, use voting
            ensemble_pred = np.round(ensemble_pred).astype(int)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
            
            ensemble_metrics = {
                "test_accuracy": ensemble_acc,
                "test_score": ensemble_acc
            }
        
        # Compare with individual models
        comparison = {
            "ensemble": ensemble_metrics["test_score"]
        }
        
        for model_info in ensemble_models:
            model_type = model_info["model_type"]
            score = model_info["metrics"]["test_score"]
            comparison[model_type] = score
        
        logger.info(
            "Ensemble training complete",
            ensemble_score=ensemble_metrics["test_score"],
            comparison=comparison
        )
        
        return {
            "ensemble_models": ensemble_models,
            "ensemble_metrics": ensemble_metrics,
            "comparison": comparison,
            "best_individual": max(
                ensemble_models,
                key=lambda x: x["metrics"]["test_score"]
            )["model_type"]
        }
    
    async def validate_model(
        self,
        model_info: Dict,
        validation_features: pd.DataFrame,
        validation_target: pd.Series
    ) -> Dict[str, float]:
        """Validate model on new data.
        
        Args:
            model_info: Trained model information
            validation_features: Validation features
            validation_target: Validation target
            
        Returns:
            Validation metrics
        """
        model = model_info["model"]
        scaler = model_info["scaler"]
        prediction_type = PredictionType(model_info["prediction_type"])
        
        # Scale features
        X_val_scaled = scaler.transform(validation_features)
        
        # Make predictions
        val_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        if prediction_type == PredictionType.REGRESSION:
            val_mse = mean_squared_error(validation_target, val_pred)
            val_mae = mean_absolute_error(validation_target, val_pred)
            val_dir_acc = np.mean(
                np.sign(val_pred) == np.sign(validation_target)
            )
            
            metrics = {
                "validation_mse": val_mse,
                "validation_mae": val_mae,
                "validation_direction_accuracy": val_dir_acc,
                "validation_score": 1 - val_mse
            }
        else:
            val_acc = accuracy_score(validation_target, val_pred)
            
            metrics = {
                "validation_accuracy": val_acc,
                "validation_score": val_acc
            }
        
        logger.info(
            "Model validation complete",
            model_type=model_info["model_type"],
            validation_score=metrics["validation_score"]
        )
        
        return metrics
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get summary of model performance history.
        
        Returns:
            DataFrame with performance history
        """
        if not self.performance_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.performance_history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Extract key metrics
        for record in self.performance_history:
            metrics = record["metrics"]
            for key, value in metrics.items():
                if key not in df.columns:
                    df[key] = None
                df.loc[df["timestamp"] == record["timestamp"], key] = value
        
        return df.set_index("timestamp")
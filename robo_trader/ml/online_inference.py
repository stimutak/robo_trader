"""
Online model inference pipeline for real-time predictions.
Part of Phase 3 S5 implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for model prediction results."""

    symbol: str
    timestamp: datetime
    prediction: float
    confidence: float
    features_used: Dict[str, float]
    model_version: str
    latency_ms: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "prediction": self.prediction,
            "confidence": self.confidence,
            "features_count": len(self.features_used),
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
        }


class OnlineModelInference:
    """
    Real-time model inference engine with low latency.
    Manages model loading, caching, and prediction serving.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        prediction_threshold: float = 0.6,
        use_ensemble: bool = False,
    ):
        """
        Initialize online inference engine.

        Args:
            model_path: Path to saved model
            feature_names: List of feature names in order
            prediction_threshold: Confidence threshold for signals
            use_ensemble: Use ensemble of models
        """
        self.model_path = model_path
        self.feature_names = feature_names or self._get_default_features()
        self.prediction_threshold = prediction_threshold
        self.use_ensemble = use_ensemble

        # Model cache
        self.models: Dict[str, Any] = {}
        self.model_version = "v1.0"

        # Performance tracking
        self.prediction_history: List[PredictionResult] = []
        self.max_history = 1000

        # Load models
        if model_path:
            self.load_model(model_path)

    def _get_default_features(self) -> List[str]:
        """Get default feature list."""
        return [
            "returns",
            "log_returns",
            "sma_10",
            "sma_20",
            "sma_50",
            "price_to_sma_10",
            "price_to_sma_20",
            "ema_10",
            "ema_20",
            "price_to_ema_10",
            "price_to_ema_20",
            "rsi",
            "macd",
            "macd_signal",
            "macd_histogram",
            "bb_position",
            "bb_width",
            "volume_ratio",
            "volatility",
            "volatility_ratio",
            "momentum",
            "price_momentum",
            "price_position",
        ]

    def load_model(self, model_path: str, model_name: str = "primary"):
        """
        Load a model for inference.

        Args:
            model_path: Path to model file
            model_name: Name for model identification
        """
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            logger.info(f"Loaded model {model_name} from {model_path}")

            # Update version based on file
            import os

            self.model_version = f"v{os.path.getmtime(model_path):.0f}"

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create a dummy model for testing
            self._create_dummy_model(model_name)

    def _create_dummy_model(self, model_name: str):
        """Create a dummy model for testing."""

        class DummyModel:
            def predict(self, X):
                """Random predictions for testing."""
                return np.random.randn(X.shape[0]) * 0.01

            def predict_proba(self, X):
                """Random probabilities for testing."""
                probs = np.random.random((X.shape[0], 2))
                return probs / probs.sum(axis=1, keepdims=True)

        self.models[model_name] = DummyModel()
        logger.warning(f"Created dummy model for {model_name}")

    def predict(
        self, features: Dict[str, float], symbol: str, model_name: str = "primary"
    ) -> Optional[PredictionResult]:
        """
        Make a real-time prediction.

        Args:
            features: Feature dictionary
            symbol: Stock symbol
            model_name: Model to use

        Returns:
            PredictionResult or None if prediction fails
        """
        start_time = time.time()

        # Check if model exists
        if model_name not in self.models:
            if not self.models:  # No models loaded
                self._create_dummy_model("primary")
            model_name = list(self.models.keys())[0]

        try:
            # Prepare feature vector
            feature_vector = self._prepare_features(features)

            if feature_vector is None:
                return None

            # Make prediction
            model = self.models[model_name]

            # Get prediction and confidence
            if hasattr(model, "predict_proba"):
                # Classification model
                proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
                prediction = proba[1] - proba[0]  # Positive class probability
                confidence = max(proba)
            else:
                # Regression model
                prediction = model.predict(feature_vector.reshape(1, -1))[0]
                # Estimate confidence based on prediction magnitude
                confidence = min(abs(prediction) / 0.05, 1.0)  # Normalize to 0-1

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Create result
            result = PredictionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                prediction=float(prediction),
                confidence=float(confidence),
                features_used=features,
                model_version=self.model_version,
                latency_ms=latency_ms,
            )

            # Store in history
            self.prediction_history.append(result)
            if len(self.prediction_history) > self.max_history:
                self.prediction_history.pop(0)

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

    def _prepare_features(self, features: Dict[str, float]) -> Optional[np.ndarray]:
        """Prepare feature vector from dictionary."""
        try:
            # Extract features in correct order
            vector = []
            missing_features = []

            for name in self.feature_names:
                if name in features:
                    value = features[name]
                    # Handle NaN and inf
                    if np.isnan(value) or np.isinf(value):
                        value = 0
                    vector.append(value)
                else:
                    missing_features.append(name)
                    vector.append(0)  # Default value

            if missing_features and len(missing_features) > len(self.feature_names) * 0.5:
                logger.warning(f"Too many missing features: {missing_features[:5]}...")
                return None

            return np.array(vector, dtype=np.float32)

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return None

    def predict_ensemble(
        self, features: Dict[str, float], symbol: str
    ) -> Optional[PredictionResult]:
        """
        Make ensemble prediction using all loaded models.

        Args:
            features: Feature dictionary
            symbol: Stock symbol

        Returns:
            Ensemble PredictionResult
        """
        if not self.models:
            return None

        predictions = []
        confidences = []

        # Get predictions from all models
        for model_name in self.models:
            result = self.predict(features, symbol, model_name)
            if result:
                predictions.append(result.prediction)
                confidences.append(result.confidence)

        if not predictions:
            return None

        # Ensemble aggregation
        ensemble_prediction = np.mean(predictions)
        ensemble_confidence = np.mean(confidences)

        # Weighted average by confidence
        if sum(confidences) > 0:
            weighted_prediction = np.average(predictions, weights=confidences)
        else:
            weighted_prediction = ensemble_prediction

        return PredictionResult(
            symbol=symbol,
            timestamp=datetime.now(),
            prediction=float(weighted_prediction),
            confidence=float(ensemble_confidence),
            features_used=features,
            model_version=f"ensemble_{self.model_version}",
            latency_ms=0,  # Already calculated in individual predictions
        )

    def get_trading_signal(self, prediction_result: PredictionResult) -> str:
        """
        Convert prediction to trading signal.

        Args:
            prediction_result: Model prediction result

        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        if not prediction_result:
            return "HOLD"

        # Check confidence threshold
        if prediction_result.confidence < self.prediction_threshold:
            return "HOLD"

        # Generate signal based on prediction
        if prediction_result.prediction > 0.01:  # Buy threshold
            return "BUY"
        elif prediction_result.prediction < -0.01:  # Sell threshold
            return "SELL"
        else:
            return "HOLD"

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get inference performance metrics."""
        if not self.prediction_history:
            return {}

        latencies = [p.latency_ms for p in self.prediction_history]
        confidences = [p.confidence for p in self.prediction_history]

        return {
            "total_predictions": len(self.prediction_history),
            "avg_latency_ms": np.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "max_latency_ms": max(latencies),
            "avg_confidence": np.mean(confidences),
            "model_version": self.model_version,
            "models_loaded": list(self.models.keys()),
        }

    async def predict_async(
        self, features: Dict[str, float], symbol: str, model_name: str = "primary"
    ) -> Optional[PredictionResult]:
        """Async wrapper for predictions."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, features, symbol, model_name)


class ModelUpdateManager:
    """
    Manages model updates and A/B testing for online inference.
    """

    def __init__(self, inference_engine: OnlineModelInference):
        """
        Initialize model update manager.

        Args:
            inference_engine: Online inference engine
        """
        self.inference_engine = inference_engine
        self.model_versions: Dict[str, str] = {}
        self.ab_test_allocation: Dict[str, str] = {}  # symbol -> model mapping
        self.performance_tracking: Dict[str, List[float]] = {}

    def deploy_model(self, model_path: str, model_name: str, allocation_pct: float = 0.1):
        """
        Deploy a new model with gradual rollout.

        Args:
            model_path: Path to new model
            model_name: Name for the model
            allocation_pct: Percentage of traffic for new model
        """
        # Load new model
        self.inference_engine.load_model(model_path, model_name)

        # Set up A/B test allocation
        import random

        for symbol in self.ab_test_allocation:
            if random.random() < allocation_pct:
                self.ab_test_allocation[symbol] = model_name

        logger.info(f"Deployed model {model_name} with {allocation_pct*100}% allocation")

    def get_model_for_symbol(self, symbol: str) -> str:
        """Get assigned model for a symbol (A/B testing)."""
        if symbol not in self.ab_test_allocation:
            # Assign model based on current allocation
            if len(self.inference_engine.models) > 1:
                # Simple round-robin for new symbols
                models = list(self.inference_engine.models.keys())
                model_idx = hash(symbol) % len(models)
                self.ab_test_allocation[symbol] = models[model_idx]
            else:
                self.ab_test_allocation[symbol] = "primary"

        return self.ab_test_allocation[symbol]

    def track_performance(
        self, model_name: str, prediction_result: PredictionResult, actual_return: float
    ):
        """
        Track model performance for comparison.

        Args:
            model_name: Model identifier
            prediction_result: Prediction made
            actual_return: Actual market return
        """
        if model_name not in self.performance_tracking:
            self.performance_tracking[model_name] = []

        # Calculate prediction accuracy
        prediction_error = abs(prediction_result.prediction - actual_return)
        self.performance_tracking[model_name].append(prediction_error)

        # Limit history
        if len(self.performance_tracking[model_name]) > 1000:
            self.performance_tracking[model_name].pop(0)

    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """Compare performance of different models."""
        comparison = {}

        for model_name, errors in self.performance_tracking.items():
            if errors:
                comparison[model_name] = {
                    "mean_error": np.mean(errors),
                    "std_error": np.std(errors),
                    "max_error": max(errors),
                    "sample_size": len(errors),
                }

        return comparison

    def promote_best_model(self):
        """Promote best performing model to primary."""
        comparison = self.compare_models()

        if not comparison:
            return

        # Find model with lowest mean error
        best_model = min(comparison.items(), key=lambda x: x[1]["mean_error"])[0]

        if best_model != "primary" and best_model in self.inference_engine.models:
            # Swap models
            self.inference_engine.models["primary"] = self.inference_engine.models[best_model]
            logger.info(f"Promoted {best_model} to primary model")

            # Update allocations
            for symbol in self.ab_test_allocation:
                self.ab_test_allocation[symbol] = "primary"

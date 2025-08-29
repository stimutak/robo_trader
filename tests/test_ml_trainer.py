"""Tests for ML model trainer."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from robo_trader.config import Config
from robo_trader.ml.model_trainer import ModelTrainer, ModelType, PredictionType


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Create synthetic features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    
    # Create synthetic target (regression)
    target_regression = pd.Series(
        features.iloc[:, :5].mean(axis=1) + 0.1 * np.random.randn(n_samples),
        name="target"
    )
    
    # Create synthetic target (classification)
    target_classification = pd.Series(
        (target_regression > target_regression.median()).astype(int),
        name="target_class"
    )
    
    return features, target_regression, target_classification


@pytest.fixture
def config():
    """Create test configuration."""
    return Config()


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for models."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestModelTrainer:
    """Test cases for ModelTrainer."""
    
    def test_initialization(self, config, temp_model_dir):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        assert trainer.config == config
        assert trainer.model_dir == temp_model_dir
        assert temp_model_dir.exists()
    
    @pytest.mark.asyncio
    async def test_train_random_forest_regression(self, config, temp_model_dir, sample_data):
        """Test Random Forest regression training."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        model_info = await trainer.train_model(
            features,
            target_regression,
            ModelType.RANDOM_FOREST,
            PredictionType.REGRESSION,
            tune_hyperparams=False  # Disable for speed
        )
        
        assert model_info is not None
        assert model_info["model_type"] == "random_forest"
        assert model_info["prediction_type"] == "regression"
        assert "metrics" in model_info
        assert "feature_importance" in model_info
        assert len(model_info["features"]) == features.shape[1]
        
        # Check that model files were saved
        assert len(list(temp_model_dir.glob("*.pkl"))) >= 1
        assert len(list(temp_model_dir.glob("*.json"))) >= 1
    
    @pytest.mark.asyncio
    async def test_train_xgboost_classification(self, config, temp_model_dir, sample_data):
        """Test XGBoost classification training."""
        features, _, target_classification = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        model_info = await trainer.train_model(
            features,
            target_classification,
            ModelType.XGBOOST,
            PredictionType.CLASSIFICATION,
            tune_hyperparams=False
        )
        
        assert model_info is not None
        assert model_info["model_type"] == "xgboost"
        assert model_info["prediction_type"] == "classification"
        assert "test_accuracy" in model_info["metrics"]
        assert "test_precision" in model_info["metrics"]
        assert "test_recall" in model_info["metrics"]
        assert "test_f1" in model_info["metrics"]
    
    @pytest.mark.asyncio
    async def test_train_lightgbm(self, config, temp_model_dir, sample_data):
        """Test LightGBM training."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        model_info = await trainer.train_model(
            features,
            target_regression,
            ModelType.LIGHTGBM,
            PredictionType.REGRESSION,
            tune_hyperparams=False
        )
        
        assert model_info is not None
        assert model_info["model_type"] == "lightgbm"
        assert "test_mse" in model_info["metrics"]
        assert "test_mae" in model_info["metrics"]
        assert "test_direction_accuracy" in model_info["metrics"]
    
    @pytest.mark.asyncio
    async def test_train_ensemble(self, config, temp_model_dir, sample_data):
        """Test ensemble training."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        # Use smaller subset for speed
        features_small = features.iloc[:200]
        target_small = target_regression.iloc[:200]
        
        ensemble_info = await trainer.train_ensemble(
            features_small,
            target_small,
            model_types=[ModelType.RANDOM_FOREST, ModelType.XGBOOST],
            prediction_type=PredictionType.REGRESSION
        )
        
        assert ensemble_info is not None
        assert "ensemble_models" in ensemble_info
        assert "ensemble_metrics" in ensemble_info
        assert "comparison" in ensemble_info
        assert len(ensemble_info["ensemble_models"]) == 2
    
    @pytest.mark.asyncio
    async def test_validate_model(self, config, temp_model_dir, sample_data):
        """Test model validation."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        # Train a model first
        model_info = await trainer.train_model(
            features.iloc[:500],
            target_regression.iloc[:500],
            ModelType.RANDOM_FOREST,
            PredictionType.REGRESSION,
            tune_hyperparams=False
        )
        
        # Validate on remaining data
        validation_metrics = await trainer.validate_model(
            model_info,
            features.iloc[500:],
            target_regression.iloc[500:]
        )
        
        assert validation_metrics is not None
        assert "validation_score" in validation_metrics
        assert "validation_mse" in validation_metrics
        assert "validation_mae" in validation_metrics
    
    def test_get_model_and_params(self, config, temp_model_dir):
        """Test model and parameter retrieval."""
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        # Test Random Forest
        model, params = trainer._get_model_and_params(
            ModelType.RANDOM_FOREST,
            PredictionType.REGRESSION
        )
        
        assert model is not None
        assert isinstance(params, dict)
        assert "n_estimators" in params
        
        # Test XGBoost
        model, params = trainer._get_model_and_params(
            ModelType.XGBOOST,
            PredictionType.CLASSIFICATION
        )
        
        assert model is not None
        assert "max_depth" in params
        
        # Test LightGBM
        model, params = trainer._get_model_and_params(
            ModelType.LIGHTGBM,
            PredictionType.REGRESSION
        )
        
        assert model is not None
        assert "num_leaves" in params
    
    def test_feature_importance_extraction(self, config, temp_model_dir, sample_data):
        """Test feature importance extraction."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(features, target_regression)
        
        importance = trainer._get_feature_importance(
            model,
            features.columns.tolist(),
            ModelType.RANDOM_FOREST
        )
        
        assert isinstance(importance, dict)
        assert len(importance) == len(features.columns)
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_performance_summary(self, config, temp_model_dir):
        """Test performance summary generation."""
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        # Add some dummy performance history
        trainer.performance_history = [
            {
                "timestamp": pd.Timestamp.now(),
                "model_type": "random_forest",
                "metrics": {"test_score": 0.8, "train_score": 0.85}
            },
            {
                "timestamp": pd.Timestamp.now(),
                "model_type": "xgboost",
                "metrics": {"test_score": 0.82, "train_score": 0.87}
            }
        ]
        
        summary = trainer.get_performance_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2
        assert "model_type" in summary.columns
        assert "test_score" in summary.columns
    
    def test_invalid_model_type(self, config, temp_model_dir):
        """Test handling of invalid model type."""
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._get_model_and_params(
                "invalid_model",  # Invalid model type
                PredictionType.REGRESSION
            )
    
    @pytest.mark.asyncio
    async def test_empty_data(self, config, temp_model_dir):
        """Test handling of empty data."""
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        empty_features = pd.DataFrame()
        empty_target = pd.Series(dtype=float)
        
        with pytest.raises((ValueError, IndexError)):
            await trainer.train_model(
                empty_features,
                empty_target,
                ModelType.RANDOM_FOREST,
                PredictionType.REGRESSION
            )
    
    @pytest.mark.asyncio
    async def test_hyperparameter_tuning(self, config, temp_model_dir, sample_data):
        """Test hyperparameter tuning."""
        features, target_regression, _ = sample_data
        trainer = ModelTrainer(config, model_dir=temp_model_dir)
        
        # Use smaller dataset for speed
        features_small = features.iloc[:100]
        target_small = target_regression.iloc[:100]
        
        model_info = await trainer.train_model(
            features_small,
            target_small,
            ModelType.RANDOM_FOREST,
            PredictionType.REGRESSION,
            tune_hyperparams=True,
            n_splits=3  # Fewer splits for speed
        )
        
        assert model_info is not None
        # Should complete without errors
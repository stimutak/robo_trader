"""Train ML models for the RoboTrader system.

This script trains and saves ML models using historical market data.
"""

import asyncio
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project to path
import sys

sys.path.append(".")

from robo_trader.config import Config
from robo_trader.features.feature_pipeline import FeaturePipeline
from robo_trader.ml.model_trainer import ModelTrainer, ModelType, PredictionType


async def fetch_training_data(symbols: list, period: str = "2y") -> dict:
    """Fetch historical data for training.

    Args:
        symbols: List of symbols to fetch
        period: Time period for data (1y, 2y, 5y, etc.)

    Returns:
        Dictionary of DataFrames by symbol
    """
    data = {}

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if not df.empty:
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]

            # Add returns
            df["returns"] = df["close"].pct_change()

            # Add target (next day return)
            df["target"] = df["returns"].shift(-1)

            # Add direction (1 for up, -1 for down)
            df["direction"] = np.where(df["target"] > 0, 1, -1)

            data[symbol] = df
            logger.info(f"  Fetched {len(df)} days of data for {symbol}")
        else:
            logger.warning(f"  No data available for {symbol}")

    return data


async def prepare_training_data(data: dict, config: Config) -> tuple:
    """Prepare features and targets for training.

    Args:
        data: Dictionary of DataFrames by symbol
        config: Configuration object

    Returns:
        Tuple of (features_df, targets_df, feature_columns)
    """
    feature_pipeline = FeaturePipeline(config)

    all_features = []
    all_targets = []

    for symbol, df in data.items():
        if len(df) < 100:  # Need minimum data
            continue

        logger.info(f"Calculating features for {symbol}...")

        # Calculate features
        features = await feature_pipeline.calculate_features(
            symbol=symbol, price_data=df, store_features=False
        )

        if features is not None and not features.empty:
            # The feature pipeline returns aggregate features (1 row)
            # We need to calculate features for each day in a sliding window

            # For now, use the aggregate features with the last day's target
            if len(features) == 1:
                # Use the single row of features with the last target
                features_row = features.iloc[0].to_dict()
                # Remove non-numeric columns
                features_row = {
                    k: v for k, v in features_row.items() if k not in ["symbol", "timestamp"]
                }

                # Create a DataFrame with the features
                features_df_single = pd.DataFrame([features_row])
                features_df_single["symbol"] = symbol

                # Use the last day's direction as target
                target_single = df["direction"].iloc[-1]

                all_features.append(features_df_single)
                all_targets.append(pd.Series([target_single]))
            else:
                # If we have multiple rows, use them as is
                features = features.iloc[:-1]  # Remove last row (no target)
                targets = df["direction"].iloc[len(df) - len(features) :]

                # Add symbol column
                features["symbol"] = symbol

                all_features.append(features)
                all_targets.append(targets)

    if all_features:
        # Combine all data
        features_df = pd.concat(all_features, ignore_index=True)
        targets_df = pd.concat(all_targets, ignore_index=True)

        # Remove any remaining NaN values
        mask = ~(features_df.isna().any(axis=1) | targets_df.isna())
        features_df = features_df[mask]
        targets_df = targets_df[mask]

        # Get feature columns (exclude symbol)
        feature_columns = [col for col in features_df.columns if col != "symbol"]

        logger.info(f"Prepared {len(features_df)} samples with {len(feature_columns)} features")

        return features_df, targets_df, feature_columns

    return None, None, []


async def train_models(
    features_df: pd.DataFrame, targets_df: pd.Series, feature_columns: list, config: Config
):
    """Train ML models.

    Args:
        features_df: Feature DataFrame
        targets_df: Target Series
        feature_columns: List of feature column names
        config: Configuration object
    """
    # Initialize model trainer
    model_trainer = ModelTrainer(config=config, model_dir=Path("trained_models"))

    # Prepare data for training
    X = features_df[feature_columns].values
    y = targets_df.values

    # Train different model types
    model_types = [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.LIGHTGBM]

    trained_models = {}

    for model_type in model_types:
        logger.info(f"\nTraining {model_type.value} model...")

        try:
            # Train model
            model, metrics = await model_trainer.train_model(
                X=X,
                y=y,
                model_type=model_type,
                prediction_type=PredictionType.CLASSIFICATION,
                hyperparameter_tuning=False,  # Set to True for better models (slower)
            )

            trained_models[model_type.value] = {"model": model, "metrics": metrics}

            logger.info(f"  {model_type.value} metrics:")
            logger.info(f"    Train Score: {metrics.get('train_score', 0):.4f}")
            logger.info(f"    Test Score: {metrics.get('test_score', 0):.4f}")
            logger.info(f"    Cross-Val Score: {metrics.get('cv_score', 0):.4f}")

            # Save model
            model_path = Path("trained_models") / f"{model_type.value}_model.pkl"
            await model_trainer.save_model(
                model=model,
                model_type=model_type,
                metrics=metrics,
                feature_columns=feature_columns,
                model_path=model_path,
            )
            logger.info(f"  Saved model to {model_path}")

        except Exception as e:
            logger.error(f"  Failed to train {model_type.value}: {e}")

    return trained_models


async def validate_models(
    trained_models: dict, features_df: pd.DataFrame, targets_df: pd.Series, feature_columns: list
):
    """Validate trained models on recent data.

    Args:
        trained_models: Dictionary of trained models
        features_df: Feature DataFrame
        targets_df: Target Series
        feature_columns: List of feature column names
    """
    # Use last 20% of data for validation
    split_idx = int(len(features_df) * 0.8)

    X_val = features_df[feature_columns].iloc[split_idx:].values
    y_val = targets_df.iloc[split_idx:].values

    logger.info(f"\nValidating models on {len(X_val)} samples...")

    for model_name, model_info in trained_models.items():
        model = model_info["model"]

        # Make predictions
        predictions = model.predict(X_val)

        # Calculate accuracy
        accuracy = np.mean(predictions == y_val)

        # Calculate directional accuracy (more relevant for trading)
        directional_accuracy = np.mean(np.sign(predictions) == np.sign(y_val))

        logger.info(f"\n{model_name} validation:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.4f}")

        # Calculate profit factor (simple backtest)
        returns = features_df["returns"].iloc[split_idx:].values
        strategy_returns = returns * predictions

        winning_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]

        if len(losing_trades) > 0 and losing_trades.sum() != 0:
            profit_factor = winning_trades.sum() / abs(losing_trades.sum())
        else:
            profit_factor = np.inf if len(winning_trades) > 0 else 0

        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Win Rate: {len(winning_trades) / len(strategy_returns):.2%}")


async def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("RoboTrader ML Model Training")
    logger.info("=" * 60)

    # Configuration
    config = Config()

    # Symbols to train on (use liquid stocks with good data)
    symbols = [
        "SPY",  # S&P 500 ETF
        "QQQ",  # Nasdaq ETF
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL",  # Google
        "AMZN",  # Amazon
        "NVDA",  # Nvidia
        "TSLA",  # Tesla
        "META",  # Meta
        "JPM",  # JP Morgan
    ]

    # Step 1: Fetch training data
    logger.info("\n1. Fetching training data...")
    data = await fetch_training_data(symbols, period="2y")

    if not data:
        logger.error("No data fetched. Exiting.")
        return

    # Step 2: Prepare features
    logger.info("\n2. Preparing features...")
    features_df, targets_df, feature_columns = await prepare_training_data(data, config)

    if features_df is None or features_df.empty:
        logger.error("No features prepared. Exiting.")
        return

    logger.info(f"\nDataset statistics:")
    logger.info(f"  Total samples: {len(features_df)}")
    logger.info(f"  Features: {len(feature_columns)}")
    logger.info(f"  Class distribution:")
    logger.info(f"    Up days: {(targets_df == 1).sum()} ({(targets_df == 1).mean():.1%})")
    logger.info(f"    Down days: {(targets_df == -1).sum()} ({(targets_df == -1).mean():.1%})")

    # Step 3: Train models
    logger.info("\n3. Training models...")
    trained_models = await train_models(features_df, targets_df, feature_columns, config)

    if not trained_models:
        logger.error("No models trained. Exiting.")
        return

    # Step 4: Validate models
    logger.info("\n4. Validating models...")
    await validate_models(trained_models, features_df, targets_df, feature_columns)

    logger.info("\n" + "=" * 60)
    logger.info("✅ Model training complete!")
    logger.info("=" * 60)

    logger.info("\nNext steps:")
    logger.info("1. Models saved in 'trained_models/' directory")
    logger.info("2. ML strategy will automatically load these models")
    logger.info("3. Run trading system to use ML predictions")
    logger.info("\nTo use different data or parameters:")
    logger.info("- Modify symbols list for different stocks")
    logger.info("- Change period for more/less historical data")
    logger.info("- Enable hyperparameter_tuning=True for better models (slower)")


if __name__ == "__main__":
    asyncio.run(main())

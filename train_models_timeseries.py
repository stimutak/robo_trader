"""Train ML models with time-series features for RoboTrader."""

import asyncio
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators as features."""
    features = pd.DataFrame(index=df.index)

    # Price features
    features["returns"] = df["close"].pct_change()
    features["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    features["sma_5"] = df["close"].rolling(5).mean() / df["close"]
    features["sma_20"] = df["close"].rolling(20).mean() / df["close"]
    features["sma_50"] = df["close"].rolling(50).mean() / df["close"]
    features["ema_12"] = df["close"].ewm(span=12).mean() / df["close"]
    features["ema_26"] = df["close"].ewm(span=26).mean() / df["close"]

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    features["macd"] = (ema_12 - ema_26) / df["close"]
    features["macd_signal"] = features["macd"].ewm(span=9).mean()
    features["macd_diff"] = features["macd"] - features["macd_signal"]

    # Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    features["bb_upper"] = (sma_20 + 2 * std_20) / df["close"]
    features["bb_lower"] = (sma_20 - 2 * std_20) / df["close"]
    features["bb_width"] = features["bb_upper"] - features["bb_lower"]
    features["bb_position"] = (df["close"] - features["bb_lower"] * df["close"]) / (
        features["bb_width"] * df["close"]
    )

    # Volume features
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    features["volume_sma"] = df["volume"].rolling(20).mean() / df["volume"].mean()

    # Volatility
    features["volatility_20"] = df["returns"].rolling(20).std()
    features["volatility_5"] = df["returns"].rolling(5).std()
    features["volatility_ratio"] = features["volatility_5"] / features["volatility_20"]

    # Price position
    features["high_low_ratio"] = df["high"] / df["low"]
    features["close_high_ratio"] = df["close"] / df["high"]
    features["close_low_ratio"] = df["close"] / df["low"]

    # Momentum
    features["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    features["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    # Market microstructure
    features["spread"] = (df["high"] - df["low"]) / df["close"]
    features["overnight_return"] = df["open"] / df["close"].shift(1) - 1

    return features


async def main():
    print("=" * 60)
    print("RoboTrader ML Model Training (Time-Series)")
    print("=" * 60)

    # Create models directory
    models_dir = Path("trained_models")
    models_dir.mkdir(exist_ok=True)

    # Fetch training data - expanded symbol list for better diversity
    print("\n1. Fetching training data...")
    symbols = [
        # Major indices
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        # Tech giants
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA",
        "AMZN",
        "GOOGL",
        "META",
        "NFLX",
        # Financial
        "JPM",
        "BAC",
        "GS",
        "WFC",
        "MS",
        # Healthcare
        "JNJ",
        "UNH",
        "PFE",
        "ABBV",
        "LLY",
        # Energy
        "XOM",
        "CVX",
        "COP",
        # Consumer
        "WMT",
        "HD",
        "DIS",
        "MCD",
        "NKE",
        # Industrial
        "BA",
        "CAT",
        "GE",
        "HON",
        # Additional high-volume stocks
        "AMD",
        "INTC",
        "ORCL",
        "CRM",
        "ADBE",
        "PYPL",
        "SQ",
        "UBER",
    ]

    all_features = []
    all_targets = []

    for symbol in symbols:
        print(f"  Processing {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")  # Changed from 2y to 5y

        if df.empty or len(df) < 100:
            continue

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        df["returns"] = df["close"].pct_change()

        # Calculate features
        features = calculate_technical_features(df)

        # Create target for significant moves (>1% up or down)
        # 1 = >1% up, 0 = <1% down, -1 = neutral (removed later)
        next_return = df["returns"].shift(-1)
        df["target"] = np.where(
            next_return > 0.01, 1, np.where(next_return < -0.01, 0, -1)  # >1% up  # >1% down
        )  # Neutral, will be filtered out

        # Remove NaN values and neutral targets
        valid_idx = ~(features.isna().any(axis=1) | df["target"].isna()) & (df["target"] != -1)
        features_clean = features[valid_idx]
        targets_clean = df["target"][valid_idx]

        # Remove last row (no target)
        if len(features_clean) > 0 and targets_clean.iloc[-1] == -1:
            features_clean = features_clean[:-1]
            targets_clean = targets_clean[:-1]

        if len(features_clean) > 0:
            all_features.append(features_clean)
            all_targets.append(targets_clean)
            print(f"    Added {len(features_clean)} samples")

    # Combine all data
    print("\n2. Preparing training data...")
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)

    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Up days: {y.sum()} ({y.mean():.1%})")
    print(f"  Down days: {(1-y).sum()} ({(1-y).mean():.1%})")

    # Split data (time-series split, no shuffle)
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Store feature columns
    feature_columns = X.columns.tolist()

    print("\n3. Training models...")

    # Train Random Forest with stronger regularization
    print("\n  Training Random Forest with regularization...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,  # Reduced from 10 to prevent overfitting
        min_samples_split=50,  # Increased from 20
        min_samples_leaf=25,  # Increased from 10
        max_features="sqrt",  # Limit features per split
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_scaled, y_train)

    rf_train_score = rf_model.score(X_train_scaled, y_train)
    rf_test_score = rf_model.score(X_test_scaled, y_test)
    print(f"    Train accuracy: {rf_train_score:.4f}")
    print(f"    Test accuracy: {rf_test_score:.4f}")

    # Save Random Forest
    model_data = {
        "model": rf_model,
        "model_type": "random_forest",
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": {"train_score": rf_train_score, "test_score": rf_test_score},
    }

    rf_path = models_dir / "random_forest_model.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Saved to {rf_path}")

    # Train XGBoost with regularization
    print("\n  Training XGBoost with regularization...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,  # Reduced from 6
        learning_rate=0.05,  # Reduced from 0.1
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        subsample=0.8,  # Sample 80% of data
        colsample_bytree=0.8,  # Sample 80% of features
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_scaled, y_train)

    xgb_train_score = xgb_model.score(X_train_scaled, y_train)
    xgb_test_score = xgb_model.score(X_test_scaled, y_test)
    print(f"    Train accuracy: {xgb_train_score:.4f}")
    print(f"    Test accuracy: {xgb_test_score:.4f}")

    # Save XGBoost
    model_data = {
        "model": xgb_model,
        "model_type": "xgboost",
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": {"train_score": xgb_train_score, "test_score": xgb_test_score},
    }

    xgb_path = models_dir / "xgboost_model.pkl"
    with open(xgb_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Saved to {xgb_path}")

    # Train LightGBM with regularization
    print("\n  Training LightGBM with regularization...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,  # Reduced from 6
        learning_rate=0.05,  # Reduced from 0.1
        num_leaves=31,  # Default, controls model complexity
        min_child_samples=50,  # Minimum data in leaf
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        subsample=0.8,  # Sample 80% of data
        colsample_bytree=0.8,  # Sample 80% of features
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train_scaled, y_train)

    lgb_train_score = lgb_model.score(X_train_scaled, y_train)
    lgb_test_score = lgb_model.score(X_test_scaled, y_test)
    print(f"    Train accuracy: {lgb_train_score:.4f}")
    print(f"    Test accuracy: {lgb_test_score:.4f}")

    # Save LightGBM
    model_data = {
        "model": lgb_model,
        "model_type": "lightgbm",
        "scaler": scaler,
        "feature_columns": feature_columns,
        "metrics": {"train_score": lgb_train_score, "test_score": lgb_test_score},
    }

    lgb_path = models_dir / "lightgbm_model.pkl"
    with open(lgb_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"    Saved to {lgb_path}")

    # Save metadata for each model
    for model_name, model_score in [
        ("random_forest", rf_test_score),
        ("xgboost", xgb_test_score),
        ("lightgbm", lgb_test_score),
    ]:
        metadata = {
            "model_type": model_name,
            "trained_at": datetime.now().isoformat(),
            "symbols": symbols,
            "features": feature_columns,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_score": float(model_score),
        }

        metadata_path = models_dir / f"{model_name}_model.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Model training complete!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print(f"  Random Forest: {rf_test_score:.1%}")
    print(f"  XGBoost: {xgb_test_score:.1%}")
    print(f"  LightGBM: {lgb_test_score:.1%}")
    print(f"\nModels saved in: {models_dir}/")
    print("\nThe ML strategy will now use these trained models for predictions.")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Simplified high-accuracy training focusing on what works."""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def calculate_features(df):
    """Calculate proven features that work."""
    features = pd.DataFrame(index=df.index)

    # Core features that matter
    features["returns"] = df["close"].pct_change()
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()

    # Simple but effective indicators
    for period in [5, 10, 20, 50]:
        features[f"sma_{period}"] = df["close"] / df["close"].rolling(period).mean() - 1
        features[f"volume_sma_{period}"] = df["volume"] / df["volume"].rolling(period).mean() - 1

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features["rsi"] = 100 - (100 / (1 + rs))

    # Volatility
    features["volatility"] = features["returns"].rolling(20).std()
    features["volatility_ratio"] = (
        features["volatility"] / features["volatility"].rolling(50).mean()
    )

    # Price patterns
    features["high_low"] = (df["high"] - df["low"]) / df["close"]
    features["close_open"] = (df["close"] - df["open"]) / df["open"]

    # Momentum
    for period in [3, 5, 10]:
        features[f"momentum_{period}"] = df["close"].pct_change(period)

    # Market regime
    features["trend"] = df["close"] / df["close"].rolling(50).mean() - 1

    # Lag features (crucial for time series)
    for i in range(1, 6):
        features[f"returns_lag_{i}"] = features["returns"].shift(i)
        features[f"volume_lag_{i}"] = features["volume_ratio"].shift(i)

    return features


def main():
    print("High-Accuracy ML Training")
    print("=" * 40)

    # Focus on liquid ETFs for cleaner patterns
    symbols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLF", "XLE", "XLK", "VXX"]

    all_X = []
    all_y = []

    print("\n1. Loading data...")
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")

            if len(df) < 100:
                continue

            df.columns = [col.lower() for col in df.columns]

            # Calculate features
            X = calculate_features(df)

            # Simple binary target: up or down next day
            y = (df["close"].pct_change().shift(-1) > 0).astype(int)

            # Clean data
            valid = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid]
            y = y[valid]

            if len(X) > 50:
                all_X.append(X)
                all_y.append(y)
                print(f"  {symbol}: {len(X)} samples")

        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    # Combine all data
    X = pd.concat(all_X, axis=0)
    y = pd.concat(all_y, axis=0)

    print(f"\n2. Total dataset: {len(X)} samples, {X.shape[1]} features")
    print(f"   Up days: {y.sum()} ({y.mean():.1%})")
    print(f"   Down days: {(1-y).sum()} ({(1-y).mean():.1%})")

    # Handle any remaining issues
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series
    )

    print(f"\n3. Training models...")

    # Model 1: Random Forest with balanced classes
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=50,
        class_weight="balanced",  # Important for imbalanced data
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"\nRandom Forest accuracy: {rf_acc:.3f}")

    # Model 2: XGBoost with tuned parameters
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y_train).sum() / y_train.sum(),  # Handle imbalance
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"XGBoost accuracy: {xgb_acc:.3f}")

    # High confidence predictions only
    print("\n4. High-confidence predictions (>60% probability):")

    for name, proba, pred in [("RF", rf_proba, rf_pred), ("XGB", xgb_proba, xgb_pred)]:
        high_conf = np.max(proba, axis=1) > 0.6
        if high_conf.sum() > 0:
            high_conf_acc = accuracy_score(y_test[high_conf], pred[high_conf])
            print(f"  {name}: {high_conf.sum()} trades, {high_conf_acc:.3f} accuracy")

    # Feature importance
    print("\n5. Top 10 most important features:")
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    # Save best model
    best_model = rf if rf_acc > xgb_acc else xgb_model
    best_acc = max(rf_acc, xgb_acc)

    if best_acc > 0.52:  # Better than random
        Path("trained_models").mkdir(exist_ok=True)

        with open("trained_models/high_accuracy_model.pkl", "wb") as f:
            pickle.dump({"model": best_model, "features": list(X.columns), "accuracy": best_acc}, f)

        print(f"\n✅ Model saved with {best_acc:.1%} accuracy!")

        # Test on very recent data
        print("\n6. Testing on last 20 days:")
        last_20 = X_test.iloc[-20:]
        last_20_y = y_test.iloc[-20:]
        last_20_pred = best_model.predict(last_20)
        last_20_acc = accuracy_score(last_20_y, last_20_pred)
        print(f"  Recent accuracy: {last_20_acc:.3f}")


if __name__ == "__main__":
    main()

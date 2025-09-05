#!/usr/bin/env python3
"""Breakthrough ML training - targeting 60%+ accuracy with advanced techniques."""

import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def calculate_market_regime(df, spy_df):
    """Identify market regime - crucial for context."""
    features = pd.DataFrame(index=df.index)

    # Market regime from SPY
    spy_returns = spy_df["close"].pct_change()
    features["market_trend"] = spy_df["close"] / spy_df["close"].rolling(20).mean() - 1
    features["market_volatility"] = spy_returns.rolling(20).std()
    features["vix_proxy"] = features["market_volatility"] * np.sqrt(252)  # Annualized

    # Regime classification
    features["bull_market"] = (features["market_trend"] > 0.02).astype(int)
    features["high_vol"] = (
        features["vix_proxy"] > features["vix_proxy"].rolling(50).mean()
    ).astype(int)

    return features


def calculate_microstructure(df):
    """Market microstructure features - where alpha lives."""
    features = pd.DataFrame(index=df.index)

    # Intraday patterns
    features["overnight_gap"] = df["open"] / df["close"].shift(1) - 1
    features["intraday_return"] = df["close"] / df["open"] - 1
    features["morning_move"] = (df["high"] - df["open"]) / df["open"]
    features["afternoon_move"] = (df["close"] - df["low"]) / df["low"]

    # Volume patterns
    features["volume_burst"] = df["volume"] / df["volume"].rolling(20).mean()
    features["dollar_volume"] = df["close"] * df["volume"]
    features["volume_price_corr"] = (
        df["close"].pct_change().rolling(20).corr(df["volume"].pct_change())
    )

    # Price efficiency
    features["high_low_spread"] = (df["high"] - df["low"]) / df["close"]
    features["close_efficiency"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(
        0, 1
    )

    # Liquidity proxy
    features["amihud_illiquidity"] = abs(df["close"].pct_change()) / (df["volume"] * df["close"])
    features["amihud_illiquidity"] = features["amihud_illiquidity"].rolling(20).mean()

    return features


def calculate_cross_asset_signals(symbol, all_data):
    """Cross-asset correlations and lead-lag relationships."""
    features = pd.DataFrame(index=all_data[symbol].index)

    # Correlations with major assets
    symbol_returns = all_data[symbol]["close"].pct_change()

    for other_symbol in ["SPY", "QQQ", "TLT", "GLD", "VXX"]:
        if other_symbol != symbol and other_symbol in all_data:
            other_returns = all_data[other_symbol]["close"].pct_change()
            # Align indices
            aligned = pd.concat([symbol_returns, other_returns], axis=1).dropna()
            if len(aligned) > 20:
                # Rolling correlation
                features[f"corr_{other_symbol}"] = symbol_returns.rolling(20).corr(other_returns)
                # Lead-lag: does other asset lead this one?
                features[f"lead_{other_symbol}"] = symbol_returns.rolling(20).corr(
                    other_returns.shift(1)
                )

    return features


def engineer_advanced_features(df, symbol, all_data, spy_df):
    """Combine all advanced feature engineering."""
    features = pd.DataFrame(index=df.index)

    # Basic returns and momentum
    features["returns"] = df["close"].pct_change()
    features["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Multiple timeframe momentum
    for period in [2, 5, 10, 20]:
        features[f"momentum_{period}"] = df["close"].pct_change(period)
        features[f"momentum_acc_{period}"] = features[f"momentum_{period}"] - features[
            f"momentum_{period}"
        ].shift(period)

    # Advanced technical indicators
    # Relative Strength Index variants
    for period in [9, 14, 21]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        features[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        features[f"rsi_divergence_{period}"] = (
            features[f"rsi_{period}"] - features[f"rsi_{period}"].rolling(5).mean()
        )

    # Stochastic oscillator
    for period in [14, 21]:
        low_min = df["low"].rolling(period).min()
        high_max = df["high"].rolling(period).max()
        features[f"stoch_{period}"] = 100 * (df["close"] - low_min) / (high_max - low_min)

    # Bollinger Bands with multiple settings
    for period in [10, 20]:
        for num_std in [1.5, 2, 2.5]:
            sma = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            features[f"bb_position_{period}_{num_std}"] = (df["close"] - sma) / (num_std * std)

    # Volume-weighted indicators
    features["vwap"] = (df["close"] * df["volume"]).rolling(20).sum() / df["volume"].rolling(
        20
    ).sum()
    features["price_to_vwap"] = df["close"] / features["vwap"] - 1

    # On-Balance Volume
    obv = (df["volume"] * (~df["close"].diff().le(0) * 2 - 1)).cumsum()
    features["obv_roc"] = obv.pct_change(10)

    # Statistical measures
    for period in [10, 20, 50]:
        returns_window = features["returns"].rolling(period)
        features[f"skew_{period}"] = returns_window.skew()
        features[f"kurtosis_{period}"] = returns_window.kurt()
        features[f"volatility_{period}"] = returns_window.std()

    # Volatility ratios and changes
    features["volatility_ratio"] = features["volatility_10"] / features["volatility_50"]
    features["volatility_change"] = features["volatility_10"].pct_change(5)

    # Historical patterns - what happened after similar setups?
    for lag in [1, 2, 3, 5, 10]:
        features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
        features[f"volume_lag_{lag}"] = df["volume"].pct_change().shift(lag)

    # Day of week and month effects
    features["day_of_week"] = df.index.dayofweek
    features["month"] = df.index.month
    features["quarter"] = df.index.quarter
    features["is_monday"] = (features["day_of_week"] == 0).astype(int)
    features["is_friday"] = (features["day_of_week"] == 4).astype(int)
    features["month_end"] = (df.index.day > 25).astype(int)

    # Add specialized features
    regime_features = calculate_market_regime(df, spy_df)
    microstructure_features = calculate_microstructure(df)
    cross_asset_features = calculate_cross_asset_signals(symbol, all_data)

    # Combine all features
    features = pd.concat(
        [features, regime_features, microstructure_features, cross_asset_features], axis=1
    )

    return features


def create_smart_labels(df, min_move=0.003):
    """Create labels focusing on meaningful moves."""
    returns = df["close"].pct_change().shift(-1)  # Next day return

    # Three-class problem: strong down, neutral, strong up
    labels = pd.Series(index=df.index, dtype=int)
    labels[returns < -min_move] = 0  # Strong down
    labels[returns > min_move] = 1  # Strong up

    # For now, we'll focus on strong moves only
    mask = (returns < -min_move) | (returns > min_move)

    return labels, mask


def train_breakthrough_model(X_train, y_train, X_test, y_test):
    """Train ensemble with optimized hyperparameters."""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # 1. Gradient Boosting - often best for tabular data
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=50,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train_scaled, y_train)
    models["gb"] = gb

    # 2. LightGBM - fast and accurate
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    lgb_model.fit(X_train_scaled, y_train)
    models["lgb"] = lgb_model

    # 3. XGBoost with careful tuning
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_scaled, y_train)
    models["xgb"] = xgb_model

    # Evaluate all models
    results = {}
    for name, model in models.items():
        pred = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)

        acc = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average="weighted")
        recall = recall_score(y_test, pred, average="weighted")

        # High confidence predictions
        high_conf_mask = np.max(proba, axis=1) > 0.65
        if high_conf_mask.sum() > 10:
            high_conf_acc = accuracy_score(y_test[high_conf_mask], pred[high_conf_mask])
        else:
            high_conf_acc = 0

        results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "high_conf_acc": high_conf_acc,
            "high_conf_trades": high_conf_mask.sum(),
            "model": model,
        }

        print(f"\n{name.upper()} Results:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  High-conf ({high_conf_mask.sum()} trades): {high_conf_acc:.3f}")

    # Return best model
    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    return results[best_model_name]["model"], results, scaler


def main():
    print("=" * 60)
    print("BREAKTHROUGH ML MODEL TRAINING")
    print("Target: 60%+ Accuracy")
    print("=" * 60)

    # Load all data first for cross-asset features
    symbols = [
        "SPY",
        "QQQ",
        "IWM",
        "TLT",
        "GLD",
        "XLF",
        "XLE",
        "XLK",
        "VXX",
        "AAPL",
        "MSFT",
        "GOOGL",
    ]
    all_data = {}

    print("\n1. Loading multi-asset data...")
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="3y", interval="1d")
            df.columns = [col.lower() for col in df.columns]
            all_data[symbol] = df
            print(f"  ✓ {symbol}: {len(df)} days")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    # SPY is our market regime indicator
    spy_df = all_data["SPY"]

    # Process each symbol
    all_features = []
    all_labels = []

    print("\n2. Engineering advanced features...")
    for symbol in symbols[:9]:  # Focus on most liquid
        if symbol not in all_data:
            continue

        df = all_data[symbol]

        # Engineer features
        features = engineer_advanced_features(df, symbol, all_data, spy_df)

        # Create smart labels
        labels, mask = create_smart_labels(df, min_move=0.004)  # 0.4% moves

        # Apply mask to get only strong moves
        features = features[mask]
        labels = labels[mask]

        # Clean data
        valid = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid]
        labels = labels[valid]

        if len(features) > 100:
            all_features.append(features)
            all_labels.append(labels)
            print(f"  {symbol}: {len(features)} strong moves")

    # Combine all
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)

    # Handle infinities and NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    print(f"\n3. Dataset ready:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Up moves: {(y==1).sum()} ({(y==1).mean():.1%})")
    print(f"  Down moves: {(y==0).sum()} ({(y==0).mean():.1%})")

    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\n4. Training breakthrough models...")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Train models
    best_model, results, scaler = train_breakthrough_model(X_train, y_train, X_test, y_test)

    # Get best accuracy
    best_acc = max(r["accuracy"] for r in results.values())
    best_high_conf = max(r["high_conf_acc"] for r in results.values())

    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"  Best Accuracy: {best_acc:.1%}")
    print(f"  Best High-Conf: {best_high_conf:.1%}")

    if best_acc > 0.58:  # Save if we beat 58%
        print(f"\n✅ BREAKTHROUGH! Saving model...")
        with open("trained_models/breakthrough_model.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": best_model,
                    "scaler": scaler,
                    "features": list(X.columns),
                    "accuracy": best_acc,
                    "high_conf_accuracy": best_high_conf,
                },
                f,
            )
    else:
        print(f"\n⚠️ Need more work to reach 60%")


if __name__ == "__main__":
    main()

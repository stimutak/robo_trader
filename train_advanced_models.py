#!/usr/bin/env python3
"""Advanced ML training with better features and techniques for higher accuracy."""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_advanced_features(df):
    """Calculate advanced technical features for better accuracy."""
    features = pd.DataFrame(index=df.index)
    
    # Price and returns
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility features
    for period in [5, 10, 20]:
        features[f'volatility_{period}'] = features['returns'].rolling(period).std()
    
    # Volatility ratios
    for period in [5, 10]:
        features[f'volatility_ratio_{period}'] = features[f'volatility_{period}'] / features['volatility_20']
    
    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['dollar_volume'] = df['close'] * df['volume']
    features['volume_momentum'] = df['volume'].pct_change(5)
    
    # Price patterns
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_to_high'] = df['close'] / df['high']
    features['close_to_low'] = df['close'] / df['low']
    features['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Advanced momentum
    for fast, slow in [(5, 20), (10, 30), (20, 50)]:
        features[f'momentum_{fast}_{slow}'] = df['close'].rolling(fast).mean() / df['close'].rolling(slow).mean()
    
    # RSI with multiple periods
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    features['macd'] = exp1 - exp2
    features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
    features['macd_histogram'] = features['macd'] - features['macd_signal']
    
    # Bollinger Bands
    for period in [10, 20]:
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        features[f'bb_upper_{period}'] = (df['close'] - (sma + 2*std)) / df['close']
        features[f'bb_lower_{period}'] = (df['close'] - (sma - 2*std)) / df['close']
        features[f'bb_width_{period}'] = (4 * std) / df['close']
    
    # Market microstructure
    features['spread'] = (df['high'] - df['low']) / df['close']
    features['overnight_gap'] = df['open'] / df['close'].shift(1) - 1
    
    # Time features
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    
    # Lag features for time series
    for lag in [1, 2, 3, 5]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
        features[f'volume_lag_{lag}'] = features['volume_ratio'].shift(lag)
    
    # Rolling statistics
    for period in [5, 10, 20]:
        features[f'returns_mean_{period}'] = features['returns'].rolling(period).mean()
        features[f'returns_std_{period}'] = features['returns'].rolling(period).std()
        features[f'returns_skew_{period}'] = features['returns'].rolling(period).skew()
        features[f'returns_kurt_{period}'] = features['returns'].rolling(period).kurt()
    
    return features

def create_labels(df, threshold=0.002):
    """Create labels for larger moves (default 0.2% threshold)."""
    # Target: Next day's return
    next_return = df['close'].pct_change().shift(-1)
    
    # Three classes: down, neutral, up
    labels = pd.Series(index=df.index, dtype=int)
    labels[next_return < -threshold] = 0  # Down
    labels[abs(next_return) <= threshold] = 1  # Neutral (skip these)
    labels[next_return > threshold] = 2  # Up
    
    # For binary classification, remove neutral
    binary_labels = labels.copy()
    binary_labels[labels == 0] = 0  # Down
    binary_labels[labels == 2] = 1  # Up
    
    return binary_labels, labels

def train_ensemble_model(X_train, y_train, X_test, y_test):
    """Train an ensemble of models for better accuracy."""
    
    # Individual models with optimized parameters
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
    
    # Ensemble voting classifier
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        voting='soft'  # Use probability voting
    )
    
    # Train ensemble
    print("Training ensemble model...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    train_pred = ensemble.predict(X_train)
    test_pred = ensemble.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"Ensemble - Train accuracy: {train_acc:.4f}")
    print(f"Ensemble - Test accuracy: {test_acc:.4f}")
    
    # Get prediction probabilities for high-confidence filtering
    test_proba = ensemble.predict_proba(X_test)
    high_conf_mask = np.max(test_proba, axis=1) > 0.65  # Only high confidence predictions
    
    if high_conf_mask.sum() > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], test_pred[high_conf_mask])
        print(f"High confidence trades ({high_conf_mask.sum()}): {high_conf_acc:.4f} accuracy")
    
    return ensemble, test_acc

def main():
    print("=" * 60)
    print("Advanced ML Model Training for Higher Accuracy")
    print("=" * 60)
    
    # Fetch more data for better training
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA']
    period = '5y'  # More data
    
    all_features = []
    all_labels = []
    
    print(f"\n1. Fetching {period} of data for training...")
    for symbol in symbols:
        print(f"  Processing {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval='1d')
        
        if len(df) < 100:
            continue
        
        # Fix column names
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate advanced features
        features = calculate_advanced_features(df)
        
        # Create labels (only for moves > 0.3%)
        labels, _ = create_labels(df, threshold=0.003)
        
        # Remove neutral labels
        mask = labels != 1
        features = features[mask]
        labels = labels[mask]
        
        # Drop NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        if len(features) > 50:
            all_features.append(features)
            all_labels.append(labels)
    
    # Combine all data
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_labels, axis=0)
    
    print(f"\n2. Dataset prepared:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Up moves: {(y == 1).sum()} ({(y == 1).mean():.1%})")
    print(f"  Down moves: {(y == 0).sum()} ({(y == 0).mean():.1%})")
    
    # Remove any remaining NaN or inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    # Split data (time series split)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n3. Training advanced models...")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Train ensemble model
    ensemble_model, test_acc = train_ensemble_model(
        X_train_scaled, y_train, 
        X_test_scaled, y_test
    )
    
    # Save the best model
    if test_acc > 0.55:  # Only save if better than baseline
        Path('trained_models').mkdir(exist_ok=True)
        
        model_data = {
            'model': ensemble_model,
            'scaler': scaler,
            'features': list(X.columns),
            'accuracy': test_acc
        }
        
        with open('trained_models/advanced_ensemble_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✅ Advanced model saved with {test_acc:.1%} accuracy!")
    else:
        print(f"\n⚠️ Model accuracy {test_acc:.1%} not better than baseline")
    
    # Time series cross-validation for robustness
    print("\n4. Cross-validation results:")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(ensemble_model, X_train_scaled, y_train, cv=tscv, scoring='accuracy')
    print(f"  CV scores: {cv_scores}")
    print(f"  Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

if __name__ == "__main__":
    main()
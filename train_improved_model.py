#!/usr/bin/env python3
"""Improved ML training with better features and targets."""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def calculate_advanced_features(df):
    """Calculate advanced technical features."""
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    features['volatility_20'] = features['returns'].rolling(20).std()
    features['volatility_5'] = features['returns'].rolling(5).std()
    
    # Volume features
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    features['dollar_volume'] = df['close'] * df['volume']
    
    # Time features
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    
    # Microstructure
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_to_high'] = df['close'] / df['high']
    features['close_to_low'] = df['close'] / df['low']
    
    # Momentum
    for period in [5, 10, 20, 50]:
        features[f'return_{period}d'] = df['close'].pct_change(period)
        features[f'sma_{period}'] = df['close'].rolling(period).mean() / df['close']
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    features['macd'] = (ema_12 - ema_26) / df['close']
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(20).mean()
    std_20 = df['close'].rolling(20).std()
    features['bb_position'] = (df['close'] - sma_20) / (2 * std_20)
    
    return features

def main():
    print("Training Improved ML Models...")
    
    # Use 0.5% threshold instead of 1%
    THRESHOLD = 0.005  # 0.5% moves
    
    # Fetch data
    symbols = ["SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "GOOGL", "AMZN"]
    
    all_features = []
    all_targets = []
    all_probs = []  # Store probability of move size
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        
        if df.empty or len(df) < 100:
            continue
        
        df.columns = [col.lower() for col in df.columns]
        
        # Calculate features
        features = calculate_advanced_features(df)
        
        # Create targets with confidence
        next_return = df['close'].pct_change().shift(-1)
        
        # Binary target for direction
        target = np.where(next_return > THRESHOLD, 1,
                 np.where(next_return < -THRESHOLD, 0, -1))
        
        # Probability based on move size
        prob = np.abs(next_return) / 0.02  # Normalize by 2% move
        prob = np.clip(prob, 0, 1)  # Cap at 100%
        
        # Remove neutral and NaN
        valid_idx = ~features.isna().any(axis=1) & (target != -1) & ~pd.isna(next_return)
        
        if valid_idx.sum() > 0:
            all_features.append(features[valid_idx])
            all_targets.append(pd.Series(target[valid_idx], index=features[valid_idx].index))
            all_probs.append(pd.Series(prob[valid_idx], index=features[valid_idx].index))
    
    # Combine all data
    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_targets, ignore_index=True)
    probs = pd.concat(all_probs, ignore_index=True)
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Up moves: {(y==1).sum()} ({(y==1).mean():.1%})")
    print(f"Down moves: {(y==0).sum()} ({(y==0).mean():.1%})")
    
    # Train with sample weights based on move size
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    weights_train = probs.iloc[:split_idx]
    
    # Train model with sample weights
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=100,
        min_samples_leaf=50,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"\nModel Performance:")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Get prediction probabilities
    test_probs = model.predict_proba(X_test)
    
    # Only trade high confidence predictions
    confidence_threshold = 0.6
    high_conf_mask = np.max(test_probs, axis=1) > confidence_threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (model.predict(X_test)[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"\nHigh confidence trades (>{confidence_threshold:.0%}):")
        print(f"Number of trades: {high_conf_mask.sum()} ({high_conf_mask.mean():.1%} of total)")
        print(f"Accuracy: {high_conf_accuracy:.4f}")
    
    # Save model
    Path("trained_models").mkdir(exist_ok=True)
    model_data = {
        'model': model,
        'features': X.columns.tolist(),
        'threshold': THRESHOLD,
        'confidence_threshold': confidence_threshold,
        'metrics': {
            'train_score': train_score,
            'test_score': test_score
        }
    }
    
    with open("trained_models/improved_model.pkl", 'wb') as f:
        pickle.dump(model_data, f)
    
    print("\n✅ Model saved to trained_models/improved_model.pkl")

if __name__ == "__main__":
    main()

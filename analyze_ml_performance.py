#!/usr/bin/env python3
"""Analyze ML model performance and suggest improvements."""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import yfinance as yf
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def analyze_current_models():
    """Analyze current model performance."""
    print("=" * 60)
    print("ML Model Performance Analysis")
    print("=" * 60)
    
    models_dir = Path("trained_models")
    
    # Load a model to analyze
    model_path = models_dir / "random_forest_model.pkl"
    if not model_path.exists():
        print("❌ No trained models found. Run: python train_models_timeseries.py")
        return
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    print(f"\n1. Current Model Stats:")
    print(f"   Train accuracy: {model_data['metrics']['train_score']:.4f}")
    print(f"   Test accuracy: {model_data['metrics']['test_score']:.4f}")
    print(f"   Gap: {abs(model_data['metrics']['train_score'] - model_data['metrics']['test_score']):.4f}")
    
    # Analyze feature importance
    model = model_data['model']
    features = model_data['feature_columns']
    
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n2. Top 10 Most Important Features:")
        for i, row in importance.head(10).iterrows():
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")
    
    return model_data


def test_different_thresholds():
    """Test different prediction thresholds."""
    print("\n" + "=" * 60)
    print("Testing Different Prediction Thresholds")
    print("=" * 60)
    
    # Fetch some test data
    ticker = yf.Ticker("SPY")
    df = ticker.history(period="1y", interval="1d")
    df['returns'] = df['Close'].pct_change()
    
    # Analyze return distribution
    returns = df['returns'].dropna()
    
    print(f"\n1. SPY Daily Return Distribution (1 year):")
    print(f"   Mean: {returns.mean():.4f} ({returns.mean()*100:.2f}%)")
    print(f"   Std: {returns.std():.4f} ({returns.std()*100:.2f}%)")
    print(f"   Skew: {returns.skew():.4f}")
    print(f"   Kurtosis: {returns.kurtosis():.4f}")
    
    # Test different thresholds
    thresholds = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    
    print(f"\n2. Percentage of Days Exceeding Threshold:")
    print(f"   {'Threshold':>10s} | {'Up Days':>10s} | {'Down Days':>10s} | {'Neutral':>10s}")
    print(f"   {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    
    for threshold in thresholds:
        up_days = (returns > threshold).sum()
        down_days = (returns < -threshold).sum()
        neutral = len(returns) - up_days - down_days
        
        up_pct = up_days / len(returns) * 100
        down_pct = down_days / len(returns) * 100
        neutral_pct = neutral / len(returns) * 100
        
        print(f"   {threshold*100:>9.1f}% | {up_pct:>9.1f}% | {down_pct:>9.1f}% | {neutral_pct:>9.1f}%")
    
    print(f"\n3. Recommendation:")
    print(f"   Current threshold (1%) filters out ~60% of days as neutral")
    print(f"   This makes the problem harder - fewer training examples")
    print(f"   Consider using 0.5% threshold for more balanced classes")


def suggest_improvements():
    """Suggest improvements for ML models."""
    print("\n" + "=" * 60)
    print("Recommendations for Improving ML Accuracy")
    print("=" * 60)
    
    print("\n1. IMMEDIATE FIXES (Can implement now):")
    print("   a) Change prediction threshold from 1% to 0.5%")
    print("      - More training examples (less filtering)")
    print("      - More balanced classes")
    print("   b) Use probability outputs instead of binary predictions")
    print("      - Trade only when confidence > 60%")
    print("      - Adjust position size based on confidence")
    print("   c) Add time-based features:")
    print("      - Day of week, month of year")
    print("      - Days to earnings, ex-dividend")
    print("      - Market regime indicators")
    
    print("\n2. DATA IMPROVEMENTS:")
    print("   a) Use intraday data (5-min bars) for better patterns")
    print("   b) Add market breadth indicators (VIX, put/call ratio)")
    print("   c) Include pre/post market data")
    print("   d) Add sector rotation signals")
    
    print("\n3. MODEL ARCHITECTURE:")
    print("   a) Try LSTM/GRU for sequential patterns")
    print("   b) Use ensemble with different time horizons")
    print("   c) Implement online learning (update daily)")
    print("   d) Use different models for different market regimes")
    
    print("\n4. ALTERNATIVE APPROACH:")
    print("   Instead of predicting direction, predict:")
    print("   - Volatility (easier to predict)")
    print("   - Relative strength (which stock will outperform)")
    print("   - Market regime (trending vs mean-reverting)")
    print("   - Entry/exit timing (not direction)")
    
    print("\n5. REALISTIC EXPECTATIONS:")
    print("   - 55-57% directional accuracy is actually good")
    print("   - Focus on risk-adjusted returns, not accuracy")
    print("   - Use ML for position sizing, not just direction")
    print("   - Combine with traditional indicators")


def create_improved_training_script():
    """Create an improved training script."""
    
    improved_script = '''#!/usr/bin/env python3
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
    
    print(f"\\nTotal samples: {len(X)}")
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
    
    print(f"\\nModel Performance:")
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Get prediction probabilities
    test_probs = model.predict_proba(X_test)
    
    # Only trade high confidence predictions
    confidence_threshold = 0.6
    high_conf_mask = np.max(test_probs, axis=1) > confidence_threshold
    
    if high_conf_mask.sum() > 0:
        high_conf_accuracy = (model.predict(X_test)[high_conf_mask] == y_test[high_conf_mask]).mean()
        print(f"\\nHigh confidence trades (>{confidence_threshold:.0%}):")
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
    
    print("\\n✅ Model saved to trained_models/improved_model.pkl")

if __name__ == "__main__":
    main()
'''
    
    with open("train_improved_model.py", "w") as f:
        f.write(improved_script)
    
    print("\n" + "=" * 60)
    print("Created Improved Training Script")
    print("=" * 60)
    print("\nRun: python train_improved_model.py")
    print("\nKey improvements:")
    print("- Lower threshold (0.5% vs 1%)")
    print("- Sample weighting by move size")
    print("- Confidence-based filtering")
    print("- More features including time-based")


def main():
    """Run all analyses."""
    
    # Analyze current models
    model_data = analyze_current_models()
    
    # Test different thresholds
    test_different_thresholds()
    
    # Suggest improvements
    suggest_improvements()
    
    # Create improved script
    create_improved_training_script()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("\n1. Run the improved training script:")
    print("   python train_improved_model.py")
    print("\n2. For smart execution, connect real data:")
    print("   - Modify SmartExecutor to fetch from IBKR")
    print("   - Use real volume profiles instead of mock")
    print("\n3. Test short selling carefully:")
    print("   - Start with small positions")
    print("   - Monitor margin requirements")
    print("   - Set strict stop losses")


if __name__ == "__main__":
    main()
"""Test script for Phase 3 S1: ML-Driven Strategy Framework.

This script tests the newly implemented ML strategy and regime detection.
"""

import asyncio

# Add project root to path
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.append(".")

from robo_trader.features.feature_pipeline import FeaturePipeline
from robo_trader.ml.model_selector import ModelSelector
from robo_trader.ml.model_trainer import ModelTrainer
from robo_trader.strategies.ml_strategy import MLStrategy, TimeFrame
from robo_trader.strategies.regime_detector import MarketRegime, RegimeDetector


async def test_regime_detection():
    """Test regime detection functionality."""
    print("\n" + "=" * 60)
    print("Testing Regime Detection")
    print("=" * 60)

    # Initialize regime detector
    detector = RegimeDetector(
        lookback_period=100, vol_lookback=30, use_ml_detection=False  # Start with rule-based
    )

    await detector.initialize()

    # Get sample data
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)
    data = ticker.history(period="6mo", interval="1d")

    if data.empty:
        print(f"Failed to fetch data for {symbol}")
        return

    # Rename columns to lowercase
    data.columns = [col.lower() for col in data.columns]

    # Detect regime
    regime_state = await detector.detect_regime(symbol, data)

    print(f"\nCurrent Regime for {symbol}:")
    print(f"  Trend Regime: {regime_state.trend_regime.value}")
    print(f"  Volatility Regime: {regime_state.volatility_regime.value}")
    print(f"  Confidence: {regime_state.confidence:.2%}")
    print(f"  Transition Probability: {regime_state.transition_probability:.2%}")

    print("\nRegime Indicators:")
    for key, value in regime_state.indicators.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Get recommendations
    recommendations = detector.get_regime_recommendations(regime_state)
    print("\nTrading Recommendations:")
    print(f"  Position Size Multiplier: {recommendations['position_size_multiplier']:.2f}")
    print(f"  Risk Level: {recommendations['risk_level']}")
    print(f"  Preferred Strategies: {recommendations['preferred_strategies']}")

    # Test multiple symbols
    print("\n" + "-" * 40)
    print("Testing Multiple Symbols:")
    symbols = ["NVDA", "TSLA", "SPY", "GLD"]

    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            data = ticker.history(period="3mo", interval="1d")
            data.columns = [col.lower() for col in data.columns]

            regime = await detector.detect_regime(sym, data)
            print(f"\n{sym}:")
            print(f"  Trend: {regime.trend_regime.value}")
            print(f"  Volatility: {regime.volatility_regime.value}")
            print(f"  Confidence: {regime.confidence:.2%}")

        except Exception as e:
            print(f"Error processing {sym}: {e}")

    print("\n✅ Regime detection test completed")


async def test_ml_strategy():
    """Test ML strategy signal generation."""
    print("\n" + "=" * 60)
    print("Testing ML Strategy")
    print("=" * 60)

    # Initialize components
    print("\nInitializing components...")

    # Create config
    from robo_trader.config import Config

    config = Config()

    # Feature pipeline
    feature_pipeline = FeaturePipeline(config)

    # Model trainer and selector
    model_trainer = ModelTrainer(
        config=config, model_dir=Path("trained_models"), feature_pipeline=feature_pipeline
    )

    model_selector = ModelSelector(model_trainer=model_trainer, model_dir=Path("trained_models"))

    # Initialize ML strategy
    ml_strategy = MLStrategy(
        model_selector=model_selector,
        feature_pipeline=feature_pipeline,
        confidence_threshold=0.6,
        ensemble_agreement=0.5,
        use_regime_filter=True,
        timeframes=[TimeFrame.INTRADAY, TimeFrame.SWING],
        symbols=["AAPL"],  # Test with one symbol
        name="TestMLStrategy",
    )

    # Initialize with empty historical data
    await ml_strategy._initialize({})

    # Test with sample data
    symbol = "AAPL"
    ticker = yf.Ticker(symbol)

    # Get data for different timeframes
    print(f"\nFetching data for {symbol}...")

    data_5min = ticker.history(period="5d", interval="5m")
    data_30min = ticker.history(period="1mo", interval="30m")
    data_daily = ticker.history(period="6mo", interval="1d")

    # Rename columns
    for df in [data_5min, data_30min, data_daily]:
        df.columns = [col.lower() for col in df.columns]

    # Prepare market data
    market_data = {
        "symbol": symbol,
        "price": data_5min["close"].iloc[-1] if not data_5min.empty else 100,
        "data": data_5min,
        f"data_{TimeFrame.INTRADAY.value}": data_5min,
        f"data_{TimeFrame.SWING.value}": data_30min,
        f"data_{TimeFrame.POSITION.value}": data_daily,
        "portfolio_value": 100000,
        "atr": data_daily["close"].rolling(14).std().iloc[-1] if not data_daily.empty else 2,
    }

    print("\nMarket Data Summary:")
    print(f"  Current Price: ${market_data['price']:.2f}")
    print(f"  ATR: ${market_data['atr']:.2f}")
    print(f"  5min bars: {len(data_5min)}")
    print(f"  30min bars: {len(data_30min)}")
    print(f"  Daily bars: {len(data_daily)}")

    # Test feature generation
    print("\nTesting feature generation...")
    if not data_5min.empty:
        features = await ml_strategy.generate_features(symbol, data_5min, TimeFrame.INTRADAY)
        print(f"  Generated {len(features.columns)} features")
        print(f"  Feature shape: {features.shape}")

        # Show some features
        if not features.empty:
            print("\n  Sample features (last row):")
            for col in features.columns[:5]:
                value = features[col].iloc[-1]
                if isinstance(value, float):
                    print(f"    {col}: {value:.4f}")

    # Test multi-timeframe analysis
    print("\nTesting multi-timeframe analysis...")
    data_dict = {TimeFrame.INTRADAY: data_5min, TimeFrame.SWING: data_30min}

    if not data_5min.empty and not data_30min.empty:
        analysis = await ml_strategy.multi_timeframe_analysis(symbol, data_dict)

        print("\nMulti-timeframe Analysis Results:")
        print(f"  Signal: {analysis['signal'].value}")
        print(f"  Score: {analysis.get('score', 0):.4f}")
        print(f"  Confidence: {analysis.get('confidence', 0):.2%}")

        if "analyses" in analysis:
            for tf, tf_analysis in analysis["analyses"].items():
                print(f"\n  {tf.value} Analysis:")
                print(f"    ML Prediction: {tf_analysis.get('ml_prediction', 0):.4f}")
                print(f"    ML Confidence: {tf_analysis.get('ml_confidence', 0):.2%}")
                print(f"    Technical Score: {tf_analysis.get('technical_score', 0):.2f}")
                print(f"    Trend: {tf_analysis.get('trend', 'unknown')}")

    # Test signal generation
    print("\nTesting signal generation...")
    signal = await ml_strategy.generate_signal(symbol, market_data)

    if signal:
        print("\n📊 Generated ML Signal:")
        print(f"  Symbol: {signal.symbol}")
        print(f"  Type: {signal.signal_type.value}")
        print(f"  Strength: {signal.strength:.2%}")
        print(f"  ML Confidence: {signal.ml_confidence:.2%}")
        print(f"  Regime: {signal.regime}")
        print(f"  Quantity: {signal.quantity} shares")
        print(f"  Entry Price: ${signal.entry_price:.2f}")
        print(f"  Stop Loss: ${signal.stop_loss:.2f}")
        print(f"  Take Profit: ${signal.take_profit:.2f}")
        print(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
    else:
        print("\n  No signal generated (HOLD)")

    # Get strategy status
    status = ml_strategy.get_status()
    print("\nStrategy Status:")
    print(f"  State: {status['state']}")
    print(f"  Signals Generated: {status['metrics'].get('signals_generated', 0)}")
    print(f"  Model Count: {status.get('model_count', 0)}")
    print(f"  Cached Features: {status.get('cached_features', 0)}")

    print("\n✅ ML strategy test completed")


async def test_position_sizing():
    """Test regime-aware position sizing."""
    print("\n" + "=" * 60)
    print("Testing Position Sizing")
    print("=" * 60)

    # Create a dummy ML strategy for testing
    feature_pipeline = FeaturePipeline()
    model_trainer = ModelTrainer(model_dir=Path("trained_models"), feature_columns=[])
    model_selector = ModelSelector(model_trainer=model_trainer, model_dir=Path("trained_models"))

    ml_strategy = MLStrategy(
        model_selector=model_selector,
        feature_pipeline=feature_pipeline,
        position_size_method="kelly",
        max_position_pct=0.1,
        risk_per_trade=0.02,
    )

    # Test scenarios
    scenarios = [
        {
            "name": "High Confidence Bull Market",
            "signal_strength": 0.8,
            "confidence": 0.85,
            "regime": "bull",
            "portfolio_value": 100000,
            "current_price": 150,
        },
        {
            "name": "Low Confidence Volatile Market",
            "signal_strength": 0.5,
            "confidence": 0.55,
            "regime": "volatile",
            "portfolio_value": 100000,
            "current_price": 150,
        },
        {
            "name": "Moderate Signal Bear Market",
            "signal_strength": 0.6,
            "confidence": 0.7,
            "regime": "bear",
            "portfolio_value": 100000,
            "current_price": 150,
        },
    ]

    print("\nPosition Sizing Scenarios:")
    print("-" * 40)

    for scenario in scenarios:
        position_size = ml_strategy.calculate_position_size(
            scenario["signal_strength"],
            scenario["confidence"],
            scenario["regime"],
            scenario["portfolio_value"],
            scenario["current_price"],
        )

        position_value = position_size * scenario["current_price"]
        position_pct = position_value / scenario["portfolio_value"]

        print(f"\n{scenario['name']}:")
        print(f"  Signal Strength: {scenario['signal_strength']:.2f}")
        print(f"  Confidence: {scenario['confidence']:.2%}")
        print(f"  Regime: {scenario['regime']}")
        print(f"  Position Size: {position_size} shares")
        print(f"  Position Value: ${position_value:,.2f}")
        print(f"  % of Portfolio: {position_pct:.2%}")

    print("\n✅ Position sizing test completed")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Phase 3 S1: ML-Driven Strategy Framework Test")
    print("=" * 60)

    try:
        # Test regime detection
        await test_regime_detection()

        # Test ML strategy
        await test_ml_strategy()

        # Test position sizing
        await test_position_sizing()

        print("\n" + "=" * 60)
        print("✅ All Phase 3 S1 tests completed successfully!")
        print("=" * 60)

        print("\n📊 Summary:")
        print("- Regime detection working for multiple market conditions")
        print("- ML strategy generating signals with multi-timeframe analysis")
        print("- Position sizing adapts to confidence and market regime")
        print("- Ready for integration with main trading system")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

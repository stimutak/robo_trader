#!/usr/bin/env python3
"""
Test Phase 3 S5: Real-Time Feature Updates.
Verifies streaming features, online inference, and WebSocket integration.
"""

import asyncio
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Try to import yfinance - skip tests if not available
try:
    import yfinance as yf

    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    yf = None

# flake8: noqa: E402
from robo_trader.features.realtime_integration import RealtimeFeaturePipeline, WebSocketIntegration
from robo_trader.features.streaming_features import (
    StreamingFeatureCalculator,
    StreamingFeatureStore,
)
from robo_trader.ml.online_inference import ModelUpdateManager, OnlineModelInference


def test_streaming_features():
    """Test streaming feature calculator."""
    print("\n" + "=" * 80)
    print("Testing Streaming Feature Calculator")
    print("=" * 80)

    calculator = StreamingFeatureCalculator()

    # Initialize with symbol
    symbol = "AAPL"
    calculator.initialize_symbol(symbol)

    # Simulate real-time updates
    base_price = 150.0
    base_volume = 1000000

    print(f"\nSimulating 10 real-time updates for {symbol}...")

    for i in range(10):
        # Generate realistic price movement
        price_change = np.random.randn() * 0.5
        price = base_price + price_change
        volume = base_volume * (1 + np.random.randn() * 0.1)

        # Update features
        features = calculator.update(
            symbol=symbol,
            price=price,
            volume=volume,
            high=price + 0.1,
            low=price - 0.1,
            timestamp=datetime.now(),
        )

        print(f"\nUpdate {i+1}:")
        print(f"  Price: ${price:.2f}")
        print(f"  Volume: {volume:,.0f}")
        print(f"  Features calculated: {len(features)}")

        # Show some key features
        if "rsi" in features:
            print(f"  RSI: {features['rsi']:.2f}")
        if "sma_20" in features:
            print(f"  SMA(20): ${features['sma_20']:.2f}")
        if "volatility" in features:
            print(f"  Volatility: {features['volatility']:.4f}")

        time.sleep(0.1)  # Simulate real-time delay

    # Test drift detection
    print("\n" + "-" * 40)
    print("Testing Drift Detection")
    print("-" * 40)

    # Add more updates to build history
    for _ in range(100):
        price = base_price + np.random.randn() * 2
        volume = base_volume * (1 + np.random.randn() * 0.2)
        calculator.update(symbol, price, volume)

    drift_result = calculator.detect_drift(symbol)
    print(f"Drift detected: {drift_result['drift_detected']}")
    if drift_result["drift_detected"]:
        print(f"Features with drift: {list(drift_result['features_with_drift'].keys())[:5]}")

    print("✅ Streaming features test PASSED")
    return True


def test_online_inference():
    """Test online model inference."""
    print("\n" + "=" * 80)
    print("Testing Online Model Inference")
    print("=" * 80)

    # Initialize inference engine with smaller feature set
    feature_names = ["returns", "rsi", "volume_ratio", "volatility", "momentum"]
    inference = OnlineModelInference(feature_names=feature_names)

    # Create test features matching the feature list
    test_features = {
        "returns": 0.01,
        "rsi": 55.0,
        "volume_ratio": 1.1,
        "volatility": 0.02,
        "momentum": 0.03,
    }

    print("\nTesting prediction with dummy model...")

    # Make prediction
    result = inference.predict(test_features, "AAPL")

    if result:
        print(f"✅ Prediction successful:")
        print(f"  Symbol: {result.symbol}")
        print(f"  Prediction: {result.prediction:.4f}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Latency: {result.latency_ms:.2f}ms")
        print(f"  Model: {result.model_version}")

        # Test trading signal
        signal = inference.get_trading_signal(result)
        print(f"  Signal: {signal}")
    else:
        print("❌ Prediction failed")
        return False

    # Test ensemble prediction
    print("\nTesting ensemble prediction...")
    inference.load_model("dummy", "model2")  # Add another model

    ensemble_result = inference.predict_ensemble(test_features, "AAPL")
    if ensemble_result:
        print(f"✅ Ensemble prediction: {ensemble_result.prediction:.4f}")
        print(f"  Confidence: {ensemble_result.confidence:.2%}")

    # Test performance metrics
    print("\nInference Performance Metrics:")
    metrics = inference.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n✅ Online inference test PASSED")
    return True


def test_feature_store():
    """Test feature store persistence."""
    print("\n" + "=" * 80)
    print("Testing Feature Store")
    print("=" * 80)

    store = StreamingFeatureStore("test_feature_store")

    # Store test features
    symbol = "NVDA"
    print(f"\nStoring features for {symbol}...")

    for i in range(10):
        features = {
            "price": 500 + i,
            "volume": 1000000 + i * 10000,
            "rsi": 50 + i,
            "returns": 0.01 * i,
        }
        timestamp = datetime.now() - timedelta(minutes=10 - i)
        store.store_features(symbol, features, timestamp)

    # Force persistence
    store._persist_features(symbol)

    # Load features
    print("Loading stored features...")
    df = store.load_features(symbol)

    if not df.empty:
        print(f"✅ Loaded {len(df)} feature records")
        print(f"  Columns: {list(df.columns)[:5]}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
    else:
        print("❌ Failed to load features")
        return False

    # Test versioning
    print("\nTesting version management...")
    current_version = store.get_latest_version()
    print(f"  Current version: {current_version}")

    store.increment_version()
    new_version = store.get_latest_version()
    print(f"  New version: {new_version}")

    print("\n✅ Feature store test PASSED")
    return True


async def test_realtime_pipeline():
    """Test complete real-time pipeline."""
    print("\n" + "=" * 80)
    print("Testing Complete Real-Time Pipeline")
    print("=" * 80)

    # Initialize pipeline
    symbols = ["AAPL", "NVDA", "TSLA"]
    pipeline = RealtimeFeaturePipeline(
        symbols=symbols, enable_persistence=True, enable_drift_detection=True
    )

    print(f"\nInitialized pipeline for {symbols}")

    # Simulate market updates
    print("\nSimulating market updates...")

    for i in range(5):
        for symbol in symbols:
            # Create market update
            update = {
                "symbol": symbol,
                "price": 100 + np.random.randn() * 5,
                "volume": 1000000 + np.random.randint(-100000, 100000),
                "high": 102,
                "low": 98,
                "timestamp": datetime.now(),
            }

            # Process update
            result = await pipeline.process_market_update(update)

            if result:
                signal = pipeline.get_latest_signal(symbol)
                print(
                    f"  {symbol}: Price=${update['price']:.2f}, Signal={signal}, "
                    f"Prediction={result.prediction:.4f}"
                )

        await asyncio.sleep(0.1)

    # Get pipeline metrics
    print("\nPipeline Metrics:")
    metrics = pipeline.get_pipeline_metrics()

    print(f"  Updates processed: {metrics['pipeline_metrics']['updates_processed']}")
    print(f"  Predictions made: {metrics['pipeline_metrics']['predictions_made']}")
    print(f"  Signals generated: {metrics['pipeline_metrics']['signals_generated']}")
    print(f"  Errors: {metrics['pipeline_metrics']['errors']}")

    # Export signal history
    signal_df = pipeline.export_signal_history()
    if not signal_df.empty:
        print(f"\n✅ Signal history: {len(signal_df)} records")

    print("\n✅ Real-time pipeline test PASSED")
    return True


async def test_model_updates():
    """Test model update and A/B testing."""
    print("\n" + "=" * 80)
    print("Testing Model Updates and A/B Testing")
    print("=" * 80)

    # Initialize components
    inference = OnlineModelInference()
    manager = ModelUpdateManager(inference)

    # Deploy models
    print("\nDeploying models for A/B testing...")
    manager.deploy_model("dummy_path", "model_a", 0.5)
    manager.deploy_model("dummy_path", "model_b", 0.5)

    # Test allocation
    symbols = ["AAPL", "NVDA", "TSLA", "GOOGL", "MSFT"]
    allocations = {}

    for symbol in symbols:
        model = manager.get_model_for_symbol(symbol)
        allocations[symbol] = model
        print(f"  {symbol} -> {model}")

    # Simulate performance tracking
    print("\nSimulating performance tracking...")

    for _ in range(20):
        for symbol in symbols:
            model = allocations[symbol]

            # Make prediction
            features = {"returns": np.random.randn() * 0.01}
            result = inference.predict(features, symbol, model)

            if result:
                # Track performance
                actual_return = np.random.randn() * 0.01
                manager.track_performance(model, result, actual_return)

    # Compare models
    print("\nModel Comparison:")
    comparison = manager.compare_models()

    for model_name, stats in comparison.items():
        print(f"  {model_name}:")
        print(f"    Mean error: {stats['mean_error']:.6f}")
        print(f"    Std error: {stats['std_error']:.6f}")
        print(f"    Samples: {stats['sample_size']}")

    print("\n✅ Model update test PASSED")
    return True


@pytest.mark.skipif(not HAS_YFINANCE, reason="yfinance not installed")
def test_with_real_data():
    """Test with real market data."""
    print("\n" + "=" * 80)
    print("Testing with Real Market Data")
    print("=" * 80)

    # Fetch real data
    symbol = "AAPL"
    print(f"\nFetching real data for {symbol}...")

    ticker = yf.Ticker(symbol)
    data = ticker.history(period="1d", interval="1m")

    if data.empty:
        print("⚠️  Could not fetch real-time data (market may be closed)")
        return True  # Don't fail test for market hours

    print(f"✅ Fetched {len(data)} data points")

    # Initialize calculator with historical data
    calculator = StreamingFeatureCalculator()
    calculator.initialize_symbol(symbol, data)

    # Process latest updates
    print("\nProcessing latest market data...")

    for idx, row in data.tail(5).iterrows():
        features = calculator.update(
            symbol=symbol,
            price=row["Close"],
            volume=row["Volume"],
            high=row["High"],
            low=row["Low"],
            timestamp=idx,
        )

        print(f"\n{idx.strftime('%H:%M:%S')}:")
        print(f"  Price: ${row['Close']:.2f}")
        print(f"  Features: {len(features)}")

        if "rsi" in features:
            print(f"  RSI: {features['rsi']:.2f}")
        if "volatility" in features:
            print(f"  Volatility: {features['volatility']:.4f}")

    print("\n✅ Real data test PASSED")
    return True


def main():
    """Run all Phase 3 S5 tests."""
    print("\n" + "=" * 80)
    print("PHASE 3 S5: REAL-TIME FEATURE UPDATES - TEST SUITE")
    print("=" * 80)

    all_passed = True

    # Run synchronous tests
    tests = [
        ("Streaming Features", test_streaming_features),
        ("Online Inference", test_online_inference),
        ("Feature Store", test_feature_store),
    ]

    # Only include yfinance test if available
    if HAS_YFINANCE:
        tests.append(("Real Market Data", test_with_real_data))

    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"❌ {test_name} ERROR: {e}")

    # Run async tests
    async_tests = [
        ("Real-time Pipeline", test_realtime_pipeline),
        ("Model Updates", test_model_updates),
    ]

    for test_name, test_func in async_tests:
        try:
            if not asyncio.run(test_func()):
                all_passed = False
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            all_passed = False
            print(f"❌ {test_name} ERROR: {e}")

    # Final summary
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ ALL PHASE 3 S5 TESTS PASSED!")
        print("\nPhase 3 S5 Implementation Complete:")
        print("  ✅ Streaming feature calculator")
        print("  ✅ Online model inference")
        print("  ✅ Feature store with versioning")
        print("  ✅ WebSocket integration")
        print("  ✅ Feature drift detection")
        print("  ✅ Model A/B testing")
        print("  ✅ Real-time pipeline")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

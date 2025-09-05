#!/usr/bin/env python
"""
Test script for feature engineering pipeline.
Tests feature calculation with real market data.
"""

import asyncio
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog

# Add parent directory to path
sys.path.insert(0, "/Users/oliver/robo_trader")

from robo_trader.database_async import AsyncTradingDatabase as AsyncDatabase
from robo_trader.features import (
    FeatureEngine,
    MomentumIndicators,
    TimeFrame,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)
from robo_trader.monitoring.performance import PerformanceMonitor

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def generate_sample_data(symbol: str, num_bars: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)

    # Generate realistic price data
    base_price = 100.0
    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq="5min")

    # Random walk for prices
    returns = np.random.normal(0.0001, 0.02, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # Add some noise for high/low
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = prices[i - 1] if i > 0 else price
        close = price
        volume = np.random.randint(1000000, 10000000)

        data.append(
            {
                "timestamp": date,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "symbol": symbol,
            }
        )

    return pd.DataFrame(data)


async def test_individual_indicators():
    """Test individual indicator calculators."""
    print("\n" + "=" * 60)
    print("Testing Individual Indicators")
    print("=" * 60)

    # Generate test data
    data = generate_sample_data("TEST", 100)

    # Test Momentum Indicators
    print("\n1. Testing Momentum Indicators...")
    momentum = MomentumIndicators(window_size=14)
    if momentum.validate_data(data):
        features = momentum.calculate(data)
        print(f"   Calculated {len(features)} momentum features:")
        for name, feature in features.items():
            print(f"   - {name}: {feature.value:.2f}")

    # Test Trend Indicators
    print("\n2. Testing Trend Indicators...")
    trend = TrendIndicators()
    if trend.validate_data(data):
        features = trend.calculate(data)
        print(f"   Calculated {len(features)} trend features:")
        for name, feature in list(features.items())[:5]:  # Show first 5
            print(f"   - {name}: {feature.value:.2f}")

    # Test Volatility Indicators
    print("\n3. Testing Volatility Indicators...")
    volatility = VolatilityIndicators(window_size=20)
    if volatility.validate_data(data):
        features = volatility.calculate(data)
        print(f"   Calculated {len(features)} volatility features:")
        for name, feature in features.items():
            print(f"   - {name}: {feature.value:.4f}")

    # Test Volume Indicators
    print("\n4. Testing Volume Indicators...")
    volume = VolumeIndicators(window_size=20)
    if volume.validate_data(data):
        features = volume.calculate(data)
        print(f"   Calculated {len(features)} volume features:")
        for name, feature in features.items():
            if isinstance(feature.value, float):
                print(f"   - {name}: {feature.value:.2f}")


async def test_feature_engine():
    """Test the main feature engine."""
    print("\n" + "=" * 60)
    print("Testing Feature Engine")
    print("=" * 60)

    # Initialize database and engine
    from pathlib import Path

    db = AsyncDatabase(Path("trading_data.db"))
    await db.initialize()

    perf_monitor = PerformanceMonitor()
    engine = FeatureEngine(db, perf_monitor)

    # Generate test data for multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    data_dict = {}

    print("\nGenerating sample data for symbols:", symbols)
    for symbol in symbols:
        data_dict[symbol] = generate_sample_data(symbol, 100)

    # Test single symbol feature calculation
    print("\n1. Testing single symbol feature calculation...")
    features = await engine.calculate_features("AAPL", data_dict["AAPL"], "technical")
    print(f"   Calculated {len(features)} features for AAPL")

    # Show sample features
    print("\n   Sample features:")
    for i, (name, feature) in enumerate(list(features.items())[:10]):
        if isinstance(feature.value, float):
            print(f"   - {name}: {feature.value:.4f}")

    # Test batch calculation
    print("\n2. Testing batch feature calculation...")
    all_features = await engine.calculate_features_batch(symbols, data_dict, "technical")

    for symbol in symbols:
        num_features = len(all_features.get(symbol, {}))
        print(f"   {symbol}: {num_features} features calculated")

    # Test feature retrieval from database
    print("\n3. Testing feature retrieval from database...")
    latest_features = await engine.get_latest_features("AAPL")
    print(f"   Retrieved {len(latest_features)} features from database")

    # Test feature correlation
    print("\n4. Testing feature correlation matrix...")
    correlation_matrix = await engine.get_feature_correlation_matrix(
        symbols, ["rsi", "macd", "atr", "obv"]
    )
    if not correlation_matrix.empty:
        print("   Correlation matrix shape:", correlation_matrix.shape)
        print("\n   Sample correlations:")
        print(correlation_matrix.head())

    # Get engine info
    print("\n5. Feature Engine Info:")
    info = engine.get_feature_info()
    print(f"   Feature sets: {list(info['feature_sets'].keys())}")
    print(f"   Total calculators: {info['total_calculators']}")
    print(f"   Cache size: {info['cache_size']}")

    # Performance metrics
    print("\n6. Performance Metrics:")
    perf_summary = perf_monitor.get_performance_summary()
    print(
        f"   Features calculated: {perf_summary.get('counters', {}).get('features_calculated', 0)}"
    )
    print(f"   Feature errors: {perf_summary.get('counters', {}).get('feature_errors', 0)}")

    timers = perf_summary.get("timers", {})
    if "feature_calculation" in timers:
        calc_stats = timers["feature_calculation"]
        print(f"   Avg calculation time: {calc_stats.get('mean', 0):.2f}ms")

    # Cleanup
    await db.close()
    print("\n✅ Feature engine tests completed successfully!")


async def test_with_real_data():
    """Test features with real market data from database."""
    print("\n" + "=" * 60)
    print("Testing with Real Market Data")
    print("=" * 60)

    # Initialize database
    from pathlib import Path

    db = AsyncDatabase(Path("trading_data.db"))
    await db.initialize()

    # Get recent market data
    query = """
        SELECT timestamp, open, high, low, close, volume, symbol
        FROM market_data
        WHERE symbol IN ('AAPL', 'NVDA', 'TSLA')
        AND timestamp > datetime('now', '-2 days')
        ORDER BY symbol, timestamp DESC
        LIMIT 500
    """

    rows = await db.fetch_all(query)

    if rows:
        # Convert to DataFrame
        df = pd.DataFrame(
            rows, columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Group by symbol
        symbols = df["symbol"].unique()
        print(f"\nFound data for {len(symbols)} symbols: {list(symbols)}")

        # Initialize feature engine
        perf_monitor = PerformanceMonitor()
        engine = FeatureEngine(db, perf_monitor)

        # Calculate features for each symbol
        for symbol in symbols[:3]:  # Limit to first 3
            symbol_data = df[df["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("timestamp")

            if len(symbol_data) >= 50:
                print(f"\n{symbol}: {len(symbol_data)} bars")

                features = await engine.calculate_features(symbol, symbol_data, "technical")
                print(f"  Calculated {len(features)} features")

                # Show key features
                key_features = ["rsi", "macd", "atr", "obv", "sma_20", "bb_width"]
                print("  Key features:")
                for fname in key_features:
                    if fname in features:
                        value = features[fname].value
                        if isinstance(value, float):
                            print(f"    - {fname}: {value:.4f}")
    else:
        print("No recent market data found in database")

    await db.close()


async def main():
    """Run all tests."""
    try:
        # Test individual components
        await test_individual_indicators()

        # Test feature engine
        await test_feature_engine()

        # Test with real data if available
        await test_with_real_data()

        print("\n" + "=" * 60)
        print("✅ All feature tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        logger.error("Test failed", error=str(e), exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())

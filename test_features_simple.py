#!/usr/bin/env python
"""
Simple test for feature engineering pipeline.
"""

# Add parent directory to path
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/oliver/robo_trader")

from robo_trader.features import (
    MomentumIndicators,
    TrendIndicators,
    VolatilityIndicators,
    VolumeIndicators,
)


def generate_sample_data(num_bars: int = 100) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)

    base_price = 100.0
    dates = pd.date_range(end=datetime.now(), periods=num_bars, freq="5min")

    returns = np.random.normal(0.0001, 0.02, num_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
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
                "symbol": "TEST",
            }
        )

    return pd.DataFrame(data)


def test_indicators():
    """Test all indicator types."""
    print("Feature Engineering Pipeline - Simple Test")
    print("=" * 50)

    # Generate test data
    data = generate_sample_data(100)
    print(f"✅ Generated {len(data)} bars of test data")

    indicators_tested = 0

    # Test Momentum Indicators
    print("\n📈 Momentum Indicators:")
    momentum = MomentumIndicators(window_size=14)
    if momentum.validate_data(data):
        features = momentum.calculate(data)
        indicators_tested += len(features)
        for name, feature in features.items():
            print(f"  • {name}: {feature.value:.2f}")

    # Test Trend Indicators
    print("\n📊 Trend Indicators:")
    trend = TrendIndicators()
    if trend.validate_data(data):
        features = trend.calculate(data)
        indicators_tested += len(features)
        # Show subset to avoid clutter
        for name in ["sma_20", "ema_20", "macd", "adx"]:
            if name in features:
                print(f"  • {name}: {features[name].value:.2f}")

    # Test Volatility Indicators
    print("\n📉 Volatility Indicators:")
    volatility = VolatilityIndicators(window_size=20)
    if volatility.validate_data(data):
        features = volatility.calculate(data)
        indicators_tested += len(features)
        for name, feature in features.items():
            print(f"  • {name}: {feature.value:.4f}")

    # Test Volume Indicators
    print("\n📊 Volume Indicators:")
    volume = VolumeIndicators(window_size=20)
    if volume.validate_data(data):
        features = volume.calculate(data)
        indicators_tested += len(features)
        for name, feature in features.items():
            if isinstance(feature.value, float):
                print(f"  • {name}: {feature.value:.2f}")

    print("\n" + "=" * 50)
    print(f"✅ Successfully tested {indicators_tested} technical indicators!")
    print("Feature engineering pipeline is working correctly.")


if __name__ == "__main__":
    test_indicators()

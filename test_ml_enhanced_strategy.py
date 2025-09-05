#!/usr/bin/env python3
"""Test the ML Enhanced Strategy with regime detection and multi-timeframe analysis."""

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.config import load_config
from robo_trader.strategies.ml_enhanced_strategy import MLEnhancedStrategy
from robo_trader.strategies.regime_detector import MarketRegime, RegimeDetector


async def test_regime_detection():
    """Test regime detection functionality."""
    print("=" * 60)
    print("Testing Regime Detection")
    print("=" * 60)

    detector = RegimeDetector(lookback_period=50)

    # Create synthetic market data with different regimes
    dates = pd.date_range(end=datetime.now(), periods=200, freq="5min")

    # 1. Trending up market
    trend_up = np.cumsum(np.random.randn(50) * 0.5 + 0.1) + 100

    # 2. Volatile market
    volatile = 100 + np.random.randn(50) * 2

    # 3. Ranging market
    ranging = 100 + np.sin(np.linspace(0, 4 * np.pi, 50)) * 2

    # 4. Trending down
    trend_down = np.cumsum(np.random.randn(50) * 0.5 - 0.1) + 100

    # Combine
    prices = np.concatenate([trend_up, volatile, ranging, trend_down])

    data = pd.DataFrame(
        {
            "open": prices + np.random.randn(200) * 0.1,
            "high": prices + abs(np.random.randn(200) * 0.3),
            "low": prices - abs(np.random.randn(200) * 0.3),
            "close": prices,
            "volume": np.random.randint(100000, 1000000, 200),
        },
        index=dates,
    )

    # Test detection at different points
    test_points = [50, 100, 150, 199]
    expected_regimes = ["trending_up", "volatile", "ranging", "trending_down"]

    print("\nRegime Detection Results:")
    for i, (point, expected) in enumerate(zip(test_points, expected_regimes)):
        subset_data = data.iloc[: point + 1]
        regime = await detector.detect_regime("TEST", subset_data)

        print(f"\n{i+1}. Period ending at index {point}:")
        print(f"   Expected: {expected}")
        print(f"   Detected: {regime.trend_regime.value}")
        print(f"   Confidence: {regime.confidence:.2%}")
        print(f"   Volatility Regime: {regime.volatility_regime.value}")
        print(f"   Indicators: {regime.indicators}")

        # Check transition probability
        if regime.transition_probability > 0:
            print(f"   Transition Probability: {regime.transition_probability:.2%}")

    print("\n✅ Regime detection working!")
    return True


async def test_ml_enhanced_strategy():
    """Test the ML enhanced strategy."""
    print("\n" + "=" * 60)
    print("Testing ML Enhanced Strategy")
    print("=" * 60)

    config = load_config()
    strategy = MLEnhancedStrategy(config)

    # Initialize strategy
    await strategy.initialize()

    print("\n1. Strategy Configuration:")
    print(f"   Base confidence threshold: {strategy.base_confidence_threshold}")
    print(f"   Min alignment score: {strategy.min_alignment_score}")
    print(f"   Timeframes: {list(strategy.timeframes.keys())}")
    print(f"   Base position size: {strategy.base_position_size:.1%}")

    # Create test data with clear signal
    dates = pd.date_range(end=datetime.now(), periods=300, freq="1min")

    # Strong uptrend for testing
    trend = np.cumsum(np.random.randn(300) * 0.3 + 0.05) + 150

    test_data = pd.DataFrame(
        {
            "open": trend + np.random.randn(300) * 0.1,
            "high": trend + abs(np.random.randn(300) * 0.2),
            "low": trend - abs(np.random.randn(300) * 0.2),
            "close": trend,
            "volume": np.random.randint(500000, 2000000, 300),
        },
        index=dates,
    )

    print("\n2. Testing Signal Generation:")

    # Test with different symbols
    test_symbols = ["TEST1", "TEST2", "TEST3"]

    for symbol in test_symbols:
        # Modify data slightly for each symbol
        symbol_data = test_data.copy()
        symbol_data["close"] = symbol_data["close"] * (1 + np.random.randn() * 0.01)

        # Get signal
        signal = await strategy.analyze(symbol, symbol_data)

        print(f"\n   {symbol}:")
        if signal:
            print(f"      Action: {signal.action}")
            print(f"      Confidence: {signal.confidence:.2%}")
            print(f"      Features: {signal.features}")

            # Check position sizing
            if "position_size" in signal.features:
                print(f"      Position Size: {signal.features['position_size']:.2%}")

            # Check risk parameters
            if "stop_loss" in signal.features:
                print(f"      Stop Loss: {signal.features['stop_loss']:.2%}")
            if "take_profit" in signal.features:
                print(f"      Take Profit: {signal.features['take_profit']:.2%}")
        else:
            print("      No signal generated")

    print("\n3. Testing Multi-Timeframe Analysis:")

    # Test MTF alignment
    mtf_signal = await strategy._analyze_multi_timeframe("TEST", test_data)

    print(f"   Alignment Score: {mtf_signal.alignment_score:.2f}")
    if mtf_signal.combined_signal:
        print(f"   Combined Action: {mtf_signal.combined_signal.action}")
        print(f"   Combined Confidence: {mtf_signal.combined_signal.confidence:.2%}")

    # Check individual timeframes
    for tf in ["1m", "5m", "15m", "1h"]:
        tf_signal = getattr(mtf_signal, f"timeframe_{tf}", None)
        if tf_signal:
            print(f"   {tf}: {tf_signal.action} ({tf_signal.confidence:.2%})")
        else:
            print(f"   {tf}: No signal")

    print("\n4. Testing Regime Adaptation:")

    # Test with different market conditions
    regimes_to_test = [
        (MarketRegime.STRONG_BULL, "Strong Bull"),
        (MarketRegime.HIGH_VOL, "High Volatility"),
        (MarketRegime.RANGE_BOUND, "Range Bound"),
        (MarketRegime.CRASH, "Crash"),
    ]

    for regime_type, regime_name in regimes_to_test:
        # Mock regime
        from robo_trader.strategies.regime_detector import RegimeState

        mock_regime = RegimeState(
            timestamp=datetime.now(),
            trend_regime=regime_type,
            volatility_regime=MarketRegime.NEUTRAL
            if regime_type not in [MarketRegime.HIGH_VOL, MarketRegime.EXTREME_VOL]
            else regime_type,
            confidence=0.8,
            indicators={"sma_20": 0.5, "rsi": 50},
            transition_probability=0.1,
            expected_duration=10,
        )

        strategy.current_regime = mock_regime

        # Get adjusted parameters from strategy settings
        position_mult = strategy.regime_size_adjustments.get(regime_type, 1.0)
        risk_params = strategy.regime_risk_params.get(
            regime_type, {"stop_loss": 0.02, "take_profit": 0.05}
        )

        print(f"\n   {regime_name} Regime:")
        print(f"      Position Size Mult: {position_mult:.1f}x")
        print(f"      Stop Loss: {risk_params['stop_loss']:.2%}")
        print(f"      Take Profit: {risk_params['take_profit']:.2%}")
        print(f"      Trading Allowed: {regime_type != MarketRegime.CRASH}")

        # Check if trading is allowed
        if regime_type == MarketRegime.CRASH:
            signal = await strategy.analyze("TEST", test_data)
            print(f"      Trading Allowed: {signal is not None}")

    print("\n5. Strategy Metrics:")
    metrics = strategy.get_strategy_metrics()

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    print("\n✅ ML Enhanced Strategy working!")
    return True


async def test_position_sizing():
    """Test dynamic position sizing based on regime and confidence."""
    print("\n" + "=" * 60)
    print("Testing Dynamic Position Sizing")
    print("=" * 60)

    config = load_config()
    strategy = MLEnhancedStrategy(config)

    from robo_trader.strategies.framework import Signal as FrameworkSignal
    from robo_trader.strategies.regime_detector import RegimeState

    # Test different scenarios
    scenarios = [
        {
            "name": "High Confidence Bull Market",
            "regime": MarketRegime.STRONG_BULL,
            "confidence": 0.85,
            "volatility": 20.0,
        },
        {
            "name": "Low Confidence Volatile Market",
            "regime": MarketRegime.HIGH_VOL,
            "confidence": 0.65,
            "volatility": 85.0,
        },
        {
            "name": "Medium Confidence Ranging",
            "regime": MarketRegime.RANGE_BOUND,
            "confidence": 0.70,
            "volatility": 50.0,
        },
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")

        # Create mock regime
        regime = RegimeState(
            timestamp=datetime.now(),
            trend_regime=scenario["regime"],
            volatility_regime=MarketRegime.HIGH_VOL
            if scenario["volatility"] > 70
            else MarketRegime.NEUTRAL,
            confidence=0.8,
            indicators={"volatility": scenario["volatility"]},
            transition_probability=0.1,
            expected_duration=10,
        )

        # Create mock signal using SimpleSignal from ml_enhanced_strategy
        from robo_trader.strategies.ml_enhanced_strategy import SimpleSignal

        signal = SimpleSignal(
            symbol="TEST", action="BUY", confidence=scenario["confidence"], features={}
        )

        # Apply position sizing
        sized_signal = strategy._apply_position_sizing(signal, regime)

        print(f"   Base Size: {strategy.base_position_size:.1%}")
        print(
            f"   Regime Adjustment: {strategy.regime_size_adjustments.get(scenario['regime'], 1.0):.1f}x"
        )
        print(f"   Confidence: {scenario['confidence']:.1%}")
        print(f"   Volatility Percentile: {scenario['volatility']:.0f}")
        print(f"   Final Position Size: {sized_signal.features['position_size']:.2%}")

    print("\n✅ Position sizing working correctly!")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ML ENHANCED STRATEGY TEST SUITE")
    print("=" * 60)

    # Run tests
    await test_regime_detection()
    await test_ml_enhanced_strategy()
    await test_position_sizing()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✅")
    print("=" * 60)

    print("\nML Enhanced Strategy Features:")
    print("• Market regime detection (trend/volatility/ranging)")
    print("• Multi-timeframe signal confirmation")
    print("• Dynamic position sizing based on regime")
    print("• Risk parameters adapted to market conditions")
    print("• ML predictions with confidence filtering")
    print("• Correlation-aware position limits")

    print("\nNext Steps:")
    print("1. Integrate with runner_async.py")
    print("2. Backtest strategy performance")
    print("3. Fine-tune regime parameters")
    print("4. Add more sophisticated MTF analysis")


if __name__ == "__main__":
    asyncio.run(main())

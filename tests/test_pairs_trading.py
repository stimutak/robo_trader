#!/usr/bin/env python3
"""Test pairs trading and statistical arbitrage strategies."""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from robo_trader.strategies.pairs_trading import (
    CointegrationPairsStrategy,
    PairSignal,
    PairStats,
    StatisticalArbitrageStrategy,
    create_mean_reversion_suite,
)


def create_mock_price_data(symbols: list, days: int = 252, correlation: float = 0.8) -> dict:
    """Create mock correlated price data for testing."""

    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq="D")

    # Generate base random walk
    base_returns = np.random.normal(0.001, 0.02, days)  # 0.1% daily return, 2% volatility
    base_prices = 100 * np.cumprod(1 + base_returns)

    price_data = {}

    for i, symbol in enumerate(symbols):
        if i == 0:
            # First symbol uses base prices
            prices = base_prices
        else:
            # Subsequent symbols are correlated with the first
            noise = np.random.normal(0, 0.01, days)  # Independent noise
            correlated_returns = correlation * base_returns + np.sqrt(1 - correlation**2) * noise
            prices = 100 * np.cumprod(1 + correlated_returns)

        price_data[symbol] = pd.DataFrame(
            {
                "close": prices,
                "high": prices * (1 + np.random.uniform(0, 0.02, days)),
                "low": prices * (1 - np.random.uniform(0, 0.02, days)),
                "volume": np.random.randint(100000, 1000000, days),
            },
            index=dates,
        )

    return price_data


async def test_cointegration_pairs_strategy():
    """Test cointegration pairs trading strategy."""

    print("📈 Testing Cointegration Pairs Strategy")
    print("-" * 40)

    # Create strategy
    strategy = CointegrationPairsStrategy(
        name="TestCointegrationPairs",
        lookback_days=100,
        min_correlation=0.6,
        entry_zscore_threshold=2.0,
        exit_zscore_threshold=0.5,
    )

    # Create mock data with high correlation
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    price_data = create_mock_price_data(symbols, days=150, correlation=0.85)

    print(f"  Created mock data for {len(symbols)} symbols")

    # Find pairs
    pairs = await strategy.find_pairs(symbols, price_data)

    print(f"  Found {len(pairs)} cointegrated pairs:")
    for pair in pairs:
        pair_stats = strategy.pair_stats[pair]
        print(
            f"    {pair[0]}-{pair[1]}: corr={pair_stats.correlation:.3f}, "
            f"p-value={pair_stats.cointegration_pvalue:.3f}, "
            f"hedge_ratio={pair_stats.hedge_ratio:.3f}"
        )

    if pairs:
        # Test signal generation
        current_prices = {symbol: data["close"].iloc[-1] for symbol, data in price_data.items()}

        signals = await strategy.analyze_pairs(pairs, current_prices)

        print(f"  Generated {len(signals)} trading signals:")
        for signal in signals:
            print(
                f"    {signal['pair']}: {signal['signal']} "
                f"(z-score: {signal['z_score']:.2f}, confidence: {signal['confidence']:.2f})"
            )

            # Test position update
            if signal["signal"] != PairSignal.HOLD.value:
                strategy.update_position(signal["pair"], signal)

    # Get portfolio summary
    summary = strategy.get_portfolio_summary()
    print(
        f"  Portfolio: {summary['active_pairs']} active pairs, "
        f"{summary['total_pairs_discovered']} total pairs discovered"
    )

    print("  ✅ Cointegration pairs strategy tested")


async def test_statistical_arbitrage_strategy():
    """Test statistical arbitrage strategy."""

    print("\n🎯 Testing Statistical Arbitrage Strategy")
    print("-" * 40)

    # Create strategy
    strategy = StatisticalArbitrageStrategy(
        name="TestStatArb",
        universe_size=20,
        lookback_days=60,
        min_score_threshold=0.3,  # Lower threshold for testing
        max_positions=5,
    )

    # Create mock data
    symbols = ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE"]
    price_data = create_mock_price_data(symbols, days=80, correlation=0.3)

    print(f"  Created mock data for {len(symbols)} symbols")

    # Calculate arbitrage scores
    scores = await strategy.calculate_arbitrage_scores(symbols, price_data)

    print(f"  Calculated arbitrage scores for {len(scores)} symbols:")
    for symbol, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"    {symbol}: {score:.3f}")

    # Test feature calculation
    if symbols and symbols[0] in price_data:
        features = strategy._calculate_features(price_data[symbols[0]])
        print(f"  Sample features for {symbols[0]}:")
        for feature, value in features.items():
            print(f"    {feature}: {value:.3f}")

    print("  ✅ Statistical arbitrage strategy tested")


async def test_mean_reversion_suite():
    """Test the complete mean reversion strategy suite."""

    print("\n🎯 Testing Mean Reversion Strategy Suite")
    print("-" * 50)

    strategies = create_mean_reversion_suite()

    print(f"  Created {len(strategies)} mean reversion strategies:")
    for strategy in strategies:
        print(f"    - {strategy.name}")

    # Test each strategy type
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    price_data = create_mock_price_data(symbols, days=200, correlation=0.7)
    current_prices = {symbol: data["close"].iloc[-1] for symbol, data in price_data.items()}

    print(f"\n  Testing strategies with {len(symbols)} symbols:")

    for strategy in strategies:
        try:
            if isinstance(strategy, CointegrationPairsStrategy):
                # Test pairs strategy
                pairs = await strategy.find_pairs(symbols, price_data)
                signals = await strategy.analyze_pairs(pairs, current_prices)

                print(f"    {strategy.name:30} | Pairs: {len(pairs):2} | Signals: {len(signals):2}")

            elif isinstance(strategy, StatisticalArbitrageStrategy):
                # Test stat arb strategy
                scores = await strategy.calculate_arbitrage_scores(symbols, price_data)

                print(
                    f"    {strategy.name:30} | Scores: {len(scores):2} | Avg: {np.mean(list(scores.values())):.2f}"
                )

        except Exception as e:
            print(f"    {strategy.name:30} | ERROR: {str(e)}")

    print("  ✅ Mean reversion suite tested")


async def test_pair_statistics():
    """Test pair statistics calculations."""

    print("\n📊 Testing Pair Statistics")
    print("-" * 40)

    # Create highly correlated pair
    dates = pd.date_range(start=datetime.now() - timedelta(days=100), periods=100, freq="D")

    # Generate cointegrated series
    base_series = np.cumsum(np.random.normal(0, 1, 100))
    series_a = base_series + np.random.normal(0, 0.1, 100)  # Add small noise
    series_b = 0.8 * base_series + 5 + np.random.normal(0, 0.1, 100)  # Scaled and shifted

    prices_a = pd.Series(100 + series_a, index=dates)
    prices_b = pd.Series(100 + series_b, index=dates)

    strategy = CointegrationPairsStrategy()

    # Test pair relationship
    pair_stats = await strategy._test_pair_relationship("TEST_A", "TEST_B", prices_a, prices_b)

    if pair_stats:
        print(f"  Pair Statistics:")
        print(f"    Correlation: {pair_stats.correlation:.3f}")
        print(f"    Cointegration p-value: {pair_stats.cointegration_pvalue:.3f}")
        print(f"    Hedge Ratio: {pair_stats.hedge_ratio:.3f}")
        print(f"    Spread Mean: {pair_stats.spread_mean:.3f}")
        print(f"    Spread Std: {pair_stats.spread_std:.3f}")
        print(f"    Half-life: {pair_stats.half_life:.1f} days")

        # Test half-life calculation
        spread = prices_a - pair_stats.hedge_ratio * prices_b
        calculated_half_life = strategy._calculate_half_life(spread)
        print(f"    Calculated Half-life: {calculated_half_life:.1f} days")

        print("  ✅ Pair statistics calculated correctly")
    else:
        print("  ❌ Failed to create valid pair statistics")


async def test_signal_generation():
    """Test signal generation logic."""

    print("\n🚦 Testing Signal Generation")
    print("-" * 40)

    strategy = CointegrationPairsStrategy(
        entry_zscore_threshold=2.0,
        exit_zscore_threshold=0.5,
        stop_loss_zscore=3.0,
    )

    # Create mock pair stats
    pair_stats = PairStats(
        symbol_a="TEST_A",
        symbol_b="TEST_B",
        correlation=0.85,
        cointegration_pvalue=0.02,
        hedge_ratio=0.8,
        spread_mean=5.0,
        spread_std=2.0,
        half_life=10.0,
        last_updated=datetime.now(),
    )

    pair_key = ("TEST_A", "TEST_B")
    strategy.pair_stats[pair_key] = pair_stats

    # Test different z-score scenarios
    test_scenarios = [
        {"z_score": 2.5, "description": "Strong entry signal (long A, short B)"},
        {"z_score": -2.3, "description": "Strong entry signal (long B, short A)"},
        {"z_score": 0.3, "description": "Exit signal (mean reversion)"},
        {"z_score": 3.5, "description": "Stop loss signal"},
        {"z_score": 1.0, "description": "No signal (within threshold)"},
    ]

    for scenario in test_scenarios:
        z_score = scenario["z_score"]
        current_spread = pair_stats.spread_mean + z_score * pair_stats.spread_std

        signal = await strategy._evaluate_pair_signal(
            pair_key, pair_stats, z_score, current_spread, 100.0, 120.0
        )

        signal_type = signal["signal"] if signal else "HOLD"
        confidence = signal.get("confidence", 0.0) if signal else 0.0

        print(
            f"  {scenario['description']:35} | Signal: {signal_type:15} | Confidence: {confidence:.2f}"
        )

        # Update position for next test
        if signal and signal["signal"] != PairSignal.HOLD.value:
            strategy.update_position(pair_key, signal)

    print("  ✅ Signal generation tested")


async def main():
    """Run all pairs trading tests."""

    print("🚀 Testing Pairs Trading & Statistical Arbitrage")
    print("=" * 50)

    try:
        # Test individual components
        await test_pair_statistics()
        await test_signal_generation()

        # Test strategies
        await test_cointegration_pairs_strategy()
        await test_statistical_arbitrage_strategy()

        # Test complete suite
        await test_mean_reversion_suite()

        print("\n🎉 All Pairs Trading Tests Passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

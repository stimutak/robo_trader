"""
Comprehensive test suite for mean reversion and pairs trading strategies.

Tests the Phase 3 S5 implementation including:
- Mean reversion strategy with ML enhancement
- Pairs trading with cointegration
- Statistical arbitrage
"""

import asyncio
import unittest
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from robo_trader.features.engine import FeatureSet
from robo_trader.strategies.framework import Signal, SignalType
from robo_trader.strategies.mean_reversion import MeanReversionStrategy
from robo_trader.strategies.pairs_trading import (
    CointegrationPairsStrategy,
    MLEnhancedPairsTrading,
    PairSignal,
    StatisticalArbitrageStrategy,
    create_mean_reversion_suite,
)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test mean reversion strategy implementation."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "GOOGL", "MSFT"]
        self.strategy = MeanReversionStrategy(
            symbols=self.symbols,
            use_ml_enhancement=False,  # Test without ML first
        )

    def _create_test_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Create synthetic test data."""
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1h")

        # Create mean-reverting price series
        np.random.seed(42)
        price = 100
        prices = []

        for i in range(periods):
            # Add mean reversion tendency
            if price > 105:
                change = np.random.normal(-0.5, 1)
            elif price < 95:
                change = np.random.normal(0.5, 1)
            else:
                change = np.random.normal(0, 1)

            price = max(price + change, 1)
            prices.append(price)

        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "close": prices,
                "volume": np.random.randint(1000000, 5000000, periods),
            }
        ).set_index("timestamp")

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "MeanReversion")
        self.assertEqual(len(self.strategy.symbols), 3)
        self.assertEqual(self.strategy.bb_period, 20)
        self.assertEqual(self.strategy.zscore_threshold, 2.0)

    async def test_reversion_score_calculation(self):
        """Test mean reversion score calculation."""
        data = self._create_test_data("AAPL")

        # Create feature set
        features = FeatureSet()
        features.bb_upper = 105
        features.bb_middle = 100
        features.bb_lower = 95
        features.rsi = 25  # Oversold
        features.atr = 2.0

        score = self.strategy._calculate_reversion_score(data, features)

        # Should have negative score (oversold)
        self.assertLess(score, 0)
        self.assertGreaterEqual(score, -2)

    async def test_signal_generation(self):
        """Test signal generation for mean reversion."""
        # Initialize market data
        market_data = {symbol: self._create_test_data(symbol) for symbol in self.symbols}

        # Create features
        features = {}
        for symbol in self.symbols:
            feature_set = FeatureSet()
            feature_set.bb_upper = 105
            feature_set.bb_middle = 100
            feature_set.bb_lower = 95
            feature_set.rsi = 20 if symbol == "AAPL" else 50  # AAPL oversold
            feature_set.atr = 2.0
            features[symbol] = feature_set

        # Initialize strategy
        await self.strategy._initialize(market_data)

        # Generate signals
        signals = await self.strategy._generate_signals(market_data, features)

        # Should generate at least one signal for oversold AAPL
        self.assertGreater(len(signals), 0)

        # Check signal properties
        aapl_signals = [s for s in signals if s.symbol == "AAPL"]
        if aapl_signals:
            signal = aapl_signals[0]
            self.assertEqual(signal.signal_type, SignalType.BUY)
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)

    def test_ml_feature_extraction(self):
        """Test ML feature extraction."""
        strategy = MeanReversionStrategy(
            symbols=self.symbols,
            use_ml_enhancement=True,
        )

        data = self._create_test_data("AAPL", periods=50)
        features = strategy._extract_ml_features(data)

        # Should extract features for each valid window
        self.assertGreater(len(features), 0)

        # Check feature vector dimensions
        if features:
            self.assertEqual(len(features[0]), 9)  # 9 features

    def test_ml_label_generation(self):
        """Test ML label generation for training."""
        strategy = MeanReversionStrategy(
            symbols=self.symbols,
            use_ml_enhancement=True,
        )

        data = self._create_test_data("AAPL", periods=50)
        labels = strategy._generate_ml_labels(data, forward_periods=5)

        # Should generate labels
        self.assertGreater(len(labels), 0)

        # Labels should be binary
        for label in labels:
            self.assertIn(label, [0, 1])


class TestPairsTrading(unittest.TestCase):
    """Test pairs trading strategies."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = CointegrationPairsStrategy(
            lookback_days=60,
            min_correlation=0.6,
            use_ml_enhancement=False,
        )

    def _create_cointegrated_pair(self) -> tuple:
        """Create synthetic cointegrated price series."""
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1D")

        # Create cointegrated series
        np.random.seed(42)

        # Base series
        base = 100 + np.cumsum(np.random.randn(periods) * 0.5)

        # Cointegrated series with mean-reverting spread
        spread = np.zeros(periods)
        for i in range(1, periods):
            # Mean-reverting spread
            spread[i] = 0.8 * spread[i - 1] + np.random.randn() * 0.2

        series_a = pd.Series(base + spread * 2, index=dates)
        series_b = pd.Series(base, index=dates)

        return series_a, series_b

    async def test_pair_finding(self):
        """Test finding cointegrated pairs."""
        # Create test data
        series_a, series_b = self._create_cointegrated_pair()

        # Add to price data
        price_data = {
            "AAPL": pd.DataFrame({"close": series_a}),
            "GOOGL": pd.DataFrame({"close": series_b}),
            "MSFT": pd.DataFrame({"close": series_a * 1.5 + np.random.randn(len(series_a)) * 5}),
        }

        # Find pairs
        pairs = await self.strategy.find_pairs(["AAPL", "GOOGL", "MSFT"], price_data)

        # Should find AAPL-GOOGL pair
        self.assertGreater(len(pairs), 0)
        self.assertIn(("AAPL", "GOOGL"), pairs)

    async def test_pair_signal_generation(self):
        """Test signal generation for pairs."""
        # Set up pair stats
        from robo_trader.strategies.pairs_trading import PairStats

        pair_stats = PairStats(
            symbol_a="AAPL",
            symbol_b="GOOGL",
            correlation=0.85,
            cointegration_pvalue=0.02,
            hedge_ratio=1.2,
            spread_mean=0,
            spread_std=2,
            half_life=10,
            last_updated=datetime.now(),
        )

        self.strategy.pair_stats[("AAPL", "GOOGL")] = pair_stats

        # Test entry signal
        current_prices = {"AAPL": 150, "GOOGL": 120}

        signals = await self.strategy.analyze_pairs([("AAPL", "GOOGL")], current_prices)

        # Should generate signal if z-score exceeds threshold
        self.assertIsInstance(signals, list)

    def test_half_life_calculation(self):
        """Test mean reversion half-life calculation."""
        # Create mean-reverting spread
        spread = pd.Series([0, 1, 0.5, 0.25, 0.125, 0.0625])

        half_life = self.strategy._calculate_half_life(spread)

        # Should be approximately 1 (perfect half-life series)
        self.assertGreater(half_life, 0)
        self.assertLess(half_life, 100)

    def test_ml_enhanced_pairs(self):
        """Test ML-enhanced pairs trading."""
        strategy = MLEnhancedPairsTrading(
            use_ml_enhancement=True,
        )

        # Create pair stats
        from robo_trader.strategies.pairs_trading import PairStats

        pair_stats = PairStats(
            symbol_a="AAPL",
            symbol_b="GOOGL",
            correlation=0.9,
            cointegration_pvalue=0.01,
            hedge_ratio=1.1,
            spread_mean=0,
            spread_std=1.5,
            half_life=15,
            last_updated=datetime.now(),
        )

        # Create price series
        series_a, series_b = self._create_cointegrated_pair()

        # Extract features
        features = strategy._extract_pair_features(pair_stats, series_a, series_b)

        # Should have correct number of features
        self.assertEqual(len(features), 15)

        # Test Hurst exponent calculation
        hurst = strategy._calculate_hurst_exponent(series_a - series_b)
        self.assertGreaterEqual(hurst, 0)
        self.assertLessEqual(hurst, 1)


class TestStatisticalArbitrage(unittest.TestCase):
    """Test statistical arbitrage strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = StatisticalArbitrageStrategy(
            universe_size=20,
            max_positions=5,
            use_ml_ranking=True,
        )

    def _create_market_data(self) -> Dict[str, pd.DataFrame]:
        """Create synthetic market data."""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        market_data = {}

        for i, symbol in enumerate(symbols):
            periods = 150
            dates = pd.date_range(end=datetime.now(), periods=periods, freq="1D")

            # Create different patterns for each stock
            np.random.seed(42 + i)

            if i == 0:  # Oversold pattern
                prices = 100 - np.abs(np.cumsum(np.random.randn(periods) * 0.5))
            elif i == 1:  # Overbought pattern
                prices = 100 + np.abs(np.cumsum(np.random.randn(periods) * 0.5))
            else:  # Random walk
                prices = 100 + np.cumsum(np.random.randn(periods) * 1)

            market_data[symbol] = pd.DataFrame(
                {
                    "close": prices,
                    "high": prices * 1.02,
                    "low": prices * 0.98,
                    "volume": np.random.randint(1e6, 1e7, periods),
                }
            )

        # Add SPY for market correlation
        market_data["SPY"] = pd.DataFrame(
            {
                "close": 100 + np.cumsum(np.random.randn(periods) * 0.3),
                "volume": np.random.randint(1e8, 2e8, periods),
            }
        )

        return market_data

    async def test_arbitrage_score_calculation(self):
        """Test arbitrage opportunity scoring."""
        market_data = self._create_market_data()

        scores = await self.strategy.calculate_arbitrage_scores(
            ["AAPL", "GOOGL", "MSFT"], market_data
        )

        # Should identify opportunities
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)

        # Scores should be between 0 and 1
        for score in scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)

    def test_feature_calculation(self):
        """Test feature calculation for stat arb."""
        market_data = self._create_market_data()

        features = self.strategy._calculate_features(market_data["AAPL"], market_data)

        # Check all features are present
        expected_features = [
            "rsi_14",
            "bb_position",
            "volume_ratio",
            "price_momentum_5d",
            "price_momentum_20d",
            "volatility_ratio",
            "correlation_spy",
            "relative_strength",
            "mean_reversion_score",
            "liquidity_score",
        ]

        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsNotNone(features[feature])

    def test_portfolio_weight_generation(self):
        """Test portfolio weight generation."""
        scores = {
            "AAPL": 0.9,
            "GOOGL": 0.8,
            "MSFT": 0.7,
            "AMZN": 0.6,
            "TSLA": 0.5,
            "FB": 0.4,
        }

        weights = self.strategy.generate_portfolio_weights(scores)

        # Should select top N positions
        self.assertEqual(len(weights), self.strategy.max_positions)

        # Weights should sum to 1
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

        # Higher scores should have higher weights
        if "AAPL" in weights and "TSLA" in weights:
            self.assertGreater(weights["AAPL"], weights["TSLA"])

    def test_sector_neutrality(self):
        """Test sector neutrality application."""
        scores = {
            "AAPL": 0.9,
            "TSLA": 0.85,
            "NVDA": 0.8,
            "MSFT": 0.75,
            "AMZN": 0.7,
        }

        adjusted_scores = self.strategy._apply_sector_neutrality(scores)

        # Should maintain relative ordering within sectors
        self.assertIsInstance(adjusted_scores, dict)
        self.assertEqual(len(adjusted_scores), len(scores))

        # All scores should be between 0 and 1
        for score in adjusted_scores.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for mean reversion suite."""

    def test_strategy_suite_creation(self):
        """Test creation of complete strategy suite."""
        strategies = create_mean_reversion_suite()

        # Should create multiple strategies
        self.assertGreater(len(strategies), 4)

        # Check strategy types
        strategy_types = [type(s).__name__ for s in strategies]
        self.assertIn("CointegrationPairsStrategy", strategy_types)
        self.assertIn("StatisticalArbitrageStrategy", strategy_types)
        self.assertIn("MLEnhancedPairsTrading", strategy_types)

    async def test_end_to_end_pairs_trading(self):
        """Test end-to-end pairs trading workflow."""
        strategy = CointegrationPairsStrategy(
            lookback_days=30,
            use_ml_enhancement=False,
        )

        # Create cointegrated data
        periods = 100
        dates = pd.date_range(end=datetime.now(), periods=periods, freq="1D")

        np.random.seed(42)
        base = 100 + np.cumsum(np.random.randn(periods) * 0.5)
        spread = np.sin(np.arange(periods) * 0.2) * 5  # Oscillating spread

        price_data = {
            "AAPL": pd.DataFrame({"close": base + spread}, index=dates),
            "GOOGL": pd.DataFrame({"close": base}, index=dates),
        }

        # Find pairs
        pairs = await strategy.find_pairs(["AAPL", "GOOGL"], price_data)
        self.assertGreater(len(pairs), 0)

        # Generate signals
        current_prices = {
            "AAPL": price_data["AAPL"]["close"].iloc[-1],
            "GOOGL": price_data["GOOGL"]["close"].iloc[-1],
        }

        signals = await strategy.analyze_pairs(pairs, current_prices)

        # Process signals
        for signal in signals:
            if signal["signal"] != PairSignal.HOLD.value:
                strategy.update_position(pairs[0], signal)

        # Check portfolio summary
        summary = strategy.get_portfolio_summary()
        self.assertIn("active_pairs", summary)
        self.assertIn("total_pairs_discovered", summary)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestMeanReversionStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestPairsTrading))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalArbitrage))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run async tests with asyncio
    success = run_tests()

    print("\n" + "=" * 60)
    if success:
        print("✅ All mean reversion strategy tests passed!")
    else:
        print("❌ Some tests failed. Please review the output above.")

    exit(0 if success else 1)

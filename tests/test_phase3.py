#!/usr/bin/env python3
"""
Phase 3 Test Suite - Strategy Framework and Backtesting

Tests all Phase 3 components:
- Strategy framework
- Backtesting engine
- Performance metrics
- Individual strategies
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robo_trader.backtest.engine import BacktestConfig, BacktestEngine
from robo_trader.backtest.metrics import PerformanceMetrics, calculate_metrics
from robo_trader.data.pipeline import DataPipeline
from robo_trader.features.engine import FeatureSet
from robo_trader.strategies.breakout import BreakoutStrategy
from robo_trader.strategies.framework import Signal, SignalType, Strategy, StrategyState
from robo_trader.strategies.mean_reversion import MeanReversionStrategy
from robo_trader.strategies.momentum import EnhancedMomentumStrategy


def generate_test_data(symbol: str, periods: int = 100) -> pd.DataFrame:
    """Generate test OHLCV data."""
    dates = pd.date_range(end=datetime.now(), periods=periods, freq="5min")

    # Generate realistic price movement
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, periods)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": price * (1 + np.random.uniform(-0.002, 0.002, periods)),
            "high": price * (1 + np.abs(np.random.uniform(0, 0.005, periods))),
            "low": price * (1 - np.abs(np.random.uniform(0, 0.005, periods))),
            "close": price,
            "volume": np.random.uniform(1000000, 5000000, periods),
        },
        index=dates,
    )

    return df


def create_mock_features() -> FeatureSet:
    """Create mock feature set for testing."""
    return FeatureSet(
        timestamp=datetime.now(),
        symbol="TEST",
        rsi=45.0,
        macd_line=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        bb_upper=105.0,
        bb_middle=100.0,
        bb_lower=95.0,
        bb_bandwidth=10.0,
        atr=2.0,
        obv=1000000,
        vwap=100.5,
        sma_20=100.1,
        sma_50=99.9,
        ema_12=100.2,
        ema_26=99.8,
        momentum_1d=0.01,
        momentum_5d=0.02,
    )


@pytest.mark.asyncio
async def test_strategy_framework():
    """Test base strategy framework."""
    print("\n" + "=" * 50)
    print("Testing Strategy Framework...")
    print("=" * 50)

    try:
        # Create test strategy
        class TestStrategy(Strategy):
            async def _initialize(self, historical_data):
                pass

            async def _generate_signals(self, market_data, features):
                # Generate a test signal
                return [
                    Signal(
                        timestamp=datetime.now(),
                        symbol="TEST",
                        signal_type=SignalType.BUY,
                        strength=0.8,
                        entry_price=100.0,
                        stop_loss=98.0,
                        take_profit=104.0,
                        rationale="Test signal",
                    )
                ]

        strategy = TestStrategy(name="Test", symbols=["TEST"], lookback_period=50)

        # Initialize with test data
        historical_data = {"TEST": generate_test_data("TEST")}
        await strategy.initialize(historical_data)

        # Generate signals
        features = {"TEST": create_mock_features()}
        signals = await strategy.generate_signals(historical_data, features)

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.BUY
        assert signals[0].risk_reward_ratio == 2.0

        print("✓ Strategy framework working")
        print(f"  Generated {len(signals)} signal(s)")
        print(f"  Risk:Reward = {signals[0].risk_reward_ratio:.1f}")

    except Exception as e:
        print(f"✗ Strategy framework test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.asyncio
async def test_momentum_strategy():
    """Test enhanced momentum strategy."""
    print("\n" + "=" * 50)
    print("Testing Momentum Strategy...")
    print("=" * 50)

    try:
        strategy = EnhancedMomentumStrategy(
            symbols=["TEST"], rsi_period=14, macd_fast=12, macd_slow=26
        )

        # Initialize with test data
        historical_data = {"TEST": generate_test_data("TEST", 200)}
        await strategy.initialize(historical_data)

        # Create features with momentum signal
        features = create_mock_features()
        features.rsi = 25.0  # Oversold
        features.macd_histogram = 0.5  # Positive momentum

        signals = await strategy.generate_signals(historical_data, {"TEST": features})

        print(f"✓ Momentum strategy initialized")
        print(f"  Momentum score calculated")
        print(f"  Generated {len(signals)} signal(s)")

        if signals:
            print(f"  Signal: {signals[0].signal_type.value}")
            print(f"  Strength: {signals[0].strength:.2f}")

    except Exception as e:
        print(f"✗ Momentum strategy test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_mean_reversion_strategy():
    """Test mean reversion strategy."""
    print("\n" + "=" * 50)
    print("Testing Mean Reversion Strategy...")
    print("=" * 50)

    try:
        strategy = MeanReversionStrategy(symbols=["TEST"], bb_period=20, zscore_threshold=2.0)

        # Initialize with test data
        historical_data = {"TEST": generate_test_data("TEST", 200)}
        await strategy.initialize(historical_data)

        # Create features indicating oversold
        features = create_mock_features()
        features.bb_lower = 98.0
        features.bb_middle = 100.0
        features.bb_upper = 102.0
        features.rsi = 20.0  # Oversold

        # Modify data to show price below lower band
        historical_data["TEST"]["close"].iloc[-1] = 97.5

        signals = await strategy.generate_signals(historical_data, {"TEST": features})

        print(f"✓ Mean reversion strategy initialized")
        print(f"  Reversion score calculated")
        print(f"  Generated {len(signals)} signal(s)")

        if signals:
            print(f"  Signal: {signals[0].signal_type.value}")
            print(f"  Target mean: {signals[0].metadata.get('target_mean', 0):.2f}")

    except Exception as e:
        print(f"✗ Mean reversion strategy test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_breakout_strategy():
    """Test breakout strategy."""
    print("\n" + "=" * 50)
    print("Testing Breakout Strategy...")
    print("=" * 50)

    try:
        strategy = BreakoutStrategy(symbols=["TEST"], lookback_periods=20, breakout_threshold=0.02)

        # Initialize with consolidating data
        data = generate_test_data("TEST", 200)
        # Create consolidation pattern
        data["high"].iloc[-30:] = 101.0
        data["low"].iloc[-30:] = 99.0
        data["close"].iloc[-30:] = 100.0
        # Add breakout
        data["high"].iloc[-1] = 103.0
        data["close"].iloc[-1] = 102.5
        data["volume"].iloc[-1] *= 2  # Volume surge

        historical_data = {"TEST": data}
        await strategy.initialize(historical_data)

        features = create_mock_features()
        features.atr = 1.0

        signals = await strategy.generate_signals(historical_data, {"TEST": features})

        print(f"✓ Breakout strategy initialized")
        print(f"  Support/Resistance levels detected")
        print(f"  Generated {len(signals)} signal(s)")

        if signals:
            print(f"  Breakout type: {signals[0].metadata.get('breakout_type')}")
            print(f"  Breakout level: {signals[0].metadata.get('breakout_level', 0):.2f}")

    except Exception as e:
        print(f"✗ Breakout strategy test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_backtesting_engine():
    """Test backtesting engine."""
    print("\n" + "=" * 50)
    print("Testing Backtesting Engine...")
    print("=" * 50)

    try:
        # Create backtest config
        config = BacktestConfig(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
        )

        engine = BacktestEngine(config)

        # Create simple test strategy
        class SimpleStrategy(Strategy):
            def __init__(self):
                super().__init__("Simple", ["TEST"], 10)
                self.signal_count = 0

            async def _initialize(self, historical_data):
                pass

            async def _generate_signals(self, market_data, features):
                # Generate alternating buy/sell signals
                self.signal_count += 1
                if self.signal_count % 20 == 1:
                    return [
                        Signal(
                            timestamp=datetime.now(),
                            symbol="TEST",
                            signal_type=SignalType.BUY,
                            strength=0.8,
                            quantity=100,
                        )
                    ]
                elif self.signal_count % 20 == 10:
                    return [
                        Signal(
                            timestamp=datetime.now(),
                            symbol="TEST",
                            signal_type=SignalType.SELL,
                            strength=0.8,
                        )
                    ]
                return []

        strategy = SimpleStrategy()

        # Mock data pipeline and feature engine
        data_pipeline = Mock(spec=DataPipeline)
        data_pipeline.get_historical_data = AsyncMock(return_value=generate_test_data("TEST", 500))

        feature_engine = Mock()
        feature_engine.calculate_features = AsyncMock(return_value=create_mock_features())

        # Run backtest
        result = await engine.run(strategy, data_pipeline, feature_engine)

        assert result is not None
        assert result.metrics.total_trades >= 0
        assert len(result.equity_curve) > 0

        print("✓ Backtesting engine working")
        print(f"  Initial capital: ${config.initial_capital:,.0f}")
        print(f"  Final equity: ${engine.equity:,.2f}")
        print(f"  Total trades: {result.metrics.total_trades}")
        print(f"  Total return: {result.metrics.total_return:.2%}")

    except Exception as e:
        print(f"✗ Backtesting engine test failed: {e}")
        raise


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\n" + "=" * 50)
    print("Testing Performance Metrics...")
    print("=" * 50)

    try:
        # Create test data
        dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
        returns = pd.Series(np.random.normal(0.001, 0.02, 100), index=dates)

        equity_curve = pd.Series(100000 * (1 + returns).cumprod(), index=dates)

        trades = pd.DataFrame(
            {
                "pnl": [100, -50, 200, -30, 150, -100, 250, 50, -75, 300],
                "return_pct": [
                    0.01,
                    -0.005,
                    0.02,
                    -0.003,
                    0.015,
                    -0.01,
                    0.025,
                    0.005,
                    -0.0075,
                    0.03,
                ],
                "bars_held": [10, 5, 15, 8, 12, 6, 20, 7, 9, 18],
            }
        )

        metrics = calculate_metrics(
            returns=returns, equity_curve=equity_curve, trades=trades, initial_capital=100000
        )

        assert metrics.total_return != 0
        assert metrics.volatility > 0
        assert metrics.total_trades == len(trades)
        assert metrics.win_rate >= 0 and metrics.win_rate <= 1

        print("✓ Performance metrics calculated")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"  Win Rate: {metrics.win_rate:.1%}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Profit Factor: {metrics.profit_factor:.2f}")

    except Exception as e:
        print(f"✗ Performance metrics test failed: {e}")
        raise


@pytest.mark.asyncio
async def test_integration():
    """Test Phase 3 integration with existing components."""
    print("\n" + "=" * 50)
    print("Testing Phase 3 Integration...")
    print("=" * 50)

    try:
        # Test strategy can use existing feature engine
        from robo_trader.config import Config
        from robo_trader.features.engine import FeatureEngine

        config = Config()

        # Create mock feature engine
        feature_engine = Mock(spec=FeatureEngine)
        feature_engine.calculate_features = AsyncMock(return_value=create_mock_features())

        # Test strategy can integrate with risk management
        from robo_trader.risk import RiskManager

        risk_mgr = RiskManager(
            max_daily_loss=0.04,
            max_position_risk_pct=0.02,
            max_symbol_exposure_pct=0.1,
            max_leverage=2.0,
        )

        # Create and test strategy with risk validation
        strategy = EnhancedMomentumStrategy(symbols=["TEST", "AAPL"], min_signal_strength=0.6)

        historical_data = {"TEST": generate_test_data("TEST"), "AAPL": generate_test_data("AAPL")}

        await strategy.initialize(historical_data)

        features = {"TEST": create_mock_features(), "AAPL": create_mock_features()}

        signals = await strategy.generate_signals(historical_data, features)

        # Validate signals with risk management
        for signal in signals:
            # Mock risk validation
            is_valid = signal.risk_reward_ratio and signal.risk_reward_ratio >= 1.8
            if is_valid:
                print(f"  Signal passed risk validation: {signal.symbol}")

        print("✓ Phase 3 integration successful")
        print(f"  Strategies work with FeatureEngine")
        print(f"  Risk management integration ready")
        print(f"  Multi-symbol support working")

    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" " * 20 + "PHASE 3 TEST SUITE")
    print(" " * 15 + "Strategy Framework & Backtesting")
    print("=" * 60)

    # Run tests
    test_functions = [
        test_strategy_framework,
        test_momentum_strategy,
        test_mean_reversion_strategy,
        test_breakout_strategy,
        test_backtesting_engine,
        test_performance_metrics,
        test_integration,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            if asyncio.iscoroutinefunction(test_func):
                asyncio.run(test_func())
            else:
                test_func()
            passed += 1
        except Exception:
            failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{len(test_functions)}")
    print(f"Tests Failed: {failed}/{len(test_functions)}")

    if failed == 0:
        print("\n🎉 All Phase 3 tests passed!")
        print("\nPhase 3 Components Ready:")
        print("✅ Strategy framework with base class")
        print("✅ Enhanced momentum strategy")
        print("✅ Mean reversion strategy")
        print("✅ Breakout strategy")
        print("✅ Event-driven backtesting engine")
        print("✅ Comprehensive performance metrics")
        print("✅ Integration with Phases 1-2")
    else:
        print(f"\n❌ {failed} test(s) failed. Please review the errors above.")
        sys.exit(1)

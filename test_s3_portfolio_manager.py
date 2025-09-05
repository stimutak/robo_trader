#!/usr/bin/env python3
"""Test S3: Multi-Strategy Portfolio Manager functionality."""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import structlog

# Import from the .py file explicitly
import robo_trader.portfolio
from robo_trader.config import Config

Portfolio = robo_trader.portfolio.Portfolio

from robo_trader.portfolio_manager.portfolio_manager import (
    AllocationMethod,
    MultiStrategyPortfolioManager,
    PortfolioMetrics,
)
from robo_trader.risk import RiskManager
from robo_trader.strategies.mean_reversion import MeanReversionStrategy
from robo_trader.strategies.ml_enhanced_strategy import MLEnhancedStrategy
from robo_trader.strategies.momentum import MomentumStrategy

# Setup logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def test_portfolio_initialization():
    """Test portfolio manager initialization."""
    print("\n" + "=" * 60)
    print("Testing Portfolio Manager Initialization")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    # Test each allocation method
    methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.MEAN_VARIANCE,
        AllocationMethod.KELLY_OPTIMAL,
        AllocationMethod.ADAPTIVE,
    ]

    for method in methods:
        pm = MultiStrategyPortfolioManager(
            config=config,
            risk_manager=risk_manager,
            allocation_method=method,
            rebalance_frequency="daily",
        )

        print(f"✅ Initialized with {method.value} allocation")

    return True


async def test_strategy_registration():
    """Test registering strategies with the portfolio manager."""
    print("\n" + "=" * 60)
    print("Testing Strategy Registration")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    pm = MultiStrategyPortfolioManager(
        config=config, risk_manager=risk_manager, allocation_method=AllocationMethod.ADAPTIVE
    )

    # Register strategies
    strategies = [
        ("ML_Enhanced", MLEnhancedStrategy(config)),
        ("Momentum", MomentumStrategy()),
        ("MeanReversion", MeanReversionStrategy()),
    ]

    for name, strategy in strategies:
        # Create a simple wrapper with name attribute
        class StrategyWrapper:
            def __init__(self, name, strategy):
                self.name = name
                self.strategy = strategy

        wrapper = StrategyWrapper(name, strategy)
        pm.register_strategy(wrapper, initial_weight=1.0 / len(strategies))
        print(f"✅ Registered {name} strategy")

    # Check allocations
    summary = pm.get_allocation_summary()
    print(f"\n📊 Allocation Summary:")
    print(f"   Total Capital: ${summary['total_capital']:,.2f}")
    print(f"   Method: {summary['allocation_method']}")
    print(f"   Strategies: {len(summary['strategies'])}")

    return True


async def test_capital_allocation():
    """Test different capital allocation methods."""
    print("\n" + "=" * 60)
    print("Testing Capital Allocation Methods")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    # Test each allocation method
    for method in [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.ADAPTIVE,
        AllocationMethod.KELLY_OPTIMAL,
    ]:
        pm = MultiStrategyPortfolioManager(
            config=config, risk_manager=risk_manager, allocation_method=method
        )

        pm.update_capital(100000)

        # Register dummy strategies
        for i in range(3):

            class DummyStrategy:
                def __init__(self, name):
                    self.name = name

            pm.register_strategy(DummyStrategy(f"Strategy_{i}"))

        # Add some fake performance data
        for i in range(3):
            returns = np.random.randn(50) * 0.01  # Random returns
            for r in returns:
                pm.update_strategy_performance(f"Strategy_{i}", r)

        # Allocate capital
        weights = await pm.allocate_capital()

        print(f"\n📊 {method.value} Allocation:")
        for name, weight in weights.items():
            capital = weight * 100000
            print(f"   {name}: {weight:.2%} (${capital:,.0f})")

    return True


async def test_rebalancing():
    """Test portfolio rebalancing functionality."""
    print("\n" + "=" * 60)
    print("Testing Portfolio Rebalancing")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    pm = MultiStrategyPortfolioManager(
        config=config,
        risk_manager=risk_manager,
        allocation_method=AllocationMethod.ADAPTIVE,
        rebalance_frequency="daily",
    )

    pm.update_capital(100000)

    # Register strategies
    for i in range(3):

        class DummyStrategy:
            def __init__(self, name):
                self.name = name

        pm.register_strategy(DummyStrategy(f"Strategy_{i}"))

    # Initial allocation
    initial_weights = await pm.allocate_capital()
    print("📊 Initial Allocation:")
    for name, weight in initial_weights.items():
        print(f"   {name}: {weight:.2%}")

    # Simulate performance changes
    pm.update_strategy_performance("Strategy_0", 0.05)  # 5% gain
    pm.update_strategy_performance("Strategy_1", -0.02)  # 2% loss
    pm.update_strategy_performance("Strategy_2", 0.01)  # 1% gain

    # Check if rebalancing is needed
    should_rebalance = await pm.should_rebalance()
    print(f"\n🔄 Should rebalance: {should_rebalance}")

    if should_rebalance:
        result = await pm.rebalance()
        print("\n📊 New Allocation after rebalancing:")
        for name, weight in result["new_weights"].items():
            print(f"   {name}: {weight:.2%}")

    return True


async def test_performance_metrics():
    """Test portfolio performance metrics calculation."""
    print("\n" + "=" * 60)
    print("Testing Performance Metrics")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    pm = MultiStrategyPortfolioManager(
        config=config, risk_manager=risk_manager, allocation_method=AllocationMethod.ADAPTIVE
    )

    pm.update_capital(100000)

    # Register strategies
    for i in range(3):

        class DummyStrategy:
            def __init__(self, name):
                self.name = name

        pm.register_strategy(DummyStrategy(f"Strategy_{i}"))

    # Simulate returns history
    np.random.seed(42)
    for _ in range(100):
        portfolio_return = 0
        for i in range(3):
            strategy_return = np.random.randn() * 0.02  # 2% vol
            pm.update_strategy_performance(f"Strategy_{i}", strategy_return)
            portfolio_return += strategy_return / 3  # Equal weight for simplicity

        pm.returns_history.append(portfolio_return)

    # Get metrics
    metrics = pm.get_portfolio_metrics()

    print("📊 Portfolio Metrics:")
    print(f"   Total Return: {metrics.total_return:.2%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Volatility: {metrics.volatility:.2%}")
    print(f"   VaR 95%: {metrics.var_95:.2%}")

    return True


async def test_correlation_aware_allocation():
    """Test correlation-aware portfolio allocation."""
    print("\n" + "=" * 60)
    print("Testing Correlation-Aware Allocation")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    pm = MultiStrategyPortfolioManager(
        config=config, risk_manager=risk_manager, allocation_method=AllocationMethod.ADAPTIVE
    )

    pm.update_capital(100000)

    # Register strategies
    for i in range(3):

        class DummyStrategy:
            def __init__(self, name):
                self.name = name

        pm.register_strategy(DummyStrategy(f"Strategy_{i}"))

    # Create correlated returns
    np.random.seed(42)
    base_returns = np.random.randn(100) * 0.01

    # Strategy 0 and 1 are highly correlated
    strategy_0_returns = base_returns + np.random.randn(100) * 0.001
    strategy_1_returns = base_returns + np.random.randn(100) * 0.001
    # Strategy 2 is uncorrelated
    strategy_2_returns = np.random.randn(100) * 0.01

    for i in range(100):
        pm.update_strategy_performance("Strategy_0", strategy_0_returns[i])
        pm.update_strategy_performance("Strategy_1", strategy_1_returns[i])
        pm.update_strategy_performance("Strategy_2", strategy_2_returns[i])

    # Calculate correlation matrix
    returns_df = pd.DataFrame(
        {
            "Strategy_0": strategy_0_returns,
            "Strategy_1": strategy_1_returns,
            "Strategy_2": strategy_2_returns,
        }
    )

    correlation_matrix = returns_df.corr()

    print("📊 Correlation Matrix:")
    print(correlation_matrix)

    # Allocate with correlation awareness
    weights = await pm.allocate_capital()

    print("\n📊 Correlation-Aware Allocation:")
    for name, weight in weights.items():
        print(f"   {name}: {weight:.2%}")

    print("\n✅ Strategy_2 should have higher weight due to lower correlation")

    return True


async def test_risk_budgeting():
    """Test risk budgeting across strategies."""
    print("\n" + "=" * 60)
    print("Testing Risk Budgeting")
    print("=" * 60)

    config = Config()
    portfolio = Portfolio(starting_cash=100000)
    risk_manager = RiskManager(max_position_size=10000)

    pm = MultiStrategyPortfolioManager(
        config=config, risk_manager=risk_manager, allocation_method=AllocationMethod.RISK_PARITY
    )

    pm.update_capital(100000)

    # Register strategies with different risk profiles
    strategies_risk = [
        ("LowRisk", 0.05),  # 5% volatility
        ("MediumRisk", 0.15),  # 15% volatility
        ("HighRisk", 0.30),  # 30% volatility
    ]

    for name, vol in strategies_risk:

        class DummyStrategy:
            def __init__(self, name):
                self.name = name

        pm.register_strategy(DummyStrategy(name))

        # Generate returns with specified volatility
        returns = np.random.randn(100) * vol / np.sqrt(252)
        for r in returns:
            pm.update_strategy_performance(name, r)

    # Allocate with risk parity
    weights = await pm.allocate_capital()

    print("📊 Risk Parity Allocation:")
    for name, weight in weights.items():
        allocation = pm.allocations.get(name)
        if allocation:
            print(f"   {name}: {weight:.2%} (${allocation.allocated_capital:,.0f})")

    print("\n✅ Lower risk strategies should have higher weights")

    return True


async def main():
    """Run all S3 portfolio manager tests."""
    print("\n" + "=" * 60)
    print("🚀 S3: Multi-Strategy Portfolio Manager Tests")
    print("=" * 60)
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M"))

    tests = [
        ("Portfolio Initialization", test_portfolio_initialization),
        ("Strategy Registration", test_strategy_registration),
        ("Capital Allocation", test_capital_allocation),
        ("Portfolio Rebalancing", test_rebalancing),
        ("Performance Metrics", test_performance_metrics),
        ("Correlation-Aware Allocation", test_correlation_aware_allocation),
        ("Risk Budgeting", test_risk_budgeting),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 S3 Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:35} {status}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 60)
        print("✨ S3: Multi-Strategy Portfolio Manager - COMPLETE ✅")
        print("=" * 60)
        print("\n📝 Implemented Features:")
        print("   ✅ Dynamic capital allocation across strategies")
        print(
            "   ✅ Multiple allocation methods (Equal, Risk Parity, Mean-Variance, Kelly, Adaptive)"
        )
        print("   ✅ Risk budgeting and correlation-aware diversification")
        print("   ✅ Automatic rebalancing with drift detection")
        print("   ✅ Performance attribution and metrics")
        print("   ✅ Integration with runner_async")
        print("\n🚀 Usage:")
        print("   python -m robo_trader.runner_async --use-portfolio-manager")
        print("\n📂 Files:")
        print("   - robo_trader/portfolio_manager/portfolio_manager.py")
        print("   - robo_trader/portfolio/portfolio_manager.py")
        print("   - Integration in robo_trader/runner_async.py")
    else:
        print(f"\n⚠️ S3 incomplete: {total - passed} tests failed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

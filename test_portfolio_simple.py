#!/usr/bin/env python3
"""Simple test for the Multi-Strategy Portfolio Manager."""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from robo_trader.portfolio_pkg.portfolio_manager import (
    AllocationMethod,
    MultiStrategyPortfolioManager,
    StrategyAllocation,
)


class MockConfig:
    """Mock configuration for testing."""
    pass


class MockRiskManager:
    """Mock risk manager for testing."""
    pass


class MockStrategy:
    """Mock strategy for testing."""
    
    def __init__(self, name: str):
        self.name = name


async def test_basic_functionality():
    """Test basic portfolio manager functionality."""
    
    print("🚀 Testing Basic Portfolio Manager Functionality")
    print("=" * 50)
    
    # Create mock dependencies
    config = MockConfig()
    risk_manager = MockRiskManager()
    
    # Create portfolio manager
    portfolio_manager = MultiStrategyPortfolioManager(
        config=config,
        risk_manager=risk_manager,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        max_strategy_weight=0.4,
        min_strategy_weight=0.1,
    )
    
    print("✅ Portfolio manager created successfully")
    print(f"   Allocation method: {portfolio_manager.allocation_method.value}")
    print(f"   Max strategy weight: {portfolio_manager.max_strategy_weight:.1%}")
    print(f"   Min strategy weight: {portfolio_manager.min_strategy_weight:.1%}")
    
    # Test strategy registration
    strategies = [
        MockStrategy("Momentum"),
        MockStrategy("Mean_Reversion"),
        MockStrategy("Trend_Following"),
    ]
    
    for strategy in strategies:
        portfolio_manager.register_strategy(
            strategy=strategy,
            initial_weight=1.0 / len(strategies),
        )
    
    print(f"✅ Registered {len(strategies)} strategies")
    
    # Test capital allocation
    initial_capital = 1_000_000
    portfolio_manager.update_capital(initial_capital)
    
    print(f"✅ Set capital: ${initial_capital:,.0f}")
    
    # Test equal weight allocation
    print("\n📊 Testing Equal Weight Allocation:")
    weights = await portfolio_manager.allocate_capital()
    
    total_weight = 0.0
    for strategy_name, weight in weights.items():
        allocated = initial_capital * weight
        print(f"   {strategy_name}: {weight:.1%} (${allocated:,.0f})")
        total_weight += weight
    
    print(f"   Total weight: {total_weight:.1%}")
    assert abs(total_weight - 1.0) < 0.001, "Weights should sum to 1.0"
    
    # Test allocation summary
    print("\n📋 Allocation Summary:")
    summary = portfolio_manager.get_allocation_summary()
    
    print(f"   Total Capital: ${summary['total_capital']:,.0f}")
    print(f"   Allocation Method: {summary['allocation_method']}")
    print(f"   Number of Strategies: {len(summary['strategies'])}")
    
    # Test performance tracking
    print("\n📈 Testing Performance Tracking:")
    
    for i, strategy_name in enumerate(weights.keys()):
        return_pct = 0.01 * (i + 1)  # Different returns for each strategy
        portfolio_manager.update_strategy_performance(
            strategy_name=strategy_name,
            return_pct=return_pct,
            metrics={"trades": 10, "win_rate": 0.6}
        )
        print(f"   Updated {strategy_name} with {return_pct:.1%} return")
    
    # Test portfolio metrics
    portfolio_manager.returns_history = [0.01, 0.02, -0.005, 0.015, 0.008]  # Mock returns
    
    print("\n📊 Portfolio Metrics:")
    metrics = portfolio_manager.get_portfolio_metrics()
    
    print(f"   Total Return: {metrics.total_return:.2%}")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"   Volatility: {metrics.volatility:.2%}")
    
    # Test different allocation methods
    print("\n🔄 Testing Different Allocation Methods:")
    
    allocation_methods = [
        AllocationMethod.EQUAL_WEIGHT,
        AllocationMethod.RISK_PARITY,
        AllocationMethod.ADAPTIVE,
    ]
    
    for method in allocation_methods:
        portfolio_manager.allocation_method = method
        weights = await portfolio_manager.allocate_capital()
        
        print(f"\n   {method.value.upper()}:")
        for strategy_name, weight in weights.items():
            print(f"     {strategy_name}: {weight:.1%}")
    
    # Test rebalancing logic
    print("\n⚖️ Testing Rebalancing Logic:")
    
    should_rebalance = await portfolio_manager.should_rebalance()
    print(f"   Should rebalance: {should_rebalance}")
    
    if should_rebalance:
        rebalance_result = await portfolio_manager.rebalance()
        print(f"   Rebalance completed at: {rebalance_result['timestamp']}")
        print(f"   New weights: {rebalance_result['new_weights']}")
    
    print("\n✅ All basic functionality tests passed!")
    return True


async def test_weight_constraints():
    """Test weight constraint enforcement."""
    
    print("\n🔒 Testing Weight Constraints")
    print("-" * 30)
    
    config = MockConfig()
    risk_manager = MockRiskManager()
    
    # Create portfolio manager with tight constraints
    portfolio_manager = MultiStrategyPortfolioManager(
        config=config,
        risk_manager=risk_manager,
        allocation_method=AllocationMethod.EQUAL_WEIGHT,
        max_strategy_weight=0.3,  # Max 30%
        min_strategy_weight=0.15,  # Min 15%
    )
    
    # Register strategies
    strategies = [MockStrategy(f"Strategy_{i}") for i in range(5)]
    
    for strategy in strategies:
        portfolio_manager.register_strategy(strategy, initial_weight=0.2)
    
    # Test constraint application
    weights = await portfolio_manager.allocate_capital()
    
    print("   Constrained weights:")
    total_weight = 0.0
    for strategy_name, weight in weights.items():
        print(f"     {strategy_name}: {weight:.1%}")
        assert weight >= 0.15, f"Weight {weight:.1%} below minimum 15%"
        assert weight <= 0.3, f"Weight {weight:.1%} above maximum 30%"
        total_weight += weight
    
    print(f"   Total weight: {total_weight:.1%}")
    assert abs(total_weight - 1.0) < 0.001, "Constrained weights should sum to 1.0"
    
    print("✅ Weight constraints enforced correctly")


async def main():
    """Run all tests."""
    
    try:
        # Run basic functionality tests
        await test_basic_functionality()
        
        # Run constraint tests
        await test_weight_constraints()
        
        print("\n🎉 All Portfolio Manager Tests Passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

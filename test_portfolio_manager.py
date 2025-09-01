#!/usr/bin/env python3
"""Test the Multi-Strategy Portfolio Manager."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from robo_trader.config import Config
from robo_trader.risk import RiskManager
from robo_trader.portfolio_manager import AllocationMethod, MultiStrategyPortfolioManager
from robo_trader.strategies.framework import Strategy, StrategyState, StrategyMetrics


class MockStrategy(Strategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str, expected_return: float = 0.1, volatility: float = 0.2):
        super().__init__(name)
        self.expected_return = expected_return
        self.volatility = volatility
        self.returns = []
    
    async def analyze(self, symbol: str, df: pd.DataFrame) -> dict:
        """Mock analysis that generates random returns."""
        # Generate random return based on expected return and volatility
        daily_return = np.random.normal(
            self.expected_return / 252,  # Daily expected return
            self.volatility / np.sqrt(252)  # Daily volatility
        )
        
        self.returns.append(daily_return)
        
        return {
            "signal": "BUY" if daily_return > 0 else "SELL",
            "confidence": min(abs(daily_return) * 10, 1.0),
            "expected_return": daily_return,
        }


async def test_portfolio_manager():
    """Test the multi-strategy portfolio manager."""
    
    print("🚀 Testing Multi-Strategy Portfolio Manager")
    print("=" * 50)
    
    # Create configuration and risk manager
    config = Config()
    risk_manager = RiskManager(
        max_daily_loss=0.05,
        max_position_risk_pct=0.1,
        max_symbol_exposure_pct=0.2,
        max_leverage=2.0,
    )
    
    # Create portfolio manager
    portfolio_manager = MultiStrategyPortfolioManager(
        config=config,
        risk_manager=risk_manager,
        allocation_method=AllocationMethod.ADAPTIVE,
        rebalance_frequency="daily",
        max_strategy_weight=0.4,
        min_strategy_weight=0.1,
    )
    
    print(f"✅ Created portfolio manager with {portfolio_manager.allocation_method.value} allocation")
    
    # Create mock strategies with different characteristics
    strategies = [
        MockStrategy("Momentum", expected_return=0.15, volatility=0.25),
        MockStrategy("Mean_Reversion", expected_return=0.12, volatility=0.18),
        MockStrategy("Trend_Following", expected_return=0.10, volatility=0.20),
        MockStrategy("Statistical_Arbitrage", expected_return=0.08, volatility=0.12),
    ]
    
    # Register strategies
    for strategy in strategies:
        portfolio_manager.register_strategy(
            strategy=strategy,
            initial_weight=0.25,  # Equal initial weights
        )
    
    print(f"✅ Registered {len(strategies)} strategies")
    
    # Set initial capital
    initial_capital = 1_000_000  # $1M
    portfolio_manager.update_capital(initial_capital)
    
    print(f"✅ Set initial capital: ${initial_capital:,.0f}")
    
    # Test initial allocation
    print("\n📊 Initial Capital Allocation:")
    weights = await portfolio_manager.allocate_capital()
    for strategy_name, weight in weights.items():
        allocated = initial_capital * weight
        print(f"  {strategy_name}: {weight:.1%} (${allocated:,.0f})")
    
    # Simulate trading over time
    print("\n📈 Simulating Trading Performance:")
    
    # Create mock market data
    dates = pd.date_range(start="2024-01-01", periods=60, freq="D")
    mock_data = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.normal(0, 1, 60)),
        "volume": np.random.randint(1000000, 5000000, 60),
    }, index=dates)
    
    # Simulate daily performance
    for day in range(30):  # 30 days of trading
        daily_returns = {}
        
        # Get strategy returns for the day
        for strategy in strategies:
            analysis = await strategy.analyze("TEST", mock_data.iloc[:day+10])
            daily_return = analysis["expected_return"]
            daily_returns[strategy.name] = daily_return
            
            # Update portfolio manager with strategy performance
            portfolio_manager.update_strategy_performance(
                strategy_name=strategy.name,
                return_pct=daily_return,
                metrics={
                    "trades": 1,
                    "win_rate": 0.6 if daily_return > 0 else 0.4,
                    "avg_return": daily_return,
                }
            )
        
        # Calculate portfolio return
        portfolio_return = 0.0
        for strategy_name, allocation in portfolio_manager.allocations.items():
            if strategy_name in daily_returns:
                portfolio_return += allocation.current_weight * daily_returns[strategy_name]
        
        portfolio_manager.returns_history.append(portfolio_return)
        
        # Check if rebalancing is needed
        if await portfolio_manager.should_rebalance():
            rebalance_result = await portfolio_manager.rebalance()
            print(f"  Day {day+1}: Rebalanced portfolio")
            
            # Show new weights
            for strategy_name, weight in rebalance_result["new_weights"].items():
                print(f"    {strategy_name}: {weight:.1%}")
    
    # Get final metrics
    print("\n📊 Final Portfolio Metrics:")
    metrics = portfolio_manager.get_portfolio_metrics()
    
    print(f"  Total Return: {metrics.total_return:.2%}")
    print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"  Volatility: {metrics.volatility:.2%}")
    print(f"  VaR (95%): {metrics.var_95:.2%}")
    
    print("\n🎯 Strategy Contributions:")
    for strategy_name, contribution in metrics.strategy_contributions.items():
        print(f"  {strategy_name}: {contribution:.4f}")
    
    # Get allocation summary
    print("\n📋 Final Allocation Summary:")
    summary = portfolio_manager.get_allocation_summary()
    
    print(f"  Total Capital: ${summary['total_capital']:,.0f}")
    print(f"  Allocation Method: {summary['allocation_method']}")
    print(f"  Last Rebalance: {summary['last_rebalance']}")
    
    for strategy_name, info in summary["strategies"].items():
        print(f"  {strategy_name}:")
        print(f"    Target Weight: {info['target_weight']:.1%}")
        print(f"    Current Weight: {info['current_weight']:.1%}")
        print(f"    Allocated Capital: ${info['allocated_capital']:,.0f}")
    
    print("\n✅ Portfolio Manager Test Completed Successfully!")
    
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
        
        print(f"\n  {method.value.upper()}:")
        for strategy_name, weight in weights.items():
            print(f"    {strategy_name}: {weight:.1%}")
    
    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_portfolio_manager())

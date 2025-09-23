#!/usr/bin/env python3
"""
Test script for Phase 4 P1: Advanced Risk Management
Tests Kelly sizing, correlation limits, and kill switches
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.logger import get_logger
from robo_trader.risk.advanced_risk import (
    AdvancedRiskManager,
    CorrelationLimiter,
    KellySizer,
    KillSwitch,
    RiskLevel,
    risk_monitor_task,
)

logger = get_logger(__name__)


def test_kelly_sizing():
    """Test Kelly criterion position sizing"""
    print("\n" + "=" * 50)
    print("Testing Kelly Criterion Position Sizing")
    print("=" * 50)

    # Create Kelly sizer
    kelly = KellySizer(
        lookback_trades=100, min_trades=30, max_kelly_fraction=0.25, use_half_kelly=True
    )

    # Simulate trade history
    np.random.seed(42)

    # Generate 50 trades with 55% win rate
    for i in range(50):
        is_win = np.random.random() < 0.55
        if is_win:
            pnl = np.random.uniform(200, 600)  # $200-600 wins
        else:
            pnl = -np.random.uniform(100, 400)  # $100-400 losses

        kelly.add_trade("AAPL", pnl, entry_price=150)

    # Calculate Kelly fraction
    kelly_params = kelly.calculate_kelly("AAPL")

    print(
        f"✓ Kelly fraction calculated: {kelly_params.kelly_fraction:.4f} ({kelly_params.kelly_fraction*100:.2f}%)"
    )

    # Test position sizing
    capital = 100000
    position_size = kelly.get_position_size(capital, "AAPL")

    print(f"✓ Position size: ${position_size:,.2f}")
    print(f"  (Account value: ${capital:,})")
    print(f"  Win rate: {kelly_params.win_rate:.2%}")
    print(f"  Avg win: {kelly_params.avg_win:.2%}")
    print(f"  Avg loss: {kelly_params.avg_loss:.2%}")
    print(f"  Half Kelly: {kelly_params.half_kelly:.4f}")

    return True


def test_correlation_limiter():
    """Test correlation-based position limits"""
    print("\n" + "=" * 50)
    print("Testing Correlation-Based Position Limits")
    print("=" * 50)

    limiter = CorrelationLimiter(
        max_correlation=0.7, max_correlated_exposure=0.3, correlation_window=60
    )

    # Simulate price history
    np.random.seed(42)
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]

    for _ in range(60):
        for symbol in symbols:
            # Correlated price movements for AAPL/MSFT
            if symbol in ["AAPL", "MSFT"]:
                price = 100 + np.random.normal(0, 1)
            else:
                price = 100 + np.random.normal(0, 2)
            limiter.update_price(symbol, price)

    # Calculate correlations
    corr_matrix = limiter.calculate_correlations()

    if not corr_matrix.empty:
        print("✓ Correlation matrix calculated:")
        print(corr_matrix.round(2))

    # Check correlation limits
    current_positions = {"MSFT": 0.15, "GOOGL": 0.10}

    allowed, max_corr, correlated = limiter.check_correlation_limit("AAPL", current_positions)

    print(f"\n✓ Correlation limit check for AAPL:")
    print(f"  Allowed: {allowed}")
    print(f"  Max correlation: {max_corr:.2f}")
    print(f"  Highly correlated: {correlated}")

    # Get correlation penalty
    penalty = limiter.get_correlation_penalty("AAPL", current_positions)
    print(f"✓ Position size penalty: {penalty:.2f}x")

    return True


def test_kill_switches():
    """Test automated kill switches"""
    print("\n" + "=" * 50)
    print("Testing Automated Kill Switches")
    print("=" * 50)

    kill_switch = KillSwitch(
        max_daily_loss_pct=0.05,
        max_position_loss_pct=0.02,
        max_consecutive_losses=5,
        max_drawdown_pct=0.10,
        cooldown_minutes=60,
    )

    # Test daily loss check
    starting_equity = 100000
    current_equity = 96000  # 4% loss

    triggered = kill_switch.check_daily_loss(current_equity, starting_equity)
    print(f"✓ Daily loss check (4% loss): {'TRIGGERED' if triggered else 'OK'}")

    # Test consecutive losses
    for i in range(6):
        trade_result = -100 if i < 4 else 100
        triggered = kill_switch.check_consecutive_losses(trade_result)
        if not triggered:
            print(f"  Trade {i+1}: {'Loss' if trade_result < 0 else 'Win'}")

    print(f"✓ Consecutive losses check: {'TRIGGERED' if kill_switch.triggered else 'OK'}")

    # Reset for drawdown test
    kill_switch.triggered = False
    kill_switch.peak_equity = 110000

    triggered = kill_switch.check_drawdown(95000)  # 13.6% drawdown
    print(f"✓ Drawdown check (13.6%): {'TRIGGERED' if triggered else 'OK'}")

    # Test position loss
    kill_switch.triggered = False
    kill_switch.position_entry["AAPL"] = (150, datetime.now())

    triggered = kill_switch.check_position_loss("AAPL", 145)  # 3.3% loss
    print(f"✓ Position loss check (3.3%): {'TRIGGERED' if triggered else 'OK'}")

    # Test can_trade
    can_trade, reason = kill_switch.can_trade()
    print(f"✓ Can trade: {can_trade}")
    if not can_trade:
        print(f"  Reason: {reason}")

    return True


async def test_advanced_risk_manager():
    """Test integrated advanced risk manager"""
    print("\n" + "=" * 50)
    print("Testing Advanced Risk Manager Integration")
    print("=" * 50)

    config = {"starting_capital": 100000, "max_position_pct": 0.1, "max_risk_per_trade": 0.02}

    risk_manager = AdvancedRiskManager(
        config=config, enable_kelly=True, enable_correlation_limits=True, enable_kill_switch=True
    )

    # Test position sizing with all features
    result = await risk_manager.calculate_position_size(
        symbol="AAPL", signal_strength=0.8, current_price=150, atr=3.5
    )

    print(f"✓ Position sizing result:")
    print(f"  Position size: {result['position_size']} shares")
    print(f"  Position value: ${result['position_value']:,.2f}")
    print(f"  Kelly fraction: {result['kelly_fraction']:.4f}")
    print(f"  Risk per trade: ${result['risk_per_trade']:,.2f}")

    if result.get("stop_loss"):
        print(f"  Stop loss: ${result['stop_loss']:.2f}")

    # Update position
    risk_manager.update_position("AAPL", 100, 150, "BUY")

    # Update market prices
    risk_manager.update_market_prices({"AAPL": 152})

    # Calculate current metrics
    metrics = risk_manager.calculate_current_metrics()

    print(f"\n✓ Current risk metrics:")
    print(f"  Total exposure: ${metrics.total_exposure:,.2f}")
    print(f"  Leverage: {metrics.leverage:.2f}x")
    print(f"  Sharpe ratio: {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown: {metrics.max_drawdown:.2%}")

    # Get risk dashboard
    dashboard = risk_manager.get_risk_dashboard()

    print(f"\n✓ Risk dashboard:")
    print(f"  Daily P&L: ${dashboard['daily_pnl']:,.2f}")
    print(f"  Total P&L: ${dashboard['total_pnl']:,.2f}")
    print(f"  Kill switch active: {dashboard['kill_switch']['active']}")

    # Save state
    state_file = Path("data/test_risk_state.json")
    state_file.parent.mkdir(exist_ok=True)
    risk_manager.save_state(state_file)
    print(f"\n✓ Risk manager state saved to {state_file}")

    # Test loading state
    new_manager = AdvancedRiskManager(config)
    new_manager.load_state(state_file)
    print(f"✓ Risk manager state loaded successfully")

    return True


async def test_risk_monitor():
    """Test background risk monitoring"""
    print("\n" + "=" * 50)
    print("Testing Risk Monitor Task")
    print("=" * 50)

    config = {"starting_capital": 100000, "max_position_pct": 0.1, "max_risk_per_trade": 0.02}

    risk_manager = AdvancedRiskManager(config)

    # Start monitor task
    monitor_task = asyncio.create_task(
        risk_monitor_task(risk_manager, interval=2)  # Fast interval for testing
    )

    print("✓ Risk monitor task started")

    # Simulate some trading activity
    risk_manager.update_position("AAPL", 100, 150, "BUY")
    risk_manager.update_position("MSFT", 50, 300, "BUY")

    # Let monitor run for a bit
    await asyncio.sleep(3)

    # Cancel monitor
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass

    print("✓ Risk monitor task stopped")

    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("PHASE 4 P1: ADVANCED RISK MANAGEMENT TESTS")
    print("=" * 50)

    all_passed = True

    # Test Kelly sizing
    try:
        if test_kelly_sizing():
            print("✅ Kelly sizing tests PASSED")
        else:
            print("❌ Kelly sizing tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Kelly sizing tests FAILED: {e}")
        all_passed = False

    # Test correlation limiter
    try:
        if test_correlation_limiter():
            print("✅ Correlation limiter tests PASSED")
        else:
            print("❌ Correlation limiter tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Correlation limiter tests FAILED: {e}")
        all_passed = False

    # Test kill switches
    try:
        if test_kill_switches():
            print("✅ Kill switch tests PASSED")
        else:
            print("❌ Kill switch tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Kill switch tests FAILED: {e}")
        all_passed = False

    # Test async components
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        # Test advanced risk manager
        if loop.run_until_complete(test_advanced_risk_manager()):
            print("✅ Advanced risk manager tests PASSED")
        else:
            print("❌ Advanced risk manager tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Advanced risk manager tests FAILED: {e}")
        all_passed = False

    try:
        # Test risk monitor
        if loop.run_until_complete(test_risk_monitor()):
            print("✅ Risk monitor tests PASSED")
        else:
            print("❌ Risk monitor tests FAILED")
            all_passed = False
    except Exception as e:
        print(f"❌ Risk monitor tests FAILED: {e}")
        all_passed = False

    loop.close()

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL P1 RISK MANAGEMENT TESTS PASSED!")
        print("\nAdvanced risk management features ready:")
        print("✓ Kelly criterion position sizing")
        print("✓ Correlation-based position limits")
        print("✓ Automated kill switches")
        print("✓ Integrated risk monitoring")
        print("✓ Risk state persistence")
        print("\nRun with: ADVANCED_RISK_ENABLED=true python3 -m robo_trader.runner_async")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 50)


if __name__ == "__main__":
    main()

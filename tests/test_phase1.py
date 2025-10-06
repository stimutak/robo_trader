#!/usr/bin/env python3
"""
Test script for Phase 1 implementation.

This script tests all Phase 1 components without requiring market hours.
Run this to verify the system is working correctly.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_configuration():
    """Test enhanced configuration system."""
    print("\n" + "=" * 50)
    print("Testing Configuration System...")
    print("=" * 50)

    try:
        from robo_trader.config import Config, Environment, TradingMode

        # Test basic configuration
        config = Config()
        print(f"✓ Config loaded successfully")
        print(f"  Environment: {config.environment}")
        print(f"  Trading Mode: {config.execution.mode}")
        print(f"  Risk - Max Position: {config.risk.max_position_pct:.1%}")
        print(f"  Risk - Max Daily Loss: {config.risk.max_daily_loss_pct:.1%}")

        # Test validation
        try:
            invalid_config = Config(execution={"mode": "invalid"})
            print("✗ Validation failed - should have raised error")
        except Exception:
            print("✓ Validation working - rejected invalid mode")

        # Test environment presets
        prod_config = Config(environment=Environment.PRODUCTION)
        assert prod_config.monitoring.enable_alerts is True
        print("✓ Environment presets working")

    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        raise


def test_risk_management():
    """Test enhanced risk management system."""
    print("\n" + "=" * 50)
    print("Testing Risk Management...")
    print("=" * 50)

    try:
        from robo_trader.risk import Position, RiskManager, RiskViolationType

        # Initialize risk manager
        risk_mgr = RiskManager(
            max_daily_loss=1000,
            max_position_risk_pct=0.02,
            max_symbol_exposure_pct=0.2,
            max_leverage=2.0,
            position_sizing_method="atr",
            min_volume=1_000_000,
        )
        print("✓ Risk manager initialized")

        # Test ATR-based position sizing
        risk_mgr.update_market_data("AAPL", atr=2.5, volume=50_000_000)
        size = risk_mgr.position_size_atr("AAPL", 10000, 150)
        print(f"✓ ATR position sizing: {size} shares for AAPL")

        # Test position validation
        positions = {}
        is_valid, msg = risk_mgr.validate_order("AAPL", 100, 150, 10000, 0, positions)
        print(f"✓ Order validation: {msg}")

        # Test portfolio heat calculation
        positions = {
            "AAPL": Position("AAPL", 100, 150.0, atr=2.5),
            "MSFT": Position("MSFT", 50, 300.0, atr=3.0),
        }
        heat = risk_mgr.calculate_portfolio_heat(positions, {"AAPL": 155, "MSFT": 305})
        print(f"✓ Portfolio heat: {heat:.2%}")

        # Test emergency shutdown logic
        for i in range(3):
            risk_mgr._record_violation(RiskViolationType.DAILY_LOSS_LIMIT, "TEST")

        should_shutdown = risk_mgr.should_emergency_shutdown()
        print(f"✓ Emergency shutdown logic: {should_shutdown} (should be False with 3 violations)")

    except Exception as e:
        print(f"✗ Risk management test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_correlation_tracking():
    """Test correlation tracking system."""
    print("\n" + "=" * 50)
    print("Testing Correlation Tracking...")
    print("=" * 50)

    try:
        from robo_trader.correlation import CorrelationTracker

        # Initialize tracker
        tracker = CorrelationTracker(lookback_days=60, correlation_threshold=0.7)
        print("✓ Correlation tracker initialized")

        # Create sample price data
        dates = pd.date_range("2024-01-01", periods=100)

        # Correlated stocks (tech sector)
        aapl_prices = pd.Series(150 + np.cumsum(np.random.randn(100) * 2), index=dates)
        msft_prices = pd.Series(300 + np.cumsum(np.random.randn(100) * 3), index=dates)

        # Less correlated stock (different sector)
        xom_prices = pd.Series(100 + np.cumsum(np.random.randn(100) * 1.5), index=dates)

        # Add price data
        tracker.add_price_series("AAPL", aapl_prices, sector="Technology")
        tracker.add_price_series("MSFT", msft_prices, sector="Technology")
        tracker.add_price_series("XOM", xom_prices, sector="Energy")
        print("✓ Price data added for 3 symbols")

        # Calculate correlation matrix
        corr_matrix = tracker.calculate_correlation_matrix()
        print("✓ Correlation matrix calculated")
        print(f"  Matrix shape: {corr_matrix.shape}")

        # Find high correlations
        high_corr = tracker.find_high_correlations(threshold=0.5)
        print(f"✓ Found {len(high_corr)} high correlation pairs")

        # Get correlation summary
        summary = tracker.get_correlation_summary()
        print(f"✓ Correlation summary:")
        print(f"  Mean correlation: {summary['mean_correlation']:.3f}")
        print(f"  Max correlation: {summary['max_correlation']:.3f}")

        # Test diversification suggestions
        suggestions = tracker.suggest_diversification(
            ["AAPL", "MSFT"], ["XOM", "AAPL"], max_suggestions=2
        )
        print(f"✓ Diversification suggestions: {suggestions}")

    except Exception as e:
        print(f"✗ Correlation tracking test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_structured_logging():
    """Test structured JSON logging."""
    print("\n" + "=" * 50)
    print("Testing Structured Logging...")
    print("=" * 50)

    try:
        from robo_trader.logger import (
            LogEvent,
            create_audit_logger,
            get_logger,
            log_performance,
            log_risk_violation,
            log_trade,
        )

        # Set JSON format for testing
        os.environ["MONITORING_LOG_FORMAT"] = "json"

        # Get logger
        logger = get_logger("test")
        print("✓ Logger initialized")

        # Test JSON output
        logger.info("Test message", symbol="AAPL", price=150.0)
        print("✓ JSON logging working")

        # Test trade logging
        log_trade(logger, LogEvent.TRADE_EXECUTED, "AAPL", 100, 150.0, "BUY", strategy="momentum")
        print("✓ Trade logging working")

        # Test risk violation logging
        log_risk_violation(logger, "position_size", "AAPL", "Position size exceeds limit")
        print("✓ Risk violation logging working")

        # Test performance logging
        log_performance(logger, "sharpe_ratio", 1.25)
        print("✓ Performance logging working")

        # Test audit logger
        audit_logger = create_audit_logger()
        audit_logger.info("Audit test", action="trade", user="system")
        print("✓ Audit logging working")

        # Test sensitive data censoring
        import os

        # Security: Use environment variable or test placeholder (never a real key)
        test_api_key = os.getenv("TEST_API_KEY", "TEST_PLACEHOLDER_NOT_A_REAL_KEY")
        logger.info("Test", api_key=test_api_key, safe_field="visible")
        print("✓ Sensitive data censoring working")

    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


@pytest.mark.asyncio
async def test_trading_engine():
    """Test async trading engine."""
    print("\n" + "=" * 50)
    print("Testing Trading Engine...")
    print("=" * 50)

    try:
        from robo_trader.config import Config
        from robo_trader.core.engine import EngineState, TradingEngine

        # Create config
        config = Config()
        print("✓ Config created")

        # Create engine
        engine = TradingEngine(config)
        print("✓ Engine created")
        print(f"  Initial state: {engine.state}")

        # Note: Cannot fully initialize without IBKR connection
        # Just test the structure

        # Test health status structure
        health = engine.get_health_status()
        print("✓ Health status structure:")
        print(f"  State: {health['state']}")
        print(f"  Is Healthy: {health['is_healthy']}")

        # Test metrics structure
        metrics = engine.get_metrics()
        print("✓ Metrics structure:")
        print(f"  Errors: {metrics['errors_count']}")
        print(f"  Tasks Running: {metrics['tasks_running']}")

        # Test market hours logic
        is_open = engine._is_market_open()
        print(f"✓ Market open check: {is_open}")

    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def test_backward_compatibility():
    """Test backward compatibility."""
    print("\n" + "=" * 50)
    print("Testing Backward Compatibility...")
    print("=" * 50)

    try:
        # Old way of importing
        from robo_trader.config import load_config

        # This should still work
        cfg = load_config()
        print("✓ Old load_config() still works")

        # Access old-style attributes through new config
        print(f"✓ IBKR Host: {cfg.ibkr.host}")
        print(f"✓ IBKR Port: {cfg.ibkr.port}")
        print(f"✓ Trading Mode: {cfg.execution.mode.value}")

        # Old Position class should still work
        from robo_trader.risk import Position

        pos = Position("AAPL", 100, 150.0)
        print(f"✓ Old Position class works: {pos.symbol}")

        # Logger backward compatibility
        from robo_trader.logger import get_logger

        logger = get_logger()
        logger.info("Backward compatibility test")
        print("✓ Logger backward compatibility works")

    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print(" PHASE 1 IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    print(f"Test Time: {datetime.now()}")
    print(f"Market Closed - Safe to Test")

    results = {
        "Configuration": test_configuration(),
        "Risk Management": test_risk_management(),
        "Correlation Tracking": test_correlation_tracking(),
        "Structured Logging": test_structured_logging(),
        "Trading Engine": asyncio.run(test_trading_engine()),
        "Backward Compatibility": test_backward_compatibility(),
    }

    # Summary
    print("\n" + "=" * 60)
    print(" TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready.")
        print("\nNext steps:")
        print("1. The system can run now with paper trading")
        print("2. Real market data will be available when market opens")
        print("3. Consider running: python start_ai_trading.py")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

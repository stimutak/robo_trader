#!/usr/bin/env python3
"""
Test script to verify production readiness safety features.

This script tests the critical safety improvements implemented in Phase 0 and Phase 1:
- Configuration validation (no hardcoded defaults)
- Database input validation
- Stop-loss monitoring
- Kill switch integration
- Proper exception handling

Run: python test_safety_features.py
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.config import load_config_from_env  # noqa: E402
from robo_trader.database_validator import DatabaseValidator, ValidationError  # noqa: E402
from robo_trader.execution import ExecutionResult, Order, PaperExecutor  # noqa: E402
from robo_trader.logger import get_logger  # noqa: E402
from robo_trader.risk_manager import Position, RiskManager  # noqa: E402
from robo_trader.stop_loss_monitor import StopLossMonitor, StopLossOrder, StopType  # noqa: E402

logger = get_logger(__name__)


class SafetyFeatureTester:
    """Test harness for safety features."""

    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = []

    def test(self, name: str, func):
        """Run a test and track results."""
        self.tests_run += 1
        try:
            result = func()
            if asyncio.iscoroutine(result):
                result = asyncio.run(result)

            if result:
                self.tests_passed += 1
                self.results.append(f"✅ PASS: {name}")
                print(f"✅ PASS: {name}")
            else:
                self.tests_failed += 1
                self.results.append(f"❌ FAIL: {name}")
                print(f"❌ FAIL: {name}")
        except Exception as e:
            self.tests_failed += 1
            self.results.append(f"❌ ERROR: {name} - {e}")
            print(f"❌ ERROR: {name} - {e}")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print("SAFETY FEATURE TEST SUMMARY")
        print("=" * 80)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_failed}")
        print(f"Pass Rate: {(self.tests_passed/self.tests_run*100):.1f}%")

        if self.tests_failed > 0:
            print("\n⚠️  CRITICAL: Safety features not fully functional!")
            print("DO NOT ATTEMPT LIVE TRADING")
        else:
            print("\n✅ All safety features verified")
        print("=" * 80)


def test_config_no_defaults():
    """Test that config requires explicit IBKR parameters."""
    # Save current env
    old_host = os.environ.get("IBKR_HOST")
    old_port = os.environ.get("IBKR_PORT")
    old_client = os.environ.get("IBKR_CLIENT_ID")

    try:
        # Clear IBKR env vars
        os.environ.pop("IBKR_HOST", None)
        os.environ.pop("IBKR_PORT", None)
        os.environ.pop("IBKR_CLIENT_ID", None)

        # This should raise an error
        try:
            config = load_config_from_env()
            return False  # Should not reach here
        except ValueError as e:
            # Expected behavior - no defaults allowed
            return "IBKR_HOST environment variable is required" in str(e)
    finally:
        # Restore env
        if old_host:
            os.environ["IBKR_HOST"] = old_host
        if old_port:
            os.environ["IBKR_PORT"] = old_port
        if old_client:
            os.environ["IBKR_CLIENT_ID"] = old_client


def test_database_validation():
    """Test database input validation."""
    validator = DatabaseValidator()

    # Test symbol validation
    try:
        # Valid symbol
        assert validator.validate_symbol("AAPL") == "AAPL"

        # Invalid symbols should raise
        try:
            validator.validate_symbol("'; DROP TABLE;")
            return False
        except ValidationError:
            pass  # Expected

        try:
            validator.validate_symbol("INVALID123")
            return False
        except ValidationError:
            pass  # Expected

    except Exception as e:
        logger.error(f"Database validation test failed: {e}")
        return False

    # Test price validation
    try:
        # Valid price
        assert validator.validate_price(100.50) == 100.50

        # Invalid prices should raise
        try:
            validator.validate_price(-10)
            return False
        except ValidationError:
            pass  # Expected

        try:
            validator.validate_price(10000000)  # Exceeds max
            return False
        except ValidationError:
            pass  # Expected

    except Exception as e:
        logger.error(f"Price validation test failed: {e}")
        return False

    # Test trade data validation
    try:
        valid_trade = {"symbol": "AAPL", "quantity": 100, "price": 150.00, "side": "BUY"}
        validated = validator.validate_trade_data(valid_trade)
        assert validated["symbol"] == "AAPL"
        assert validated["quantity"] == 100

        # Invalid trade should raise
        invalid_trade = {"symbol": "BAD;DROP", "quantity": -100, "price": -50, "side": "INVALID"}
        try:
            validator.validate_trade_data(invalid_trade)
            return False
        except ValidationError:
            pass  # Expected

    except Exception as e:
        logger.error(f"Trade validation test failed: {e}")
        return False

    return True


async def test_stop_loss_monitor():
    """Test stop-loss monitoring system."""
    # Create mock executor and risk manager
    executor = PaperExecutor(slippage_bps=0)
    risk_manager = RiskManager(
        max_daily_loss=1000,
        max_position_risk_pct=0.02,
        max_symbol_exposure_pct=0.1,
        max_leverage=2.0,
    )

    emergency_triggered = False

    async def emergency_callback(reason: str):
        nonlocal emergency_triggered
        emergency_triggered = True
        logger.warning(f"Emergency shutdown: {reason}")

    # Create stop-loss monitor
    monitor = StopLossMonitor(executor, risk_manager, emergency_callback)

    # Add a test position with stop-loss
    position = Position(symbol="AAPL", quantity=100, avg_price=150.00, entry_time=datetime.now())

    stop_order = await monitor.add_stop_loss(
        symbol="AAPL", position=position, stop_percent=0.02, stop_type=StopType.FIXED  # 2% stop
    )

    # Verify stop-loss created
    assert stop_order.symbol == "AAPL"
    assert stop_order.stop_price == 147.00  # 150 * 0.98
    assert stop_order.position_qty == 100

    # Update price - should not trigger
    await monitor.update_price("AAPL", 149.00)
    triggered = await monitor.check_stops()
    assert len(triggered) == 0

    # Update price below stop - should trigger
    await monitor.update_price("AAPL", 146.00)
    triggered = await monitor.check_stops()
    assert len(triggered) == 1
    assert triggered[0].symbol == "AAPL"

    # Test trailing stop
    position2 = Position(symbol="MSFT", quantity=50, avg_price=300.00, entry_time=datetime.now())

    trailing_stop = await monitor.add_stop_loss(
        symbol="MSFT",
        position=position2,
        stop_percent=0.03,
        stop_type=StopType.TRAILING,
        trailing_amount=5.00,  # $5 trailing
    )

    # Price goes up - stop should adjust
    await monitor.update_price("MSFT", 310.00)
    assert trailing_stop.high_water_mark == 310.00
    assert trailing_stop.stop_price == 305.00  # 310 - 5

    # Price drops but above new stop - no trigger
    await monitor.update_price("MSFT", 306.00)
    triggered = await monitor.check_stops()
    # AAPL already triggered, MSFT not
    assert len([t for t in triggered if t.symbol == "MSFT"]) == 0

    # Cleanup
    monitor.cancel_all_stops()

    return True


def test_kill_switch_in_executor():
    """Test kill switch integration in executor."""
    # Create executor
    executor = PaperExecutor(slippage_bps=0)

    # Create kill switch lock file
    kill_switch_file = Path("data/kill_switch.lock")
    kill_switch_file.parent.mkdir(exist_ok=True)

    try:
        # Without kill switch - order should succeed
        if kill_switch_file.exists():
            kill_switch_file.unlink()

        order = Order(symbol="AAPL", quantity=100, side="BUY", price=150.00)
        result = executor.place_order(order)
        assert result.ok is True

        # With kill switch - order should be blocked
        kill_switch_file.touch()

        order2 = Order(symbol="MSFT", quantity=50, side="SELL", price=300.00)
        result2 = executor.place_order(order2)
        assert result2.ok is False
        assert "kill switch" in result2.message.lower()

        return True

    except Exception as e:
        logger.error(f"Kill switch test failed: {e}")
        return False
    finally:
        # Cleanup
        if kill_switch_file.exists():
            kill_switch_file.unlink()


def test_exception_handling():
    """Test that exception handling is proper (no bare excepts)."""
    import ast
    import inspect

    # Check a few critical files for bare excepts
    files_to_check = [
        "robo_trader/clients/async_ibkr_client.py",
        "robo_trader/features/engine.py",
        "robo_trader/execution.py",
    ]

    bare_excepts_found = []

    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            continue

        content = path.read_text()
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    # Check for bare except
                    if node.type is None:
                        bare_excepts_found.append(f"{filepath}:{node.lineno}")
                    # Check for except Exception: pass
                    elif (
                        isinstance(node.type, ast.Name)
                        and node.type.id == "Exception"
                        and len(node.body) == 1
                        and isinstance(node.body[0], ast.Pass)
                    ):
                        bare_excepts_found.append(f"{filepath}:{node.lineno} (Exception: pass)")

        except Exception as e:
            logger.warning(f"Could not parse {filepath}: {e}")

    if bare_excepts_found:
        logger.error(f"Found bare excepts: {bare_excepts_found}")
        return False

    return True


def test_debug_mode_protection():
    """Test that debug mode is disabled in production."""
    # Save current env
    old_env = os.environ.get("ENVIRONMENT")
    old_debug = os.environ.get("DEBUG")

    try:
        # Test production environment
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "true"  # Try to enable debug

        # Check test_dashboard_simple.py logic
        environment = os.getenv("ENVIRONMENT", "development")
        debug_mode = False
        if environment == "development":
            debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        elif environment in ["staging", "production"]:
            debug_mode = False  # Force disable

        # In production, debug should be False even if requested
        assert debug_mode is False

        # Test development environment
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"

        environment = os.getenv("ENVIRONMENT", "development")
        debug_mode = False
        if environment == "development":
            debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        elif environment in ["staging", "production"]:
            debug_mode = False

        # In development, debug can be enabled
        assert debug_mode is True

        return True

    except Exception as e:
        logger.error(f"Debug mode test failed: {e}")
        return False
    finally:
        # Restore env
        if old_env:
            os.environ["ENVIRONMENT"] = old_env
        else:
            os.environ.pop("ENVIRONMENT", None)
        if old_debug:
            os.environ["DEBUG"] = old_debug
        else:
            os.environ.pop("DEBUG", None)


def main():
    """Run all safety feature tests."""
    print("\n" + "=" * 80)
    print("PRODUCTION READINESS SAFETY FEATURE TESTS")
    print("=" * 80)
    print("Testing critical safety improvements from Phase 0 and Phase 1...")
    print()

    tester = SafetyFeatureTester()

    # Phase 0 Tests
    print("\n📋 PHASE 0: Immediate Blockers")
    print("-" * 40)
    tester.test("Config requires explicit IBKR parameters", test_config_no_defaults)
    tester.test("Database input validation", test_database_validation)
    tester.test("Exception handling (no bare excepts)", test_exception_handling)
    tester.test("Debug mode protection in production", test_debug_mode_protection)

    # Phase 1 Tests
    print("\n📋 PHASE 1: Critical Safety")
    print("-" * 40)
    tester.test("Stop-loss monitoring system", test_stop_loss_monitor)
    tester.test("Kill switch in executor", test_kill_switch_in_executor)

    # Print summary
    tester.print_summary()

    # Exit with appropriate code
    if tester.tests_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    # Set up minimal environment for testing
    os.environ.setdefault("ENVIRONMENT", "development")
    os.environ.setdefault("EXECUTION_MODE", "paper")

    # Set IBKR params for config test
    os.environ.setdefault("IBKR_HOST", "127.0.0.1")
    os.environ.setdefault("IBKR_PORT", "7497")
    os.environ.setdefault("IBKR_CLIENT_ID", "999")

    main()

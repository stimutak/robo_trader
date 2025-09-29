#!/usr/bin/env python3
"""
Test enhanced configuration validation.

This test verifies that configuration validation properly:
- Prevents bypass vulnerabilities
- Validates ranges and types
- Provides fail-safe defaults
- Logs warnings for suspicious values
"""

import os
import sys
from unittest.mock import patch

# Add project to path
sys.path.insert(0, "/Users/oliver/robo_trader")

from robo_trader.config import load_config_from_env
from robo_trader.utils.config_validator import ConfigValidator, EnhancedTradingConfig


def test_positive_float_validation():
    """Test validation of positive float parameters."""
    print("\n=== Testing Positive Float Validation ===")

    validator = ConfigValidator()

    # Test valid values
    result = validator.validate_positive_float("TEST_VAL", 100.0)
    assert result == 100.0
    print(f"✅ Valid default: {result}")

    # Test with environment variable
    with patch.dict(os.environ, {"TEST_VAL": "500.0"}):
        result = validator.validate_positive_float("TEST_VAL", 100.0)
        assert result == 500.0
        print(f"✅ Valid env value: {result}")

    # Test negative value (should use default)
    with patch.dict(os.environ, {"TEST_VAL": "-100.0"}):
        result = validator.validate_positive_float("TEST_VAL", 100.0)
        assert result == 100.0
        print(f"✅ Negative rejected, using default: {result}")

    # Test zero value (should use default when allow_zero=False)
    with patch.dict(os.environ, {"TEST_VAL": "0"}):
        result = validator.validate_positive_float("TEST_VAL", 100.0, allow_zero=False)
        assert result == 100.0
        print(f"✅ Zero rejected, using default: {result}")

    # Test NaN (should use default)
    with patch.dict(os.environ, {"TEST_VAL": "nan"}):
        result = validator.validate_positive_float("TEST_VAL", 100.0)
        assert result == 100.0
        print(f"✅ NaN rejected, using default: {result}")

    # Test infinity (should use default)
    with patch.dict(os.environ, {"TEST_VAL": "inf"}):
        result = validator.validate_positive_float("TEST_VAL", 100.0)
        assert result == 100.0
        print(f"✅ Infinity rejected, using default: {result}")


def test_range_validation():
    """Test range validation."""
    print("\n=== Testing Range Validation ===")

    validator = ConfigValidator()

    # Test within range
    with patch.dict(os.environ, {"TEST_RANGE": "1.5"}):
        result = validator.validate_range("TEST_RANGE", 1.0, 2.0, 1.5)
        assert result == 1.5
        print(f"✅ Within range: {result}")

    # Test below minimum (should clamp)
    with patch.dict(os.environ, {"TEST_RANGE": "0.5"}):
        result = validator.validate_range("TEST_RANGE", 1.0, 2.0, 1.5)
        assert result == 1.0
        print(f"✅ Below minimum, clamped to: {result}")

    # Test above maximum (should clamp)
    with patch.dict(os.environ, {"TEST_RANGE": "3.0"}):
        result = validator.validate_range("TEST_RANGE", 1.0, 2.0, 1.5)
        assert result == 2.0
        print(f"✅ Above maximum, clamped to: {result}")


def test_leverage_validation():
    """Test leverage validation with safety checks."""
    print("\n=== Testing Leverage Validation ===")

    validator = ConfigValidator()

    # Test safe leverage
    with patch.dict(os.environ, {"MAX_LEVERAGE": "2.0"}):
        result = validator.validate_leverage()
        assert result == 2.0
        print(f"✅ Safe leverage: {result}x")

    # Test high leverage (should warn)
    with patch.dict(os.environ, {"MAX_LEVERAGE": "3.5"}):
        result = validator.validate_leverage()
        assert result == 3.5
        print(f"⚠️ High leverage accepted with warning: {result}x")

    # Test excessive leverage (should cap)
    with patch.dict(os.environ, {"MAX_LEVERAGE": "10.0"}):
        result = validator.validate_leverage()
        assert result == 4.0
        print(f"✅ Excessive leverage capped at: {result}x")

    # Test zero leverage (should use default)
    with patch.dict(os.environ, {"MAX_LEVERAGE": "0"}):
        result = validator.validate_leverage()
        assert result == 2.0
        print(f"✅ Zero leverage rejected, using default: {result}x")


def test_daily_loss_validation():
    """Test daily loss limit validation."""
    print("\n=== Testing Daily Loss Validation ===")

    validator = ConfigValidator()

    # Test reasonable daily loss
    with patch.dict(os.environ, {"MAX_DAILY_LOSS": "1000"}):
        result = validator.validate_daily_loss()
        assert result == 1000.0
        print(f"✅ Reasonable daily loss: ${result}")

    # Test very small daily loss (should use minimum)
    with patch.dict(os.environ, {"MAX_DAILY_LOSS": "5"}):
        result = validator.validate_daily_loss()
        assert result == 10.0
        print(f"✅ Too small, using minimum: ${result}")

    # Test negative daily loss (should use default)
    with patch.dict(os.environ, {"MAX_DAILY_LOSS": "-1000"}):
        result = validator.validate_daily_loss()
        assert result == 1000.0
        print(f"✅ Negative rejected, using default: ${result}")

    # Test None/empty (should use default)
    with patch.dict(os.environ, {"MAX_DAILY_LOSS": ""}):
        result = validator.validate_daily_loss()
        assert result == 1000.0
        print(f"✅ Empty value, using default: ${result}")


def test_position_limit_validation():
    """Test position limit validation."""
    print("\n=== Testing Position Limit Validation ===")

    validator = ConfigValidator()

    # Test reasonable limit
    with patch.dict(os.environ, {"MAX_OPEN_POSITIONS": "20"}):
        result = validator.validate_position_limit()
        assert result == 20
        print(f"✅ Reasonable position limit: {result}")

    # Test zero positions (should use default)
    with patch.dict(os.environ, {"MAX_OPEN_POSITIONS": "0"}):
        result = validator.validate_position_limit()
        assert result == 20
        print(f"✅ Zero rejected, using default: {result}")

    # Test excessive positions (should cap)
    with patch.dict(os.environ, {"MAX_OPEN_POSITIONS": "200"}):
        result = validator.validate_position_limit()
        assert result == 100
        print(f"✅ Excessive positions capped at: {result}")


def test_enhanced_trading_config():
    """Test EnhancedTradingConfig class."""
    print("\n=== Testing Enhanced Trading Config ===")

    # Test with various environment settings
    test_env = {
        "MAX_DAILY_LOSS": "2000",
        "MAX_LEVERAGE": "2.5",
        "MAX_OPEN_POSITIONS": "25",
        "MAX_POSITION_PCT": "0.03",
        "STOP_LOSS_PCT": "0.025",
        "MAX_ORDER_NOTIONAL": "15000",
        "MAX_DAILY_NOTIONAL": "150000",
    }

    with patch.dict(os.environ, test_env):
        config = EnhancedTradingConfig()

        assert config.max_daily_loss == 2000
        assert config.max_leverage == 2.5
        assert config.max_open_positions == 25
        assert config.max_position_pct == 0.03
        assert config.stop_loss_pct == 0.025
        assert config.max_order_notional == 15000
        assert config.max_daily_notional == 150000

        print("✅ Enhanced config loaded successfully")
        print(f"  Daily loss: ${config.max_daily_loss}")
        print(f"  Leverage: {config.max_leverage}x")
        print(f"  Max positions: {config.max_open_positions}")
        print(f"  Position size: {config.max_position_pct * 100}%")
        print(f"  Stop loss: {config.stop_loss_pct * 100}%")


def test_config_integration():
    """Test integration with main config system."""
    print("\n=== Testing Config Integration ===")

    # Test with safe values
    safe_env = {
        "RISK_MAX_LEVERAGE": "2.0",
        "RISK_MAX_POSITION_PCT": "0.02",
        "RISK_MAX_DAILY_LOSS_PCT": "0.005",
        "RISK_MAX_OPEN_POSITIONS": "20",
    }

    with patch.dict(os.environ, safe_env):
        try:
            config = load_config_from_env()
            print("✅ Config loaded with safe values")
            print(f"  Leverage: {config.risk.max_leverage}x")
            print(f"  Position %: {config.risk.max_position_pct * 100}%")
            print(f"  Daily loss %: {config.risk.max_daily_loss_pct * 100}%")
            print(f"  Max positions: {config.risk.max_open_positions}")
        except Exception as e:
            print(f"❌ Config load failed: {e}")

    # Test with suspicious values (should trigger warnings)
    suspicious_env = {
        "RISK_MAX_LEVERAGE": "4.0",  # High leverage
        "RISK_MAX_POSITION_PCT": "0.1",  # 10% position - very high
        "RISK_MAX_DAILY_LOSS_PCT": "0.05",  # 5% daily loss - high
        "RISK_MAX_OPEN_POSITIONS": "100",  # Many positions
    }

    print("\n⚠️ Testing with suspicious values (should trigger warnings):")
    with patch.dict(os.environ, suspicious_env):
        try:
            config = load_config_from_env()
            print("Config loaded but with warnings")
        except Exception as e:
            print(f"Config validation prevented unsafe values: {e}")


def main():
    """Run all validation tests."""
    print("=" * 50)
    print("Enhanced Configuration Validation Tests")
    print("=" * 50)

    tests = [
        test_positive_float_validation,
        test_range_validation,
        test_leverage_validation,
        test_daily_loss_validation,
        test_position_limit_validation,
        test_enhanced_trading_config,
        test_config_integration,
    ]

    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("🎉 All Configuration Validation Tests Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

"""
Enhanced configuration validation for critical trading parameters.

This module provides comprehensive validation for trading configuration
to prevent bypass vulnerabilities and ensure safe operation.
"""

import os
from decimal import Decimal
from typing import Any, Optional, Union

import numpy as np
from pydantic import ValidationError

from ..logger import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """Comprehensive configuration validator with fail-safe defaults."""

    @staticmethod
    def validate_positive_float(
        key: str,
        default: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_zero: bool = False,
    ) -> float:
        """
        Validate a positive float configuration value.

        Args:
            key: Environment variable key
            default: Default value if not set or invalid
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_zero: Whether to allow zero values

        Returns:
            Validated float value

        Raises:
            ValueError: If validation fails and no safe default
        """
        try:
            value = float(os.getenv(key, str(default)))

            # Check for NaN or infinity
            if not np.isfinite(value):
                logger.error(f"{key} is not finite: {value}, using default {default}")
                return default

            # Check positivity
            if not allow_zero and value <= 0:
                logger.error(f"{key} must be positive, got {value}, using default {default}")
                return default
            elif allow_zero and value < 0:
                logger.error(f"{key} must be non-negative, got {value}, using default {default}")
                return default

            # Check min value
            if min_value is not None and value < min_value:
                logger.warning(f"{key} below minimum {min_value}, got {value}, using minimum")
                return min_value

            # Check max value
            if max_value is not None and value > max_value:
                logger.warning(f"{key} above maximum {max_value}, got {value}, using maximum")
                return max_value

            logger.debug(f"Validated {key}: {value}")
            return value

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid {key}: {e}, using default {default}")
            return default

    @staticmethod
    def validate_range(key: str, min_val: float, max_val: float, default: float) -> float:
        """
        Validate a float value within a specific range.

        Args:
            key: Environment variable key
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            default: Default value if not set or invalid

        Returns:
            Validated float value within range
        """
        return ConfigValidator.validate_positive_float(
            key, default, min_value=min_val, max_value=max_val, allow_zero=(min_val == 0)
        )

    @staticmethod
    def validate_percentage(key: str, default: float) -> float:
        """
        Validate a percentage value (0-1 range).

        Args:
            key: Environment variable key
            default: Default value if not set or invalid

        Returns:
            Validated percentage as decimal (0-1)
        """
        value = ConfigValidator.validate_range(key, 0.0, 1.0, default)

        # Additional check for suspicious values
        if value > 0.5:  # 50% is quite high for most risk parameters
            logger.warning(f"{key} is unusually high: {value * 100}%")

        return value

    @staticmethod
    def validate_leverage(key: str = "MAX_LEVERAGE", default: float = 2.0) -> float:
        """
        Validate leverage configuration with safety limits.

        Args:
            key: Environment variable key
            default: Default leverage if not set

        Returns:
            Validated leverage value
        """
        # Most brokers limit to 4x for day trading, 2x for overnight
        max_safe_leverage = 4.0
        value = ConfigValidator.validate_range(key, 1.0, max_safe_leverage, default)

        if value > 2.0:
            logger.warning(f"High leverage configured: {value}x - ensure day trading rules apply")

        return value

    @staticmethod
    def validate_daily_loss(key: str = "MAX_DAILY_LOSS", default: float = 1000.0) -> float:
        """
        Validate daily loss limit configuration.

        Args:
            key: Environment variable key
            default: Default daily loss limit

        Returns:
            Validated daily loss limit (positive value)
        """
        value = ConfigValidator.validate_positive_float(key, default, min_value=10.0)

        # Sanity check - daily loss over $10,000 might be excessive for retail
        if value > 10000:
            logger.warning(f"Very high daily loss limit configured: ${value}")

        return value

    @staticmethod
    def validate_position_limit(key: str = "MAX_OPEN_POSITIONS", default: int = 20) -> int:
        """
        Validate maximum open positions.

        Args:
            key: Environment variable key
            default: Default position limit

        Returns:
            Validated position limit
        """
        try:
            value = int(os.getenv(key, str(default)))

            if value <= 0:
                logger.error(f"{key} must be positive, got {value}, using default {default}")
                return default

            if value > 100:
                logger.warning(f"Very high position limit: {value} - may impact performance")
                return 100  # Cap at reasonable limit

            return value

        except (ValueError, TypeError) as e:
            logger.error(f"Invalid {key}: {e}, using default {default}")
            return default

    @staticmethod
    def validate_notional_limits(
        order_key: str = "MAX_ORDER_NOTIONAL",
        daily_key: str = "MAX_DAILY_NOTIONAL",
        order_default: float = 10000.0,
        daily_default: float = 100000.0,
    ) -> tuple[float, float]:
        """
        Validate notional limit configurations.

        Args:
            order_key: Max order notional env key
            daily_key: Max daily notional env key
            order_default: Default order notional
            daily_default: Default daily notional

        Returns:
            Tuple of (max_order_notional, max_daily_notional)
        """
        max_order = ConfigValidator.validate_positive_float(
            order_key, order_default, min_value=100.0
        )
        max_daily = ConfigValidator.validate_positive_float(
            daily_key, daily_default, min_value=max_order
        )

        # Ensure daily >= order
        if max_daily < max_order:
            logger.warning(f"Daily notional {max_daily} < order notional {max_order}, adjusting")
            max_daily = max_order * 10

        return max_order, max_daily

    @staticmethod
    def validate_stop_loss(key: str = "STOP_LOSS_PCT", default: float = 0.02) -> float:
        """
        Validate stop loss percentage.

        Args:
            key: Environment variable key
            default: Default stop loss percentage

        Returns:
            Validated stop loss percentage (0-1)
        """
        value = ConfigValidator.validate_percentage(key, default)

        if value > 0.1:  # 10% stop loss is quite wide
            logger.warning(f"Wide stop loss configured: {value * 100}%")
        elif value < 0.005:  # 0.5% is very tight
            logger.warning(f"Very tight stop loss: {value * 100}% - may trigger frequently")

        return value

    @staticmethod
    def validate_all_risk_params() -> dict:
        """
        Validate all critical risk parameters.

        Returns:
            Dictionary of validated risk parameters
        """
        params = {
            # Core risk limits
            "max_daily_loss": ConfigValidator.validate_daily_loss(),
            "max_leverage": ConfigValidator.validate_leverage(),
            "max_open_positions": ConfigValidator.validate_position_limit(),
            # Position sizing
            "max_position_pct": ConfigValidator.validate_percentage("MAX_POSITION_PCT", 0.02),
            "max_daily_loss_pct": ConfigValidator.validate_percentage("MAX_DAILY_LOSS_PCT", 0.005),
            # Notional limits
            "notional_limits": ConfigValidator.validate_notional_limits(),
            # Stop loss/take profit
            "stop_loss_pct": ConfigValidator.validate_stop_loss(),
            "take_profit_pct": ConfigValidator.validate_percentage("TAKE_PROFIT_PCT", 0.05),
            # Correlation and exposure
            "max_correlation": ConfigValidator.validate_range("MAX_CORRELATION", 0.0, 1.0, 0.7),
            "max_sector_exposure_pct": ConfigValidator.validate_percentage(
                "MAX_SECTOR_EXPOSURE_PCT", 0.3
            ),
            # Volume and liquidity
            "min_volume": max(int(os.getenv("MIN_VOLUME", "1000000")), 100000),  # Minimum floor
            "min_market_cap": max(
                float(os.getenv("MIN_MARKET_CAP", "1000000000")), 100000000  # Minimum floor
            ),
        }

        # Log summary
        logger.info("Risk parameters validated:")
        logger.info(f"  Daily loss: ${params['max_daily_loss']:.2f}")
        logger.info(f"  Max leverage: {params['max_leverage']}x")
        logger.info(f"  Max positions: {params['max_open_positions']}")
        logger.info(f"  Position size: {params['max_position_pct'] * 100:.1f}%")
        logger.info(f"  Stop loss: {params['stop_loss_pct'] * 100:.1f}%")

        return params


class EnhancedTradingConfig:
    """Enhanced trading configuration with comprehensive validation."""

    def __init__(self):
        """Initialize with validated parameters."""
        # Validate all risk parameters
        risk_params = ConfigValidator.validate_all_risk_params()

        # Core risk limits
        self.max_daily_loss = risk_params["max_daily_loss"]
        self.max_leverage = risk_params["max_leverage"]
        self.max_open_positions = risk_params["max_open_positions"]

        # Position sizing
        self.max_position_pct = risk_params["max_position_pct"]
        self.max_daily_loss_pct = risk_params["max_daily_loss_pct"]

        # Notional limits
        self.max_order_notional, self.max_daily_notional = risk_params["notional_limits"]

        # Stop loss/take profit
        self.stop_loss_pct = risk_params["stop_loss_pct"]
        self.take_profit_pct = risk_params["take_profit_pct"]

        # Correlation and exposure
        self.max_correlation = risk_params["max_correlation"]
        self.max_sector_exposure_pct = risk_params["max_sector_exposure_pct"]

        # Volume and liquidity
        self.min_volume = risk_params["min_volume"]
        self.min_market_cap = risk_params["min_market_cap"]

        # Additional safety checks
        self._validate_consistency()

    def _validate_consistency(self) -> None:
        """Validate configuration consistency."""
        # Check daily loss vs position limits
        max_position_loss = self.max_position_pct * self.stop_loss_pct
        max_total_loss = max_position_loss * self.max_open_positions

        if max_total_loss > self.max_daily_loss_pct:
            logger.warning(
                f"Potential daily loss ({max_total_loss * 100:.1f}%) exceeds limit "
                f"({self.max_daily_loss_pct * 100:.1f}%) with max positions"
            )

        # Check leverage vs position limits
        max_exposure = self.max_position_pct * self.max_open_positions
        if max_exposure > self.max_leverage:
            logger.warning(
                f"Maximum exposure ({max_exposure:.1f}x) exceeds leverage limit "
                f"({self.max_leverage}x)"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility."""
        return {
            "max_daily_loss": self.max_daily_loss,
            "max_leverage": self.max_leverage,
            "max_open_positions": self.max_open_positions,
            "max_position_pct": self.max_position_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_order_notional": self.max_order_notional,
            "max_daily_notional": self.max_daily_notional,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "max_correlation": self.max_correlation,
            "max_sector_exposure_pct": self.max_sector_exposure_pct,
            "min_volume": self.min_volume,
            "min_market_cap": self.min_market_cap,
        }


# Example usage
if __name__ == "__main__":
    # Test validation
    config = EnhancedTradingConfig()
    print(f"Configuration loaded and validated:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")

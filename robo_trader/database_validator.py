"""
Database Input Validation Layer

This module provides comprehensive validation for all database inputs to prevent
SQL injection, data corruption, and ensure data integrity. All database operations
must pass through this validation layer before execution.

CRITICAL: This is a safety-critical component. All database inputs MUST be validated.
"""

import re
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


class OrderSide(str, Enum):
    """Valid order sides."""

    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class OrderType(str, Enum):
    """Valid order types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class DatabaseValidator:
    """
    Comprehensive database input validation.

    This class provides validation for all types of data that may be stored
    in the database, preventing SQL injection and ensuring data integrity.
    """

    # Valid symbol pattern: 1-5 uppercase letters, optionally followed by dot and 1-2 letters
    SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}(\.[A-Z]{1,2})?$")

    # Maximum reasonable values for sanity checks
    MAX_PRICE = 1_000_000.0  # $1M per share max
    MIN_PRICE = 0.0001  # $0.0001 min (penny stocks)
    MAX_QUANTITY = 1_000_000  # 1M shares max per order
    MIN_QUANTITY = 1  # At least 1 share
    MAX_NOTIONAL = 100_000_000.0  # $100M max per trade
    MAX_DAILY_VOLUME = 10_000_000_000  # 10B shares daily volume max

    # Time constraints
    MAX_DATA_AGE_SECONDS = 86400  # 24 hours
    MIN_TIMESTAMP = datetime(2000, 1, 1)  # No data before year 2000

    @staticmethod
    def validate_symbol(symbol: Any) -> str:
        """
        Validate a trading symbol.

        Args:
            symbol: The symbol to validate

        Returns:
            str: The validated symbol

        Raises:
            ValidationError: If symbol is invalid
        """
        if not symbol:
            raise ValidationError("Symbol cannot be empty")

        if not isinstance(symbol, str):
            raise ValidationError(f"Symbol must be a string, got {type(symbol).__name__}")

        symbol = symbol.strip().upper()

        if not DatabaseValidator.SYMBOL_PATTERN.match(symbol):
            raise ValidationError(
                f"Invalid symbol format: '{symbol}'. "
                "Must be 1-5 uppercase letters, optionally followed by .XX"
            )

        # Check for SQL injection attempts
        if any(char in symbol for char in ["'", '"', ";", "--", "/*", "*/", "DROP", "DELETE"]):
            logger.error(f"Potential SQL injection attempt in symbol: {symbol}")
            raise ValidationError("Invalid characters in symbol")

        return symbol

    @staticmethod
    def validate_price(
        price: Any, min_val: float = None, max_val: float = None, field_name: str = "price"
    ) -> float:
        """
        Validate a price value.

        Args:
            price: The price to validate
            min_val: Minimum allowed value (default: MIN_PRICE)
            max_val: Maximum allowed value (default: MAX_PRICE)
            field_name: Name of the field for error messages

        Returns:
            float: The validated price

        Raises:
            ValidationError: If price is invalid
        """
        if price is None:
            raise ValidationError(f"{field_name} cannot be None")

        try:
            # Use Decimal for precision
            price_decimal = Decimal(str(price))
            price_float = float(price_decimal)
        except (InvalidOperation, ValueError, TypeError) as e:
            raise ValidationError(f"Invalid {field_name}: {price} - {e}")

        if not min_val:
            min_val = DatabaseValidator.MIN_PRICE
        if not max_val:
            max_val = DatabaseValidator.MAX_PRICE

        if price_float < min_val:
            raise ValidationError(f"{field_name} {price_float} below minimum {min_val}")

        if price_float > max_val:
            raise ValidationError(f"{field_name} {price_float} exceeds maximum {max_val}")

        if price_float != price_float:  # NaN check
            raise ValidationError(f"{field_name} is NaN")

        return price_float

    @staticmethod
    def validate_quantity(quantity: Any, allow_negative: bool = False, max_val: int = None) -> int:
        """
        Validate a quantity value.

        Args:
            quantity: The quantity to validate
            allow_negative: Whether to allow negative quantities (for short positions)
            max_val: Maximum allowed value

        Returns:
            int: The validated quantity

        Raises:
            ValidationError: If quantity is invalid
        """
        if quantity is None:
            raise ValidationError("Quantity cannot be None")

        try:
            qty_int = int(quantity)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid quantity: {quantity} - {e}")

        if not allow_negative and qty_int < DatabaseValidator.MIN_QUANTITY:
            raise ValidationError(
                f"Quantity {qty_int} must be at least {DatabaseValidator.MIN_QUANTITY}"
            )

        if allow_negative and qty_int == 0:
            raise ValidationError("Quantity cannot be zero")

        max_check = max_val if max_val else DatabaseValidator.MAX_QUANTITY
        if abs(qty_int) > max_check:
            raise ValidationError(f"Quantity {qty_int} exceeds maximum {max_check}")

        return qty_int

    @staticmethod
    def validate_order_side(side: Any) -> str:
        """
        Validate an order side.

        Args:
            side: The order side to validate

        Returns:
            str: The validated order side

        Raises:
            ValidationError: If side is invalid
        """
        if not side:
            raise ValidationError("Order side cannot be empty")

        if not isinstance(side, str):
            raise ValidationError(f"Order side must be a string, got {type(side).__name__}")

        side = side.strip().upper()

        try:
            return OrderSide(side).value
        except ValueError:
            valid_sides = [s.value for s in OrderSide]
            raise ValidationError(f"Invalid order side: '{side}'. Must be one of {valid_sides}")

    @staticmethod
    def validate_timestamp(
        timestamp: Any, max_age_seconds: int = None, allow_future: bool = False
    ) -> datetime:
        """
        Validate a timestamp.

        Args:
            timestamp: The timestamp to validate
            max_age_seconds: Maximum age in seconds (None for no limit)
            allow_future: Whether to allow future timestamps

        Returns:
            datetime: The validated timestamp

        Raises:
            ValidationError: If timestamp is invalid
        """
        if timestamp is None:
            return datetime.now()

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError as e:
                raise ValidationError(f"Invalid timestamp string: {timestamp} - {e}")
        elif isinstance(timestamp, (int, float)):
            try:
                timestamp = datetime.fromtimestamp(timestamp)
            except (ValueError, OSError) as e:
                raise ValidationError(f"Invalid timestamp number: {timestamp} - {e}")
        elif not isinstance(timestamp, datetime):
            raise ValidationError(
                f"Timestamp must be datetime, string, or number, got {type(timestamp).__name__}"
            )

        # Check bounds
        if timestamp < DatabaseValidator.MIN_TIMESTAMP:
            raise ValidationError(f"Timestamp {timestamp} is before year 2000")

        now = datetime.now()
        if not allow_future and timestamp > now:
            raise ValidationError(f"Timestamp {timestamp} is in the future")

        if max_age_seconds:
            max_age = timedelta(seconds=max_age_seconds)
            if now - timestamp > max_age:
                raise ValidationError(
                    f"Timestamp {timestamp} is older than {max_age_seconds} seconds"
                )

        return timestamp

    @staticmethod
    def validate_trade_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete trade data.

        Args:
            data: Trade data dictionary

        Returns:
            Dict: Validated and sanitized trade data

        Raises:
            ValidationError: If any field is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Trade data must be a dictionary, got {type(data).__name__}")

        validated = {}

        # Required fields
        validated["symbol"] = DatabaseValidator.validate_symbol(data.get("symbol"))
        validated["quantity"] = DatabaseValidator.validate_quantity(data.get("quantity"))
        validated["price"] = DatabaseValidator.validate_price(data.get("price"))
        validated["side"] = DatabaseValidator.validate_order_side(data.get("side"))

        # Optional fields
        if "timestamp" in data:
            validated["timestamp"] = DatabaseValidator.validate_timestamp(data["timestamp"])
        else:
            validated["timestamp"] = datetime.now()

        if "order_id" in data:
            validated["order_id"] = DatabaseValidator._validate_string(
                data["order_id"], "order_id", max_length=50
            )

        if "commission" in data:
            validated["commission"] = DatabaseValidator.validate_price(
                data["commission"], min_val=0, max_val=1000, field_name="commission"
            )

        # Calculate and validate notional
        notional = abs(validated["quantity"] * validated["price"])
        if notional > DatabaseValidator.MAX_NOTIONAL:
            raise ValidationError(
                f"Trade notional ${notional:,.2f} exceeds maximum ${DatabaseValidator.MAX_NOTIONAL:,.2f}"
            )

        validated["notional"] = notional

        return validated

    @staticmethod
    def validate_position_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate position data.

        Args:
            data: Position data dictionary

        Returns:
            Dict: Validated and sanitized position data

        Raises:
            ValidationError: If any field is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Position data must be a dictionary, got {type(data).__name__}")

        validated = {}

        # Required fields
        validated["symbol"] = DatabaseValidator.validate_symbol(data.get("symbol"))
        validated["quantity"] = DatabaseValidator.validate_quantity(
            data.get("quantity"), allow_negative=True
        )
        validated["avg_cost"] = DatabaseValidator.validate_price(
            data.get("avg_cost"), field_name="avg_cost"
        )

        # Optional fields
        if "current_price" in data:
            validated["current_price"] = DatabaseValidator.validate_price(
                data["current_price"], field_name="current_price"
            )

        if "unrealized_pnl" in data:
            validated["unrealized_pnl"] = DatabaseValidator._validate_numeric(
                data["unrealized_pnl"], "unrealized_pnl", min_val=-DatabaseValidator.MAX_NOTIONAL
            )

        if "realized_pnl" in data:
            validated["realized_pnl"] = DatabaseValidator._validate_numeric(
                data["realized_pnl"], "realized_pnl", min_val=-DatabaseValidator.MAX_NOTIONAL
            )

        return validated

    @staticmethod
    def validate_market_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate market data (OHLCV).

        Args:
            data: Market data dictionary

        Returns:
            Dict: Validated and sanitized market data

        Raises:
            ValidationError: If any field is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Market data must be a dictionary, got {type(data).__name__}")

        validated = {}

        # Symbol
        validated["symbol"] = DatabaseValidator.validate_symbol(data.get("symbol"))

        # OHLC prices
        open_price = DatabaseValidator.validate_price(data.get("open"), field_name="open")
        high_price = DatabaseValidator.validate_price(data.get("high"), field_name="high")
        low_price = DatabaseValidator.validate_price(data.get("low"), field_name="low")
        close_price = DatabaseValidator.validate_price(data.get("close"), field_name="close")

        # Validate OHLC relationships
        if low_price > high_price:
            raise ValidationError(f"Low price {low_price} cannot exceed high price {high_price}")

        if open_price < low_price or open_price > high_price:
            raise ValidationError(
                f"Open price {open_price} must be between low {low_price} and high {high_price}"
            )

        if close_price < low_price or close_price > high_price:
            raise ValidationError(
                f"Close price {close_price} must be between low {low_price} and high {high_price}"
            )

        validated["open"] = open_price
        validated["high"] = high_price
        validated["low"] = low_price
        validated["close"] = close_price

        # Volume
        if "volume" in data:
            volume = DatabaseValidator._validate_numeric(
                data["volume"], "volume", min_val=0, max_val=DatabaseValidator.MAX_DAILY_VOLUME
            )
            validated["volume"] = int(volume)

        # Timestamp
        validated["timestamp"] = DatabaseValidator.validate_timestamp(
            data.get("timestamp"), max_age_seconds=DatabaseValidator.MAX_DATA_AGE_SECONDS
        )

        return validated

    @staticmethod
    def validate_account_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate account data.

        Args:
            data: Account data dictionary

        Returns:
            Dict: Validated and sanitized account data

        Raises:
            ValidationError: If any field is invalid
        """
        if not isinstance(data, dict):
            raise ValidationError(f"Account data must be a dictionary, got {type(data).__name__}")

        validated = {}

        # Cash and equity
        if "cash" in data:
            validated["cash"] = DatabaseValidator._validate_numeric(
                data["cash"], "cash", min_val=0, max_val=1e9  # Max $1B
            )

        if "equity" in data:
            validated["equity"] = DatabaseValidator._validate_numeric(
                data["equity"], "equity", min_val=0, max_val=1e9
            )

        # P&L values (can be negative)
        if "daily_pnl" in data:
            validated["daily_pnl"] = DatabaseValidator._validate_numeric(
                data["daily_pnl"], "daily_pnl", min_val=-1e6, max_val=1e6  # +/- $1M daily
            )

        if "realized_pnl" in data:
            validated["realized_pnl"] = DatabaseValidator._validate_numeric(
                data["realized_pnl"], "realized_pnl", min_val=-1e8, max_val=1e8
            )

        if "unrealized_pnl" in data:
            validated["unrealized_pnl"] = DatabaseValidator._validate_numeric(
                data["unrealized_pnl"], "unrealized_pnl", min_val=-1e8, max_val=1e8
            )

        return validated

    @staticmethod
    def _validate_string(
        value: Any, field_name: str, max_length: int = 255, allow_empty: bool = False
    ) -> str:
        """
        Validate a string field.

        Args:
            value: The string to validate
            field_name: Name of the field for error messages
            max_length: Maximum allowed length
            allow_empty: Whether to allow empty strings

        Returns:
            str: The validated string

        Raises:
            ValidationError: If string is invalid
        """
        if value is None:
            if allow_empty:
                return ""
            raise ValidationError(f"{field_name} cannot be None")

        if not isinstance(value, str):
            value = str(value)

        value = value.strip()

        if not allow_empty and not value:
            raise ValidationError(f"{field_name} cannot be empty")

        if len(value) > max_length:
            raise ValidationError(f"{field_name} exceeds maximum length of {max_length}")

        # Check for SQL injection attempts
        dangerous_patterns = ["'", '"', ";", "--", "/*", "*/", "DROP", "DELETE", "INSERT", "UPDATE"]
        for pattern in dangerous_patterns:
            if pattern in value.upper():
                logger.warning(f"Potential SQL injection in {field_name}: {value}")
                # Escape rather than reject for non-critical fields
                value = value.replace("'", "''").replace('"', '""')

        return value

    @staticmethod
    def _validate_numeric(
        value: Any, field_name: str, min_val: float = None, max_val: float = None
    ) -> float:
        """
        Validate a numeric field.

        Args:
            value: The value to validate
            field_name: Name of the field for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            float: The validated number

        Raises:
            ValidationError: If value is invalid
        """
        if value is None:
            raise ValidationError(f"{field_name} cannot be None")

        try:
            num_float = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Invalid {field_name}: {value} - {e}")

        if num_float != num_float:  # NaN check
            raise ValidationError(f"{field_name} is NaN")

        if min_val is not None and num_float < min_val:
            raise ValidationError(f"{field_name} {num_float} below minimum {min_val}")

        if max_val is not None and num_float > max_val:
            raise ValidationError(f"{field_name} {num_float} exceeds maximum {max_val}")

        return num_float

    @staticmethod
    def sanitize_for_logging(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data for safe logging (remove sensitive information).

        Args:
            data: Data to sanitize

        Returns:
            Dict: Sanitized data safe for logging
        """
        if not isinstance(data, dict):
            return {"type": type(data).__name__, "value": "[REDACTED]"}

        sanitized = {}
        sensitive_fields = ["password", "api_key", "token", "secret", "account"]

        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = DatabaseValidator.sanitize_for_logging(value)
            else:
                sanitized[key] = value

        return sanitized

"""
Custom exceptions for RoboTrader.

This module defines a hierarchy of exceptions for better error handling
and to replace generic `except Exception` blocks with specific types.

Usage:
    from robo_trader.exceptions import (
        IBKRConnectionError,
        IBKRTimeoutError,
        DataValidationError,
        TradingError,
    )

    try:
        await client.connect()
    except IBKRConnectionError as e:
        # Handle connection failures
        logger.error(f"IBKR connection failed: {e}")
    except IBKRTimeoutError as e:
        # Handle timeouts specifically
        logger.warning(f"IBKR timeout, will retry: {e}")
"""


class RoboTraderError(Exception):
    """Base exception for all RoboTrader errors."""

    pass


# =============================================================================
# IBKR Connection Errors
# =============================================================================


class IBKRError(RoboTraderError):
    """Base exception for IBKR-related errors."""

    pass


class IBKRConnectionError(IBKRError):
    """Failed to connect to IBKR Gateway/TWS."""

    pass


class IBKRTimeoutError(IBKRError):
    """IBKR operation timed out."""

    pass


class IBKRDisconnectedError(IBKRError):
    """IBKR connection was lost unexpectedly."""

    pass


class IBKRRateLimitError(IBKRError):
    """Too many requests to IBKR API."""

    pass


class IBKRDataError(IBKRError):
    """Error retrieving market data from IBKR."""

    pass


# =============================================================================
# Trading Errors
# =============================================================================


class TradingError(RoboTraderError):
    """Base exception for trading-related errors."""

    pass


class OrderError(TradingError):
    """Error placing or managing an order."""

    pass


class OrderRejectedError(OrderError):
    """Order was rejected by the broker."""

    pass


class PositionError(TradingError):
    """Error with position management."""

    pass


class InsufficientFundsError(TradingError):
    """Insufficient funds for the requested operation."""

    pass


class RiskLimitExceededError(TradingError):
    """Operation would exceed risk limits."""

    pass


# =============================================================================
# Data Errors
# =============================================================================


class DataError(RoboTraderError):
    """Base exception for data-related errors."""

    pass


class DataValidationError(DataError):
    """Data failed validation checks."""

    pass


class DataStaleError(DataError):
    """Data is too old to be usable."""

    pass


class DataMissingError(DataError):
    """Required data is missing."""

    pass


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(RoboTraderError):
    """Error in configuration."""

    pass


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration is missing."""

    pass


# =============================================================================
# Strategy Errors
# =============================================================================


class StrategyError(RoboTraderError):
    """Base exception for strategy-related errors."""

    pass


class StrategyInitError(StrategyError):
    """Failed to initialize strategy."""

    pass


class SignalGenerationError(StrategyError):
    """Failed to generate trading signal."""

    pass


# =============================================================================
# ML Errors
# =============================================================================


class MLError(RoboTraderError):
    """Base exception for ML-related errors."""

    pass


class ModelNotFoundError(MLError):
    """ML model file not found."""

    pass


class ModelLoadError(MLError):
    """Failed to load ML model."""

    pass


class PredictionError(MLError):
    """Failed to generate ML prediction."""

    pass


# =============================================================================
# Circuit Breaker Errors
# =============================================================================


class CircuitBreakerError(RoboTraderError):
    """Circuit breaker is open, operation blocked."""

    pass


class CircuitBreakerOpenError(CircuitBreakerError):
    """Circuit breaker is in OPEN state."""

    pass

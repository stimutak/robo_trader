"""
Enhanced logging system with structured JSON output and comprehensive context.

This module provides structured logging with:
- JSON format for log aggregation
- Contextual information (trade, risk, performance)
- Log rotation and management
- Performance metrics logging
- Audit trail for compliance
"""

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.processors import CallsiteParameter

# Global flag to track configuration
_CONFIGURED = False

# Source loggers whose output must NEVER be forwarded to WebSocket clients
# (these are authoritative/security-sensitive streams).
WS_LOGGER_DENYLIST = (
    "audit",
    "security",
    "robo_trader.production",
    "robo_trader.security",
)

# High-entropy / known-shape secret patterns that may appear in arbitrary log
# values. These are scrubbed in addition to key-name matching.
_SECRET_VALUE_PATTERNS = [
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),  # GitHub tokens
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access keys
    re.compile(
        r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"
    ),  # JWTs
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"(?i)password[=:\s]+\S{4,}"),
]


def _scrub_value(v: Any) -> Any:
    """Redact known secret patterns inside string values."""
    if not isinstance(v, str):
        return v
    for pat in _SECRET_VALUE_PATTERNS:
        v = pat.sub("***REDACTED***", v)
    return v


class LogEvent(str, Enum):
    """Standard log events for trading system."""

    # Trade events
    TRADE_PLACED = "trade.placed"
    TRADE_EXECUTED = "trade.executed"
    TRADE_CANCELLED = "trade.cancelled"
    TRADE_REJECTED = "trade.rejected"

    # Risk events
    RISK_VIOLATION = "risk.violation"
    RISK_WARNING = "risk.warning"
    RISK_CLEARED = "risk.cleared"
    EMERGENCY_SHUTDOWN = "risk.emergency_shutdown"

    # Market data events
    DATA_RECEIVED = "data.received"
    DATA_ERROR = "data.error"
    DATA_GAP = "data.gap"

    # System events
    ENGINE_STARTED = "engine.started"
    ENGINE_STOPPED = "engine.stopped"
    CONNECTION_ESTABLISHED = "connection.established"
    CONNECTION_LOST = "connection.lost"
    HEALTH_CHECK = "health.check"

    # Strategy events
    SIGNAL_GENERATED = "signal.generated"
    SIGNAL_IGNORED = "signal.ignored"
    STRATEGY_ERROR = "strategy.error"

    # Performance events
    PERFORMANCE_METRIC = "performance.metric"
    PORTFOLIO_UPDATE = "portfolio.update"
    PNL_UPDATE = "pnl.update"


class WebSocketLogProcessor:
    """Structlog processor that forwards logs to WebSocket for real-time streaming.

    Uses late binding to avoid circular imports. Supports two modes:
    1. Server mode (ws_manager): When this process hosts the WebSocket server
    2. Client mode (ws_client): When connecting to an external WebSocket server

    The processor automatically falls back to client mode if the server isn't running.
    """

    _ws_manager = None
    _ws_client = None

    @classmethod
    def set_ws_manager(cls, ws_manager):
        """Set the WebSocket manager instance for log streaming (server mode)."""
        cls._ws_manager = ws_manager

    @classmethod
    def set_ws_client(cls, ws_client):
        """Set the WebSocket client instance for log streaming (client mode)."""
        cls._ws_client = ws_client

    def __call__(self, logger, method_name, event_dict):
        """Forward log message to WebSocket if manager or client is configured."""
        level = method_name.upper()

        # Only stream INFO and above to avoid flooding clients
        if level not in ("INFO", "WARNING", "ERROR", "CRITICAL"):
            return event_dict

        # Never forward audit/security/production logs to WebSocket clients.
        source = str(event_dict.get("logger", ""))
        for prefix in WS_LOGGER_DENYLIST:
            if source == prefix or source.startswith(prefix + "."):
                return event_dict

        # Try server mode first (if ws_manager is running)
        if self._ws_manager is not None:
            thread = getattr(self._ws_manager, "thread", None)
            if thread is not None and thread.is_alive():
                try:
                    self._ws_manager.send_log_message(
                        level=level,
                        source=event_dict.get("logger", "robo_trader"),
                        message=event_dict.get("event", str(event_dict)),
                        context={
                            k: v
                            for k, v in event_dict.items()
                            if k not in ("event", "logger", "timestamp", "level")
                        },
                    )
                    return event_dict
                except Exception:
                    pass  # Fall through to client mode

        # Fall back to client mode (connect to external WebSocket server)
        if self._ws_client is not None:
            try:
                self._ws_client.send_log_message(
                    level=level,
                    source=event_dict.get("logger", "robo_trader"),
                    message=event_dict.get("event", str(event_dict)),
                    context={
                        k: v
                        for k, v in event_dict.items()
                        if k not in ("event", "logger", "timestamp", "level")
                    },
                )
            except Exception:
                pass  # Don't let log streaming failures break logging

        return event_dict


def add_timestamp(logger, method_name, event_dict):
    """Add timestamp to all log entries."""
    # Use local timezone instead of UTC
    event_dict["timestamp"] = datetime.now().isoformat()
    return event_dict


def add_hostname(logger, method_name, event_dict):
    """Add hostname to identify log source."""
    import socket

    event_dict["hostname"] = socket.gethostname()
    return event_dict


def add_environment(logger, method_name, event_dict):
    """Add environment information."""
    event_dict["environment"] = os.getenv("ENVIRONMENT", "dev")
    event_dict["trading_mode"] = os.getenv("EXECUTION_MODE", "paper")
    return event_dict


def censor_sensitive(logger, method_name, event_dict):
    """Censor sensitive information like API keys.

    Strategy:
    1. Redact entire values whose KEY matches a sensitive name.
    2. Scrub VALUES of all string fields (and the message) for known secret
       shapes — this catches secrets accidentally interpolated into log
       messages by callers.
    """
    sensitive_keys = ["password", "api_key", "secret", "token", "credential"]

    for key in list(event_dict.keys()):
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
            continue
        # Scrub value-level patterns for any string field
        event_dict[key] = _scrub_value(event_dict[key])

    return event_dict


def setup_structlog():
    """Configure structlog for the application."""

    # Determine output format
    log_format = os.getenv("MONITORING_LOG_FORMAT", "json").lower()

    # Common processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        add_timestamp,
        add_hostname,
        add_environment,
        censor_sensitive,
        structlog.processors.CallsiteParameterAdder(
            parameters=[
                CallsiteParameter.FILENAME,
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
            ]
        ),
        # WebSocket log streaming (before final renderer)
        WebSocketLogProcessor(),
    ]

    # Format-specific processors
    if log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_stdlib_logging():
    """Configure standard library logging."""

    # Get configuration from environment
    level_str = os.getenv("MONITORING_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = os.getenv("MONITORING_LOG_FORMAT", "json").lower()

    # Create formatter based on format type
    if log_format == "json":
        # JSON formatter for stdlib logs
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.now().isoformat(),  # Use local time, not UTC
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                }

                if record.exc_info:
                    log_obj["exception"] = self.formatException(record.exc_info)

                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in [
                        "name",
                        "msg",
                        "args",
                        "created",
                        "filename",
                        "funcName",
                        "levelname",
                        "levelno",
                        "lineno",
                        "module",
                        "msecs",
                        "pathname",
                        "process",
                        "processName",
                        "relativeCreated",
                        "thread",
                        "threadName",
                        "exc_info",
                        "exc_text",
                        "stack_info",
                    ]:
                        log_obj[key] = value

                return json.dumps(log_obj)

        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler with rotation (if enabled)
    log_file = os.getenv("LOG_FILE")
    if log_file:
        from logging.handlers import RotatingFileHandler

        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    global _CONFIGURED

    if not _CONFIGURED:
        configure_stdlib_logging()
        setup_structlog()
        _CONFIGURED = True

    return structlog.get_logger(name)


def log_trade(
    logger: structlog.BoundLogger,
    event: LogEvent,
    symbol: str,
    quantity: int,
    price: float,
    side: str,
    **kwargs,
) -> None:
    """
    Log trade-related events with context.

    Args:
        logger: Logger instance
        event: Trade event type
        symbol: Trading symbol
        quantity: Trade quantity
        price: Trade price
        side: Trade side (BUY/SELL)
        **kwargs: Additional context
    """
    logger.info(
        event.value,
        symbol=symbol,
        quantity=quantity,
        price=price,
        side=side,
        notional=quantity * price,
        **kwargs,
    )


def log_risk_violation(
    logger: structlog.BoundLogger,
    violation_type: str,
    symbol: str,
    message: str,
    **kwargs,
) -> None:
    """
    Log risk violation with context.

    Args:
        logger: Logger instance
        violation_type: Type of violation
        symbol: Related symbol
        message: Violation message
        **kwargs: Additional context
    """
    logger.warning(
        LogEvent.RISK_VIOLATION.value,
        violation_type=violation_type,
        symbol=symbol,
        message=message,
        **kwargs,
    )


def log_performance(
    logger: structlog.BoundLogger, metric_name: str, value: float, **kwargs
) -> None:
    """
    Log performance metrics.

    Args:
        logger: Logger instance
        metric_name: Name of the metric
        value: Metric value
        **kwargs: Additional context
    """
    logger.info(LogEvent.PERFORMANCE_METRIC.value, metric=metric_name, value=value, **kwargs)


def log_system_event(
    logger: structlog.BoundLogger, event: LogEvent, message: str, **kwargs
) -> None:
    """
    Log system events.

    Args:
        logger: Logger instance
        event: System event type
        message: Event message
        **kwargs: Additional context
    """
    logger.info(event.value, message=message, **kwargs)


def create_audit_logger(name: str = "audit") -> structlog.BoundLogger:
    """
    Create a logger specifically for audit trail.

    Args:
        name: Audit logger name

    Returns:
        Configured audit logger
    """
    audit_logger = get_logger(name)

    # Bind audit-specific context
    audit_logger = audit_logger.bind(
        log_type="audit",
        compliance=True,
    )

    return audit_logger


def create_performance_logger(name: str = "performance") -> structlog.BoundLogger:
    """
    Create a logger specifically for performance metrics.

    Args:
        name: Performance logger name

    Returns:
        Configured performance logger
    """
    perf_logger = get_logger(name)

    # Bind performance-specific context
    perf_logger = perf_logger.bind(
        log_type="performance",
        metrics=True,
    )

    return perf_logger


class LogContext:
    """Context manager for adding temporary log context."""

    def __init__(self, logger: structlog.BoundLogger, **kwargs):
        """
        Initialize log context.

        Args:
            logger: Logger to bind context to
            **kwargs: Context to bind
        """
        self.logger = logger
        self.context = kwargs
        self.original_logger = None

    def __enter__(self):
        """Enter context and bind values."""
        self.original_logger = self.logger
        self.logger = self.logger.bind(**self.context)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore original logger."""
        self.logger = self.original_logger


# Convenience function for backward compatibility
def _configure_root_logger() -> None:
    """Configure root logger (backward compatibility)."""
    global _CONFIGURED
    if not _CONFIGURED:
        configure_stdlib_logging()
        setup_structlog()
        _CONFIGURED = True


# Export convenience functions
__all__ = [
    "get_logger",
    "LogEvent",
    "log_trade",
    "log_risk_violation",
    "log_performance",
    "log_system_event",
    "create_audit_logger",
    "create_performance_logger",
    "LogContext",
    "WebSocketLogProcessor",
]

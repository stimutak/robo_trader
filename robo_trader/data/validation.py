"""
Data validation and quality checks for market data.

This module implements:
- Tick data validation
- Outlier detection
- Corporate action detection
- Market hours validation
- Data feed health monitoring
- Gap detection and handling
"""

import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..logger import get_logger
from .pipeline import BarData, TickData


@dataclass
class ValidationResult:
    """Result of data validation check."""

    is_valid: bool
    check_name: str
    message: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    data: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.check_name}: {self.message}"


@dataclass
class DataQualityMetrics:
    """Metrics for data quality monitoring."""

    total_ticks: int = 0
    valid_ticks: int = 0
    invalid_ticks: int = 0
    outliers_detected: int = 0
    gaps_detected: int = 0
    corporate_actions: int = 0
    validation_errors: int = 0
    last_update: Optional[datetime] = None

    @property
    def validity_rate(self) -> float:
        """Calculate data validity rate."""
        if self.total_ticks == 0:
            return 0.0
        return self.valid_ticks / self.total_ticks * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_ticks == 0:
            return 0.0
        return self.invalid_ticks / self.total_ticks * 100


class OutlierDetector:
    """Detect outliers in market data."""

    def __init__(self, window_size: int = 100, z_threshold: float = 3.0):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.price_history: Dict[str, deque] = {}
        self.volume_history: Dict[str, deque] = {}
        self.logger = get_logger("data.outlier_detector")

    def is_price_outlier(self, symbol: str, price: float) -> bool:
        """Check if price is an outlier using z-score method."""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)

        history = self.price_history[symbol]

        if len(history) < 20:  # Need minimum history
            history.append(price)
            return False

        # Calculate z-score
        mean = statistics.mean(history)
        stdev = statistics.stdev(history)

        if stdev == 0:
            history.append(price)
            return False

        z_score = abs((price - mean) / stdev)

        # Add to history
        history.append(price)

        if z_score > self.z_threshold:
            self.logger.warning(
                f"Price outlier detected for {symbol}: price={price:.2f}, "
                f"mean={mean:.2f}, z-score={z_score:.2f}"
            )
            return True

        return False

    def is_volume_outlier(self, symbol: str, volume: int) -> bool:
        """Check if volume is an outlier."""
        if symbol not in self.volume_history:
            self.volume_history[symbol] = deque(maxlen=self.window_size)

        history = self.volume_history[symbol]

        if len(history) < 20:
            history.append(volume)
            return False

        # Use IQR method for volume
        q1 = np.percentile(list(history), 25)
        q3 = np.percentile(list(history), 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 3 * iqr  # More lenient for volume spikes

        history.append(volume)

        if volume < lower_bound or volume > upper_bound:
            self.logger.warning(
                f"Volume outlier detected for {symbol}: volume={volume:,}, "
                f"expected range=[{lower_bound:,.0f}, {upper_bound:,.0f}]"
            )
            return True

        return False


class MarketHoursValidator:
    """Validate data against market hours."""

    def __init__(self):
        self.logger = get_logger("data.market_hours")

        # Regular market hours (EST)
        self.regular_open = time(9, 30)
        self.regular_close = time(16, 0)

        # Extended hours
        self.premarket_open = time(4, 0)
        self.afterhours_close = time(20, 0)

        # Market holidays (simplified - should load from calendar)
        self.holidays = set(
            [
                datetime(2025, 1, 1).date(),  # New Year's Day
                datetime(2025, 1, 20).date(),  # MLK Day
                datetime(2025, 2, 17).date(),  # Presidents Day
                datetime(2025, 4, 18).date(),  # Good Friday
                datetime(2025, 5, 26).date(),  # Memorial Day
                datetime(2025, 7, 4).date(),  # Independence Day
                datetime(2025, 9, 1).date(),  # Labor Day
                datetime(2025, 11, 27).date(),  # Thanksgiving
                datetime(2025, 12, 25).date(),  # Christmas
            ]
        )

    def is_market_open(self, timestamp: datetime, extended: bool = False) -> bool:
        """Check if market is open at given timestamp."""
        # Check if holiday
        if timestamp.date() in self.holidays:
            return False

        # Check if weekend
        if timestamp.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check time
        current_time = timestamp.time()

        if extended:
            return self.premarket_open <= current_time <= self.afterhours_close
        else:
            return self.regular_open <= current_time <= self.regular_close

    def get_session_type(self, timestamp: datetime) -> str:
        """Get market session type."""
        if not self.is_market_open(timestamp, extended=True):
            return "closed"

        current_time = timestamp.time()

        if current_time < self.regular_open:
            return "premarket"
        elif current_time <= self.regular_close:
            return "regular"
        else:
            return "afterhours"


class DataValidator:
    """Main data validation class."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("data.validator")

        # Components
        self.outlier_detector = OutlierDetector()
        self.market_hours = MarketHoursValidator()

        # Metrics
        self.metrics = DataQualityMetrics()

        # Gap detection
        self.last_tick_time: Dict[str, datetime] = {}
        self.max_gap_seconds = 60  # Max acceptable gap

        # Corporate action detection
        self.price_change_threshold = 0.2  # 20% change indicates possible corporate action

    def validate_tick(self, tick: TickData) -> List[ValidationResult]:
        """Validate a single tick."""
        results = []
        self.metrics.total_ticks += 1

        # Basic validation
        basic_result = self._validate_tick_basic(tick)
        results.append(basic_result)

        if not basic_result.is_valid:
            self.metrics.invalid_ticks += 1
            return results

        # Price validation
        price_result = self._validate_tick_prices(tick)
        results.append(price_result)

        # Spread validation
        spread_result = self._validate_spread(tick)
        results.append(spread_result)

        # Outlier detection
        outlier_result = self._check_outliers(tick)
        results.append(outlier_result)

        # Gap detection
        gap_result = self._check_gap(tick)
        results.append(gap_result)

        # Market hours validation
        hours_result = self._validate_market_hours(tick)
        results.append(hours_result)

        # Update metrics
        if all(r.is_valid for r in results):
            self.metrics.valid_ticks += 1
        else:
            self.metrics.invalid_ticks += 1
            self.metrics.validation_errors += 1

        self.metrics.last_update = datetime.now()

        return results

    def _validate_tick_basic(self, tick: TickData) -> ValidationResult:
        """Basic tick validation."""
        # Check for null values
        if tick.bid is None or tick.ask is None or tick.last is None:
            return ValidationResult(
                is_valid=False,
                check_name="basic_validation",
                message="Tick contains null prices",
                severity="error",
            )

        # Check for negative prices
        if tick.bid < 0 or tick.ask < 0 or tick.last < 0:
            return ValidationResult(
                is_valid=False,
                check_name="basic_validation",
                message="Tick contains negative prices",
                severity="error",
            )

        # Check for zero prices
        if tick.bid == 0 and tick.ask == 0 and tick.last == 0:
            return ValidationResult(
                is_valid=False,
                check_name="basic_validation",
                message="All prices are zero",
                severity="error",
            )

        # Check sizes
        if tick.bid_size < 0 or tick.ask_size < 0 or tick.last_size < 0:
            return ValidationResult(
                is_valid=False,
                check_name="basic_validation",
                message="Negative size values",
                severity="error",
            )

        return ValidationResult(
            is_valid=True,
            check_name="basic_validation",
            message="Basic validation passed",
            severity="info",
        )

    def _validate_tick_prices(self, tick: TickData) -> ValidationResult:
        """Validate tick price relationships with epsilon tolerance."""
        EPSILON = 1e-6  # Tolerance for floating-point comparison

        # Bid should be less than or equal to ask (with tolerance)
        if tick.bid > tick.ask + EPSILON:
            return ValidationResult(
                is_valid=False,
                check_name="price_validation",
                message=f"Inverted market: bid ({tick.bid:.6f}) > ask ({tick.ask:.6f})",
                severity="error",
            )

        # Last price should generally be between bid and ask
        # (but can be outside during fast markets)
        if tick.last < tick.bid * 0.9 or tick.last > tick.ask * 1.1:
            return ValidationResult(
                is_valid=False,
                check_name="price_validation",
                message=f"Last price ({tick.last}) far outside bid-ask spread",
                severity="warning",
            )

        return ValidationResult(
            is_valid=True,
            check_name="price_validation",
            message="Price relationships valid",
            severity="info",
        )

    def _validate_spread(self, tick: TickData) -> ValidationResult:
        """Validate bid-ask spread."""
        spread_bps = tick.spread_bps

        # Check for excessive spread
        if spread_bps > 100:  # More than 1%
            return ValidationResult(
                is_valid=False,
                check_name="spread_validation",
                message=f"Excessive spread: {spread_bps:.1f} bps",
                severity="warning",
                data={"spread_bps": spread_bps},
            )

        # Check for zero spread (locked market) with epsilon tolerance
        EPSILON = 1e-6
        if abs(spread_bps) < EPSILON:
            return ValidationResult(
                is_valid=True,
                check_name="spread_validation",
                message="Locked market (zero spread)",
                severity="info",
            )

        # Check for negative spread (should be caught earlier)
        if spread_bps < 0:
            return ValidationResult(
                is_valid=False,
                check_name="spread_validation",
                message=f"Negative spread: {spread_bps:.1f} bps",
                severity="error",
            )

        return ValidationResult(
            is_valid=True,
            check_name="spread_validation",
            message=f"Normal spread: {spread_bps:.1f} bps",
            severity="info",
        )

    def _check_outliers(self, tick: TickData) -> ValidationResult:
        """Check for outliers."""
        is_price_outlier = self.outlier_detector.is_price_outlier(tick.symbol, tick.last)
        is_volume_outlier = self.outlier_detector.is_volume_outlier(tick.symbol, tick.volume)

        if is_price_outlier or is_volume_outlier:
            self.metrics.outliers_detected += 1

            outlier_type = []
            if is_price_outlier:
                outlier_type.append("price")
            if is_volume_outlier:
                outlier_type.append("volume")

            return ValidationResult(
                is_valid=False,
                check_name="outlier_detection",
                message=f"Outlier detected: {', '.join(outlier_type)}",
                severity="warning",
                data={
                    "price_outlier": is_price_outlier,
                    "volume_outlier": is_volume_outlier,
                },
            )

        return ValidationResult(
            is_valid=True,
            check_name="outlier_detection",
            message="No outliers detected",
            severity="info",
        )

    def _check_gap(self, tick: TickData) -> ValidationResult:
        """Check for data gaps."""
        if tick.symbol in self.last_tick_time:
            gap_seconds = (tick.timestamp - self.last_tick_time[tick.symbol]).total_seconds()

            if gap_seconds > self.max_gap_seconds:
                self.metrics.gaps_detected += 1

                return ValidationResult(
                    is_valid=False,
                    check_name="gap_detection",
                    message=f"Data gap detected: {gap_seconds:.1f} seconds",
                    severity="warning",
                    data={"gap_seconds": gap_seconds},
                )

        self.last_tick_time[tick.symbol] = tick.timestamp

        return ValidationResult(
            is_valid=True,
            check_name="gap_detection",
            message="No data gaps",
            severity="info",
        )

    def _validate_market_hours(self, tick: TickData) -> ValidationResult:
        """Validate tick against market hours."""
        session_type = self.market_hours.get_session_type(tick.timestamp)

        if session_type == "closed":
            return ValidationResult(
                is_valid=False,
                check_name="market_hours",
                message="Data received during market closed hours",
                severity="warning",
                data={"session": session_type},
            )

        return ValidationResult(
            is_valid=True,
            check_name="market_hours",
            message=f"Market session: {session_type}",
            severity="info",
            data={"session": session_type},
        )

    def check_corporate_action(
        self, symbol: str, prev_close: float, current_price: float
    ) -> Optional[ValidationResult]:
        """Check for potential corporate actions."""
        if prev_close <= 0:
            return None

        price_change = abs((current_price - prev_close) / prev_close)

        if price_change > self.price_change_threshold:
            self.metrics.corporate_actions += 1

            return ValidationResult(
                is_valid=False,
                check_name="corporate_action",
                message=f"Potential corporate action: {price_change*100:.1f}% price change",
                severity="critical",
                data={
                    "prev_close": prev_close,
                    "current_price": current_price,
                    "change_pct": price_change * 100,
                },
            )

        return None

    def validate_bar(self, bar: BarData) -> List[ValidationResult]:
        """Validate OHLCV bar data."""
        results = []

        # OHLC relationship
        if not (bar.low <= bar.open <= bar.high and bar.low <= bar.close <= bar.high):
            results.append(
                ValidationResult(
                    is_valid=False,
                    check_name="ohlc_validation",
                    message="Invalid OHLC relationship",
                    severity="error",
                )
            )
        else:
            results.append(
                ValidationResult(
                    is_valid=True,
                    check_name="ohlc_validation",
                    message="Valid OHLC relationship",
                    severity="info",
                )
            )

        # Volume validation
        if bar.volume < 0:
            results.append(
                ValidationResult(
                    is_valid=False,
                    check_name="volume_validation",
                    message="Negative volume",
                    severity="error",
                )
            )
        elif bar.volume == 0:
            results.append(
                ValidationResult(
                    is_valid=True,
                    check_name="volume_validation",
                    message="Zero volume bar",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    is_valid=True,
                    check_name="volume_validation",
                    message="Valid volume",
                    severity="info",
                )
            )

        return results

    def get_metrics(self) -> DataQualityMetrics:
        """Get data quality metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = DataQualityMetrics()


class DataHealthMonitor:
    """Monitor overall data feed health."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("data.health")

        # Health status
        self.last_data_time: Dict[str, datetime] = {}
        self.data_rate: Dict[str, float] = {}  # Ticks per second
        self.health_status: Dict[str, str] = {}  # 'healthy', 'degraded', 'unhealthy'

        # Thresholds
        self.unhealthy_gap = 30  # seconds
        self.degraded_gap = 10  # seconds
        self.min_tick_rate = 0.1  # ticks per second

    def update(self, symbol: str) -> str:
        """Update health status for symbol."""
        now = datetime.now()

        if symbol not in self.last_data_time:
            self.last_data_time[symbol] = now
            self.health_status[symbol] = "healthy"
            return "healthy"

        gap = (now - self.last_data_time[symbol]).total_seconds()
        self.last_data_time[symbol] = now

        # Determine health status
        if gap > self.unhealthy_gap:
            status = "unhealthy"
        elif gap > self.degraded_gap:
            status = "degraded"
        else:
            status = "healthy"

        # Log status changes
        if symbol in self.health_status and self.health_status[symbol] != status:
            self.logger.warning(
                f"Data health changed for {symbol}: {self.health_status[symbol]} -> {status}"
            )

        self.health_status[symbol] = status
        return status

    def get_overall_health(self) -> str:
        """Get overall data feed health."""
        if not self.health_status:
            return "unknown"

        statuses = list(self.health_status.values())

        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"

    def get_health_report(self) -> Dict[str, Any]:
        """Get detailed health report."""
        return {
            "overall": self.get_overall_health(),
            "symbols": self.health_status.copy(),
            "last_update": {
                symbol: time.isoformat() if time else None
                for symbol, time in self.last_data_time.items()
            },
            "data_rates": self.data_rate.copy(),
        }

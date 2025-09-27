"""
Market Data Validation Layer

Validates market data quality before using in trading strategies.
Checks for stale data, invalid prices, wide spreads, and anomalies.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from robo_trader.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    reason: str
    warnings: List[str] = None
    metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.metrics is None:
            self.metrics = {}


class DataValidator:
    """Comprehensive market data validation."""

    def __init__(
        self,
        max_staleness_seconds: int = None,
        max_spread_percent: float = None,
        min_price: float = 0.01,
        max_price_change_percent: float = 20.0,
        require_volume: bool = True,
        min_volume: int = 100,
    ):
        # Load from environment or use defaults
        self.max_staleness_seconds = max_staleness_seconds or int(
            os.getenv("DATA_STALENESS_SECONDS", "60")
        )
        self.max_spread_percent = max_spread_percent or float(
            os.getenv("MAX_SPREAD_PERCENT", "1.0")
        )
        self.min_price = min_price
        self.max_price_change_percent = max_price_change_percent
        self.require_volume = require_volume
        self.min_volume = min_volume

        # Track last valid prices for anomaly detection
        self.last_valid_prices: Dict[str, float] = {}
        self.validation_stats = {
            "total_validations": 0,
            "passed": 0,
            "failed_stale": 0,
            "failed_spread": 0,
            "failed_price": 0,
            "failed_volume": 0,
            "failed_anomaly": 0,
        }

    def validate_price_data(
        self, data: Dict[str, Any], symbol: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate price data dictionary.

        Expected fields: timestamp, bid, ask, last, volume
        """
        self.validation_stats["total_validations"] += 1

        # Check for empty data
        if not data:
            self.validation_stats["failed_price"] += 1
            return ValidationResult(False, "No data received")

        # Check timestamp freshness
        if "timestamp" in data:
            result = self._check_staleness(data["timestamp"])
            if not result.is_valid:
                self.validation_stats["failed_stale"] += 1
                return result

        # Validate prices
        price_result = self._validate_prices(data, symbol)
        if not price_result.is_valid:
            self.validation_stats["failed_price"] += 1
            return price_result

        # Check spread
        if "bid" in data and "ask" in data:
            spread_result = self._check_spread(data["bid"], data["ask"])
            if not spread_result.is_valid:
                self.validation_stats["failed_spread"] += 1
                return spread_result

        # Check volume
        if self.require_volume and "volume" in data:
            volume_result = self._check_volume(data["volume"])
            if not volume_result.is_valid:
                self.validation_stats["failed_volume"] += 1
                return volume_result

        # Check for anomalies
        if symbol and "last" in data:
            anomaly_result = self._check_anomalies(symbol, data["last"])
            if not anomaly_result.is_valid:
                self.validation_stats["failed_anomaly"] += 1
                return anomaly_result

        self.validation_stats["passed"] += 1
        return ValidationResult(
            True,
            "Data validation passed",
            metrics={
                "staleness": self._calculate_staleness(data.get("timestamp")),
                "spread_pct": self._calculate_spread_percent(data.get("bid"), data.get("ask")),
                "volume": data.get("volume", 0),
            },
        )

    def validate_dataframe(
        self, df: pd.DataFrame, symbol: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate a pandas DataFrame of market data.

        Expected columns: open, high, low, close, volume
        """
        self.validation_stats["total_validations"] += 1

        # Check for empty dataframe
        if df is None or df.empty:
            self.validation_stats["failed_price"] += 1
            return ValidationResult(False, "Empty dataframe")

        warnings = []

        # Check for required columns
        required_columns = ["close"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            self.validation_stats["failed_price"] += 1
            return ValidationResult(False, f"Missing required columns: {missing_columns}")

        # Check for NaN values
        if df["close"].isna().any():
            nan_count = df["close"].isna().sum()
            warnings.append(f"Found {nan_count} NaN values in close prices")
            # Remove NaN rows
            df = df.dropna(subset=["close"])
            if df.empty:
                self.validation_stats["failed_price"] += 1
                return ValidationResult(False, "All close prices are NaN")

        # Check for zero or negative prices
        if (df["close"] <= 0).any():
            invalid_count = (df["close"] <= 0).sum()
            self.validation_stats["failed_price"] += 1
            return ValidationResult(False, f"Found {invalid_count} invalid prices (<=0)")

        # Check for extreme price changes
        if len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            max_change = price_changes.max() * 100
            if max_change > self.max_price_change_percent:
                warnings.append(f"Large price change detected: {max_change:.1f}%")

        # Check OHLC consistency if available
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            ohlc_result = self._validate_ohlc(df)
            if not ohlc_result.is_valid:
                self.validation_stats["failed_price"] += 1
                return ohlc_result
            if ohlc_result.warnings:
                warnings.extend(ohlc_result.warnings)

        # Check volume if required
        if self.require_volume and "volume" in df.columns:
            low_volume_count = (df["volume"] < self.min_volume).sum()
            if low_volume_count > len(df) * 0.5:  # More than 50% low volume
                warnings.append(f"Low volume in {low_volume_count}/{len(df)} bars")

        # Check timestamp if available
        if df.index.name == "timestamp" or "timestamp" in df.columns:
            timestamp_col = df.index if df.index.name == "timestamp" else df["timestamp"]
            latest_timestamp = timestamp_col.max()
            if pd.Timestamp.now() - latest_timestamp > pd.Timedelta(
                seconds=self.max_staleness_seconds
            ):
                self.validation_stats["failed_stale"] += 1
                return ValidationResult(
                    False, f"Data is stale (>{ self.max_staleness_seconds}s old)"
                )

        self.validation_stats["passed"] += 1
        return ValidationResult(
            True,
            "Dataframe validation passed",
            warnings=warnings,
            metrics={
                "rows": len(df),
                "min_price": float(df["close"].min()),
                "max_price": float(df["close"].max()),
                "avg_volume": float(df["volume"].mean()) if "volume" in df.columns else 0,
            },
        )

    def _check_staleness(self, timestamp: Any) -> ValidationResult:
        """Check if data is too old."""
        try:
            if isinstance(timestamp, (int, float)):
                # Unix timestamp
                age_seconds = time.time() - timestamp
            elif isinstance(timestamp, str):
                # Parse string timestamp
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_seconds = (datetime.now() - dt).total_seconds()
            elif isinstance(timestamp, datetime):
                age_seconds = (datetime.now() - timestamp).total_seconds()
            else:
                return ValidationResult(True, "Cannot determine timestamp age")

            if age_seconds > self.max_staleness_seconds:
                return ValidationResult(
                    False,
                    f"Stale data: {age_seconds:.1f}s old (max: {self.max_staleness_seconds}s)",
                )

            return ValidationResult(True, "Data is fresh")

        except Exception as e:
            logger.warning(f"Error checking staleness: {e}")
            return ValidationResult(True, "Cannot validate timestamp")

    def _validate_prices(
        self, data: Dict[str, Any], symbol: Optional[str] = None
    ) -> ValidationResult:
        """Validate price fields."""
        price_fields = ["last", "bid", "ask", "open", "high", "low", "close"]

        for field in price_fields:
            if field in data:
                price = data[field]

                # Check for None
                if price is None:
                    continue

                # Check for zero or negative
                if price <= 0:
                    return ValidationResult(False, f"Invalid {field} price: {price}")

                # Check minimum price
                if price < self.min_price:
                    return ValidationResult(
                        False, f"{field} price below minimum: {price} < {self.min_price}"
                    )

        # Validate bid <= last <= ask if all present
        if all(k in data for k in ["bid", "last", "ask"]):
            if data["bid"] and data["ask"] and data["last"]:
                if not (data["bid"] <= data["last"] <= data["ask"]):
                    return ValidationResult(
                        False,
                        f"Price inconsistency: bid={data['bid']}, last={data['last']}, ask={data['ask']}",
                    )

        return ValidationResult(True, "Prices valid")

    def _check_spread(self, bid: float, ask: float) -> ValidationResult:
        """Check bid-ask spread."""
        if not bid or not ask or bid <= 0 or ask <= 0:
            return ValidationResult(True, "Cannot calculate spread")

        spread = ask - bid
        spread_percent = (spread / bid) * 100

        if spread_percent > self.max_spread_percent:
            return ValidationResult(
                False, f"Wide spread: {spread_percent:.2f}% (max: {self.max_spread_percent}%)"
            )

        return ValidationResult(True, f"Spread acceptable: {spread_percent:.2f}%")

    def _check_volume(self, volume: Any) -> ValidationResult:
        """Check volume requirements."""
        if volume is None:
            return ValidationResult(True, "No volume data")

        try:
            vol = int(volume)
            if vol < self.min_volume:
                return ValidationResult(False, f"Low volume: {vol} (min: {self.min_volume})")
        except (ValueError, TypeError):
            return ValidationResult(False, f"Invalid volume: {volume}")

        return ValidationResult(True, "Volume acceptable")

    def _check_anomalies(self, symbol: str, price: float) -> ValidationResult:
        """Check for price anomalies."""
        if symbol not in self.last_valid_prices:
            # First price for this symbol
            self.last_valid_prices[symbol] = price
            return ValidationResult(True, "First price recorded")

        last_price = self.last_valid_prices[symbol]
        if last_price <= 0:
            self.last_valid_prices[symbol] = price
            return ValidationResult(True, "Reset from invalid last price")

        # Calculate price change
        change_percent = abs((price - last_price) / last_price) * 100

        if change_percent > self.max_price_change_percent:
            return ValidationResult(
                False,
                f"Anomaly detected: {change_percent:.1f}% price change (max: {self.max_price_change_percent}%)",
            )

        # Update last valid price
        self.last_valid_prices[symbol] = price
        return ValidationResult(True, "No anomalies detected")

    def _validate_ohlc(self, df: pd.DataFrame) -> ValidationResult:
        """Validate OHLC data consistency."""
        warnings = []

        # Check high >= low
        invalid_hl = df["high"] < df["low"]
        if invalid_hl.any():
            count = invalid_hl.sum()
            return ValidationResult(False, f"Found {count} bars where high < low")

        # Check close within high-low range
        close_above_high = df["close"] > df["high"]
        close_below_low = df["close"] < df["low"]

        if close_above_high.any():
            count = close_above_high.sum()
            warnings.append(f"Found {count} bars where close > high")

        if close_below_low.any():
            count = close_below_low.sum()
            warnings.append(f"Found {count} bars where close < low")

        # Check open within high-low range
        if "open" in df.columns:
            open_above_high = df["open"] > df["high"]
            open_below_low = df["open"] < df["low"]

            if open_above_high.any():
                count = open_above_high.sum()
                warnings.append(f"Found {count} bars where open > high")

            if open_below_low.any():
                count = open_below_low.sum()
                warnings.append(f"Found {count} bars where open < low")

        return ValidationResult(True, "OHLC validation passed", warnings=warnings)

    def _calculate_staleness(self, timestamp: Any) -> float:
        """Calculate data age in seconds."""
        try:
            if isinstance(timestamp, (int, float)):
                return time.time() - timestamp
            elif isinstance(timestamp, datetime):
                return (datetime.now() - timestamp).total_seconds()
        except Exception:
            pass
        return 0.0

    def _calculate_spread_percent(self, bid: Any, ask: Any) -> float:
        """Calculate bid-ask spread percentage."""
        try:
            if bid and ask and bid > 0:
                return ((ask - bid) / bid) * 100
        except Exception:
            pass
        return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()

        if stats["total_validations"] > 0:
            stats["pass_rate"] = (stats["passed"] / stats["total_validations"]) * 100
            stats["failure_reasons"] = {
                "stale": stats["failed_stale"],
                "spread": stats["failed_spread"],
                "price": stats["failed_price"],
                "volume": stats["failed_volume"],
                "anomaly": stats["failed_anomaly"],
            }

        return stats

    def reset_statistics(self) -> None:
        """Reset validation statistics."""
        for key in self.validation_stats:
            if key != "total_validations":
                self.validation_stats[key] = 0
        self.validation_stats["total_validations"] = 0

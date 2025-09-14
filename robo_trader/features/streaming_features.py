"""
Real-time streaming feature calculator for Phase 3 S5.
Provides incremental feature updates without full recalculation.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StreamingWindow:
    """Manages a rolling window of data for streaming calculations."""

    size: int
    data: Deque[float] = field(default_factory=deque)
    timestamps: Deque[datetime] = field(default_factory=deque)

    def add(self, value: float, timestamp: datetime):
        """Add a new value to the window."""
        self.data.append(value)
        self.timestamps.append(timestamp)

        # Maintain window size
        while len(self.data) > self.size:
            self.data.popleft()
            self.timestamps.popleft()

    def get_array(self) -> np.ndarray:
        """Get data as numpy array."""
        return np.array(self.data)

    def get_series(self) -> pd.Series:
        """Get data as pandas Series with timestamps."""
        if not self.data:
            return pd.Series()
        return pd.Series(list(self.data), index=list(self.timestamps))

    @property
    def is_full(self) -> bool:
        """Check if window has enough data."""
        return len(self.data) >= self.size


class StreamingFeatureCalculator:
    """
    Real-time feature calculator with incremental updates.
    Maintains rolling windows and efficiently updates features.
    """

    def __init__(
        self,
        window_sizes: Dict[str, int] = None,
        feature_list: List[str] = None,
    ):
        """
        Initialize streaming calculator.

        Args:
            window_sizes: Window sizes for different features
            feature_list: List of features to calculate
        """
        self.window_sizes = window_sizes or {
            "sma_short": 10,
            "sma_medium": 20,
            "sma_long": 50,
            "rsi": 14,
            "volume": 20,
            "volatility": 20,
            "momentum": 10,
        }

        self.feature_list = feature_list or [
            "price",
            "volume",
            "returns",
            "sma_10",
            "sma_20",
            "sma_50",
            "ema_10",
            "ema_20",
            "rsi",
            "macd",
            "macd_signal",
            "bb_upper",
            "bb_lower",
            "bb_position",
            "volume_ratio",
            "volatility",
            "momentum",
            "price_momentum",
        ]

        # Initialize windows for each symbol
        self.windows: Dict[str, Dict[str, StreamingWindow]] = {}

        # Cache for exponential calculations
        self.ema_cache: Dict[str, Dict[str, float]] = {}

        # Latest features
        self.latest_features: Dict[str, Dict[str, float]] = {}

        # Feature history for drift detection
        self.feature_history: Dict[str, deque] = {}
        self.max_history = 1000

    def initialize_symbol(self, symbol: str, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize streaming for a symbol with optional historical data.

        Args:
            symbol: Stock symbol
            historical_data: Optional historical OHLCV data for initialization
        """
        # Initialize windows
        self.windows[symbol] = {
            "price": StreamingWindow(max(self.window_sizes.values())),
            "volume": StreamingWindow(self.window_sizes["volume"]),
            "high": StreamingWindow(20),
            "low": StreamingWindow(20),
        }

        # Initialize caches
        self.ema_cache[symbol] = {}
        self.latest_features[symbol] = {}
        self.feature_history[symbol] = deque(maxlen=self.max_history)

        # Warm up with historical data if provided
        if historical_data is not None and not historical_data.empty:
            self._warmup_with_history(symbol, historical_data)
            logger.info(f"Initialized {symbol} with {len(historical_data)} historical points")

    def _warmup_with_history(self, symbol: str, data: pd.DataFrame):
        """Warm up windows with historical data."""
        for idx, row in data.iterrows():
            timestamp = idx if isinstance(idx, datetime) else datetime.now()
            # Handle both lowercase and capitalized column names
            close_col = "close" if "close" in row else "Close"
            volume_col = "volume" if "volume" in row else "Volume"
            high_col = "high" if "high" in row else "High"
            low_col = "low" if "low" in row else "Low"

            self.windows[symbol]["price"].add(row[close_col], timestamp)
            self.windows[symbol]["volume"].add(row[volume_col], timestamp)
            self.windows[symbol]["high"].add(row[high_col], timestamp)
            self.windows[symbol]["low"].add(row[low_col], timestamp)

            # Initialize EMA values
            if len(self.windows[symbol]["price"].data) == 10:
                self.ema_cache[symbol]["ema_10"] = np.mean(list(self.windows[symbol]["price"].data))
            elif len(self.windows[symbol]["price"].data) == 20:
                self.ema_cache[symbol]["ema_20"] = np.mean(list(self.windows[symbol]["price"].data))

    def update(
        self,
        symbol: str,
        price: float,
        volume: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Update features with new market data.

        Args:
            symbol: Stock symbol
            price: Current price
            volume: Current volume
            high: High price
            low: Low price
            timestamp: Data timestamp

        Returns:
            Dictionary of updated features
        """
        if symbol not in self.windows:
            self.initialize_symbol(symbol)

        timestamp = timestamp or datetime.now()

        # Update windows
        self.windows[symbol]["price"].add(price, timestamp)
        self.windows[symbol]["volume"].add(volume, timestamp)
        if high is not None:
            self.windows[symbol]["high"].add(high, timestamp)
        if low is not None:
            self.windows[symbol]["low"].add(low, timestamp)

        # Calculate features
        features = self._calculate_features(symbol, price, volume)

        # Store latest features
        self.latest_features[symbol] = features

        # Add to history for drift detection
        self.feature_history[symbol].append({"timestamp": timestamp, "features": features.copy()})

        return features

    def _calculate_features(
        self, symbol: str, current_price: float, current_volume: float
    ) -> Dict[str, float]:
        """Calculate all streaming features."""
        features = {}

        # Basic features
        features["price"] = current_price
        features["volume"] = current_volume

        price_window = self.windows[symbol]["price"]
        volume_window = self.windows[symbol]["volume"]

        if len(price_window.data) > 1:
            # Returns
            features["returns"] = (current_price - price_window.data[-2]) / price_window.data[-2]
            features["log_returns"] = np.log(current_price / price_window.data[-2])
        else:
            features["returns"] = 0
            features["log_returns"] = 0

        # Moving averages
        if len(price_window.data) >= 10:
            features["sma_10"] = np.mean(list(price_window.data)[-10:])
            features["price_to_sma_10"] = current_price / features["sma_10"]

        if len(price_window.data) >= 20:
            features["sma_20"] = np.mean(list(price_window.data)[-20:])
            features["price_to_sma_20"] = current_price / features["sma_20"]

        if len(price_window.data) >= 50:
            features["sma_50"] = np.mean(list(price_window.data)[-50:])
            features["price_to_sma_50"] = current_price / features["sma_50"]

        # Exponential moving averages (incremental)
        features.update(self._calculate_ema(symbol, current_price))

        # RSI
        if len(price_window.data) >= 14:
            features["rsi"] = self._calculate_rsi(list(price_window.data)[-14:])

        # Bollinger Bands
        if len(price_window.data) >= 20:
            bb_features = self._calculate_bollinger_bands(
                list(price_window.data)[-20:], current_price
            )
            features.update(bb_features)

        # Volume features
        if len(volume_window.data) >= 20:
            avg_volume = np.mean(list(volume_window.data))
            features["volume_ratio"] = current_volume / avg_volume if avg_volume > 0 else 0
            features["volume_sma_20"] = avg_volume

        # Volatility
        if len(price_window.data) >= 20:
            returns = np.diff(list(price_window.data)[-20:]) / list(price_window.data)[-20:-1]
            features["volatility"] = np.std(returns) * np.sqrt(252)
            features["volatility_ratio"] = (
                features["volatility"] / 0.15
            )  # Normalized to 15% annual vol

        # Momentum
        if len(price_window.data) >= 10:
            features["momentum"] = (current_price - price_window.data[-10]) / price_window.data[-10]
            features["price_momentum"] = current_price / price_window.data[-10]

        # High/Low features
        if len(self.windows[symbol]["high"].data) >= 20:
            high_20 = max(list(self.windows[symbol]["high"].data))
            low_20 = min(list(self.windows[symbol]["low"].data))

            features["high_20"] = high_20
            features["low_20"] = low_20
            features["price_position"] = (
                (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            )

        # MACD
        if "ema_12" in self.ema_cache[symbol] and "ema_26" in self.ema_cache[symbol]:
            features["macd"] = self.ema_cache[symbol]["ema_12"] - self.ema_cache[symbol]["ema_26"]

            # MACD signal (EMA of MACD)
            if "macd_signal" not in self.ema_cache[symbol]:
                self.ema_cache[symbol]["macd_signal"] = features["macd"]
            else:
                alpha = 2 / (9 + 1)
                self.ema_cache[symbol]["macd_signal"] = (
                    alpha * features["macd"] + (1 - alpha) * self.ema_cache[symbol]["macd_signal"]
                )
            features["macd_signal"] = self.ema_cache[symbol]["macd_signal"]
            features["macd_histogram"] = features["macd"] - features["macd_signal"]

        return features

    def _calculate_ema(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Calculate exponential moving averages incrementally."""
        ema_features = {}

        for period in [10, 12, 20, 26]:
            key = f"ema_{period}"

            if key not in self.ema_cache[symbol]:
                # Initialize with SMA if we have enough data
                if len(self.windows[symbol]["price"].data) >= period:
                    self.ema_cache[symbol][key] = np.mean(
                        list(self.windows[symbol]["price"].data)[-period:]
                    )
            else:
                # Update EMA incrementally
                alpha = 2 / (period + 1)
                self.ema_cache[symbol][key] = (
                    alpha * current_price + (1 - alpha) * self.ema_cache[symbol][key]
                )

            if key in self.ema_cache[symbol]:
                ema_features[key] = self.ema_cache[symbol][key]
                if period in [10, 20]:  # Only for displayed EMAs
                    ema_features[f"price_to_{key}"] = current_price / self.ema_cache[symbol][key]

        return ema_features

    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI from price list."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(
        self, prices: List[float], current_price: float
    ) -> Dict[str, float]:
        """Calculate Bollinger Bands features."""
        mean = np.mean(prices)
        std = np.std(prices)

        upper = mean + 2 * std
        lower = mean - 2 * std

        position = (current_price - lower) / (upper - lower) if upper > lower else 0.5

        return {
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_middle": mean,
            "bb_width": upper - lower,
            "bb_position": position,
        }

    def get_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest features for a symbol."""
        return self.latest_features.get(symbol)

    def get_feature_vector(
        self, symbol: str, feature_names: Optional[List[str]] = None
    ) -> Optional[np.ndarray]:
        """
        Get feature vector for ML model input.

        Args:
            symbol: Stock symbol
            feature_names: Specific features to include

        Returns:
            Numpy array of features
        """
        features = self.get_features(symbol)
        if not features:
            return None

        feature_names = feature_names or self.feature_list

        # Extract features in order
        vector = []
        for name in feature_names:
            if name in features:
                vector.append(features[name])
            else:
                vector.append(0)  # Default value for missing features

        return np.array(vector)

    def detect_drift(self, symbol: str, threshold: float = 2.0) -> Dict[str, Any]:
        """
        Detect feature drift using statistical tests.

        Args:
            symbol: Stock symbol
            threshold: Z-score threshold for drift detection

        Returns:
            Dictionary with drift detection results
        """
        if symbol not in self.feature_history or len(self.feature_history[symbol]) < 100:
            return {"drift_detected": False, "message": "Insufficient history"}

        history = list(self.feature_history[symbol])

        # Split into reference and recent windows
        split_point = len(history) // 2
        reference = history[:split_point]
        recent = history[split_point:]

        drift_results = {}
        drift_detected = False

        # Check each feature for drift
        for feature_name in self.feature_list:
            if feature_name in ["price", "volume"]:  # Skip absolute values
                continue

            ref_values = [h["features"].get(feature_name, 0) for h in reference]
            recent_values = [h["features"].get(feature_name, 0) for h in recent]

            if not ref_values or not recent_values:
                continue

            # Calculate statistics
            ref_mean = np.mean(ref_values)
            ref_std = np.std(ref_values)
            recent_mean = np.mean(recent_values)

            if ref_std > 0:
                z_score = abs(recent_mean - ref_mean) / ref_std

                if z_score > threshold:
                    drift_detected = True
                    drift_results[feature_name] = {
                        "z_score": z_score,
                        "reference_mean": ref_mean,
                        "recent_mean": recent_mean,
                        "drift": True,
                    }

        return {
            "drift_detected": drift_detected,
            "features_with_drift": drift_results,
            "timestamp": datetime.now(),
        }


class StreamingFeatureStore:
    """
    Persistent storage for streaming features with time-series versioning.
    """

    def __init__(self, storage_path: str = "feature_store"):
        """
        Initialize feature store.

        Args:
            storage_path: Path for feature storage
        """
        self.storage_path = storage_path
        self.features_buffer: Dict[str, List[Dict]] = {}
        self.buffer_size = 100
        self.version = 0

        # Create storage directory
        import os

        os.makedirs(storage_path, exist_ok=True)

    def store_features(self, symbol: str, features: Dict[str, float], timestamp: datetime):
        """
        Store features with versioning.

        Args:
            symbol: Stock symbol
            features: Feature dictionary
            timestamp: Feature timestamp
        """
        if symbol not in self.features_buffer:
            self.features_buffer[symbol] = []

        # Add versioned features
        versioned_features = {
            "timestamp": timestamp.isoformat(),
            "version": self.version,
            "features": features,
        }

        self.features_buffer[symbol].append(versioned_features)

        # Persist when buffer is full
        if len(self.features_buffer[symbol]) >= self.buffer_size:
            self._persist_features(symbol)

    def _persist_features(self, symbol: str):
        """Persist buffered features to disk."""
        if symbol not in self.features_buffer or not self.features_buffer[symbol]:
            return

        import os
        import pickle

        # Create symbol directory
        symbol_path = os.path.join(self.storage_path, symbol)
        os.makedirs(symbol_path, exist_ok=True)

        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"features_{timestamp}_v{self.version}.pkl"
        filepath = os.path.join(symbol_path, filename)

        with open(filepath, "wb") as f:
            pickle.dump(self.features_buffer[symbol], f)

        logger.info(f"Persisted {len(self.features_buffer[symbol])} features for {symbol}")

        # Clear buffer
        self.features_buffer[symbol] = []

    def load_features(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load historical features.

        Args:
            symbol: Stock symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame of historical features
        """
        import os
        import pickle

        symbol_path = os.path.join(self.storage_path, symbol)
        if not os.path.exists(symbol_path):
            return pd.DataFrame()

        all_features = []

        # Load all feature files
        for filename in sorted(os.listdir(symbol_path)):
            if filename.startswith("features_") and filename.endswith(".pkl"):
                filepath = os.path.join(symbol_path, filename)

                with open(filepath, "rb") as f:
                    features = pickle.load(f)
                    all_features.extend(features)

        if not all_features:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame([f["features"] for f in all_features])
        df["timestamp"] = pd.to_datetime([f["timestamp"] for f in all_features])
        df["version"] = [f["version"] for f in all_features]
        df.set_index("timestamp", inplace=True)

        # Filter by time range
        if start_time:
            df = df[df.index >= start_time]
        if end_time:
            df = df[df.index <= end_time]

        return df

    def increment_version(self):
        """Increment feature version (e.g., after model update)."""
        self.version += 1
        logger.info(f"Feature store version incremented to {self.version}")

    def get_latest_version(self) -> int:
        """Get current feature version."""
        return self.version

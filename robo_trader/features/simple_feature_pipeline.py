"""
Standalone feature pipeline for M1 that works without configuration.
Provides simple interface for feature engineering.
"""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class TechnicalIndicators:
    """Technical indicator calculations."""

    @staticmethod
    def sma(prices: pd.Series, window: int = 20) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window=window).mean()

    @staticmethod
    def ema(prices: pd.Series, window: int = 20) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2):
        """Bollinger Bands."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14):
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent


class FeaturePipeline:
    """
    Standalone feature pipeline that works without configuration.
    Generates 50+ technical features from OHLCV data.
    """

    def __init__(self, lookback_window: int = 100):
        """
        Initialize feature pipeline.

        Args:
            lookback_window: Maximum lookback period for features
        """
        self.lookback_window = lookback_window
        self.indicators = TechnicalIndicators()
        self.feature_names: List[str] = []

    def generate_features(
        self, data: pd.DataFrame, include_volume: bool = True, include_returns: bool = True
    ) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume
            include_volume: Include volume-based features
            include_returns: Include return-based features

        Returns:
            DataFrame with 50+ technical features
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features["price"] = data["close"]
        features["log_price"] = np.log(data["close"])

        # Returns
        if include_returns:
            features["returns"] = data["close"].pct_change()
            features["log_returns"] = np.log(data["close"] / data["close"].shift(1))
            features["returns_squared"] = features["returns"] ** 2

            # Multi-period returns
            for period in [5, 10, 20]:
                features[f"returns_{period}d"] = data["close"].pct_change(period)

        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            if window <= len(data):
                features[f"sma_{window}"] = self.indicators.sma(data["close"], window)
                features[f"ema_{window}"] = self.indicators.ema(data["close"], window)

        # Price relative to MAs
        for window in [20, 50]:
            if f"sma_{window}" in features:
                features[f"price_to_sma_{window}"] = data["close"] / features[f"sma_{window}"]

        # RSI
        features["rsi_14"] = self.indicators.rsi(data["close"], 14)
        features["rsi_7"] = self.indicators.rsi(data["close"], 7)
        features["rsi_21"] = self.indicators.rsi(data["close"], 21)

        # MACD
        macd, signal, histogram = self.indicators.macd(data["close"])
        features["macd"] = macd
        features["macd_signal"] = signal
        features["macd_histogram"] = histogram

        # Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(data["close"])
        features["bb_upper"] = upper
        features["bb_middle"] = middle
        features["bb_lower"] = lower
        features["bb_width"] = upper - lower
        features["bb_position"] = (data["close"] - lower) / (upper - lower)

        # ATR
        features["atr_14"] = self.indicators.atr(data["high"], data["low"], data["close"], 14)
        features["atr_7"] = self.indicators.atr(data["high"], data["low"], data["close"], 7)

        # Stochastic
        k, d = self.indicators.stochastic(data["high"], data["low"], data["close"])
        features["stoch_k"] = k
        features["stoch_d"] = d

        # Volatility features
        features["volatility_20"] = data["close"].pct_change().rolling(20).std()
        features["volatility_60"] = data["close"].pct_change().rolling(60).std()

        # High/Low features
        features["high_low_ratio"] = data["high"] / data["low"]
        features["close_to_high"] = data["close"] / data["high"]
        features["close_to_low"] = data["close"] / data["low"]

        # Range features
        features["daily_range"] = data["high"] - data["low"]
        features["range_pct"] = (data["high"] - data["low"]) / data["close"]

        # Volume features
        if include_volume and "volume" in data.columns:
            features["volume"] = data["volume"]
            features["volume_sma_20"] = data["volume"].rolling(20).mean()
            features["volume_ratio"] = data["volume"] / features["volume_sma_20"]
            features["volume_change"] = data["volume"].pct_change()

            # VWAP
            typical_price = (data["high"] + data["low"] + data["close"]) / 3
            features["vwap"] = (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()
            features["price_to_vwap"] = data["close"] / features["vwap"]

            # On-Balance Volume
            obv = (np.sign(data["close"].diff()) * data["volume"]).fillna(0).cumsum()
            features["obv"] = obv
            features["obv_sma_20"] = obv.rolling(20).mean()

        # Momentum features
        for period in [5, 10, 20]:
            features[f"momentum_{period}"] = data["close"] - data["close"].shift(period)
            features[f"roc_{period}"] = data["close"].pct_change(period) * 100

        # Support/Resistance levels
        features["resistance_20"] = data["high"].rolling(20).max()
        features["support_20"] = data["low"].rolling(20).min()
        features["price_to_resistance"] = data["close"] / features["resistance_20"]
        features["price_to_support"] = data["close"] / features["support_20"]

        # Store feature names
        self.feature_names = list(features.columns)

        return features

    def calculate_feature_importance(self, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        """
        Calculate feature importance using correlation.

        Args:
            features: Feature DataFrame
            target: Target series (e.g., future returns)

        Returns:
            Series with feature importance scores
        """
        # Simple correlation-based importance
        importance = features.corrwith(target).abs()
        return importance.sort_values(ascending=False)

    def select_features(
        self, features: pd.DataFrame, target: pd.Series, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Select top N most important features.

        Args:
            features: Feature DataFrame
            target: Target series
            top_n: Number of features to select

        Returns:
            DataFrame with selected features
        """
        importance = self.calculate_feature_importance(features, target)
        top_features = importance.head(top_n).index.tolist()
        return features[top_features]

    def get_feature_stats(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for all features.

        Returns:
            DataFrame with feature statistics
        """
        stats = pd.DataFrame(
            {
                "mean": features.mean(),
                "std": features.std(),
                "min": features.min(),
                "max": features.max(),
                "skew": features.skew(),
                "kurtosis": features.kurtosis(),
                "null_count": features.isnull().sum(),
                "null_pct": features.isnull().sum() / len(features),
            }
        )
        return stats


class FeatureStore:
    """Simple feature store for caching and versioning."""

    def __init__(self, cache_dir: str = "./feature_cache"):
        """Initialize feature store."""
        self.cache_dir = cache_dir
        import os

        os.makedirs(cache_dir, exist_ok=True)
        self.cache: Dict[str, pd.DataFrame] = {}

    def save_features(self, symbol: str, features: pd.DataFrame, version: str = "latest") -> None:
        """Save features to store."""
        key = f"{symbol}_{version}"
        self.cache[key] = features.copy()

        # Also save to disk (using pickle instead of parquet to avoid dependency)
        import pickle

        filepath = f"{self.cache_dir}/{key}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(features, f)

    def load_features(self, symbol: str, version: str = "latest") -> Optional[pd.DataFrame]:
        """Load features from store."""
        key = f"{symbol}_{version}"

        # Check memory cache first
        if key in self.cache:
            return self.cache[key].copy()

        # Try loading from disk
        import pickle

        filepath = f"{self.cache_dir}/{key}.pkl"
        import os

        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                features = pickle.load(f)
            self.cache[key] = features
            return features

        return None

    def list_versions(self, symbol: str) -> List[str]:
        """List available versions for a symbol."""
        import os
        import re

        versions = []
        pattern = re.compile(f"{symbol}_(.+)\\.pkl")

        for filename in os.listdir(self.cache_dir):
            match = pattern.match(filename)
            if match:
                versions.append(match.group(1))

        return versions

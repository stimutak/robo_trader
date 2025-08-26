"""
Feature calculation engine for real-time feature updates.

This module implements:
- Efficient calculation with pandas/numpy
- Real-time feature updates
- Multi-timeframe analysis
- Caching layer for performance
- Feature normalization and scaling
"""

import asyncio
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ..config import Config
from ..data.pipeline import BarData, DataSubscriber, TickData
from ..logger import get_logger
from .indicators import IndicatorConfig, TechnicalIndicators


@dataclass
class FeatureSet:
    """Container for calculated features."""

    timestamp: datetime
    symbol: str

    # Price features
    returns_1m: Optional[float] = None
    returns_5m: Optional[float] = None
    returns_15m: Optional[float] = None
    returns_1h: Optional[float] = None
    log_returns: Optional[float] = None

    # Technical indicators
    rsi: Optional[float] = None
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    bb_bandwidth: Optional[float] = None
    atr: Optional[float] = None

    # Volume indicators
    vwap: Optional[float] = None
    obv: Optional[float] = None
    volume_ratio: Optional[float] = None

    # Market microstructure
    spread_bps: Optional[float] = None
    bid_ask_imbalance: Optional[float] = None
    tick_direction: Optional[int] = None  # 1=uptick, -1=downtick, 0=unchanged

    # Multi-timeframe
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # Momentum
    momentum_1d: Optional[float] = None
    momentum_5d: Optional[float] = None
    roc: Optional[float] = None  # Rate of change

    # Volatility
    historical_volatility: Optional[float] = None
    volatility_ratio: Optional[float] = None

    # Custom signals
    trend_strength: Optional[float] = None
    mean_reversion_signal: Optional[float] = None
    breakout_signal: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def get_non_null_features(self) -> Dict[str, float]:
        """Get only calculated (non-null) features."""
        return {
            k: v
            for k, v in self.to_dict().items()
            if v is not None and k not in ["timestamp", "symbol"]
        }


class FeatureCache:
    """Thread-safe feature cache with TTL."""

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache: Dict[str, Tuple[FeatureSet, datetime]] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[FeatureSet]:
        """Get feature from cache if valid."""
        with self.lock:
            if key in self.cache:
                features, cached_at = self.cache[key]
                if (datetime.now() - cached_at).seconds < self.ttl:
                    return features
                del self.cache[key]
        return None

    def set(self, key: str, features: FeatureSet) -> None:
        """Set feature in cache."""
        with self.lock:
            self.cache[key] = (features, datetime.now())

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()


class FeatureEngine(DataSubscriber):
    """Main feature calculation engine."""

    def __init__(self, config: Config):
        super().__init__("feature_engine")
        self.config = config
        self.logger = get_logger("features.engine")

        # Technical indicators calculator
        self.indicators = TechnicalIndicators(IndicatorConfig())

        # Data storage
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.tick_data: Dict[str, List[TickData]] = defaultdict(list)
        self.feature_data: Dict[str, FeatureSet] = {}

        # Feature cache
        self.cache = FeatureCache(config.data.cache_ttl)

        # Control flags
        self.running = False

        # Performance metrics
        self.metrics = {
            "features_calculated": 0,
            "calculation_time_ms": [],
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Initialize data structures for symbols
        for symbol in config.symbols:
            self.price_data[symbol] = pd.DataFrame()

    async def start(self) -> None:
        """Start feature engine."""
        if self.running:
            return

        self.running = True
        self.logger.info("Starting feature engine")

        # Start feature update loop
        asyncio.create_task(self._update_features_loop())

    async def stop(self) -> None:
        """Stop feature engine."""
        self.logger.info("Stopping feature engine")
        self.running = False

    async def on_tick(self, tick: TickData) -> None:
        """Handle incoming tick data."""
        # Store tick for microstructure features
        self.tick_data[tick.symbol].append(tick)

        # Keep only recent ticks (last 1000)
        if len(self.tick_data[tick.symbol]) > 1000:
            self.tick_data[tick.symbol] = self.tick_data[tick.symbol][-1000:]

    async def on_bar(self, bar: BarData) -> None:
        """Handle incoming bar data."""
        # Update price data
        new_row = pd.DataFrame(
            [
                {
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "vwap": bar.vwap,
                }
            ]
        )

        if bar.symbol in self.price_data:
            self.price_data[bar.symbol] = pd.concat(
                [self.price_data[bar.symbol], new_row], ignore_index=True
            )

            # Keep only recent data (configurable window)
            max_rows = (
                self.config.data.feature_window * 10
            )  # Keep extra for longer indicators
            if len(self.price_data[bar.symbol]) > max_rows:
                self.price_data[bar.symbol] = self.price_data[bar.symbol].tail(max_rows)

        # Calculate features for this symbol
        await self.calculate_features(bar.symbol)

    async def _update_features_loop(self) -> None:
        """Periodically update features for all symbols."""
        while self.running:
            try:
                # Update features for all symbols with data
                for symbol in self.price_data.keys():
                    if len(self.price_data[symbol]) > 0:
                        await self.calculate_features(symbol)

                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                self.logger.error(f"Feature update error: {e}")

    async def calculate_features(self, symbol: str) -> Optional[FeatureSet]:
        """Calculate all features for a symbol."""
        try:
            start_time = datetime.now()

            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M')}"
            cached = self.cache.get(cache_key)
            if cached:
                self.metrics["cache_hits"] += 1
                return cached

            self.metrics["cache_misses"] += 1

            # Get price data
            df = self.price_data.get(symbol)
            if df is None or len(df) < 2:
                return None

            # Create feature set
            features = FeatureSet(timestamp=datetime.now(), symbol=symbol)

            # Calculate returns
            if len(df) >= 2:
                features.returns_1m = (
                    df["close"].iloc[-1] / df["close"].iloc[-2] - 1
                ) * 100
                features.log_returns = np.log(
                    df["close"].iloc[-1] / df["close"].iloc[-2]
                )

            if len(df) >= 5:
                features.returns_5m = (
                    df["close"].iloc[-1] / df["close"].iloc[-5] - 1
                ) * 100

            if len(df) >= 15:
                features.returns_15m = (
                    df["close"].iloc[-1] / df["close"].iloc[-15] - 1
                ) * 100

            if len(df) >= 12:  # 1 hour = 12 * 5min bars
                features.returns_1h = (
                    df["close"].iloc[-1] / df["close"].iloc[-12] - 1
                ) * 100

            # Calculate technical indicators
            if len(df) >= 14:  # Minimum for RSI
                features.rsi = self.indicators.rsi(df["close"])

            if len(df) >= 26:  # Minimum for MACD
                macd = self.indicators.macd(df["close"])
                if macd is not None:
                    features.macd_line = macd["macd"]
                    features.macd_signal = macd["signal"]
                    features.macd_histogram = macd["histogram"]

            if len(df) >= 20:  # Minimum for Bollinger Bands
                bb = self.indicators.bollinger_bands(df["close"])
                if bb is not None:
                    features.bb_upper = bb["upper"]
                    features.bb_middle = bb["middle"]
                    features.bb_lower = bb["lower"]
                    features.bb_bandwidth = bb["bandwidth"]

                # SMA
                features.sma_20 = df["close"].rolling(20).mean().iloc[-1]

            if len(df) >= 50:
                features.sma_50 = df["close"].rolling(50).mean().iloc[-1]

            if len(df) >= 200:
                features.sma_200 = df["close"].rolling(200).mean().iloc[-1]

            # ATR
            if len(df) >= 14:
                features.atr = self.indicators.atr(df)

            # Volume indicators
            if "volume" in df.columns and len(df) >= 20:
                features.obv = self.indicators.obv(df)
                features.volume_ratio = (
                    df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
                )

            if "vwap" in df.columns:
                features.vwap = df["vwap"].iloc[-1]

            # Market microstructure from ticks
            if symbol in self.tick_data and self.tick_data[symbol]:
                recent_ticks = self.tick_data[symbol][-100:]  # Last 100 ticks

                # Average spread
                spreads = [t.spread_bps for t in recent_ticks]
                features.spread_bps = np.mean(spreads) if spreads else None

                # Bid-ask imbalance
                bid_sizes = [t.bid_size for t in recent_ticks]
                ask_sizes = [t.ask_size for t in recent_ticks]
                if bid_sizes and ask_sizes:
                    total_bid = sum(bid_sizes)
                    total_ask = sum(ask_sizes)
                    if total_bid + total_ask > 0:
                        features.bid_ask_imbalance = (total_bid - total_ask) / (
                            total_bid + total_ask
                        )

                # Tick direction
                if len(recent_ticks) >= 2:
                    if recent_ticks[-1].last > recent_ticks[-2].last:
                        features.tick_direction = 1
                    elif recent_ticks[-1].last < recent_ticks[-2].last:
                        features.tick_direction = -1
                    else:
                        features.tick_direction = 0

            # Momentum indicators
            if len(df) >= 20:
                features.momentum_1d = (
                    df["close"].iloc[-1] / df["close"].iloc[-12] - 1
                ) * 100  # Approx 1 day
                features.roc = (
                    (df["close"].iloc[-1] - df["close"].iloc[-20])
                    / df["close"].iloc[-20]
                ) * 100

            # Volatility
            if len(df) >= 20:
                returns = df["close"].pct_change().dropna()
                features.historical_volatility = returns.tail(20).std() * np.sqrt(
                    252
                )  # Annualized

                if features.atr and df["close"].iloc[-1] > 0:
                    features.volatility_ratio = features.atr / df["close"].iloc[-1]

            # Custom signals
            features.trend_strength = self._calculate_trend_strength(df)
            features.mean_reversion_signal = self._calculate_mean_reversion_signal(
                features
            )
            features.breakout_signal = self._calculate_breakout_signal(df, features)

            # Store features
            self.feature_data[symbol] = features

            # Cache features
            self.cache.set(cache_key, features)

            # Update metrics
            self.metrics["features_calculated"] += 1
            calc_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["calculation_time_ms"].append(calc_time)
            if len(self.metrics["calculation_time_ms"]) > 100:
                self.metrics["calculation_time_ms"] = self.metrics[
                    "calculation_time_ms"
                ][-100:]

            return features

        except Exception as e:
            self.logger.error(f"Feature calculation error for {symbol}: {e}")
            return None

    def _calculate_trend_strength(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate trend strength indicator."""
        try:
            if len(df) < 20:
                return None

            # Use ADX-like calculation
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift())
            low_close = abs(df["low"] - df["close"].shift())

            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()

            # Directional movement
            up_move = df["high"] - df["high"].shift()
            down_move = df["low"].shift() - df["low"]

            pos_dm = pd.Series(
                np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                index=df.index,
            )
            neg_dm = pd.Series(
                np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                index=df.index,
            )

            pos_di = 100 * (pos_dm.rolling(14).mean() / atr)
            neg_di = 100 * (neg_dm.rolling(14).mean() / atr)

            dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 0.0001)
            adx = dx.rolling(14).mean()

            return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else None

        except Exception:
            return None

    def _calculate_mean_reversion_signal(self, features: FeatureSet) -> Optional[float]:
        """Calculate mean reversion signal."""
        try:
            if features.rsi is None:
                return None

            # Combine RSI and Bollinger Band position
            signal = 0.0

            # RSI signal (oversold < 30, overbought > 70)
            if features.rsi < 30:
                signal += (30 - features.rsi) / 30  # Stronger signal as RSI gets lower
            elif features.rsi > 70:
                signal -= (features.rsi - 70) / 30  # Negative signal for overbought

            # Bollinger Band signal
            if features.bb_upper and features.bb_lower and features.bb_middle:
                # Normalize price position within bands
                band_width = features.bb_upper - features.bb_lower
                if band_width > 0:
                    position = (features.bb_middle - features.bb_lower) / band_width
                    if position < 0.2:  # Near lower band
                        signal += (0.2 - position) * 2
                    elif position > 0.8:  # Near upper band
                        signal -= (position - 0.8) * 2

            return signal

        except Exception:
            return None

    def _calculate_breakout_signal(
        self, df: pd.DataFrame, features: FeatureSet
    ) -> Optional[float]:
        """Calculate breakout signal."""
        try:
            if len(df) < 20:
                return None

            signal = 0.0

            # Volume breakout
            if features.volume_ratio and features.volume_ratio > 1.5:
                signal += 0.5

            # Price breakout from Bollinger Bands
            if features.bb_upper and features.bb_lower:
                current_price = df["close"].iloc[-1]
                if current_price > features.bb_upper:
                    signal += 1.0
                elif current_price < features.bb_lower:
                    signal -= 1.0

            # Volatility breakout
            if features.volatility_ratio:
                recent_volatility = df["close"].pct_change().tail(5).std()
                avg_volatility = df["close"].pct_change().tail(20).std()
                if recent_volatility > avg_volatility * 1.5:
                    signal += 0.5

            return signal

        except Exception:
            return None

    def get_features(self, symbol: str) -> Optional[FeatureSet]:
        """Get latest calculated features for symbol."""
        return self.feature_data.get(symbol)

    def get_all_features(self) -> Dict[str, FeatureSet]:
        """Get all calculated features."""
        return self.feature_data.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        avg_calc_time = (
            np.mean(self.metrics["calculation_time_ms"])
            if self.metrics["calculation_time_ms"]
            else 0
        )

        return {
            "features_calculated": self.metrics["features_calculated"],
            "avg_calculation_time_ms": avg_calc_time,
            "cache_hits": self.metrics["cache_hits"],
            "cache_misses": self.metrics["cache_misses"],
            "cache_hit_rate": self.metrics["cache_hits"]
            / (self.metrics["cache_hits"] + self.metrics["cache_misses"] + 0.0001),
            "symbols_tracked": len(self.price_data),
            "features_per_symbol": len(FeatureSet.__dataclass_fields__)
            - 2,  # Exclude timestamp and symbol
        }

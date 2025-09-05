"""
Real-time data pipeline for market data ingestion from IBKR.

This module implements:
- Real-time tick data streaming from IBKR
- WebSocket-style event publishing
- Data buffering and queuing
- Subscriber pattern for strategies
- Historical data backfill
- Missing data handling
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Full, Queue
from typing import Any, Callable, Deque, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from ..config import Config
from ..logger import get_logger


@dataclass
class TickData:
    """Real-time tick data structure."""

    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    last: float
    bid_size: int
    ask_size: int
    last_size: int
    volume: int
    open_interest: Optional[int] = None

    @property
    def mid(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        if self.mid > 0:
            return (self.spread / self.mid) * 10000
        return 0


@dataclass
class BarData:
    """OHLCV bar data structure."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    trades: Optional[int] = None

    @property
    def range(self) -> float:
        """Calculate bar range."""
        return self.high - self.low

    @property
    def true_range(self, prev_close: Optional[float] = None) -> float:
        """Calculate true range."""
        if prev_close is None:
            return self.range
        return max(
            self.high - self.low,
            abs(self.high - prev_close),
            abs(self.low - prev_close),
        )


class DataBuffer:
    """Thread-safe circular buffer for market data."""

    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.buffer: Deque[Any] = deque(maxlen=maxsize)
        self.lock = threading.Lock()

    def append(self, item: Any) -> None:
        """Add item to buffer."""
        with self.lock:
            self.buffer.append(item)

    def get_all(self) -> List[Any]:
        """Get all items from buffer."""
        with self.lock:
            return list(self.buffer)

    def get_latest(self, n: int = 1) -> List[Any]:
        """Get latest n items."""
        with self.lock:
            if n >= len(self.buffer):
                return list(self.buffer)
            return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()

    def __len__(self) -> int:
        with self.lock:
            return len(self.buffer)


class DataSubscriber:
    """Base class for data pipeline subscribers."""

    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"subscriber.{name}")

    async def on_tick(self, tick: TickData) -> None:
        """Handle tick data."""
        pass

    async def on_bar(self, bar: BarData) -> None:
        """Handle bar data."""
        pass

    async def on_error(self, error: Exception) -> None:
        """Handle errors."""
        self.logger.error(f"Subscriber error: {error}")


class DataPublisher:
    """Publisher for market data events."""

    def __init__(self):
        self.subscribers: Set[DataSubscriber] = set()
        self.logger = get_logger("data.publisher")

    def subscribe(self, subscriber: DataSubscriber) -> None:
        """Add subscriber."""
        self.subscribers.add(subscriber)
        self.logger.info(f"Added subscriber: {subscriber.name}")

    def unsubscribe(self, subscriber: DataSubscriber) -> None:
        """Remove subscriber."""
        self.subscribers.discard(subscriber)
        self.logger.info(f"Removed subscriber: {subscriber.name}")

    async def publish_tick(self, tick: TickData) -> None:
        """Publish tick to all subscribers."""
        tasks = []
        for subscriber in self.subscribers:
            tasks.append(subscriber.on_tick(tick))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def publish_bar(self, bar: BarData) -> None:
        """Publish bar to all subscribers."""
        tasks = []
        for subscriber in self.subscribers:
            tasks.append(subscriber.on_bar(bar))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class DataPipeline:
    """Main data pipeline for real-time market data."""

    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("data.pipeline")

        # Data buffers
        self.tick_buffer: Dict[str, DataBuffer] = {}
        self.bar_buffer: Dict[str, DataBuffer] = {}

        # Publisher
        self.publisher = DataPublisher()

        # Control flags
        self.running = False
        self.streaming_enabled = config.data.enable_realtime

        # Performance metrics
        self.metrics = {
            "ticks_received": 0,
            "bars_received": 0,
            "last_tick_time": None,
            "data_gaps": 0,
            "reconnects": 0,
        }

        # Historical data cache
        self.historical_cache: Dict[str, pd.DataFrame] = {}

        # Initialize buffers for symbols
        for symbol in config.symbols:
            self.tick_buffer[symbol] = DataBuffer(config.data.tick_buffer)
            self.bar_buffer[symbol] = DataBuffer(100)  # Keep 100 bars

    async def start(self) -> None:
        """Start data pipeline."""
        if self.running:
            self.logger.warning("Data pipeline already running")
            return

        self.running = True
        self.logger.info("Starting data pipeline")

        # Start streaming if enabled
        if self.streaming_enabled:
            asyncio.create_task(self._streaming_loop())

        # Start data quality monitoring
        asyncio.create_task(self._monitor_data_quality())

    async def stop(self) -> None:
        """Stop data pipeline."""
        self.logger.info("Stopping data pipeline")
        self.running = False

    async def _streaming_loop(self) -> None:
        """Main streaming loop for real-time data."""
        while self.running:
            try:
                # This will be replaced with actual IBKR streaming
                # For now, simulate with mock data
                await self._process_mock_tick()
                await asyncio.sleep(0.1)  # Simulate tick rate

            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                await asyncio.sleep(1)

    async def _process_mock_tick(self) -> None:
        """Process mock tick for testing."""
        # Generate mock tick data
        for symbol in self.config.symbols[:3]:  # Limit to 3 symbols for testing
            tick = TickData(
                timestamp=datetime.now(),
                symbol=symbol,
                bid=100 + np.random.randn(),
                ask=100.05 + np.random.randn(),
                last=100.02 + np.random.randn(),
                bid_size=100,
                ask_size=100,
                last_size=100,
                volume=int(np.random.uniform(1000000, 5000000)),
            )

            # Store in buffer
            if symbol in self.tick_buffer:
                self.tick_buffer[symbol].append(tick)

            # Publish to subscribers
            await self.publisher.publish_tick(tick)

            # Update metrics with market time
            import pytz

            market_tz = pytz.timezone("US/Eastern")
            self.metrics["ticks_received"] += 1
            self.metrics["last_tick_time"] = datetime.now(market_tz)

    async def _monitor_data_quality(self) -> None:
        """Monitor data quality and detect issues."""
        import pytz

        while self.running:
            try:
                # Check for data gaps using market time
                market_tz = pytz.timezone("US/Eastern")
                now = datetime.now(market_tz)

                if self.metrics["last_tick_time"]:
                    # Ensure both timestamps are timezone-aware
                    if self.metrics["last_tick_time"].tzinfo is None:
                        last_tick = market_tz.localize(self.metrics["last_tick_time"])
                    else:
                        last_tick = self.metrics["last_tick_time"].astimezone(market_tz)

                    gap = (now - last_tick).total_seconds()
                    if gap > 5 and self.streaming_enabled:  # 5 second gap
                        self.metrics["data_gaps"] += 1
                        self.logger.warning(f"Data gap detected: {gap:.1f} seconds")

                # Log metrics periodically
                if self.metrics["ticks_received"] % 1000 == 0:
                    self.logger.info(f"Pipeline metrics: {self.metrics}")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")

    def subscribe(self, subscriber: DataSubscriber) -> None:
        """Subscribe to data events."""
        self.publisher.subscribe(subscriber)

    def unsubscribe(self, subscriber: DataSubscriber) -> None:
        """Unsubscribe from data events."""
        self.publisher.unsubscribe(subscriber)

    async def get_historical_data(
        self, symbol: str, duration: str = "30 D", bar_size: str = "5 mins"
    ) -> Optional[pd.DataFrame]:
        """Get historical data for backtesting/warmup."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{duration}_{bar_size}"
            if cache_key in self.historical_cache:
                cached_data = self.historical_cache[cache_key]
                # Check if cache is still valid (< 5 minutes old)
                if "cached_at" in cached_data.attrs:
                    age = (datetime.now() - cached_data.attrs["cached_at"]).seconds
                    if age < self.config.data.cache_ttl:
                        return cached_data

            # Fetch from IBKR (mock for now)
            self.logger.info(f"Fetching historical data: {symbol} {duration} {bar_size}")

            # Generate mock historical data
            end_date = datetime.now()
            days = int(duration.split()[0]) if "D" in duration else 1
            dates = pd.date_range(
                end=end_date, periods=days * 78, freq="5min"
            )  # 78 5-min bars per day

            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": np.random.randn(len(dates)) + 100,
                    "high": np.random.randn(len(dates)) + 101,
                    "low": np.random.randn(len(dates)) + 99,
                    "close": np.random.randn(len(dates)) + 100,
                    "volume": np.random.uniform(100000, 1000000, len(dates)),
                }
            )

            # Add metadata
            df.attrs["symbol"] = symbol
            df.attrs["bar_size"] = bar_size
            df.attrs["cached_at"] = datetime.now()

            # Cache the data
            self.historical_cache[cache_key] = df

            return df

        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            return None

    def get_latest_ticks(self, symbol: str, n: int = 100) -> List[TickData]:
        """Get latest n ticks for symbol."""
        if symbol in self.tick_buffer:
            return self.tick_buffer[symbol].get_latest(n)
        return []

    def get_latest_bars(self, symbol: str, n: int = 20) -> List[BarData]:
        """Get latest n bars for symbol."""
        if symbol in self.bar_buffer:
            return self.bar_buffer[symbol].get_latest(n)
        return []

    async def aggregate_ticks_to_bars(
        self, symbol: str, interval: timedelta = timedelta(minutes=5)
    ) -> Optional[BarData]:
        """Aggregate ticks into OHLCV bars."""
        try:
            ticks = self.get_latest_ticks(symbol, 1000)
            if not ticks:
                return None

            # Group ticks by interval
            now = datetime.now()
            cutoff = now - interval

            interval_ticks = [t for t in ticks if t.timestamp >= cutoff]
            if not interval_ticks:
                return None

            # Calculate OHLCV
            prices = [t.last for t in interval_ticks]
            volumes = [t.last_size for t in interval_ticks]

            bar = BarData(
                timestamp=now,
                symbol=symbol,
                open=prices[0],
                high=max(prices),
                low=min(prices),
                close=prices[-1],
                volume=sum(volumes),
                vwap=np.average(prices, weights=volumes) if volumes else prices[-1],
                trades=len(interval_ticks),
            )

            # Store in buffer
            self.bar_buffer[symbol].append(bar)

            # Publish to subscribers
            await self.publisher.publish_bar(bar)

            self.metrics["bars_received"] += 1

            return bar

        except Exception as e:
            self.logger.error(f"Failed to aggregate ticks: {e}")
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        return {
            **self.metrics,
            "buffer_sizes": {symbol: len(buffer) for symbol, buffer in self.tick_buffer.items()},
            "cache_size": len(self.historical_cache),
            "subscribers": len(self.publisher.subscribers),
        }


# Example usage for testing
class ExampleSubscriber(DataSubscriber):
    """Example subscriber for testing."""

    async def on_tick(self, tick: TickData) -> None:
        """Handle tick data."""
        self.logger.debug(
            f"Tick: {tick.symbol} bid={tick.bid:.2f} ask={tick.ask:.2f} "
            f"spread={tick.spread_bps:.1f}bps"
        )

    async def on_bar(self, bar: BarData) -> None:
        """Handle bar data."""
        self.logger.info(
            f"Bar: {bar.symbol} OHLCV=({bar.open:.2f}, {bar.high:.2f}, "
            f"{bar.low:.2f}, {bar.close:.2f}, {bar.volume:,})"
        )

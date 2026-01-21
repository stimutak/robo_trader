"""
Data Fetcher Module - Market data retrieval and caching.

This module handles:
- Fetching historical bars from IBKR
- Storing market data in the database
- Managing the market data cache with LRU eviction

Extracted from runner_async.py to improve modularity.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

import pandas as pd
from ib_async import Stock

from ..logger import get_logger
from ..market_hours import get_market_session, is_market_open
from ..monitoring.performance import Timer

if TYPE_CHECKING:
    from ..database_async import AsyncTradingDatabase
    from ..monitoring.performance import PerformanceMonitor
    from ..monitoring.production_monitor import ProductionMonitor

logger = get_logger(__name__)


class IBKRClient(Protocol):
    """Protocol for IBKR client interface."""

    async def get_historical_bars(
        self,
        symbol: str,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
    ) -> List[Dict[str, Any]]:
        """Fetch historical bars from IBKR."""
        ...

    def isConnected(self) -> bool:
        """Check if connected to IBKR."""
        ...


class DataFetcher:
    """Handles market data fetching, storage, and caching."""

    def __init__(
        self,
        ib_client: Any,  # IBKRClient or legacy IB
        database: AsyncTradingDatabase,
        monitor: PerformanceMonitor,
        production_monitor: Optional[ProductionMonitor] = None,
        cache: Optional[Dict[str, pd.DataFrame]] = None,
        max_cache_size: int = 100,
        duration: str = "1 D",
        bar_size: str = "1 min",
    ):
        """
        Initialize the DataFetcher.

        Args:
            ib_client: IBKR client (subprocess or legacy)
            database: Async trading database
            monitor: Performance monitor for metrics
            production_monitor: Production monitor for API call tracking
            cache: Market data cache (shared with runner)
            max_cache_size: Maximum number of symbols to cache
            duration: Default duration for historical data requests
            bar_size: Default bar size for historical data requests
        """
        self.ib = ib_client
        self.db = database
        self.monitor = monitor
        self.production_monitor = production_monitor
        self.market_data_cache = cache if cache is not None else {}
        self.max_cache_size = max_cache_size
        self.duration = duration
        self.bar_size = bar_size

    async def fetch_historical_bars(
        self,
        symbol: str,
        duration: Optional[str] = None,
        bar_size: Optional[str] = None,
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch historical bars from IBKR and return normalized DataFrame.

        Args:
            symbol: Ticker symbol
            duration: IB duration string (default: self.duration)
            bar_size: IB bar size (default: self.bar_size)
            what_to_show: Data type to request
            use_rth: Restrict to regular trading hours

        Returns:
            DataFrame with OHLCV columns

        Raises:
            ConnectionError: If not connected to IBKR
        """
        if not self.ib:
            raise ConnectionError("Not connected to IBKR")

        duration = duration or self.duration
        bar_size = bar_size or self.bar_size

        # Check if subprocess client or legacy IB client
        if hasattr(self.ib, "get_historical_bars"):
            # Subprocess client - use async method
            bars = await self.ib.get_historical_bars(
                symbol=symbol,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
            )

            if not bars:
                return pd.DataFrame()

            # Convert list of dicts to DataFrame
            df = pd.DataFrame(bars)
            if not df.empty:
                # Ensure columns are lowercase
                df.columns = [col.lower() for col in df.columns]
                # Sort by date
                if "date" in df.columns:
                    df = df.sort_values("date")
        else:
            # Legacy IB client - use synchronous methods
            if not self.ib.isConnected():
                raise ConnectionError("Not connected to IBKR")

            contract = Stock(symbol, "SMART", "USD")
            qualified = self.ib.qualifyContracts(contract)
            if not qualified:
                return pd.DataFrame()

            bars = self.ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars)
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df = df.sort_values("date")

        return df

    async def fetch_and_store(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch market data for a symbol and store in database.

        Args:
            symbol: Ticker symbol

        Returns:
            DataFrame with market data, or None if fetch failed
        """
        # Check if market is open before fetching data
        if not is_market_open():
            session = get_market_session()
            logger.debug(f"Market is {session}, skipping data fetch for {symbol}")
            return None

        try:
            start_time = asyncio.get_event_loop().time()
            with Timer("data_fetch", self.monitor):
                # Fetch historical bars using IB connection directly
                df = await self.fetch_historical_bars(
                    symbol=symbol, duration=self.duration, bar_size=self.bar_size
                )

            # Record API call metrics to ProductionMonitor (Phase 4 P2)
            if self.production_monitor:
                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
                self.production_monitor.record_api_call(
                    "fetch_historical_bars", df is not None and not df.empty, latency_ms
                )

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Store market data in database
            logger.info(f"Fetched {len(df)} bars for {symbol}")

            # Prepare batch data for efficient storage
            batch_data = []
            for timestamp, row in df.iterrows():
                # Convert pandas Timestamp to datetime for SQLite compatibility
                if hasattr(timestamp, "to_pydatetime"):
                    timestamp = timestamp.to_pydatetime()
                batch_data.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp,
                        "open": float(row.get("open", 0)),
                        "high": float(row.get("high", 0)),
                        "low": float(row.get("low", 0)),
                        "close": float(row.get("close", 0)),
                        "volume": int(row.get("volume", 0)),
                    }
                )

            if batch_data:
                with Timer("database_write", self.monitor):
                    await self.db.batch_store_market_data(batch_data)
                self.monitor.record_data_points(len(batch_data))

            # Update cache
            self.market_data_cache[symbol] = df
            self.manage_cache_size()

            return df

        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return None

    def manage_cache_size(self) -> None:
        """Ensure cache doesn't exceed max size (LRU eviction)."""
        while len(self.market_data_cache) > self.max_cache_size:
            # Remove oldest item (first item in OrderedDict)
            oldest_symbol = next(iter(self.market_data_cache))
            del self.market_data_cache[oldest_symbol]
            logger.debug(f"Evicted {oldest_symbol} from market data cache")

    def get_cached_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get cached market data for a symbol."""
        return self.market_data_cache.get(symbol)

    def clear_cache(self) -> None:
        """Clear the market data cache."""
        self.market_data_cache.clear()
        logger.debug("Market data cache cleared")

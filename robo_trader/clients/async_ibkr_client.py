"""
Async IBKR Client with retry logic, connection pooling, and proper async patterns.

This replaces the synchronous calls in ibkr_client.py with proper async/await patterns,
adds exponential backoff on failures, and implements connection pooling.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from ib_insync import IB, Contract, Stock, util
from ib_insync.util import patchAsyncio
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Enable nested event loops - REQUIRED for ib_insync to work in async context
patchAsyncio()

logger = logging.getLogger(__name__)


def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IB historical bars DataFrame for deterministic downstream use.
    - Lowercase column names
    - Keep standard OHLCV columns when present
    - Coerce numeric types where possible
    - Drop rows with NaN in close
    - Sort by index if datetime index, else by a 'date'/'time' column if present
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]
    # Standard column subset if present
    preferred_cols = [
        c for c in ["date", "time", "open", "high", "low", "close", "volume"] if c in data.columns
    ]
    if preferred_cols:
        data = data[preferred_cols]
    # Coerce numerics
    for col in ["open", "high", "low", "close", "volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    # Drop rows without close
    if "close" in data.columns:
        data = data.dropna(subset=["close"])
    # Sort chronologically
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    elif "date" in data.columns:
        try:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.sort_values("date")
        except (ValueError, AttributeError, KeyError) as e:
            logger.warning(f"Failed to process date column in market data: {e}")
            # Continue without date sorting - data may still be usable
    elif "time" in data.columns:
        try:
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
            data = data.sort_values("time")
        except (ValueError, AttributeError, KeyError) as e:
            logger.warning(f"Failed to process time column in market data: {e}")
            # Continue without time sorting - data may still be usable
    return data.reset_index(drop=False)


@dataclass
class ConnectionConfig:
    """Configuration for IBKR connection."""

    host: str = "127.0.0.1"
    port: int = 7497  # Will auto-detect: 7497 (TWS), 4001 (Gateway Paper), 4002 (Gateway Live)
    client_id: int = 1
    readonly: bool = True
    timeout: float = 10.0
    max_connections: int = 1  # TWS/Gateway only supports one connection per client ID
    retry_attempts: int = 1  # Reduced to avoid stuck connections
    retry_max_wait: float = 10.0
    auto_detect_port: bool = True  # Auto-detect TWS vs Gateway


class ConnectionPool:
    """Manages a pool of IBKR connections for concurrent operations."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.pool: List[IB] = []
        self.available: asyncio.Queue = asyncio.Queue(maxsize=config.max_connections)
        self.semaphore = asyncio.Semaphore(config.max_connections)
        self._initialized = False
        self._lock = asyncio.Lock()
        self._detect_port_if_needed()

    def _detect_port_if_needed(self):
        """Auto-detect whether TWS or IB Gateway is running."""
        if not self.config.auto_detect_port:
            return

        import socket

        # Check ports in order of preference
        ports_to_check = [
            (4002, "IB Gateway (Paper)"),
            (4001, "IB Gateway (Live)"),
            (7497, "TWS (Paper)"),
            (7496, "TWS (Live)"),
        ]

        for port, name in ports_to_check:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            try:
                result = sock.connect_ex((self.config.host, port))
                if result == 0:
                    logger.info(f"Auto-detected {name} on port {port}")
                    self.config.port = port
                    break
            except:
                pass
            finally:
                sock.close()

    async def initialize(self):
        """Initialize the connection pool."""
        async with self._lock:
            if self._initialized:
                return

            for i in range(self.config.max_connections):
                ib = await self._create_connection(self.config.client_id + i)
                self.pool.append(ib)
                await self.available.put(ib)

            self._initialized = True
            logger.info(
                f"Initialized connection pool with {self.config.max_connections} connections"
            )

    async def _create_connection(self, client_id: int) -> IB:
        """Create a single IBKR connection with retry logic."""

        @retry(
            stop=stop_after_attempt(self.config.retry_attempts),
            wait=wait_exponential(multiplier=1, max=self.config.retry_max_wait),
            retry=retry_if_exception_type((ConnectionError, asyncio.TimeoutError)),
        )
        async def connect_with_retry():
            ib = IB()
            try:
                # Use SYNC connect with patchAsyncio() enabled
                logger.info(f"Attempting connection with client ID {client_id} (timeout=20s)...")
                ib.connect(
                    self.config.host,
                    self.config.port,
                    clientId=client_id,
                    timeout=20,  # Increased timeout for fresh TWS/Gateway connections
                    readonly=self.config.readonly,
                )
                logger.info(f"✓ Successfully connected with client ID {client_id}")
                return ib
            except Exception as e:
                logger.warning(f"Connection attempt failed for client ID {client_id}: {e}")
                # Always try to disconnect to clean up resources
                try:
                    if ib and hasattr(ib, "disconnect"):
                        ib.disconnect()
                        logger.debug(f"Cleaned up failed connection for client ID {client_id}")
                except:
                    pass  # Ignore cleanup errors
                raise

        return await connect_with_retry()

    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0):
        """Acquire a connection from the pool with timeout."""
        if not self._initialized:
            await self.initialize()

        try:
            connection = await asyncio.wait_for(self.available.get(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConnectionError(f"Connection pool exhausted after {timeout}s timeout")

        try:
            # Verify connection is still alive
            if not connection.isConnected():
                logger.warning("Connection lost, reconnecting...")
                client_id = connection.client.clientId
                connection.disconnect()
                connection = await self._create_connection(client_id)
                # Update pool reference
                for i, conn in enumerate(self.pool):
                    if conn.client.clientId == client_id:
                        self.pool[i] = connection
                        break

            yield connection
        finally:
            await self.available.put(connection)

    async def close_all(self):
        """Close all connections in the pool."""
        for connection in self.pool:
            if connection.isConnected():
                connection.disconnect()
        self.pool.clear()
        self._initialized = False
        logger.info("Closed all connections in pool")


class AsyncIBKRClient:
    """
    Async IBKR client with proper async patterns, retry logic, and connection pooling.

    Key improvements over the original:
    - All operations are truly async (no synchronous qualifyContracts)
    - Connection pooling for concurrent operations
    - Exponential backoff retry logic
    - Proper error handling and logging
    - Market hours validation
    - Request throttling to avoid rate limits
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self.pool = ConnectionPool(self.config)
        self._contract_cache: Dict[str, Contract] = {}
        self._rate_limiter = asyncio.Semaphore(50)  # Max 50 concurrent requests

    async def connect(self):
        """Initialize the connection pool."""
        await self.pool.initialize()

    async def disconnect(self):
        """Close all connections."""
        await self.pool.close_all()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
    )
    async def qualify_stock(
        self, symbol: str, exchange: str = "SMART", currency: str = "USD"
    ) -> Contract:
        """
        Qualify a stock contract asynchronously with caching.

        Args:
            symbol: Stock symbol
            exchange: Exchange (default: SMART)
            currency: Currency (default: USD)

        Returns:
            Qualified contract
        """
        cache_key = f"{symbol}:{exchange}:{currency}"
        if cache_key in self._contract_cache:
            return self._contract_cache[cache_key]

        async with self.pool.acquire() as ib:
            contract = Stock(symbol, exchange, currency)

            # Use async version of qualifyContracts
            qualified = await asyncio.wait_for(ib.qualifyContractsAsync(contract), timeout=5.0)

            if not qualified:
                raise RuntimeError(f"Unable to qualify contract for {symbol}")

            self._contract_cache[cache_key] = qualified[0]
            return qualified[0]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
    )
    async def fetch_recent_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch recent bars with retry logic and rate limiting.

        Args:
            symbol: Stock symbol
            duration: IB duration string (e.g., "2 D", "30 D")
            bar_size: IB bar size (e.g., "1 min", "5 mins", "1 hour")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with normalized OHLCV data
        """
        async with self._rate_limiter:
            contract = await self.qualify_stock(symbol)

            async with self.pool.acquire() as ib:
                bars = await asyncio.wait_for(
                    ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime="",
                        durationStr=duration,
                        barSizeSetting=bar_size,
                        whatToShow=what_to_show,
                        useRTH=use_rth,
                        formatDate=1,
                    ),
                    timeout=30.0,
                )

                if not bars:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()

                raw_df = util.df(bars)
                return self._normalize_bars_df(raw_df)

    async def fetch_multiple_symbols(
        self,
        symbols: List[str],
        duration: str = "2 D",
        bar_size: str = "5 mins",
        max_concurrent: int = 8,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            duration: IB duration string
            bar_size: IB bar size
            max_concurrent: Max number of concurrent requests

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = {}

        async def fetch_with_semaphore(symbol: str):
            async with semaphore:
                try:
                    df = await self.fetch_recent_bars(symbol, duration, bar_size)
                    return symbol, df
                except Exception as e:
                    logger.error(f"Failed to fetch data for {symbol}: {e}")
                    return symbol, pd.DataFrame()

        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        completed = await asyncio.gather(*tasks, return_exceptions=False)

        for symbol, df in completed:
            results[symbol] = df

        return results

    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary with key metrics."""
        async with self.pool.acquire() as ib:
            summary = await ib.accountSummaryAsync()
            return {item.tag: item.value for item in summary}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        async with self.pool.acquire() as ib:
            positions = await ib.reqPositionsAsync()
            return [
                {
                    "symbol": pos.contract.symbol,
                    "quantity": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.marketValue,
                    "unrealized_pnl": pos.unrealizedPNL,
                }
                for pos in positions
            ]

    def _normalize_bars_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize IB historical bars DataFrame.

        Args:
            df: Raw DataFrame from IB

        Returns:
            Normalized DataFrame with standard OHLCV columns
        """
        if df is None or df.empty:
            return pd.DataFrame()

        data = df.copy()
        data.columns = [str(c).lower() for c in data.columns]

        # Standard column subset
        preferred_cols = [
            c for c in ["date", "open", "high", "low", "close", "volume"] if c in data.columns
        ]
        if preferred_cols:
            data = data[preferred_cols]

        # Coerce numerics
        for col in ["open", "high", "low", "close", "volume"]:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")

        # Drop rows without close
        if "close" in data.columns:
            data = data.dropna(subset=["close"])

        # Set datetime index
        if "date" in data.columns:
            try:
                data["date"] = pd.to_datetime(data["date"], errors="coerce")
                data = data.set_index("date").sort_index()
            except (ValueError, AttributeError, KeyError) as e:
                self.logger.warning(f"Failed to process date index in market data: {e}")
                # Continue without date indexing - data may still be usable

        return data

    def is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        weekday = now.weekday()

        # Skip weekends
        if weekday >= 5:
            return False

        # Check time (9:30 AM - 4:30 PM ET)
        # This is a simplified check - production should use exchange calendars
        hour = now.hour
        minute = now.minute
        time_minutes = hour * 60 + minute

        # Convert to ET (assuming system is in ET or adjust accordingly)
        market_open = 9 * 60 + 30  # 9:30 AM
        market_close = 16 * 60 + 30  # 4:30 PM

        return market_open <= time_minutes < market_close


# Backward compatibility wrapper
async def create_client(config: Optional[ConnectionConfig] = None) -> AsyncIBKRClient:
    """Create and initialize an async IBKR client."""
    client = AsyncIBKRClient(config)
    await client.connect()
    return client

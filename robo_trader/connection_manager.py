"""
Robust IB connection manager that handles common edge cases with ib_insync.

This module introduces two primary classes:
- ConnectionManager: Responsible for establishing and maintaining a resilient
  connection to IBKR using ib_insync with retries, backoff, and cleanup.
- IBKRClient: A thin, practical wrapper that uses ConnectionManager and exposes
  a few convenience methods plus an async context manager for ergonomic usage.

Notes:
- Defaults to connecting to local TWS/Gateway. Configure via environment vars
  IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID. Paper/live is controlled externally.
- Avoids noisy logging and prints; this module only logs via the standard logger.
- Designed to be a drop-in utility without changing existing callers elsewhere.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import random
import signal
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from ib_insync import IB, Contract, LimitOrder, MarketOrder, Option, Stock, util
except ImportError as import_err:
    print("ERROR: ib_insync not installed. Run: pip install ib_insync")
    raise


logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Bulletproof-ish connection manager for Interactive Brokers via ib_insync.

    Handles edge cases that commonly cause timeouts or stale connections by
    cleaning up previous sessions, rotating client IDs, and verifying liveness
    after connect. Intended to be used from async code.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
    ) -> None:
        # Load from environment if not provided
        self.host: str = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port: int = int(port or os.getenv("IBKR_PORT", "7497"))
        self.base_client_id: int = int(client_id or self._generate_client_id())

        self.ib: Optional[IB] = None
        self._connected: bool = False
        self._connecting: bool = False
        self._connection_count: int = 0
        self._cleanup_registered: bool = False

        # Track failed client IDs to avoid immediate reuse
        self._failed_client_ids: set[int] = set()
        self._max_retries: int = 5
        self._retry_delay: float = 2.0

        logger.info(f"Connection manager initialized for {self.host}:{self.port}")

    def _generate_client_id(self) -> int:
        """Generate a semi-unique client ID to avoid collisions."""
        base_id = int(time.time()) % 10000
        random_offset = random.randint(0, 99)
        return base_id + random_offset

    def _get_next_client_id(self) -> int:
        """Get next available client ID, avoiding those that recently failed."""
        while True:
            self._connection_count += 1
            client_id = self.base_client_id + self._connection_count

            if client_id not in self._failed_client_ids:
                return client_id

            if self._connection_count > 100:
                self.base_client_id = self._generate_client_id()
                self._connection_count = 0
                self._failed_client_ids.clear()

    async def _cleanup_connection(self) -> None:
        """Properly clean up IB connection and event loop tasks."""
        if self.ib is not None:
            try:
                if self.ib.isConnected():
                    logger.info("Disconnecting from IB...")
                    self.ib.disconnect()
                    await asyncio.sleep(0.5)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Error during disconnect: {e}")
            finally:
                self.ib = None
                self._connected = False

        await self._cleanup_pending_tasks()

    async def _cleanup_pending_tasks(self) -> None:
        """Cancel all pending tasks to prevent event loop contamination."""
        try:
            tasks = [
                t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()
            ]
            if tasks:
                logger.debug(f"Cancelling {len(tasks)} pending tasks")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Error cleaning up tasks: {e}")

    async def connect(self) -> IB:
        """Establish connection with comprehensive retry logic and cleanup."""
        if self._connecting:
            logger.warning("Already attempting connection")
            return self.ib  # type: ignore[return-value]

        if self._connected and self.ib and self.ib.isConnected():
            try:
                await asyncio.wait_for(self._verify_connection(), timeout=2.0)
                logger.info("Using existing connection")
                return self.ib
            except Exception:  # noqa: BLE001
                logger.warning("Existing connection is dead, reconnecting...")
                await self._cleanup_connection()

        self._connecting = True

        try:
            await self._cleanup_connection()

            for attempt in range(self._max_retries):
                client_id = self._get_next_client_id()
                logger.info(
                    f"Connection attempt {attempt + 1}/{self._max_retries} with client_id={client_id}"
                )

                try:
                    self.ib = IB()
                    # Configure connection parameters
                    self.ib.RequestTimeout = 10.0
                    self.ib.RaiseRequestErrors = False

                    # Attempt connection
                    await asyncio.wait_for(
                        self.ib.connectAsync(
                            self.host,
                            self.port,
                            clientId=client_id,
                            timeout=10.0,
                            readonly=False,
                        ),
                        timeout=12.0,
                    )

                    # Verify connection works
                    await asyncio.wait_for(self._verify_connection(), timeout=5.0)

                    self._connected = True
                    logger.info(f"\u2713 Successfully connected with client_id={client_id}")

                    if not self._cleanup_registered:
                        self._register_cleanup_handlers()
                        self._cleanup_registered = True

                    self._failed_client_ids.clear()
                    return self.ib

                except asyncio.TimeoutError:
                    logger.warning(f"Connection timeout on attempt {attempt + 1}")
                    self._failed_client_ids.add(client_id)
                    await self._cleanup_connection()

                except Exception as e:  # noqa: BLE001
                    error_msg = str(e).lower()
                    if "already connected" in error_msg or "client id" in error_msg:
                        self._failed_client_ids.add(client_id)
                        logger.warning(f"Client ID {client_id} already in use")
                    else:
                        logger.warning(f"Connection failed: {e}")
                    await self._cleanup_connection()

                if attempt < self._max_retries - 1:
                    delay = self._retry_delay * (1.5**attempt)
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)

            raise ConnectionError(f"Failed to connect after {self._max_retries} attempts")

        finally:
            self._connecting = False

    async def _verify_connection(self) -> None:
        """Verify that the connection is actually functional by querying accounts."""
        if not self.ib:
            raise ConnectionError("IB client not initialized")

        accounts = self.ib.managedAccounts()
        if not accounts:
            await asyncio.sleep(1)
            accounts = self.ib.managedAccounts()

        if not accounts:
            raise ConnectionError("No managed accounts - connection invalid")

        logger.debug(f"Connection verified - accounts: {accounts}")

    async def disconnect(self) -> None:
        """Gracefully disconnect from IB."""
        await self._cleanup_connection()
        self._connection_count = 0
        logger.info("Disconnected from IB")

    async def fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical bars and return a normalized DataFrame.

        Args:
            symbol: Ticker symbol (e.g., "AAPL")
            duration: IB duration string (e.g., "2 D", "10 D", "30 D")
            bar_size: IB bar size (e.g., "1 min", "5 mins", "30 mins")
            what_to_show: Data type to request (TRADES, MIDPOINT, BID, ASK)
            use_rth: Restrict to regular trading hours

        Returns:
            pandas.DataFrame with lowercased OHLCV columns when present, time-ordered.
        """
        if not self.ib or not self.ib.isConnected():
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

        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize
        data = df.copy()
        data.columns = [str(c).lower() for c in data.columns]
        preferred = [
            c
            for c in ["date", "time", "open", "high", "low", "close", "volume"]
            if c in data.columns
        ]
        if preferred:
            data = data[preferred]
        for c in ["open", "high", "low", "close", "volume"]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors="coerce")
        if "close" in data.columns:
            data = data.dropna(subset=["close"])  # ensure valid rows
        # Sort by time if available
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.sort_index()
        elif "date" in data.columns:
            try:
                data["date"] = pd.to_datetime(data["date"], errors="coerce")
                data = data.sort_values("date")
            except Exception:
                pass
        elif "time" in data.columns:
            try:
                data["time"] = pd.to_datetime(data["time"], errors="coerce")
                data = data.sort_values("time")
            except Exception:
                pass

        return data.reset_index(drop=False)

    def _register_cleanup_handlers(self) -> None:
        """Register handlers for graceful shutdown."""

        def cleanup_sync() -> None:
            """Synchronous cleanup for atexit and signals."""
            if self.ib and self.ib.isConnected():
                logger.info("Cleaning up IB connection...")
                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    loop.run_until_complete(self._cleanup_connection())
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Cleanup error: {e}")

        atexit.register(cleanup_sync)

        def signal_handler(signum, _frame) -> None:  # type: ignore[no-untyped-def]
            logger.info(f"Received signal {signum}, shutting down...")
            cleanup_sync()
            try:
                sys.exit(0)
            except SystemExit:
                pass

        # Register only if available (SIGTERM not always on Windows)
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except Exception:  # noqa: BLE001
            # Best-effort on platforms that restrict signals
            pass

    @asynccontextmanager
    async def connection_context(self):
        """Async context manager that connects on enter and disconnects on exit."""
        try:
            ib = await self.connect()
            yield ib
        finally:
            await self.disconnect()


class IBKRClient:
    """
    IBKR client with robust connection management.

    Designed as a drop-in style client that holds an `IB` instance while in
    context. It sets up error/connection handlers and exposes a few helper
    methods for market data and account queries.
    """

    def __init__(self) -> None:
        self.manager = ConnectionManager()
        self._ib: Optional[IB] = None
        self._subscriptions: Dict[str, Any] = {}

    async def __aenter__(self) -> "IBKRClient":
        self._ib = await self.manager.connect()
        self._setup_handlers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001, D401
        await self.manager.disconnect()

    def _setup_handlers(self) -> None:
        """Set up event handlers on the underlying IB instance."""
        if not self._ib:
            return

        self._ib.errorEvent += self._on_error
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected

    def _on_error(self, reqId, errorCode, errorString, contract):  # type: ignore[no-untyped-def]
        """Handle IB errors dispatched via ib_insync events."""
        if errorCode == 1100:
            logger.error("Lost connection to TWS")
            asyncio.create_task(self._handle_reconnect())
        elif errorCode == 1102:
            logger.info("Connection restored")
        elif errorCode in [2104, 2106, 2158]:
            logger.debug(f"Market data farm: {errorString}")
        else:
            logger.warning(f"IB Error {errorCode}: {errorString}")

    def _on_connected(self) -> None:
        logger.info("Connected to IB")

    def _on_disconnected(self) -> None:
        logger.warning("Disconnected from IB")

    async def _handle_reconnect(self) -> None:
        """Handle automatic reconnection attempts after a brief delay."""
        logger.info("Attempting automatic reconnection...")
        await asyncio.sleep(5)
        try:
            self._ib = await self.manager.connect()
            self._setup_handlers()
            logger.info("Reconnection successful")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Reconnection failed: {e}")

    # Market data methods
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get simple market data snapshot for a symbol."""
        if not self._ib:
            raise ConnectionError("Not connected")

        contract = Stock(symbol, "SMART", "USD")
        ticker = self._ib.reqMktData(contract, "", False, False)

        for _ in range(50):
            await asyncio.sleep(0.1)
            # ib_insync ticker fields may be None until first update
            if (
                getattr(ticker, "last", None) is not None
                and getattr(ticker, "bid", None) is not None
                and getattr(ticker, "ask", None) is not None
            ):
                break

        return {
            "symbol": symbol,
            "last": getattr(ticker, "last", None),
            "bid": getattr(ticker, "bid", None),
            "ask": getattr(ticker, "ask", None),
            "time": time.time(),
        }

    # Account methods
    async def get_positions(self) -> List[Any]:
        """Get current positions."""
        if not self._ib:
            raise ConnectionError("Not connected")
        return self._ib.positions()

    async def get_account_values(self) -> List[Any]:
        """Get account values."""
        if not self._ib:
            raise ConnectionError("Not connected")
        return self._ib.accountValues()

    async def fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "2 D",
        bar_size: str = "5 mins",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Convenience method delegating to the manager for historical data."""
        return await self.manager.fetch_historical_bars(
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            what_to_show=what_to_show,
            use_rth=use_rth,
        )


async def diagnose_connection() -> None:
    """Run connection diagnostics for quick verification."""
    print("\n" + "=" * 60)
    print("IB CONNECTION DIAGNOSTICS")
    print("=" * 60)

    manager = ConnectionManager()

    # Test 1: Basic connection
    print("\n1. Testing basic connection...")
    try:
        ib = await manager.connect()
        print("\u2713 Connected successfully")
        print(f"  Accounts: {ib.managedAccounts()}")
        await manager.disconnect()
    except Exception as e:  # noqa: BLE001
        print(f"\u2717 Connection failed: {e}")

    # Test 2: Rapid reconnection
    print("\n2. Testing rapid reconnection...")
    for i in range(3):
        try:
            _ = await manager.connect()
            print(f"\u2713 Connection {i + 1} successful")
            await manager.disconnect()
            await asyncio.sleep(1)
        except Exception as e:  # noqa: BLE001
            print(f"\u2717 Connection {i + 1} failed: {e}")

    # Test 3: Context manager
    print("\n3. Testing context manager...")
    try:
        async with manager.connection_context() as ib:
            print("\u2713 Context manager works")
            print(f"  Connected: {ib.isConnected()}")
    except Exception as e:  # noqa: BLE001
        print(f"\u2717 Context manager failed: {e}")

    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)


async def _demo_ibkr_client() -> None:
    """Simple demo of IBKRClient usage."""
    print("\n\nTesting IBKRClient...")
    async with IBKRClient() as client:
        try:
            positions = await client.get_positions()
            print(f"Positions: {positions}")
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching positions: {e}")

        try:
            values = await client.get_account_values()
            print(f"Account values: {len(values)} entries")
        except Exception as e:  # noqa: BLE001
            print(f"Error fetching account values: {e}")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def main() -> None:
    """Run connection diagnostics and a brief client demo."""
    _configure_logging()
    await diagnose_connection()
    await _demo_ibkr_client()


if __name__ == "__main__":
    asyncio.run(main())

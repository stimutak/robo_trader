"""
Simplified Async IBKR Client with direct connection approach.

This version removes the complex connection pooling that was causing issues
and uses a simpler, more reliable direct connection approach similar to
the working test_connection.py script.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Try to use ib_insync for connection (which works) and ib_async for operations
try:
    from ib_insync import IB as IB_insync
    from ib_insync import Contract as Contract_insync
    from ib_insync import Stock as Stock_insync
    from ib_insync import util as util_insync
    from ib_insync.util import patchAsyncio as patchAsyncio_insync

    HAS_IB_INSYNC = True
except ImportError:
    HAS_IB_INSYNC = False

try:
    from ib_async import IB as IB_async
    from ib_async import Contract as Contract_async
    from ib_async import Stock as Stock_async
    from ib_async import util as util_async
    from ib_async.util import patchAsyncio as patchAsyncio_async

    HAS_IB_ASYNC = True
except ImportError:
    HAS_IB_ASYNC = False

# Use the library that works for connections
if HAS_IB_INSYNC:
    IB = IB_insync
    Contract = Contract_insync
    Stock = Stock_insync
    util = util_insync
    patchAsyncio = patchAsyncio_insync
    USING_LIBRARY = "ib_insync"
elif HAS_IB_ASYNC:
    IB = IB_async
    Contract = Contract_async
    Stock = Stock_async
    util = util_async
    patchAsyncio = patchAsyncio_async
    USING_LIBRARY = "ib_async"
else:
    raise ImportError("Neither ib_insync nor ib_async is available")

# Enable nested event loops
patchAsyncio()

logger = logging.getLogger(__name__)
logger.info(f"Using {USING_LIBRARY} for IBKR connections")


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


def _detect_port():
    """Auto-detect whether TWS or IB Gateway is running."""
    import socket

    # Check ports in order of preference
    ports_to_check = [
        (7497, "TWS (Paper)"),
        (4002, "IB Gateway (Paper)"),
        (7496, "TWS (Live)"),
        (4001, "IB Gateway (Live)"),
    ]

    for port, name in ports_to_check:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            result = sock.connect_ex(("127.0.0.1", port))
            if result == 0:
                logger.info(f"Auto-detected {name} on port {port}")
                return port
        except Exception:
            pass
        finally:
            sock.close()

    # Default to TWS paper trading port
    logger.warning("No IBKR service detected, defaulting to TWS paper trading port 7497")
    return 7497


async def _create_direct_connection(
    host: str, port: int, client_id: int, readonly: bool = True, timeout: float = 10.0
) -> IB:
    """Create a direct IBKR connection by running in a separate process without async patches."""
    max_retries = 3
    base_delay = 1.0

    for attempt in range(max_retries):
        try:
            current_client_id = client_id + attempt
            logger.info(
                f"Attempt {attempt + 1}/{max_retries}: Connecting to {host}:{port} with client ID {current_client_id}"
            )

            # Create a simple connection script that doesn't use patchAsyncio
            script_content = f"""
import sys
import json
from ib_insync import IB
# Don't call patchAsyncio() - run in clean environment

def test_connection():
    try:
        ib = IB()
        ib.connect("{host}", {port}, clientId={current_client_id}, timeout={min(timeout, 15.0)}, readonly={readonly})

        # Get basic info to validate connection
        server_version = ib.client.serverVersion()
        accounts = ib.managedAccounts()

        # Disconnect cleanly
        ib.disconnect()

        return {{
            "success": True,
            "server_version": server_version,
            "accounts": accounts,
            "client_id": {current_client_id}
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "client_id": {current_client_id}
        }}

if __name__ == "__main__":
    result = test_connection()
    print(json.dumps(result))
"""

            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Run the connection test in subprocess
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=min(timeout, 15.0) + 5,  # Add buffer
                )

                if result.returncode == 0:
                    # Parse result
                    try:
                        output = json.loads(result.stdout.strip())
                        if output["success"]:
                            logger.info(
                                f"✓ Subprocess validation successful with client ID {current_client_id}"
                            )
                            logger.info(f"Server version: {output['server_version']}")

                            # The subprocess confirmed the connection works
                            # Now try to create the connection in this process using the same client ID
                            # Since we know it works, we can be more aggressive with timeout
                            ib = IB()

                            # Try the connection - if it fails due to async issues, we'll handle it
                            try:
                                ib.connect(
                                    host=host,
                                    port=port,
                                    clientId=current_client_id,
                                    timeout=min(timeout, 15.0),
                                    readonly=readonly,
                                )
                                logger.info(
                                    f"✓ Main process connection successful with client ID {current_client_id}"
                                )
                                return ib
                            except Exception as main_e:
                                logger.warning(
                                    f"Main process connection failed even though subprocess worked: {main_e}"
                                )
                                # If main process fails but subprocess worked, there's an async context issue
                                # For now, raise the error - we may need a different approach
                                raise RuntimeError(f"Async context prevents connection: {main_e}")
                        else:
                            raise RuntimeError(f"Subprocess connection failed: {output['error']}")
                    except json.JSONDecodeError as je:
                        raise RuntimeError(
                            f"Failed to parse subprocess output: {result.stdout}, error: {je}"
                        )
                else:
                    raise RuntimeError(
                        f"Subprocess failed with return code {result.returncode}: {result.stderr}"
                    )

            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except Exception as e:
            logger.warning(
                f"Connection attempt {attempt + 1} failed with client ID {client_id + attempt}: {e}"
            )

            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} connection attempts failed")
                raise

            # Wait before retrying
            delay = base_delay * (2**attempt)
            logger.info(f"Waiting {delay:.1f}s before retry...")
            await asyncio.sleep(delay)

            logger.info(f"✓ Successfully connected with client ID {current_client_id}")
            logger.info(f"Server version: {ib.client.serverVersion()}")
            return ib

        except Exception as e:
            logger.warning(
                f"Connection attempt {attempt + 1} failed with client ID {client_id + attempt}: {e}"
            )

            # Clean up failed connection
            try:
                if ib and hasattr(ib, "disconnect"):
                    ib.disconnect()
                    await asyncio.sleep(0.5)  # Give time for cleanup
            except Exception:
                pass

            # If this was the last attempt, raise the error
            if attempt == max_retries - 1:
                logger.error(f"All {max_retries} connection attempts failed")
                raise

            # Wait before retrying with exponential backoff
            delay = base_delay * (2**attempt)
            logger.info(f"Waiting {delay:.1f}s before retry...")
            await asyncio.sleep(delay)


class AsyncIBKRClient:
    """
    Async IBKR client using synchronous wrapper to avoid async context issues.

    This version uses a separate thread for all IBKR operations to completely
    avoid the patchAsyncio() conflicts that cause connection timeouts.
    """

    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig()
        self._wrapper: Optional[Any] = None
        self._contract_cache: Dict[str, Contract] = {}
        self._rate_limiter = asyncio.Semaphore(50)  # Max 50 concurrent requests

    async def connect(self):
        """Create a connection using the sync wrapper."""
        if self._wrapper:
            logger.info("Already have wrapper instance")
            return

        # Import here to avoid circular imports
        from .sync_ibkr_wrapper import SyncIBKRWrapper

        # Auto-detect port if needed
        port = _detect_port() if self.config.port == 7497 else self.config.port

        # Create wrapper
        self._wrapper = SyncIBKRWrapper(
            host=self.config.host, port=port, readonly=self.config.readonly
        )

        # Connect using wrapper
        result = await self._wrapper.connect()
        if not result["success"]:
            raise RuntimeError(f"Connection failed: {result.get('error', 'Unknown error')}")

        logger.info("✓ IBKR connection established successfully")
        logger.info(f"✓ Connected with client ID {result.get('client_id')}")
        logger.info(f"Server version: {result.get('server_version')}")

    async def disconnect(self):
        """Close the connection."""
        if self._wrapper:
            result = await self._wrapper.disconnect()
            if result["success"]:
                logger.info("Disconnected from IBKR")
            else:
                logger.warning(f"Disconnect warning: {result.get('error')}")
        self._wrapper = None

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

        if not self._connection or not self._connection.isConnected():
            raise RuntimeError("Not connected to IBKR")

        async with self._rate_limiter:
            await asyncio.sleep(0.1)  # Rate limiting

        contract = Stock(symbol, exchange, currency)

        # Use async version of qualifyContracts
        qualified = await asyncio.wait_for(
            self._connection.qualifyContractsAsync(contract), timeout=5.0
        )

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
        Fetch recent bars using the sync wrapper.

        Args:
            symbol: Stock symbol
            duration: IB duration string (e.g., "2 D", "30 D")
            bar_size: IB bar size (e.g., "1 min", "5 mins", "1 hour")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)
            use_rth: Use regular trading hours only

        Returns:
            DataFrame with normalized OHLCV data
        """
        if not self._wrapper:
            raise RuntimeError("Not connected to IBKR")

        async with self._rate_limiter:
            # Use the wrapper to get historical data
            result = await self._wrapper.get_historical_data(symbol, duration, bar_size)

            if not result["success"]:
                logger.warning(f"No data returned for {symbol}: {result.get('error')}")
                return pd.DataFrame()

            # Convert the data back to DataFrame
            data = result["data"]
            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(data)
            return normalize_bars_df(df)

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
        if not self._connection or not self._connection.isConnected():
            raise RuntimeError("Not connected to IBKR")

        summary = await self._connection.accountSummaryAsync()
        return {item.tag: item.value for item in summary}

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        if not self._connection or not self._connection.isConnected():
            raise RuntimeError("Not connected to IBKR")

        positions = await self._connection.reqPositionsAsync()
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

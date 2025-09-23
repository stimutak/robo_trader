"""
Synchronous IBKR wrapper that can be called from async code.

This completely avoids the patchAsyncio() issues by running IBKR operations
in a separate thread with a clean synchronous environment.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from ib_async import IB, Contract, Stock, util

logger = logging.getLogger(__name__)


class SyncIBKRWrapper:
    """Synchronous IBKR wrapper that runs in a separate thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, readonly: bool = True):
        self.host = host
        self.port = port
        self.readonly = readonly
        self.ib: Optional[IB] = None
        self.client_id: Optional[int] = None
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _generate_client_id(self) -> int:
        """Generate unique client ID."""
        timestamp = int(time.time() * 1000) % 10000
        pid = threading.get_ident() % 1000
        return 20000 + (timestamp % 1000) * 100 + (pid % 100)

    def _sync_connect(self) -> Dict[str, Any]:
        """Synchronous connection method using subprocess to avoid patchAsyncio."""
        try:
            if self.ib and self.ib.isConnected():
                return {"success": True, "message": "Already connected"}

            self.client_id = self._generate_client_id()

            logger.info(f"Connecting to {self.host}:{self.port} with client ID {self.client_id}")

            # Create a subprocess script that doesn't use patchAsyncio
            script_content = f"""
import sys
import json
from ib_async import IB
# Don't call patchAsyncio() - run in clean environment

def test_and_keep_connection():
    try:
        ib = IB()
        ib.connect("{self.host}", {self.port}, clientId={self.client_id}, timeout=15, readonly={self.readonly})

        # Get basic info to validate connection
        server_version = ib.client.serverVersion()
        accounts = ib.managedAccounts()

        # Keep connection alive and return success
        return {{
            "success": True,
            "server_version": server_version,
            "accounts": accounts,
            "client_id": {self.client_id}
        }}
    except Exception as e:
        return {{
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "client_id": {self.client_id}
        }}

if __name__ == "__main__":
    result = test_and_keep_connection()
    print(json.dumps(result))
"""

            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Run the connection test in subprocess
                result = subprocess.run(
                    [sys.executable, script_path], capture_output=True, text=True, timeout=20
                )

                if result.returncode == 0:
                    # Parse result
                    try:
                        output = json.loads(result.stdout.strip())
                        if output["success"]:
                            logger.info(
                                f"✓ Subprocess validation successful with client ID {self.client_id}"
                            )

                            # Now create the actual connection in this thread
                            # Since subprocess confirmed it works, try with same client ID
                            self.ib = IB()
                            self.ib.connect(
                                host=self.host,
                                port=self.port,
                                clientId=self.client_id,
                                timeout=15,
                                readonly=self.readonly,
                            )

                            server_version = self.ib.client.serverVersion()
                            accounts = self.ib.managedAccounts()

                            logger.info(f"✓ Connected successfully with client ID {self.client_id}")
                            logger.info(f"Server version: {server_version}")

                            return {
                                "success": True,
                                "server_version": server_version,
                                "accounts": accounts,
                                "client_id": self.client_id,
                            }
                        else:
                            return {"success": False, "error": output.get("error", "Unknown error")}
                    except json.JSONDecodeError as je:
                        return {"success": False, "error": f"JSON decode error: {je}"}
                else:
                    return {"success": False, "error": f"Subprocess failed: {result.stderr}"}

            finally:
                # Clean up temp file
                try:
                    os.unlink(script_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            if self.ib:
                try:
                    self.ib.disconnect()
                except Exception:
                    pass
                self.ib = None
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def _sync_disconnect(self) -> Dict[str, Any]:
        """Synchronous disconnect method."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IBKR")
            self.ib = None
            self.client_id = None
            return {"success": True}
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return {"success": False, "error": str(e)}

    def _sync_get_historical_data(
        self, symbol: str, duration: str = "2 D", bar_size: str = "5 mins"
    ) -> Dict[str, Any]:
        """Get historical data synchronously."""
        try:
            if not self.ib or not self.ib.isConnected():
                return {"success": False, "error": "Not connected"}

            # Create and qualify contract
            contract = Stock(symbol, "SMART", "USD")
            qualified = self.ib.qualifyContracts(contract)

            if not qualified:
                return {"success": False, "error": f"Could not qualify contract for {symbol}"}

            # Get historical data
            bars = self.ib.reqHistoricalData(
                qualified[0],
                endDateTime="",
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                return {"success": False, "error": f"No data returned for {symbol}"}

            # Convert to DataFrame
            df = util.df(bars)

            # Normalize the DataFrame
            if not df.empty:
                df.columns = [str(c).lower() for c in df.columns]
                # Convert to JSON-serializable format
                data = df.to_dict("records")
            else:
                data = []

            return {"success": True, "symbol": symbol, "data": data, "rows": len(data)}

        except Exception as e:
            logger.error(f"Historical data request failed for {symbol}: {e}")
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    async def connect(self) -> Dict[str, Any]:
        """Async wrapper for connection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_connect)

    async def disconnect(self) -> Dict[str, Any]:
        """Async wrapper for disconnection."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._sync_disconnect)

    async def get_historical_data(
        self, symbol: str, duration: str = "2 D", bar_size: str = "5 mins"
    ) -> Dict[str, Any]:
        """Async wrapper for historical data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._sync_get_historical_data, symbol, duration, bar_size
        )

    def __del__(self):
        """Cleanup on destruction."""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
        except Exception:
            pass

        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

"""
DEPRECATED: Use `robo_trader.connection_manager.ConnectionManager` or
`robo_trader.connection_manager.IBKRClient` instead.

This module remains for backward compatibility only and will be removed in a
future release. Importing will emit a deprecation warning and stub methods will
raise clear exceptions when used.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Dict, List, Optional

import pandas as pd

warnings.warn(
    "robo_trader.clients.async_ibkr_client is deprecated; use robo_trader.connection_manager",
    DeprecationWarning,
    stacklevel=2,
)


class ConnectionConfig:  # type: ignore[too-many-instance-attributes]
    """Deprecated placeholder; kept for import compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        warnings.warn(
            "ConnectionConfig is deprecated; configure via environment and use ConnectionManager",
            DeprecationWarning,
            stacklevel=2,
        )


def _detect_port() -> int:
    warnings.warn(
        "_detect_port is deprecated; honor IBKR_PORT and let ConnectionManager verify",
        DeprecationWarning,
        stacklevel=2,
    )
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
    """Deprecated async client placeholder."""

    def __init__(self, config: Optional[ConnectionConfig] = None):  # noqa: D401
        warnings.warn(
            "AsyncIBKRClient is deprecated; use ConnectionManager/IBKRClient",
            DeprecationWarning,
            stacklevel=2,
        )

    async def connect(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        raise RuntimeError("Deprecated: use ConnectionManager.connect() instead")

    async def disconnect(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
    )
    async def qualify_stock(self, *args: Any, **kwargs: Any):  # noqa: D401
        raise RuntimeError("Deprecated: use ib_insync directly via ConnectionManager")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type((RuntimeError, asyncio.TimeoutError)),
    )
    async def fetch_recent_bars(self, *args: Any, **kwargs: Any) -> pd.DataFrame:  # noqa: D401
        raise RuntimeError("Deprecated: use ConnectionManager.fetch_historical_bars()")

    async def fetch_multiple_symbols(
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, pd.DataFrame]:  # noqa: D401
        raise RuntimeError("Deprecated: iterate ConnectionManager.fetch_historical_bars per symbol")

    async def get_account_summary(self) -> Dict[str, Any]:  # noqa: D401
        raise RuntimeError("Deprecated: use ib.accountSummary() via ConnectionManager")

    async def get_positions(self) -> List[Dict[str, Any]]:  # noqa: D401
        raise RuntimeError("Deprecated: use ib.positions() via ConnectionManager")

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
async def create_client(config: Optional[ConnectionConfig] = None) -> AsyncIBKRClient:  # noqa: D401
    raise RuntimeError("Deprecated: use ConnectionManager() or IBKRClient()")

#!/usr/bin/env python3
"""
IBKR Subprocess Worker

Runs in a separate process to isolate ib_async from the main trading system's
complex async environment. Communicates via JSON over stdin/stdout.

This solves the ib_async library incompatibility with complex async environments
where API handshakes timeout despite successful TCP connections.
"""
import asyncio
import atexit
import json
import os
import queue
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import Optional

# CRITICAL: Enable real disconnect BEFORE importing ib_async or ibkr_safe
# This prevents zombie connections when the worker process exits
os.environ["IBKR_FORCE_DISCONNECT"] = "1"

from ib_async import IB  # noqa: E402

from robo_trader.utils.ibkr_safe import safe_disconnect  # noqa: E402

# Global IB instance
ib: Optional[IB] = None

# Global shutdown flag
shutdown_requested = False

# Tracks Gateway API failure state to avoid hammering a dead Gateway
gateway_api_down = False
gateway_failure_detail = ""

# CRITICAL FIX: Use a dedicated thread for stdin reading to avoid
# run_in_executor race condition where orphaned threads consume data
stdin_queue: queue.Queue = queue.Queue()
stdin_reader_thread: Optional[threading.Thread] = None


def _stdin_reader():
    """Dedicated thread to read stdin lines and put them in queue.

    This avoids the race condition with run_in_executor where:
    1. run_in_executor submits readline() to thread pool
    2. asyncio timeout cancels the future after 1s
    3. BUT the thread pool thread continues blocking on readline()
    4. Next iteration submits ANOTHER readline()
    5. When data arrives, the orphaned thread consumes it
    6. Result is never returned because its future was cancelled

    Using a dedicated thread ensures exactly one readline() is active.
    """
    while not shutdown_requested:
        try:
            line = sys.stdin.readline()
            if not line:  # EOF
                stdin_queue.put(None)
                break
            stdin_queue.put(line)
        except Exception as e:
            print(f"DEBUG: stdin reader error: {e}", file=sys.stderr, flush=True)
            stdin_queue.put(None)
            break
    print("DEBUG: stdin reader thread exiting", file=sys.stderr, flush=True)


def _cleanup_on_exit():
    """Atexit handler to ensure we disconnect from IBKR to prevent zombies"""
    global ib
    if ib is not None:
        print("atexit: Disconnecting from IBKR...", file=sys.stderr, flush=True)
        try:
            safe_disconnect(ib)
            print("atexit: Disconnected successfully", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"atexit: Disconnect error: {e}", file=sys.stderr, flush=True)
        ib = None


# Register atexit handler as safety net
atexit.register(_cleanup_on_exit)


def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    print(
        f"Received signal {signal_name} ({signum}), initiating graceful shutdown...",
        file=sys.stderr,
        flush=True,
    )
    shutdown_requested = True


# Register signal handlers at module level
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def handle_connect(params: dict) -> dict:
    """Handle connect command with proper async handling.

    CRITICAL FIX (2025-11-27): Removed blocking waitOnUpdate() call which was
    freezing the async event loop. Now uses proper async patterns:
    1. connectAsync() with explicit timeout
    2. serverVersion() check to verify API handshake completion
    3. Async polling with asyncio.sleep() for account data
    4. Event-based waiting using ib_async's internal event loop
    """
    global ib, gateway_api_down, gateway_failure_detail

    try:
        if gateway_api_down:
            return {
                "status": "error",
                "error": "Gateway API layer is unresponsive. Manual restart required.",
                "error_type": "GatewayRequiresRestartError",
                "requires_restart": True,
                "detail": gateway_failure_detail,
            }

        # Create new IB instance
        ib = IB()

        # Extract parameters
        host = params.get("host", "127.0.0.1")
        port = params.get("port", 4002)
        client_id = params.get("client_id", 1)
        readonly = params.get("readonly", True)
        timeout = params.get("timeout", 30.0)

        print(
            f"DEBUG: Connecting to {host}:{port} client_id={client_id} timeout={timeout}",
            file=sys.stderr,
            flush=True,
        )

        # Track connection timing separately from handshake verification
        connect_start = time.time()

        # Connect to IBKR using native async
        # The timeout here only applies to the initial TCP connection
        await ib.connectAsync(
            host=host,
            port=port,
            clientId=client_id,
            readonly=readonly,
            timeout=timeout,
        )

        print(
            f"DEBUG: connectAsync() returned after {time.time() - connect_start:.2f}s",
            file=sys.stderr,
            flush=True,
        )

        # CRITICAL FIX #1: Verify API handshake completed by checking serverVersion
        # The serverVersion is only available after the API protocol handshake succeeds
        # This is more reliable than isConnected() which only checks TCP state
        # NOTE: This timeout must match WORKER_HANDSHAKE_TIMEOUT in subprocess_ibkr_client.py
        max_handshake_wait = 15.0  # seconds for full API handshake
        handshake_poll_interval = 0.25  # 250ms polling - balanced for CPU efficiency

        handshake_start = time.time()
        server_version = None
        while time.time() - handshake_start < max_handshake_wait:
            # Check if connected at TCP level first
            if not ib.isConnected():
                await asyncio.sleep(handshake_poll_interval)
                continue

            # Now check if API handshake is complete by checking serverVersion
            try:
                server_version = ib.client.serverVersion()
                if server_version and server_version > 0:
                    print(
                        f"DEBUG: API handshake complete! serverVersion={server_version} "
                        f"after {time.time() - handshake_start:.2f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
            except AttributeError:
                # client.serverVersion() not available yet - handshake incomplete
                pass
            except (ConnectionError, OSError) as e:
                # Connection-related errors during handshake
                print(
                    f"DEBUG: serverVersion() connection error: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as e:
                # Unexpected error - log but continue polling
                print(
                    f"DEBUG: serverVersion() unexpected error ({type(e).__name__}): {e}",
                    file=sys.stderr,
                    flush=True,
                )

            # CRITICAL FIX #2: Use async sleep, NOT blocking waitOnUpdate()
            # waitOnUpdate() is synchronous and blocks the event loop, which
            # prevents ib_async from processing incoming Gateway messages
            await asyncio.sleep(handshake_poll_interval)

        if not server_version or server_version <= 0:
            elapsed = time.time() - handshake_start
            raise TimeoutError(
                f"API handshake timeout after {elapsed:.1f}s. "
                f"TCP connected but Gateway did not complete protocol negotiation. "
                f"isConnected={ib.isConnected()}"
            )

        # Small stabilization delay after handshake (reduced from 2.0s)
        # This gives Gateway time to send initial account data
        # NOTE: This delay must match WORKER_STABILIZATION_DELAY in subprocess_ibkr_client.py
        await asyncio.sleep(0.5)

        # CRITICAL FIX #3: Wait for account data with pure async polling
        # Do NOT use waitOnUpdate() as it's blocking
        print(
            f"DEBUG: Waiting for account data to arrive...",
            file=sys.stderr,
            flush=True,
        )

        accounts = []
        account_wait_start = time.time()
        # NOTE: This timeout must match WORKER_ACCOUNT_TIMEOUT in subprocess_ibkr_client.py
        max_account_wait = 10.0  # seconds
        account_poll_interval = 0.3  # 300ms polling - balanced for CPU efficiency

        while time.time() - account_wait_start < max_account_wait:
            # Check for managed accounts
            accounts = ib.managedAccounts()
            if accounts:
                print(
                    f"DEBUG: Received accounts after {time.time() - account_wait_start:.2f}s: {accounts}",
                    file=sys.stderr,
                    flush=True,
                )
                break

            # Pure async sleep - let ib_async's internal event loop process messages
            # This is critical: ib_async processes incoming data in the background
            # when we yield control with await asyncio.sleep()
            await asyncio.sleep(account_poll_interval)

        if not accounts:
            elapsed = time.time() - handshake_start
            raise ConnectionError(
                f"No managed accounts received after {elapsed:.1f}s total wait. "
                f"API handshake succeeded (serverVersion={server_version}) but "
                f"Gateway did not send account data. Check Gateway API permissions."
            )

        # Connection fully established
        gateway_api_down = False
        gateway_failure_detail = ""

        total_time = time.time() - handshake_start
        print(
            f"DEBUG: Connection fully established in {total_time:.2f}s "
            f"(serverVersion={server_version}, accounts={accounts})",
            file=sys.stderr,
            flush=True,
        )

        return {
            "status": "success",
            "data": {
                "connected": True,
                "accounts": accounts,
                "client_id": client_id,
                "server_version": server_version,
            },
        }

    except Exception as e:
        # Clean up on error
        # NOTE: Do NOT call ib.disconnect() here! It crashes IBKR Gateway's API layer.
        # Gateway has a bug where disconnect() during/after a failed connection
        # causes the API client to go RED. Let Python's cleanup handle it naturally.
        ib = None

        error_text = str(e).lower()
        if isinstance(e, TimeoutError) or "timeout" in error_text:
            gateway_api_down = True
            gateway_failure_detail = (
                "Handshake timed out at "
                f"{datetime.utcnow().isoformat()}Z. Restart IB Gateway before retrying."
            )

        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "requires_restart": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
        }


async def handle_get_accounts() -> dict:
    """Handle get_accounts command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        accounts = ib.managedAccounts()

        return {"status": "success", "data": {"accounts": accounts}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_get_positions() -> dict:
    """Handle get_positions command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        positions = ib.positions()

        # Convert Position objects to dicts
        positions_data = []
        for pos in positions:
            positions_data.append(
                {
                    "account": pos.account,
                    "contract": {
                        "symbol": pos.contract.symbol,
                        "secType": pos.contract.secType,
                        "exchange": pos.contract.exchange,
                        "currency": pos.contract.currency,
                    },
                    "position": float(pos.position),
                    "avgCost": float(pos.avgCost),
                }
            )

        return {"status": "success", "data": {"positions": positions_data}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_get_account_summary() -> dict:
    """Handle get_account_summary command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        # Get account values
        account_values = ib.accountValues()

        # Convert to dict
        summary = {}
        for av in account_values:
            key = f"{av.tag}_{av.currency}" if av.currency else av.tag
            summary[key] = av.value

        return {"status": "success", "data": {"summary": summary}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_disconnect() -> dict:
    """Handle disconnect command"""
    global ib

    try:
        if ib:
            # Properly disconnect to avoid zombie connections
            print("Disconnecting from IBKR...", file=sys.stderr, flush=True)
            safe_disconnect(ib)
            ib = None

        return {"status": "success", "data": {"disconnected": True}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_ping() -> dict:
    """Handle ping command (health check) - also triggers IBKR keep-alive"""
    global ib
    connected = ib is not None and ib.isConnected()

    # If we have an IB instance but it's disconnected, try to keep it alive
    # by running a simple async loop iteration - this helps maintain connection
    if ib is not None:
        try:
            # Running sleep(0) through ib_async's event loop helps keep connection alive
            await asyncio.sleep(0)
            # Check connection status after the async yield
            connected = ib.isConnected()
            if connected:
                print("DEBUG: Ping keep-alive check passed", file=sys.stderr, flush=True)
            else:
                print(
                    "DEBUG: IBKR connection lost, will need reconnection",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as e:
            print(f"DEBUG: Keep-alive check failed: {e}", file=sys.stderr, flush=True)
            connected = False

    return {
        "status": "success",
        "data": {
            "pong": True,
            "connected": connected,
            "gateway_api_down": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
        },
    }


async def handle_health() -> dict:
    """Provide extended health detail for diagnostics."""
    connected = ib is not None and ib.isConnected()

    return {
        "status": "success",
        "data": {
            "connected": connected,
            "gateway_api_down": gateway_api_down,
            "detail": gateway_failure_detail if gateway_api_down else "",
        },
    }


async def handle_get_historical_bars(params: dict) -> dict:
    """Handle get_historical_bars command"""
    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        # Extract parameters
        symbol = params.get("symbol")
        duration = params.get("duration", "2 D")
        bar_size = params.get("bar_size", "5 mins")
        what_to_show = params.get("what_to_show", "TRADES")
        use_rth = params.get("use_rth", True)

        if not symbol:
            raise ValueError("symbol parameter is required")

        # Create contract
        from ib_async import Stock

        contract = Stock(symbol, "SMART", "USD")

        # Qualify contract (must await - it's a coroutine in ib_async)
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract for {symbol}")

        # Request historical data (must await - it's a coroutine in ib_async)
        bars = await ib.reqHistoricalDataAsync(
            qualified[0],
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )

        # Convert bars to dict format
        bars_data = []
        for bar in bars:
            bars_data.append(
                {
                    "date": (
                        bar.date.isoformat() if hasattr(bar.date, "isoformat") else str(bar.date)
                    ),
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "average": float(bar.average) if hasattr(bar, "average") else 0.0,
                    "barCount": int(bar.barCount) if hasattr(bar, "barCount") else 0,
                }
            )

        return {"status": "success", "data": {"bars": bars_data}}

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


async def handle_command(command: dict) -> dict:
    """Route command to appropriate handler"""
    cmd = command.get("command")

    # DEBUG: Log received command
    print(f"DEBUG: Received command: {cmd}", file=sys.stderr, flush=True)
    if cmd == "connect":
        params = command.get("params", {})
        print(f"DEBUG: Extracted params: {params}", file=sys.stderr, flush=True)
        return await handle_connect(params)
    elif cmd == "get_accounts":
        return await handle_get_accounts()
    elif cmd == "get_positions":
        return await handle_get_positions()
    elif cmd == "get_account_summary":
        return await handle_get_account_summary()
    elif cmd == "get_historical_bars":
        return await handle_get_historical_bars(command.get("params", {}))
    elif cmd == "disconnect":
        return await handle_disconnect()
    elif cmd == "ping":
        return await handle_ping()
    elif cmd == "health":
        return await handle_health()
    else:
        return {
            "status": "error",
            "error": f"Unknown command: {cmd}",
            "error_type": "UnknownCommandError",
        }


async def main():
    """Main loop - read commands from stdin, write responses to stdout"""
    global stdin_reader_thread

    # Start dedicated stdin reader thread
    stdin_reader_thread = threading.Thread(target=_stdin_reader, daemon=True, name="StdinReader")
    stdin_reader_thread.start()
    print("DEBUG: Started stdin reader thread", file=sys.stderr, flush=True)

    try:
        while not shutdown_requested:
            # Read command from queue (populated by dedicated reader thread)
            # This avoids the run_in_executor race condition
            try:
                # Use run_in_executor to wait on queue without blocking event loop
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: stdin_queue.get(timeout=1.0)
                    ),
                    timeout=2.0,  # Slightly longer than queue timeout
                )
            except (asyncio.TimeoutError, queue.Empty):
                continue  # Check shutdown flag and loop again

            # EOF or shutdown - exit gracefully
            if line is None or shutdown_requested:
                break

            # Parse command
            try:
                command = json.loads(line.strip())
            except json.JSONDecodeError as e:
                response = {
                    "status": "error",
                    "error": f"Invalid JSON: {e}",
                    "error_type": "JSONDecodeError",
                }
                print(json.dumps(response), flush=True)
                continue

            # Log receipt of command for debugging pipe issues
            cmd = command.get("command", "unknown")
            print(f"DEBUG: Processing command: {cmd}", file=sys.stderr, flush=True)

            # Handle command
            response = await handle_command(command)

            # Write response to stdout
            print(json.dumps(response), flush=True)

    except KeyboardInterrupt:
        print("Received KeyboardInterrupt, shutting down...", file=sys.stderr, flush=True)
    except Exception as e:
        error_response = {
            "status": "error",
            "error": f"Fatal error: {e}",
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_response), flush=True)
    finally:
        # Cleanup on exit - MUST disconnect to avoid zombies
        print("Worker shutting down gracefully...", file=sys.stderr, flush=True)
        if ib is not None:
            print("Disconnecting from IBKR to prevent zombie...", file=sys.stderr, flush=True)
            try:
                safe_disconnect(ib)
                print("Disconnected successfully", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"Disconnect error (non-fatal): {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
IBKR Subprocess Worker

Runs in a separate process to isolate ib_async from the main trading system's
complex async environment. Communicates via JSON over stdin/stdout.

This solves the ib_async library incompatibility with complex async environments
where API handshakes timeout despite successful TCP connections.
"""
import asyncio
import json
import signal
import sys
import traceback
from datetime import datetime
from typing import Optional

from ib_async import IB

from robo_trader.utils import ibkr_safe as _ibkr_safe

# Global IB instance
ib: Optional[IB] = None

# Global shutdown flag
shutdown_requested = False

# Tracks Gateway API failure state to avoid hammering a dead Gateway
gateway_api_down = False
gateway_failure_detail = ""


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
    """Handle connect command"""
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

        # DEBUG: Log to stderr
        print(
            f"DEBUG: Connecting to {host}:{port} client_id={client_id} timeout={timeout}",
            file=sys.stderr,
            flush=True,
        )

        # Connect to IBKR synchronously in executor to avoid async patch issues
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: ib.connect(
                host=host,
                port=port,
                clientId=client_id,
                readonly=readonly,
                timeout=timeout,
            ),
        )

        print(f"DEBUG: Connected successfully!", file=sys.stderr, flush=True)

        # Verify connection and get accounts
        if not ib.isConnected():
            raise ConnectionError("Connection failed - not connected")

        accounts = ib.managedAccounts()

        # Retry once if no accounts
        if not accounts:
            await asyncio.sleep(1)
            accounts = ib.managedAccounts()

        if not accounts:
            raise ConnectionError("No managed accounts found")

        gateway_api_down = False
        gateway_failure_detail = ""

        return {
            "status": "success",
            "data": {"connected": True, "accounts": accounts, "client_id": client_id},
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
            # Do NOT call ib.disconnect() - let process exit naturally to avoid Gateway crash
            ib = None

        return {"status": "success", "data": {"disconnected": True}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_ping() -> dict:
    """Handle ping command (health check)"""
    connected = ib is not None and ib.isConnected()

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

        # Qualify contract
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            raise ValueError(f"Could not qualify contract for {symbol}")

        # Request historical data
        bars = ib.reqHistoricalData(
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
    try:
        while not shutdown_requested:
            # Read command from stdin (blocking)
            try:
                line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
                    timeout=1.0,  # Check shutdown flag every second
                )
            except asyncio.TimeoutError:
                continue  # Check shutdown flag and loop again

            # EOF - exit gracefully
            if not line or shutdown_requested:
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
        # Cleanup on exit
        print("Worker shutting down gracefully...", file=sys.stderr, flush=True)
        # NOTE: Do NOT call ib.disconnect() here! It crashes IBKR Gateway's API layer.
        # When the process exits, Python will clean up connections naturally without
        # triggering Gateway's disconnect bug. Explicit disconnect() calls cause Gateway
        # API client to go RED.
        pass


if __name__ == "__main__":
    asyncio.run(main())

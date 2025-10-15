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
import sys
import traceback
from typing import Optional

from ib_async import IB

# Global IB instance
ib: Optional[IB] = None


async def handle_connect(params: dict) -> dict:
    """Handle connect command"""
    global ib

    try:
        # Create new IB instance
        ib = IB()

        # Extract parameters
        host = params.get("host", "127.0.0.1")
        port = params.get("port", 4002)
        client_id = params.get("client_id", 1)
        readonly = params.get("readonly", True)
        timeout = params.get("timeout", 15.0)

        # Connect to IBKR
        await ib.connectAsync(
            host=host, port=port, clientId=client_id, readonly=readonly, timeout=timeout
        )

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

        return {
            "status": "success",
            "data": {"connected": True, "accounts": accounts, "client_id": client_id},
        }

    except Exception as e:
        # Clean up on error
        if ib:
            try:
                ib.disconnect()
            except Exception:
                pass
        ib = None

        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


async def handle_get_accounts() -> dict:
    """Handle get_accounts command"""
    global ib

    try:
        if not ib or not ib.isConnected():
            raise ConnectionError("Not connected to IBKR")

        accounts = ib.managedAccounts()

        return {"status": "success", "data": {"accounts": accounts}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_get_positions() -> dict:
    """Handle get_positions command"""
    global ib

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
    global ib

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
            ib.disconnect()
            ib = None

        return {"status": "success", "data": {"disconnected": True}}

    except Exception as e:
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}


async def handle_ping() -> dict:
    """Handle ping command (health check)"""
    global ib

    connected = ib is not None and ib.isConnected()

    return {"status": "success", "data": {"pong": True, "connected": connected}}


async def handle_command(command: dict) -> dict:
    """Route command to appropriate handler"""
    cmd = command.get("command")

    if cmd == "connect":
        return await handle_connect(command)
    elif cmd == "get_accounts":
        return await handle_get_accounts()
    elif cmd == "get_positions":
        return await handle_get_positions()
    elif cmd == "get_account_summary":
        return await handle_get_account_summary()
    elif cmd == "disconnect":
        return await handle_disconnect()
    elif cmd == "ping":
        return await handle_ping()
    else:
        return {
            "status": "error",
            "error": f"Unknown command: {cmd}",
            "error_type": "UnknownCommandError",
        }


async def main():
    """Main loop - read commands from stdin, write responses to stdout"""
    global ib

    try:
        while True:
            # Read command from stdin (blocking)
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)

            # EOF - exit gracefully
            if not line:
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
        pass
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
        if ib:
            try:
                ib.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())

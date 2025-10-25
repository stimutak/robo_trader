#!/usr/bin/env python3
"""
Minimal test that mimics runner_async environment
"""
import asyncio
import os

# Import ib_async at top level like runner_async does
from ib_async import Stock

from robo_trader.utils.robust_connection import connect_ibkr_robust

# Import websocket client like runner_async does
from robo_trader.websocket_client import ws_client


async def test_like_runner():
    """Test connection in environment similar to runner_async"""

    print("=" * 60)
    print("Testing in Runner-Like Environment")
    print("=" * 60)

    # Connect to WebSocket like runner does
    print("\n1. Connecting to WebSocket...")
    try:
        await ws_client.connect()
        print("   ✅ WebSocket connected")
    except Exception as e:
        print(f"   ⚠️  WebSocket failed (expected): {e}")

    # Now try IBKR connection
    print("\n2. Connecting to IBKR...")
    try:
        client = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=1,
            readonly=True,
            timeout=15.0,
            max_retries=2,
            use_subprocess=True,
        )

        print(f"   ✅ Connected! Type: {type(client).__name__}")

        accounts = await client.get_accounts()
        print(f"   ✅ Accounts: {accounts}")

        await client.disconnect()
        await client.stop()
        print("   ✅ Disconnected")

    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_like_runner())

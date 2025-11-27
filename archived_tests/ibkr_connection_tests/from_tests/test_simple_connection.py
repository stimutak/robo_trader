#!/usr/bin/env python3
"""Simple test for IBKR connection without event loop conflicts"""

import asyncio
import sys

from ib_async import IB
from ib_async.util import patchAsyncio

# Enable nested event loops
patchAsyncio()


async def test_simple_connection():
    """Test simple async connection to TWS"""
    ib = IB()

    try:
        print("Testing connection to TWS on port 7497...")

        # Try with a unique client ID to avoid conflicts
        import os
        import time

        client_id = 10000 + (int(time.time()) % 1000) + (os.getpid() % 100)
        print(f"Using client ID: {client_id}")

        await ib.connectAsync("127.0.0.1", 7497, clientId=client_id, timeout=15)

        print("✓ Successfully connected!")
        print(f"Server version: {ib.client.serverVersion()}")
        print(f"Connection time: {ib.client.connectionTime()}")

        # Get account info
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        # Clean disconnect
        ib.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Simple IBKR Connection Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_simple_connection())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

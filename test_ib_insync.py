#!/usr/bin/env python3
"""Test with the old ib_insync library to see if it works"""

import asyncio
import sys

from ib_insync import IB
from ib_insync.util import patchAsyncio

# Enable nested event loops
patchAsyncio()


async def test_ib_insync_connection():
    """Test connection with ib_insync"""
    ib = IB()

    try:
        print("Testing ib_insync connection to TWS on port 7497...")

        # Try with a unique client ID
        import os
        import time

        client_id = 20000 + (int(time.time()) % 1000) + (os.getpid() % 100)
        print(f"Using client ID: {client_id}")

        # Use async connection
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


def test_sync_connection():
    """Test synchronous connection"""
    ib = IB()

    try:
        print("Testing synchronous ib_insync connection...")

        # Try with a unique client ID
        import os
        import time

        client_id = 30000 + (int(time.time()) % 1000) + (os.getpid() % 100)
        print(f"Using client ID: {client_id}")

        # Synchronous connection
        ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=15)

        print("✓ Successfully connected!")
        print(f"Server version: {ib.client.serverVersion()}")

        # Get account info
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        # Clean disconnect
        ib.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Sync connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ib_insync Connection Test")
    print("=" * 60)

    # Test sync first
    sync_result = test_sync_connection()
    print()

    # Test async
    async_result = asyncio.run(test_ib_insync_connection())

    print("=" * 60)
    print("Test Results:")
    print(f"  Sync connection:  {'PASS ✓' if sync_result else 'FAIL ✗'}")
    print(f"  Async connection: {'PASS ✓' if async_result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if (sync_result or async_result) else 1)

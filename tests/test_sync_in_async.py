#!/usr/bin/env python3
"""Test using synchronous connect in async context"""

import asyncio
import sys

from ib_async import IB
from ib_async.util import patchAsyncio

# Enable nested event loops
patchAsyncio()


async def test_sync_in_async():
    """Test using sync connect in async function"""
    ib = IB()

    try:
        print("Testing sync connect in async context...")

        # Use sync connect (like the working test)
        client_id = 12345
        print(f"Using client ID: {client_id}")

        # Synchronous connection in async context
        ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=10)

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
        print(f"✗ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Sync Connect in Async Context Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_sync_in_async())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

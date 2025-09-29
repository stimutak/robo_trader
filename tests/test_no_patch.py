#!/usr/bin/env python3
"""Test connection without patchAsyncio"""

import asyncio
import sys

from ib_async import IB

# DON'T import patchAsyncio


async def test_no_patch_connection():
    """Test connection without patchAsyncio"""
    ib = IB()

    try:
        print("Testing connection WITHOUT patchAsyncio...")

        # Use sync connect (like the working test)
        client_id = 15555
        print(f"Using client ID: {client_id}")

        # Synchronous connection without async patches
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
    print("No patchAsyncio Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_no_patch_connection())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

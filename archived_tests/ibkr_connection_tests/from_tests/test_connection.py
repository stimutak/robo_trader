#!/usr/bin/env python3
"""Test IBKR connection directly"""

import asyncio
import sys

from ib_async import IB, util


def test_connection():
    """Test basic connection to TWS/Gateway"""
    ib = IB()

    try:
        print("Attempting to connect to TWS on port 7497...")

        # Simple synchronous connection
        ib.connect("127.0.0.1", 7497, clientId=999, timeout=10)

        print("✓ Successfully connected!")
        print(f"Server version: {ib.client.serverVersion()}")
        print(f"Connection time: {ib.client.connectionTime()}")

        # Request account info
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


async def test_async_connection():
    """Test async connection to TWS/Gateway"""
    ib = IB()

    try:
        print("\n\nTesting ASYNC connection to TWS on port 7497...")

        # Async connection
        await ib.connectAsync("127.0.0.1", 7497, clientId=998, timeout=10)

        print("✓ Successfully connected (async)!")
        print(f"Server version: {ib.client.serverVersion()}")

        # Clean disconnect
        ib.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Async connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("IBKR Connection Test")
    print("=" * 60)

    # Test sync connection
    sync_result = test_connection()

    # Test async connection
    async_result = asyncio.run(test_async_connection())

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Sync connection:  {'PASS ✓' if sync_result else 'FAIL ✗'}")
    print(f"  Async connection: {'PASS ✓' if async_result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if (sync_result and async_result) else 1)

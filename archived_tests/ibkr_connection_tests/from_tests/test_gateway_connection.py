#!/usr/bin/env python3
"""
Simple test script to verify IB Gateway connection.
Tests both sync and async connection methods.
"""

import asyncio
import sys

from ib_async import IB, util


def test_sync_connection():
    """Test synchronous connection (standard method)."""
    print("\n=== Testing SYNC Connection ===")
    ib = IB()
    try:
        # Standard synchronous connection
        print("Connecting to IB Gateway on port 4002...")
        ib.connect("127.0.0.1", 4002, clientId=999, timeout=10)
        print("✅ Connected successfully!")
        print(f"Server version: {ib.client.serverVersion()}")

        # Get account info to verify API is working
        account_values = ib.accountValues()
        if account_values:
            print(f"✅ API working - got {len(account_values)} account values")
            # Show NetLiquidation value
            for av in account_values:
                if av.tag == "NetLiquidation":
                    print(f"   NetLiquidation: ${av.value}")
                    break

        ib.disconnect()
        return True
    except Exception as e:
        print(f"❌ Sync connection failed: {e}")
        return False


async def test_async_connection():
    """Test async connection."""
    print("\n=== Testing ASYNC Connection ===")
    ib = IB()
    try:
        print("Connecting async to IB Gateway on port 4002...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=998, timeout=10)
        print("✅ Connected successfully!")
        print(f"Server version: {ib.client.serverVersion()}")

        # Get account info to verify API is working
        account_values = await ib.accountValuesAsync()
        if account_values:
            print(f"✅ API working - got {len(account_values)} account values")
            # Show NetLiquidation value
            for av in account_values:
                if av.tag == "NetLiquidation":
                    print(f"   NetLiquidation: ${av.value}")
                    break

        ib.disconnect()
        return True
    except Exception as e:
        print(f"❌ Async connection failed: {e}")
        return False


def main():
    """Run connection tests."""
    print("Testing IB Gateway Connection Methods")
    print("=" * 50)

    # Test sync connection
    sync_ok = test_sync_connection()

    # Test async connection
    async_ok = asyncio.run(test_async_connection())

    print("\n" + "=" * 50)
    print("Results:")
    print(f"  Sync connection:  {'✅ PASS' if sync_ok else '❌ FAIL'}")
    print(f"  Async connection: {'✅ PASS' if async_ok else '❌ FAIL'}")

    if sync_ok or async_ok:
        print("\n✅ Gateway API is enabled and working!")
        print("The trading system should be able to connect.")
    else:
        print("\n❌ Both connection methods failed.")
        print("Please check Gateway API settings.")

    return 0 if (sync_ok or async_ok) else 1


if __name__ == "__main__":
    sys.exit(main())

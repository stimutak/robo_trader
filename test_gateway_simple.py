#!/usr/bin/env python3
"""
Test if ib_async works differently with Gateway vs TWS.
Try using the synchronous IB.connect() instead of connectAsync().
"""

import logging

from ib_async import IB, util

# Enable debug logging
util.logToConsole(logging.DEBUG)


def test_sync_connect():
    """Test using synchronous connect (runs its own event loop)."""
    ib = IB()

    print("=" * 60)
    print("Testing SYNCHRONOUS Gateway connection")
    print("=" * 60)

    try:
        # Use synchronous connect - it manages its own event loop
        print("Attempting sync connection to 127.0.0.1:4002...")
        ib.connect(host="127.0.0.1", port=4002, clientId=999, readonly=False, timeout=20)

        print("✅ Connected synchronously!")

        # Check if we're really connected
        if ib.isConnected():
            print("✅ Connection verified")

            # Try to get account info
            accounts = ib.managedAccounts()
            if accounts:
                print(f"✅ SUCCESS! Accounts: {accounts}")
            else:
                print("⚠️ Connected but no accounts")

            # Get server version
            print(f"Server version: {ib.client.serverVersion()}")

            ib.disconnect()
            return True
        else:
            print("❌ Not connected after connect() call")
            return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        try:
            ib.disconnect()
        except:
            pass
        return False


if __name__ == "__main__":
    success = test_sync_connect()

    print("\n" + "=" * 60)
    if success:
        print("SYNCHRONOUS CONNECTION WORKS!")
        print("\nThis means the issue is with async event loop handling.")
        print("The Gateway might need the IB.run() event loop, not asyncio.")
    else:
        print("Synchronous connection also failed.")
        print("Gateway API is not responding to any connection method.")
    print("=" * 60)

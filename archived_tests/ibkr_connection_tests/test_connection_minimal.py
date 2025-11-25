#!/usr/bin/env python3
"""
Minimal connection test - no managedAccounts() call
"""
import asyncio

from ib_async import IB


async def test_minimal_connection():
    """Test connection without calling managedAccounts()"""
    ib = IB()

    try:
        print("🔍 Testing minimal Gateway connection...")

        # Just connect - don't call any API methods
        await ib.connectAsync("127.0.0.1", 4002, clientId=1, readonly=True, timeout=10)

        print(f"✅ Connected: {ib.isConnected()}")
        print(f"✅ Client ID: {ib.client.clientId}")

        # Test managedAccounts() call - this might be the issue
        if ib.isConnected():
            print("✅ Connection successful - API handshake completed!")
            print("🔍 Testing managedAccounts() call...")
            accounts = ib.managedAccounts()
            print(f"✅ Accounts: {accounts}")
        else:
            print("❌ Connection failed")

        # Clean disconnect
        ib.disconnect()
        print("✅ Disconnected cleanly")

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        try:
            ib.disconnect()
        except:  # noqa: E722
            pass


if __name__ == "__main__":
    asyncio.run(test_minimal_connection())

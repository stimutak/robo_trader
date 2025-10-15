#!/usr/bin/env python3
"""
Test the exact robust connection logic to find the issue
"""
import asyncio

from ib_async import IB


async def test_robust_logic():
    """Test the exact logic from robust_connection.py"""
    ib = IB()

    try:
        print("🔍 Testing robust connection logic...")

        # Step 1: Connect (same as simple test)
        print("Step 1: Connecting...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=1, readonly=True, timeout=30)
        print(f"✅ Connected: {ib.isConnected()}")

        # Step 2: Get accounts (this is what robust connection does)
        print("Step 2: Getting managed accounts...")
        accounts = ib.managedAccounts()
        print(f"✅ Accounts (first try): {accounts}")

        # Step 3: Retry if no accounts (robust connection logic)
        if not accounts:
            print("Step 3: No accounts, sleeping 1 second and retrying...")
            await asyncio.sleep(1)
            accounts = ib.managedAccounts()
            print(f"✅ Accounts (second try): {accounts}")

        # Step 4: Check if accounts exist (robust connection validation)
        if not accounts:
            print("❌ No managed accounts - connection invalid")
            raise ConnectionError("No managed accounts - connection invalid")

        print(f"✅ SUCCESS! Found {len(accounts)} accounts: {accounts}")

        # Clean disconnect
        ib.disconnect()
        print("✅ Disconnected cleanly")

    except Exception as e:
        print(f"❌ Failed: {e}")
        try:
            ib.disconnect()
        except:
            pass


if __name__ == "__main__":
    asyncio.run(test_robust_logic())

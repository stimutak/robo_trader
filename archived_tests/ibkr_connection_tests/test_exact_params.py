#!/usr/bin/env python3
"""
Test with EXACT same parameters as robust connection
"""
import asyncio

from ib_async import IB


async def test_exact_params():
    """Test with exact parameters from robust connection"""
    ib = IB()

    try:
        print("🔍 Testing with EXACT robust connection parameters...")

        # EXACT parameters from robust_connection.py line 691
        print("Parameters: host=127.0.0.1 port=4002 clientId=1 readonly=True timeout=30.0")

        # This is the EXACT call from robust_connection.py
        await ib.connectAsync(host="127.0.0.1", port=4002, clientId=1, readonly=True, timeout=30.0)

        print(f"✅ Connected: {ib.isConnected()}")

        # EXACT validation from robust_connection.py
        accounts = ib.managedAccounts()
        print(f"✅ Accounts (first try): {accounts}")

        if not accounts:
            print("No accounts, sleeping 1 second...")
            await asyncio.sleep(1)
            accounts = ib.managedAccounts()
            print(f"✅ Accounts (second try): {accounts}")

        if not accounts:
            raise ConnectionError("No managed accounts - connection invalid")

        print(f"✅ SUCCESS! Found {len(accounts)} accounts: {accounts}")

        # Clean disconnect
        ib.disconnect()
        print("✅ Disconnected cleanly")

    except Exception as e:
        print(f"❌ Failed: {e}")
        try:
            ib.disconnect()
        except:  # noqa: E722
            pass


if __name__ == "__main__":
    asyncio.run(test_exact_params())

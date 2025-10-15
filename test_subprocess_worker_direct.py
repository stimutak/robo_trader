#!/usr/bin/env python3
"""
Test the subprocess worker logic directly without stdin/stdout
"""
import asyncio

from ib_async import IB


async def test_direct_connection():
    """Test direct connection without stdin/stdout"""
    ib = IB()

    try:
        print("Connecting...")
        await ib.connectAsync(host="127.0.0.1", port=4002, clientId=1, readonly=True, timeout=15.0)

        print(f"Connected: {ib.isConnected()}")

        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        ib.disconnect()
        print("Disconnected")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_direct_connection())

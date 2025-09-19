#!/usr/bin/env python3
"""Direct IBKR connection test to diagnose timeout issues."""

import asyncio
import sys

from ib_insync import IB


async def test_connection(client_id):
    """Test a single IBKR connection."""
    ib = IB()
    try:
        print(f"Testing connection with client ID {client_id}...")

        # Try to connect with a 5 second timeout
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", 7497, clientId=client_id, readonly=True), timeout=5.0
        )

        print(f"✓ Connection successful with client ID {client_id}")
        print(f"  Connected: {ib.isConnected()}")
        print(f"  Client ID: {ib.client.clientId}")

        # Try to get account info
        account_values = await ib.accountValuesAsync()
        print(f"  Account values retrieved: {len(account_values)} items")

        ib.disconnect()
        print(f"✓ Disconnected successfully")
        return True

    except asyncio.TimeoutError:
        print(f"✗ Timeout connecting with client ID {client_id}")
        if ib.isConnected():
            ib.disconnect()
        return False

    except Exception as e:
        print(f"✗ Error with client ID {client_id}: {e}")
        if ib.isConnected():
            ib.disconnect()
        return False


async def main():
    """Test multiple client IDs to find what works."""
    print("IBKR Connection Test")
    print("=" * 50)

    # Test specific problematic ID
    print("\n1. Testing problematic client ID 140:")
    await test_connection(140)

    # Test low range IDs
    print("\n2. Testing low range client IDs (1-5):")
    for client_id in range(1, 6):
        await test_connection(client_id)
        await asyncio.sleep(0.5)  # Small delay between attempts

    # Test some random IDs
    print("\n3. Testing other random client IDs:")
    for client_id in [100, 200, 500, 999]:
        await test_connection(client_id)
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())

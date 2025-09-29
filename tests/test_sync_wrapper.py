#!/usr/bin/env python3
"""Test the ConnectionManager (ib_insync based)"""

import asyncio
import logging
import sys

from robo_trader.connection_manager import ConnectionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_sync_wrapper():
    """Test the new connection manager approach"""
    try:
        print("Testing ConnectionManager...")

        mgr = ConnectionManager(host="127.0.0.1", port=7497)

        print("Attempting to connect...")
        ib = await mgr.connect()
        print("✓ Successfully connected!")
        print(f"Server version: {ib.client.serverVersion()}")
        print(f"Accounts: {ib.managedAccounts()}")

        # Test historical data
        print("\nTesting historical data...")
        from pandas import DataFrame

        df = await mgr.fetch_historical_bars("AAPL", "1 D", "5 mins")
        if isinstance(df, DataFrame) and not df.empty:
            print(f"✓ Got {len(df)} bars for AAPL")
            print(df.head(1).to_dict("records")[0])
        else:
            print("✗ Historical data failed: empty result")

        print("\nDisconnecting...")
        await mgr.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Connection Manager Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_sync_wrapper())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

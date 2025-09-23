#!/usr/bin/env python3
"""Test the synchronous IBKR wrapper"""

import asyncio
import logging
import sys

from robo_trader.clients.sync_ibkr_wrapper import SyncIBKRWrapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_sync_wrapper():
    """Test the synchronous wrapper approach"""
    try:
        print("Testing SyncIBKRWrapper...")

        # Create wrapper (auto-detects port)
        wrapper = SyncIBKRWrapper(host="127.0.0.1", port=7497, readonly=True)

        # Test connection
        print("Attempting to connect...")
        result = await wrapper.connect()

        if result["success"]:
            print("✓ Successfully connected!")
            print(f"Server version: {result['server_version']}")
            print(f"Accounts: {result['accounts']}")
            print(f"Client ID: {result['client_id']}")

            # Test historical data
            print("\nTesting historical data...")
            data_result = await wrapper.get_historical_data("AAPL", "1 D", "5 mins")

            if data_result["success"]:
                print(f"✓ Got {data_result['rows']} bars for AAPL")
                if data_result["data"]:
                    print(f"Sample data: {data_result['data'][0]}")
            else:
                print(f"✗ Historical data failed: {data_result['error']}")

            # Disconnect
            print("\nDisconnecting...")
            disc_result = await wrapper.disconnect()
            if disc_result["success"]:
                print("✓ Disconnected successfully")
            else:
                print(f"✗ Disconnect failed: {disc_result['error']}")

            return True
        else:
            print(f"✗ Connection failed: {result['error']}")
            return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Sync Wrapper Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_sync_wrapper())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

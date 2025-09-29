#!/usr/bin/env python3
"""Test the connection approach directly via ConnectionManager"""

import asyncio
import logging
import sys

from robo_trader.connection_manager import ConnectionManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_runner_connection():
    """Test the ConnectionManager approach"""
    try:
        print("Testing ConnectionManager connection...")

        mgr = ConnectionManager(host="127.0.0.1", port=7497)
        print("Attempting to connect...")
        ib = await mgr.connect()
        print("✓ Successfully connected!")

        # Try to get account info
        try:
            print(f"Accounts: {ib.managedAccounts()}")
        except Exception as e:
            print(f"Account info failed: {e}")

        # Disconnect
        await mgr.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Runner Connection Test")
    print("=" * 60)

    # Run the test
    result = asyncio.run(test_runner_connection())

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")
    print("=" * 60)

    sys.exit(0 if result else 1)

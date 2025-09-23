#!/usr/bin/env python3
"""Test the runner's connection approach directly"""

import asyncio
import logging
import sys

from robo_trader.clients.async_ibkr_client import AsyncIBKRClient, ConnectionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_runner_connection():
    """Test the runner's connection approach"""
    try:
        print("Testing AsyncIBKRClient connection...")

        # Create client with auto-detected port
        config = ConnectionConfig(
            host="127.0.0.1",
            port=7497,  # Will auto-detect
            readonly=True,
            timeout=20.0,
        )

        client = AsyncIBKRClient(config)

        # Try to connect
        print("Attempting to connect...")
        await client.connect()

        print("✓ Successfully connected!")

        # Try to get account info
        try:
            accounts = await client.get_account_summary()
            print(f"Account summary: {accounts}")
        except Exception as e:
            print(f"Account info failed: {e}")

        # Disconnect
        await client.disconnect()
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

#!/usr/bin/env python3
"""Test with longer timeout and different client IDs"""

import sys
import time


def test_with_longer_timeout():
    """Test connection with longer timeout"""
    try:
        from ib_async import IB

        ib = IB()

        # Try with longer timeout and different client ID
        client_id = int(time.time()) % 10000 + 50000  # Use high client ID
        print(f"Testing with client ID {client_id} and 30s timeout...")

        ib.connect("127.0.0.1", 7497, clientId=client_id, timeout=30)
        print("SUCCESS!")

        # Get some info
        print(f"Server version: {ib.client.serverVersion()}")
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        ib.disconnect()
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        print(f"Error type: {type(e).__name__}")
        return False


if __name__ == "__main__":
    print("Testing with longer timeout and high client ID:")
    print("=" * 60)

    result = test_with_longer_timeout()

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")

    sys.exit(0 if result else 1)

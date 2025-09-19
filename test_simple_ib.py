#!/usr/bin/env python3
"""Simple synchronous IBKR test."""

import time

from ib_insync import IB


def test_sync():
    """Test synchronous connection."""
    ib = IB()

    print("Attempting synchronous connection...")
    print("Make sure TWS/IB Gateway API Settings are:")
    print("  - Enable ActiveX and Socket Clients: CHECKED")
    print("  - Read-Only API: UNCHECKED (if you want trading)")
    print("  - Master API client ID: BLANK or 0")
    print("  - Trusted IPs: 127.0.0.1")
    print("")

    try:
        # Try synchronous connection
        ib.connect("127.0.0.1", 7497, clientId=1)
        print(f"✓ Connected successfully!")
        print(f"  Is Connected: {ib.isConnected()}")

        # Get some data to verify API is working
        print("  Getting account data...")
        accounts = ib.managedAccounts()
        print(f"  Accounts: {accounts}")

        ib.disconnect()
        print("✓ Disconnected successfully")
        return True

    except Exception as e:
        print(f"✗ Connection failed: {e}")
        if hasattr(e, "__cause__"):
            print(f"  Cause: {e.__cause__}")
        return False


if __name__ == "__main__":
    test_sync()

#!/usr/bin/env python3
"""Test the exact command that was mentioned as working in the handoff"""

import sys


def test_exact_command():
    """Test the exact command from the handoff document"""
    try:
        from ib_async import IB

        ib = IB()
        ib.connect("127.0.0.1", 7497, clientId=999, timeout=5)
        print("SUCCESS!")

        # Get some info
        print(f"Server version: {ib.client.serverVersion()}")
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        ib.disconnect()
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


if __name__ == "__main__":
    print("Testing exact command from handoff document:")
    print("ib = IB(); ib.connect('127.0.0.1', 7497, clientId=999, timeout=5)")
    print("=" * 60)

    result = test_exact_command()

    print("=" * 60)
    print(f"Result: {'PASS ✓' if result else 'FAIL ✗'}")

    sys.exit(0 if result else 1)

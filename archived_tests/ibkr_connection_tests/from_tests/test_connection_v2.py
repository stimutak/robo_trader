#!/usr/bin/env python3
"""Test IBKR connection with both IPv4 and IPv6"""

import sys

from ib_async import IB


def test_connection(host, port, client_id):
    """Test connection to TWS/Gateway"""
    ib = IB()

    try:
        print(f"Attempting connection to {host}:{port} with client ID {client_id}...")
        ib.connect(host, port, clientId=client_id, timeout=5)

        print(f"✓ Successfully connected to {host}:{port}!")
        print(f"  Server version: {ib.client.serverVersion()}")
        print(f"  Accounts: {ib.managedAccounts()}")

        ib.disconnect()
        return True

    except Exception as e:
        print(f"✗ Failed to connect to {host}:{port}: {e}")
        try:
            ib.disconnect()
        except Exception:
            pass
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("IBKR Connection Test - IPv4 vs IPv6")
    print("=" * 60)

    # Test different connection options
    tests = [
        ("127.0.0.1", 7497, 900),  # IPv4 localhost - Paper
        ("localhost", 7497, 901),  # Hostname - Paper
        ("::1", 7497, 902),  # IPv6 localhost - Paper
        ("127.0.0.1", 7496, 903),  # IPv4 - Live port
        ("127.0.0.1", 4001, 904),  # IB Gateway paper port
        ("127.0.0.1", 4002, 905),  # IB Gateway live port
    ]

    results = []
    for host, port, client_id in tests:
        success = test_connection(host, port, client_id)
        results.append((host, port, success))
        print()

    print("=" * 60)
    print("Summary:")
    for host, port, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {host:15} port {port:4} : {status}")
    print("=" * 60)

    # Exit with success if any connection worked
    sys.exit(0 if any(r[2] for r in results) else 1)

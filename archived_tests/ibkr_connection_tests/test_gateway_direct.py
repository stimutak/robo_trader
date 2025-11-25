#!/usr/bin/env python3
"""
Minimal test script to connect to IBKR Gateway directly.
This bypasses all our wrapper code to isolate the issue.
"""

import asyncio
import logging

from ib_async import IB, util

# Enable debug logging
util.logToConsole(logging.DEBUG)


async def test_gateway():
    """Test direct connection to Gateway."""
    ib = IB()

    print("=" * 60)
    print("TESTING DIRECT GATEWAY CONNECTION")
    print("=" * 60)
    print(f"Target: 127.0.0.1:4002")
    print(f"Client ID: 999")
    print("=" * 60)

    try:
        # Try connecting with minimal parameters
        print("\n1. Attempting connection...")
        await ib.connectAsync(
            host="127.0.0.1", port=4002, clientId=999, timeout=20  # Shorter timeout for testing
        )

        print("2. TCP connection established!")

        # Try to get account info
        print("3. Requesting account info...")
        accounts = ib.managedAccounts()

        if accounts:
            print(f"✅ SUCCESS! Connected to Gateway. Accounts: {accounts}")

            # Get some basic info
            print("\n4. Getting server version...")
            print(f"   Server Version: {ib.client.serverVersion()}")

            print("\n5. Getting connection time...")
            print(f"   Connected at: {ib.client.connTime}")

            return True
        else:
            print("❌ Connected but no accounts returned")
            return False

    except asyncio.TimeoutError:
        print("\n❌ TIMEOUT: Connection timed out after 20 seconds")
        print("   This means TCP connected but API handshake failed")
        return False

    except ConnectionRefusedError as e:
        print(f"\n❌ CONNECTION REFUSED: {e}")
        print("   Gateway is not listening on port 4002")
        return False

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        return False

    finally:
        if ib.isConnected():
            print("\n6. Disconnecting...")
            ib.disconnect()
            print("   Disconnected")


if __name__ == "__main__":
    print("Starting Gateway connection test...\n")
    success = asyncio.run(test_gateway())

    print("\n" + "=" * 60)
    if success:
        print("TEST PASSED: Gateway connection working!")
    else:
        print("TEST FAILED: Could not connect to Gateway")
        print("\nPossible issues:")
        print("1. Gateway API not enabled (check Configuration > API > Settings)")
        print("2. Gateway needs restart after configuration change")
        print("3. Firewall blocking connection")
        print("4. Gateway in bad state (restart needed)")
    print("=" * 60)

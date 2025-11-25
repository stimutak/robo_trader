#!/usr/bin/env python3
"""
Minimal test to isolate IBKR Gateway connection issue.

This script tests the exact connection flow used by the system.
"""

import asyncio
import sys

from ib_async import IB


async def test_gateway_connection():
    """Test direct connection to IBKR Gateway."""

    print("=" * 60)
    print("TESTING IBKR GATEWAY CONNECTION")
    print("=" * 60)

    # Test parameters
    host = "127.0.0.1"
    port = 4002
    client_id = 0
    timeout = 60.0
    readonly = False

    print(f"\nConnection parameters:")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Client ID: {client_id}")
    print(f"  Timeout: {timeout}s")
    print(f"  Readonly: {readonly}")
    print()

    ib = IB()

    try:
        print("Attempting connection...")
        print(f"Calling ib.connectAsync()...")

        # This is the exact call from robust_connection.py line 572-574
        await ib.connectAsync(
            host=host, port=port, clientId=client_id, timeout=timeout, readonly=readonly
        )

        print(f"✅ Connection successful!")
        print(f"   isConnected: {ib.isConnected()}")

        # Verify we can get accounts
        print("\nVerifying connection...")
        accounts = ib.managedAccounts()
        print(f"   Managed accounts: {accounts}")

        if not accounts:
            print("   Waiting 1 second for accounts...")
            await asyncio.sleep(1)
            accounts = ib.managedAccounts()
            print(f"   Managed accounts (retry): {accounts}")

        if accounts:
            print(f"✅ Connection fully functional!")
            return True
        else:
            print(f"❌ No accounts returned - connection incomplete")
            return False

    except asyncio.TimeoutError as e:
        print(f"❌ Connection timed out after {timeout}s")
        print(f"   Error: {e}")
        return False

    except asyncio.CancelledError as e:
        print(f"❌ Connection cancelled")
        print(f"   Error: {e}")
        return False

    except Exception as e:
        print(f"❌ Connection failed with exception:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if ib.isConnected():
            print("\nDisconnecting...")
            ib.disconnect()
            print("✅ Disconnected")
        else:
            print("\n❌ Not connected (no disconnect needed)")


async def test_with_variations():
    """Test connection with different parameter variations."""

    print("\n" + "=" * 60)
    print("TESTING PARAMETER VARIATIONS")
    print("=" * 60)

    test_configs = [
        {"client_id": 0, "readonly": False, "desc": "client_id=0, readonly=False (current)"},
        {"client_id": 0, "readonly": True, "desc": "client_id=0, readonly=True"},
        {"client_id": 1, "readonly": False, "desc": "client_id=1, readonly=False"},
        {"client_id": 1, "readonly": True, "desc": "client_id=1, readonly=True"},
    ]

    results = []

    for config in test_configs:
        print(f"\n--- Test: {config['desc']} ---")

        ib = IB()
        success = False

        try:
            await ib.connectAsync(
                host="127.0.0.1",
                port=4002,
                clientId=config["client_id"],
                timeout=15.0,  # Shorter timeout for testing
                readonly=config["readonly"],
            )

            accounts = ib.managedAccounts()
            if accounts:
                print(f"✅ SUCCESS - Accounts: {accounts}")
                success = True
            else:
                print(f"⚠️ PARTIAL - Connected but no accounts")
                success = False

        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT")
        except Exception as e:
            print(f"❌ FAILED - {type(e).__name__}: {e}")
        finally:
            if ib.isConnected():
                ib.disconnect()
                await asyncio.sleep(0.5)  # Give time to clean up

        results.append((config["desc"], success))

        # Wait between tests
        await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for desc, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {desc}")


async def main():
    """Main test runner."""

    # Test 1: Basic connection test
    print("TEST 1: Basic Connection")
    success = await test_gateway_connection()

    if not success:
        print("\n⚠️ Basic connection failed. Trying parameter variations...")
        print()

        # Test 2: Try different parameters
        print("TEST 2: Parameter Variations")
        await test_with_variations()

    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

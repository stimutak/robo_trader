#!/usr/bin/env python3
"""Test TWS connection directly to diagnose handshake timeout issues."""

import asyncio
import sys

from ib_async import IB, util

# Enable debug logging to see what's happening
util.logToConsole("DEBUG")


async def test_connection():
    """Test connection to TWS with various client IDs."""
    ib = IB()

    # Try different client IDs
    for client_id in [0, 1, 99, 123]:
        print(f"\n{'='*60}")
        print(f"Testing connection with client_id={client_id}...")
        print(f"{'='*60}")

        try:
            # Attempt connection with a longer timeout
            await asyncio.wait_for(
                ib.connectAsync(
                    "127.0.0.1",
                    7497,
                    clientId=client_id,
                    timeout=20.0,  # Longer timeout to see what happens
                ),
                timeout=25.0,
            )

            print(f"✅ Connected successfully!")

            # Try to get accounts
            accounts = ib.managedAccounts()
            print(f"Managed accounts: {accounts}")

            # Disconnect cleanly
            ib.disconnect()
            print("Disconnected cleanly")

            return True

        except asyncio.TimeoutError:
            print(f"❌ Connection timed out after 25 seconds")
        except Exception as e:
            print(f"❌ Connection failed: {type(e).__name__}: {e}")

        finally:
            if ib.isConnected():
                ib.disconnect()

    return False


async def main():
    """Run the test."""
    print("Testing TWS API connection...")
    print("Make sure TWS is running and API is enabled on port 7497")

    success = await test_connection()
    if success:
        print("\n✅ Connection test PASSED - TWS API is working")
    else:
        print("\n❌ Connection test FAILED")
        print("\nTroubleshooting steps:")
        print("1. Open TWS")
        print("2. Go to File → Global Configuration → API → Settings")
        print("3. Enable 'Enable ActiveX and Socket Clients'")
        print("4. Add '127.0.0.1' to Trusted IP Addresses")
        print("5. Set 'Master API client ID' to 0")
        print("6. Uncheck 'Read-Only API'")
        print("7. Apply and restart TWS")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

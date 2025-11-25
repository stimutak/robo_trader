#!/usr/bin/env python3
"""
Test Gateway connection using exact same approach that worked with TWS.
Maybe Gateway needs readonly=True after all?
"""

import asyncio

from ib_async import IB


async def test_variations():
    """Test different connection variations."""

    variations = [
        {"readonly": True, "client_id": 0, "desc": "readonly=True, clientId=0 (like TWS)"},
        {"readonly": True, "client_id": 1, "desc": "readonly=True, clientId=1"},
        {"readonly": False, "client_id": 0, "desc": "readonly=False, clientId=0"},
        {"readonly": False, "client_id": 123, "desc": "readonly=False, clientId=123"},
    ]

    for var in variations:
        print("\n" + "=" * 60)
        print(f"Testing: {var['desc']}")
        print("=" * 60)

        ib = IB()
        try:
            print(f"Connecting to 127.0.0.1:4002...")
            await asyncio.wait_for(
                ib.connectAsync(
                    host="127.0.0.1",
                    port=4002,
                    clientId=var["client_id"],
                    readonly=var["readonly"],
                    timeout=10,
                ),
                timeout=10,
            )

            # If we get here, connection worked!
            print("✅ CONNECTION SUCCESSFUL!")

            # Try to get accounts
            await asyncio.sleep(1)
            accounts = ib.managedAccounts()
            if accounts:
                print(f"   Accounts: {accounts}")
            else:
                print("   Warning: No accounts returned")

            ib.disconnect()
            print(f"\n🎉 WORKING CONFIG: {var['desc']}")
            return True

        except asyncio.TimeoutError:
            print("❌ Timeout - API handshake failed")
            try:
                ib.disconnect()
            except:  # noqa: E722
                pass

        except Exception as e:
            print(f"❌ Error: {e}")
            try:
                ib.disconnect()
            except:  # noqa: E722
                pass

    return False


if __name__ == "__main__":
    print("Testing different Gateway connection configurations...")
    print("Looking for a working combination...\n")

    success = asyncio.run(test_variations())

    if not success:
        print("\n" + "=" * 60)
        print("ALL VARIATIONS FAILED")
        print("\nThis confirms Gateway API is not responding.")
        print("\nDespite being configured correctly, the Gateway is not")
        print("processing API connections. This usually means:")
        print("\n1. Gateway needs a full restart (not just stop/start)")
        print("2. Gateway needs to be fully closed and restarted")
        print("3. There may be a Gateway process stuck in background")
        print("\nTry:")
        print("- Fully quit Gateway (Cmd+Q or File > Exit)")
        print("- Wait 10 seconds")
        print("- Start Gateway fresh")
        print("- Make sure to login to PAPER trading account")
        print("=" * 60)

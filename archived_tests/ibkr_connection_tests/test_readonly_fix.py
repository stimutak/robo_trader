#!/usr/bin/env python3
"""
Test to verify readonly=True fixes the Gateway timeout issue.
"""

import asyncio

from ib_async import IB


async def test_readonly_modes():
    """Compare readonly=False vs readonly=True."""

    test_configs = [
        {"readonly": False, "desc": "readonly=False (CURRENT - BROKEN)"},
        {"readonly": True, "desc": "readonly=True (EXPECTED FIX)"},
    ]

    for config in test_configs:
        print("\n" + "=" * 60)
        print(f"TEST: {config['desc']}")
        print("=" * 60)

        ib = IB()

        try:
            print(f"Connecting with readonly={config['readonly']}...")
            start = asyncio.get_event_loop().time()

            await ib.connectAsync(
                host="127.0.0.1", port=4002, clientId=0, timeout=60.0, readonly=config["readonly"]
            )

            elapsed = asyncio.get_event_loop().time() - start
            print(f"✅ Connected in {elapsed:.2f}s")
            print(f"   isConnected: {ib.isConnected()}")

            accounts = ib.managedAccounts()
            print(f"   Accounts: {accounts}")

            if not accounts:
                await asyncio.sleep(1)
                accounts = ib.managedAccounts()
                print(f"   Accounts (retry): {accounts}")

            if accounts:
                print(f"✅ SUCCESS - Full connection in {elapsed:.2f}s")
            else:
                print(f"❌ PARTIAL - Connected but no accounts")

        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start
            print(f"❌ TIMEOUT after {elapsed:.2f}s")

        except Exception as e:
            print(f"❌ FAILED: {type(e).__name__}: {e}")

        finally:
            if ib.isConnected():
                ib.disconnect()
                await asyncio.sleep(0.5)

        # Wait between tests
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(test_readonly_modes())

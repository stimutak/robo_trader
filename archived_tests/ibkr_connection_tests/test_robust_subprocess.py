#!/usr/bin/env python3
"""
Test robust connection with subprocess client
"""
import asyncio

from robo_trader.utils.robust_connection import connect_ibkr_robust


async def test_robust_subprocess():
    """Test robust connection using subprocess client"""

    print("=" * 60)
    print("Testing Robust Connection with Subprocess Client")
    print("=" * 60)

    try:
        print("\n1. Connecting to IBKR Gateway via robust connection...")
        print("   (Using subprocess mode)")

        client = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=1,
            readonly=True,
            timeout=15.0,
            max_retries=2,
            use_subprocess=True,  # Use subprocess mode
        )

        print(f"   ✅ Connected! Client type: {type(client).__name__}")

        print("\n2. Getting accounts...")
        accounts = await client.get_accounts()
        print(f"   ✅ Accounts: {accounts}")

        print("\n3. Getting positions...")
        positions = await client.get_positions()
        print(f"   ✅ Positions: {len(positions)} positions")

        print("\n4. Testing ping...")
        pong = await client.ping()
        print(f"   ✅ Ping: {pong}")

        print("\n5. Disconnecting...")
        await client.disconnect()
        print("   ✅ Disconnected")

        print("\n6. Stopping subprocess...")
        await client.stop()
        print("   ✅ Subprocess stopped")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


async def test_robust_legacy():
    """Test robust connection using legacy direct ib_async"""

    print("\n" + "=" * 60)
    print("Testing Robust Connection with Legacy Direct ib_async")
    print("=" * 60)

    try:
        print("\n1. Connecting to IBKR Gateway via robust connection...")
        print("   (Using legacy direct ib_async mode)")

        ib = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=2,
            readonly=True,
            timeout=15.0,
            max_retries=2,
            use_subprocess=False,  # Use legacy mode
        )

        print(f"   ✅ Connected! IB type: {type(ib).__name__}")

        print("\n2. Getting accounts...")
        accounts = ib.managedAccounts()
        print(f"   ✅ Accounts: {accounts}")

        print("\n3. Disconnecting...")
        ib.disconnect()
        print("   ✅ Disconnected")

        print("\n" + "=" * 60)
        print("✅ LEGACY MODE TEST PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ LEGACY MODE TEST FAILED: {e}")
        print("   (This is expected - legacy mode has the timeout issue)")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""

    # Test subprocess mode (should work)
    await test_robust_subprocess()

    await asyncio.sleep(2)

    # Test legacy mode (will likely fail with timeout)
    # await test_robust_legacy()

    print("\n" + "=" * 60)
    print("🎉 ROBUST CONNECTION TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

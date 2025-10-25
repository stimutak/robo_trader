#!/usr/bin/env python3
"""
Test subprocess-based IBKR client

Tests the new subprocess isolation approach to verify it solves the
ib_async async environment incompatibility issue.
"""
import asyncio

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient


async def test_subprocess_connection():
    """Test subprocess-based IBKR connection"""

    print("=" * 60)
    print("Testing Subprocess-Based IBKR Client")
    print("=" * 60)

    client = SubprocessIBKRClient()

    try:
        # Start subprocess
        print("\n1. Starting subprocess worker...")
        await client.start()
        print("   ✅ Subprocess started")

        # Test ping
        print("\n2. Testing ping...")
        pong = await client.ping()
        print(f"   ✅ Ping successful: {pong}")

        # Connect to IBKR
        print("\n3. Connecting to IBKR Gateway (port 4002)...")
        connected = await client.connect(
            host="127.0.0.1", port=4002, client_id=1, readonly=True, timeout=15.0
        )
        print(f"   ✅ Connected: {connected}")

        # Get accounts
        print("\n4. Getting managed accounts...")
        accounts = await client.get_accounts()
        print(f"   ✅ Accounts: {accounts}")

        # Get positions
        print("\n5. Getting positions...")
        positions = await client.get_positions()
        print(f"   ✅ Positions: {len(positions)} positions")
        for pos in positions:
            print(f"      - {pos['contract']['symbol']}: {pos['position']} @ {pos['avgCost']}")

        # Get account summary
        print("\n6. Getting account summary...")
        summary = await client.get_account_summary()
        print(f"   ✅ Account summary: {len(summary)} values")
        # Print a few key values
        for key in ["NetLiquidation_USD", "TotalCashValue_USD", "BuyingPower"]:
            if key in summary:
                print(f"      - {key}: {summary[key]}")

        # Test ping while connected
        print("\n7. Testing ping while connected...")
        pong = await client.ping()
        print(f"   ✅ Ping successful: {pong}")

        # Disconnect
        print("\n8. Disconnecting...")
        await client.disconnect()
        print("   ✅ Disconnected")

        # Test ping after disconnect
        print("\n9. Testing ping after disconnect...")
        pong = await client.ping()
        print(f"   ✅ Ping successful: {pong}")

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Stop subprocess
        print("\n10. Stopping subprocess...")
        await client.stop()
        print("    ✅ Subprocess stopped")


async def test_subprocess_reconnection():
    """Test subprocess reconnection after disconnect"""

    print("\n" + "=" * 60)
    print("Testing Subprocess Reconnection")
    print("=" * 60)

    client = SubprocessIBKRClient()

    try:
        await client.start()

        # First connection
        print("\n1. First connection...")
        await client.connect("127.0.0.1", 4002, 1)
        accounts1 = await client.get_accounts()
        print(f"   ✅ Connected, accounts: {accounts1}")

        # Disconnect
        print("\n2. Disconnecting...")
        await client.disconnect()
        print("   ✅ Disconnected")

        # Second connection (different client ID)
        print("\n3. Second connection (different client ID)...")
        await client.connect("127.0.0.1", 4002, 2)
        accounts2 = await client.get_accounts()
        print(f"   ✅ Connected, accounts: {accounts2}")

        # Verify same accounts
        assert accounts1 == accounts2, "Accounts mismatch!"
        print("   ✅ Accounts match")

        await client.disconnect()

        print("\n✅ RECONNECTION TEST PASSED!")

    except Exception as e:
        print(f"\n❌ RECONNECTION TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    finally:
        await client.stop()


async def test_subprocess_context_manager():
    """Test subprocess client as async context manager"""

    print("\n" + "=" * 60)
    print("Testing Subprocess Context Manager")
    print("=" * 60)

    try:
        async with SubprocessIBKRClient() as client:
            print("\n1. Context manager entered (subprocess started)")

            await client.connect("127.0.0.1", 4002, 1)
            print("2. Connected")

            accounts = await client.get_accounts()
            print(f"3. Accounts: {accounts}")

            await client.disconnect()
            print("4. Disconnected")

        print("5. Context manager exited (subprocess stopped)")
        print("\n✅ CONTEXT MANAGER TEST PASSED!")

    except Exception as e:
        print(f"\n❌ CONTEXT MANAGER TEST FAILED: {e}")
        import traceback

        traceback.print_exc()


async def main():
    """Run all tests"""

    # Test 1: Basic connection
    await test_subprocess_connection()

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Test 2: Reconnection
    await test_subprocess_reconnection()

    # Wait a bit between tests
    await asyncio.sleep(2)

    # Test 3: Context manager
    await test_subprocess_context_manager()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

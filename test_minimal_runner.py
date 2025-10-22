#!/usr/bin/env python3
"""
Minimal runner to test subprocess IBKR connection
Bypasses all the WebSocket and other complexity
"""
import asyncio

from robo_trader.utils.robust_connection import connect_ibkr_robust


async def minimal_runner():
    """Minimal trading runner"""

    print("=" * 60)
    print("Minimal Trading Runner - Testing Subprocess IBKR")
    print("=" * 60)

    client = None

    try:
        # Connect to IBKR
        print("\n1. Connecting to IBKR Gateway...")
        client = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=1,
            readonly=True,
            timeout=15.0,
            max_retries=2,
            use_subprocess=True,
        )

        print(f"   ✅ Connected! Type: {type(client).__name__}")

        # Get accounts
        print("\n2. Getting accounts...")
        accounts = await client.get_accounts()
        print(f"   ✅ Accounts: {accounts}")

        # Get positions
        print("\n3. Getting positions...")
        positions = await client.get_positions()
        print(f"   ✅ Positions: {len(positions)}")
        for pos in positions:
            print(f"      {pos['contract']['symbol']}: {pos['position']} @ ${pos['avgCost']:.2f}")

        # Get account summary
        print("\n4. Getting account summary...")
        summary = await client.get_account_summary()
        print(f"   ✅ Account summary: {len(summary)} values")

        # Print key values
        for key in ["NetLiquidation_USD", "TotalCashValue_USD", "BuyingPower"]:
            if key in summary:
                print(f"      {key}: ${summary[key]}")

        # Keep connection alive for a bit
        print("\n5. Keeping connection alive for 10 seconds...")
        for i in range(10):
            await asyncio.sleep(1)
            pong = await client.ping()
            if i % 3 == 0:
                print(f"      Ping {i+1}/10: {pong}")

        print("\n   ✅ Connection stable!")

        print("\n" + "=" * 60)
        print("✅ MINIMAL RUNNER SUCCESS!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if client:
            print("\nCleaning up...")
            try:
                await client.disconnect()
                await client.stop()
                print("✅ Cleaned up")
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")


if __name__ == "__main__":
    asyncio.run(minimal_runner())

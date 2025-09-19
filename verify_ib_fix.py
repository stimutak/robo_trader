#!/usr/bin/env python3
"""
Verification script to test IB Gateway connection after API settings fix.
Run this after enabling API settings in IB Gateway.
"""

import asyncio
import sys
import time

from ib_insync import IB, Stock


async def verify_connection():
    """Verify IB Gateway connection is working properly."""

    print("=" * 60)
    print("VERIFYING IB GATEWAY API FIX")
    print("=" * 60)

    ib = IB()

    try:
        print("🔄 Testing API connection...")
        start_time = time.time()

        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", 4002, clientId=998),
            timeout=10,  # Should connect quickly now
        )

        connection_time = time.time() - start_time
        print(f"✅ Connected in {connection_time:.2f}s")

        # Test account access
        accounts = ib.managedAccounts()
        print(f"✅ Accounts: {accounts}")

        # Test market data
        print("🔄 Testing market data...")
        contract = Stock("AAPL", "SMART", "USD")
        details = await ib.reqContractDetailsAsync(contract)

        if details:
            print("✅ Market data access working")

        # Test historical data
        print("🔄 Testing historical data...")
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr="1 D",
            barSizeSetting="1 hour",
            whatToShow="TRADES",
            useRTH=True,
        )

        if bars:
            print(f"✅ Historical data: {len(bars)} bars received")

        ib.disconnect()

        print("\n🎉 SUCCESS! IB Gateway API is working correctly!")
        print("✅ Your trading system should now connect without issues.")
        return True

    except asyncio.TimeoutError:
        print("❌ Still timing out - API settings may not be applied yet")
        print("   Try restarting IB Gateway and run this test again")
        return False

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


async def main():
    success = await verify_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

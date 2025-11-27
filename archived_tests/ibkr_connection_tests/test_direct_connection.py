#!/usr/bin/env python3
"""Direct connection test that mimics what should work."""

import asyncio

from ib_async import IB


async def connect_and_get_data():
    ib = IB()
    try:
        # Connect with client_id=0 (which works)
        print("Connecting with client_id=0...")
        await ib.connectAsync("127.0.0.1", 7497, clientId=0, timeout=10)
        print("✅ Connected successfully!")

        # Get accounts to verify
        accounts = ib.managedAccounts()
        print(f"Managed accounts: {accounts}")

        # Subscribe to market data for a symbol
        from ib_async import MarketDataType, Stock

        contract = Stock("AAPL", "SMART", "USD")
        ib.reqMarketDataType(MarketDataType.DELAYED)

        ticker = ib.reqMktData(contract, "", False, False)
        await asyncio.sleep(5)  # Wait for data

        print(f"AAPL bid: {ticker.bid}, ask: {ticker.ask}, last: {ticker.last}")

        if ticker.last and ticker.last > 0:
            print("✅ Market data flowing! Signals can be generated.")
        else:
            print("⚠️ No market data yet")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        ib.disconnect()


asyncio.run(connect_and_get_data())

#!/usr/bin/env python3
"""Quick TWS connection test"""

import asyncio
from ib_insync import IB

async def test():
    ib = IB()
    try:
        print("Connecting to TWS on port 7497...")
        await ib.connectAsync('127.0.0.1', 7497, clientId=0)
        print("✅ SUCCESS! Connected to TWS Paper Trading")
        print(f"Account: {ib.managedAccounts()}")
        ib.disconnect()
    except Exception as e:
        print(f"❌ Failed: {e}")
        print("\nPlease check TWS:")
        print("1. File → Global Configuration → API → Settings")
        print("2. Enable 'Enable ActiveX and Socket Clients'")
        print("3. Add 127.0.0.1 to Trusted IP Addresses")
        print("4. Socket port should be 7497")

asyncio.run(test())
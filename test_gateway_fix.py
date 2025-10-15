#!/usr/bin/env python3
"""
Quick test script to verify Gateway API fix.
Run this after configuring Gateway API settings.
"""

import asyncio
import time

from ib_async import IB


async def quick_connection_test():
    """Test if Gateway API is now working."""
    print("🔍 Testing Gateway API connection...")

    ib = IB()
    try:
        start_time = time.time()

        await asyncio.wait_for(
            ib.connectAsync(host="127.0.0.1", port=4002, clientId=1, timeout=10, readonly=True),
            timeout=12,
        )

        connect_time = time.time() - start_time
        print(f"✅ SUCCESS! Connected in {connect_time:.2f}s")

        # Test basic functionality
        accounts = ib.managedAccounts()
        print(f"✅ Accounts: {accounts}")

        print(f"✅ Connected: {ib.isConnected()}")

        print("\n🎉 Gateway API is working! You can now run the trading system.")
        return True

    except asyncio.TimeoutError:
        print("❌ Still timing out. Check Gateway API settings again.")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    finally:
        if ib.isConnected():
            ib.disconnect()


async def main():
    success = await quick_connection_test()

    if success:
        print("\n🚀 NEXT STEPS:")
        print("1. Kill any running processes:")
        print("   pkill -9 -f 'runner_async' && pkill -9 -f 'app.py'")
        print("\n2. Start the trading system:")
        print("   source .venv/bin/activate")
        print("   python3 -m robo_trader.websocket_server &")
        print("   python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA &")
        print("   export DASH_PORT=5555 && python3 app.py &")
    else:
        print("\n❌ Gateway API still not working.")
        print("Double-check these Gateway settings:")
        print("- Configure → Settings → API")
        print("- Enable ActiveX and Socket Clients ✓")
        print("- Socket port: 4002")
        print("- Trusted IPs: 127.0.0.1")
        print("- Restart Gateway after changes")


if __name__ == "__main__":
    asyncio.run(main())

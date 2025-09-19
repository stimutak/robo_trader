#!/usr/bin/env python3
"""
IB Gateway connection test that handles API connection dialogs.
This script tests connection and provides guidance on handling Gateway-specific issues.
"""

import asyncio
import sys
import time

from ib_insync import IB


async def test_gateway_with_dialog_handling():
    """Test IB Gateway connection with dialog handling guidance."""

    print("=" * 70)
    print("IB GATEWAY CONNECTION TEST - DIALOG HANDLING")
    print("=" * 70)

    print("\n🔍 STEP 1: Checking IB Gateway Status...")

    # Check if Gateway is running and on correct port
    import socket

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex(("127.0.0.1", 4002))
        sock.close()

        if result == 0:
            print("✅ IB Gateway is running on port 4002")
        else:
            print("❌ IB Gateway not running on port 4002")
            print("   Please start IB Gateway and log in completely")
            return False
    except Exception as e:
        print(f"❌ Socket test failed: {e}")
        return False

    print("\n🔍 STEP 2: Testing API Connection...")
    print("⚠️  IMPORTANT: Watch IB Gateway window for popup dialogs!")
    print("   If you see any connection dialogs, click 'Accept' or 'Yes'")
    print("   Common dialogs:")
    print("   - 'Incoming connection' → Accept")
    print("   - 'API client requesting connection' → Accept")
    print("   - 'Allow API connections' → Yes")

    ib = IB()

    try:
        print("\n🔄 Attempting API connection (30s timeout)...")
        print("   Client ID: 997 (test)")
        print("   If connection hangs, check Gateway for dialogs!")

        start_time = time.time()

        # Use longer timeout and provide progress updates
        connection_task = asyncio.create_task(ib.connectAsync("127.0.0.1", 4002, clientId=997))

        # Wait with progress updates
        for i in range(30):
            if connection_task.done():
                break
            await asyncio.sleep(1)
            if i % 5 == 4:  # Every 5 seconds
                elapsed = i + 1
                print(f"   ⏳ Still connecting... {elapsed}s elapsed")
                print("      👀 Check IB Gateway for popup dialogs!")

        # Check if connection completed
        if not connection_task.done():
            print("\n❌ CONNECTION TIMEOUT AFTER 30 SECONDS")
            print("\n🔧 LIKELY CAUSES:")
            print("1. 📋 API connection dialog waiting for user input")
            print("   → Check IB Gateway window for popup dialogs")
            print("   → Click 'Accept' on any connection requests")
            print("\n2. 🔐 Gateway not fully logged in")
            print("   → Ensure you see account balance in Gateway")
            print("   → Check for login errors or prompts")
            print("\n3. ⚙️  API settings issue")
            print("   → Configure → Settings → API")
            print("   → Add 127.0.0.1 to Trusted IPs")
            print("   → Uncheck 'Read-Only API'")

            connection_task.cancel()
            return False

        # Connection successful
        connection_time = time.time() - start_time
        print(f"\n✅ API CONNECTION SUCCESSFUL in {connection_time:.2f}s!")

        # Test basic functionality
        print("\n🔍 STEP 3: Testing API Functionality...")

        # Get server version
        server_version = ib.client.serverVersion()
        print(f"   Server version: {server_version}")

        # Get managed accounts
        accounts = ib.managedAccounts()
        print(f"   Managed accounts: {accounts}")

        if not accounts:
            print("⚠️  No managed accounts found - check Gateway login")

        # Test contract details
        print("   Testing contract details request...")
        from ib_insync import Stock

        contract = Stock("AAPL", "SMART", "USD")

        try:
            details = await asyncio.wait_for(ib.reqContractDetailsAsync(contract), timeout=10)
            if details:
                print("✅ Contract details request successful")
            else:
                print("⚠️  Contract details request returned empty")
        except asyncio.TimeoutError:
            print("⚠️  Contract details request timed out")
        except Exception as e:
            print(f"⚠️  Contract details error: {e}")

        # Clean disconnect
        ib.disconnect()

        print("\n🎉 SUCCESS! IB Gateway API is working correctly!")
        print("\n📋 NEXT STEPS:")
        print("1. Your IB Gateway is properly configured")
        print("2. Run your trading system: python -m robo_trader.runner_async")
        print("3. If you get dialogs again, always click 'Accept'")

        return True

    except Exception as e:
        print(f"\n❌ API CONNECTION FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")

        print("\n🔧 TROUBLESHOOTING STEPS:")
        print("1. 🔄 Restart IB Gateway completely")
        print("2. 🔐 Log in again with your credentials")
        print("3. ⚙️  Check API settings (Configure → Settings → API)")
        print("4. 🔥 Check firewall/antivirus blocking connections")
        print("5. 📞 Contact IB support if issues persist")

        if ib.isConnected():
            ib.disconnect()
        return False


async def main():
    """Main test function."""

    print("Testing IB Gateway API connection with dialog handling...")

    success = await test_gateway_with_dialog_handling()

    if success:
        print("\n" + "=" * 70)
        print("🎉 IB GATEWAY CONNECTION TEST PASSED!")
        print("=" * 70)
        print("Your trading system should now connect successfully.")
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ IB GATEWAY CONNECTION TEST FAILED")
        print("=" * 70)
        print("Please fix the issues above before running your trading system.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

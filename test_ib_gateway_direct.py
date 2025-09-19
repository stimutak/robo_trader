#!/usr/bin/env python3
"""
Direct IB Gateway connection test to isolate API configuration issues.
This script tests the exact same connection pattern as the trading system.
"""

import asyncio
import sys
import time

from ib_insync import IB


async def test_gateway_connection():
    """Test IB Gateway connection with detailed logging."""

    print("=" * 60)
    print("IB GATEWAY CONNECTION DIAGNOSTIC TEST")
    print("=" * 60)

    # Test socket connectivity first
    import socket

    print("\n1. Testing socket connectivity...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(("127.0.0.1", 7497))
        sock.close()

        if result == 0:
            print("✅ Socket connection to port 7497 successful")
        else:
            print("❌ Socket connection failed")
            return False
    except Exception as e:
        print(f"❌ Socket test failed: {e}")
        return False

    # Test IB API connection
    print("\n2. Testing IB API connection...")
    ib = IB()

    try:
        print("   Attempting API connection with 30s timeout...")
        start_time = time.time()

        # Use the same connection pattern as the trading system
        await asyncio.wait_for(ib.connectAsync("127.0.0.1", 7497, clientId=999), timeout=30)

        connection_time = time.time() - start_time
        print(f"✅ API connection successful in {connection_time:.2f}s")

        # Test basic API functionality
        print("\n3. Testing API functionality...")

        # Get server version
        server_version = ib.client.serverVersion()
        print(f"   Server version: {server_version}")

        # Get managed accounts
        accounts = ib.managedAccounts()
        print(f"   Managed accounts: {accounts}")

        # Test market data subscription
        print("   Testing market data subscription...")
        from ib_insync import Stock

        contract = Stock("AAPL", "SMART", "USD")

        # Request contract details
        contract_details = await ib.reqContractDetailsAsync(contract)
        if contract_details:
            print("✅ Contract details request successful")
        else:
            print("⚠️  Contract details request failed")

        print("\n✅ ALL TESTS PASSED - IB Gateway API is working correctly")

        ib.disconnect()
        return True

    except asyncio.TimeoutError:
        print("❌ API CONNECTION TIMEOUT")
        print("   This indicates IB Gateway API is not properly enabled")
        print("\n🔧 REQUIRED FIXES:")
        print("   1. Open IB Gateway")
        print("   2. Go to Configure → Settings → API → Settings")
        print("   3. Check 'Enable ActiveX and Socket Clients'")
        print("   4. Add '127.0.0.1' to Trusted IP Addresses")
        print("   5. Uncheck 'Read-Only API' if you want to place orders")
        print("   6. Click Apply and OK")
        print("   7. Restart IB Gateway")

        if ib.isConnected():
            ib.disconnect()
        return False

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        if ib.isConnected():
            ib.disconnect()
        return False


async def main():
    """Main test function."""
    success = await test_gateway_connection()

    if success:
        print("\n🎉 IB Gateway is properly configured!")
        print("   Your trading system should now connect successfully.")
        sys.exit(0)
    else:
        print("\n❌ IB Gateway configuration issues detected.")
        print("   Please fix the API settings before running the trading system.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

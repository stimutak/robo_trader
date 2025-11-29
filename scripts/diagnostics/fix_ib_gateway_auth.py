#!/usr/bin/env python3
"""
Fix IB Gateway authentication issues.
The connection succeeds but authentication is incomplete.
"""

import asyncio
import sys
import time

from ib_async import IB


async def diagnose_auth_issue():
    """Diagnose IB Gateway authentication problems."""

    print("=" * 70)
    print("IB GATEWAY AUTHENTICATION DIAGNOSTIC")
    print("=" * 70)

    ib = IB()

    try:
        print("🔄 Connecting to IB Gateway...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=996)

        print("✅ Socket connection established")

        # Check connection state
        print(f"   Is connected: {ib.isConnected()}")
        print(f"   Client ID: {ib.client.clientId}")
        print(f"   Server version: {ib.client.serverVersion()}")

        # Wait for full initialization
        print("\n🔄 Waiting for full API initialization...")
        await asyncio.sleep(5)  # Give time for full handshake

        # Check again after waiting
        print(f"   Server version (after wait): {ib.client.serverVersion()}")

        # Try to get account info
        print("\n🔄 Testing account access...")
        accounts = ib.managedAccounts()
        print(f"   Managed accounts: {accounts}")

        if not accounts:
            print("\n❌ NO MANAGED ACCOUNTS - AUTHENTICATION INCOMPLETE")
            print("\n🔧 POSSIBLE CAUSES:")
            print("1. 🔐 IB Gateway not fully logged in")
            print("   → Check Gateway window for login status")
            print("   → Look for 'Connected' or account balance display")
            print("   → Check for any error messages")

            print("\n2. 📋 API connection dialog not accepted")
            print("   → Look for popup dialogs in Gateway")
            print("   → Accept any 'API connection' requests")

            print("\n3. ⚙️  API permissions issue")
            print("   → Configure → Settings → API")
            print("   → Ensure 'Read-Only API' is UNCHECKED")
            print("   → Add 127.0.0.1 to Trusted IPs")

            print("\n4. 🏦 Account/subscription issue")
            print("   → Check your IB account status")
            print("   → Ensure paper trading is enabled")
            print("   → Verify market data subscriptions")

        else:
            print(f"✅ Authentication successful! Accounts: {accounts}")

            # Test market data access
            print("\n🔄 Testing market data access...")
            from ib_async import Stock

            contract = Stock("AAPL", "SMART", "USD")

            try:
                details = await asyncio.wait_for(ib.reqContractDetailsAsync(contract), timeout=10)
                if details:
                    print("✅ Market data access working")
                    print(f"   Contract: {details[0].contract.symbol}")
                else:
                    print("⚠️  No contract details returned")
            except Exception as e:
                print(f"❌ Market data error: {e}")

        ib.disconnect()
        return len(accounts) > 0

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        if ib.isConnected():
            ib.disconnect()
        return False


async def test_different_client_ids():
    """Test with different client IDs to avoid conflicts."""

    print("\n" + "=" * 70)
    print("TESTING DIFFERENT CLIENT IDs")
    print("=" * 70)

    client_ids = [1, 10, 100, 999, 1001]

    for client_id in client_ids:
        print(f"\n🔄 Testing client ID {client_id}...")

        ib = IB()
        try:
            await asyncio.wait_for(
                ib.connectAsync("127.0.0.1", 4002, clientId=client_id), timeout=10
            )

            # Wait for initialization
            await asyncio.sleep(3)

            accounts = ib.managedAccounts()
            server_version = ib.client.serverVersion()

            print(f"   Server version: {server_version}")
            print(f"   Accounts: {accounts}")

            if accounts and server_version > 0:
                print(f"✅ SUCCESS with client ID {client_id}!")
                ib.disconnect()
                return client_id

            ib.disconnect()

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            if ib.isConnected():
                ib.disconnect()

    return None


async def main():
    """Main diagnostic function."""

    print("Diagnosing IB Gateway authentication issues...")

    # First, try standard connection
    auth_success = await diagnose_auth_issue()

    if not auth_success:
        print("\n🔄 Trying different client IDs...")
        working_client_id = await test_different_client_ids()

        if working_client_id:
            print(f"\n✅ Found working client ID: {working_client_id}")
            print(f"Update your .env file: IBKR_CLIENT_ID={working_client_id}")
        else:
            print("\n❌ No client IDs worked - deeper issue exists")

            print("\n🔧 MANUAL STEPS TO TRY:")
            print("1. 🔄 Completely restart IB Gateway")
            print("2. 🔐 Log in again with fresh credentials")
            print("3. 👀 Watch for ANY popup dialogs and accept them")
            print("4. ⚙️  Check Configure → Settings → API settings")
            print("5. 🏦 Verify your IB account is active and funded")
            print("6. 📞 Contact IB support if issues persist")

            sys.exit(1)
    else:
        print("\n🎉 Authentication is working correctly!")
        print("Your trading system should connect successfully.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Debug TWS connection issues step by step.
"""

import asyncio
import socket
import time

from ib_async import IB


def test_socket_connection():
    """Test raw socket connection to TWS."""
    print("=" * 50)
    print("TESTING SOCKET CONNECTION")
    print("=" * 50)

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)

        print("Attempting socket connection to 127.0.0.1:7497...")
        result = sock.connect_ex(("127.0.0.1", 7497))

        if result == 0:
            print("✅ Socket connection successful")

            # Try to send a simple message
            try:
                sock.send(b"test")
                print("✅ Socket send successful")
            except Exception as e:
                print(f"⚠️  Socket send failed: {e}")

            sock.close()
            return True
        else:
            print(f"❌ Socket connection failed with code: {result}")
            return False

    except Exception as e:
        print(f"❌ Socket test failed: {e}")
        return False


async def test_api_connection_with_timeout():
    """Test API connection with detailed timeout handling."""
    print("\n" + "=" * 50)
    print("TESTING API CONNECTION WITH TIMEOUTS")
    print("=" * 50)

    ib = IB()

    try:
        print("Attempting API connection with 10s timeout...")

        # Try with very short timeout first
        start_time = time.time()

        await asyncio.wait_for(ib.connectAsync("127.0.0.1", 7497, clientId=990), timeout=10)

        connection_time = time.time() - start_time
        print(f"✅ API connection successful in {connection_time:.2f}s")

        # Test basic functionality
        server_version = ib.client.serverVersion()
        print(f"Server version: {server_version}")

        accounts = ib.managedAccounts()
        print(f"Managed accounts: {accounts}")

        ib.disconnect()
        return True

    except asyncio.TimeoutError:
        print("❌ API connection TIMEOUT")
        print("This usually means:")
        print("1. TWS API is not enabled")
        print("2. Connection dialog waiting for user acceptance")
        print("3. IP address not in trusted list")
        print("4. TWS is not fully logged in")

        if ib.isConnected():
            ib.disconnect()
        return False

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        if ib.isConnected():
            ib.disconnect()
        return False


def check_tws_process():
    """Check if TWS process is running."""
    print("\n" + "=" * 50)
    print("CHECKING TWS PROCESS")
    print("=" * 50)

    import subprocess

    try:
        # Check for TWS processes
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        tws_processes = [
            line
            for line in result.stdout.split("\n")
            if "tws" in line.lower() or "trader" in line.lower() or "workstation" in line.lower()
        ]

        if tws_processes:
            print("✅ Found TWS-related processes:")
            for proc in tws_processes:
                print(f"  {proc.strip()}")
        else:
            print("❌ No TWS processes found")

        return len(tws_processes) > 0

    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False


async def main():
    """Run all diagnostic tests."""
    print("TWS CONNECTION DIAGNOSTIC")
    print("=" * 50)

    # Test 1: Check TWS process
    process_ok = check_tws_process()

    # Test 2: Test socket connection
    socket_ok = test_socket_connection()

    # Test 3: Test API connection
    api_ok = await test_api_connection_with_timeout()

    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"TWS Process Running: {'✅' if process_ok else '❌'}")
    print(f"Socket Connection: {'✅' if socket_ok else '❌'}")
    print(f"API Connection: {'✅' if api_ok else '❌'}")

    if not api_ok:
        print("\n🔧 NEXT STEPS:")
        print("1. Check TWS window for popup dialogs")
        print("2. Go to TWS: File → Global Configuration → API → Settings")
        print("3. Ensure 'Enable ActiveX and Socket Clients' is checked")
        print("4. Add '127.0.0.1' to Trusted IP Addresses")
        print("5. Uncheck 'Read-Only API' if you want to place orders")
        print("6. Click Apply and restart TWS if needed")

    return api_ok


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

#!/usr/bin/env python3
"""Diagnose TWS connection issue."""

import asyncio
import socket

from ib_async import IB


async def test_connection():
    print("=" * 60)
    print("TWS CONNECTION DIAGNOSTIC")
    print("=" * 60)

    # 1. Check if port is open
    print("\n1. Testing TCP port 7497...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        result = sock.connect_ex(("127.0.0.1", 7497))
        if result == 0:
            print("   ✅ Port 7497 is OPEN")
        else:
            print(f"   ❌ Port 7497 is CLOSED (error: {result})")
            return
    except Exception as e:
        print(f"   ❌ Cannot connect to port: {e}")
        return
    finally:
        sock.close()

    # 2. Test API connection
    print("\n2. Testing TWS API handshake...")
    ib = IB()
    try:
        print("   Attempting connection with client_id=0...")
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", 7497, clientId=0, timeout=10), timeout=12
        )
        print("   ✅ API HANDSHAKE SUCCESS!")

        # 3. Check accounts
        print("\n3. Checking managed accounts...")
        accounts = ib.managedAccounts()
        if accounts:
            print(f"   ✅ Accounts: {accounts}")
        else:
            print("   ⚠️  No managed accounts")

    except asyncio.TimeoutError:
        print("   ❌ API HANDSHAKE TIMEOUT")
        print("\n   This means:")
        print("   - TCP connection succeeded (port is open)")
        print("   - TWS accepted the socket")
        print("   - BUT TWS did not respond with API protocol handshake")
        print("\n   SOLUTION:")
        print("   → Open TWS")
        print("   → File → Global Configuration → API → Settings")
        print("   → CHECK 'Enable ActiveX and Socket Clients'")
        print("   → Click OK")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(test_connection())

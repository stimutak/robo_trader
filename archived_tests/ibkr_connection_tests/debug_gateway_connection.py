#!/usr/bin/env python3
"""
Debug IBKR Gateway connection issues.
Tests socket vs API handshake to isolate the problem.
"""

import asyncio
import random
import socket
import sys
import time

from ib_async import IB


def test_socket_connection(host="127.0.0.1", port=4002, timeout=5):
    """Test raw socket connectivity."""
    print(f"\n=== SOCKET TEST ===")
    print(f"Testing socket connection to {host}:{port}")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        start_time = time.time()
        result = sock.connect_ex((host, port))
        connect_time = time.time() - start_time
        sock.close()

        if result == 0:
            print(f"✅ Socket connection SUCCESS in {connect_time:.3f}s")
            return True
        else:
            print(f"❌ Socket connection FAILED: error code {result}")
            return False
    except Exception as e:
        print(f"❌ Socket connection FAILED: {e}")
        return False


async def test_api_handshake(host="127.0.0.1", port=4002, client_id=None, timeout=15):
    """Test IBKR API handshake."""
    print(f"\n=== API HANDSHAKE TEST ===")

    if client_id is None:
        client_id = random.randint(1000, 9999)

    print(f"Testing API handshake to {host}:{port} with client_id={client_id}")

    ib = IB()
    try:
        start_time = time.time()
        print(f"Starting handshake at {time.strftime('%H:%M:%S')}")

        # Try connection with detailed timeout
        await asyncio.wait_for(
            ib.connectAsync(
                host=host,
                port=port,
                clientId=client_id,
                timeout=timeout,
                readonly=True,  # Use readonly to avoid TWS dialogs
            ),
            timeout=timeout + 2,
        )

        connect_time = time.time() - start_time
        print(f"✅ API handshake SUCCESS in {connect_time:.3f}s")

        # Test basic functionality
        accounts = ib.managedAccounts()
        print(f"✅ Managed accounts: {accounts}")

        # Test server version
        print(f"✅ Server version: {ib.serverVersion()}")

        return True

    except asyncio.TimeoutError:
        connect_time = time.time() - start_time
        print(f"❌ API handshake TIMEOUT after {connect_time:.3f}s")
        return False
    except Exception as e:
        connect_time = time.time() - start_time
        print(f"❌ API handshake FAILED after {connect_time:.3f}s: {e}")
        return False
    finally:
        if ib.isConnected():
            ib.disconnect()


def check_gateway_process():
    """Check if Gateway process is running."""
    print(f"\n=== GATEWAY PROCESS CHECK ===")
    import subprocess

    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)

        gateway_lines = [
            line
            for line in result.stdout.split("\n")
            if "gateway" in line.lower() or "ibgateway" in line.lower()
        ]

        if gateway_lines:
            print("✅ Gateway process found:")
            for line in gateway_lines:
                print(f"  {line}")
            return True
        else:
            print("❌ No Gateway process found")
            return False

    except Exception as e:
        print(f"❌ Process check failed: {e}")
        return False


def check_port_status(port=4002):
    """Check port listening status."""
    print(f"\n=== PORT STATUS CHECK ===")
    import subprocess

    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)

        port_lines = [line for line in result.stdout.split("\n") if str(port) in line]

        if port_lines:
            print(f"✅ Port {port} status:")
            for line in port_lines:
                print(f"  {line}")
            return True
        else:
            print(f"❌ Port {port} not found in netstat")
            return False

    except Exception as e:
        print(f"❌ Port check failed: {e}")
        return False


async def main():
    """Run comprehensive Gateway connection diagnostics."""
    print("🔍 IBKR Gateway Connection Diagnostics")
    print("=" * 50)

    # Check Gateway process
    gateway_running = check_gateway_process()

    # Check port status
    port_listening = check_port_status()

    # Test socket connection
    socket_ok = test_socket_connection()

    # Test API handshake with different client IDs
    api_results = []
    for i, client_id in enumerate([0, 1, 100, random.randint(1000, 9999)]):
        print(f"\n--- API Test {i+1}/4 ---")
        result = await test_api_handshake(client_id=client_id)
        api_results.append((client_id, result))

        if result:
            break  # Success, no need to test more

        await asyncio.sleep(1)  # Brief pause between attempts

    # Summary
    print(f"\n{'='*50}")
    print("🔍 DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")
    print(f"Gateway Process: {'✅' if gateway_running else '❌'}")
    print(f"Port Listening:  {'✅' if port_listening else '❌'}")
    print(f"Socket Connect:  {'✅' if socket_ok else '❌'}")

    print(f"\nAPI Handshake Results:")
    for client_id, success in api_results:
        print(f"  Client ID {client_id}: {'✅' if success else '❌'}")

    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    if not gateway_running:
        print("❌ Start IB Gateway")
    elif not port_listening:
        print("❌ Check Gateway port configuration (should be 4002 for paper)")
    elif not socket_ok:
        print("❌ Check firewall/network settings")
    elif not any(result for _, result in api_results):
        print("❌ Gateway API settings issue:")
        print("   1. Open Gateway configuration")
        print("   2. Go to API settings")
        print("   3. Enable 'Enable ActiveX and Socket Clients'")
        print("   4. Add 127.0.0.1 to Trusted IPs")
        print("   5. Set Socket port to 4002")
        print("   6. Restart Gateway")
    else:
        print("✅ Connection working!")


if __name__ == "__main__":
    asyncio.run(main())

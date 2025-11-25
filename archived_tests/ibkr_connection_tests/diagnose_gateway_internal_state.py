#!/usr/bin/env python3
"""
Gateway Internal State Diagnostic Tool

This script performs deep diagnostics on Gateway's internal API state
to identify why handshakes timeout even though TCP connections succeed.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


async def test_tcp_connection():
    """Test raw TCP connection to Gateway port."""
    print("=" * 60)
    print("TEST 1: Raw TCP Connection")
    print("=" * 60)

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 4002), timeout=5.0
        )

        print("✅ TCP connection successful")

        # Try to send some data and see if Gateway responds
        writer.write(b"API\0")
        await writer.drain()

        try:
            data = await asyncio.wait_for(reader.read(100), timeout=2.0)
            print(f"✅ Gateway responded with: {data}")
        except asyncio.TimeoutError:
            print("⚠️ Gateway accepted connection but didn't respond to API probe")

        writer.close()
        await writer.wait_closed()
        return True

    except Exception as e:
        print(f"❌ TCP connection failed: {e}")
        return False


async def test_ib_async_handshake_steps():
    """Test IB API handshake step by step to identify failure point."""
    print("\n" + "=" * 60)
    print("TEST 2: IB API Handshake Step Analysis")
    print("=" * 60)

    try:
        from ib_async import IB
        from ib_async.client import Client

        # Create IB instance with detailed logging
        ib = IB()

        # Hook into client events to see exactly where it fails
        client = ib.client

        handshake_events = []

        def log_event(event_name):
            timestamp = time.time()
            handshake_events.append((timestamp, event_name))
            print(f"[{timestamp:.3f}] {event_name}")

        # Hook key client methods
        original_connect = client.connect
        original_startApi = client.startApi

        def hooked_connect(*args, **kwargs):
            log_event("CLIENT: connect() called")
            return original_connect(*args, **kwargs)

        def hooked_startApi(*args, **kwargs):
            log_event("CLIENT: startApi() called")
            return original_startApi(*args, **kwargs)

        client.connect = hooked_connect
        client.startApi = hooked_startApi

        print("Starting detailed handshake analysis...")
        start_time = time.time()

        try:
            log_event("STARTING: connectAsync()")
            await asyncio.wait_for(
                ib.connectAsync("127.0.0.1", 4002, clientId=777, timeout=10), timeout=15.0
            )

            duration = time.time() - start_time
            log_event(f"SUCCESS: Connected in {duration:.2f}s")

            # Test account data retrieval
            log_event("TESTING: managedAccounts()")
            accounts = ib.managedAccounts()
            log_event(f"ACCOUNTS: {accounts}")

            ib.disconnect()
            return True, handshake_events

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            log_event(f"TIMEOUT: After {duration:.2f}s")
            return False, handshake_events

        except Exception as e:
            duration = time.time() - start_time
            log_event(f"ERROR: {e} after {duration:.2f}s")
            return False, handshake_events

    except ImportError:
        print("❌ ib_async not available")
        return False, []


async def test_multiple_client_ids():
    """Test if specific client IDs work better than others."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple Client ID Test")
    print("=" * 60)

    try:
        from ib_async import IB

        client_ids = [1, 100, 500, 999, 1234, 9999]
        results = {}

        for client_id in client_ids:
            print(f"\nTesting client ID {client_id}...")
            ib = IB()

            start_time = time.time()
            try:
                await asyncio.wait_for(
                    ib.connectAsync("127.0.0.1", 4002, clientId=client_id, timeout=5), timeout=8.0
                )

                duration = time.time() - start_time
                accounts = ib.managedAccounts()

                results[client_id] = {
                    "success": True,
                    "duration": duration,
                    "accounts": len(accounts),
                }

                print(f"✅ Client ID {client_id}: {duration:.2f}s, {len(accounts)} accounts")
                ib.disconnect()

            except Exception as e:
                duration = time.time() - start_time
                results[client_id] = {"success": False, "duration": duration, "error": str(e)}
                print(f"❌ Client ID {client_id}: Failed after {duration:.2f}s - {e}")

            # Small delay between attempts
            await asyncio.sleep(1)

        return results

    except ImportError:
        print("❌ ib_async not available")
        return {}


async def check_gateway_process_health():
    """Check Gateway process health and resource usage."""
    print("\n" + "=" * 60)
    print("TEST 4: Gateway Process Health")
    print("=" * 60)

    try:
        # Get Gateway process info
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)

        gateway_lines = [
            line
            for line in result.stdout.split("\n")
            if "JavaAppli" in line and ("gateway" in line.lower() or "tws" in line.lower())
        ]

        if not gateway_lines:
            print("❌ No Gateway process found")
            return False

        for line in gateway_lines:
            parts = line.split()
            if len(parts) >= 11:
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                time_used = parts[9]

                print(f"Gateway Process:")
                print(f"  PID: {pid}")
                print(f"  CPU: {cpu}%")
                print(f"  Memory: {mem}%")
                print(f"  Time: {time_used}")

        # Check file descriptors
        try:
            fd_result = subprocess.run(
                ["lsof", "-p", pid], capture_output=True, text=True, timeout=5
            )
            fd_count = len(fd_result.stdout.split("\n")) - 1
            print(f"  Open FDs: {fd_count}")

            # Check specifically for port 4002 connections
            port_result = subprocess.run(
                ["lsof", "-nP", "-iTCP:4002"], capture_output=True, text=True, timeout=5
            )
            port_lines = [line for line in port_result.stdout.split("\n") if line.strip()]
            print(f"  Port 4002 connections: {len(port_lines) - 1}")

            for line in port_lines[1:]:  # Skip header
                print(f"    {line}")

        except Exception as e:
            print(f"  ⚠️ Could not check file descriptors: {e}")

        return True

    except Exception as e:
        print(f"❌ Process health check failed: {e}")
        return False


async def main():
    """Run comprehensive Gateway diagnostics."""
    print("GATEWAY API HANDSHAKE TIMEOUT DIAGNOSTIC")
    print("=" * 60)
    print("This tool diagnoses why Gateway accepts TCP connections")
    print("but API handshakes consistently timeout after 15-30 seconds.")
    print()

    # Test 1: Raw TCP
    tcp_ok = await test_tcp_connection()

    # Test 2: IB API handshake analysis
    if tcp_ok:
        handshake_ok, events = await test_ib_async_handshake_steps()

        print("\nHandshake Event Timeline:")
        if events:
            start_time = events[0][0]
            for timestamp, event in events:
                relative_time = timestamp - start_time
                print(f"  +{relative_time:.3f}s: {event}")

    # Test 3: Multiple client IDs
    client_results = await test_multiple_client_ids()

    # Test 4: Gateway process health
    await check_gateway_process_health()

    # Summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    if tcp_ok:
        print("✅ TCP Layer: Working")
    else:
        print("❌ TCP Layer: Failed")

    if "handshake_ok" in locals() and handshake_ok:
        print("✅ IB API Layer: Working")
    else:
        print("❌ IB API Layer: Handshake timeout")

    if client_results:
        successful_clients = [cid for cid, result in client_results.items() if result["success"]]
        if successful_clients:
            print(f"✅ Working Client IDs: {successful_clients}")
        else:
            print("❌ No client IDs work")

    print("\nNext steps based on results:")
    if not tcp_ok:
        print("1. Check Gateway is running and listening on port 4002")
    elif "handshake_ok" in locals() and not handshake_ok:
        print("1. Gateway API layer is unresponsive - needs internal restart")
        print("2. Try force-killing Gateway process and restarting")
        print("3. Check for Gateway version/library compatibility issues")
    else:
        print("1. Diagnostics inconclusive - manual investigation needed")


if __name__ == "__main__":
    asyncio.run(main())

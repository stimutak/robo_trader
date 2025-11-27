#!/usr/bin/env python3
"""
Test script to verify the Gateway connection fix (2025-11-27).

⚠️  CRITICAL WARNING ⚠️
=======================
DO NOT run this script immediately before starting the trader!

Even with proper cleanup, this test creates Gateway connections that may
leave brief zombie states. Running this and then immediately starting
./START_TRADER.sh will likely fail due to CLOSE_WAIT zombies.

SAFE USAGE:
- Run this ONLY to diagnose connection problems AFTER startup fails
- If you run this test, wait 30+ seconds before starting the trader
- OR restart Gateway (File → Exit, relaunch with 2FA) after testing

For quick connectivity checks without zombie risk, use:
    ./force_gateway_reconnect.sh

=======================

This script tests the critical fixes made to the subprocess worker:
1. Removed blocking waitOnUpdate() call
2. Added serverVersion() check for API handshake verification
3. Proper async polling for account data

IMPORTANT: This test uses IBKR_FORCE_DISCONNECT=1 to properly clean up
connections and avoid creating zombies.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))

# CRITICAL: Enable force disconnect to avoid creating zombies
os.environ["IBKR_FORCE_DISCONNECT"] = "1"


async def test_direct_minimal_connection():
    """Test direct ib_async connection without subprocess (baseline)."""
    print("=" * 60)
    print("TEST 1: Direct ib_async Connection (Baseline)")
    print("=" * 60)

    ib = None
    try:
        from ib_async import IB
        from robo_trader.utils.ibkr_safe import safe_disconnect

        ib = IB()
        start_time = time.time()

        print("Connecting directly with ib_async...")
        await ib.connectAsync("127.0.0.1", 4002, clientId=888, readonly=True, timeout=15)

        connect_time = time.time() - start_time
        print(f"connectAsync() returned in {connect_time:.2f}s")

        # Check serverVersion
        server_version = ib.client.serverVersion()
        print(f"Server version: {server_version}")

        # Check accounts
        accounts = ib.managedAccounts()
        print(f"Accounts: {accounts}")

        total_time = time.time() - start_time

        if server_version and server_version > 0 and accounts:
            print(f"PASS: Direct connection successful in {total_time:.2f}s")
            return True
        else:
            print(f"FAIL: Connection incomplete (serverVersion={server_version}, accounts={accounts})")
            return False

    except Exception as e:
        print(f"FAIL: Direct connection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # CRITICAL: Use safe_disconnect with force flag to avoid zombies
        if ib is not None:
            try:
                from robo_trader.utils.ibkr_safe import safe_disconnect
                safe_disconnect(ib, context="test_direct_minimal_connection")
                print("Disconnected cleanly via safe_disconnect")
            except Exception as e:
                print(f"Warning: Disconnect failed: {e}")


async def test_worker_process_direct():
    """Test worker process directly with JSON command."""
    print("\n" + "=" * 60)
    print("TEST 2: Subprocess Worker Direct Test")
    print("=" * 60)

    test_cmd = {
        "command": "connect",
        "params": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 889,
            "readonly": True,
            "timeout": 30.0
        }
    }

    print(f"Sending command: {json.dumps(test_cmd)}")
    start_time = time.time()

    try:
        # Run worker directly
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "robo_trader.clients.ibkr_subprocess_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent
        )

        # Send command with timeout
        cmd_json = json.dumps(test_cmd) + "\n"
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(cmd_json.encode()),
                timeout=90.0  # Generous timeout for full handshake
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            print(f"FAIL: Worker timed out after 90s")
            return False

        duration = time.time() - start_time

        print(f"\nWorker completed in {duration:.2f}s")
        print(f"Return code: {process.returncode}")

        # Print stderr (debug output)
        if stderr:
            print("\n--- Worker Debug Output (stderr) ---")
            for line in stderr.decode().strip().split('\n'):
                print(f"  {line}")
            print("--- End Debug Output ---\n")

        # Parse response
        if stdout:
            stdout_text = stdout.decode().strip()

            # Filter out any non-JSON lines (e.g., ib_async log messages)
            json_lines = [line for line in stdout_text.split('\n')
                         if line.strip().startswith('{"status":')]

            if not json_lines:
                print(f"FAIL: No valid JSON response found")
                print(f"Raw stdout: {stdout_text}")
                return False

            response_text = json_lines[-1]  # Take last valid response

            try:
                response = json.loads(response_text)

                if response.get("status") == "success":
                    data = response.get("data", {})
                    connected = data.get("connected", False)
                    accounts = data.get("accounts", [])
                    server_version = data.get("server_version")

                    print(f"Response: connected={connected}, accounts={accounts}, serverVersion={server_version}")

                    if connected and accounts and server_version:
                        print(f"PASS: Worker connection successful in {duration:.2f}s")
                        return True
                    else:
                        print(f"FAIL: Incomplete connection data")
                        return False
                else:
                    error = response.get("error", "Unknown error")
                    error_type = response.get("error_type", "Unknown")
                    print(f"FAIL: Worker returned error: {error_type}: {error}")
                    return False

            except json.JSONDecodeError as e:
                print(f"FAIL: Invalid JSON response: {e}")
                print(f"Raw response: {response_text}")
                return False
        else:
            print("FAIL: No stdout from worker")
            return False

    except Exception as e:
        print(f"FAIL: Worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_subprocess_client():
    """Test full SubprocessIBKRClient."""
    print("\n" + "=" * 60)
    print("TEST 3: SubprocessIBKRClient Full Test")
    print("=" * 60)

    try:
        from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient

        client = SubprocessIBKRClient()

        print("Starting subprocess client...")
        await client.start()

        print("Connecting via client...")
        start_time = time.time()

        connected = await client.connect(
            host="127.0.0.1",
            port=4002,
            client_id=890,
            readonly=True,
            timeout=30.0
        )

        duration = time.time() - start_time
        print(f"Connection attempt completed in {duration:.2f}s")

        if connected:
            # Test getting accounts
            accounts = await client.get_accounts()
            print(f"Accounts: {accounts}")

            # Test ping
            ping_ok = await client.ping()
            print(f"Ping: {ping_ok}")

            print(f"PASS: SubprocessIBKRClient connection successful in {duration:.2f}s")
            return True
        else:
            print("FAIL: SubprocessIBKRClient connection failed")
            return False

    except Exception as e:
        print(f"FAIL: SubprocessIBKRClient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def check_zombies():
    """Check for zombie connections before testing."""
    print("=" * 60)
    print("PRE-CHECK: Zombie Connection Detection")
    print("=" * 60)

    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:4002", "-sTCP:CLOSE_WAIT"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.stdout.strip():
            lines = [l for l in result.stdout.split('\n') if l.strip() and not l.startswith('COMMAND')]
            if lines:
                print(f"WARNING: Found {len(lines)} zombie connection(s)!")
                for line in lines:
                    print(f"  {line}")
                print("\nZombie connections may cause connection failures.")
                print("Restart Gateway (File->Exit, relaunch with 2FA) to clear.")
                return False

        print("OK: No zombie connections detected")
        return True

    except Exception as e:
        print(f"Could not check for zombies: {e}")
        return True  # Proceed anyway


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("⚠️  CRITICAL WARNING ⚠️")
    print("=" * 60)
    print("")
    print("DO NOT run this test immediately before starting the trader!")
    print("")
    print("This test creates Gateway connections that may leave zombies.")
    print("After running, either:")
    print("  - Wait 30+ seconds before ./START_TRADER.sh")
    print("  - OR restart Gateway (File→Exit, relaunch with 2FA)")
    print("")
    print("For quick checks, use: ./force_gateway_reconnect.sh")
    print("")
    print("=" * 60)
    print("GATEWAY CONNECTION FIX VERIFICATION")
    print("=" * 60)
    print("\nThis test verifies the 2025-11-27 fix for:")
    print("1. Blocking waitOnUpdate() removed")
    print("2. serverVersion() check for handshake verification")
    print("3. Proper async polling for account data")
    print("")

    # Pre-check for zombies
    zombies_clean = await check_zombies()
    if not zombies_clean:
        print("\nWARNING: Zombie connections detected. Tests may fail.")
        print("Consider restarting Gateway before running tests.\n")

    print("")

    # Run tests with delays between to avoid connection conflicts
    results = {}

    # Test 1: Direct connection
    results["direct"] = await test_direct_minimal_connection()
    await asyncio.sleep(2)  # Wait between tests

    # Test 2: Worker direct
    results["worker"] = await test_worker_process_direct()
    await asyncio.sleep(2)

    # Test 3: Full client
    results["client"] = await test_subprocess_client()

    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Direct ib_async:      {'PASS' if results['direct'] else 'FAIL'}")
    print(f"Subprocess Worker:    {'PASS' if results['worker'] else 'FAIL'}")
    print(f"SubprocessIBKRClient: {'PASS' if results['client'] else 'FAIL'}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("The Gateway connection fix is working correctly.")
        print("=" * 60)
        print("")
        print("⚠️  REMINDER: Wait 30+ seconds before running ./START_TRADER.sh")
        print("   or restart Gateway to avoid zombie connections.")
        print("")
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)

        if not results["direct"]:
            print("\nDirect connection failed - Gateway may not be running or configured.")
            print("Check: Gateway running, API enabled, port 4002 open")

        if results["direct"] and not results["worker"]:
            print("\nWorker failed but direct works - check worker process/module path")

        if results["worker"] and not results["client"]:
            print("\nClient failed but worker works - check client timeout/error handling")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

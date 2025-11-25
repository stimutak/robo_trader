#!/usr/bin/env python3
"""
Test script for subprocess worker connection fix.

This script validates the timing and zombie connection fixes implemented
to resolve the subprocess worker connection failure issue.
"""

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient
from robo_trader.logger import get_logger

logger = get_logger(__name__)


async def test_zombie_detection():
    """Test zombie connection detection logic."""
    print("=" * 60)
    print("TEST 1: Zombie Connection Detection")
    print("=" * 60)

    client = SubprocessIBKRClient()

    # Test zombie detection on port 4002
    zombie_count, message = await client._check_zombie_connections(4002)

    print(f"Zombie count: {zombie_count}")
    print(f"Message: {message}")

    if zombie_count > 0:
        print("❌ ZOMBIES DETECTED - Connection will be blocked")
        print("   Solution: Restart Gateway (File→Exit, relaunch with 2FA)")
        return False
    else:
        print("✅ No zombies detected - Connection should work")
        return True


async def test_direct_worker():
    """Test direct worker connection with timing analysis."""
    print("\n" + "=" * 60)
    print("TEST 2: Direct Worker Connection Test")
    print("=" * 60)

    # Test command
    test_cmd = {
        "command": "connect",
        "params": {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 998,
            "readonly": True,
            "timeout": 30.0,
        },
    }

    print("Testing direct worker connection...")
    print(f"Command: {json.dumps(test_cmd, indent=2)}")

    start_time = time.time()

    try:
        # Run worker directly
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "robo_trader.clients.ibkr_subprocess_worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent,
        )

        # Send command
        cmd_json = json.dumps(test_cmd) + "\n"
        stdout, stderr = await process.communicate(cmd_json.encode())

        duration = time.time() - start_time

        print(f"Duration: {duration:.2f}s")
        print(f"Return code: {process.returncode}")

        if stdout:
            raw_stdout = stdout.decode()
            lines = [line.strip() for line in raw_stdout.splitlines() if line.strip()]
            response_line = next(
                (line for line in reversed(lines) if line.startswith('{"status":')), None
            )
            if response_line:
                try:
                    response = json.loads(response_line)
                    print(f"Response: {json.dumps(response, indent=2)}")
                    if response.get("status") == "success":
                        data = response.get("data", {})
                        connected = data.get("connected", False)
                        accounts = data.get("accounts", [])
                        print(f"✅ Connection successful: {connected}, Accounts: {accounts}")
                        return True
                    error = response.get("error", "Unknown error")
                    print(f"❌ Connection failed: {error}")
                    return False
                except json.JSONDecodeError as e:
                    print(f"❌ Invalid JSON response: {e}")
            else:
                print("❌ Worker stdout did not contain a JSON status payload.")

            print("Raw stdout:")
            print(raw_stdout)
            return False

        if stderr:
            print("Worker stderr output:")
            print(stderr.decode())

        return process.returncode == 0

    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Test failed after {duration:.2f}s: {e}")
        return False


async def test_subprocess_client():
    """Test full subprocess client connection."""
    print("\n" + "=" * 60)
    print("TEST 3: Subprocess Client Connection Test")
    print("=" * 60)

    client = SubprocessIBKRClient()

    try:
        print("Starting subprocess client...")
        await client.start()

        print("Attempting connection...")
        start_time = time.time()

        connected = await client.connect(
            host="127.0.0.1", port=4002, client_id=997, readonly=True, timeout=30.0
        )

        duration = time.time() - start_time
        print(f"Connection attempt completed in {duration:.2f}s")

        if connected:
            print("✅ Subprocess client connection successful")

            # Test getting accounts
            accounts = await client.get_accounts()
            print(f"Accounts: {accounts}")

            return True
        else:
            print("❌ Subprocess client connection failed")
            return False

    except Exception as e:
        print(f"❌ Subprocess client test failed: {e}")
        return False
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def main():
    """Run all tests."""
    print("SUBPROCESS WORKER CONNECTION FIX VALIDATION")
    print("=" * 60)

    # Test 1: Check for zombies
    zombies_clean = await test_zombie_detection()

    if not zombies_clean:
        print("\n❌ CRITICAL: Zombie connections detected!")
        print("   Gateway restart required before connection tests will work.")
        print("   Please restart Gateway (File→Exit, relaunch with 2FA) and retry.")
        return False

    # Test 2: Direct worker test
    direct_works = await test_direct_worker()

    # Test 3: Full subprocess client test
    client_works = await test_subprocess_client()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Zombie Detection: {'✅ PASS' if zombies_clean else '❌ FAIL'}")
    print(f"Direct Worker:    {'✅ PASS' if direct_works else '❌ FAIL'}")
    print(f"Subprocess Client: {'✅ PASS' if client_works else '❌ FAIL'}")

    all_passed = zombies_clean and direct_works and client_works

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Subprocess connection fix is working!")
    else:
        print("\n❌ SOME TESTS FAILED - Further investigation needed")

    return all_passed


if __name__ == "__main__":
    asyncio.run(main())

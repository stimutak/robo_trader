#!/usr/bin/env python3
"""
Alternative Connection Methods Test

This script tests different approaches to connecting to Gateway
that haven't been tried based on handoff document analysis.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.logger import get_logger

logger = get_logger(__name__)


async def test_sync_connection_in_thread():
    """Test sync connection in thread (not executor)."""
    print("=" * 60)
    print("TEST 1: Sync Connection in Thread")
    print("=" * 60)

    try:
        import threading

        from ib_async import IB

        result = {"success": False, "error": None, "accounts": []}

        def sync_connect():
            try:
                ib = IB()
                print("Thread: Starting sync connection...")

                # Use sync connect (not connectAsync)
                ib.connect("127.0.0.1", 4002, clientId=888, readonly=True, timeout=15)
                print("Thread: Connected!")

                # Wait for account data with explicit event processing
                for i in range(30):  # 15 seconds
                    accounts = ib.managedAccounts()
                    if accounts:
                        result["accounts"] = accounts
                        result["success"] = True
                        break

                    # Process events
                    ib.waitOnUpdate(timeout=0.5)

                if not result["success"]:
                    result["error"] = "No accounts received after 15s"

                ib.disconnect()

            except Exception as e:
                result["error"] = str(e)

        # Run in thread
        thread = threading.Thread(target=sync_connect)
        thread.start()
        thread.join(timeout=20)  # 20 second timeout

        if thread.is_alive():
            result["error"] = "Thread timeout"

        if result["success"]:
            print(f"✅ Sync thread connection successful: {result['accounts']}")
            return True
        else:
            print(f"❌ Sync thread connection failed: {result['error']}")
            return False

    except Exception as e:
        print(f"❌ Thread test failed: {e}")
        return False


async def test_minimal_async_connection():
    """Test minimal async connection without any subprocess complexity."""
    print("\n" + "=" * 60)
    print("TEST 2: Minimal Async Connection")
    print("=" * 60)

    try:
        from ib_async import IB

        ib = IB()

        print("Starting minimal async connection...")
        start_time = time.time()

        # Minimal connectAsync call
        await ib.connectAsync("127.0.0.1", 4002, clientId=889, readonly=True, timeout=15)

        connect_time = time.time() - start_time
        print(f"Connected in {connect_time:.2f}s")

        # Wait for accounts with event processing
        print("Waiting for account data...")
        accounts = []

        for i in range(20):  # 10 seconds
            accounts = ib.managedAccounts()
            if accounts:
                break

            try:
                ib.waitOnUpdate(timeout=0.5)
            except Exception:
                await asyncio.sleep(0.5)

        total_time = time.time() - start_time

        if accounts:
            print(f"✅ Minimal async connection successful in {total_time:.2f}s: {accounts}")
            ib.disconnect()
            return True
        else:
            print(f"❌ No accounts received after {total_time:.2f}s")
            ib.disconnect()
            return False

    except Exception as e:
        print(f"❌ Minimal async connection failed: {e}")
        return False


async def test_connection_with_different_timeouts():
    """Test connections with various timeout values."""
    print("\n" + "=" * 60)
    print("TEST 3: Different Timeout Values")
    print("=" * 60)

    timeouts = [5, 10, 30, 60]
    results = {}

    for timeout in timeouts:
        print(f"\nTesting with {timeout}s timeout...")

        try:
            from ib_async import IB

            ib = IB()
            start_time = time.time()

            await asyncio.wait_for(
                ib.connectAsync(
                    "127.0.0.1", 4002, clientId=890 + timeout, readonly=True, timeout=timeout
                ),
                timeout=timeout + 5,  # asyncio timeout slightly longer
            )

            duration = time.time() - start_time
            accounts = ib.managedAccounts()

            results[timeout] = {"success": True, "duration": duration, "accounts": len(accounts)}

            print(f"✅ {timeout}s timeout: Success in {duration:.2f}s, {len(accounts)} accounts")
            ib.disconnect()

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            results[timeout] = {"success": False, "duration": duration, "error": "asyncio timeout"}
            print(f"❌ {timeout}s timeout: asyncio timeout after {duration:.2f}s")

        except Exception as e:
            duration = time.time() - start_time
            results[timeout] = {"success": False, "duration": duration, "error": str(e)}
            print(f"❌ {timeout}s timeout: {e} after {duration:.2f}s")

        # Delay between attempts
        await asyncio.sleep(2)

    return results


async def test_connection_without_readonly():
    """Test connection without readonly flag."""
    print("\n" + "=" * 60)
    print("TEST 4: Connection Without Readonly Flag")
    print("=" * 60)

    try:
        from ib_async import IB

        ib = IB()

        print("Testing connection without readonly=True...")
        start_time = time.time()

        # Connect without readonly flag
        await ib.connectAsync("127.0.0.1", 4002, clientId=895, timeout=15)

        duration = time.time() - start_time
        accounts = ib.managedAccounts()

        if accounts:
            print(f"✅ Non-readonly connection successful in {duration:.2f}s: {accounts}")
            ib.disconnect()
            return True
        else:
            print(f"❌ Non-readonly connection: no accounts after {duration:.2f}s")
            ib.disconnect()
            return False

    except Exception as e:
        print(f"❌ Non-readonly connection failed: {e}")
        return False


async def main():
    """Run all alternative connection tests."""
    print("ALTERNATIVE CONNECTION METHODS TEST")
    print("=" * 60)
    print("Testing different connection approaches that haven't been tried")
    print("based on handoff document analysis.")
    print()

    results = {}

    # Test 1: Sync in thread
    results["sync_thread"] = await test_sync_connection_in_thread()

    # Test 2: Minimal async
    results["minimal_async"] = await test_minimal_async_connection()

    # Test 3: Different timeouts
    timeout_results = await test_connection_with_different_timeouts()
    results["timeouts"] = timeout_results

    # Test 4: Without readonly
    results["non_readonly"] = await test_connection_without_readonly()

    # Summary
    print("\n" + "=" * 60)
    print("ALTERNATIVE CONNECTION TEST SUMMARY")
    print("=" * 60)

    working_methods = []

    if results["sync_thread"]:
        working_methods.append("Sync connection in thread")

    if results["minimal_async"]:
        working_methods.append("Minimal async connection")

    if results["non_readonly"]:
        working_methods.append("Non-readonly connection")

    # Check timeout results
    if "timeouts" in results:
        successful_timeouts = [t for t, r in results["timeouts"].items() if r["success"]]
        if successful_timeouts:
            working_methods.append(f"Timeouts: {successful_timeouts}")

    if working_methods:
        print("✅ Working connection methods found:")
        for method in working_methods:
            print(f"  - {method}")
        print("\nRecommendation: Use the working method in production")
    else:
        print("❌ No alternative connection methods work")
        print("\nThis confirms Gateway API layer is fundamentally unresponsive")
        print("Recommendation: Force restart Gateway process")

    print("\nNext steps:")
    if working_methods:
        print("1. Implement working connection method in subprocess worker")
        print("2. Test with full trading system")
    else:
        print("1. Run: ./force_gateway_restart.sh")
        print("2. After restart: python3 diagnose_gateway_internal_state.py")
        print("3. If still failing: Check Gateway version compatibility")


if __name__ == "__main__":
    asyncio.run(main())

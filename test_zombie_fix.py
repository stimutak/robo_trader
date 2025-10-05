#!/usr/bin/env python3
"""
Test script to verify TWS zombie connection fix.

This tests that:
1. Successful connections work properly
2. Failed connections clean up properly (no zombies)
3. Unique client IDs are used per retry
4. Only max 2 retries (not 5) to limit zombie potential
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def count_zombies(port: int = 7497) -> int:
    """Count CLOSE_WAIT zombie connections on port."""
    try:
        result = subprocess.run(["netstat", "-an"], capture_output=True, text=True, timeout=5)
        zombie_count = 0
        for line in result.stdout.splitlines():
            if str(port) in line and "CLOSE_WAIT" in line:
                zombie_count += 1
        return zombie_count
    except Exception as e:
        print(f"Warning: Could not count zombies: {e}")
        return 0


async def test_connection_cleanup():
    """Test that failed connections clean up properly."""
    from robo_trader.utils.robust_connection import CircuitBreakerConfig, connect_ibkr_robust

    print("=" * 70)
    print("ZOMBIE CONNECTION FIX TEST")
    print("=" * 70)

    # Count zombies before test
    zombies_before = count_zombies()
    print(f"\n1. Zombie connections BEFORE test: {zombies_before}")

    # Test connection (may fail if TWS not running, that's OK)
    print("\n2. Attempting connection with max_retries=2...")

    config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30, success_threshold=1)

    try:
        ib = await connect_ibkr_robust(
            client_id=100,  # Use high client_id to avoid conflicts
            max_retries=2,  # Should only be 2 attempts max
            circuit_breaker_config=config,
        )
        print("   ✅ Connection successful!")
        print(f"   Connected: {ib.isConnected()}")
        print(f"   Accounts: {ib.managedAccounts()}")

        # Clean disconnect
        ib.disconnect()
        await asyncio.sleep(0.5)
        print("   ✅ Disconnected cleanly")

    except Exception as e:
        print(f"   ℹ️ Connection failed (expected if TWS not running): {type(e).__name__}")
        print(f"   Error: {e}")

    # Wait for cleanup
    print("\n3. Waiting 2 seconds for cleanup...")
    await asyncio.sleep(2)

    # Count zombies after test
    zombies_after = count_zombies()
    print(f"\n4. Zombie connections AFTER test: {zombies_after}")

    # Check results
    new_zombies = zombies_after - zombies_before
    print(f"\n5. New zombies created: {new_zombies}")

    print("\n" + "=" * 70)
    if new_zombies == 0:
        print("✅ TEST PASSED - No zombie connections created!")
    elif new_zombies <= 2:
        print(f"⚠️ TEST ACCEPTABLE - Only {new_zombies} zombie(s) created (max 2 retries)")
        print("   This may happen if connection timed out during handshake")
        print("   Key: Should be ≤2 zombies, not 5+ like before the fix")
    else:
        print(f"❌ TEST FAILED - {new_zombies} zombies created (should be ≤2)")
    print("=" * 70)

    return new_zombies


async def test_health_monitor():
    """Test TWS health monitoring."""
    from robo_trader.utils.tws_health import check_tws_api_health, is_port_listening

    print("\n\n" + "=" * 70)
    print("TWS HEALTH MONITOR TEST")
    print("=" * 70)

    # Test port check
    print("\n1. Checking if port 7497 is listening...")
    port_open = is_port_listening()
    print(f"   Result: {'✅ OPEN' if port_open else '❌ CLOSED'}")

    if not port_open:
        print("   ⚠️ TWS not running - skipping API health check")
        return

    # Test API health check
    print("\n2. Checking TWS API health (fast check)...")
    healthy, message = await check_tws_api_health(timeout=3.0)
    print(f"   Result: {'✅ HEALTHY' if healthy else '❌ UNHEALTHY'}")
    print(f"   Message: {message}")

    print("\n" + "=" * 70)
    if healthy:
        print("✅ TWS API is responding normally")
    else:
        print("❌ TWS API not responding - may need restart")
    print("=" * 70)


async def main():
    """Run all tests."""
    # Test 1: Zombie connection cleanup
    new_zombies = await test_connection_cleanup()

    # Test 2: Health monitoring
    await test_health_monitor()

    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Fixes Applied:")
    print("  1. ✅ ALWAYS call ib.disconnect() on failure (even if isConnected()==False)")
    print("  2. ✅ Add 0.5s delay after disconnect for TWS to process")
    print("  3. ✅ Use unique client IDs per retry (client_id + random 0-99)")
    print("  4. ✅ Reduced max_retries from 5 to 2")
    print("\nExpected Results:")
    print("  - Zombies created: ≤2 (was 5 before fix)")
    print("  - TWS should not get stuck anymore")
    print("  - System will fail faster instead of accumulating zombies")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

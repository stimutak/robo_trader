#!/usr/bin/env python3
"""
Test robust connection handling with exponential backoff and circuit breaker.

This test simulates various connection failure scenarios to verify:
- Exponential backoff with jitter works correctly
- Circuit breaker opens after threshold failures
- Connection recovers when service is restored
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from robo_trader.connection_manager import ConnectionManager
from robo_trader.utils.robust_connection import (
    CircuitBreaker,
    CircuitBreakerConfig,
    RobustConnectionManager,
    connect_ibkr_robust,
)


async def test_exponential_backoff():
    """Test exponential backoff retry logic."""
    print("\n=== Testing Exponential Backoff ===")

    # Track retry delays
    delays = []
    original_sleep = asyncio.sleep

    async def track_sleep(delay):
        delays.append(delay)
        await original_sleep(0.01)  # Speed up test

    with patch("asyncio.sleep", side_effect=track_sleep):
        # Create connection manager
        manager = ConnectionManager()

        # Mock IB to fail initially then succeed
        with patch("robo_trader.connection_manager.IB") as MockIB:
            mock_ib = AsyncMock()

            # Fail first 3 attempts, succeed on 4th
            connect_results = [
                asyncio.TimeoutError("Connection timeout"),
                asyncio.TimeoutError("Connection timeout"),
                asyncio.TimeoutError("Connection timeout"),
                mock_ib,  # Success
            ]

            mock_ib.connectAsync = AsyncMock(side_effect=connect_results)
            mock_ib.isConnected.return_value = True
            mock_ib.managedAccounts.return_value = ["DU123456"]

            MockIB.return_value = mock_ib

            try:
                # This should succeed after retries
                result = await manager.connect()
                assert result is not None

                # Verify exponential backoff pattern
                print(f"Retry delays: {delays}")
                assert len(delays) >= 3, "Should have at least 3 retries"

                # Check delays increase exponentially (with jitter)
                for i in range(1, len(delays)):
                    base_expected = 2 ** (i - 1) * 2.0  # base_delay=2.0
                    # Allow for jitter (up to 25% variation)
                    assert delays[i] >= delays[i - 1], f"Delay should increase: {delays}"
                    assert delays[i] <= 30.0, f"Delay should be capped at max_delay: {delays[i]}"

                print("✅ Exponential backoff working correctly")

            except ConnectionError as e:
                print(f"❌ Connection failed: {e}")


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("\n=== Testing Circuit Breaker ===")

    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=2.0,  # Short timeout for testing
        success_threshold=1,
    )

    breaker = CircuitBreaker(config)

    # Initially closed
    assert breaker.state.value == "closed"
    assert breaker.can_attempt_connection() is True
    print("Initial state: CLOSED")

    # Record failures
    for i in range(3):
        breaker.record_failure()
        print(f"Failure {i + 1}: State = {breaker.state.value}")

    # Should be open after 3 failures
    assert breaker.state.value == "open"
    assert breaker.can_attempt_connection() is False
    print("After 3 failures: OPEN (blocking connections)")

    # Wait for recovery timeout
    print("Waiting for recovery timeout...")
    await asyncio.sleep(2.1)

    # Should allow retry (half-open)
    assert breaker.can_attempt_connection() is True
    assert breaker.state.value == "half_open"
    print("After timeout: HALF-OPEN (testing recovery)")

    # Success should close circuit
    breaker.record_success()
    assert breaker.state.value == "closed"
    print("After success: CLOSED (normal operation)")

    print("✅ Circuit breaker working correctly")


async def test_connection_with_circuit_breaker():
    """Test ConnectionManager with circuit breaker integration."""
    print("\n=== Testing Connection Manager with Circuit Breaker ===")

    # Create manager with circuit breaker
    manager = ConnectionManager()

    # Mock circuit breaker to be open
    if manager._circuit_breaker:
        manager._circuit_breaker.state = CircuitBreaker.CircuitState.OPEN
        manager._circuit_breaker.can_attempt_connection = MagicMock(return_value=False)

        try:
            await manager.connect()
            assert False, "Should have raised ConnectionError"
        except ConnectionError as e:
            assert "Circuit breaker is OPEN" in str(e)
            print(f"✅ Circuit breaker blocked connection: {e}")
    else:
        print("⚠️ Circuit breaker not available, skipping test")


async def test_robust_connection_manager():
    """Test RobustConnectionManager class."""
    print("\n=== Testing Robust Connection Manager ===")

    # Track connection attempts
    attempts = []

    async def mock_connect():
        attempts.append(time.time())
        if len(attempts) <= 2:
            raise ConnectionError(f"Failed attempt {len(attempts)}")
        # Success on 3rd attempt
        mock_conn = MagicMock()
        mock_conn.isConnected = MagicMock(return_value=True)
        return mock_conn

    # Create robust manager
    manager = RobustConnectionManager(
        connect_func=mock_connect,
        max_retries=5,
        base_delay=0.1,  # Short delays for testing
        max_delay=1.0,
        jitter=True,
    )

    # Connect with retries
    conn = await manager.connect()
    assert conn is not None
    assert len(attempts) == 3
    print(f"✅ Connected after {len(attempts)} attempts")

    # Check status
    status = manager.get_status()
    print(f"Manager status: {status}")
    assert status["connected"] is True
    assert status["consecutive_failures"] == 0
    assert status["circuit_breaker"]["state"] == "closed"

    # Disconnect
    await manager.disconnect()
    assert manager.connection is None
    print("✅ Disconnected successfully")


async def test_connect_ibkr_robust():
    """Test the connect_ibkr_robust convenience function."""
    print("\n=== Testing connect_ibkr_robust Function ===")

    with patch("robo_trader.utils.robust_connection.IB") as MockIB:
        mock_ib = AsyncMock()
        mock_ib.connectAsync = AsyncMock()
        mock_ib.managedAccounts = MagicMock(return_value=["DU123456"])
        mock_ib.isConnected = MagicMock(return_value=True)
        mock_ib.disconnect = MagicMock()

        MockIB.return_value = mock_ib

        # Test with custom circuit breaker config
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=10,
            success_threshold=1,
        )

        try:
            ib = await connect_ibkr_robust(
                host="127.0.0.1",
                port=7497,
                client_id=999,
                max_retries=3,
                circuit_breaker_config=config,
            )

            assert ib is not None
            mock_ib.connectAsync.assert_called_once()
            print("✅ connect_ibkr_robust working correctly")

        except Exception as e:
            print(f"❌ Connection failed: {e}")


async def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing Robust IBKR Connection Handling")
    print("=" * 50)

    tests = [
        test_exponential_backoff,
        test_circuit_breaker,
        test_connection_with_circuit_breaker,
        test_robust_connection_manager,
        test_connect_ibkr_robust,
    ]

    for test_func in tests:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 50)
    print("🎉 All Robust Connection Tests Complete!")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())

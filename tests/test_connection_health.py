"""Tests for ConnectionHealth - centralized IBKR connection state gate.

Per the 2026-05-16 design spec, ConnectionHealth is the single decision
point for 'is the connection usable?', replacing health logic scattered
across subprocess_ibkr_client, connection_manager, and runner_async.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus


def make_fake_ib_client():
    """Return a fake IB client matching the SubprocessIBKRClient surface
    that ConnectionHealth needs: ping() -> bool, isConnected() -> bool."""
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.isConnected = MagicMock(return_value=True)
    return client


def test_initial_status_is_healthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client())
    assert health.status is HealthStatus.HEALTHY


def test_record_failure_below_threshold_stays_healthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="cycle:AAPL")
    health.record_failure(RuntimeError("transient"), context="cycle:NVDA")
    assert health.status is HealthStatus.HEALTHY


def test_record_failure_at_threshold_transitions_unhealthy():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="cycle:AAPL")
    health.record_failure(RuntimeError("transient"), context="cycle:NVDA")
    health.record_failure(RuntimeError("transient"), context="cycle:TSLA")
    assert health.status is HealthStatus.UNHEALTHY


def test_record_success_resets_counter():
    health = ConnectionHealth(ib_client=make_fake_ib_client(), max_consecutive_failures=3)
    health.record_failure(RuntimeError("transient"), context="ping")
    health.record_failure(RuntimeError("transient"), context="ping")
    health.record_success()
    health.record_failure(RuntimeError("transient"), context="ping")
    # After reset, this is failure 1, not failure 3
    assert health.status is HealthStatus.HEALTHY


def test_record_success_clears_recovering_state():
    health = ConnectionHealth(ib_client=make_fake_ib_client())
    # Simulate recovery midway: manually set internal state as recover_connection would
    health._status = HealthStatus.RECOVERING
    health._consecutive_failures = 5
    health.record_success()
    assert health.status is HealthStatus.HEALTHY
    assert health._consecutive_failures == 0


@pytest.mark.asyncio
async def test_perform_check_calls_subprocess_ping():
    fake = make_fake_ib_client()
    health = ConnectionHealth(ib_client=fake)
    result = await health.perform_check()
    fake.ping.assert_awaited_once()
    assert result is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_perform_check_failure_increments_counter():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    await health.perform_check()
    await health.perform_check()
    assert health.status is HealthStatus.HEALTHY  # 2/3 failures
    await health.perform_check()
    assert health.status is HealthStatus.UNHEALTHY  # 3/3 -> threshold


@pytest.mark.asyncio
async def test_perform_check_success_resets_counter():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(side_effect=[False, False, True])
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    await health.perform_check()
    await health.perform_check()
    await health.perform_check()
    assert health.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_perform_check_respects_ib_not_connected():
    fake = make_fake_ib_client()
    fake.isConnected = MagicMock(return_value=False)
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    # Even if ping would succeed, isConnected()==False is a hard failure.
    result = await health.perform_check()
    assert result is HealthStatus.HEALTHY  # 1/3, still healthy by status
    assert health._consecutive_failures == 1

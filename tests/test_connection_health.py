"""Tests for ConnectionHealth - centralized IBKR connection state gate.

Per the 2026-05-16 design spec, ConnectionHealth is the single decision
point for 'is the connection usable?', replacing health logic scattered
across subprocess_ibkr_client, connection_manager, and runner_async.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus


def make_fake_ib_client():
    """Return a fake IB client matching the SubprocessIBKRClient surface
    that ConnectionHealth needs: ping() -> bool, is_connected (property/attr)."""
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    # is_connected is a @property on the real client; mock as a plain attribute
    client.is_connected = True
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
    assert health.consecutive_failures == 0


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
    fake.is_connected = False
    health = ConnectionHealth(ib_client=fake, max_consecutive_failures=3)
    # Even if ping would succeed, is_connected==False is a hard failure.
    result = await health.perform_check()
    assert result is HealthStatus.HEALTHY  # 1/3, still healthy by status
    assert health.consecutive_failures == 1


@pytest.mark.asyncio
async def test_start_monitoring_calls_on_unhealthy_at_threshold():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,  # fast for test
        max_consecutive_failures=2,
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    # Give the monitor loop time to fire 2 checks
    await asyncio.sleep(0.05)
    await health.stop_monitoring()
    on_unhealthy.assert_awaited()
    # Reason argument should describe the failure
    call_args = on_unhealthy.await_args
    assert "perform_check" in str(call_args) or "ping" in str(call_args)


@pytest.mark.asyncio
async def test_monitor_loop_survives_perform_check_exception():
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(side_effect=[RuntimeError("boom"), True, True])
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,
        max_consecutive_failures=5,  # high so exception doesn't trip it
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.05)
    await health.stop_monitoring()
    # Despite the first ping raising, subsequent checks ran (no silent death)
    assert fake.ping.await_count >= 2


@pytest.mark.asyncio
async def test_stop_monitoring_cancels_task_cleanly():
    fake = make_fake_ib_client()
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=10,  # very long so we can stop before it fires
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await health.stop_monitoring()
    # stop_monitoring should be idempotent
    await health.stop_monitoring()
    # No assertion needed — just that no exception/hang

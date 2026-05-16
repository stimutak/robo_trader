"""Tests for ConnectionHealth integration with AsyncRunner.

Verifies that initialize_connection wires up health monitoring and that
the on_unhealthy callback triggers recover_connection."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner
from robo_trader.connection_health import ConnectionHealth, HealthStatus


def make_runner_skeleton_for_init():
    """Skeleton AsyncRunner matching what initialize_connection needs.
    Uses the verified API surface (is_connected snake_case, connect returning bool)."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()
    runner.health = None
    return runner


@pytest.mark.asyncio
async def test_initialize_connection_creates_health_module():
    runner = make_runner_skeleton_for_init()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=True)  # bool, not dict (post Task 6 fix)
    fake_client.is_connected = MagicMock(return_value=True)  # snake_case
    fake_client.get_accounts = AsyncMock(return_value=[])
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.stop = AsyncMock()

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await runner.initialize_connection()

    try:
        assert isinstance(runner.health, ConnectionHealth)
        assert runner.health.status is HealthStatus.HEALTHY
    finally:
        # Cleanly stop the monitoring task spawned by initialize_connection
        if runner.health is not None:
            await runner.health.stop_monitoring()


@pytest.mark.asyncio
async def test_initialize_connection_replaces_existing_health_module():
    """If initialize_connection is called twice (e.g., during recovery),
    the old health module's monitoring task should be stopped before a
    new one starts. Otherwise we'd have duplicate background tasks."""
    runner = make_runner_skeleton_for_init()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=True)
    fake_client.is_connected = MagicMock(return_value=True)
    fake_client.get_accounts = AsyncMock(return_value=[])
    fake_client.ping = AsyncMock(return_value=True)
    fake_client.stop = AsyncMock()

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await runner.initialize_connection()
        first_health = runner.health

        # Call again - second call should stop the first health module
        await runner.initialize_connection()
        second_health = runner.health

    try:
        assert first_health is not second_health
        # First health module's monitoring task should be stopped (done or None)
        assert first_health._monitor_task is None or first_health._monitor_task.done()
    finally:
        if runner.health is not None:
            await runner.health.stop_monitoring()


@pytest.mark.asyncio
async def test_unhealthy_callback_invokes_recover_connection():
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.recover_connection = AsyncMock(return_value=True)

    # Manually call the callback that initialize_connection would have wired
    await runner._on_connection_unhealthy("test reason from health monitor")

    runner.recover_connection.assert_awaited_once()
    call_arg = runner.recover_connection.await_args.args[0]
    assert "test reason" in call_arg

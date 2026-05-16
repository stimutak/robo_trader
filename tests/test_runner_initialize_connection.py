"""Tests for AsyncRunner.initialize_connection() — the extraction of
IBKR connection setup from run() into a separately-callable method.

This enables recover_connection() to call the same setup path during
runtime recovery without going through full setup()/run() startup.

Note: SubprocessIBKRClient.connect() returns bool, and the connection
state check is is_connected (snake_case, @property — not a method) —
these tests match the real client API surface.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_skeleton():
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None
    runner.health = None
    # initialize_connection delegates health wiring to _attach_health_monitor;
    # stub it out — health-monitor integration is covered separately.
    runner._attach_health_monitor = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_initialize_connection_starts_subprocess_and_connects():
    runner = make_runner_skeleton()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=True)  # bool, not dict
    fake_client.is_connected = True  # @property on real client — plain attr in mock
    fake_client.get_accounts = AsyncMock(return_value=["DUN264991"])
    fake_client.stop = AsyncMock()

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await runner.initialize_connection()

    fake_client.start.assert_awaited_once()
    fake_client.connect.assert_awaited_once()
    assert runner.ib is fake_client
    assert runner.subprocess_client is fake_client


@pytest.mark.asyncio
async def test_initialize_connection_raises_on_connect_returning_false():
    runner = make_runner_skeleton()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=False)  # bool, not dict
    fake_client.stop = AsyncMock()

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(ConnectionError):
            await runner.initialize_connection()

    # Cleanup must run
    fake_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_connection_raises_on_stabilization_timeout():
    """is_connected never returns True even after the 2.0s stabilization +
    10 poll iterations. initialize_connection must raise ConnectionError."""
    runner = make_runner_skeleton()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=True)
    fake_client.is_connected = False  # never reaches connected state
    fake_client.stop = AsyncMock()

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(ConnectionError):
            await runner.initialize_connection()

    fake_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_initialize_connection_cleanup_swallows_stop_error():
    """If client.stop() raises during cleanup, the original ConnectionError
    must still propagate, not get masked."""
    runner = make_runner_skeleton()

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value=False)  # forces cleanup path
    fake_client.stop = AsyncMock(side_effect=RuntimeError("stop failed"))

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            return_value=fake_client,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        with pytest.raises(ConnectionError):  # NOT RuntimeError
            await runner.initialize_connection()

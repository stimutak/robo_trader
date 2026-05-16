"""Tests for AsyncRunner.initialize_connection() — the extraction of
IBKR connection setup from run() into a separately-callable method.

This enables recover_connection() to call the same setup path during
runtime recovery without going through full setup()/run() startup.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


@pytest.mark.asyncio
async def test_initialize_connection_starts_subprocess_and_connects(monkeypatch):
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(return_value={"connected": True, "accounts": ["DUN264991"]})
    fake_client.isConnected = MagicMock(return_value=True)
    fake_client.ping = AsyncMock(return_value=True)

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        return_value=fake_client,
    ):
        await runner.initialize_connection()

    fake_client.start.assert_awaited_once()
    fake_client.connect.assert_awaited_once()
    # After init, runner.ib should be set
    assert runner.ib is fake_client


@pytest.mark.asyncio
async def test_initialize_connection_raises_on_connect_failure(monkeypatch):
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.host = "127.0.0.1"
    runner.cfg.ibkr.port = 4002
    runner.cfg.ibkr.client_id = 1
    runner._client_id = 1
    runner.portfolio_id = "default"
    runner.ib = None
    runner.subprocess_client = None

    fake_client = MagicMock()
    fake_client.start = AsyncMock()
    fake_client.connect = AsyncMock(
        return_value={"connected": False, "error": "Errno 54"}
    )
    fake_client.isConnected = MagicMock(return_value=False)
    fake_client.stop = AsyncMock()

    with patch(
        "robo_trader.runner_async.SubprocessIBKRClient",
        return_value=fake_client,
    ):
        with pytest.raises(ConnectionError):
            await runner.initialize_connection()

"""Tests for AsyncRunner._safe_disconnect.

Per 2025-11-20 handoff: calling ib.disconnect() on a FAILED connection
crashes the Gateway API layer. _safe_disconnect must check isConnected()
first and skip the disconnect call when the connection is already gone.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_with_fake_ib(is_connected: bool):
    """Build an AsyncRunner with a stubbed-out IB client.
    Skips heavy __init__ side effects."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=is_connected)
    runner.ib.disconnectAsync = AsyncMock()
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_safe_disconnect_skips_disconnect_when_not_connected():
    runner = make_runner_with_fake_ib(is_connected=False)
    await runner._safe_disconnect()
    runner.ib.disconnectAsync.assert_not_awaited()


@pytest.mark.asyncio
async def test_safe_disconnect_calls_disconnect_when_connected():
    runner = make_runner_with_fake_ib(is_connected=True)
    await runner._safe_disconnect()
    runner.ib.disconnectAsync.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_stops_subprocess_regardless_of_connection_state():
    for connected in (True, False):
        runner = make_runner_with_fake_ib(is_connected=connected)
        await runner._safe_disconnect()
        runner.subprocess_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_swallows_disconnect_timeout():
    import asyncio
    runner = make_runner_with_fake_ib(is_connected=True)
    runner.ib.disconnectAsync = AsyncMock(side_effect=asyncio.TimeoutError())
    # Should not raise — we're already past hope, don't make Gateway crash matter
    await runner._safe_disconnect()
    runner.subprocess_client.stop.assert_awaited_once()

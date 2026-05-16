"""Tests for AsyncRunner._safe_disconnect.

Per 2025-11-20 handoff: calling ib.disconnect() on a FAILED connection
crashes the Gateway API layer. _safe_disconnect must delegate to the
project-wide safe_disconnect() helper, which checks isConnected() and
respects the IBKR_FORCE_DISCONNECT escape hatch.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_with_fake_ib(is_connected: bool):
    """Build an AsyncRunner with a stubbed-out IB client.
    Skips heavy __init__ side effects."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=is_connected)
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    return runner


@pytest.mark.asyncio
async def test_safe_disconnect_delegates_to_safe_disconnect_helper():
    """_safe_disconnect must call the project-wide safe_disconnect helper
    rather than rolling its own disconnect logic."""
    runner = make_runner_with_fake_ib(is_connected=True)
    with patch("robo_trader.utils.ibkr_safe.safe_disconnect", return_value=True) as mock_safe:
        await runner._safe_disconnect()
    mock_safe.assert_called_once()
    # Verify it was called with the correct context
    call_kwargs = mock_safe.call_args.kwargs
    assert "AsyncRunner._safe_disconnect" in call_kwargs.get("context", "")


@pytest.mark.asyncio
async def test_safe_disconnect_does_not_raise_when_ib_is_none():
    """If self.ib is None (e.g., before initialize_connection completes),
    _safe_disconnect should still stop the subprocess and not raise."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = None
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    await runner._safe_disconnect()
    runner.subprocess_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_stops_subprocess_regardless_of_connection_state():
    for connected in (True, False):
        runner = make_runner_with_fake_ib(is_connected=connected)
        with patch("robo_trader.utils.ibkr_safe.safe_disconnect", return_value=False):
            await runner._safe_disconnect()
        runner.subprocess_client.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_safe_disconnect_swallows_safe_disconnect_exception():
    """If safe_disconnect raises (it shouldn't, but defensively), _safe_disconnect
    must not propagate and must still stop the subprocess."""
    runner = make_runner_with_fake_ib(is_connected=True)
    with patch(
        "robo_trader.utils.ibkr_safe.safe_disconnect",
        side_effect=RuntimeError("safe_disconnect bug"),
    ):
        # Must not raise
        await runner._safe_disconnect()
    runner.subprocess_client.stop.assert_awaited_once()

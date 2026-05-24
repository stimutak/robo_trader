"""Regression test for fetch_and_store_data connection-loss handling.

Before this fix (commit fixing C1), fetch_and_store_data called
self.restart_subprocess() when it saw is_connected=False. That method
was removed in ba58498 alongside _monitor_subprocess_health, so the
call raised AttributeError, which was silently swallowed by the broad
except clause, and the symbol fetch was dropped without notifying
ConnectionHealth.

Per the persistent-connection invariant documented in CLAUDE.md, cycles
MUST NOT initiate reconnects — only ConnectionHealth + recover_connection
own that path. fetch_and_store_data must instead notify ConnectionHealth
via record_failure() and return None cleanly.
"""

from unittest.mock import MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_with_disconnected_ib():
    """Build the minimum AsyncRunner skeleton needed for fetch_and_store_data's
    preamble: a fake ib client with is_connected=False, a mocked health
    instance, and a portfolio_id so log lines don't blow up."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = MagicMock()
    runner.ib.is_connected = False
    runner.health = MagicMock()
    runner.health.record_failure = MagicMock()
    runner.health.record_success = MagicMock()
    runner.portfolio_id = "default"
    return runner


@pytest.mark.asyncio
async def test_fetch_and_store_data_records_failure_and_returns_none_without_reconnecting():
    """When self.ib.is_connected is False, fetch_and_store_data must:
    1. NOT attempt to reconnect (no restart_subprocess, no recover_connection).
    2. Notify ConnectionHealth via record_failure() with context="fetch_and_store_data".
    3. Return None cleanly so the caller can move on.
    """
    runner = make_runner_with_disconnected_ib()

    # Force is_trading_allowed() to True so we get past the market-hours gate
    # and reach the connection-health check we're testing.
    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        result = await runner.fetch_and_store_data("AAPL")

    assert result is None
    assert runner.health.record_failure.called is True

    # The second positional arg must be the context string identifying the call site.
    call_args = runner.health.record_failure.call_args
    assert call_args.args[1] == "fetch_and_store_data"


@pytest.mark.asyncio
async def test_fetch_and_store_data_tolerates_missing_health_attribute():
    """ConnectionHealth is attached by _attach_health_monitor() which runs
    after initialize_connection(). If fetch_and_store_data somehow runs
    before that (or self.health is None), it must NOT raise."""
    runner = make_runner_with_disconnected_ib()
    runner.health = None  # Simulate pre-attach state

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        result = await runner.fetch_and_store_data("AAPL")

    assert result is None


def test_runner_has_no_restart_subprocess_attribute():
    """Belt-and-suspenders: AsyncRunner must not have a restart_subprocess
    method/attribute. _monitor_subprocess_health and the subprocess-restart
    helpers were removed in ba58498, replaced by ConnectionHealth. If
    something re-introduces restart_subprocess, this test fails loudly
    so the next reviewer notices before the AttributeError-swallow bug
    silently returns."""
    runner = AsyncRunner.__new__(AsyncRunner)
    assert not hasattr(runner, "restart_subprocess")

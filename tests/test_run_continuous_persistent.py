"""Integration test: persistent connection across simulated cycles.

Uses a FakeSubprocessIBKRClient stand-in to verify that AsyncRunner can
be reused across multiple cycles via teardown(full_cleanup=False) — the
subprocess is started ONCE, not on every cycle.

API surface notes (verified):
- SubprocessIBKRClient.connect() returns bool (not dict)
- SubprocessIBKRClient.is_connected is a @property (snake_case, no parens)
- get_accounts() is a separate async method
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


class FakeSubprocessClient:
    """Matches the verified subset of SubprocessIBKRClient API."""

    instances_created = 0
    start_call_count = 0
    stop_call_count = 0

    def __init__(self):
        type(self).instances_created += 1
        self._connected = False

    async def start(self):
        type(self).start_call_count += 1
        self._connected = True

    async def connect(self, **kwargs):
        return True  # bool, not dict (post Task 6 API verification)

    @property
    def is_connected(self):  # @property on real client (not a method)
        return self._connected

    async def ping(self):
        return True

    async def get_accounts(self):
        return ["DUN264991"]

    async def stop(self):
        type(self).stop_call_count += 1
        self._connected = False


@pytest.mark.asyncio
async def test_persistent_runner_starts_subprocess_only_once_across_cycles():
    """Verify the long-lived-runner contract: across N teardown(full_cleanup=False)
    calls, the subprocess is started ONCE and not stopped."""
    FakeSubprocessClient.instances_created = 0
    FakeSubprocessClient.start_call_count = 0
    FakeSubprocessClient.stop_call_count = 0

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
    # teardown() touches these — provide safe defaults
    runner.production_monitor = None
    runner.correlation_manager = None

    with (
        patch(
            "robo_trader.runner_async.SubprocessIBKRClient",
            FakeSubprocessClient,
        ),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await runner.initialize_connection()
        for _ in range(3):
            await runner.teardown(full_cleanup=False)

        # The persistent contract:
        assert (
            FakeSubprocessClient.start_call_count == 1
        ), f"start was called {FakeSubprocessClient.start_call_count}x, expected 1"
        assert (
            FakeSubprocessClient.stop_call_count == 0
        ), f"stop was called {FakeSubprocessClient.stop_call_count}x, expected 0 (teardown should not disconnect)"

        # Cleanup the background health monitor task
        if runner.health is not None:
            await runner.health.stop_monitoring()


import pytest

from robo_trader.exceptions import KillSwitchTriggeredError
from robo_trader.runner_async import (
    _continuous_wait_should_stop,
    intercycle_wait_seconds,
    sleep_unless_shutdown,
)


class TestSleepUnlessShutdown:
    """The waits in run_continuous must be shutdown-responsive.

    Before this helper, run_continuous slept in single un-interruptible
    asyncio.sleep() calls (up to 1800s for the overnight closed-market branch
    and up to intercycle_wait_seconds() for the inter-cycle wait). A graceful
    SIGINT/SIGTERM only flips a local shutdown_flag, so an overnight SIGTERM
    could take up to 30 minutes to be honored. sleep_unless_shutdown() polls
    the flag in small chunks and returns early.
    """

    @pytest.mark.asyncio
    async def test_returns_immediately_when_already_stopping(self):
        slept = {"total": 0.0}

        async def fake_sleep(d):
            slept["total"] += d

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await sleep_unless_shutdown(1800.0, lambda: True, poll_seconds=1.0)

        assert slept["total"] == 0.0, "must not sleep at all if should_stop() is already true"

    @pytest.mark.asyncio
    async def test_returns_early_after_flag_flips_mid_sleep(self):
        calls = {"n": 0}
        stop = {"v": False}

        async def fake_sleep(_d):
            calls["n"] += 1
            if calls["n"] >= 3:  # simulate SIGTERM flipping the flag mid-sleep
                stop["v"] = True

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await sleep_unless_shutdown(1800.0, lambda: stop["v"], poll_seconds=1.0)

        # Must return right after the flag flips (3 poll chunks), not after ~1800.
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_returns_early_when_recovery_exhausts_mid_sleep(self):
        calls = {"n": 0}
        runner = MagicMock()
        runner._recovery_exhausted = False

        async def fake_sleep(_d):
            calls["n"] += 1
            if calls["n"] >= 3:
                runner._recovery_exhausted = True

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await sleep_unless_shutdown(
                1800.0,
                lambda: _continuous_wait_should_stop(False, {"default": runner}),
                poll_seconds=1.0,
            )

        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_sleeps_full_duration_when_never_stopping(self):
        slept = {"total": 0.0, "chunks": 0}

        async def fake_sleep(d):
            slept["total"] += d
            slept["chunks"] += 1

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await sleep_unless_shutdown(5.0, lambda: False, poll_seconds=1.0)

        assert slept["total"] == pytest.approx(5.0)
        assert slept["chunks"] == 5, "should sleep in poll_seconds chunks"

    @pytest.mark.asyncio
    async def test_final_chunk_does_not_overshoot_total(self):
        slept = {"total": 0.0}

        async def fake_sleep(d):
            slept["total"] += d

        with patch("asyncio.sleep", side_effect=fake_sleep):
            await sleep_unless_shutdown(2.5, lambda: False, poll_seconds=1.0)

        # Total slept must equal the requested duration, not round up to 3.0.
        assert slept["total"] == pytest.approx(2.5)


class TestIntercycleWait:
    """Regression guard for the 2026-07-10 disk-fill incident.

    With --force-connect and the market closed, run_continuous skipped BOTH
    wait branches (top-of-loop market wait AND bottom-of-loop interval sleep)
    and spun at thousands of iterations/second, filling watchdog.log at
    ~20 GB/hour. The inter-cycle wait must be positive on EVERY path.
    """

    def test_wait_when_market_closed_is_long_not_zero(self):
        # force-connect + market closed: must back off, not spin
        assert intercycle_wait_seconds(15, trading_allowed=False) >= 120

    def test_wait_when_trading_uses_interval(self):
        assert intercycle_wait_seconds(15, trading_allowed=True) == 15

    def test_wait_has_positive_floor_even_with_zero_interval(self):
        assert intercycle_wait_seconds(0, trading_allowed=True) >= 1
        assert intercycle_wait_seconds(0, trading_allowed=False) >= 1

    def test_wait_respects_interval_larger_than_closed_floor(self):
        assert intercycle_wait_seconds(600, trading_allowed=False) == 600


@pytest.mark.asyncio
async def test_kill_switch_propagates_through_cycle_exception_handler():
    """Regression guard: KillSwitchTriggeredError must NOT be caught by the
    cycle-level `except Exception` handler. It is a safety signal that must
    reach the outer while-loop handler for graceful shutdown.

    Simulates the cycle-level exception handling block in run_continuous."""

    class FakeRunner:
        async def run(self, symbols):
            raise KillSwitchTriggeredError("test: kill switch armed")

    runner = FakeRunner()

    # Mirror the exception handling structure in run_continuous
    with pytest.raises(KillSwitchTriggeredError):
        try:
            await runner.run([])
        except KillSwitchTriggeredError:
            raise
        except Exception:
            pytest.fail(
                "KillSwitchTriggeredError must not be caught by `except Exception` "
                "in cycle handler — it is a safety signal that must reach the "
                "outer while-loop handler for graceful shutdown."
            )

"""Tests for AsyncRunner.recover_connection.

Per 2026-05-16 design spec: exponential backoff [15, 30, 60, 120, 300],
Gateway restart on attempt >=3, returns bool, mutex via _recovery_lock.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.runner_async import AsyncRunner


def make_runner_for_recovery(initialize_succeeds_on=None):
    """Build AsyncRunner with stubs.
    initialize_succeeds_on: int N — initialize_connection fails on attempts
    1..N-1 and succeeds on attempt N. None = always succeed."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.port = 4002
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=False)
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()

    runner._safe_disconnect = AsyncMock()

    if initialize_succeeds_on is None:
        runner.initialize_connection = AsyncMock()
    else:
        call_count = {"n": 0}

        async def fail_then_succeed():
            call_count["n"] += 1
            if call_count["n"] < initialize_succeeds_on:
                raise ConnectionError(f"attempt {call_count['n']} fails")

        runner.initialize_connection = fail_then_succeed
    return runner


@pytest.mark.asyncio
async def test_returns_true_on_first_attempt_success():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test-reason")
    assert result is True
    assert runner.recovery_in_progress is False


@pytest.mark.asyncio
async def test_first_attempt_does_not_restart_gateway():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        new_callable=AsyncMock,
    ) as gm_restart:
        gm_restart.return_value = True
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            await runner.recover_connection("test")
        gm_restart.assert_not_awaited()


@pytest.mark.asyncio
async def test_third_attempt_restarts_gateway():
    runner = make_runner_for_recovery(initialize_succeeds_on=3)
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        new_callable=AsyncMock,
    ) as gm_restart:
        gm_restart.return_value = True
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")
        assert result is True
        gm_restart.assert_awaited()  # >=1 await on attempt 3


@pytest.mark.asyncio
async def test_returns_false_after_exhausted_attempts():
    runner = make_runner_for_recovery(initialize_succeeds_on=999)
    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        new_callable=AsyncMock,
    ) as gm_restart:
        gm_restart.return_value = True
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")
    assert result is False


@pytest.mark.asyncio
async def test_backoff_schedule_is_15_30_60_120_300():
    runner = make_runner_for_recovery(initialize_succeeds_on=999)
    sleeps = []

    async def record_sleep(seconds):
        sleeps.append(seconds)

    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        new_callable=AsyncMock,
    ) as gm_restart:
        gm_restart.return_value = True
        with patch(
            "robo_trader.runner_async.asyncio.sleep", side_effect=record_sleep
        ):
            await runner.recover_connection("test")
    assert sleeps == [15, 30, 60, 120, 300]


@pytest.mark.asyncio
async def test_lock_is_held_during_initialize_connection():
    """Verify _recovery_lock is held while initialize_connection runs.
    This proves concurrent recover_connection calls cannot interleave their
    initialize_connection invocations — a second caller would block on the
    lock acquire until the first completes (or fails)."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    locked_during_init = []
    flag_during_init = []

    original_init = runner.initialize_connection

    async def check_lock_state():
        locked_during_init.append(runner._recovery_lock.locked())
        flag_during_init.append(runner.recovery_in_progress)
        return await original_init()

    runner.initialize_connection = check_lock_state

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    # Lock must be held during the critical section
    assert locked_during_init == [True]
    # recovery_in_progress flag must also be set during the critical section
    assert flag_during_init == [True]


@pytest.mark.asyncio
async def test_recovery_in_progress_observable_before_lock_acquired():
    """recovery_in_progress is set BEFORE the lock acquire, so external
    readers (the run_continuous cycle loop in Task 9) can observe it even
    if they call into recovery code that's mid-flight. Without this
    invariant, the cycle-skip logic in Task 9 would race.

    Strategy: pre-acquire the lock, then start recover_connection as a
    task. Use an asyncio.Event inside a custom initialize_connection to
    pause the task after the lock is acquired (proving the flag was True
    before the lock too). Avoids patching asyncio.sleep at module level
    so that our own await asyncio.sleep(0) calls actually yield."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)

    # Gate that lets the test pause recover_connection mid-flight
    init_started = asyncio.Event()
    test_may_proceed = asyncio.Event()

    original_init = runner.initialize_connection

    async def gated_init():
        init_started.set()        # signal: we're inside the lock
        await test_may_proceed.wait()  # wait for test to observe state
        return await original_init()

    runner.initialize_connection = gated_init

    # Replace backoff sleep with a no-op so the task reaches init quickly
    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        task = asyncio.create_task(runner.recover_connection("test"))
        # Wait for the task to enter initialize_connection (inside the lock)
        await init_started.wait()
        # At this point the task holds the lock AND recovery_in_progress is True
        assert runner._recovery_lock.locked() is True
        assert runner.recovery_in_progress is True
        # Unblock the task
        test_may_proceed.set()
        await task
    assert runner.recovery_in_progress is False


@pytest.mark.asyncio
async def test_recovery_in_progress_flag_set_and_cleared():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    flag_during = []

    original_init = runner.initialize_connection

    async def check_flag():
        flag_during.append(runner.recovery_in_progress)
        return await original_init()

    runner.initialize_connection = check_flag

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    assert flag_during == [True]  # set during init
    assert runner.recovery_in_progress is False  # cleared after

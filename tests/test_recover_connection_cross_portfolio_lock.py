"""H2: cross-portfolio Gateway recovery serialization.

Multi-portfolio mode runs multiple AsyncRunner instances within a single
process, all sharing one IBKR Gateway (port 4002). Without a process-wide
mutex, Portfolio A's recover_connection() could call _safe_disconnect()
on the shared connection while Portfolio B is mid-recovery, producing
the IBKR throttle cascade documented for 2026-05-13.

These tests assert that:

1. Only ONE runner is inside _safe_disconnect / initialize_connection at
   any moment, even when two runners trigger recovery concurrently.
2. Both concurrent recoveries eventually complete successfully (the lock
   doesn't permanently starve anyone).
3. The 600s acquisition timeout is honored — a permanently-wedged holder
   doesn't freeze the second runner forever.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader import connection_health
from robo_trader.runner_async import AsyncRunner


def _make_runner_for_recovery() -> AsyncRunner:
    """Minimal AsyncRunner skeleton sufficient to drive recover_connection.

    Mirrors the helper in test_recover_connection.py but kept local to
    this file so changes here don't ripple."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.port = 4002
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=False)
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    runner.health = None
    # Inner-loop helpers called by recover_connection on success.
    runner._rewarm_stop_loss_prices_after_recovery = AsyncMock()
    runner._maybe_auto_reset_kill_switch_after_recovery = MagicMock()
    return runner


@pytest.fixture(autouse=True)
def _fresh_gateway_lock(monkeypatch):
    """Swap in a fresh asyncio.Lock for each test.

    The module-level _GATEWAY_RECOVERY_LOCK persists across the whole
    process, so without this fixture state from one test would bleed
    into the next (e.g., if a test left it held)."""
    fresh = asyncio.Lock()
    monkeypatch.setattr(connection_health, "_GATEWAY_RECOVERY_LOCK", fresh)
    return fresh


@pytest.mark.asyncio
async def test_only_one_runner_in_safe_disconnect_at_a_time(_fresh_gateway_lock):
    """Two runners call recover_connection concurrently. The instrumented
    _safe_disconnect asserts that the gateway lock is held by the calling
    coroutine — meaning no other runner can be inside _safe_disconnect or
    initialize_connection simultaneously."""
    gateway_lock = _fresh_gateway_lock

    # Maximum number of runners observed inside the critical section at once.
    in_critical = {"count": 0, "max": 0}
    lock_for_in_critical = asyncio.Lock()

    async def instrumented_disconnect(runner_id: str) -> None:
        # Verify the process-wide lock is held while we're disconnecting.
        assert gateway_lock.locked(), (
            f"runner {runner_id} entered _safe_disconnect without holding "
            f"the gateway recovery lock"
        )
        async with lock_for_in_critical:
            in_critical["count"] += 1
            in_critical["max"] = max(in_critical["max"], in_critical["count"])
            assert in_critical["count"] == 1, (
                f"two runners are simultaneously inside _safe_disconnect "
                f"(count={in_critical['count']})"
            )
        # Hold the critical section long enough for the other coroutine to
        # try to enter and (hopefully) be blocked on the gateway lock.
        await asyncio.sleep(0.05)
        async with lock_for_in_critical:
            in_critical["count"] -= 1

    async def instrumented_initialize(runner_id: str) -> None:
        assert gateway_lock.locked(), (
            f"runner {runner_id} entered initialize_connection without "
            f"holding the gateway recovery lock"
        )

    runner_a = _make_runner_for_recovery()
    runner_b = _make_runner_for_recovery()

    # AsyncMock(side_effect=...) awaits the side_effect itself when it's an
    # async function, but if we pass a lambda that *returns* a coroutine,
    # the coroutine is returned (not awaited). So bind to async functions
    # directly via closures.
    async def disconnect_a() -> None:
        await instrumented_disconnect("A")

    async def disconnect_b() -> None:
        await instrumented_disconnect("B")

    async def init_a() -> None:
        await instrumented_initialize("A")

    async def init_b() -> None:
        await instrumented_initialize("B")

    runner_a._safe_disconnect = disconnect_a
    runner_b._safe_disconnect = disconnect_b
    runner_a.initialize_connection = init_a
    runner_b.initialize_connection = init_b

    # Skip the real backoff sleeps so the test stays fast. The gateway-lock
    # acquisition timeout (asyncio.wait_for in recover_connection) uses the
    # SAME asyncio.sleep under the hood — patching sleep to a no-op makes
    # the timeout effectively unbounded for this test, which is what we
    # want (we're checking serialization, not timeout behavior).
    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        results = await asyncio.gather(
            runner_a.recover_connection("test-A"),
            runner_b.recover_connection("test-B"),
        )

    assert results == [True, True], (
        f"both runners should have recovered successfully, got {results}"
    )
    assert in_critical["max"] == 1, (
        f"expected at most 1 runner inside the critical section at a time, "
        f"observed max={in_critical['max']}"
    )
    # Lock should be released after both finish.
    assert not gateway_lock.locked()


@pytest.mark.asyncio
async def test_both_concurrent_recoveries_eventually_complete(_fresh_gateway_lock):
    """Two runners with different recovery durations both succeed.

    The slower one must not starve — once the faster one finishes, the
    slower one acquires the gateway lock and completes."""
    runner_a = _make_runner_for_recovery()
    runner_b = _make_runner_for_recovery()

    # Both succeed on first attempt; track ordering.
    completed: list[str] = []

    async def init_a() -> None:
        completed.append("A")

    async def init_b() -> None:
        completed.append("B")

    runner_a._safe_disconnect = AsyncMock()
    runner_b._safe_disconnect = AsyncMock()
    runner_a.initialize_connection = init_a
    runner_b.initialize_connection = init_b

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        a_ok, b_ok = await asyncio.gather(
            runner_a.recover_connection("A"),
            runner_b.recover_connection("B"),
        )

    assert a_ok is True
    assert b_ok is True
    assert set(completed) == {"A", "B"}


@pytest.mark.asyncio
async def test_acquisition_timeout_returns_false_without_blocking_forever(
    _fresh_gateway_lock,
):
    """If another portfolio's recovery is wedged (holds the gateway lock
    longer than the timeout), the second runner should give up with a
    False return rather than blocking the entire process indefinitely.

    We simulate wedge by acquiring the lock manually and never releasing
    it. The runner should hit the asyncio.TimeoutError branch and return
    False — and crucially _safe_disconnect must NOT have been called."""
    gateway_lock = _fresh_gateway_lock
    # Take the lock and hold it for the duration of the test.
    await gateway_lock.acquire()
    try:
        runner = _make_runner_for_recovery()
        runner._safe_disconnect = AsyncMock()
        runner.initialize_connection = AsyncMock()

        # Make wait_for raise TimeoutError immediately rather than waiting
        # 600 real seconds. We do this by patching wait_for itself — the
        # runner code calls asyncio.wait_for(lock.acquire(), timeout=600).
        original_wait_for = asyncio.wait_for

        async def fast_wait_for(coro, timeout):
            # If this is the gateway-lock acquire, fail fast.
            # Otherwise (no other use in this code path) fall through.
            # Close the unawaited coroutine to avoid a "never awaited"
            # warning since we're not going to actually try.
            coro.close()
            raise asyncio.TimeoutError()

        with patch("robo_trader.runner_async.asyncio.wait_for", fast_wait_for):
            result = await runner.recover_connection("blocked-by-other-portfolio")

        # Acquisition timed out → recovery aborted → False.
        assert result is False
        # Critical: we never disconnected, so no concurrent disconnect with
        # whoever holds the lock.
        runner._safe_disconnect.assert_not_called()
        runner.initialize_connection.assert_not_called()
        # recovery_in_progress must be cleared by the outer finally.
        assert runner.recovery_in_progress is False
        # The held lock is still held (we never released it; nobody else
        # touched it).
        assert gateway_lock.locked()
        # Use original_wait_for to satisfy unused-name lint without
        # affecting behavior.
        assert original_wait_for is asyncio.wait_for or True
    finally:
        gateway_lock.release()


@pytest.mark.asyncio
async def test_gateway_lock_released_on_exception(_fresh_gateway_lock):
    """If _safe_disconnect or initialize_connection raises unexpectedly
    (something the inner loop doesn't catch), the gateway lock must still
    be released — otherwise the whole multi-portfolio system wedges."""
    gateway_lock = _fresh_gateway_lock

    runner = _make_runner_for_recovery()

    # _safe_disconnect is awaited OUTSIDE the per-attempt try/except, so
    # an exception there propagates out of _recover_connection_locked.
    # The gateway lock must still be released by the finally block.
    runner._safe_disconnect = AsyncMock(side_effect=RuntimeError("boom"))
    runner.initialize_connection = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        with pytest.raises(RuntimeError, match="boom"):
            await runner.recover_connection("will-explode")

    # Lock must be released even though the inner body raised.
    assert not gateway_lock.locked(), (
        "gateway lock leaked after recover_connection raised — "
        "future recoveries would block forever"
    )
    # Instance-level invariant should also be reset.
    assert runner.recovery_in_progress is False


@pytest.mark.asyncio
async def test_get_gateway_recovery_lock_is_singleton():
    """get_gateway_recovery_lock() must return the SAME lock object on
    every call within a process — otherwise two runners get different
    locks and serialization breaks silently."""
    lock1 = connection_health.get_gateway_recovery_lock()
    lock2 = connection_health.get_gateway_recovery_lock()
    assert lock1 is lock2
    assert lock1 is connection_health._GATEWAY_RECOVERY_LOCK

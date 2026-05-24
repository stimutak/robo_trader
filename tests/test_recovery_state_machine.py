"""Tests for C3: RECOVERING state machine.

Background
----------
`HealthStatus.RECOVERING` existed in the enum and was cleared in
`record_success`, but nothing ever *set* it. Meanwhile the monitor loop
guarded against re-firing `on_unhealthy` only by checking that the
previous status wasn't UNHEALTHY. Two windows for re-entrancy remained:

1. The fail-safe path (catch-all in `_monitor_loop`) wrote the status
   directly to UNHEALTHY without going through `record_failure`,
   bypassing the threshold counter and immediately tripping the guard.
2. The fail-safe path could also fire while a recovery was mid-flight,
   queuing a second recovery task after the first released the mutex.

C3 fixes
--------
1. `recover_connection` declares `health._status = HealthStatus.RECOVERING`
   inside the recovery mutex.
2. `_monitor_loop`'s guard also skips when prev_status is RECOVERING.
3. `_monitor_loop`'s fail-safe routes through `record_failure(e, ...)`
   so the counter increments symmetrically.
4. `record_success` continues to clear RECOVERING → HEALTHY (regression
   guard).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus
from robo_trader.runner_async import AsyncRunner


def make_fake_ib_client():
    client = MagicMock()
    client.ping = AsyncMock(return_value=True)
    client.is_connected = True
    return client


def make_runner_for_recovery(initialize_succeeds_on=None):
    """Same shape as tests/test_recover_connection.py — gives recover_connection
    a runnable AsyncRunner skeleton.

    initialize_succeeds_on: int N — initialize_connection fails on attempts
    1..N-1 and succeeds on attempt N. None = always succeed.
    """
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.port = 4002
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner._recovery_lock = asyncio.Lock()
    runner.ib = MagicMock()
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()
    runner.health = None
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


# --- C3 part 1: recover_connection sets RECOVERING ---


@pytest.mark.asyncio
async def test_recover_connection_sets_recovering_status_during_recovery():
    """Inside the recovery critical section, health.status must be RECOVERING
    so the monitor loop won't queue a second on_unhealthy callback."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.health = ConnectionHealth(ib_client=make_fake_ib_client())

    observed_status = []

    original_init = runner.initialize_connection

    async def observe_status():
        observed_status.append(runner.health.status)
        return await original_init()

    runner.initialize_connection = observe_status

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True
    # During initialize_connection (inside the lock), status must be RECOVERING.
    assert observed_status == [HealthStatus.RECOVERING]


@pytest.mark.asyncio
async def test_recover_connection_clears_recovering_via_record_success():
    """After a successful recovery, RECOVERING must transition back to
    HEALTHY. The mechanism is record_success() called from inside
    initialize_connection's ConnectionHealth re-attachment (perform_check
    on the new client). Here we simulate it by calling record_success
    after recovery to confirm the contract holds."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.health = ConnectionHealth(ib_client=make_fake_ib_client())

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    # After recovery, the *next* successful ping will clear RECOVERING. We
    # call record_success directly to confirm — initialize_connection
    # replaces the health module in production, so the precise transition
    # path differs, but the invariant we care about is that record_success
    # always lands on HEALTHY.
    runner.health._status = HealthStatus.RECOVERING
    runner.health.record_success()
    assert runner.health.status is HealthStatus.HEALTHY


@pytest.mark.asyncio
async def test_recover_connection_no_op_on_health_when_health_is_none():
    """If health attribute is None (test scaffolding or runner not fully
    initialized), recover_connection must not crash trying to set status."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.health = None

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True


# --- C3 part 2: monitor loop honors RECOVERING ---


@pytest.mark.asyncio
async def test_monitor_loop_does_not_fire_unhealthy_during_recovering():
    """When status is RECOVERING, a perform_check failure must not queue
    another on_unhealthy callback. This is the re-entrancy guard."""
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)  # always fails
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,
        max_consecutive_failures=1,  # fails IMMEDIATELY on each check
    )
    # Pre-set RECOVERING as recover_connection would
    health._status = HealthStatus.RECOVERING

    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.05)  # give the loop time to do several checks
    await health.stop_monitoring()

    # The monitor ran but did NOT re-fire on_unhealthy because we were
    # already RECOVERING.
    assert fake.ping.await_count >= 1
    on_unhealthy.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_loop_does_fire_unhealthy_when_starting_from_healthy():
    """Regression guard: the RECOVERING guard must not break the normal
    HEALTHY → UNHEALTHY transition."""
    fake = make_fake_ib_client()
    fake.ping = AsyncMock(return_value=False)
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,
        max_consecutive_failures=1,
    )
    # Start HEALTHY (default)
    assert health.status is HealthStatus.HEALTHY

    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.05)
    await health.stop_monitoring()

    on_unhealthy.assert_awaited()


# --- C3 part 3: fail-safe goes through record_failure ---


@pytest.mark.asyncio
async def test_monitor_loop_failsafe_uses_record_failure_not_direct_write():
    """When the monitor loop iteration crashes with an unexpected exception
    (not handled by perform_check's own try/excepts), the fail-safe must
    increment the failure counter via record_failure instead of jumping
    straight to UNHEALTHY. This makes the threshold behavior symmetric:
    a single random crash is not enough to trip recovery."""
    fake = make_fake_ib_client()
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.01,
        max_consecutive_failures=3,
    )

    # Patch perform_check to raise from OUTSIDE its try/except — i.e.,
    # something so unexpected (e.g., asyncio internals) that it bubbles
    # to the monitor's fail-safe. We do this by replacing perform_check
    # entirely.
    crashed = {"n": 0}

    async def crashing_perform_check():
        crashed["n"] += 1
        raise RuntimeError(f"surprise crash #{crashed['n']}")

    health.perform_check = crashing_perform_check

    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.03)  # ~3 iterations
    await health.stop_monitoring()

    # After ~3 crashes, status must be UNHEALTHY via record_failure
    # (not via direct write on the first crash).
    assert crashed["n"] >= 1
    # The counter must reflect those crashes — proving record_failure
    # was called, not the old direct write.
    assert health._consecutive_failures >= 1


@pytest.mark.asyncio
async def test_single_failsafe_crash_does_not_immediately_trip_unhealthy():
    """One crash should bump the counter but stay HEALTHY (counter 1 of 3).
    Before C3, the fail-safe wrote UNHEALTHY immediately on a single crash,
    bypassing the threshold."""
    fake = make_fake_ib_client()
    on_unhealthy = AsyncMock()
    health = ConnectionHealth(
        ib_client=fake,
        ping_interval_seconds=0.5,  # slow so only ONE iteration happens
        max_consecutive_failures=3,
    )

    async def crashing_perform_check():
        raise RuntimeError("one-shot crash")

    health.perform_check = crashing_perform_check

    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.05)  # only one iteration
    await health.stop_monitoring()

    # After one crash: counter == 1, status still HEALTHY (1 of 3).
    assert health._consecutive_failures == 1
    assert health.status is HealthStatus.HEALTHY


# --- C3 part 4: record_success still clears RECOVERING ---


def test_record_success_still_clears_recovering():
    """Regression guard: record_success must continue to transition
    RECOVERING → HEALTHY. This was already the behavior; C3 must not
    break it."""
    health = ConnectionHealth(ib_client=make_fake_ib_client())
    health._status = HealthStatus.RECOVERING
    health._consecutive_failures = 5

    health.record_success()

    assert health.status is HealthStatus.HEALTHY
    assert health._consecutive_failures == 0

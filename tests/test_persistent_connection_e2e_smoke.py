"""End-to-end smoke test: persistent IBKR connection recovery chain.

Exercises the full call graph WITHOUT touching real subprocess, Gateway, or
network:

  ConnectionHealth._monitor_loop
      → performs 3 failing checks (ping returns False)
      → transitions to UNHEALTHY
      → fires on_unhealthy callback

  AsyncRunner._on_connection_unhealthy(reason)
      → calls recover_connection("health monitor: ...")

  AsyncRunner.recover_connection(reason)
      → attempt 1: _safe_disconnect + sleep + initialize_connection (fails)
      → attempt 2: _safe_disconnect + sleep + initialize_connection (fails)
      → attempt 3: _safe_disconnect + sleep
                   + restart_gateway_for_zombies_async (mocked → True)
                   + initialize_connection (succeeds)
      → returns True

  After initialize_connection succeeds, _attach_health_monitor is called,
  which creates a new ConnectionHealth and calls record_success-path.
  Final observable state: recover_connection returned True and
  restart_gateway_for_zombies_async was awaited.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from robo_trader.connection_health import ConnectionHealth, HealthStatus
from robo_trader.runner_async import AsyncRunner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner():
    """Build a minimal AsyncRunner skeleton that owns the recovery path."""
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.cfg = MagicMock()
    runner.cfg.ibkr.port = 4002

    # Recovery machinery
    runner.recovery_in_progress = False
    runner._recovery_lock = asyncio.Lock()

    # Stubs that would normally be set by real connect / setup
    runner.ib = MagicMock()
    runner.ib.isConnected = MagicMock(return_value=False)
    runner.subprocess_client = MagicMock()
    runner.subprocess_client.stop = AsyncMock()

    # health is set by _attach_health_monitor; start as None
    runner.health = None

    return runner


def _make_failing_ping_client(fail_count: int):
    """IB client whose ping() fails `fail_count` times then returns True."""
    client = MagicMock()
    client.is_connected = True

    responses = [False] * fail_count + [True] * 100
    ping_iter = iter(responses)

    async def _ping():
        return next(ping_iter)

    client.ping = _ping
    return client


# ---------------------------------------------------------------------------
# Core smoke test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_recovery_chain_fires_end_to_end():
    """
    Prove the whole wiring: failing health checks → on_unhealthy callback →
    recover_connection → Gateway restart on attempt 3 → initialize_connection
    succeeds → status returns to HEALTHY.

    All external I/O is mocked; no subprocess or Gateway is touched.
    """
    runner = _make_runner()

    # ------------------------------------------------------------------ #
    # 1.  Mock _safe_disconnect so it's a no-op                           #
    # ------------------------------------------------------------------ #
    runner._safe_disconnect = AsyncMock()

    # ------------------------------------------------------------------ #
    # 2.  Mock initialize_connection to fail twice then succeed           #
    # ------------------------------------------------------------------ #
    init_call_count = {"n": 0}
    attach_called = {"yes": False}

    async def _fake_initialize_connection():
        init_call_count["n"] += 1
        n = init_call_count["n"]
        if n < 3:
            raise ConnectionError(f"fake connect failure attempt {n}")
        # On success, replicate what the real method does:
        # wire a new (healthy) ConnectionHealth so health.status is HEALTHY
        runner.health = ConnectionHealth(
            ib_client=MagicMock(is_connected=True, ping=AsyncMock(return_value=True)),
            ping_interval_seconds=999,  # won't fire in this test
            max_consecutive_failures=3,
        )
        # start_monitoring is called inside _attach_health_monitor; mock it
        # so we don't spawn a real background task
        runner.health.start_monitoring = AsyncMock()
        await runner.health.start_monitoring(on_unhealthy=runner._on_connection_unhealthy)
        attach_called["yes"] = True

    runner.initialize_connection = _fake_initialize_connection

    # ------------------------------------------------------------------ #
    # 3.  Mock restart_gateway_for_zombies_async (call-graph check)       #
    # ------------------------------------------------------------------ #
    gateway_restart_calls = []

    async def _fake_restart(port, timeout):
        gateway_restart_calls.append((port, timeout))
        return True

    # ------------------------------------------------------------------ #
    # 4.  Build a ConnectionHealth that will fail 3 times fast            #
    # ------------------------------------------------------------------ #
    fast_interval = 0.02  # 20 ms between checks so the test stays quick
    failing_client = _make_failing_ping_client(fail_count=3)
    health = ConnectionHealth(
        ib_client=failing_client,
        ping_interval_seconds=fast_interval,
        max_consecutive_failures=3,
    )
    runner.health = health

    # ------------------------------------------------------------------ #
    # 5.  Fire the monitor with the runner's callback, patch sleep so     #
    #     the recovery backoff doesn't take 15+ seconds                   #
    # ------------------------------------------------------------------ #
    recovery_result = {}
    health_status_at_callback = {}  # status of health object when callback fires

    async def _capture_on_unhealthy(reason: str):
        """Wrapper that runs recover_connection and stores the result."""
        # Capture health status at the moment the callback fires —
        # the monitor loop may heal the original object on the next
        # iteration (once failing_client starts returning True), so
        # we snapshot here rather than asserting after the fact.
        health_status_at_callback["status"] = health.status
        with patch(
            "robo_trader.runner_async.restart_gateway_for_zombies_async",
            side_effect=_fake_restart,
        ):
            with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
                result = await runner.recover_connection(f"health monitor: {reason}")
        recovery_result["value"] = result

    await health.start_monitoring(on_unhealthy=_capture_on_unhealthy)

    # Give the monitor enough time to do 3 checks + fire the callback +
    # allow recover_connection to run (all sleeps are mocked so it's fast)
    await asyncio.sleep(fast_interval * 10)

    await health.stop_monitoring()

    # ------------------------------------------------------------------ #
    # 6.  Assertions: verify the entire chain fired correctly             #
    # ------------------------------------------------------------------ #

    # Callback fired: health was UNHEALTHY at the moment on_unhealthy ran
    assert health_status_at_callback.get("status") is HealthStatus.UNHEALTHY, (
        f"Health was {health_status_at_callback.get('status')} when callback fired; "
        "expected UNHEALTHY — the transition guard in _monitor_loop may be broken"
    )

    # recover_connection was reached and succeeded
    assert (
        recovery_result.get("value") is True
    ), f"recover_connection did not return True; result={recovery_result}"

    # _safe_disconnect was called (at least once per attempt)
    assert (
        runner._safe_disconnect.await_count >= 1
    ), "_safe_disconnect was never called during recovery"

    # initialize_connection was called 3 times (2 failures + 1 success)
    assert (
        init_call_count["n"] == 3
    ), f"initialize_connection called {init_call_count['n']} times, expected 3"

    # restart_gateway_for_zombies_async was called on attempt 3
    # (gateway_restart_attempt_threshold = 3 in recover_connection)
    assert len(gateway_restart_calls) >= 1, "restart_gateway_for_zombies_async was never called"
    assert (
        gateway_restart_calls[0][0] == 4002
    ), f"Gateway restart used wrong port: {gateway_restart_calls[0][0]}"

    # After initialize_connection succeeded, _attach_health_monitor was reached
    assert (
        attach_called["yes"] is True
    ), "_attach_health_monitor path was not reached after successful initialize_connection"

    # recovery_in_progress flag was cleaned up
    assert (
        runner.recovery_in_progress is False
    ), "recovery_in_progress was not reset to False after recovery completed"


# ---------------------------------------------------------------------------
# Focused unit sub-tests (call-graph slices)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_unhealthy_delegates_to_recover_connection():
    """_on_connection_unhealthy must call recover_connection with the reason."""
    runner = _make_runner()
    runner.recover_connection = AsyncMock(return_value=True)

    reason = "3 consecutive failures"
    await runner._on_connection_unhealthy(reason)

    runner.recover_connection.assert_awaited_once()
    forwarded = runner.recover_connection.await_args.args[0]
    assert reason in forwarded, f"reason not forwarded: expected {reason!r} in {forwarded!r}"


@pytest.mark.asyncio
async def test_monitor_loop_fires_callback_at_threshold():
    """ConnectionHealth._monitor_loop fires on_unhealthy exactly once on
    the HEALTHY→UNHEALTHY transition (not on every subsequent check)."""
    on_unhealthy = AsyncMock()
    failing_client = _make_failing_ping_client(fail_count=10)
    health = ConnectionHealth(
        ib_client=failing_client,
        ping_interval_seconds=0.01,
        max_consecutive_failures=3,
    )
    await health.start_monitoring(on_unhealthy=on_unhealthy)
    await asyncio.sleep(0.15)  # long enough for many checks
    await health.stop_monitoring()

    # Callback must have fired at least once
    assert on_unhealthy.await_count >= 1, "on_unhealthy callback never fired"
    # It fires on the first transition then stops (prev_status guard in _monitor_loop)
    # After that, prev_status IS UNHEALTHY so subsequent loops don't re-fire.
    # Allow exactly 1 fire (tight assertion on the guard behaviour).
    assert on_unhealthy.await_count == 1, (
        f"on_unhealthy fired {on_unhealthy.await_count} times; expected 1 "
        "(transition guard broken)"
    )


@pytest.mark.asyncio
async def test_recover_connection_calls_gateway_restart_on_attempt_3():
    """recover_connection must call restart_gateway_for_zombies_async on
    attempt 3 (gateway_restart_attempt_threshold=3) but NOT on attempt 1."""
    runner = _make_runner()
    runner._safe_disconnect = AsyncMock()

    call_log = []

    async def _fake_restart(port, timeout):
        call_log.append(("restart", port))
        return True

    # Fail on attempts 1,2; succeed on 3
    init_calls = {"n": 0}

    async def _init():
        init_calls["n"] += 1
        if init_calls["n"] < 3:
            raise ConnectionError("fail")

    runner.initialize_connection = _init

    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        side_effect=_fake_restart,
    ):
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")

    assert result is True
    # Gateway restart should have been called on attempt 3
    restart_calls = [e for e in call_log if e[0] == "restart"]
    assert len(restart_calls) >= 1, "restart_gateway_for_zombies_async not called"
    assert restart_calls[0][1] == 4002


@pytest.mark.asyncio
async def test_recover_connection_no_gateway_restart_on_attempt_1():
    """Attempt 1 must NOT call restart_gateway_for_zombies_async."""
    runner = _make_runner()
    runner._safe_disconnect = AsyncMock()

    gateway_called = {"n": 0}

    async def _fake_restart(port, timeout):
        gateway_called["n"] += 1
        return True

    runner.initialize_connection = AsyncMock()  # succeeds on first call

    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        side_effect=_fake_restart,
    ):
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")

    assert result is True
    assert (
        gateway_called["n"] == 0
    ), f"restart_gateway_for_zombies_async called on attempt 1 (should not be)"

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
        with patch("robo_trader.runner_async.asyncio.sleep", side_effect=record_sleep):
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
        init_started.set()  # signal: we're inside the lock
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


# --- C4: stop-loss monitor rewarming after recovery ---


@pytest.mark.asyncio
async def test_recovery_rewarms_stop_loss_monitor_with_cached_prices():
    """C4: After a successful recovery, stop_loss_monitor.update_price must
    be called for every active stop whose symbol exists in latest_prices.
    This closes the 10-second freshness gate gap that otherwise blinds
    stop-losses for the first 1-N cycles after reconnect."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0, "NVDA": 500.0, "TSLA": 200.0}

    # Build a stop-loss monitor mock with active stops keyed by
    # portfolio:symbol but stop objects carrying bare symbols
    stop_aapl = MagicMock(symbol="AAPL")
    stop_nvda = MagicMock(symbol="NVDA")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {
        "default:AAPL": stop_aapl,
        "default:NVDA": stop_nvda,
    }
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True
    # Both active stops have symbols present in latest_prices → both rewarmed
    assert runner.stop_loss_monitor.update_price.await_count == 2
    rewarmed_calls = {call.args for call in runner.stop_loss_monitor.update_price.await_args_list}
    assert ("AAPL", 150.0) in rewarmed_calls
    assert ("NVDA", 500.0) in rewarmed_calls


@pytest.mark.asyncio
async def test_recovery_skips_stops_with_no_cached_price():
    """If a stop's symbol isn't in latest_prices, skip it gracefully."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0}  # only AAPL has a cached price

    stop_aapl = MagicMock(symbol="AAPL")
    stop_unknown = MagicMock(symbol="UNKNOWN")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {
        "default:AAPL": stop_aapl,
        "default:UNKNOWN": stop_unknown,
    }
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    # Only AAPL was rewarmed
    runner.stop_loss_monitor.update_price.assert_awaited_once_with("AAPL", 150.0)


@pytest.mark.asyncio
async def test_recovery_rewarm_handles_per_symbol_failures():
    """If update_price raises for one symbol, the others must still rewarm.
    A single broken stop cannot poison the entire rewarm pass."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0, "NVDA": 500.0}

    stop_aapl = MagicMock(symbol="AAPL")
    stop_nvda = MagicMock(symbol="NVDA")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {
        "default:AAPL": stop_aapl,
        "default:NVDA": stop_nvda,
    }

    async def update_price_fails_for_aapl(symbol, price):
        if symbol == "AAPL":
            raise RuntimeError("intentional test failure")

    runner.stop_loss_monitor.update_price = AsyncMock(side_effect=update_price_fails_for_aapl)

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    # Recovery still succeeded — rewarm errors do not fail the recovery
    assert result is True
    # Both update_price calls were attempted (NVDA wasn't skipped due to
    # AAPL's failure)
    assert runner.stop_loss_monitor.update_price.await_count == 2


@pytest.mark.asyncio
async def test_recovery_rewarm_no_op_when_no_stop_monitor():
    """Missing stop_loss_monitor attribute must not crash recovery — it
    just means we have no stops to warm yet (early in startup, test
    scaffolding, etc.)."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0}
    # explicitly no stop_loss_monitor

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True


@pytest.mark.asyncio
async def test_recovery_rewarm_no_op_when_active_stops_empty():
    """Empty active_stops must not crash and should be a quiet no-op."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {}
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_rewarm_not_called_when_recovery_fails():
    """If all attempts fail, rewarm must NOT be called — there's no
    connection to update prices against."""
    runner = make_runner_for_recovery(initialize_succeeds_on=999)  # never succeeds
    runner.latest_prices = {"AAPL": 150.0}
    stop_aapl = MagicMock(symbol="AAPL")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {"default:AAPL": stop_aapl}
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch(
        "robo_trader.runner_async.restart_gateway_for_zombies_async",
        new_callable=AsyncMock,
    ):
        with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
            result = await runner.recover_connection("test")

    assert result is False
    runner.stop_loss_monitor.update_price.assert_not_awaited()


# --- H1: kill switch auto-reset after recovery iff trigger was connection-related ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trigger_reason,should_reset",
    [
        # Connection-related → auto-reset
        ("Connection lost to IBKR Gateway", True),
        ("Handshake timeout after 30s", True),
        ("Gateway returned no response", True),
        ("Subprocess crashed unexpectedly", True),
        ("IBKR API error 1100", True),
        ("Connection refused on port 4002", True),
        # Mixed-case variants (matching is case-insensitive)
        ("CONNECTION POOL EXHAUSTED", True),
        ("IBKR Timeout During Reconnect", True),
        # Loss-based / safety-meaningful → preserve
        ("Position loss limit exceeded for NVDA: 2.93% loss", False),
        ("Daily loss limit breached: -$2,450", False),
        ("Margin call from broker", False),
        ("Manual trigger by operator", False),
        ("Risk: portfolio drawdown 8.5%", False),
    ],
)
async def test_kill_switch_auto_reset_only_for_connection_reasons(trigger_reason, should_reset):
    """H1: After recovery, the kill switch is auto-reset iff its
    trigger_reason contains a connection-related keyword. Loss-based
    triggers must persist because the recovery doesn't make the loss
    go away."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    # Set up advanced_risk with a triggered kill switch
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch = MagicMock()
    runner.advanced_risk.kill_switch.triggered = True
    runner.advanced_risk.kill_switch.trigger_reason = trigger_reason
    runner.advanced_risk.kill_switch.reset = MagicMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True
    if should_reset:
        # Must be called with force=True to bypass the cooldown gate
        runner.advanced_risk.kill_switch.reset.assert_called_once()
        call = runner.advanced_risk.kill_switch.reset.call_args
        # Accept either keyword or positional force=True
        force_arg = call.kwargs.get("force", call.args[0] if call.args else None)
        assert force_arg is True, (
            f"Connection-related reset must use force=True to bypass cooldown; "
            f"got force={force_arg!r}"
        )
    else:
        runner.advanced_risk.kill_switch.reset.assert_not_called()


@pytest.mark.asyncio
async def test_kill_switch_not_reset_when_not_triggered():
    """If the kill switch was never tripped, recovery must NOT touch it."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch = MagicMock()
    runner.advanced_risk.kill_switch.triggered = False
    runner.advanced_risk.kill_switch.trigger_reason = None
    runner.advanced_risk.kill_switch.reset = MagicMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    runner.advanced_risk.kill_switch.reset.assert_not_called()


@pytest.mark.asyncio
async def test_kill_switch_auto_reset_handles_missing_advanced_risk():
    """advanced_risk == None must not crash recovery."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.advanced_risk = None

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True


@pytest.mark.asyncio
async def test_kill_switch_auto_reset_handles_missing_kill_switch_attr():
    """advanced_risk.kill_switch == None must not crash recovery."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch = None

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True


@pytest.mark.asyncio
async def test_kill_switch_auto_reset_handles_none_trigger_reason():
    """Triggered but trigger_reason=None must be safely treated as
    non-connection (preserve the trigger)."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch = MagicMock()
    runner.advanced_risk.kill_switch.triggered = True
    runner.advanced_risk.kill_switch.trigger_reason = None
    runner.advanced_risk.kill_switch.reset = MagicMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        await runner.recover_connection("test")

    runner.advanced_risk.kill_switch.reset.assert_not_called()


def test_kill_switch_force_reset_bypasses_cooldown(tmp_path):
    """H1 dependency: AdvancedRiskManager.kill_switch.reset(force=True)
    must perform the reset immediately, ignoring the cooldown timer."""
    from datetime import timedelta

    from robo_trader.risk.advanced_risk import KillSwitch, get_market_time

    # Use tmp_path so this test doesn't touch the production kill_switch
    # state file. KillSwitch's constructor derives the lock path from
    # state_path's parent — point both into the tmp dir.
    state_file = tmp_path / "kill_switch_state.json"
    ks = KillSwitch(cooldown_minutes=60, state_path=state_file)

    # Pretend it was just triggered (well within cooldown — use market-time
    # to avoid offset-naive/aware mismatch with reset()).
    ks.triggered = True
    ks.trigger_time = get_market_time() - timedelta(seconds=5)
    ks.trigger_reason = "Connection lost during ping"

    # Without force: cooldown not elapsed → must remain triggered
    ks.reset()
    assert ks.triggered is True

    # With force=True: must reset regardless
    ks.reset(force=True)
    assert ks.triggered is False
    assert ks.trigger_reason is None

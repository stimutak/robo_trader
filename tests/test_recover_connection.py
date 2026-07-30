"""Tests for AsyncRunner.recover_connection.

Per 2026-05-16 design spec: exponential backoff [15, 30, 60, 120, 300],
Gateway restart on attempt >=3, returns bool, mutex via _recovery_lock.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robo_trader.connection_health import HealthStatus
from robo_trader.market_data_contract import (
    BrokerProtectiveQuote,
    MarketDataSource,
    MarketSession,
)
from robo_trader.paper_reduction_gateway import PaperReductionGateway
from robo_trader.protective_quote_evidence import ProtectiveQuoteSource
from robo_trader.reconciliation.runtime_integration import RuntimeReconciliationController
from robo_trader.risk_manager import Position
from robo_trader.runner_async import AsyncRunner
from robo_trader.stop_loss_monitor import StopLossMonitor


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
    reconciliation = object.__new__(RuntimeReconciliationController)
    reconciliation.reconcile_reconnect = AsyncMock()
    runner.reconciliation_controller = reconciliation

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
async def test_recovery_refreshes_separate_paper_gateway_before_success():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway.refresh_diagnostic_connection = AsyncMock()
    runner.paper_reduction_gateway = gateway

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("shared Gateway recovered")

    assert result is True
    gateway.refresh_diagnostic_connection.assert_awaited_once()
    runner.reconciliation_controller.reconcile_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovery_fails_and_disconnects_when_paper_gateway_refresh_fails():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    gateway = object.__new__(PaperReductionGateway)
    gateway._started = False
    gateway.refresh_diagnostic_connection = AsyncMock(
        side_effect=RuntimeError("diagnostic reconnect failed")
    )
    runner.paper_reduction_gateway = gateway

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("shared Gateway recovered")

    assert result is False
    assert gateway.refresh_diagnostic_connection.await_count == 5
    # Every failed post-connect gateway refresh must disconnect the newly
    # installed runner client before the next attempt or final False result.
    assert runner._safe_disconnect.await_count == 10


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
    test_task = asyncio.current_task()
    real_sleep = asyncio.sleep

    async def record_sleep(seconds):
        # Patching runner_async.asyncio.sleep replaces the attribute on the
        # shared asyncio module.  A reconciliation task left alive by an
        # earlier test may therefore call this side effect while this test is
        # running.  Record only the recovery coroutine under test and preserve
        # real scheduling for unrelated tasks.
        if asyncio.current_task() is test_task:
            sleeps.append(seconds)
            return
        await real_sleep(seconds)

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
async def test_recovery_cannot_rewarm_from_cached_prices_without_new_lineage():
    """Cached labels cannot manufacture quote authority after reconnect."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0, "NVDA": 500.0, "TSLA": 200.0}
    event_time = datetime.now(timezone.utc)
    runner.latest_price_times = {
        "AAPL": event_time,
        "NVDA": event_time,
        "TSLA": event_time,
    }
    runner.latest_price_sources = {
        "AAPL": "live_protective",
        "NVDA": "live_protective",
        "TSLA": "historical_bar",
    }

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

    assert result is False
    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_skips_stops_with_no_cached_price():
    """A missing required cached price prevents recovery from resuming."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0}  # only AAPL has a cached price
    event_time = datetime.now(timezone.utc)
    runner.latest_price_times = {"AAPL": event_time}
    runner.latest_price_sources = {"AAPL": "live_protective"}

    stop_aapl = MagicMock(symbol="AAPL")
    stop_unknown = MagicMock(symbol="UNKNOWN")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {
        "default:AAPL": stop_aapl,
        "default:UNKNOWN": stop_unknown,
    }
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is False
    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_rewarm_handles_per_symbol_failures():
    """All symbols are attempted, but any failure blocks recovery."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.latest_prices = {"AAPL": 150.0, "NVDA": 500.0}
    event_time = datetime.now(timezone.utc)
    runner.latest_price_times = {"AAPL": event_time, "NVDA": event_time}
    runner.latest_price_sources = {
        "AAPL": "live_protective",
        "NVDA": "live_protective",
    }

    stop_aapl = MagicMock(symbol="AAPL")
    stop_nvda = MagicMock(symbol="NVDA")
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {
        "default:AAPL": stop_aapl,
        "default:NVDA": stop_nvda,
    }

    async def update_price_fails_for_aapl(symbol, price, *, source_timestamp):
        if symbol == "AAPL":
            raise RuntimeError("intentional test failure")
        return True

    runner.stop_loss_monitor.update_price = AsyncMock(side_effect=update_price_fails_for_aapl)

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is False
    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_accepts_exact_fresh_event_already_owned_by_monitor():
    """A new transport callback may republish exact authority during reconnect."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.portfolio_id = "default"
    now = datetime.now(timezone.utc)
    event_time = now
    runner.latest_prices = {"AAPL": 150.0}
    runner.latest_price_times = {"AAPL": event_time}
    runner.latest_price_sources = {"AAPL": "live_protective"}
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=MagicMock(),
    )
    await monitor.add_stop_loss(
        "AAPL",
        Position("AAPL", 10, Decimal("100")),
    )
    runner.stop_loss_monitor = monitor

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = True
    gateway._diagnostic_recovery_required = False
    gateway.refresh_diagnostic_connection = AsyncMock()
    gateway.refresh_protective_quotes = AsyncMock(
        return_value=(
            BrokerProtectiveQuote(
                schema_version=1,
                symbol="AAPL",
                con_id=265598,
                exchange="SMART",
                primary_exchange="NASDAQ",
                currency="USD",
                security_type="STK",
                price=Decimal("150.0"),
                source_timestamp=event_time,
                retrieval_timestamp=event_time,
                session=MarketSession.REGULAR,
                source=MarketDataSource.IBKR_LIVE_LAST_TRADE,
                source_event_id="recovery-ticker-1",
                transport_generation="recovered-generation",
                market_data_type=1,
            ),
        )
    )
    runner.paper_reduction_gateway = gateway

    async def initialize_with_new_transport_quote() -> None:
        assert await monitor.update_price(
            "AAPL",
            150.0,
            source_timestamp=event_time,
            source=ProtectiveQuoteSource.LIVE_BROKER,
            con_id=265598,
            transport_generation="recovered-generation",
            source_event_id="recovery-ticker-1",
        )
        runner._protective_feed_status = {
            "AAPL": {
                "available": True,
                "live_grade": True,
                "source": "live_protective",
                "con_id": 265598,
                "transport_generation": "recovered-generation",
            }
        }

    runner.initialize_connection = initialize_with_new_transport_quote

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True
    quote = monitor.get_protective_quote_evidence("AAPL")
    assert quote is not None
    assert quote.transport_generation == "recovered-generation"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    (
        "monitor_price",
        "cached_event_offset",
        "monitor_event_offset",
        "receipt_offset",
        "publish_feed_status",
    ),
    [
        (149.0, -2, -2, -2, True),  # different broker price
        (150.0, -2, -1, -2, True),  # different broker event timestamp
        (150.0, -11, -11, -2, True),  # stale broker event
        (150.0, -2, -2, -11, True),  # stale monitor receipt
        (150.0, 1, 1, -2, True),  # exact but future broker event
        (150.0, -2, -2, -2, False),  # missing live-feed status
    ],
)
async def test_recovery_rejects_nonexact_or_unfresh_monitor_evidence(
    monitor_price,
    cached_event_offset,
    monitor_event_offset,
    receipt_offset,
    publish_feed_status,
):
    """Duplicate rejection cannot mask mismatched, stale, or future evidence."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    now = datetime(2026, 7, 23, 15, 0, 5, tzinfo=timezone.utc)
    cached_event_time = now + timedelta(seconds=cached_event_offset)
    runner.latest_prices = {"AAPL": 150.0}
    runner.latest_price_times = {"AAPL": cached_event_time}
    runner.latest_price_sources = {"AAPL": "live_protective"}
    if publish_feed_status:
        runner._protective_feed_status = {
            "AAPL": {
                "available": True,
                "live_grade": True,
                "source": "live_protective",
            }
        }
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {"default:AAPL": MagicMock(symbol="AAPL")}
    runner.stop_loss_monitor.update_price = AsyncMock(return_value=False)
    runner.stop_loss_monitor.last_prices = {"AAPL": monitor_price}
    runner.stop_loss_monitor.price_event_times = {
        "AAPL": now + timedelta(seconds=monitor_event_offset)
    }
    runner.stop_loss_monitor.price_receipt_monotonic = {"AAPL": 1000.0 + receipt_offset}
    runner.stop_loss_monitor._utcnow = MagicMock(return_value=now)
    runner.stop_loss_monitor._monotonic = MagicMock(return_value=1000.0)
    runner.stop_loss_monitor.max_price_age_seconds = 10

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is False
    runner.stop_loss_monitor.update_price.assert_not_awaited()
    assert runner._safe_disconnect.await_count == 2


@pytest.mark.asyncio
async def test_stale_cached_event_after_backoff_keeps_recovery_unhealthy():
    runner = make_runner_for_recovery()
    runner.portfolio_id = "default"
    runner.positions = {}
    event_time = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    elapsed = 0
    runner.latest_prices = {"AAPL": 150.0}
    runner.latest_price_times = {"AAPL": event_time}
    runner.latest_price_sources = {"AAPL": "live_protective"}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {"default:AAPL": MagicMock(symbol="AAPL")}

    async def accept_only_fresh(_symbol, _price, *, source_timestamp):
        age = (event_time + timedelta(seconds=elapsed) - source_timestamp).total_seconds()
        return age <= 10

    runner.stop_loss_monitor.update_price = AsyncMock(side_effect=accept_only_fresh)
    runner.health = MagicMock()
    runner.health._status = HealthStatus.UNHEALTHY
    runner.health.record_success = MagicMock()
    runner.advanced_risk = MagicMock()
    runner.advanced_risk.kill_switch.triggered = True
    runner.advanced_risk.kill_switch.trigger_reason = "Connection lost"

    async def advance_clock(seconds):
        nonlocal elapsed
        elapsed += seconds

    with patch("robo_trader.runner_async.asyncio.sleep", side_effect=advance_clock):
        result = await runner.recover_connection("test")

    assert elapsed == 15
    assert result is False
    assert runner.health._status is HealthStatus.UNHEALTHY
    runner.health.record_success.assert_not_called()
    runner.advanced_risk.kill_switch.reset.assert_not_called()
    runner.initialize_connection.assert_awaited_once()
    runner.stop_loss_monitor.update_price.assert_not_awaited()
    assert runner._safe_disconnect.await_count == 2


@pytest.mark.asyncio
async def test_final_generic_post_connect_failure_disconnects_before_false():
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.positions = {}
    runner.health = MagicMock()
    runner.health._status = HealthStatus.UNHEALTHY
    runner.initialize_connection = AsyncMock()
    runner._assert_existing_position_protection = MagicMock(
        side_effect=RuntimeError("protection assertion infrastructure failed")
    )

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner._recover_connection_locked("test", [0], 3)

    assert result is False
    runner.initialize_connection.assert_awaited_once()
    assert runner.health._status is HealthStatus.UNHEALTHY
    # One disconnect starts the attempt and the second immediately closes the
    # connection that initialize_connection installed before the generic
    # post-connect failure. There is no next attempt to clean it up.
    assert runner._safe_disconnect.await_count == 2


@pytest.mark.asyncio
async def test_recovery_rewarm_no_op_when_no_stop_monitor():
    """Missing stop_loss_monitor attribute must not crash recovery — it
    just means we have no stops to warm yet (early in startup, test
    scaffolding, etc.)."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.positions = {}
    runner.latest_prices = {"AAPL": 150.0}
    # explicitly no stop_loss_monitor

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True


@pytest.mark.asyncio
async def test_recovery_rewarm_no_op_when_active_stops_empty():
    """Empty active_stops must not crash and should be a quiet no-op."""
    runner = make_runner_for_recovery(initialize_succeeds_on=1)
    runner.positions = {}
    runner.latest_prices = {"AAPL": 150.0}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.active_stops = {}
    runner.stop_loss_monitor.update_price = AsyncMock()

    with patch("robo_trader.runner_async.asyncio.sleep", AsyncMock()):
        result = await runner.recover_connection("test")

    assert result is True
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

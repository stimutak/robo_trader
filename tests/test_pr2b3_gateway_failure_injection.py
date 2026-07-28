"""Failure injection for the PR2B3 terminal paper-settlement boundary.

These tests deliberately cross the one-shot paper submission point, then fail
each later stage.  No failure may cause a second executor call or allow another
account order to pass a terminally quarantined gateway.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from robo_trader.execution import PaperExecutor
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    PaperReductionGatewayError,
)
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
    PaperReductionSubmitter,
)
from robo_trader.paper_runtime_settlement import (
    PaperRuntimeProjection,
    PaperRuntimeSettlementError,
    PaperRuntimeSettlementParticipant,
)
from robo_trader.protective_quote_evidence import ProtectiveQuoteSource
from robo_trader.safety import (
    IdempotencyConflict,
    RuntimeAuthorizationBlocked,
    RuntimeStartupBlocked,
    SafetyJournal,
    SafetyRuntimeCoordinator,
)
from robo_trader.safety import readiness as paper_readiness
from robo_trader.stop_loss_monitor import StopLossMonitor
from tests.test_pr2b2_paper_reduction_gateway import (
    CON_ID,
    SYMBOL,
    GatewayHarness,
    _bind_runtime,
    _build_harness,
    _install_broker_boundary,
    _order,
)


@pytest.fixture(autouse=True)
def _enable_contained_terminal_settlement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paper_readiness, "PAPER_TERMINAL_SETTLEMENT_READY", True)


@pytest_asyncio.fixture
async def failure_harness(tmp_path):
    value = await _build_harness(tmp_path)
    try:
        yield value
    finally:
        await value.gateway.close()
        await value.database.close()


async def _ledger_counts(harness: GatewayHarness) -> tuple[int, int]:
    async with harness.database.get_connection() as connection:
        trade_count = (await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone())[
            0
        ]
        settlement_count = (
            await (
                await connection.execute("SELECT COUNT(*) FROM paper_reduction_settlements")
            ).fetchone()
        )[0]
    return trade_count, settlement_count


async def _bind_projection(
    harness: GatewayHarness,
    executor: PaperExecutor,
    apply_callback,
) -> None:
    async def unused_execute(*_args):
        raise AssertionError("failure-injection monitor must not execute a stop")

    monitor = StopLossMonitor(
        execute_reduction=unused_execute,
        risk_manager=None,
        portfolio_id="portfolio-a",
    )
    participant = PaperRuntimeSettlementParticipant(
        "portfolio-a",
        apply_callback=apply_callback,
        quarantine_callback=harness.quarantine_reasons.append,
    )
    harness.gateway.attach_protective_quote_producer("portfolio-a", monitor)
    harness.gateway.register_paper_executor(
        "portfolio-a",
        executor,
        protective_quote_producer=monitor,
        settlement_participant=participant,
    )
    accepted = await monitor.update_price(
        SYMBOL,
        123.45,
        source_timestamp=datetime.now(timezone.utc),
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=CON_ID,
        transport_generation="gateway-generation",
        source_event_id="pr2b3-failure-injection",
    )
    assert accepted is True
    harness.runtime_monitors["portfolio-a"] = monitor


def _unsafe_outcome(
    *,
    order_ref: str,
    status: LocalPaperOrderStatus,
) -> LocalPaperTerminalOutcome:
    partial = status is LocalPaperOrderStatus.PARTIALLY_FILLED
    return LocalPaperTerminalOutcome(
        order_ref=order_ref,
        status=status,
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("1") if partial else Decimal("0"),
        remaining_quantity=Decimal("1") if partial else Decimal("2"),
        exact_fill_price=Decimal("123.45") if partial else None,
        observed_at=datetime.now(timezone.utc),
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=False,
        message=f"injected {status.value.lower()} outcome",
    )


def _unfilled_terminal_outcome(
    *,
    order_ref: str,
    status: LocalPaperOrderStatus,
) -> LocalPaperTerminalOutcome:
    return LocalPaperTerminalOutcome(
        order_ref=order_ref,
        status=status,
        requested_quantity=Decimal("2"),
        filled_quantity=Decimal("0"),
        remaining_quantity=Decimal("2"),
        exact_fill_price=None,
        observed_at=datetime.now(timezone.utc),
        provenance=LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        terminal=True,
        message=f"injected terminal {status.value.lower()} outcome",
    )


def _assert_terminal_quarantine(harness: GatewayHarness) -> None:
    state = harness.journal.replay()
    assert len(state.active_reservations) == 1
    assert len(state.quarantined_reservations) == 1
    assert harness.gateway.can_attempt_order_admission is False
    assert harness.gateway.terminal_quarantine_reason
    assert harness.quarantine_reasons


def _assert_gateway_quarantined_after_terminal_release(harness: GatewayHarness) -> None:
    state = harness.journal.replay()
    assert state.active_reservations == ()
    assert state.quarantined_reservations == ()
    assert harness.gateway.can_attempt_order_admission is False
    assert harness.gateway.terminal_quarantine_reason
    assert harness.quarantine_reasons


def _assert_restart_blocked_by_unresolved_terminal_boundary(
    harness: GatewayHarness,
) -> None:
    replacement = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        harness.journal,
    )
    with pytest.raises(RuntimeStartupBlocked) as captured:
        replacement.start()
    message = str(captured.value)
    assert "ACTIVE_RESERVATION_AT_STARTUP" in message
    assert "QUARANTINED_RESERVATION_AT_STARTUP" in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        LocalPaperOrderStatus.PARTIALLY_FILLED,
        LocalPaperOrderStatus.SUBMITTED,
        LocalPaperOrderStatus.UNKNOWN,
    ],
)
async def test_partial_nonterminal_and_unknown_outcomes_are_quarantined(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
    status: LocalPaperOrderStatus,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    order_ref = f"unsafe-{status.value.lower()}"
    submit_calls = 0

    def injected_submit(_submitter, _envelope, *, pre_position_quantity):
        nonlocal submit_calls
        assert pre_position_quantity == Decimal("7")
        submit_calls += 1
        return _unsafe_outcome(order_ref=order_ref, status=status)

    monkeypatch.setattr(PaperReductionSubmitter, "_submit_once", injected_submit)

    with pytest.raises(
        PaperReductionGatewayError,
        match="partial, nonterminal, or unknown",
    ):
        await harness.gateway.submit_reduction(
            order=_order(order_ref=order_ref),
            portfolio_id="portfolio-a",
        )

    assert submit_calls == 1
    assert executor.fills == {}
    assert await _ledger_counts(harness) == (0, 0)
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)

    with pytest.raises(PaperReductionGatewayError, match="terminally quarantined"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref=f"blocked-after-{order_ref}"),
            portfolio_id="portfolio-a",
        )
    assert submit_calls == 1


@pytest.mark.asyncio
async def test_failure_after_durable_dispatch_before_executor_submit_blocks_restart(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    submit_calls = 0

    def fail_before_executor(_submitter, _envelope, *, pre_position_quantity):
        nonlocal submit_calls
        assert pre_position_quantity == Decimal("7")
        submit_calls += 1
        raise RuntimeError("injected failure before executor submit")

    monkeypatch.setattr(PaperReductionSubmitter, "_submit_once", fail_before_executor)

    with pytest.raises(RuntimeError, match="before executor submit"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="durable-dispatch-before-submit"),
            portfolio_id="portfolio-a",
        )

    assert submit_calls == 1
    assert executor.fills == {}
    assert await _ledger_counts(harness) == (0, 0)
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)

    with pytest.raises(PaperReductionGatewayError, match="terminally quarantined"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="durable-dispatch-no-retry"),
            portfolio_id="portfolio-a",
        )
    assert submit_calls == 1


@pytest.mark.asyncio
async def test_consume_post_commit_failure_latches_terminal_quarantine(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consume error may happen after OUTCOME_UNKNOWN is already durable."""

    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    original_consume = harness.coordinator._consume_authorization_for_paper_submission

    def consume_then_fail(authorization, proof):
        original_consume(authorization, proof)
        raise RuntimeError("injected post-commit consume verification failure")

    monkeypatch.setattr(
        harness.coordinator,
        "_consume_authorization_for_paper_submission",
        consume_then_fail,
    )

    with pytest.raises(RuntimeError, match="post-commit consume verification"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="consume-post-commit-failure"),
            portfolio_id="portfolio-a",
        )

    assert executor.fills == {}
    replay = harness.journal.replay()
    assert len(replay.active_reservations) == 1
    assert replay.active_reservations[0].outcome_unknown is True
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)


@pytest.mark.asyncio
async def test_newer_stream_quote_does_not_break_fresh_latched_stop_authority(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    monitor = harness.runtime_monitors["portfolio-a"]
    latched = monitor.get_protective_quote_evidence(SYMBOL)
    assert latched is not None

    async def publish_newer_quote() -> None:
        accepted = await monitor.update_price(
            SYMBOL,
            122.95,
            source_timestamp=datetime.now(timezone.utc),
            source=ProtectiveQuoteSource.LIVE_BROKER,
            con_id=CON_ID,
            transport_generation="gateway-generation",
            source_event_id="newer-after-stop-crossing",
        )
        assert accepted is True

    _install_broker_boundary(
        harness,
        monkeypatch,
        before_final=publish_newer_quote,
    )
    outcome = await harness.gateway.submit_reduction(
        order=_order(order_ref="fresh-latched-with-newer-stream-quote"),
        portfolio_id="portfolio-a",
        protective_quote=latched,
    )

    assert outcome.ok is True
    assert len(executor.fills) == 1
    assert harness.journal.replay().active_reservations == ()


@pytest.mark.asyncio
async def test_successful_terminal_restart_never_reauthorizes_same_order(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    first_executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", first_executor)
    _install_broker_boundary(harness, monkeypatch)
    order = _order(order_ref="terminal-success-survives-restart")

    result = await harness.gateway.submit_reduction(
        order=order,
        portfolio_id="portfolio-a",
    )
    assert result.ok is True
    assert len(first_executor.fills) == 1
    assert harness.journal.replay().active_reservations == ()

    await harness.gateway.close()
    reopened_journal = SafetyJournal(harness.context.runtime_contract.safety_journal_path)
    restarted = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        reopened_journal,
    )
    restarted.start()
    fresh_gateway = PaperReductionGateway(
        harness.context,
        restarted,
        harness.database,
    )
    fresh_gateway._client.start = AsyncMock(return_value=None)
    fresh_gateway._client.connect = AsyncMock(return_value=True)
    fresh_gateway._client.ping = AsyncMock(return_value=True)
    fresh_gateway._client.stop = AsyncMock(return_value=None)
    await fresh_gateway.start()
    fresh_harness = GatewayHarness(
        harness.context,
        harness.database,
        reopened_journal,
        restarted,
        fresh_gateway,
    )
    second_executor = PaperExecutor()
    try:
        await _bind_runtime(fresh_harness, "portfolio-a", second_executor)
        _install_broker_boundary(fresh_harness, monkeypatch)
        with pytest.raises(
            IdempotencyConflict,
            match="different authorization evidence",
        ):
            await fresh_gateway.submit_reduction(
                order=order,
                portfolio_id="portfolio-a",
            )
        assert second_executor.fills == {}
        assert len(first_executor.fills) == 1
    finally:
        await fresh_gateway.close()


@pytest.mark.asyncio
async def test_terminal_rejection_releases_once_and_restart_cannot_retry(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    rejection_calls = 0

    def reject(_submitter, _envelope, *, pre_position_quantity):
        nonlocal rejection_calls
        assert pre_position_quantity == Decimal("7")
        rejection_calls += 1
        return _unfilled_terminal_outcome(
            order_ref="terminal-rejection-survives-restart",
            status=LocalPaperOrderStatus.REJECTED,
        )

    monkeypatch.setattr(PaperReductionSubmitter, "_submit_once", reject)
    first_executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", first_executor)
    _install_broker_boundary(harness, monkeypatch)
    order = _order(order_ref="terminal-rejection-survives-restart")

    result = await harness.gateway.submit_reduction(
        order=order,
        portfolio_id="portfolio-a",
    )
    assert result.status is LocalPaperOrderStatus.REJECTED
    assert result.terminal is True
    assert result.filled_quantity == 0
    assert rejection_calls == 1
    assert await _ledger_counts(harness) == (0, 1)
    assert harness.journal.replay().active_reservations == ()
    _assert_gateway_quarantined_after_terminal_release(harness)

    await harness.gateway.close()
    reopened_journal = SafetyJournal(harness.context.runtime_contract.safety_journal_path)
    restarted = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        reopened_journal,
    )
    restarted.start()
    fresh_gateway = PaperReductionGateway(
        harness.context,
        restarted,
        harness.database,
    )
    fresh_gateway._client.start = AsyncMock(return_value=None)
    fresh_gateway._client.connect = AsyncMock(return_value=True)
    fresh_gateway._client.ping = AsyncMock(return_value=True)
    fresh_gateway._client.stop = AsyncMock(return_value=None)
    await fresh_gateway.start()
    fresh_harness = GatewayHarness(
        harness.context,
        harness.database,
        reopened_journal,
        restarted,
        fresh_gateway,
    )
    try:
        await _bind_runtime(fresh_harness, "portfolio-a", PaperExecutor())
        _install_broker_boundary(fresh_harness, monkeypatch)
        with pytest.raises((IdempotencyConflict, RuntimeAuthorizationBlocked)):
            await fresh_gateway.submit_reduction(
                order=order,
                portfolio_id="portfolio-a",
            )
        assert rejection_calls == 1
    finally:
        await fresh_gateway.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [LocalPaperOrderStatus.CANCELLED, LocalPaperOrderStatus.EXPIRED],
)
async def test_unfilled_terminal_status_releases_exactly_once(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
    status: LocalPaperOrderStatus,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    order_ref = f"terminal-{status.value.lower()}"
    submit_calls = 0

    def terminal_without_fill(_submitter, _envelope, *, pre_position_quantity):
        nonlocal submit_calls
        assert pre_position_quantity == Decimal("7")
        submit_calls += 1
        return _unfilled_terminal_outcome(order_ref=order_ref, status=status)

    monkeypatch.setattr(
        PaperReductionSubmitter,
        "_submit_once",
        terminal_without_fill,
    )
    result = await harness.gateway.submit_reduction(
        order=_order(order_ref=order_ref),
        portfolio_id="portfolio-a",
    )

    assert result.status is status
    assert result.terminal is True
    assert submit_calls == 1
    assert executor.fills == {}
    assert await _ledger_counts(harness) == (0, 1)
    assert harness.journal.replay().active_reservations == ()
    _assert_gateway_quarantined_after_terminal_release(harness)
    restarted = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        SafetyJournal(harness.context.runtime_contract.safety_journal_path),
    )
    restarted.start()
    assert restarted.started is True

    with pytest.raises(PaperReductionGatewayError, match="terminally quarantined"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref=order_ref),
            portfolio_id="portfolio-a",
        )
    assert submit_calls == 1


@pytest.mark.asyncio
async def test_database_failure_after_fill_quarantines_without_executor_retry(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    commit = AsyncMock(side_effect=RuntimeError("injected DB settlement failure"))
    monkeypatch.setattr(harness.database, "commit_paper_reduction_outcome", commit)

    with pytest.raises(RuntimeError, match="injected DB settlement failure"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="db-fails-after-fill"),
            portfolio_id="portfolio-a",
        )

    assert len(executor.fills) == 1
    commit.assert_awaited_once()
    assert await _ledger_counts(harness) == (0, 0)
    assert harness.runtime_positions == {}
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)

    with pytest.raises(PaperReductionGatewayError, match="terminally quarantined"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="db-failure-must-not-retry"),
            portfolio_id="portfolio-a",
        )
    assert len(executor.fills) == 1
    commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_after_trade_insert_fault_rolls_back_and_blocks_restart(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)

    def fault(step: str) -> None:
        if step == "AFTER_TRADE_INSERT":
            raise RuntimeError("injected AFTER_TRADE_INSERT failure")

    harness.database._paper_settlement_fault_hook = fault
    try:
        with pytest.raises(RuntimeError, match="AFTER_TRADE_INSERT"):
            await harness.gateway.submit_reduction(
                order=_order(order_ref="fault-after-trade-insert"),
                portfolio_id="portfolio-a",
            )
    finally:
        harness.database._paper_settlement_fault_hook = None

    assert len(executor.fills) == 1
    assert await _ledger_counts(harness) == (0, 0)
    allocation = await harness.database.get_safety_allocation_snapshot(
        SYMBOL,
        runtime_contract=harness.context.runtime_contract,
    )
    position = next(row for row in allocation.allocations if row.portfolio_id == "portfolio-a")
    assert position.quantity == Decimal("7")
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_mode", ["raises", "mismatch"])
async def test_projection_failure_after_commit_withholds_journal_release(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
    projection_mode: str,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()

    async def apply(receipt):
        if projection_mode == "raises":
            raise RuntimeError("injected runtime projection failure")
        expected = receipt.request.expected_post_position_quantity
        return PaperRuntimeProjection(
            settlement_id=receipt.settlement_id,
            settlement_receipt_fingerprint=receipt.fingerprint(),
            portfolio_id=receipt.request.portfolio_id,
            symbol=receipt.request.symbol,
            runner_position_quantity=expected + Decimal("1"),
            portfolio_position_quantity=expected,
            account_cash=receipt.request.expected_post_cash,
            account_realized_pnl=receipt.request.expected_post_realized_pnl,
            risk_visible_daily_pnl_before=receipt.request.expected_pre_daily_pnl,
            risk_visible_daily_pnl=receipt.request.expected_post_daily_pnl,
            protective_stop_quantity=None if expected == 0 else expected,
            advanced_risk_position_quantity=None,
            advanced_risk_position_avg_price=None,
            advanced_risk_total_pnl=None,
            advanced_risk_daily_pnl=None,
        )

    await _bind_projection(harness, executor, apply)
    _install_broker_boundary(harness, monkeypatch)
    expected_exception = (
        RuntimeError if projection_mode == "raises" else PaperRuntimeSettlementError
    )

    with pytest.raises(expected_exception):
        await harness.gateway.submit_reduction(
            order=_order(order_ref=f"projection-{projection_mode}"),
            portfolio_id="portfolio-a",
        )

    assert len(executor.fills) == 1
    assert await _ledger_counts(harness) == (1, 1)
    allocation = await harness.database.get_safety_allocation_snapshot(
        SYMBOL,
        runtime_contract=harness.context.runtime_contract,
    )
    position = next(row for row in allocation.allocations if row.portfolio_id == "portfolio-a")
    assert position.quantity == Decimal("5")
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)


@pytest.mark.asyncio
async def test_journal_release_failure_preserves_committed_projection_and_blocks_retry(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    release = Mock(side_effect=RuntimeError("injected journal release failure"))
    monkeypatch.setattr(
        harness.coordinator,
        "release_after_local_paper_settlement",
        release,
    )

    with pytest.raises(RuntimeError, match="injected journal release failure"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="journal-release-failure"),
            portfolio_id="portfolio-a",
        )

    assert len(executor.fills) == 1
    assert await _ledger_counts(harness) == (1, 1)
    assert harness.runtime_positions["portfolio-a"] == Decimal("5")
    release.assert_called_once()
    _assert_terminal_quarantine(harness)
    _assert_restart_blocked_by_unresolved_terminal_boundary(harness)

    with pytest.raises(PaperReductionGatewayError, match="terminally quarantined"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="release-failure-must-not-retry"),
            portfolio_id="portfolio-a",
        )
    assert len(executor.fills) == 1
    release.assert_called_once()


@pytest.mark.asyncio
async def test_cancellation_after_fill_drains_settlement_and_release_before_returning(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    original_commit = harness.database.commit_paper_reduction_outcome
    commit_entered = asyncio.Event()
    release_commit = asyncio.Event()

    async def blocking_commit(*args, **kwargs):
        commit_entered.set()
        await release_commit.wait()
        return await original_commit(*args, **kwargs)

    monkeypatch.setattr(
        harness.database,
        "commit_paper_reduction_outcome",
        blocking_commit,
    )
    operation = asyncio.create_task(
        harness.gateway.submit_reduction(
            order=_order(order_ref="cancel-after-fill"),
            portfolio_id="portfolio-a",
        )
    )
    await commit_entered.wait()
    assert len(executor.fills) == 1

    operation.cancel()
    await asyncio.sleep(0)
    assert operation.done() is False
    assert harness.gateway._account_order_gate.locked() is True
    assert await _ledger_counts(harness) == (0, 0)

    release_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert len(executor.fills) == 1
    assert await _ledger_counts(harness) == (1, 1)
    assert harness.runtime_positions["portfolio-a"] == Decimal("5")
    assert harness.journal.replay().active_reservations == ()
    assert harness.gateway.can_attempt_order_admission is False
    assert harness.quarantine_reasons


@pytest.mark.asyncio
async def test_terminal_quarantine_emits_one_operator_diagnostic_and_one_freeze(
    failure_harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = failure_harness
    executor = PaperExecutor()
    await _bind_runtime(harness, "portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)
    monkeypatch.setattr(
        harness.database,
        "commit_paper_reduction_outcome",
        AsyncMock(side_effect=RuntimeError("diagnostic settlement failure")),
    )

    with caplog.at_level("CRITICAL", logger="robo_trader.paper_reduction_gateway"):
        with pytest.raises(RuntimeError, match="diagnostic settlement failure"):
            await harness.gateway.submit_reduction(
                order=_order(order_ref="terminal-diagnostic"),
                portfolio_id="portfolio-a",
            )

    diagnostics = [
        record
        for record in caplog.records
        if record.getMessage().startswith("event=paper_reduction_gateway_terminal_quarantine")
    ]
    assert len(diagnostics) == 1
    assert len(harness.quarantine_reasons) == 1
    assert "paper terminal settlement failed" in diagnostics[0].getMessage()
    assert harness.gateway.can_attempt_order_admission is False
    assert len(executor.fills) == 1

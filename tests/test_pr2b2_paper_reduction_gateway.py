"""Focused adversarial coverage for the account-scoped paper reduction gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from robo_trader.broker_safety_evidence import (
    BrokerContractSafetySnapshot,
    BrokerSafetyContract,
    _issue_broker_contract_snapshot_capability,
    _produce_broker_contract_safety_snapshot,
)
from robo_trader.config import _derive_safety_account_scope
from robo_trader.database_async import (
    AsyncTradingDatabase,
    SafetyAllocationSnapshotError,
)
from robo_trader.execution import Order, PaperExecutor
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    PaperReductionGatewayError,
)
from robo_trader.paper_reduction_submitter import PaperReductionSubmissionError
from robo_trader.reconciliation.errors import RuntimeSafetyError as RuntimeIdentityError
from robo_trader.reconciliation.identity import (
    RuntimeSafetyContext,
    validate_runtime_safety,
)
from robo_trader.safety import (
    IdempotencyConflict,
    PaperExecutionIdentity,
    ReservationConflict,
    RuntimeSafetyError,
    RuntimeStartupBlocked,
    SafetyJournal,
    SafetyRuntimeCoordinator,
    ValidationError,
)
from robo_trader.safety import readiness as paper_readiness
from robo_trader.safety_runtime_evidence import (
    assemble_local_paper_safety_evidence,
)

ACCOUNT = "DU1234567"
SCOPE_KEY = "0123456789abcdef" * 4
ACCOUNT_SCOPE = _derive_safety_account_scope(SCOPE_KEY, ACCOUNT)
SYMBOL = "AAPL"
CON_ID = 265598


@pytest.fixture(autouse=True)
def _enable_gateway_only_for_contained_tests(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(paper_readiness, "PAPER_TERMINAL_SETTLEMENT_READY", True)


@dataclass
class GatewayHarness:
    context: RuntimeSafetyContext
    database: AsyncTradingDatabase
    journal: SafetyJournal
    coordinator: SafetyRuntimeCoordinator
    gateway: PaperReductionGateway


def _runtime_context(
    tmp_path: Path,
    *,
    database_path: str | None = None,
) -> RuntimeSafetyContext:
    ibc = tmp_path / "config" / "ibc" / "config.ini"
    ibc.parent.mkdir(parents=True)
    ibc.write_text("ReadOnlyApi=yes\nTradingMode=paper\n")
    configured_database_path = database_path or str(tmp_path / "paper-ledger.db")
    journal_path = tmp_path / "paper-safety.db"
    environment = {
        "ENVIRONMENT": "dev",
        "EXECUTION_MODE": "paper",
        "TRADING_MODE": "paper",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4002",
        "IBKR_READONLY": "true",
        "IBKR_CLIENT_ID": "1",
        "IBKR_RECONCILIATION_CLIENT_ID": "7",
        "IBKR_ACCOUNT": ACCOUNT,
        "IBKR_APPROVED_ACCOUNTS": ACCOUNT,
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_STATE_NAMESPACE": "paper",
        "RT_DB_PATH": configured_database_path,
        "SAFETY_ACCOUNT_SCOPE_KEY": SCOPE_KEY,
        "SAFETY_ACCOUNT_SCOPE": ACCOUNT_SCOPE,
        "SAFETY_JOURNAL_PATH": str(journal_path),
        "MODEL_ARTIFACT_SET": "gateway-tests",
        "BUILD_ID": "gateway-tests",
    }
    return validate_runtime_safety(tmp_path, environment)


async def _seed_allocations(
    database: AsyncTradingDatabase,
    *,
    portfolio_a_quantity: int = 7,
    portfolio_b_quantity: int = 3,
) -> None:
    async with database.get_connection() as connection:
        await connection.executemany(
            "INSERT INTO portfolios (id, name) VALUES (?, ?)",
            (("portfolio-a", "Portfolio A"), ("portfolio-b", "Portfolio B")),
        )
        await connection.commit()
    await database.update_position(
        SYMBOL,
        portfolio_a_quantity,
        100.0,
        101.0,
        portfolio_id="portfolio-a",
    )
    await database.update_position(
        SYMBOL,
        portfolio_b_quantity,
        100.0,
        101.0,
        portfolio_id="portfolio-b",
    )


async def _build_harness(
    tmp_path: Path,
    *,
    portfolio_a_quantity: int = 7,
    portfolio_b_quantity: int = 3,
) -> GatewayHarness:
    context = _runtime_context(tmp_path)
    database = AsyncTradingDatabase(Path(context.runtime_contract.database_path), pool_size=1)
    await database.initialize()
    await _seed_allocations(
        database,
        portfolio_a_quantity=portfolio_a_quantity,
        portfolio_b_quantity=portfolio_b_quantity,
    )
    journal = SafetyJournal(Path(context.runtime_contract.safety_journal_path))
    journal.initialize(
        execution_domain_scope=context.runtime_contract.safety_execution_domain_scope,
        account_scope=context.runtime_contract.safety_account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            context.runtime_contract.safety_execution_domain_scope,
            context.runtime_contract.safety_account_scope,
        ),
        journal,
    )
    coordinator.start()
    gateway = PaperReductionGateway(context, coordinator, database)
    gateway._client.start = AsyncMock(return_value=None)
    gateway._client.connect = AsyncMock(return_value=True)
    gateway._client.ping = AsyncMock(return_value=True)
    gateway._client.stop = AsyncMock(return_value=None)
    await gateway.start()
    return GatewayHarness(context, database, journal, coordinator, gateway)


@pytest.mark.asyncio
async def test_gateway_start_is_blocked_at_submission_boundary_until_pr2b3(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(paper_readiness, "PAPER_TERMINAL_SETTLEMENT_READY", False)
    context = _runtime_context(tmp_path)
    database = AsyncTradingDatabase(Path(context.runtime_contract.database_path), pool_size=1)
    await database.initialize()
    journal = SafetyJournal(Path(context.runtime_contract.safety_journal_path))
    journal.initialize(
        execution_domain_scope=context.runtime_contract.safety_execution_domain_scope,
        account_scope=context.runtime_contract.safety_account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            context.runtime_contract.safety_execution_domain_scope,
            context.runtime_contract.safety_account_scope,
        ),
        journal,
    )
    coordinator.start()
    gateway = PaperReductionGateway(context, coordinator, database)
    gateway._client.start = AsyncMock(return_value=None)

    try:
        with pytest.raises(RuntimeError, match="PR 2B.3 terminal settlement"):
            await gateway.start()
        gateway._client.start.assert_not_awaited()
        assert gateway.started is False
    finally:
        await database.close()


@pytest_asyncio.fixture
async def harness(tmp_path: Path):
    value = await _build_harness(tmp_path)
    try:
        yield value
    finally:
        await value.gateway.close()
        await value.database.close()


@pytest.mark.asyncio
async def test_gateway_start_cancellation_stops_partial_client(
    harness: GatewayHarness,
):
    await harness.gateway.close()
    harness.gateway._client.stop.reset_mock()
    harness.gateway._client.start = AsyncMock(return_value=None)
    harness.gateway._client.connect = AsyncMock(
        side_effect=asyncio.CancelledError,
    )

    with pytest.raises(asyncio.CancelledError):
        await harness.gateway.start()

    harness.gateway._client.stop.assert_awaited_once()
    assert harness.gateway.started is False


@pytest.mark.asyncio
async def test_entry_serialization_fails_closed_when_diagnostic_ping_is_false(
    harness: GatewayHarness,
) -> None:
    harness.gateway._client.ping = AsyncMock(return_value=False)
    entered = False

    with pytest.raises(PaperReductionGatewayError, match="unavailable for entry"):
        async with harness.gateway.serialize_entry():
            entered = True

    assert entered is False
    assert harness.gateway.started is False
    harness.gateway._client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_entry_serialization_fails_closed_when_diagnostic_ping_errors(
    harness: GatewayHarness,
) -> None:
    harness.gateway._client.ping = AsyncMock(side_effect=OSError("diagnostic failed"))

    with pytest.raises(PaperReductionGatewayError, match="health could not be proven"):
        async with harness.gateway.serialize_entry():
            pytest.fail("entry admission must not yield with an unhealthy diagnostic client")

    assert harness.gateway.started is False


@pytest.mark.asyncio
async def test_refresh_reconnects_and_proves_gateway_health_before_started(
    harness: GatewayHarness,
) -> None:
    client = harness.gateway._client
    client.start.reset_mock()
    client.connect.reset_mock()
    client.ping.reset_mock()
    client.stop.reset_mock()

    await harness.gateway.refresh_diagnostic_connection()

    assert harness.gateway.started is True
    client.stop.assert_awaited_once()
    client.start.assert_awaited_once()
    client.connect.assert_awaited_once_with(
        host=harness.context.diagnostic_connection.host,
        port=harness.context.diagnostic_connection.port,
        client_id=harness.context.diagnostic_connection.client_id,
        readonly=True,
        timeout=30.0,
    )
    client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_failure_keeps_gateway_stopped_and_cleans_worker(
    harness: GatewayHarness,
) -> None:
    client = harness.gateway._client
    client.start.reset_mock()
    client.connect.reset_mock()
    client.ping = AsyncMock(return_value=False)
    client.stop.reset_mock()

    with pytest.raises(PaperReductionGatewayError, match="did not recover"):
        await harness.gateway.refresh_diagnostic_connection()

    assert harness.gateway.started is False
    assert client.stop.await_count == 2
    client.start.assert_awaited_once()
    client.connect.assert_awaited_once()
    client.ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_identity_drift_drains_stop_through_repeated_cancellation(
    harness: GatewayHarness,
) -> None:
    stop_entered = asyncio.Event()
    release_stop = asyncio.Event()
    stop_finished = asyncio.Event()

    async def blocking_stop() -> None:
        stop_entered.set()
        await release_stop.wait()
        stop_finished.set()

    harness.gateway._runtime_context = object()
    harness.gateway._client.start.reset_mock()
    harness.gateway._client.stop = AsyncMock(side_effect=blocking_stop)
    refresh = asyncio.create_task(harness.gateway.refresh_diagnostic_connection())
    await stop_entered.wait()

    refresh.cancel()
    await asyncio.sleep(0)
    assert refresh.done() is False
    refresh.cancel()
    await asyncio.sleep(0)
    assert refresh.done() is False
    release_stop.set()

    with pytest.raises(RuntimeIdentityError, match="validated RuntimeSafetyContext"):
        await refresh

    assert stop_finished.is_set()
    assert harness.gateway.started is False
    harness.gateway._client.start.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_identity_drift_preserves_primary_and_logs_stop_failure(
    harness: GatewayHarness,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness.gateway._runtime_context = object()
    harness.gateway._client.start.reset_mock()
    original_stop = harness.gateway._client.stop
    harness.gateway._client.stop = AsyncMock(side_effect=OSError("diagnostic client stop failed"))

    try:
        with caplog.at_level(
            "ERROR",
            logger="robo_trader.paper_reduction_gateway",
        ):
            with pytest.raises(
                RuntimeIdentityError,
                match="validated RuntimeSafetyContext",
            ):
                await harness.gateway.refresh_diagnostic_connection()

        matching_records = [
            record
            for record in caplog.records
            if record.getMessage()
            == "event=paper_reduction_gateway_client_stop_failed_after_primary_error"
        ]
        assert len(matching_records) == 1
        assert matching_records[0].exc_info is not None
        assert matching_records[0].exc_info[0] is OSError
        assert str(matching_records[0].exc_info[1]) == "diagnostic client stop failed"
        assert harness.gateway.started is False
        harness.gateway._client.start.assert_not_awaited()
    finally:
        harness.gateway._client.stop = original_stop


def _broker_snapshot(
    context: RuntimeSafetyContext,
    *,
    con_id: int = CON_ID,
    generation: str = "gateway-generation",
    symbol: str = SYMBOL,
) -> BrokerContractSafetySnapshot:
    now = datetime.now(timezone.utc)
    connection = context.diagnostic_connection
    capability = _issue_broker_contract_snapshot_capability(
        context,
        connection_identity=(
            connection.host,
            connection.port,
            connection.client_id,
            connection.readonly,
        ),
        transport_generation=generation,
        requested_symbol=symbol,
    )
    contract = BrokerSafetyContract(
        con_id=con_id,
        symbol=symbol,
        local_symbol=symbol,
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
    )
    return _produce_broker_contract_safety_snapshot(
        capability=capability,
        broker_time_before=now - timedelta(milliseconds=2),
        broker_time_after=now - timedelta(milliseconds=1),
        retrieved_at=now,
        snapshot_id=f"broker-contract-{generation}-{con_id}-{now.timestamp()}",
        source="gateway-test-broker",
        qualified_contract=contract,
    )


def _install_broker_boundary(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    initial_factory=None,
    final_factory=None,
    before_final=None,
) -> None:
    initial_factory = initial_factory or (lambda: _broker_snapshot(harness.context))
    final_factory = final_factory or (lambda: _broker_snapshot(harness.context))

    async def get_snapshot(context, symbol, *, max_age_seconds=30.0):
        assert context is harness.context
        assert symbol == SYMBOL
        assert max_age_seconds == 30.0
        return initial_factory()

    async def run_locked(
        context,
        symbol,
        callback,
        *,
        max_age_seconds=30.0,
    ):
        assert context is harness.context
        assert symbol == SYMBOL
        assert max_age_seconds == 30.0
        if before_final is not None:
            await before_final()
        return await callback(final_factory())

    monkeypatch.setattr(
        harness.gateway._client,
        "get_broker_contract_safety_snapshot",
        get_snapshot,
    )
    monkeypatch.setattr(
        harness.gateway._client,
        "run_with_locked_broker_contract_safety_snapshot",
        run_locked,
    )


def _order(
    *,
    side: str = "SELL",
    quantity: int = 2,
    price: Decimal = Decimal("123.4500"),
    order_ref: str = "gateway-reduction-1",
) -> Order:
    return Order(
        symbol=SYMBOL,
        quantity=quantity,
        side=side,
        price=price,
        order_ref=order_ref,
    )


def test_gateway_requires_exact_runtime_coordinator_database_and_executor_binding(
    harness: GatewayHarness,
    tmp_path: Path,
):
    with pytest.raises(RuntimeIdentityError, match="not validation-produced"):
        PaperReductionGateway(replace(harness.context), harness.coordinator, harness.database)

    wrong_database = AsyncTradingDatabase(tmp_path / "wrong-ledger.db")
    with pytest.raises(PaperReductionGatewayError, match="runtime ledger path"):
        PaperReductionGateway(harness.context, harness.coordinator, wrong_database)

    other_scope = _derive_safety_account_scope("fedcba9876543210" * 4, ACCOUNT)
    other_journal = SafetyJournal(tmp_path / "other-safety.db")
    other_journal.initialize(
        execution_domain_scope=(harness.context.runtime_contract.safety_execution_domain_scope),
        account_scope=other_scope,
    )
    other_coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            harness.context.runtime_contract.safety_execution_domain_scope,
            other_scope,
        ),
        other_journal,
    )
    other_coordinator.start()
    with pytest.raises(PaperReductionGatewayError, match="coordinator.*do not match"):
        PaperReductionGateway(harness.context, other_coordinator, harness.database)

    alternate_journal = SafetyJournal(tmp_path / "same-scope-other-safety.db")
    alternate_journal.initialize(
        execution_domain_scope=(harness.context.runtime_contract.safety_execution_domain_scope),
        account_scope=harness.context.runtime_contract.safety_account_scope,
    )
    alternate_coordinator = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        alternate_journal,
    )
    alternate_coordinator.start()
    with pytest.raises(PaperReductionGatewayError, match="configured safety journal"):
        PaperReductionGateway(
            harness.context,
            alternate_coordinator,
            harness.database,
        )

    unstarted_journal = SafetyJournal(tmp_path / "unstarted-safety.db")
    unstarted_journal.initialize(
        execution_domain_scope=(harness.context.runtime_contract.safety_execution_domain_scope),
        account_scope=harness.context.runtime_contract.safety_account_scope,
    )
    unstarted = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        unstarted_journal,
    )
    with pytest.raises(PaperReductionGatewayError, match="started exact"):
        PaperReductionGateway(harness.context, unstarted, harness.database)

    executor = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor)
    harness.gateway.register_paper_executor("portfolio-a", executor)
    with pytest.raises(PaperReductionGatewayError, match="already registered"):
        harness.gateway.register_paper_executor("portfolio-a", PaperExecutor())
    with pytest.raises(PaperReductionGatewayError, match="exactly PaperExecutor"):
        harness.gateway.register_paper_executor("portfolio-b", object())  # type: ignore[arg-type]

    class ExecutorSubclass(PaperExecutor):
        pass

    with pytest.raises(PaperReductionGatewayError, match="exactly PaperExecutor"):
        harness.gateway.register_paper_executor("portfolio-b", ExecutorSubclass())


def test_gateway_accepts_configured_relative_runtime_ledger_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("robo_trader.config._PROJECT_ROOT", tmp_path)
    context = _runtime_context(tmp_path, database_path="paper-ledger.db")
    database = AsyncTradingDatabase(Path(context.runtime_contract.database_path))
    journal = SafetyJournal(Path(context.runtime_contract.safety_journal_path))
    journal.initialize(
        execution_domain_scope=context.runtime_contract.safety_execution_domain_scope,
        account_scope=context.runtime_contract.safety_account_scope,
    )
    coordinator = SafetyRuntimeCoordinator(
        PaperExecutionIdentity(
            context.runtime_contract.safety_execution_domain_scope,
            context.runtime_contract.safety_account_scope,
        ),
        journal,
    )
    coordinator.start()

    gateway = PaperReductionGateway(context, coordinator, database)

    assert Path(context.runtime_contract.database_path).is_absolute()
    assert database.db_path == Path(context.runtime_contract.database_path)
    assert gateway.started is False


@pytest.mark.asyncio
async def test_account_lock_serializes_entry_and_cross_portfolio_reductions(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor_a = PaperExecutor()
    executor_b = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor_a)
    harness.gateway.register_paper_executor("portfolio-b", executor_b)
    first_snapshot_entered = asyncio.Event()
    release_first_snapshot = asyncio.Event()
    snapshot_calls = 0

    async def get_snapshot(context, symbol, *, max_age_seconds=30.0):
        nonlocal snapshot_calls
        del context, symbol, max_age_seconds
        snapshot_calls += 1
        if snapshot_calls == 1:
            first_snapshot_entered.set()
            await release_first_snapshot.wait()
        return _broker_snapshot(harness.context)

    async def run_locked(context, symbol, callback, *, max_age_seconds=30.0):
        del context, symbol, max_age_seconds
        return await callback(_broker_snapshot(harness.context))

    monkeypatch.setattr(
        harness.gateway._client,
        "get_broker_contract_safety_snapshot",
        get_snapshot,
    )
    monkeypatch.setattr(
        harness.gateway._client,
        "run_with_locked_broker_contract_safety_snapshot",
        run_locked,
    )

    first = asyncio.create_task(
        harness.gateway.submit_reduction(
            order=_order(order_ref="reduce-a"),
            portfolio_id="portfolio-a",
        )
    )
    await first_snapshot_entered.wait()
    entry_entered = asyncio.Event()

    async def entry():
        async with harness.gateway.serialize_entry():
            entry_entered.set()

    entry_task = asyncio.create_task(entry())
    second = asyncio.create_task(
        harness.gateway.submit_reduction(
            order=_order(quantity=1, order_ref="reduce-b"),
            portfolio_id="portfolio-b",
        )
    )
    await asyncio.sleep(0)
    assert not entry_entered.is_set()
    assert not second.done()
    assert snapshot_calls == 1

    release_first_snapshot.set()
    assert (await first).ok is True
    await entry_task
    with pytest.raises(ReservationConflict, match=r"active account\+conId reservation"):
        await second
    assert entry_entered.is_set()
    assert len(executor_a.fills) == 1
    assert len(executor_b.fills) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("side", "portfolio_a_quantity"),
    [("SELL", 7), ("BUY_TO_COVER", -7)],
)
async def test_reduction_preserves_side_order_ref_and_decimal_limit_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    side: str,
    portfolio_a_quantity: int,
):
    value = await _build_harness(
        tmp_path,
        portfolio_a_quantity=portfolio_a_quantity,
        portfolio_b_quantity=0,
    )
    try:
        executor = PaperExecutor(slippage_bps=1.0)
        value.gateway.register_paper_executor("portfolio-a", executor)
        _install_broker_boundary(value, monkeypatch)
        order = _order(
            side=side,
            price=Decimal("123.4500"),
            order_ref=f"exact-{side.lower()}",
        )

        result = await value.gateway.submit_reduction(
            order=order,
            portfolio_id="portfolio-a",
        )

        assert result.ok is True
        assert len(executor.fills) == 1
        submitted = next(iter(executor.fills.values()))[1]
        assert submitted.side == side
        assert submitted.quantity == 2
        assert type(submitted.price) is Decimal
        assert submitted.price == Decimal("123.4500")
        assert any(
            f'"order_ref":"exact-{side.lower()}"' in event.payload_json
            for event in value.journal.replay().events
        )
    finally:
        await value.gateway.close()
        await value.database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "order",
    [
        _order(side="BUY"),
        _order(order_ref=" bad-ref"),
        Order(SYMBOL, Decimal("2"), "SELL", Decimal("123.45"), "decimal-quantity"),
        _order(price=Decimal("NaN")),
    ],
)
async def test_invalid_reduction_semantics_fail_before_broker_authority(
    harness: GatewayHarness,
    order: Order,
):
    harness.gateway.register_paper_executor("portfolio-a", PaperExecutor())
    get_snapshot = AsyncMock()
    harness.gateway._client.get_broker_contract_safety_snapshot = get_snapshot

    with pytest.raises(PaperReductionGatewayError):
        await harness.gateway.submit_reduction(
            order=order,
            portfolio_id="portfolio-a",
        )
    get_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_or_final_drift_invalidates_without_submission(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor)
    _install_broker_boundary(
        harness,
        monkeypatch,
        final_factory=lambda: _broker_snapshot(harness.context, con_id=CON_ID + 1),
    )

    with pytest.raises(RuntimeSafetyError, match="final contract does not match"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="contract-drift"),
            portfolio_id="portfolio-a",
        )

    assert executor.fills == {}
    state = harness.journal.replay()
    assert len(state.active_reservations) == 1
    assert len(state.quarantined_reservations) == 1


@pytest.mark.asyncio
async def test_final_allocation_drift_invalidates_and_blocks_restart(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor)

    async def reduce_allocation_below_request():
        await harness.database.update_position(
            SYMBOL,
            1,
            100.0,
            101.0,
            portfolio_id="portfolio-a",
        )

    _install_broker_boundary(
        harness,
        monkeypatch,
        before_final=reduce_allocation_below_request,
    )

    with pytest.raises(RuntimeSafetyError, match="no longer authorizes"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="allocation-drift"),
            portfolio_id="portfolio-a",
        )

    assert executor.fills == {}
    state = harness.journal.replay()
    assert len(state.active_reservations) == 1
    replacement = SafetyRuntimeCoordinator(
        harness.coordinator.paper_execution_identity,
        harness.journal,
    )
    with pytest.raises(RuntimeStartupBlocked, match="ACTIVE_RESERVATION_AT_STARTUP"):
        replacement.start()


@pytest.mark.asyncio
async def test_copied_broker_and_allocation_evidence_rejected_before_authority(
    harness: GatewayHarness,
):
    broker = _broker_snapshot(harness.context)
    allocation = await harness.database.get_safety_allocation_snapshot(
        SYMBOL,
        runtime_contract=harness.context.runtime_contract,
    )
    identity = harness.coordinator.paper_execution_identity
    contract = harness.context.runtime_contract

    with pytest.raises(ValidationError, match="not producer-owned"):
        assemble_local_paper_safety_evidence(
            identity,
            contract,
            replace(broker),
            allocation,
        )
    with pytest.raises(SafetyAllocationSnapshotError, match="not registered producer-owned"):
        assemble_local_paper_safety_evidence(
            identity,
            contract,
            broker,
            replace(allocation),
        )
    with pytest.raises(ValidationError, match="producer boundary"):
        replace(broker, _producer_marker=object())
    with pytest.raises(SafetyAllocationSnapshotError, match="trusted ledger producer"):
        replace(allocation, _producer_marker=object())
    assert harness.journal.replay().active_reservations == ()


@pytest.mark.asyncio
async def test_lifecycle_callback_remains_held_through_exactly_one_submission(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor)

    async def get_snapshot(context, symbol, *, max_age_seconds=30.0):
        del context, symbol, max_age_seconds
        return _broker_snapshot(harness.context)

    callback_active = False

    async def run_locked(context, symbol, callback, *, max_age_seconds=30.0):
        nonlocal callback_active
        del context, symbol, max_age_seconds
        assert not callback_active
        callback_active = True
        try:
            result = await callback(_broker_snapshot(harness.context))
            assert len(executor.fills) == 1
            return result
        finally:
            callback_active = False

    monkeypatch.setattr(
        harness.gateway._client,
        "get_broker_contract_safety_snapshot",
        get_snapshot,
    )
    monkeypatch.setattr(
        harness.gateway._client,
        "run_with_locked_broker_contract_safety_snapshot",
        run_locked,
    )

    result = await harness.gateway.submit_reduction(
        order=_order(order_ref="lifecycle-held"),
        portfolio_id="portfolio-a",
    )
    assert result.ok is True
    assert callback_active is False
    assert len(executor.fills) == 1

    with pytest.raises(IdempotencyConflict, match="different authorization evidence"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="lifecycle-held"),
            portfolio_id="portfolio-a",
        )
    assert len(executor.fills) == 1


@pytest.mark.asyncio
async def test_executor_preflight_failure_consumes_authority_without_retry(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor = PaperExecutor()
    executor.slippage_bps = float("nan")
    harness.gateway.register_paper_executor("portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)

    with pytest.raises(PaperReductionSubmissionError, match="slippage"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="executor-failure"),
            portfolio_id="portfolio-a",
        )
    assert executor.fills == {}
    state = harness.journal.replay()
    assert len(state.quarantined_reservations) == 1

    with pytest.raises(IdempotencyConflict, match="different authorization evidence"):
        await harness.gateway.submit_reduction(
            order=_order(order_ref="executor-failure"),
            portfolio_id="portfolio-a",
        )
    assert executor.fills == {}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known settlement gap: PaperReductionGateway does not apply the local fill to "
        "the allocation ledger or release the journal reservation before unlocking."
    ),
)
@pytest.mark.asyncio
async def test_successful_fill_is_settled_and_journal_released_before_gateway_unlock(
    harness: GatewayHarness,
    monkeypatch: pytest.MonkeyPatch,
):
    executor = PaperExecutor()
    harness.gateway.register_paper_executor("portfolio-a", executor)
    _install_broker_boundary(harness, monkeypatch)

    result = await harness.gateway.submit_reduction(
        order=_order(order_ref="settlement-gap"),
        portfolio_id="portfolio-a",
    )
    assert result.ok is True

    allocation = await harness.database.get_safety_allocation_snapshot(
        SYMBOL,
        runtime_contract=harness.context.runtime_contract,
    )
    portfolio_a = next(row for row in allocation.allocations if row.portfolio_id == "portfolio-a")
    assert portfolio_a.quantity == 5
    assert harness.journal.replay().active_reservations == ()

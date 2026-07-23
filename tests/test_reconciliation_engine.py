import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from robo_trader.reconciliation.engine import reconcile
from robo_trader.reconciliation.errors import BrokerEvidenceError
from robo_trader.reconciliation.models import (
    AggregatedLedgerPosition,
    BrokerExecution,
    BrokerExecutionScope,
    BrokerOpenOrder,
    BrokerPosition,
    BrokerSnapshot,
    ContractIdentity,
    LedgerPosition,
    LedgerSnapshot,
    LedgerTrade,
)

NOW = datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)


def _contract(symbol: str, con_id: int) -> ContractIdentity:
    return ContractIdentity(
        con_id=con_id,
        symbol=symbol,
        local_symbol=symbol,
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class=symbol,
    )


def _snapshot(*positions, orders=(), executions=(), age_seconds=0) -> BrokerSnapshot:
    at = NOW - timedelta(seconds=age_seconds)
    return BrokerSnapshot(
        schema_version=1,
        account_alias="***4567",
        broker_time_before=at - timedelta(seconds=1),
        broker_time_after=at,
        retrieved_at=at,
        execution_scope=BrokerExecutionScope(
            kind="bounded_execution_filter",
            start_at=at - timedelta(hours=24, seconds=1),
            end_at=at,
        ),
        positions=positions,
        open_orders=orders,
        recent_executions=executions,
        balances={
            "NetLiquidation:USD": Decimal("1000"),
            "TotalCashValue:USD": Decimal("500"),
        },
    )


def _ledger(*positions, blockers=(), trades=()) -> LedgerSnapshot:
    aggregated = []
    for position in positions:
        aggregated.append(
            AggregatedLedgerPosition(
                symbol=position.symbol,
                quantity=position.quantity,
                average_cost=position.average_cost,
                allocations=(position,),
            )
        )
    return LedgerSnapshot(
        selected_portfolio_ids=("default",),
        known_portfolio_ids=("default",),
        active_portfolio_ids=("default",) if positions else (),
        positions=positions,
        aggregated_positions=tuple(aggregated),
        recent_trades=trades,
        blockers=blockers,
        caveats=("LOCAL_LEDGER_HAS_NO_CONID_EXCHANGE_OR_CURRENCY",),
    )


def _local(symbol: str, quantity: str, cost: str) -> LedgerPosition:
    return LedgerPosition(
        portfolio_id="default",
        symbol=symbol,
        quantity=Decimal(quantity),
        average_cost=Decimal(cost),
        timestamp=NOW - timedelta(minutes=1),
    )


def test_position_diff_is_deterministic_for_match_broker_only_ledger_only_qty_and_cost():
    broker = _snapshot(
        BrokerPosition(_contract("AAPL", 1), Decimal("3"), Decimal("100.00")),
        BrokerPosition(_contract("MSFT", 2), Decimal("4"), Decimal("50.00")),
        BrokerPosition(_contract("TSLA", 3), Decimal("2"), Decimal("200.00")),
    )
    ledger = _ledger(
        _local("AAPL", "3", "100.009"),
        _local("GOOG", "1", "90"),
        _local("MSFT", "5", "50"),
        _local("TSLA", "2", "201"),
    )

    report = reconcile(
        broker,
        ledger,
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert [item.symbol for item in report.position_comparisons] == [
        "AAPL",
        "GOOG",
        "MSFT",
        "TSLA",
    ]
    assert [item.status for item in report.position_comparisons] == [
        "quantity_cost_match",
        "ledger_only",
        "quantity_cost_mismatch",
        "quantity_cost_mismatch",
    ]
    assert report.status == "MISMATCH"
    assert report.mutated_state is False
    assert report.authorizes_startup is False


def test_fractional_broker_quantity_is_a_schema_blocker():
    report = reconcile(
        _snapshot(BrokerPosition(_contract("AAPL", 1), Decimal("1.5"), Decimal("10"))),
        _ledger(_local("AAPL", "1", "10")),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )
    assert report.status == "BLOCKED"
    assert "BROKER_FRACTIONAL_QUANTITY_UNREPRESENTABLE:AAPL" in report.blockers


def test_stale_ledger_projection_cannot_report_comparable_only():
    stale = LedgerPosition(
        portfolio_id="default",
        symbol="AAPL",
        quantity=Decimal("3"),
        average_cost=Decimal("100"),
        timestamp=NOW - timedelta(minutes=6),
    )
    report = reconcile(
        _snapshot(BrokerPosition(_contract("AAPL", 1), Decimal("3"), Decimal("100"))),
        _ledger(stale),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "BLOCKED"
    assert "LEDGER_POSITION_PROJECTION_STALE:default:AAPL" in report.blockers


def test_future_ledger_position_timestamp_blocks_comparable_only_report():
    future = LedgerPosition(
        portfolio_id="default",
        symbol="AAPL",
        quantity=Decimal("3"),
        average_cost=Decimal("100"),
        timestamp=NOW + timedelta(seconds=6),
    )
    report = reconcile(
        _snapshot(BrokerPosition(_contract("AAPL", 1), Decimal("3"), Decimal("100"))),
        _ledger(future),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "BLOCKED"
    assert report.status != "QUANTITY_COST_COMPARABLE_ONLY"
    assert "LEDGER_POSITION_TIMESTAMP_FUTURE:default:AAPL" in report.blockers


def test_future_ledger_trade_timestamp_blocks_comparable_only_report():
    future_trade = LedgerTrade(
        local_trade_id=42,
        portfolio_id="default",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("3"),
        price=Decimal("100"),
        timestamp=NOW + timedelta(seconds=6),
    )
    report = reconcile(
        _snapshot(),
        _ledger(trades=(future_trade,)),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "BLOCKED"
    assert report.status != "QUANTITY_COST_COMPARABLE_ONLY"
    assert "LEDGER_TRADE_TIMESTAMP_FUTURE:42" in report.blockers


def test_open_orders_and_executions_are_deterministically_non_comparable():
    contract = _contract("AAPL", 1)
    order = BrokerOpenOrder(
        order_id="20",
        client_id=7,
        contract=contract,
        side="BUY",
        quantity=Decimal("3"),
        filled=Decimal("1"),
        remaining=Decimal("2"),
        order_type="LMT",
        status="Submitted",
        limit_price=Decimal("99"),
        permanent_id="501",
        unavailable={"stop_price": "not supplied for this order type"},
    )
    same_order_id_other_client = BrokerOpenOrder(
        order_id="20",
        client_id=8,
        contract=contract,
        side="SELL",
        quantity=Decimal("1"),
        filled=Decimal("0"),
        remaining=Decimal("1"),
        order_type="STP",
        status="Submitted",
        auxiliary_price=Decimal("90"),
        permanent_id="502",
        unavailable={"limit_price": "not supplied for this order type"},
    )
    execution = BrokerExecution(
        execution_id="exec-1",
        order_id="19",
        contract=contract,
        side="BOT",
        quantity=Decimal("2"),
        price=Decimal("98"),
        executed_at=NOW - timedelta(seconds=5),
    )
    report = reconcile(
        _snapshot(orders=(same_order_id_other_client, order), executions=(execution,)),
        _ledger(),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "BLOCKED"
    assert "UNMATCHED_ACTIVE_BROKER_OPEN_ORDERS" in report.blockers
    assert report.open_order_comparisons[0].status == "not_comparable"
    assert "NO_BROKER_ORDER_ID" in report.open_order_comparisons[0].reason
    assert report.execution_comparisons[0].status == "not_comparable"
    assert "NO_BROKER_EXECUTION" in report.execution_comparisons[0].reason
    public = report.public_dict()
    assert [item["broker_identifier"] for item in public["open_orders"]] == [
        "7:20",
        "8:20",
    ]
    first_order = public["open_orders"][0]
    assert first_order["details"]["client_id"] == 7
    assert first_order["details"]["broker_order_id"] == "20"
    assert first_order["details"]["permanent_id"] == "501"
    assert first_order["details"]["contract"] == {
        "con_id": 1,
        "currency": "USD",
        "exchange": "SMART",
        "local_symbol": "AAPL",
        "primary_exchange": "NASDAQ",
        "security_type": "STK",
        "symbol": "AAPL",
        "trading_class": "AAPL",
    }
    assert first_order["details"]["unavailable"] == {
        "stop_price": "not supplied for this order type"
    }
    with pytest.raises(TypeError):
        report.open_order_comparisons[0].details["contract"]["symbol"] = "MSFT"
    assert public["recent_executions"][0]["broker_identifier"] == "exec-1"
    assert public["broker_freshness"]["execution_scope"] == {
        "kind": "bounded_execution_filter",
        "start_at": (NOW - timedelta(hours=24, seconds=1)).isoformat(),
        "end_at": NOW.isoformat(),
    }


def test_public_report_serializes_all_evidence_decimals_without_exponents():
    small = Decimal("0.00000001")
    contract = _contract("AAPL", 1)
    order = BrokerOpenOrder(
        order_id="20",
        client_id=7,
        contract=contract,
        side="BUY",
        quantity=Decimal("0.00000002"),
        filled=small,
        remaining=small,
        order_type="LMT",
        status="Submitted",
        limit_price=small,
        auxiliary_price=small,
        average_fill_price=small,
    )
    execution = BrokerExecution(
        execution_id="exec-1",
        order_id="19",
        contract=contract,
        side="BOT",
        quantity=small,
        price=small,
        average_price=small,
        commission=small,
        commission_currency="USD",
        realized_pnl=small,
        executed_at=NOW - timedelta(seconds=5),
    )
    trade = LedgerTrade(
        local_trade_id=42,
        portfolio_id="default",
        symbol="AAPL",
        side="BUY",
        quantity=small,
        price=small,
        timestamp=NOW - timedelta(seconds=5),
    )

    report = reconcile(
        _snapshot(orders=(order,), executions=(execution,)),
        _ledger(trades=(trade,)),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )
    public_json = json.dumps(report.public_dict(), sort_keys=True)

    assert "1E-8" not in public_json
    assert "2E-8" not in public_json
    assert "0.00000001" in public_json
    assert "0.00000002" in public_json


def test_uncorrelated_broker_execution_is_incomplete_even_when_positions_match():
    execution = BrokerExecution(
        execution_id="exec-1",
        order_id="19",
        contract=_contract("AAPL", 1),
        side="BOT",
        quantity=Decimal("3"),
        price=Decimal("100"),
        executed_at=NOW - timedelta(seconds=5),
    )
    report = reconcile(
        _snapshot(
            BrokerPosition(_contract("AAPL", 1), Decimal("3"), Decimal("100")),
            executions=(execution,),
        ),
        _ledger(_local("AAPL", "3", "100")),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "INCOMPLETE"
    assert report.status != "QUANTITY_COST_COMPARABLE_ONLY"
    assert "BROKER_EXECUTIONS_CANNOT_BE_IDENTITY_MATCHED_TO_LOCAL_TRADES" in report.caveats


def test_uncorrelated_local_trade_is_incomplete_even_without_broker_execution():
    trade = LedgerTrade(
        local_trade_id=42,
        portfolio_id="default",
        symbol="AAPL",
        side="BUY",
        quantity=Decimal("3"),
        price=Decimal("100"),
        timestamp=NOW - timedelta(seconds=5),
    )
    report = reconcile(
        _snapshot(),
        _ledger(trades=(trade,)),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.status == "INCOMPLETE"
    assert report.status != "QUANTITY_COST_COMPARABLE_ONLY"
    assert report.execution_comparisons[0].evidence_type == "local_trade"


def test_public_report_rejects_account_fragments_in_metadata_and_execution_evidence():
    with pytest.raises(BrokerEvidenceError, match="sensitive identity"):
        reconcile(
            _snapshot(),
            _ledger(),
            runtime_fingerprint="runtime-DU1234567",
            database_identity="paper:db",
            expected_account_alias="***4567",
            now=NOW,
        )

    with pytest.raises(BrokerEvidenceError, match="sensitive identity"):
        BrokerExecution(
            execution_id="exec-DU1234567-fill",
            order_id="19",
            contract=_contract("AAPL", 1),
            side="BOT",
            quantity=Decimal("3"),
            price=Decimal("100"),
            executed_at=NOW - timedelta(seconds=5),
        )


def test_duplicate_order_identity_is_rejected_even_when_contracts_differ():
    first = BrokerOpenOrder(
        order_id="20",
        client_id=7,
        contract=_contract("AAPL", 1),
        side="BUY",
        quantity=Decimal("1"),
        filled=Decimal("0"),
        remaining=Decimal("1"),
        order_type="MKT",
        status="Submitted",
    )
    duplicate_identity = BrokerOpenOrder(
        order_id="20",
        client_id=7,
        contract=_contract("MSFT", 2),
        side="SELL",
        quantity=Decimal("1"),
        filled=Decimal("0"),
        remaining=Decimal("1"),
        order_type="MKT",
        status="Submitted",
    )

    with pytest.raises(BrokerEvidenceError, match="duplicate open-order identity"):
        _snapshot(orders=(first, duplicate_identity))


@pytest.mark.parametrize(
    "snapshot,expected",
    [
        (_snapshot(age_seconds=121), "stale"),
        (
            BrokerSnapshot(
                schema_version=1,
                account_alias="***4567",
                broker_time_before=NOW,
                broker_time_after=NOW - timedelta(seconds=1),
                retrieved_at=NOW,
                execution_scope=BrokerExecutionScope(
                    kind="bounded_execution_filter",
                    start_at=NOW - timedelta(hours=24),
                    end_at=NOW,
                ),
                balances={
                    "NetLiquidation:USD": Decimal("1000"),
                    "TotalCashValue:USD": Decimal("500"),
                },
            ),
            "reversed",
        ),
        (
            BrokerSnapshot(
                schema_version=1,
                account_alias="***4567",
                broker_time_before=NOW,
                broker_time_after=NOW,
                retrieved_at=NOW + timedelta(seconds=6),
                execution_scope=BrokerExecutionScope(
                    kind="bounded_execution_filter",
                    start_at=NOW - timedelta(hours=24),
                    end_at=NOW,
                ),
                balances={
                    "NetLiquidation:USD": Decimal("1000"),
                    "TotalCashValue:USD": Decimal("500"),
                },
            ),
            "future",
        ),
    ],
)
def test_stale_reversed_and_future_broker_evidence_fails_closed(snapshot, expected):
    with pytest.raises(BrokerEvidenceError, match=expected):
        reconcile(
            snapshot,
            _ledger(),
            runtime_fingerprint="runtime",
            database_identity="paper:db",
            expected_account_alias="***4567",
            now=NOW,
        )


def test_exact_execution_scope_can_end_before_snapshot_retrieval() -> None:
    broker_before = NOW - timedelta(seconds=10)
    snapshot = BrokerSnapshot(
        schema_version=1,
        account_alias="***4567",
        broker_time_before=broker_before,
        broker_time_after=NOW,
        retrieved_at=NOW,
        execution_scope=BrokerExecutionScope(
            kind="bounded_execution_filter",
            start_at=broker_before - timedelta(hours=24),
            end_at=NOW - timedelta(seconds=5),
        ),
        balances={
            "NetLiquidation:USD": Decimal("1000"),
            "TotalCashValue:USD": Decimal("500"),
        },
    )

    report = reconcile(
        snapshot,
        _ledger(),
        runtime_fingerprint="runtime",
        database_identity="paper:db",
        expected_account_alias="***4567",
        now=NOW,
    )

    assert report.broker_snapshot.execution_scope.end_at == NOW - timedelta(seconds=5)
    assert report.broker_snapshot.retrieved_at == NOW


def test_execution_scope_start_must_exactly_match_wire_filter() -> None:
    broker_before = NOW - timedelta(seconds=10)
    snapshot = BrokerSnapshot(
        schema_version=1,
        account_alias="***4567",
        broker_time_before=broker_before,
        broker_time_after=NOW,
        retrieved_at=NOW,
        execution_scope=BrokerExecutionScope(
            kind="bounded_execution_filter",
            start_at=broker_before - timedelta(hours=24) + timedelta(seconds=1),
            end_at=NOW - timedelta(seconds=5),
        ),
        balances={
            "NetLiquidation:USD": Decimal("1000"),
            "TotalCashValue:USD": Decimal("500"),
        },
    )

    with pytest.raises(BrokerEvidenceError, match="exact wire filter"):
        reconcile(
            snapshot,
            _ledger(),
            runtime_fingerprint="runtime",
            database_identity="paper:db",
            expected_account_alias="***4567",
            now=NOW,
        )


def test_account_mismatch_fails_without_exposing_expected_identity():
    with pytest.raises(BrokerEvidenceError) as exc_info:
        reconcile(
            _snapshot(),
            _ledger(),
            runtime_fingerprint="runtime",
            database_identity="paper:db",
            expected_account_alias="***9999",
            now=NOW,
        )
    assert "DU1234567" not in str(exc_info.value)

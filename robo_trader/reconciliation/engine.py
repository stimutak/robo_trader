"""Pure deterministic comparison of broker evidence and immutable ledger data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from .errors import BrokerEvidenceError
from .models import (
    BrokerSnapshot,
    LedgerSnapshot,
    NonComparableEvidence,
    PositionComparison,
    ReconciliationReport,
    canonical_decimal,
)

DEFAULT_COST_TOLERANCE = Decimal("0.01")
DEFAULT_MAX_SNAPSHOT_AGE = timedelta(minutes=2)
DEFAULT_MAX_COLLECTION_SPAN = timedelta(seconds=30)
DEFAULT_FUTURE_TOLERANCE = timedelta(seconds=5)
DEFAULT_MAX_LEDGER_POSITION_AGE = timedelta(minutes=5)


def validate_broker_snapshot(
    snapshot: BrokerSnapshot,
    *,
    expected_account_alias: str,
    now: datetime,
    max_age: timedelta = DEFAULT_MAX_SNAPSHOT_AGE,
) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise BrokerEvidenceError("reconciliation clock must be timezone-aware")
    if snapshot.account_alias != expected_account_alias:
        raise BrokerEvidenceError("broker snapshot account identity does not match runtime")
    if snapshot.execution_scope.kind != "bounded_execution_filter":
        raise BrokerEvidenceError("broker recent-execution scope is missing or ambiguous")
    if snapshot.broker_time_before > snapshot.broker_time_after:
        raise BrokerEvidenceError("broker snapshot timestamps are reversed")
    if snapshot.broker_time_after - snapshot.broker_time_before > DEFAULT_MAX_COLLECTION_SPAN:
        raise BrokerEvidenceError("broker snapshot collection window is too wide")
    expected_execution_start = snapshot.broker_time_before.replace(microsecond=0) - timedelta(
        hours=24
    )
    if snapshot.execution_scope.start_at != expected_execution_start:
        raise BrokerEvidenceError("broker execution scope does not match the exact wire filter")
    if not (
        snapshot.broker_time_before <= snapshot.execution_scope.end_at <= snapshot.broker_time_after
    ):
        raise BrokerEvidenceError("broker execution scope end is outside broker collection bounds")
    for timestamp in (
        snapshot.broker_time_before,
        snapshot.broker_time_after,
        snapshot.retrieved_at,
    ):
        if timestamp > now + DEFAULT_FUTURE_TOLERANCE:
            raise BrokerEvidenceError("broker snapshot timestamp is in the future")
        if now - timestamp > max_age:
            raise BrokerEvidenceError("broker snapshot is stale")
    if len({item.contract.con_id for item in snapshot.positions}) != len(snapshot.positions):
        raise BrokerEvidenceError("broker snapshot has duplicate contract identity")
    if len({item.contract.symbol for item in snapshot.positions}) != len(snapshot.positions):
        raise BrokerEvidenceError("broker snapshot has duplicate symbol identity")
    order_identities = {item.identity for item in snapshot.open_orders}
    if len(order_identities) != len(snapshot.open_orders):
        raise BrokerEvidenceError("broker snapshot has duplicate open-order identity")
    execution_ids = {item.execution_id for item in snapshot.recent_executions}
    if len(execution_ids) != len(snapshot.recent_executions):
        raise BrokerEvidenceError("broker snapshot has duplicate execution identity")
    for execution in snapshot.recent_executions:
        if (
            execution.executed_at < snapshot.execution_scope.start_at
            or execution.executed_at > snapshot.execution_scope.end_at
        ):
            raise BrokerEvidenceError("broker execution falls outside exact bounded request window")


def reconcile(
    broker: BrokerSnapshot,
    ledger: LedgerSnapshot,
    *,
    runtime_fingerprint: str,
    database_identity: str,
    expected_account_alias: str,
    now: datetime | None = None,
    cost_tolerance: Decimal = DEFAULT_COST_TOLERANCE,
) -> ReconciliationReport:
    generated_at = now or datetime.now(timezone.utc)
    validate_broker_snapshot(
        broker, expected_account_alias=expected_account_alias, now=generated_at
    )
    broker_by_symbol = {position.contract.symbol: position for position in broker.positions}
    ledger_by_symbol = {position.symbol: position for position in ledger.aggregated_positions}

    blockers = list(ledger.blockers)
    caveats = list(ledger.caveats)
    for position in ledger.positions:
        identity = f"{position.portfolio_id}:{position.symbol}"
        if position.timestamp > generated_at + DEFAULT_FUTURE_TOLERANCE:
            blockers.append(f"LEDGER_POSITION_TIMESTAMP_FUTURE:{identity}")
        elif generated_at - position.timestamp > DEFAULT_MAX_LEDGER_POSITION_AGE:
            blockers.append(f"LEDGER_POSITION_PROJECTION_STALE:{identity}")
    for trade in ledger.recent_trades:
        if trade.timestamp > generated_at + DEFAULT_FUTURE_TOLERANCE:
            blockers.append(f"LEDGER_TRADE_TIMESTAMP_FUTURE:{trade.local_trade_id}")
    comparisons = []
    for symbol in sorted(set(broker_by_symbol) | set(ledger_by_symbol)):
        broker_position = broker_by_symbol.get(symbol)
        ledger_position = ledger_by_symbol.get(symbol)
        reasons: list[str] = []
        if broker_position is None:
            status = "ledger_only"
            reasons.append("BROKER_POSITION_MISSING")
        elif ledger_position is None:
            status = "broker_only"
            reasons.append("LEDGER_POSITION_MISSING")
        else:
            if broker_position.quantity != broker_position.quantity.to_integral_value():
                blockers.append(f"BROKER_FRACTIONAL_QUANTITY_UNREPRESENTABLE:{symbol}")
                reasons.append("BROKER_QUANTITY_CANNOT_BE_REPRESENTED_BY_LEDGER_INTEGER_SCHEMA")
            if broker_position.quantity != ledger_position.quantity:
                reasons.append("QUANTITY_MISMATCH")
            if ledger_position.average_cost is None:
                reasons.append("LEDGER_AGGREGATE_COST_NOT_COMPARABLE")
            elif abs(broker_position.average_cost - ledger_position.average_cost) > cost_tolerance:
                reasons.append("AVERAGE_COST_MISMATCH")
            status = "quantity_cost_match" if not reasons else "quantity_cost_mismatch"

        comparisons.append(
            PositionComparison(
                symbol=symbol,
                status=status,
                reasons=tuple(reasons),
                broker_contract=broker_position.contract if broker_position else None,
                broker_quantity=broker_position.quantity if broker_position else None,
                ledger_quantity=ledger_position.quantity if ledger_position else None,
                broker_average_cost=broker_position.average_cost if broker_position else None,
                ledger_average_cost=ledger_position.average_cost if ledger_position else None,
                allocations=ledger_position.allocations if ledger_position else (),
            )
        )

    open_orders = tuple(
        NonComparableEvidence(
            evidence_type="open_order",
            broker_identifier=f"{order.client_id}:{order.order_id}",
            symbol=order.contract.symbol,
            status="not_comparable",
            reason="LOCAL_LEDGER_HAS_NO_BROKER_ORDER_ID_OR_DURABLE_OPEN_ORDER_TABLE",
            details={
                "client_id": order.client_id,
                "broker_order_id": order.order_id,
                "permanent_id": order.permanent_id,
                "contract": order.contract.public_dict(),
                "side": order.side,
                "quantity": canonical_decimal(order.quantity),
                "filled": canonical_decimal(order.filled),
                "remaining": canonical_decimal(order.remaining),
                "order_type": order.order_type,
                "status": order.status,
                "time_in_force": order.time_in_force,
                "limit_price": (
                    canonical_decimal(order.limit_price) if order.limit_price is not None else None
                ),
                "auxiliary_price": (
                    canonical_decimal(order.auxiliary_price)
                    if order.auxiliary_price is not None
                    else None
                ),
                "average_fill_price": (
                    canonical_decimal(order.average_fill_price)
                    if order.average_fill_price is not None
                    else None
                ),
                "last_status_at": (
                    order.last_status_at.isoformat() if order.last_status_at else None
                ),
                "unavailable": dict(order.unavailable),
            },
        )
        for order in sorted(
            broker.open_orders,
            key=lambda item: (
                item.client_id,
                int(item.order_id),
                item.contract.symbol,
            ),
        )
    )
    executions = tuple(
        NonComparableEvidence(
            evidence_type="recent_execution",
            broker_identifier=execution.execution_id,
            symbol=execution.contract.symbol,
            status="not_comparable",
            reason="LOCAL_LEDGER_HAS_NO_BROKER_EXECUTION_OR_ORDER_ID",
            details={
                "broker_order_id": execution.order_id,
                "permanent_id": execution.permanent_id,
                "client_id": execution.client_id,
                "contract": execution.contract.public_dict(),
                "side": execution.side,
                "quantity": canonical_decimal(execution.quantity),
                "price": canonical_decimal(execution.price),
                "average_price": (
                    canonical_decimal(execution.average_price)
                    if execution.average_price is not None
                    else None
                ),
                "executed_at": execution.executed_at.isoformat(),
                "execution_exchange": execution.execution_exchange,
                "commission": (
                    canonical_decimal(execution.commission)
                    if execution.commission is not None
                    else None
                ),
                "commission_currency": execution.commission_currency,
                "realized_pnl": (
                    canonical_decimal(execution.realized_pnl)
                    if execution.realized_pnl is not None
                    else None
                ),
                "unavailable": dict(execution.unavailable),
            },
        )
        for execution in sorted(
            broker.recent_executions,
            key=lambda item: (
                item.executed_at,
                item.contract.symbol,
                item.execution_id,
            ),
        )
    )
    execution_scope = broker.execution_scope
    scoped_local_trades = tuple(
        trade
        for trade in ledger.recent_trades
        if execution_scope.start_at <= trade.timestamp <= execution_scope.end_at
    )
    local_trade_evidence = tuple(
        NonComparableEvidence(
            evidence_type="local_trade",
            broker_identifier=f"local:{trade.local_trade_id}",
            symbol=trade.symbol,
            status="not_comparable",
            reason="LOCAL_LEDGER_HAS_NO_BROKER_EXECUTION_OR_ORDER_ID",
            details={
                "portfolio_id": trade.portfolio_id,
                "side": trade.side,
                "quantity": canonical_decimal(trade.quantity),
                "price": canonical_decimal(trade.price),
                "timestamp": trade.timestamp.isoformat(),
            },
        )
        for trade in sorted(
            scoped_local_trades,
            key=lambda item: (item.timestamp, item.portfolio_id, item.local_trade_id),
        )
    )
    executions = executions + local_trade_evidence
    if open_orders:
        blockers.append("UNMATCHED_ACTIVE_BROKER_OPEN_ORDERS")
        caveats.append("BROKER_OPEN_ORDERS_CANNOT_BE_MATCHED_TO_LOCAL_LEDGER")
    if executions:
        caveats.append("BROKER_EXECUTIONS_CANNOT_BE_IDENTITY_MATCHED_TO_LOCAL_TRADES")

    quantity_cost_difference = any(
        comparison.is_quantity_cost_difference for comparison in comparisons
    )
    if blockers:
        status = "BLOCKED"
    elif executions:
        status = "INCOMPLETE"
    elif quantity_cost_difference:
        status = "MISMATCH"
    else:
        status = "QUANTITY_COST_COMPARABLE_ONLY"

    return ReconciliationReport(
        generated_at=generated_at,
        runtime_fingerprint=runtime_fingerprint,
        database_identity=database_identity,
        account_alias=expected_account_alias,
        selected_portfolio_ids=ledger.selected_portfolio_ids,
        status=status,
        blockers=tuple(sorted(set(blockers))),
        caveats=tuple(sorted(set(caveats))),
        position_comparisons=tuple(comparisons),
        open_order_comparisons=open_orders,
        execution_comparisons=executions,
        broker_snapshot=broker,
        ledger_snapshot=ledger,
    )

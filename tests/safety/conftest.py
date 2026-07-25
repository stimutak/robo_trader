from datetime import datetime, timezone
from decimal import Decimal

import pytest

from robo_trader.safety import (
    EvidenceStatus,
    ExposureEvidence,
    GateContext,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioAllocationEvidence,
    ReconciliationStatus,
    SubmissionDescriptor,
    TimeInForce,
    TransportState,
    evaluate_reduce_only,
)

ACCOUNT_A = "acct_v1_" + ("a" * 64)
ACCOUNT_B = "acct_v1_" + ("b" * 64)


@pytest.fixture
def now():
    return datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)


def make_case(
    now,
    *,
    account_quantity=Decimal("10"),
    portfolio_quantity=Decimal("5"),
    order_quantity=Decimal("2"),
    side=OrderSide.SELL,
    account_target=None,
    portfolio_target=None,
    portfolio_id="portfolio-a",
    account_scope=ACCOUNT_A,
    execution_domain_scope="paper-domain",
    con_id=265598,
):
    delta = order_quantity if side in {OrderSide.BUY, OrderSide.BUY_TO_COVER} else -order_quantity
    intent = OrderIntent(
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        portfolio_id=portfolio_id,
        con_id=con_id,
        symbol="AAPL",
        side=side,
        quantity=order_quantity,
        account_current_quantity=account_quantity,
        target_quantity=account_quantity + delta if account_target is None else account_target,
        portfolio_target_quantity=(
            portfolio_quantity + delta if portfolio_target is None else portfolio_target
        ),
        portfolio_current_quantity=portfolio_quantity,
        created_at=now,
        reduce_only=False,
        reason="caller label is not authority",
        strategy="safety-looking-name",
    )
    exposure = ExposureEvidence(
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        con_id=con_id,
        symbol="AAPL",
        position_quantity=account_quantity,
        observed_at=now,
        status=EvidenceStatus.AUTHORITATIVE,
        source="independent-account-snapshot",
        snapshot_id="account-snapshot-1",
    )
    allocation = PortfolioAllocationEvidence(
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        portfolio_id=portfolio_id,
        con_id=con_id,
        symbol="AAPL",
        position_quantity=portfolio_quantity,
        aggregate_allocated_quantity=account_quantity,
        has_offsetting_allocations=False,
        observed_at=now,
        status=EvidenceStatus.AUTHORITATIVE,
        source="independent-allocation-ledger",
        snapshot_id="allocation-snapshot-1",
    )
    gates = GateContext(
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        con_id=con_id,
        evaluated_at=now,
        max_evidence_age_seconds=30,
        transport_state=TransportState.CONNECTED,
        reconciliation_status=ReconciliationStatus.PASSED,
        open_orders_complete=True,
        open_orders_all_clients=True,
        open_orders_snapshot_stable=True,
        open_orders_observed_at=now,
        open_orders_snapshot_id="open-orders-snapshot-1",
        active_order_count=0,
    )
    decision = evaluate_reduce_only(intent, exposure, allocation, gates)
    descriptor = SubmissionDescriptor(
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        con_id=con_id,
        side=side if side in {OrderSide.SELL, OrderSide.BUY_TO_COVER} else OrderSide.SELL,
        quantity=order_quantity,
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        outside_regular_hours=False,
        order_ref=f"close-{portfolio_id}-{con_id}",
    )
    return intent, exposure, allocation, gates, decision, descriptor

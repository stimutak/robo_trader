"""Pure, fail-closed reduce-only policy."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from typing import Optional, Tuple

from .models import (
    DecisionOutcome,
    EvidenceStatus,
    ExposureEvidence,
    GateContext,
    OrderIntent,
    OrderSide,
    PortfolioAllocationEvidence,
    ReconciliationStatus,
    RiskEffect,
    SafetyDecision,
    TransportState,
    ValidationError,
    _exact_decimal_add,
    sha256_text,
)


def _deny(
    fingerprint: str,
    reasons: Tuple[str, ...],
    *,
    risk_effect: RiskEffect = RiskEffect.UNKNOWN,
    current: Optional[Decimal] = None,
    target: Optional[Decimal] = None,
) -> SafetyDecision:
    return SafetyDecision(
        outcome=DecisionOutcome.DENY,
        risk_effect=risk_effect,
        reason_codes=reasons,
        current_quantity=current,
        computed_target_quantity=target,
        intent_fingerprint=fingerprint,
    )


def evaluate_reduce_only(
    intent: object,
    exposure: object,
    allocation: object,
    gates: object,
) -> SafetyDecision:
    """Evaluate an intent without trusting caller-provided safety labels.

    Invalid object types fail closed instead of raising. Valid model instances
    have already passed strict construction-time validation.
    """

    if type(intent) is not OrderIntent:
        return _deny(sha256_text("invalid-intent"), ("INVALID_INTENT",))
    if type(exposure) is not ExposureEvidence:
        return _deny(intent.fingerprint(), ("MISSING_EXPOSURE_EVIDENCE",))
    if type(allocation) is not PortfolioAllocationEvidence:
        return _deny(intent.fingerprint(), ("MISSING_PORTFOLIO_ALLOCATION_EVIDENCE",))
    if type(gates) is not GateContext:
        return _deny(intent.fingerprint(), ("MISSING_GATE_CONTEXT",))
    try:
        intent = replace(intent)
        exposure = replace(exposure)
        allocation = replace(allocation)
        gates = replace(gates)
    except (TypeError, ValueError):
        return _deny(sha256_text("invalid-boundary-model"), ("INVALID_BOUNDARY_MODEL",))
    fingerprint = intent.fingerprint()

    hard_reasons = []
    if (
        intent.execution_domain_scope != exposure.execution_domain_scope
        or intent.execution_domain_scope != allocation.execution_domain_scope
        or intent.execution_domain_scope != gates.execution_domain_scope
    ):
        hard_reasons.append("EXECUTION_DOMAIN_SCOPE_MISMATCH")
    if (
        intent.account_scope != exposure.account_scope
        or intent.account_scope != allocation.account_scope
        or intent.account_scope != gates.account_scope
    ):
        hard_reasons.append("ACCOUNT_SCOPE_MISMATCH")
    if gates.con_id != intent.con_id:
        hard_reasons.append("OPEN_ORDER_CON_ID_MISMATCH")
    if intent.portfolio_id != allocation.portfolio_id:
        hard_reasons.append("PORTFOLIO_SCOPE_MISMATCH")
    if intent.con_id != exposure.con_id or intent.con_id != allocation.con_id:
        hard_reasons.append("CON_ID_MISMATCH")
    if intent.symbol != exposure.symbol or intent.symbol != allocation.symbol:
        hard_reasons.append("SYMBOL_MISMATCH")
    if exposure.status is not EvidenceStatus.AUTHORITATIVE:
        hard_reasons.append("EXPOSURE_NOT_AUTHORITATIVE")
    if allocation.status is not EvidenceStatus.AUTHORITATIVE:
        hard_reasons.append("ALLOCATION_NOT_AUTHORITATIVE")
    if intent.created_at > gates.evaluated_at:
        hard_reasons.append("FUTURE_ORDER_INTENT")
    if exposure.observed_at > gates.evaluated_at:
        hard_reasons.append("FUTURE_EXPOSURE_EVIDENCE")
    elif gates.evaluated_at - exposure.observed_at > timedelta(
        seconds=gates.max_evidence_age_seconds
    ):
        hard_reasons.append("STALE_EXPOSURE_EVIDENCE")
    if allocation.observed_at > gates.evaluated_at:
        hard_reasons.append("FUTURE_ALLOCATION_EVIDENCE")
    elif gates.evaluated_at - allocation.observed_at > timedelta(
        seconds=gates.max_evidence_age_seconds
    ):
        hard_reasons.append("STALE_ALLOCATION_EVIDENCE")
    if allocation.aggregate_allocated_quantity != exposure.position_quantity:
        hard_reasons.append("AGGREGATE_ALLOCATION_MISMATCH")
    if allocation.has_offsetting_allocations:
        hard_reasons.append("OFFSETTING_ALLOCATIONS_EXIST")
    if (
        exposure.position_quantity != 0
        and allocation.position_quantity != 0
        and (exposure.position_quantity > 0) != (allocation.position_quantity > 0)
    ):
        hard_reasons.append("ACCOUNT_PORTFOLIO_DIRECTION_MISMATCH")
    if allocation.position_quantity.copy_abs() > exposure.position_quantity.copy_abs():
        hard_reasons.append("PORTFOLIO_ALLOCATION_EXCEEDS_ACCOUNT")
    if gates.transport_state is not TransportState.CONNECTED:
        hard_reasons.append("TRANSPORT_NOT_CERTAINLY_CONNECTED")
    if gates.reconciliation_status is not ReconciliationStatus.PASSED:
        hard_reasons.append("RECONCILIATION_NOT_PASSED")
    if not gates.open_orders_complete:
        hard_reasons.append("OPEN_ORDER_EVIDENCE_INCOMPLETE")
    if not gates.open_orders_all_clients:
        hard_reasons.append("OPEN_ORDER_EVIDENCE_NOT_ALL_CLIENTS")
    if not gates.open_orders_snapshot_stable:
        hard_reasons.append("OPEN_ORDER_SNAPSHOT_UNSTABLE")
    if gates.open_orders_observed_at > gates.evaluated_at:
        hard_reasons.append("FUTURE_OPEN_ORDER_EVIDENCE")
    elif gates.evaluated_at - gates.open_orders_observed_at > timedelta(
        seconds=gates.max_evidence_age_seconds
    ):
        hard_reasons.append("STALE_OPEN_ORDER_EVIDENCE")
    if gates.active_order_count:
        hard_reasons.append("ACTIVE_BROKER_ORDER_EXISTS")
    hard_reasons.extend(f"HARD_GATE:{reason}" for reason in gates.hard_block_reasons)
    if hard_reasons:
        return _deny(
            fingerprint,
            tuple(hard_reasons),
            current=exposure.position_quantity,
        )

    current = exposure.position_quantity
    portfolio_current = allocation.position_quantity
    assertion_reasons = []
    if intent.account_current_quantity != current:
        assertion_reasons.append("ACCOUNT_CURRENT_ASSERTION_MISMATCH")
    if intent.portfolio_current_quantity != portfolio_current:
        assertion_reasons.append("PORTFOLIO_CURRENT_ASSERTION_MISMATCH")
    if assertion_reasons:
        return _deny(
            fingerprint,
            tuple(assertion_reasons),
            current=current,
        )
    signed_delta = (
        intent.quantity
        if intent.side in {OrderSide.BUY, OrderSide.BUY_TO_COVER}
        else intent.quantity.copy_negate()
    )
    try:
        target = _exact_decimal_add(current, signed_delta, "computed account target")
        portfolio_target = _exact_decimal_add(
            portfolio_current,
            signed_delta,
            "computed portfolio target",
        )
    except ValidationError:
        return _deny(
            fingerprint,
            ("ARITHMETIC_OUT_OF_CONTRACT",),
            current=current,
        )

    arithmetic_reasons = []
    if intent.target_quantity != target:
        arithmetic_reasons.append("INCONSISTENT_ACCOUNT_TARGET_ARITHMETIC")
    if intent.portfolio_target_quantity != portfolio_target:
        arithmetic_reasons.append("INCONSISTENT_PORTFOLIO_TARGET_ARITHMETIC")
    if arithmetic_reasons:
        return _deny(
            fingerprint,
            tuple(arithmetic_reasons),
            current=current,
            target=target,
        )
    if current == 0 or portfolio_current == 0:
        return _deny(
            fingerprint,
            ("ZERO_OR_UNKNOWN_POSITION",),
            current=current,
            target=target,
        )

    reducing = False
    reasons = []
    # Both account truth and portfolio allocation must independently reduce
    # without an over-close. Caller target fields are assertions only.
    if current > 0 and portfolio_current > 0:
        if intent.side is not OrderSide.SELL:
            reasons.append("LONG_REDUCTION_REQUIRES_SELL")
        elif target < 0 or portfolio_target < 0:
            reasons.append("OVER_CLOSE_OR_REVERSAL")
        elif target >= current or portfolio_target >= portfolio_current:
            reasons.append("NOT_REDUCING")
        else:
            reducing = True
    elif current < 0 and portfolio_current < 0:
        if intent.side is not OrderSide.BUY_TO_COVER:
            reasons.append("SHORT_REDUCTION_REQUIRES_BUY_TO_COVER")
        elif target > 0 or portfolio_target > 0:
            reasons.append("OVER_CLOSE_OR_REVERSAL")
        elif target <= current or portfolio_target <= portfolio_current:
            reasons.append("NOT_REDUCING")
        else:
            reducing = True
    else:
        reasons.append("ACCOUNT_PORTFOLIO_DIRECTION_MISMATCH")

    if not reducing:
        if not gates.soft_entry_allowed:
            reasons.append("SOFT_ENTRY_GATE_BLOCKED")
        return _deny(
            fingerprint,
            tuple(reasons) or ("NOT_REDUCING",),
            risk_effect=RiskEffect.INCREASING,
            current=current,
            target=target,
        )

    return SafetyDecision(
        outcome=DecisionOutcome.ALLOW,
        risk_effect=RiskEffect.REDUCING,
        reason_codes=("AUTHORITATIVE_REDUCTION",),
        current_quantity=current,
        computed_target_quantity=target,
        intent_fingerprint=fingerprint,
    )


class ReduceOnlyPolicy:
    """Stateless callable wrapper for dependency-injection boundaries."""

    def evaluate(
        self,
        intent: object,
        exposure: object,
        allocation: object,
        gates: object,
    ) -> SafetyDecision:
        return evaluate_reduce_only(intent, exposure, allocation, gates)

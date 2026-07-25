from dataclasses import FrozenInstanceError, replace
from datetime import timedelta, timezone
from decimal import Decimal, localcontext

import pytest

from robo_trader.safety import (
    DecisionOutcome,
    EvidenceStatus,
    OrderIntent,
    OrderSide,
    RiskEffect,
    SafetyDecision,
    TransportState,
    ValidationError,
    decimal_to_fixed,
    evaluate_reduce_only,
)

from .conftest import ACCOUNT_B, make_case


@pytest.mark.parametrize(
    ("account", "portfolio", "quantity", "side", "allowed"),
    [
        ("10", "5", "2", OrderSide.SELL, True),
        ("10", "5", "5", OrderSide.SELL, True),
        ("10", "5", "5.01", OrderSide.SELL, False),
        ("10", "5", "11", OrderSide.SELL, False),
        ("10", "5", "2", OrderSide.BUY, False),
        ("10", "5", "2", OrderSide.BUY_TO_COVER, False),
        ("-10", "-5", "2", OrderSide.BUY_TO_COVER, True),
        ("-10", "-5", "5", OrderSide.BUY_TO_COVER, True),
        ("-10", "-5", "5.01", OrderSide.BUY_TO_COVER, False),
        ("-10", "-5", "11", OrderSide.BUY_TO_COVER, False),
        ("-10", "-5", "2", OrderSide.BUY, False),
        ("-10", "-5", "2", OrderSide.SELL, False),
        ("0", "0", "1", OrderSide.SELL, False),
    ],
)
def test_exact_long_short_close_matrix(now, account, portfolio, quantity, side, allowed):
    case = make_case(
        now,
        account_quantity=Decimal(account),
        portfolio_quantity=Decimal(portfolio),
        order_quantity=Decimal(quantity),
        side=side,
    )
    decision = case[4]
    assert decision.allowed is allowed
    if allowed:
        assert decision.risk_effect is RiskEffect.REDUCING
        assert abs(decision.computed_target_quantity) < abs(Decimal(account))


def test_property_allowed_implies_both_authoritative_exposures_strictly_reduce(now):
    for sign, side in ((Decimal("1"), OrderSide.SELL), (Decimal("-1"), OrderSide.BUY_TO_COVER)):
        for account_abs in range(1, 15):
            for portfolio_abs in range(1, account_abs + 1):
                for quantity in range(1, account_abs + 2):
                    decision = make_case(
                        now,
                        account_quantity=sign * account_abs,
                        portfolio_quantity=sign * portfolio_abs,
                        order_quantity=Decimal(quantity),
                        side=side,
                    )[4]
                    if decision.allowed:
                        assert quantity <= portfolio_abs
                        assert abs(decision.computed_target_quantity) < account_abs


@pytest.mark.parametrize(
    ("current_quantity", "target_quantity"),
    [
        (Decimal("10"), Decimal("-1")),
        (Decimal("-10"), Decimal("1")),
    ],
)
def test_direct_allow_decision_rejects_sign_crossing_reversal(
    current_quantity,
    target_quantity,
):
    with pytest.raises(ValidationError, match="must not cross through zero"):
        SafetyDecision(
            outcome=DecisionOutcome.ALLOW,
            risk_effect=RiskEffect.REDUCING,
            reason_codes=("REDUCE_ONLY_ALLOWED",),
            current_quantity=current_quantity,
            computed_target_quantity=target_quantity,
            intent_fingerprint="a" * 64,
        )


@pytest.mark.parametrize("current_quantity", [Decimal("10"), Decimal("-10")])
def test_direct_allow_decision_accepts_exact_close_to_zero(current_quantity):
    decision = SafetyDecision(
        outcome=DecisionOutcome.ALLOW,
        risk_effect=RiskEffect.REDUCING,
        reason_codes=("REDUCE_ONLY_ALLOWED",),
        current_quantity=current_quantity,
        computed_target_quantity=Decimal("0"),
        intent_fingerprint="a" * 64,
    )
    assert decision.allowed


@pytest.mark.parametrize(
    "bad",
    [
        1,
        1.0,
        True,
        "1",
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("1E+2"),
        Decimal("-0"),
        Decimal("1.0000000000001"),
        Decimal("12345678901234567890123456789"),
    ],
)
def test_quantity_rejects_every_non_exact_or_exponent_value(now, bad):
    intent = make_case(now)[0]
    with pytest.raises(ValidationError):
        replace(intent, quantity=bad)


@pytest.mark.parametrize(
    ("account", "portfolio", "quantity", "side", "expected_target"),
    [
        (
            "1.000000000001",
            "0.500000000001",
            "0.000000000001",
            OrderSide.SELL,
            "1",
        ),
        (
            "-1.000000000001",
            "-0.500000000001",
            "0.000000000001",
            OrderSide.BUY_TO_COVER,
            "-1",
        ),
        (
            "0.000000000001",
            "0.000000000001",
            "0.000000000001",
            OrderSide.SELL,
            "0",
        ),
        (
            "-0.000000000001",
            "-0.000000000001",
            "0.000000000001",
            OrderSide.BUY_TO_COVER,
            "0",
        ),
    ],
)
def test_max_scale_partial_and_full_closes_are_exact(
    now,
    account,
    portfolio,
    quantity,
    side,
    expected_target,
):
    decision = make_case(
        now,
        account_quantity=Decimal(account),
        portfolio_quantity=Decimal(portfolio),
        order_quantity=Decimal(quantity),
        side=side,
    )[4]
    assert decision.allowed
    assert decimal_to_fixed(decision.computed_target_quantity) == expected_target


def test_policy_arithmetic_ignores_low_ambient_decimal_precision(now):
    intent, exposure, allocation, gates, _, _ = make_case(
        now,
        account_quantity=Decimal("1.123456789012"),
        portfolio_quantity=Decimal("1.123456789012"),
        order_quantity=Decimal("0.123456789012"),
    )
    with localcontext() as context:
        context.prec = 3
        decision = evaluate_reduce_only(intent, exposure, allocation, gates)
    assert decision.allowed
    assert decision.computed_target_quantity == Decimal("1.000000000000")


def test_mixed_maximum_magnitude_and_scale_fails_closed_without_rounding(now):
    current = Decimal("9999999999999999999999999999")
    quantity = Decimal("0.000000000001")
    decision = make_case(
        now,
        account_quantity=current,
        portfolio_quantity=current,
        order_quantity=quantity,
        account_target=current,
        portfolio_target=current,
    )[4]
    assert not decision.allowed
    assert decision.reason_codes == ("ARITHMETIC_OUT_OF_CONTRACT",)


def test_negative_scale_zero_is_valid_and_serializes_without_exponent(now):
    intent = replace(make_case(now)[0], target_quantity=Decimal("0E-7"))
    assert intent.target_quantity == 0
    assert decimal_to_fixed(intent.target_quantity) == "0"


def test_models_are_frozen_and_raw_accounts_are_rejected(now):
    intent = make_case(now)[0]
    with pytest.raises(FrozenInstanceError):
        intent.symbol = "MSFT"
    with pytest.raises(ValidationError):
        replace(intent, account_scope="DU1234567")
    with pytest.raises(ValidationError):
        replace(intent, account_scope="masked-last4-1234")
    with pytest.raises(ValidationError):
        replace(intent, account_scope="generic-scope")
    with pytest.raises(ValidationError):
        replace(intent, reason="copied from DU1234567")
    with pytest.raises(ValidationError):
        replace(intent, reason="prefixxDU1234567xsuffix")
    with pytest.raises(ValidationError):
        replace(intent, strategy="embeddedDU7654321inside")
    with pytest.raises(ValidationError):
        replace(intent, reason="prefixxU1234567xsuffix")
    with pytest.raises(ValidationError):
        replace(intent, strategy="embeddedF7654321inside")


def test_fingerprint_covers_every_authorization_field(now):
    intent = make_case(now)[0]
    mutations = (
        {"execution_domain_scope": "paper-domain-2"},
        {"account_scope": ACCOUNT_B},
        {"portfolio_id": "portfolio-b"},
        {"con_id": intent.con_id + 1},
        {"symbol": "MSFT"},
        {"side": OrderSide.BUY},
        {"quantity": Decimal("1")},
        {"account_current_quantity": Decimal("9")},
        {"target_quantity": Decimal("9")},
        {"portfolio_current_quantity": Decimal("4")},
        {"portfolio_target_quantity": Decimal("4")},
        {"created_at": now + timedelta(microseconds=1)},
        {"reduce_only": True},
        {"reason": "other"},
        {"strategy": "other"},
    )
    assert all(
        replace(intent, **change).fingerprint() != intent.fingerprint() for change in mutations
    )


def test_submission_descriptor_fingerprint_covers_every_broker_affecting_term(now):
    descriptor = make_case(now)[5]
    from robo_trader.safety import OrderType, TimeInForce

    mutations = (
        {"execution_domain_scope": "other-domain"},
        {"account_scope": ACCOUNT_B},
        {"con_id": descriptor.con_id + 1},
        {"side": OrderSide.BUY_TO_COVER},
        {"quantity": Decimal("1")},
        {
            "order_type": OrderType.LIMIT,
            "limit_price": Decimal("100"),
        },
        {"time_in_force": TimeInForce.GTC},
        {"outside_regular_hours": True},
        {"order_ref": "other-order-ref"},
    )
    for change in mutations:
        assert replace(descriptor, **change).fingerprint() != descriptor.fingerprint()


def test_hard_truth_gates_block_reductions_and_soft_entry_gate_does_not(now):
    intent, exposure, allocation, gates, _, _ = make_case(now)
    soft = evaluate_reduce_only(
        intent, exposure, allocation, replace(gates, soft_entry_allowed=False)
    )
    assert soft.allowed
    hard_cases = (
        replace(gates, transport_state=TransportState.AMBIGUOUS),
        replace(gates, open_orders_complete=False),
        replace(gates, open_orders_all_clients=False),
        replace(gates, open_orders_snapshot_stable=False),
        replace(gates, active_order_count=1),
    )
    for hard in hard_cases:
        assert not evaluate_reduce_only(intent, exposure, allocation, hard).allowed


def test_missing_stale_future_failed_and_inconsistent_evidence_fail_closed(now):
    intent, exposure, allocation, gates, _, _ = make_case(now)
    cases = (
        (None, allocation, gates),
        (replace(exposure, status=EvidenceStatus.FAILED), allocation, gates),
        (replace(exposure, observed_at=now - timedelta(seconds=31)), allocation, gates),
        (replace(exposure, observed_at=now + timedelta(microseconds=1)), allocation, gates),
        (exposure, replace(allocation, aggregate_allocated_quantity=Decimal("9")), gates),
        (exposure, replace(allocation, has_offsetting_allocations=True), gates),
    )
    for account_evidence, portfolio_evidence, context in cases:
        assert not evaluate_reduce_only(
            intent, account_evidence, portfolio_evidence, context
        ).allowed


def test_target_assertions_are_recomputed_and_dishonest_labels_grant_nothing(now):
    intent, exposure, allocation, gates, _, _ = make_case(now)
    dishonest = replace(
        intent,
        side=OrderSide.BUY,
        target_quantity=Decimal("12"),
        portfolio_target_quantity=Decimal("7"),
        reduce_only=True,
        reason="stop",
        strategy="safety",
    )
    decision = evaluate_reduce_only(dishonest, exposure, allocation, gates)
    assert decision.outcome is DecisionOutcome.DENY
    inconsistent = replace(intent, target_quantity=Decimal("7.999"))
    assert not evaluate_reduce_only(inconsistent, exposure, allocation, gates).allowed
    wrong_current = replace(intent, account_current_quantity=Decimal("9"))
    assert not evaluate_reduce_only(wrong_current, exposure, allocation, gates).allowed


def test_local_timestamp_is_rejected(now):
    intent = make_case(now)[0]
    local = now.astimezone(timezone(timedelta(hours=-4)))
    with pytest.raises(ValidationError):
        replace(intent, created_at=local)


def test_caller_cannot_expand_safety_owned_evidence_freshness_window(now):
    gates = make_case(now)[3]
    with pytest.raises(ValidationError, match="safety-owned"):
        replace(gates, max_evidence_age_seconds=86400)

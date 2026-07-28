"""Adversarial tests for the dormant PR-7 Gate-A entry-risk contract."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from robo_trader.risk.entry_contract import (
    ENTRY_RISK_CONTRACT_VERSION,
    GATE_A_MAX_POSITION_FRACTION,
    CorrelationEvidence,
    EntryFeatureFlags,
    EntryIntent,
    EntryRiskContractError,
    EntryRiskEvidence,
    EntryRiskLimits,
    EntrySide,
    EntrySignal,
    LimitingCapacity,
    LiquidityEvidence,
    RefreshedQuoteEvidence,
    RiskReason,
    SignalSource,
    build_entry_intent,
    evaluate_entry_intent,
)

NOW = datetime(2026, 7, 28, 15, 0, tzinfo=timezone.utc)


def _signal(**changes: object) -> EntrySignal:
    values: dict[str, object] = {
        "signal_id": "sig-001",
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "side": EntrySide.BUY,
        "source": SignalSource.BASE_STRATEGY,
        "confidence_fraction": Decimal("0.81"),
        "requested_position_fraction": Decimal("0.10"),
        "source_data_version": "bars-20260728T145900Z",
        "observed_at": NOW - timedelta(seconds=10),
        "expires_at": NOW + timedelta(minutes=2),
    }
    values.update(changes)
    return EntrySignal(**values)  # type: ignore[arg-type]


def _intent(**signal_changes: object) -> EntryIntent:
    return build_entry_intent(
        _signal(**signal_changes),
        intent_id="intent-001",
        quote_refresh_request_id="refresh-001",
        created_at=NOW - timedelta(seconds=5),
    )


def _limits(**changes: object) -> EntryRiskLimits:
    values: dict[str, object] = {
        "max_position_fraction": GATE_A_MAX_POSITION_FRACTION,
        "max_sector_fraction": Decimal("0.25"),
        "max_portfolio_gross_fraction": Decimal("0.75"),
        "max_absolute_correlation": Decimal("0.80"),
        "minimum_average_daily_dollar_volume_usd": Decimal("1000000"),
        "max_order_fraction_of_daily_dollar_volume": Decimal("0.01"),
        "max_daily_notional_usd": Decimal("50000"),
        "max_quote_age": timedelta(seconds=10),
        "max_account_evidence_age": timedelta(seconds=30),
    }
    values.update(changes)
    return EntryRiskLimits(**values)  # type: ignore[arg-type]


def _flags(**changes: object) -> EntryFeatureFlags:
    values: dict[str, object] = {
        "risk_contract_enabled": True,
        "refreshed_quote_revalidation_enabled": True,
        "base_strategy_entries_enabled": True,
        "short_entries_enabled": False,
        "ai_discovery_entries_enabled": False,
        "pairs_entries_enabled": False,
        "smart_execution_entries_enabled": False,
    }
    values.update(changes)
    return EntryFeatureFlags(**values)  # type: ignore[arg-type]


def _quote(**changes: object) -> RefreshedQuoteEvidence:
    values: dict[str, object] = {
        "quote_id": "quote-001",
        "refresh_request_id": "refresh-001",
        "symbol": "AAPL",
        "price_usd": Decimal("333"),
        "observed_at": NOW - timedelta(seconds=1),
        "transport_generation": "ib-session-42",
        "source": "live-broker",
    }
    values.update(changes)
    return RefreshedQuoteEvidence(**values)  # type: ignore[arg-type]


def _evidence(**changes: object) -> EntryRiskEvidence:
    values: dict[str, object] = {
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "observed_at": NOW - timedelta(seconds=1),
        "quote": _quote(),
        "sector": "Technology",
        "correlation": CorrelationEvidence(
            complete=True,
            existing_position_count=0,
            max_absolute_correlation=Decimal("0"),
            observed_at=NOW - timedelta(seconds=1),
        ),
        "liquidity": LiquidityEvidence(
            complete=True,
            average_daily_dollar_volume_usd=Decimal("10000000"),
            observed_at=NOW - timedelta(seconds=1),
        ),
        "portfolio_equity_usd": Decimal("100000"),
        "cash_available_usd": Decimal("50000"),
        "buying_power_usd": Decimal("100000"),
        "current_symbol_gross_notional_usd": Decimal("0"),
        "current_sector_gross_notional_usd": Decimal("0"),
        "portfolio_gross_notional_usd": Decimal("0"),
        "daily_executed_notional_usd": Decimal("0"),
    }
    values.update(changes)
    return EntryRiskEvidence(**values)  # type: ignore[arg-type]


def _evaluate(
    *,
    intent: EntryIntent | None = None,
    evidence: EntryRiskEvidence | None = None,
    limits: EntryRiskLimits | None = None,
    flags: EntryFeatureFlags | None = None,
    evaluated_at: datetime = NOW,
):
    return evaluate_entry_intent(
        intent or _intent(),
        evidence or _evidence(),
        limits or _limits(),
        flags or _flags(),
        evaluated_at=evaluated_at,
    )


def test_signal_to_intent_to_decision_preserves_versioned_lineage() -> None:
    signal = _signal()
    intent = build_entry_intent(
        signal,
        intent_id="intent-001",
        quote_refresh_request_id="refresh-001",
        created_at=NOW - timedelta(seconds=5),
    )
    decision = _evaluate(intent=intent)

    assert intent.schema_version == ENTRY_RISK_CONTRACT_VERSION
    assert intent.signal_id == signal.signal_id
    assert intent.portfolio_id == signal.portfolio_id
    assert intent.symbol == signal.symbol
    assert intent.side is signal.side
    assert intent.source is signal.source
    assert intent.confidence_fraction == signal.confidence_fraction
    assert intent.source_data_version == signal.source_data_version
    assert decision.signal_id == signal.signal_id
    assert decision.portfolio_id == signal.portfolio_id
    assert decision.symbol == signal.symbol
    assert decision.side is signal.side
    assert decision.authorizes_order_submission is False
    assert decision.runtime_integration_ready is False
    with pytest.raises(FrozenInstanceError):
        intent.symbol = "MSFT"  # type: ignore[misc]


def test_configured_two_percent_cap_floors_quantity_and_never_rounds_up() -> None:
    decision = _evaluate()

    assert decision.risk_approved is True
    assert decision.approved_quantity == 6
    assert decision.approved_notional_usd == Decimal("1998")
    assert decision.approved_notional_usd <= Decimal("2000")
    assert decision.limiting_capacity is LimitingCapacity.SYMBOL

    boundary = _evaluate(evidence=_evidence(quote=_quote(price_usd=Decimal("666.67"))))
    assert boundary.approved_quantity == 2
    assert boundary.approved_notional_usd == Decimal("1333.34")
    assert boundary.approved_notional_usd <= Decimal("2000")


def test_gate_a_rejects_a_configured_position_cap_above_two_percent() -> None:
    with pytest.raises(EntryRiskContractError, match="cannot exceed 2%"):
        _limits(max_position_fraction=Decimal("0.0200001"))


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("sector", RiskReason.MISSING_SECTOR),
        ("correlation", RiskReason.MISSING_CORRELATION),
        ("liquidity", RiskReason.MISSING_LIQUIDITY),
        ("cash_available_usd", RiskReason.MISSING_CASH),
        ("buying_power_usd", RiskReason.MISSING_BUYING_POWER),
        ("daily_executed_notional_usd", RiskReason.MISSING_DAILY_NOTIONAL),
        ("portfolio_equity_usd", RiskReason.MISSING_PORTFOLIO_EQUITY),
        (
            "current_symbol_gross_notional_usd",
            RiskReason.MISSING_SYMBOL_EXPOSURE,
        ),
        (
            "current_sector_gross_notional_usd",
            RiskReason.MISSING_SECTOR_EXPOSURE,
        ),
        ("portfolio_gross_notional_usd", RiskReason.MISSING_PORTFOLIO_EXPOSURE),
        ("quote", RiskReason.MISSING_QUOTE),
    ],
)
def test_missing_authoritative_evidence_fails_closed(
    field_name: str,
    reason: RiskReason,
) -> None:
    decision = _evaluate(evidence=_evidence(**{field_name: None}))

    assert decision.risk_approved is False
    assert reason in decision.reasons
    assert decision.approved_quantity == 0
    assert decision.approved_notional_usd == 0


@pytest.mark.parametrize("field_name", ["portfolio_id", "symbol"])
def test_missing_evidence_scope_fails_closed(field_name: str) -> None:
    decision = _evaluate(evidence=_evidence(**{field_name: None}))

    assert decision.reasons == (RiskReason.MISSING_EVIDENCE_SCOPE,)


@pytest.mark.parametrize(
    "changes",
    [
        {"portfolio_id": "portfolio-beta"},
        {"symbol": "MSFT"},
    ],
)
def test_wrong_portfolio_or_symbol_evidence_cannot_cross_scope(changes: dict[str, str]) -> None:
    decision = _evaluate(evidence=_evidence(**changes))

    assert decision.reasons == (RiskReason.EVIDENCE_SCOPE_MISMATCH,)


@pytest.mark.parametrize(
    "quote",
    [
        _quote(refresh_request_id="refresh-other"),
        _quote(symbol="MSFT"),
        _quote(observed_at=NOW - timedelta(seconds=6)),
        _quote(observed_at=NOW + timedelta(microseconds=1)),
    ],
)
def test_quote_must_be_fresh_and_bound_to_the_intent(quote: RefreshedQuoteEvidence) -> None:
    decision = _evaluate(evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert decision.reasons == (RiskReason.QUOTE_NOT_REFRESHED,)


@pytest.mark.parametrize("kind", ["correlation", "liquidity"])
def test_incomplete_or_stale_market_evidence_fails_closed(kind: str) -> None:
    evidence = _evidence()
    item = getattr(evidence, kind)
    assert item is not None
    incomplete = replace(item, complete=False)
    stale = replace(item, observed_at=NOW - timedelta(seconds=31))
    expected = (
        RiskReason.MISSING_CORRELATION if kind == "correlation" else RiskReason.MISSING_LIQUIDITY
    )

    incomplete_decision = _evaluate(evidence=replace(evidence, **{kind: incomplete}))
    stale_decision = _evaluate(evidence=replace(evidence, **{kind: stale}))

    assert incomplete_decision.reasons == (expected,)
    assert stale_decision.reasons == (expected,)


@pytest.mark.parametrize(
    ("evidence_changes", "expected_capacity", "expected_quantity"),
    [
        (
            {"current_symbol_gross_notional_usd": Decimal("1900")},
            LimitingCapacity.SYMBOL,
            1,
        ),
        (
            {"current_sector_gross_notional_usd": Decimal("24500")},
            LimitingCapacity.SECTOR,
            8,
        ),
        (
            {"portfolio_gross_notional_usd": Decimal("74900")},
            LimitingCapacity.PORTFOLIO,
            1,
        ),
        ({"cash_available_usd": Decimal("500")}, LimitingCapacity.CASH, 8),
        ({"buying_power_usd": Decimal("400")}, LimitingCapacity.BUYING_POWER, 6),
        (
            {"daily_executed_notional_usd": Decimal("49700")},
            LimitingCapacity.DAILY_NOTIONAL,
            5,
        ),
    ],
)
def test_each_exposure_unit_constrains_the_same_usd_order_notional(
    evidence_changes: dict[str, Decimal],
    expected_capacity: LimitingCapacity,
    expected_quantity: int,
) -> None:
    price = Decimal("60")
    evidence = _evidence(quote=_quote(price_usd=price), **evidence_changes)
    decision = _evaluate(evidence=evidence)

    assert decision.limiting_capacity is expected_capacity
    assert decision.approved_quantity == expected_quantity


def test_correlation_and_liquidity_thresholds_reject_before_sizing() -> None:
    high_correlation = CorrelationEvidence(
        complete=True,
        existing_position_count=2,
        max_absolute_correlation=Decimal("0.81"),
        observed_at=NOW - timedelta(seconds=1),
    )
    low_liquidity = LiquidityEvidence(
        complete=True,
        average_daily_dollar_volume_usd=Decimal("999999"),
        observed_at=NOW - timedelta(seconds=1),
    )

    assert _evaluate(evidence=_evidence(correlation=high_correlation)).reasons == (
        RiskReason.CORRELATION_LIMIT,
    )
    assert _evaluate(evidence=_evidence(liquidity=low_liquidity)).reasons == (
        RiskReason.LIQUIDITY_LIMIT,
    )


@pytest.mark.parametrize(
    ("intent", "flags", "reason"),
    [
        (_intent(), _flags(risk_contract_enabled=False), RiskReason.CONTRACT_NOT_READY),
        (
            _intent(),
            _flags(refreshed_quote_revalidation_enabled=False),
            RiskReason.CONTRACT_NOT_READY,
        ),
        (
            _intent(source=SignalSource.AI_DISCOVERY),
            _flags(),
            RiskReason.STRATEGY_NOT_READY,
        ),
        (
            _intent(source=SignalSource.PAIRS),
            _flags(),
            RiskReason.STRATEGY_NOT_READY,
        ),
        (
            _intent(source=SignalSource.SMART_EXECUTION),
            _flags(),
            RiskReason.STRATEGY_NOT_READY,
        ),
        (
            _intent(side=EntrySide.SELL_SHORT),
            _flags(),
            RiskReason.SIDE_NOT_READY,
        ),
    ],
)
def test_readiness_flags_fail_closed(
    intent: EntryIntent,
    flags: EntryFeatureFlags,
    reason: RiskReason,
) -> None:
    decision = _evaluate(intent=intent, flags=flags)

    assert decision.risk_approved is False
    assert reason in decision.reasons


def test_expired_intent_and_stale_account_snapshot_fail_closed() -> None:
    expired = _intent(expires_at=NOW - timedelta(seconds=1))
    expired_decision = _evaluate(intent=expired)
    stale_decision = _evaluate(evidence=_evidence(observed_at=NOW - timedelta(seconds=31)))

    assert expired_decision.reasons == (RiskReason.INTENT_NOT_ACTIVE,)
    assert stale_decision.reasons == (RiskReason.STALE_ACCOUNT_EVIDENCE,)


@pytest.mark.parametrize(
    "field_name",
    [
        "portfolio_equity_usd",
        "cash_available_usd",
        "buying_power_usd",
        "daily_executed_notional_usd",
    ],
)
def test_money_and_fraction_inputs_reject_binary_float_units(field_name: str) -> None:
    if field_name == "portfolio_equity_usd":
        with pytest.raises(EntryRiskContractError, match="exact finite Decimal"):
            _evidence(**{field_name: 100000.0})
    else:
        with pytest.raises(EntryRiskContractError, match="exact finite Decimal"):
            _evidence(**{field_name: 1.0})

    with pytest.raises(EntryRiskContractError, match="exact finite Decimal"):
        _signal(requested_position_fraction=0.02)


@pytest.mark.parametrize(
    "changes",
    [
        {"cash_available_usd": Decimal("0")},
        {"buying_power_usd": Decimal("0")},
        {"daily_executed_notional_usd": Decimal("50000")},
    ],
)
def test_exhausted_capacity_rejects_without_authority(changes: dict[str, Decimal]) -> None:
    decision = _evaluate(evidence=_evidence(**changes))

    assert decision.reasons == (RiskReason.NO_CAPACITY,)
    assert decision.approved_quantity == 0
    assert decision.authorizes_order_submission is False

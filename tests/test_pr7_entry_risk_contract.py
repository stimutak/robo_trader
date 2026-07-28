"""Adversarial tests for the dormant PR-7 Gate-A entry-risk contract."""

import copy
import pickle
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_UP, Decimal, Overflow, localcontext

import pytest

from robo_trader.risk import entry_contract as entry_contract_module
from robo_trader.risk.entry_contract import (
    ENTRY_RISK_CONTRACT_VERSION,
    GATE_A_MAX_POSITION_FRACTION,
    ConsumedRiskDecision,
    CorrelationEvidence,
    EntryBrokerContractIdentity,
    EntryFeatureFlags,
    EntryIntent,
    EntryRiskContractError,
    EntryRiskEvidence,
    EntryRiskLimits,
    EntrySide,
    EntrySignal,
    LimitingCapacity,
    LiquidityEvidence,
    LiveBrokerQuoteSourceCapability,
    MLCorroborationEvidence,
    RefreshedQuoteEvidence,
    RefreshedQuoteSource,
    RiskReason,
    SignalSource,
    _DecisionReplayTombstones,
    assert_and_consume_risk_decision,
    build_correlation_evidence,
    build_entry_intent,
    build_entry_risk_evidence,
    build_entry_signal,
    build_liquidity_evidence,
    build_ml_corroboration_evidence,
    build_refreshed_quote_evidence,
    evaluate_entry_intent,
    produce_live_broker_quote_source,
)

NOW = datetime(2026, 7, 28, 15, 0, tzinfo=timezone.utc)
ACTIVE_GENERATION = "ib-session-42"
SOURCE_DATA_VERSION = "bars-20260728T145900Z"
QUOTE_PRODUCER_ID = "ibkr-readonly-worker-7"
_UNSET = object()


def _contract(**changes: object) -> EntryBrokerContractIdentity:
    values: dict[str, object] = {
        "con_id": 265598,
        "symbol": "AAPL",
        "local_symbol": "AAPL",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "trading_class": "NMS",
    }
    values.update(changes)
    return EntryBrokerContractIdentity(**values)  # type: ignore[arg-type]


def _signal(**changes: object) -> EntrySignal:
    values: dict[str, object] = {
        "signal_id": "sig-001",
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "side": EntrySide.BUY,
        "source": SignalSource.BASE_STRATEGY,
        "confidence_fraction": Decimal("0.81"),
        "requested_position_fraction": Decimal("0.10"),
        "source_data_version": SOURCE_DATA_VERSION,
        "broker_contract": _contract(),
        "transport_generation": ACTIVE_GENERATION,
        "observed_at": NOW - timedelta(seconds=10),
        "expires_at": NOW + timedelta(minutes=2),
    }
    values.update(changes)
    return build_entry_signal(**values)  # type: ignore[arg-type]


def _intent(
    *,
    intent_id: str = "intent-001",
    quote_refresh_request_id: str = "refresh-001",
    created_at: datetime = NOW - timedelta(seconds=5),
    **signal_changes: object,
) -> EntryIntent:
    return build_entry_intent(
        _signal(**signal_changes),
        intent_id=intent_id,
        quote_refresh_request_id=quote_refresh_request_id,
        created_at=created_at,
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
        "broker_contract": _contract(),
        "price_usd": Decimal("333"),
        "observed_at": NOW - timedelta(seconds=1),
        "transport_generation": ACTIVE_GENERATION,
    }
    values.update(changes)
    source = produce_live_broker_quote_source(
        producer_id=QUOTE_PRODUCER_ID,
        **values,  # type: ignore[arg-type]
    )
    return build_refreshed_quote_evidence(source_capability=source)


def _correlation(**changes: object) -> CorrelationEvidence:
    values: dict[str, object] = {
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "source_data_version": SOURCE_DATA_VERSION,
        "broker_contract": _contract(),
        "transport_generation": ACTIVE_GENERATION,
        "complete": True,
        "existing_position_count": 0,
        "max_absolute_correlation": Decimal("0"),
        "observed_at": NOW - timedelta(seconds=1),
    }
    values.update(changes)
    return build_correlation_evidence(**values)  # type: ignore[arg-type]


def _liquidity(**changes: object) -> LiquidityEvidence:
    values: dict[str, object] = {
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "source_data_version": SOURCE_DATA_VERSION,
        "broker_contract": _contract(),
        "transport_generation": ACTIVE_GENERATION,
        "complete": True,
        "average_daily_dollar_volume_usd": Decimal("10000000"),
        "observed_at": NOW - timedelta(seconds=1),
    }
    values.update(changes)
    return build_liquidity_evidence(**values)  # type: ignore[arg-type]


def _ml(**changes: object) -> MLCorroborationEvidence:
    values: dict[str, object] = {
        "signal_id": "sig-001",
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "source_data_version": SOURCE_DATA_VERSION,
        "broker_contract": _contract(),
        "transport_generation": ACTIVE_GENERATION,
        "model_id": "entry-ml-v1",
        "corroborated": True,
        "confidence_fraction": Decimal("0.77"),
        "observed_at": NOW - timedelta(seconds=1),
        "expires_at": NOW + timedelta(seconds=10),
    }
    values.update(changes)
    return build_ml_corroboration_evidence(**values)  # type: ignore[arg-type]


def _evidence(**changes: object) -> EntryRiskEvidence:
    quote = changes.pop("quote") if "quote" in changes else _quote()
    correlation = changes.pop("correlation") if "correlation" in changes else _correlation()
    liquidity = changes.pop("liquidity") if "liquidity" in changes else _liquidity()
    ml = changes.pop("ml_corroboration") if "ml_corroboration" in changes else None
    values: dict[str, object] = {
        "portfolio_id": "portfolio-alpha",
        "symbol": "AAPL",
        "observed_at": NOW - timedelta(seconds=1),
        "quote": quote,
        "sector": "Technology",
        "correlation": correlation,
        "liquidity": liquidity,
        "ml_corroboration": ml,
        "portfolio_equity_usd": Decimal("100000"),
        "cash_available_usd": Decimal("50000"),
        "buying_power_usd": Decimal("100000"),
        "current_symbol_gross_notional_usd": Decimal("0"),
        "current_sector_gross_notional_usd": Decimal("0"),
        "portfolio_gross_notional_usd": Decimal("0"),
        "daily_executed_notional_usd": Decimal("0"),
    }
    values.update(changes)
    return build_entry_risk_evidence(**values)  # type: ignore[arg-type]


def _evaluate(
    *,
    intent: EntryIntent | None = None,
    evidence: EntryRiskEvidence | None = None,
    limits: EntryRiskLimits | None = None,
    flags: EntryFeatureFlags | None = None,
    expected_broker_contract: object = _UNSET,
    expected_transport_generation: object = ACTIVE_GENERATION,
    expected_quote_producer_id: object = QUOTE_PRODUCER_ID,
    evaluated_at: datetime = NOW,
):
    authoritative_contract = (
        _contract() if expected_broker_contract is _UNSET else expected_broker_contract
    )
    return evaluate_entry_intent(
        intent or _intent(),
        evidence or _evidence(),
        limits or _limits(),
        flags or _flags(),
        expected_broker_contract=authoritative_contract,  # type: ignore[arg-type]
        expected_transport_generation=expected_transport_generation,  # type: ignore[arg-type]
        expected_quote_producer_id=expected_quote_producer_id,  # type: ignore[arg-type]
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
    assert intent.broker_contract is signal.broker_contract
    assert intent.transport_generation == signal.transport_generation
    assert decision.signal_id == signal.signal_id
    assert decision.portfolio_id == signal.portfolio_id
    assert decision.symbol == signal.symbol
    assert decision.side is signal.side
    assert decision.broker_contract == signal.broker_contract
    assert decision.transport_generation == signal.transport_generation
    assert decision.expires_at == intent.expires_at
    assert decision.authorizes_order_submission is False
    assert decision.runtime_integration_ready is False
    with pytest.raises(FrozenInstanceError):
        intent.symbol = "MSFT"  # type: ignore[misc]


def test_exact_contract_and_active_generation_are_legitimately_approved() -> None:
    contract = _contract()
    intent = _intent(
        broker_contract=contract,
        transport_generation=ACTIVE_GENERATION,
    )
    quote = _quote(
        broker_contract=contract,
        transport_generation=ACTIVE_GENERATION,
    )

    decision = _evaluate(
        intent=intent,
        evidence=_evidence(quote=quote),
        expected_broker_contract=contract,
        expected_transport_generation=ACTIVE_GENERATION,
    )

    assert decision.risk_approved is True
    assert decision.reasons == (RiskReason.APPROVED,)
    assert decision.broker_contract is contract
    assert decision.transport_generation == ACTIVE_GENERATION


def test_reconnect_retires_matching_pre_reconnect_quote_and_intent() -> None:
    retired_generation = "ib-session-41"
    intent = _intent(transport_generation=retired_generation)
    quote = _quote(transport_generation=retired_generation)

    decision = _evaluate(
        intent=intent,
        evidence=_evidence(quote=quote),
        expected_transport_generation=ACTIVE_GENERATION,
    )

    assert decision.risk_approved is False
    assert RiskReason.INTENT_BROKER_LINEAGE_MISMATCH in decision.reasons
    assert RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH in decision.reasons


def test_same_symbol_different_con_id_quote_is_rejected() -> None:
    wrong_contract = _contract(con_id=265599)
    quote = _quote(broker_contract=wrong_contract)

    decision = _evaluate(evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert decision.reasons == (RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH,)


@pytest.mark.parametrize(
    "intent_generation",
    ["ib-session-41", "ib-session-43", "ib-session-substituted"],
)
def test_intent_generation_rollback_future_and_substitution_are_rejected(
    intent_generation: str,
) -> None:
    intent = _intent(transport_generation=intent_generation)
    quote = _quote(transport_generation=intent_generation)

    decision = _evaluate(intent=intent, evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert RiskReason.INTENT_BROKER_LINEAGE_MISMATCH in decision.reasons
    assert RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH in decision.reasons


@pytest.mark.parametrize(
    "quote_generation",
    ["ib-session-41", "ib-session-43", "ib-session-substituted"],
)
def test_quote_generation_rollback_future_and_substitution_are_rejected(
    quote_generation: str,
) -> None:
    quote = _quote(transport_generation=quote_generation)

    decision = _evaluate(evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert decision.reasons == (RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH,)


def test_intent_contract_substitution_is_rejected_even_with_matching_quote() -> None:
    substituted_contract = _contract(primary_exchange="NYSE")
    intent = _intent(broker_contract=substituted_contract)
    quote = _quote(broker_contract=substituted_contract)

    decision = _evaluate(intent=intent, evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert RiskReason.INTENT_BROKER_LINEAGE_MISMATCH in decision.reasons
    assert RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH in decision.reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"con_id": None},
        {"con_id": True},
        {"con_id": 0},
        {"local_symbol": "MSFT"},
        {"security_type": "OPT"},
        {"currency": "EUR"},
        {"exchange": "NYSE"},
        {"primary_exchange": ""},
        {"trading_class": ""},
    ],
)
def test_broker_contract_identity_requires_complete_canonical_stock_lineage(
    changes: dict[str, object],
) -> None:
    with pytest.raises(EntryRiskContractError, match="broker contract"):
        _contract(**changes)


@pytest.mark.parametrize("construction", ["mutated", "raw"])
def test_broker_contract_is_revalidated_before_capability_mint(construction: str) -> None:
    valid = _contract()
    if construction == "mutated":
        corrupted = valid
    else:
        corrupted = object.__new__(EntryBrokerContractIdentity)
        for field in fields(valid):
            object.__setattr__(corrupted, field.name, getattr(valid, field.name))
    object.__setattr__(corrupted, "security_type", "OPT")

    with pytest.raises(EntryRiskContractError, match="security_type must be STK"):
        _signal(broker_contract=corrupted)


@pytest.mark.parametrize(
    ("expected_contract", "expected_generation"),
    [(None, ACTIVE_GENERATION), (_contract(), None), (_contract(), "")],
)
def test_missing_authoritative_contract_or_generation_is_rejected(
    expected_contract: object,
    expected_generation: object,
) -> None:
    with pytest.raises(EntryRiskContractError, match="expected|transport"):
        _evaluate(
            expected_broker_contract=expected_contract,
            expected_transport_generation=expected_generation,
        )


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps, replace])
@pytest.mark.parametrize("evidence_factory", [_intent, _quote])
def test_intent_and_quote_cannot_be_copied_or_replaced(
    operation,
    evidence_factory,
) -> None:
    with pytest.raises((EntryRiskContractError, TypeError, ValueError)):
        operation(evidence_factory())


@pytest.mark.parametrize("kind", ["correlation", "liquidity"])
@pytest.mark.parametrize(
    "changes",
    [
        {"portfolio_id": "portfolio-beta"},
        {"source_data_version": "bars-other"},
        {"broker_contract": _contract(con_id=265599)},
        {"transport_generation": "ib-session-other"},
    ],
)
def test_nested_market_evidence_is_bound_to_the_exact_intent_scope(
    kind: str,
    changes: dict[str, object],
) -> None:
    builder = _correlation if kind == "correlation" else _liquidity
    nested = builder(**changes)

    decision = _evaluate(evidence=_evidence(**{kind: nested}))

    assert decision.risk_approved is False
    assert RiskReason.EVIDENCE_SCOPE_MISMATCH in decision.reasons


@pytest.mark.parametrize("kind", ["correlation", "liquidity"])
def test_nested_market_evidence_rejects_cross_symbol_substitution(kind: str) -> None:
    microsoft = _contract(
        con_id=272093,
        symbol="MSFT",
        local_symbol="MSFT",
        primary_exchange="NASDAQ",
        trading_class="NMS",
    )
    builder = _correlation if kind == "correlation" else _liquidity
    nested = builder(symbol="MSFT", broker_contract=microsoft)

    decision = _evaluate(evidence=_evidence(**{kind: nested}))

    assert decision.risk_approved is False
    assert RiskReason.EVIDENCE_SCOPE_MISMATCH in decision.reasons


def test_ai_discovery_flag_alone_cannot_replace_ml_corroboration() -> None:
    intent = _intent(source=SignalSource.AI_DISCOVERY)

    decision = _evaluate(
        intent=intent,
        flags=_flags(ai_discovery_entries_enabled=True),
    )

    assert decision.risk_approved is False
    assert RiskReason.MISSING_ML_CORROBORATION in decision.reasons


def test_fresh_exact_ml_corroboration_allows_enabled_ai_discovery_risk_evaluation() -> None:
    contract = _contract()
    intent = _intent(
        source=SignalSource.AI_DISCOVERY,
        broker_contract=contract,
    )
    ml = _ml(broker_contract=contract)

    decision = _evaluate(
        intent=intent,
        evidence=_evidence(ml_corroboration=ml),
        flags=_flags(ai_discovery_entries_enabled=True),
        expected_broker_contract=contract,
    )

    assert decision.risk_approved is True
    assert decision.reasons == (RiskReason.APPROVED,)


@pytest.mark.parametrize(
    "ml_changes",
    [
        {"signal_id": "sig-other"},
        {"portfolio_id": "portfolio-beta"},
        {"source_data_version": "bars-other"},
        {"broker_contract": _contract(con_id=265599)},
        {"transport_generation": "ib-session-other"},
        {"corroborated": False},
        {"confidence_fraction": Decimal("0")},
        {
            "observed_at": NOW - timedelta(seconds=40),
            "expires_at": NOW + timedelta(seconds=1),
        },
        {
            "observed_at": NOW - timedelta(seconds=2),
            "expires_at": NOW - timedelta(seconds=1),
        },
    ],
)
def test_ai_discovery_rejects_unscoped_stale_or_negative_ml_corroboration(
    ml_changes: dict[str, object],
) -> None:
    intent = _intent(source=SignalSource.AI_DISCOVERY)
    ml = _ml(**ml_changes)

    decision = _evaluate(
        intent=intent,
        evidence=_evidence(ml_corroboration=ml),
        flags=_flags(ai_discovery_entries_enabled=True),
    )

    assert decision.risk_approved is False
    assert {
        RiskReason.EVIDENCE_SCOPE_MISMATCH,
        RiskReason.ML_CORROBORATION_REQUIRED,
    }.intersection(decision.reasons)


def test_ai_discovery_zero_confidence_signal_fails_despite_positive_ml() -> None:
    intent = _intent(
        source=SignalSource.AI_DISCOVERY,
        confidence_fraction=Decimal("0"),
    )

    decision = _evaluate(
        intent=intent,
        evidence=_evidence(ml_corroboration=_ml()),
        flags=_flags(ai_discovery_entries_enabled=True),
    )

    assert decision.risk_approved is False
    assert RiskReason.ML_CORROBORATION_REQUIRED in decision.reasons


def _raw_clone(capability):
    values = {field.name: getattr(capability, field.name) for field in fields(capability)}
    return type(capability)(**values)


def _source_capability() -> LiveBrokerQuoteSourceCapability:
    return produce_live_broker_quote_source(
        producer_id=QUOTE_PRODUCER_ID,
        quote_id="quote-source-only",
        refresh_request_id="refresh-source-only",
        broker_contract=_contract(),
        price_usd=Decimal("333"),
        observed_at=NOW - timedelta(seconds=1),
        transport_generation=ACTIVE_GENERATION,
    )


def _decision():
    return _evaluate()


def _decision_with_ids(suffix: str):
    refresh_id = f"refresh-{suffix}"
    intent = _intent(
        intent_id=f"intent-{suffix}",
        signal_id=f"sig-{suffix}",
        quote_refresh_request_id=refresh_id,
    )
    quote = _quote(
        quote_id=f"quote-{suffix}",
        refresh_request_id=refresh_id,
    )
    return _evaluate(intent=intent, evidence=_evidence(quote=quote))


@pytest.mark.parametrize(
    "factory",
    [
        _signal,
        _intent,
        _source_capability,
        _quote,
        _correlation,
        _liquidity,
        _ml,
        _evidence,
        _decision,
    ],
)
def test_sealed_contract_records_reject_raw_reconstruction(factory) -> None:
    capability = factory()

    assert getattr(capability, "_seal", None) is None
    with pytest.raises(EntryRiskContractError, match="factory-only"):
        _raw_clone(capability)


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy, pickle.dumps, replace])
@pytest.mark.parametrize(
    "factory",
    [
        _signal,
        _intent,
        _source_capability,
        _quote,
        _correlation,
        _liquidity,
        _ml,
        _evidence,
        _decision,
    ],
)
def test_all_sealed_contract_records_reject_copy_pickle_and_replace(
    operation,
    factory,
) -> None:
    with pytest.raises((EntryRiskContractError, TypeError, ValueError)):
        operation(factory())


def test_quote_builder_requires_and_consumes_exact_typed_source_capability() -> None:
    with pytest.raises(EntryRiskContractError, match="exact LiveBrokerQuoteSourceCapability"):
        build_refreshed_quote_evidence(source_capability=object())  # type: ignore[arg-type]

    source = _source_capability()
    quote = build_refreshed_quote_evidence(source_capability=source)
    assert quote.source is RefreshedQuoteSource.LIVE_BROKER
    with pytest.raises(EntryRiskContractError, match="already consumed"):
        build_refreshed_quote_evidence(source_capability=source)


def test_quote_from_another_registered_producer_is_rejected() -> None:
    source = produce_live_broker_quote_source(
        producer_id="ibkr-readonly-worker-other",
        quote_id="quote-other-producer",
        refresh_request_id="refresh-001",
        broker_contract=_contract(),
        price_usd=Decimal("333"),
        observed_at=NOW - timedelta(seconds=1),
        transport_generation=ACTIVE_GENERATION,
    )
    quote = build_refreshed_quote_evidence(source_capability=source)

    decision = _evaluate(evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert decision.reasons == (RiskReason.QUOTE_SOURCE_MISMATCH,)


def test_capability_mutation_and_replay_cannot_mint_authority() -> None:
    signal = _signal()
    object.__setattr__(signal, "symbol", "MSFT")
    with pytest.raises(EntryRiskContractError, match="altered"):
        build_entry_intent(
            signal,
            intent_id="intent-mutated",
            quote_refresh_request_id="refresh-mutated",
            created_at=NOW - timedelta(seconds=5),
        )

    quote = _quote()
    object.__setattr__(quote, "price_usd", Decimal("1"))
    with pytest.raises(EntryRiskContractError, match="altered"):
        _evidence(quote=quote)

    intent = _intent()
    evidence = _evidence()
    object.__setattr__(evidence, "portfolio_id", "portfolio-beta")
    with pytest.raises(EntryRiskContractError, match="altered"):
        _evaluate(intent=intent, evidence=evidence)


def test_signal_intent_evidence_and_decision_are_each_single_use() -> None:
    signal = _signal()
    build_entry_intent(
        signal,
        intent_id="intent-first",
        quote_refresh_request_id="refresh-first",
        created_at=NOW - timedelta(seconds=5),
    )
    with pytest.raises(EntryRiskContractError, match="already consumed"):
        build_entry_intent(
            signal,
            intent_id="intent-second",
            quote_refresh_request_id="refresh-second",
            created_at=NOW - timedelta(seconds=4),
        )

    intent = _intent()
    evidence = _evidence()
    decision = _evaluate(intent=intent, evidence=evidence)
    with pytest.raises(EntryRiskContractError, match="already consumed"):
        _evaluate(intent=intent, evidence=_evidence())
    consumed = assert_and_consume_risk_decision(decision, consumed_at=NOW)
    assert type(consumed) is ConsumedRiskDecision
    assert consumed.intent_id == decision.intent_id
    assert consumed.approved_quantity == decision.approved_quantity
    assert consumed.expires_at == decision.expires_at
    assert consumed.authorizes_order_submission is False
    assert consumed.runtime_integration_ready is False
    with pytest.raises(EntryRiskContractError, match="already consumed"):
        assert_and_consume_risk_decision(decision, consumed_at=NOW)


def test_mutated_or_replaced_risk_decision_cannot_be_consumed() -> None:
    replaced_target = _decision()
    with pytest.raises(EntryRiskContractError, match="factory-only"):
        replace(
            replaced_target,
            risk_approved=True,
            approved_quantity=999,
            approved_notional_usd=Decimal("999999"),
        )

    mutated = _decision()
    object.__setattr__(mutated, "approved_quantity", 999)
    with pytest.raises(EntryRiskContractError, match="altered"):
        assert_and_consume_risk_decision(mutated, consumed_at=NOW)


def test_equivalent_independently_minted_decision_pipeline_is_semantic_replay() -> None:
    first = _decision_with_ids("semantic-replay")
    duplicate = _decision_with_ids("semantic-replay")

    assert first is not duplicate
    assert (first.intent_id, first.signal_id, first.quote_id) == (
        duplicate.intent_id,
        duplicate.signal_id,
        duplicate.quote_id,
    )
    assert_and_consume_risk_decision(first, consumed_at=NOW)
    with pytest.raises(EntryRiskContractError, match="semantic replay"):
        assert_and_consume_risk_decision(duplicate, consumed_at=NOW)


def test_consumed_decision_is_a_deeply_immutable_non_authorizing_snapshot() -> None:
    decision = _decision_with_ids("immutable-transfer")
    consumed = assert_and_consume_risk_decision(decision, consumed_at=NOW)
    alias = consumed

    object.__setattr__(decision, "authorizes_order_submission", True)
    object.__setattr__(decision, "runtime_integration_ready", True)
    object.__setattr__(decision.broker_contract, "security_type", "OPT")

    assert alias is consumed
    assert consumed.authorizes_order_submission is False
    assert consumed.runtime_integration_ready is False
    assert consumed.broker_contract[3] == "STK"
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(consumed, "authorizes_order_submission", True)
    with pytest.raises(TypeError):
        consumed.broker_contract[3] = "OPT"  # type: ignore[index]

    for operation in (copy.copy, copy.deepcopy, pickle.dumps, replace):
        with pytest.raises((EntryRiskContractError, TypeError, ValueError)):
            operation(consumed)
    with pytest.raises(EntryRiskContractError, match="factory-only"):
        ConsumedRiskDecision(tuple(consumed))

    raw = tuple.__new__(ConsumedRiskDecision, tuple(consumed))
    assert raw.authorizes_order_submission is False
    assert raw.runtime_integration_ready is False
    with pytest.raises(EntryRiskContractError, match="exact RiskDecision"):
        assert_and_consume_risk_decision(raw, consumed_at=NOW)


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


def test_ambient_decimal_context_cannot_round_the_two_percent_cap_up() -> None:
    equity = Decimal("99999")
    exact_cap = Decimal("1999.98")
    evidence = _evidence(
        portfolio_equity_usd=equity,
        quote=_quote(price_usd=Decimal("1000")),
    )

    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_UP
        context.Emax = 3
        context.Emin = -3
        context.clamp = 1
        context.traps[Overflow] = False
        decision = _evaluate(evidence=evidence)

    assert decision.risk_approved is True
    assert decision.approved_quantity == 1
    assert decision.approved_notional_usd == Decimal("1000")
    assert decision.approved_notional_usd <= exact_cap
    assert decision.limiting_capacity is LimitingCapacity.SYMBOL


def test_large_coefficients_remain_exact_under_hostile_ambient_context() -> None:
    equity = Decimal("9" * 250 + ".99")
    generous_capacity = Decimal("9" * 252)
    price = Decimal("1" + "0" * 248)
    evidence = _evidence(
        portfolio_equity_usd=equity,
        cash_available_usd=generous_capacity,
        buying_power_usd=generous_capacity,
        current_symbol_gross_notional_usd=Decimal("0.01"),
        quote=_quote(price_usd=price),
        liquidity=_liquidity(
            average_daily_dollar_volume_usd=generous_capacity,
            observed_at=NOW - timedelta(seconds=1),
        ),
    )
    limits = _limits(max_daily_notional_usd=generous_capacity)

    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        context.Emax = 9
        context.Emin = -9
        context.clamp = 1
        context.traps[Overflow] = False
        decision = _evaluate(evidence=evidence, limits=limits)

    assert decision.risk_approved is True
    assert decision.approved_quantity == 1
    assert decision.approved_notional_usd == price
    assert decision.limiting_capacity is LimitingCapacity.SYMBOL


def test_final_postcondition_checks_every_capacity_not_only_reported_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        entry_contract_module,
        "_minimum_capacity",
        lambda capacities: (LimitingCapacity.REQUEST, Decimal("3000")),
    )

    with pytest.raises(
        EntryRiskContractError,
        match="exceeded the exact symbol capacity",
    ):
        _evaluate()


def test_gate_a_rejects_a_configured_position_cap_above_two_percent() -> None:
    with pytest.raises(EntryRiskContractError, match="cannot exceed 2%"):
        _limits(max_position_fraction=Decimal("0.0200001"))


@pytest.mark.parametrize("construction", ["mutated", "raw"])
def test_authorization_boundary_revalidates_exact_limits(construction: str) -> None:
    valid = _limits()
    if construction == "mutated":
        corrupted = valid
    else:
        corrupted = object.__new__(EntryRiskLimits)
        for field in fields(valid):
            object.__setattr__(corrupted, field.name, getattr(valid, field.name))
    object.__setattr__(corrupted, "max_position_fraction", Decimal("0.50"))

    with pytest.raises(EntryRiskContractError, match="cannot exceed 2%"):
        _evaluate(limits=corrupted)


def test_authorization_boundary_revalidates_flags_and_nested_evidence() -> None:
    flags = _flags()
    object.__setattr__(flags, "risk_contract_enabled", 1)
    with pytest.raises(EntryRiskContractError, match="explicit boolean"):
        _evaluate(flags=flags)

    evidence = _evidence()
    assert evidence.correlation is not None
    object.__setattr__(evidence.correlation, "complete", "yes")
    with pytest.raises(EntryRiskContractError, match="altered"):
        _evaluate(evidence=evidence)


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
        _quote(
            broker_contract=_contract(
                con_id=272093,
                symbol="MSFT",
                local_symbol="MSFT",
                primary_exchange="NASDAQ",
                trading_class="NMS",
            )
        ),
        _quote(observed_at=NOW - timedelta(seconds=6)),
        _quote(observed_at=NOW + timedelta(microseconds=1)),
    ],
)
def test_quote_must_be_fresh_and_bound_to_the_intent(quote: RefreshedQuoteEvidence) -> None:
    decision = _evaluate(evidence=_evidence(quote=quote))

    assert decision.risk_approved is False
    assert RiskReason.QUOTE_NOT_REFRESHED in decision.reasons


@pytest.mark.parametrize("kind", ["correlation", "liquidity"])
def test_incomplete_or_stale_market_evidence_fails_closed(kind: str) -> None:
    builder = _correlation if kind == "correlation" else _liquidity
    incomplete = builder(complete=False)
    stale = builder(observed_at=NOW - timedelta(seconds=31))
    expected = (
        RiskReason.MISSING_CORRELATION if kind == "correlation" else RiskReason.MISSING_LIQUIDITY
    )

    incomplete_decision = _evaluate(evidence=_evidence(**{kind: incomplete}))
    stale_decision = _evaluate(evidence=_evidence(**{kind: stale}))

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
    high_correlation = _correlation(
        existing_position_count=2,
        max_absolute_correlation=Decimal("0.81"),
    )
    low_liquidity = _liquidity(
        average_daily_dollar_volume_usd=Decimal("999999"),
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


def test_datetime_subclasses_cannot_override_freshness_authority() -> None:
    class AdversarialDateTime(datetime):
        def astimezone(self, tz=None):
            return self

        def __lt__(self, other):
            return False

        def __le__(self, other):
            return True

        def __ge__(self, other):
            return False

        def __sub__(self, other):
            return timedelta(0)

    adversarial = AdversarialDateTime.fromtimestamp(NOW.timestamp(), timezone.utc)
    operations = (
        lambda: _signal(observed_at=adversarial),
        lambda: _intent(created_at=adversarial),
        lambda: _quote(observed_at=adversarial),
        lambda: _correlation(observed_at=adversarial),
        lambda: _liquidity(observed_at=adversarial),
        lambda: _ml(observed_at=adversarial),
        lambda: _evidence(observed_at=adversarial),
        lambda: _evaluate(evaluated_at=adversarial),
        lambda: assert_and_consume_risk_decision(
            _decision_with_ids("adversarial-consumed-at"),
            consumed_at=adversarial,
        ),
    )

    for operation in operations:
        with pytest.raises(EntryRiskContractError, match="exact timezone-aware datetime"):
            operation()


def test_timedelta_subclasses_are_not_risk_limit_authority() -> None:
    class AdversarialTimedelta(timedelta):
        pass

    with pytest.raises(EntryRiskContractError, match="positive timedelta"):
        _limits(max_quote_age=AdversarialTimedelta(seconds=10))


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


def test_more_than_replay_limit_sequential_expired_decisions_continue() -> None:
    tombstones = _DecisionReplayTombstones(4096)

    for index in range(4097):
        consumed_at = NOW + timedelta(seconds=index)
        tombstones.record(
            ("risk-decision-v1", 1, "portfolio-alpha", f"intent-expired-{index}"),
            expires_at=consumed_at + timedelta(seconds=1),
            consumed_at=consumed_at,
        )


def test_live_replay_tombstones_exhaust_only_when_every_entry_remains_live() -> None:
    tombstones = _DecisionReplayTombstones(4096)
    live_expiry = NOW + timedelta(days=1)

    for index in range(4096):
        tombstones.record(
            ("risk-decision-v1", 1, "portfolio-alpha", f"intent-live-{index}"),
            expires_at=live_expiry,
            consumed_at=NOW,
        )

    with pytest.raises(EntryRiskContractError, match="live replay registry capacity"):
        tombstones.record(
            ("risk-decision-v1", 1, "portfolio-alpha", "intent-live-overflow"),
            expires_at=live_expiry,
            consumed_at=NOW,
        )


def test_duplicate_rejects_before_and_after_its_tombstone_expires() -> None:
    tombstones = _DecisionReplayTombstones(2)
    replay_key = ("risk-decision-v1", 1, "portfolio-alpha", "intent-replayed")
    expiry = NOW + timedelta(seconds=1)
    tombstones.record(replay_key, expires_at=expiry, consumed_at=NOW)

    with pytest.raises(EntryRiskContractError, match="semantic replay"):
        tombstones.record(
            replay_key,
            expires_at=expiry,
            consumed_at=NOW + timedelta(microseconds=1),
        )

    after_expiry = NOW + timedelta(seconds=2)
    tombstones.record(
        ("risk-decision-v1", 1, "portfolio-alpha", "intent-after-expiry"),
        expires_at=after_expiry + timedelta(seconds=1),
        consumed_at=after_expiry,
    )
    with pytest.raises(EntryRiskContractError, match="expired before consumption"):
        tombstones.record(
            replay_key,
            expires_at=expiry,
            consumed_at=after_expiry,
        )


def test_replay_tombstones_detect_clock_rollback() -> None:
    tombstones = _DecisionReplayTombstones(2)
    tombstones.record(
        ("risk-decision-v1", 1, "portfolio-alpha", "intent-newer"),
        expires_at=NOW + timedelta(seconds=10),
        consumed_at=NOW,
    )

    with pytest.raises(EntryRiskContractError, match="clock rollback"):
        tombstones.record(
            ("risk-decision-v1", 1, "portfolio-alpha", "intent-older"),
            expires_at=NOW + timedelta(seconds=10),
            consumed_at=NOW - timedelta(microseconds=1),
        )


def test_risk_decision_cannot_be_consumed_at_or_after_exact_expiry() -> None:
    decision = _decision_with_ids("expired-consumption")

    with pytest.raises(EntryRiskContractError, match="expired before consumption"):
        assert_and_consume_risk_decision(
            decision,
            consumed_at=decision.expires_at,
        )

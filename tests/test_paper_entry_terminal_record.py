"""Entry terminal records bind reservation, attempt, quote and exact accounting."""

import json
import time
import uuid
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from robo_trader.execution import LocalPaperExecutionEvidence
from robo_trader.paper_entry_settlement import build_paper_entry_terminal_record
from robo_trader.paper_reduction_submitter import (
    LocalPaperTerminalOutcome,
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
)
from robo_trader.paper_terminal_settlement import PaperAccountSettlementState
from robo_trader.protective_quote_evidence import _produce_protective_quote, ProtectiveQuoteSource
from robo_trader.risk.entry_reservations import reserve_entry_capacity
from robo_trader.safety import SafetyJournal
from robo_trader.safety.entry_capacity import claim_entry_capacity
from robo_trader.safety.models import ValidationError
from robo_trader.stop_loss_monitor import StopLossMonitor
from tests.safety.conftest import ACCOUNT_A
from tests.test_pr7_entry_risk_contract import (
    NOW,
    ACTIVE_GENERATION,
    _evaluate,
    _evidence,
    _quote,
    _intent,
)


@pytest.fixture
def case(tmp_path, request):
    price = getattr(request, "param", Decimal("333"))
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(), risk_manager=None, portfolio_id="portfolio-alpha"
    )
    quote = _produce_protective_quote(
        monitor,
        portfolio_id="portfolio-alpha",
        symbol="AAPL",
        con_id=265598,
        price=price,
        source_timestamp=NOW - timedelta(seconds=1),
        receipt_monotonic=time.monotonic(),
        receipt_order=1,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        transport_generation=ACTIVE_GENERATION,
        source_event_id="ticker-1",
    )
    decision = _evaluate(
        intent=_intent(intent_id="intent-" + uuid.uuid4().hex),
        evidence=_evidence(quote=_quote(quote_id=quote.quote_id, price_usd=price)),
    )
    journal = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
    journal.initialize(execution_domain_scope="paper-simulator-v1", account_scope=ACCOUNT_A)
    reserved = reserve_entry_capacity(
        journal, decision, sector="Technology", expected_head=(0, "0" * 64)
    )
    claim = claim_entry_capacity(
        journal,
        reservation_sequence=reserved.sequence,
        reservation_chain_hash=reserved.chain_hash,
        expected_head=(reserved.sequence, reserved.chain_hash),
    )
    quantity = Decimal(json.loads(reserved.payload_json)["quantity"])
    evidence = LocalPaperExecutionEvidence(
        "lpfill-" + "a" * 32,
        quantity,
        price,
        0,
        "USD",
        "LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1",
        NOW,
    )
    outcome = LocalPaperTerminalOutcome(
        json.loads(claim.payload_json)["order_ref"],
        LocalPaperOrderStatus.FILLED,
        quantity,
        quantity,
        Decimal("0"),
        price,
        NOW,
        LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        True,
        "filled",
        evidence,
    )
    return dict(
        reservation=reserved,
        claim=claim,
        account=PaperAccountSettlementState(
            "portfolio-alpha",
            Decimal("100000"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            "2026-07-28",
            None,
            None,
            None,
        ),
        pre_position_quantity=Decimal("0"),
        pre_aggregate_quantity=Decimal("0"),
        pre_symbol_gross_quantity=Decimal("0"),
        quote=quote,
        quote_producer=monitor,
        outcome=outcome,
    )


def test_record_is_deterministic_and_binds_all_accounting(case):
    record = build_paper_entry_terminal_record(**case)
    assert record == build_paper_entry_terminal_record(**case)
    assert len(record.fingerprint) == 64
    payload = json.loads(record.payload_json)
    assert payload["kind"] == "PAPER_ENTRY_TERMINAL_SETTLEMENT"
    assert payload["post_values"]["cash"] == "98002"
    assert payload["post_values"]["position_quantity"] == "6"
    assert payload["outcome"]["fill_evidence"]["execution_id"] == "lpfill-" + "a" * 32
    assert payload["claim"]["chain_hash"] == case["claim"].chain_hash
    assert payload["quote_id"] == case["quote"].quote_id


@pytest.mark.parametrize(
    "status",
    [
        LocalPaperOrderStatus.REJECTED,
        LocalPaperOrderStatus.CANCELLED,
        LocalPaperOrderStatus.EXPIRED,
    ],
)
def test_zero_fill_record_preserves_account(case, status):
    case["outcome"] = replace(
        case["outcome"],
        status=status,
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    payload = json.loads(build_paper_entry_terminal_record(**case).payload_json)
    assert payload["post_values"]["cash"] == "100000"
    assert payload["post_values"]["position_quantity"] == "0"


@pytest.mark.parametrize("field", ["order_ref", "before_claim", "above_reserved", "commission"])
def test_execution_must_match_claim_and_reserved_principal(case, field):
    outcome = case["outcome"]
    if field == "order_ref":
        outcome = replace(outcome, order_ref="different")
    elif field == "before_claim":
        at = NOW - timedelta(seconds=1)
        outcome = replace(
            outcome, observed_at=at, fill_evidence=replace(outcome.fill_evidence, occurred_at=at)
        )
    elif field == "above_reserved":
        outcome = replace(
            outcome,
            exact_fill_price=Decimal("334"),
            fill_evidence=replace(outcome.fill_evidence, exact_fill_price=Decimal("334")),
        )
    else:
        outcome = replace(outcome, fill_evidence=replace(outcome.fill_evidence, commission_minor=1))
    case["outcome"] = outcome
    with pytest.raises(ValidationError):
        build_paper_entry_terminal_record(**case)


@pytest.mark.parametrize(
    "field", ["pre_position_quantity", "pre_aggregate_quantity", "pre_symbol_gross_quantity"]
)
def test_existing_exposure_cannot_become_opening_entry(case, field):
    case[field] = Decimal("1")
    with pytest.raises(ValidationError):
        build_paper_entry_terminal_record(**case)


def test_wrong_account_portfolio_rejected(case):
    case["account"] = replace(case["account"], portfolio_id="other")
    with pytest.raises(ValidationError):
        build_paper_entry_terminal_record(**case)


@pytest.mark.parametrize("failure", ["copied", "producer", "missing_producer", "new_quote"])
def test_quote_must_be_exact_reserved_producer_evidence(case, failure):
    from robo_trader.protective_quote_evidence import ProtectiveQuoteValidationError

    if failure == "copied":
        case["quote"] = replace(case["quote"])
    elif failure == "producer":
        case["quote_producer"] = object()
    elif failure == "missing_producer":
        case["quote_producer"] = None
    else:
        original = case["quote"]
        case["quote"] = _produce_protective_quote(
            case["quote_producer"],
            portfolio_id=original.portfolio_id,
            symbol="AAPL",
            con_id=265598,
            price=original.price,
            source_timestamp=original.source_timestamp,
            receipt_monotonic=time.monotonic(),
            receipt_order=2,
            source=ProtectiveQuoteSource.LIVE_BROKER,
            transport_generation=ACTIVE_GENERATION,
            source_event_id="ticker-2",
        )
    with pytest.raises((ValidationError, ProtectiveQuoteValidationError)):
        build_paper_entry_terminal_record(**case)


def test_mutated_execution_evidence_is_revalidated(case):
    object.__setattr__(case["outcome"].fill_evidence, "commission_currency", "EUR")
    with pytest.raises(ValueError, match="currency"):
        build_paper_entry_terminal_record(**case)


def test_partial_outcome_cannot_become_a_terminal_record(case):
    evidence = replace(case["outcome"].fill_evidence, filled_quantity=Decimal("1"))
    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus.PARTIALLY_FILLED,
        filled_quantity=Decimal("1"),
        remaining_quantity=Decimal("5"),
        terminal=False,
        fill_evidence=evidence,
    )
    with pytest.raises(ValidationError):
        build_paper_entry_terminal_record(**case)


def test_fingerprint_changes_with_exact_fill_and_projection(case):
    original = build_paper_entry_terminal_record(**case)
    evidence = replace(case["outcome"].fill_evidence, exact_fill_price=Decimal("332.9999"))
    case["outcome"] = replace(
        case["outcome"], exact_fill_price=evidence.exact_fill_price, fill_evidence=evidence
    )
    changed = build_paper_entry_terminal_record(**case)
    assert original.fingerprint != changed.fingerprint
    assert json.loads(changed.payload_json)["post_values"]["cash"] == "98002.0006"

"""Stored entry data must be revalidated against actual journal history."""

import gc
import json
from decimal import Decimal

import pytest

from robo_trader.paper_entry_settlement import (
    build_paper_entry_terminal_record,
    validate_stored_paper_entry_terminal_record,
)
from robo_trader.safety import SafetyJournal
from robo_trader.safety.models import ValidationError, canonical_json, sha256_text
from tests.test_paper_entry_terminal_record import case  # noqa: F401


def read(record, path):
    return validate_stored_paper_entry_terminal_record(
        record.payload_json, fingerprint=record.fingerprint, journal=SafetyJournal(path)
    )


def test_historical_roundtrip_needs_no_live_quote_producer(case, tmp_path):
    record = build_paper_entry_terminal_record(**case)
    del case["quote_producer"]
    gc.collect()
    assert read(record, tmp_path / "journal.db") == record


@pytest.mark.parametrize(
    "path,value",
    [
        (("version",), True),
        (("kind",), "REDUCTION"),
        (("extra",), "value"),
        (("pre_account", "cash"), "100000.0"),
        (("pre_account", "cash"), 100000),
        (("pre_account", "extra"), "value"),
        (("post_values", "cash"), "98001"),
        (("post_values", "extra"), "value"),
        (("outcome", "filled_quantity"), "5"),
        (("outcome", "fill_evidence", "commission_minor"), True),
        (("outcome", "fill_evidence", "commission_currency"), "EUR"),
        (("outcome", "extra"), "value"),
        (("quote", "extra"), "value"),
        (("quote", "receipt_order"), True),
        (("quote", "receipt_monotonic"), "inf"),
        (("quote", "source_timestamp"), "2026-07-28T14:59:59"),
        (("quote", "price"), 333),
        (("claim", "chain_hash"), "a" * 64),
        (("reservation", "intent_fingerprint"), "a" * 64),
    ],
)
def test_altered_data_is_rejected_even_with_recomputed_record_hash(case, tmp_path, path, value):
    original = build_paper_entry_terminal_record(**case)
    payload = json.loads(original.payload_json)
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    altered = canonical_json(payload)
    with pytest.raises(ValidationError):
        validate_stored_paper_entry_terminal_record(
            altered,
            fingerprint=sha256_text(altered),
            journal=SafetyJournal(tmp_path / "journal.db"),
        )


def test_record_cannot_invent_claim_history(case, tmp_path):
    record = build_paper_entry_terminal_record(**case)
    empty = SafetyJournal(tmp_path / "empty.db")
    empty.initialize(
        execution_domain_scope=case["claim"].execution_domain_scope,
        account_scope=case["claim"].account_scope,
    )
    with pytest.raises(ValidationError, match="journal"):
        validate_stored_paper_entry_terminal_record(
            record.payload_json, fingerprint=record.fingerprint, journal=empty
        )


@pytest.mark.parametrize("failure", ["hash", "duplicate", "whitespace"])
def test_exact_fingerprint_and_canonical_json_required(case, tmp_path, failure):
    record = build_paper_entry_terminal_record(**case)
    payload, digest = record.payload_json, record.fingerprint
    if failure == "hash":
        digest = "0" * 64
    elif failure == "duplicate":
        payload = '{"version":1,' + payload[1:]
        digest = sha256_text(payload)
    else:
        payload += " "
        digest = sha256_text(payload)
    with pytest.raises(ValidationError):
        validate_stored_paper_entry_terminal_record(
            payload, fingerprint=digest, journal=SafetyJournal(tmp_path / "journal.db")
        )


def test_missing_journal_is_not_created_during_validation(case, tmp_path):
    record = build_paper_entry_terminal_record(**case)
    missing = tmp_path / "missing.db"
    with pytest.raises(ValidationError):
        read(record, missing)
    assert not missing.exists()


def test_historical_record_still_valid_after_unrelated_journal_append(case, tmp_path):
    from tests.risk.test_pending_entry_capacity import append
    from tests.test_pr7_entry_risk_contract import NOW

    record = build_paper_entry_terminal_record(**case)
    journal = SafetyJournal(tmp_path / "journal.db", clock=lambda: NOW)
    append(journal, symbol="MSFT", con_id=272093)
    assert read(record, tmp_path / "journal.db") == record


def test_replay_preserves_canonical_quote_decimal_scale(case, tmp_path):
    # A protective quote's own serialization intentionally preserves its price
    # spelling; unlike accounting values, trailing zeros affect its identity.
    import time
    from decimal import Decimal

    from robo_trader.paper_entry_record_replay import _quote
    from robo_trader.protective_quote_evidence import (
        ProtectiveQuoteSource,
        _produce_protective_quote,
    )
    from tests.test_pr7_entry_risk_contract import ACTIVE_GENERATION, NOW

    quote = _produce_protective_quote(
        case["quote_producer"],
        portfolio_id="portfolio-alpha",
        symbol="AAPL",
        con_id=265598,
        price=Decimal("333.00"),
        source_timestamp=NOW,
        receipt_monotonic=time.monotonic(),
        receipt_order=2,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        transport_generation=ACTIVE_GENERATION,
        source_event_id="scaled-tick",
    )
    restored = _quote(json.loads(quote.canonical_payload()), quote.quote_id)
    assert restored.canonical_payload() == quote.canonical_payload()


@pytest.mark.parametrize("case", [Decimal("16.2")], indirect=True)
@pytest.mark.parametrize("filled", [True, False])
@pytest.mark.parametrize("trapped", [False, True])
def test_large_quantity_replay_ignores_ambient_precision(case, tmp_path, filled, trapped):
    from dataclasses import replace
    from decimal import Decimal, Inexact, Rounded, localcontext

    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    assert case["outcome"].requested_quantity == Decimal("123")
    if not filled:
        case["outcome"] = replace(
            case["outcome"],
            status=LocalPaperOrderStatus.REJECTED,
            filled_quantity=Decimal("0"),
            remaining_quantity=Decimal("123"),
            exact_fill_price=None,
            fill_evidence=None,
        )
    record = build_paper_entry_terminal_record(**case)
    with localcontext() as context:
        context.prec = 2
        context.traps[Inexact] = trapped
        context.traps[Rounded] = trapped
        assert read(record, tmp_path / "journal.db") == record
        assert build_paper_entry_terminal_record(**case) == record

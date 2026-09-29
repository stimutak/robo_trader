"""Strict read validation of proposed entry records; never execution authority."""

import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from decimal import Decimal

from .execution import LocalPaperExecutionEvidence
from .paper_entry_settlement import _build_paper_entry_terminal_record
from .paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
)
from .paper_terminal_settlement import (
    PaperAccountSettlementState,
    _parse_protective_quote_timestamp,
)
from .protective_quote_evidence import ProtectiveQuoteSource, _canonical_payload, _text
from .safety.journal import SafetyJournal
from .safety.models import (
    JournalEvent,
    JournalEventType,
    ValidationError,
    _strict_decimal,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError("duplicate entry record field")
        result[key] = value
    return result


def _model_values(value, model):
    if type(value) is not dict or set(value) != {field.name for field in fields(model)}:
        raise ValidationError("stored entry model fields are incomplete or unknown")
    return dict(value)


def _decimals(values, names):
    for name in names:
        if values[name] is not None:
            values[name] = parse_fixed_decimal(values[name], name)
    return values


def _event(value):
    values = _model_values(value, JournalEvent)
    values["event_type"] = JournalEventType(values["event_type"])
    values["occurred_at"] = parse_utc_text(values["occurred_at"])
    return JournalEvent(**values)


@dataclass(frozen=True, slots=True)
class _StoredQuote:
    """Historical structure only; deliberately not ProtectiveQuoteEvidence."""

    portfolio_id: str
    symbol: str
    con_id: int
    price: Decimal
    source_timestamp: datetime
    source: ProtectiveQuoteSource
    transport_generation: str
    quote_id: str
    payload_json: str

    def canonical_payload(self):
        return self.payload_json


def _quote(value, quote_id):
    if type(value) is not dict or set(value) != {
        "portfolio_id",
        "symbol",
        "con_id",
        "price",
        "source_timestamp",
        "source",
        "transport_generation",
        "receipt_monotonic",
        "receipt_order",
        "source_event_id",
    }:
        raise ValidationError("stored entry quote fields are invalid")
    for name in ("portfolio_id", "symbol", "transport_generation", "source_event_id"):
        _text(value[name], name)
    if type(value["price"]) is not str:
        raise ValidationError("stored entry quote price must be exact text")
    price = _strict_decimal(Decimal(value["price"]), "stored quote price", positive=True)
    if type(value["con_id"]) is not int or value["con_id"] <= 0:
        raise ValidationError("stored entry quote contract is invalid")
    if type(value["receipt_order"]) is not int or value["receipt_order"] <= 0:
        raise ValidationError("stored entry quote sequence is invalid")
    if type(value["receipt_monotonic"]) is not str:
        raise ValidationError("stored entry quote clock is invalid")
    monotonic = float.fromhex(value["receipt_monotonic"])
    if not math.isfinite(monotonic) or monotonic < 0:
        raise ValidationError("stored entry quote clock is invalid")
    timestamp = _parse_protective_quote_timestamp(value["source_timestamp"])
    source = ProtectiveQuoteSource(value["source"])
    encoded = _canonical_payload(
        portfolio_id=value["portfolio_id"],
        symbol=value["symbol"],
        con_id=value["con_id"],
        price=price,
        source_timestamp=timestamp,
        source=source,
        transport_generation=value["transport_generation"],
        receipt_monotonic=monotonic,
        receipt_order=value["receipt_order"],
        source_event_id=value["source_event_id"],
    )
    if encoded != canonical_json(value) or quote_id != "quote:v1:" + sha256_text(encoded):
        raise ValidationError("stored entry quote fingerprint is invalid")
    return _StoredQuote(
        value["portfolio_id"],
        value["symbol"],
        value["con_id"],
        price,
        timestamp,
        source,
        value["transport_generation"],
        quote_id,
        encoded,
    )


def validate_stored_entry_record(payload_json, *, fingerprint, journal):
    """Recompute canonical data and match both events to read-only journal replay.

    This does not prove database pre-state, authentic execution, persistence or
    release eligibility. No quote, order permit or receipt is minted on recovery.
    """
    try:
        if type(payload_json) is not str or len(payload_json) > 65536:
            raise ValidationError("stored entry record text is invalid")
        if type(fingerprint) is not str or fingerprint != sha256_text(payload_json):
            raise ValidationError("stored entry record fingerprint differs")
        data = json.loads(payload_json, object_pairs_hook=_unique)
        if type(data) is not dict or canonical_json(data) != payload_json:
            raise ValidationError("stored entry record is not canonical")
        if type(data.get("version")) is not int or data["version"] != 1:
            raise ValidationError("stored entry record version is invalid")
        reservation, claim = _event(data["reservation"]), _event(data["claim"])
        account = PaperAccountSettlementState(
            **_decimals(
                _model_values(data["pre_account"], PaperAccountSettlementState),
                (
                    "cash",
                    "realized_pnl",
                    "daily_pnl",
                    "daily_pnl_baseline",
                    "position_cost_basis",
                    "position_mark_price",
                ),
            )
        )
        outcome = _decimals(
            _model_values(data["outcome"], LocalPaperTerminalOutcome),
            ("requested_quantity", "filled_quantity", "remaining_quantity", "exact_fill_price"),
        )
        outcome["status"] = LocalPaperOrderStatus(outcome["status"])
        outcome["provenance"] = LocalPaperOutcomeProvenance(outcome["provenance"])
        outcome["observed_at"] = parse_utc_text(outcome["observed_at"])
        if outcome["fill_evidence"] is not None:
            evidence = _decimals(
                _model_values(outcome["fill_evidence"], LocalPaperExecutionEvidence),
                ("filled_quantity", "exact_fill_price"),
            )
            evidence["occurred_at"] = parse_utc_text(evidence["occurred_at"])
            outcome["fill_evidence"] = LocalPaperExecutionEvidence(**evidence)
        result = _build_paper_entry_terminal_record(
            reservation=reservation,
            claim=claim,
            account=account,
            outcome=LocalPaperTerminalOutcome(**outcome),
            quote=_quote(data["quote"], data["quote_id"]),
            **{
                name: parse_fixed_decimal(data[name], name)
                for name in (
                    "pre_position_quantity",
                    "pre_aggregate_quantity",
                    "pre_symbol_gross_quantity",
                )
            },
        )
        # Rebuilding also detects unknown top-level fields, incorrect kind and
        # forged/omitted post-values, not just malformed individual inputs.
        if result.payload_json != payload_json:
            raise ValidationError("stored entry record differs from recomputed accounting")
        if type(journal) is not SafetyJournal:
            raise ValidationError("stored entry record requires an exact journal")
        replay = journal.replay(
            expected_execution_domain_scope=claim.execution_domain_scope,
            expected_account_scope=claim.account_scope,
        )
        actual = {event.sequence: event for event in replay.events}
        for event in (reservation, claim):
            found = actual.get(event.sequence)
            if found is None or canonical_json(asdict(found)) != canonical_json(asdict(event)):
                raise ValidationError("stored entry event differs from authenticated journal")
        return result
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError("stored entry record validation failed") from error

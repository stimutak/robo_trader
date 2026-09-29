"""Exact Gate-A opening-BUY accounting; no persistence or order authority.

The eventual atomic settlement writer must authenticate the terminal execution,
prove the position is flat under its write transaction, compare this projection
with FIFO, and persist cash/position/outbox before releasing the reservation.
Constructing these values does none of those things.
"""

import json
from dataclasses import asdict, dataclass
from decimal import Decimal

from .paper_terminal_settlement import (
    PaperAccountSettlementState,
    _exact_decimal_multiply,
    _strict_integral_decimal,
)
from .safety.models import (
    JournalEvent,
    JournalEventType,
    TerminalOrderStatus,
    ValidationError,
    _exact_decimal_add,
    _exact_decimal_subtract,
    _strict_decimal,
    canonical_json,
    parse_fixed_decimal,
    sha256_text,
)


@dataclass(frozen=True, slots=True)
class PaperEntrySettlementValues:
    """Calculated values, never a settlement receipt or authorization token."""

    cash: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    daily_pnl_baseline: Decimal
    daily_pnl_date: str
    position_quantity: Decimal
    position_cost_basis: Decimal | None
    position_mark_price: Decimal | None


@dataclass(frozen=True, slots=True)
class PaperEntryTerminalRecord:
    """Canonical proposed settlement data; not a receipt or replay authority."""

    payload_json: str

    @property
    def fingerprint(self) -> str:
        return sha256_text(self.payload_json)


def validate_stored_paper_entry_terminal_record(payload_json, *, fingerprint, journal):
    """Validate historical data and journal links without minting live evidence."""
    from .paper_entry_record_replay import validate_stored_entry_record

    return validate_stored_entry_record(payload_json, fingerprint=fingerprint, journal=journal)


def build_paper_entry_terminal_record(
    *, quote, quote_producer, **values
) -> PaperEntryTerminalRecord:
    """Build proposed data from the exact live quote owner, never a copied quote."""
    from .protective_quote_evidence import assert_producer_owned_protective_quote

    if quote_producer is None:
        raise ValidationError("entry quote requires its owning producer")
    quote = assert_producer_owned_protective_quote(quote, producer=quote_producer)
    return _build_paper_entry_terminal_record(quote=quote, **values)


def _build_paper_entry_terminal_record(
    *,
    reservation,
    claim,
    account,
    pre_position_quantity,
    pre_aggregate_quantity,
    pre_symbol_gross_quantity,
    quote,
    outcome,
) -> PaperEntryTerminalRecord:
    """Bind one entry attempt to its terminal values before atomic persistence.

    The writer must re-read both events from the authenticated journal and prove
    database pre-state under its transaction. The gateway must have dispatched
    this exact claimed order through its one-shot entry sink. This record alone
    cannot establish either provenance or authorize a write/release.
    """
    from .paper_reduction_submitter import LocalPaperTerminalOutcome
    from .protective_quote_evidence import (
        ProtectiveQuoteSource,
    )
    from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
    from .safety.entry_capacity import validate_entry_capacity_event, validate_entry_claim_event

    for event, kind in (
        (reservation, JournalEventType.ENTRY_CAPACITY_RESERVED),
        (claim, JournalEventType.ENTRY_SUBMISSION_CLAIMED),
    ):
        if type(event) is not JournalEvent or event.event_type is not kind:
            raise ValidationError("entry terminal record requires exact reservation and claim")
        event.__post_init__()
        if event.payload_hash != sha256_text(event.payload_json):
            raise ValidationError("entry event payload hash changed")
    capacity = json.loads(reservation.payload_json)
    claimed = json.loads(claim.payload_json)
    validate_entry_capacity_event(reservation, capacity)
    validate_entry_claim_event(claim, claimed, reservation)
    if reservation.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE:
        raise ValidationError("entry settlement requires the local paper domain")
    if (
        type(account) is not PaperAccountSettlementState
        or account.portfolio_id != claim.portfolio_id
    ):
        raise ValidationError("entry account does not match claimed portfolio")
    if _strict_integral_decimal(pre_aggregate_quantity, "pre_aggregate_quantity") != 0:
        raise ValidationError("entry settlement requires account-wide flat symbol exposure")
    if _strict_integral_decimal(pre_symbol_gross_quantity, "pre_symbol_gross_quantity") != 0:
        raise ValidationError("entry settlement cannot net opposing held symbol positions")
    if (
        quote.source is not ProtectiveQuoteSource.LIVE_BROKER
        or quote.portfolio_id != claim.portfolio_id
        or quote.symbol != capacity["symbol"]
        or quote.con_id != claim.con_id
        or quote.quote_id != capacity["quote_id"]
        or quote.transport_generation != capacity["transport_generation"]
        or quote.source_timestamp > claim.occurred_at
    ):
        raise ValidationError("entry quote does not match reserved market evidence")
    if type(outcome) is not LocalPaperTerminalOutcome:
        raise ValidationError("entry terminal outcome is malformed")
    outcome.__post_init__()
    if (
        outcome.terminal is not True
        or outcome.order_ref != claimed["order_ref"]
        or outcome.requested_quantity != Decimal(capacity["quantity"])
        or outcome.observed_at < claim.occurred_at
    ):
        raise ValidationError("entry terminal outcome does not match its claim")
    evidence = outcome.fill_evidence
    if evidence is not None:
        evidence.__post_init__()
        if _exact_decimal_multiply(
            outcome.filled_quantity, outcome.exact_fill_price, "entry actual notional"
        ) > parse_fixed_decimal(capacity["notional_usd"]):
            raise ValidationError("entry fill exceeds reserved principal")
    values = project_flat_paper_buy(
        account=account,
        pre_position_quantity=pre_position_quantity,
        requested_quantity=outcome.requested_quantity,
        filled_quantity=outcome.filled_quantity,
        fill_price=outcome.exact_fill_price,
        mark_price=quote.price,
        commission_minor=0 if evidence is None else evidence.commission_minor,
        terminal_status=TerminalOrderStatus(outcome.status.value),
    )
    return PaperEntryTerminalRecord(
        canonical_json(
            {
                "kind": "PAPER_ENTRY_TERMINAL_SETTLEMENT",
                "version": 1,
                "reservation": asdict(reservation),
                "claim": asdict(claim),
                "pre_account": asdict(account),
                "pre_position_quantity": pre_position_quantity,
                "pre_aggregate_quantity": pre_aggregate_quantity,
                "pre_symbol_gross_quantity": pre_symbol_gross_quantity,
                "quote_id": quote.quote_id,
                "quote": json.loads(quote.canonical_payload()),
                "outcome": asdict(outcome),
                "post_values": asdict(values),
            }
        )
    )


def project_flat_paper_buy(
    *,
    account: PaperAccountSettlementState,
    pre_position_quantity: Decimal,
    requested_quantity: Decimal,
    filled_quantity: Decimal,
    fill_price: Decimal | None,
    mark_price: Decimal,
    commission_minor: int,
    terminal_status: TerminalOrderStatus,
) -> PaperEntrySettlementValues:
    """Project a full opening BUY or an unfilled terminal outcome, exactly.

    Gate A prohibits adding to held positions, short entries and partial fills.
    Its simulator reports explicit zero commission. Nonzero commissions require
    a separate policy integrating deferred opening fees with FIFO and daily P&L;
    this path refuses them instead of silently changing that accounting policy.
    """
    if type(account) is not PaperAccountSettlementState:
        raise ValidationError("entry settlement requires exact account state")
    account.__post_init__()
    pre = _strict_integral_decimal(pre_position_quantity, "pre_position_quantity")
    # Closed positions retain their prior cost/mark/settlement history. Quantity,
    # verified against FIFO by the writer, establishes flatness; historical
    # metadata must not make every later re-entry impossible.
    cost_missing = account.position_cost_basis is None
    mark_missing = account.position_mark_price is None
    if (
        pre != 0
        or cost_missing != mark_missing
        or (cost_missing and account.position_source_settlement_id is not None)
    ):
        raise ValidationError("entry settlement requires an unambiguous flat position")
    requested = _strict_integral_decimal(requested_quantity, "requested_quantity")
    filled = _strict_integral_decimal(filled_quantity, "filled_quantity")
    if not 0 < requested <= 2_147_483_647 or filled < 0:
        raise ValidationError("entry settlement quantity is outside the simulator range")
    mark = _strict_decimal(mark_price, "entry mark price", positive=True)
    if type(commission_minor) is not int or commission_minor != 0:
        raise ValidationError("Gate-A entry requires explicit zero simulated commission")
    if type(terminal_status) is not TerminalOrderStatus:
        raise ValidationError("entry terminal status is malformed")
    cash, daily = account.cash, account.daily_pnl
    cost_basis, position_mark = account.position_cost_basis, account.position_mark_price
    if terminal_status is TerminalOrderStatus.FILLED:
        if filled != requested:
            raise ValidationError("entry settlement requires a full terminal fill")
        cost_basis = _strict_decimal(fill_price, "entry fill price", positive=True)
        notional = _exact_decimal_multiply(filled, cost_basis, "entry fill notional")
        cash = _exact_decimal_subtract(cash, notional, "entry post cash")
        if cash < 0:
            raise ValidationError("entry fill exceeds available cash")
        unrealized = _exact_decimal_multiply(
            filled,
            _exact_decimal_subtract(mark, cost_basis, "entry price change"),
            "entry unrealized P&L",
        )
        daily = _exact_decimal_add(daily, unrealized, "entry daily P&L")
        position_mark = mark
    elif terminal_status in {
        TerminalOrderStatus.REJECTED,
        TerminalOrderStatus.CANCELLED,
        TerminalOrderStatus.EXPIRED,
    }:
        if filled != 0 or fill_price is not None:
            raise ValidationError("unfilled entry outcome cannot carry execution values")
    else:
        raise ValidationError("entry outcome is not a supported terminal state")
    return PaperEntrySettlementValues(
        cash,
        account.realized_pnl,
        daily,
        account.daily_pnl_baseline,
        account.daily_pnl_date,
        filled,
        cost_basis,
        position_mark,
    )

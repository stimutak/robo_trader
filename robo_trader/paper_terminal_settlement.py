"""Exact, producer-owned records for local paper reduction settlement.

This module contains no database access.  The authoritative trading database
is the only producer allowed to mint :class:`PaperTerminalSettlementReceipt`
instances; consumers can verify that provenance before using a receipt to
release durable safety authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import weakref
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

from robo_trader.safety.models import (
    MODEL_VERSION,
    LocalPaperTerminalEvidence,
    OrderSide,
    TerminalOrderStatus,
    ValidationError,
    _exact_decimal_add,
    _exact_decimal_subtract,
    _strict_account_scope,
    _strict_database_identity,
    _strict_decimal,
    _strict_enum,
    _strict_hash,
    _strict_internal_id,
    _strict_text,
    _strict_utc,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
)

_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_SETTLEMENT_ID_RE = re.compile(r"^pset-[0-9a-f]{32}$")
_LOCAL_PAPER_EXECUTION_ID_RE = re.compile(r"^lpfill-[0-9a-f]{32}$")


class PaperTerminalSettlementError(RuntimeError):
    """A local paper outcome cannot be settled without ambiguity."""


class PaperTerminalSettlementConflict(PaperTerminalSettlementError):
    """An idempotency identity is already bound to different settlement data."""


def _strict_integral_decimal(value: object, field_name: str) -> Decimal:
    exact_value: Decimal = _strict_decimal(value, field_name)
    if exact_value != exact_value.to_integral_value():
        raise ValidationError(f"{field_name} must be an integral share quantity")
    return exact_value


def _signed_fill(side: OrderSide, filled_quantity: Decimal) -> Decimal:
    if side is OrderSide.BUY_TO_COVER or filled_quantity.is_zero():
        return filled_quantity
    return filled_quantity.copy_negate()


def _exact_decimal_multiply(left: Decimal, right: Decimal, field_name: str) -> Decimal:
    """Multiply validated Decimals without depending on ambient context."""

    _strict_decimal(left, f"{field_name} left operand")
    _strict_decimal(right, f"{field_name} right operand")
    with localcontext() as context:
        context.prec = 64
        product = left * right
    return _strict_decimal(product, field_name)


def _parse_protective_quote_timestamp(value: object) -> datetime:
    """Parse the two UTC spellings used by canonical quote evidence.

    Python 3.10's ``datetime.fromisoformat`` does not accept the trailing ``Z``
    form that the durable safety journal uses, while Python 3.11+ does. Normalize
    only that suffix and continue to reject naive and non-UTC timestamps.
    """

    if type(value) is not str or not value:
        raise ValueError("protective quote timestamp must be text")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    offset = parsed.utcoffset()
    if parsed.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise ValueError("protective quote timestamp must be UTC")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class PaperAccountSettlementState:
    """Exact database snapshot used to construct one settlement request."""

    portfolio_id: str
    cash: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    daily_pnl_baseline: Decimal
    daily_pnl_date: str
    position_cost_basis: Optional[Decimal]
    position_mark_price: Optional[Decimal]
    position_source_settlement_id: Optional[str]

    def __post_init__(self) -> None:
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_decimal(self.cash, "cash")
        _strict_decimal(self.realized_pnl, "realized_pnl")
        _strict_decimal(self.daily_pnl, "daily_pnl")
        _strict_decimal(self.daily_pnl_baseline, "daily_pnl_baseline")
        _strict_text(self.daily_pnl_date, "daily_pnl_date", max_length=10)
        try:
            parsed_daily_date = date.fromisoformat(self.daily_pnl_date)
        except ValueError as exc:
            raise ValidationError("daily_pnl_date is not an ISO calendar date") from exc
        if parsed_daily_date.isoformat() != self.daily_pnl_date:
            raise ValidationError("daily_pnl_date is not canonical")
        if self.position_cost_basis is not None:
            _strict_decimal(
                self.position_cost_basis,
                "position_cost_basis",
                positive=True,
            )
        if self.position_mark_price is not None:
            _strict_decimal(
                self.position_mark_price,
                "position_mark_price",
                positive=True,
            )
        if self.position_source_settlement_id is not None and not _SETTLEMENT_ID_RE.fullmatch(
            self.position_source_settlement_id
        ):
            raise ValidationError("position_source_settlement_id is malformed")

    def post_values(
        self,
        *,
        side: OrderSide,
        filled_quantity: Decimal,
        fill_price: Optional[Decimal],
        protective_mark_price: Decimal,
        pre_position_quantity: Decimal,
        commission_minor: int = 0,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        """Return exact cash/P&L after a reduction, without float arithmetic.

        Daily P&L is cumulative realized P&L plus current unrealized P&L,
        less the persisted day-start unrealized baseline.  Reducing exposure
        therefore replaces the removed shares' unrealized P&L at the
        producer-owned protective mark with realized P&L at the fill.
        """

        _strict_enum(side, OrderSide, "side")
        filled = _strict_integral_decimal(filled_quantity, "filled_quantity")
        mark = _strict_decimal(
            protective_mark_price,
            "protective_mark_price",
            positive=True,
        )
        if filled < 0:
            raise ValidationError("filled_quantity must be nonnegative")
        if type(commission_minor) is not int or abs(commission_minor) > 1_000_000_000_000:
            raise ValidationError("commission_minor is outside the exact allowed range")
        commission = Decimal(commission_minor).scaleb(-2)
        if filled.is_zero():
            if fill_price is not None:
                raise ValidationError("unfilled settlement cannot have a fill price")
            if commission_minor != 0:
                raise ValidationError("unfilled settlement cannot have a commission")
            return self.cash, self.realized_pnl, self.daily_pnl
        if (
            fill_price is None
            or self.position_cost_basis is None
            or self.position_mark_price is None
        ):
            raise ValidationError(
                "filled settlement requires fill, cost-basis, and prior-mark prices"
            )
        pre_quantity = _strict_integral_decimal(
            pre_position_quantity,
            "pre_position_quantity",
        )
        if (
            side is OrderSide.SELL
            and pre_quantity <= 0
            or side is OrderSide.BUY_TO_COVER
            and pre_quantity >= 0
        ):
            raise ValidationError("pre_position_quantity has the wrong reduction direction")
        fill = _strict_decimal(fill_price, "fill_price", positive=True)
        notional = _exact_decimal_multiply(fill, filled, "fill notional")
        if side is OrderSide.SELL:
            post_cash = _exact_decimal_subtract(
                _exact_decimal_add(self.cash, notional, "post cash before commission"),
                commission,
                "post cash",
            )
            per_share_pnl = _exact_decimal_subtract(
                fill,
                self.position_cost_basis,
                "long realized P&L per share",
            )
            removed_unrealized_per_share = _exact_decimal_subtract(
                mark,
                self.position_cost_basis,
                "long removed unrealized P&L per share",
            )
            mark_change_per_share = _exact_decimal_subtract(
                mark,
                self.position_mark_price,
                "long mark change per share",
            )
        elif side is OrderSide.BUY_TO_COVER:
            post_cash = _exact_decimal_subtract(
                _exact_decimal_subtract(self.cash, notional, "post cash before commission"),
                commission,
                "post cash",
            )
            per_share_pnl = _exact_decimal_subtract(
                self.position_cost_basis,
                fill,
                "short realized P&L per share",
            )
            removed_unrealized_per_share = _exact_decimal_subtract(
                self.position_cost_basis,
                mark,
                "short removed unrealized P&L per share",
            )
            mark_change_per_share = _exact_decimal_subtract(
                self.position_mark_price,
                mark,
                "short mark change per share",
            )
        else:
            raise ValidationError("paper account settlement side must reduce exposure")
        pnl_delta = _exact_decimal_multiply(
            per_share_pnl,
            filled,
            "realized P&L delta",
        )
        pnl_after_commission = _exact_decimal_subtract(
            pnl_delta,
            commission,
            "realized P&L delta after commission",
        )
        realized_pnl = _exact_decimal_add(
            self.realized_pnl,
            pnl_after_commission,
            "post realized P&L",
        )
        removed_unrealized = _exact_decimal_multiply(
            removed_unrealized_per_share,
            filled,
            "removed unrealized P&L",
        )
        mark_revaluation = _exact_decimal_multiply(
            mark_change_per_share,
            abs(pre_quantity),
            "pre-position mark revaluation",
        )
        daily_pnl = _exact_decimal_subtract(
            _exact_decimal_add(
                _exact_decimal_add(
                    self.daily_pnl,
                    mark_revaluation,
                    "daily P&L after mark revaluation",
                ),
                pnl_after_commission,
                "daily P&L after realized fill",
            ),
            removed_unrealized,
            "post daily P&L",
        )
        return post_cash, realized_pnl, daily_pnl


@dataclass(frozen=True)
class PaperTerminalSettlementRequest:
    """Exact local terminal outcome proposed for one atomic ledger commit."""

    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    reservation_id: str
    claim_id: str
    claim_sequence: int
    submission_descriptor_fingerprint: str
    protective_quote_fingerprint: str
    protective_quote_payload: str
    order_ref: str
    side: OrderSide
    requested_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    expected_pre_position_quantity: Decimal
    expected_post_position_quantity: Decimal
    expected_pre_aggregate_quantity: Decimal
    expected_post_aggregate_quantity: Decimal
    expected_pre_cash: Decimal
    expected_post_cash: Decimal
    expected_pre_realized_pnl: Decimal
    expected_post_realized_pnl: Decimal
    expected_pre_daily_pnl: Decimal
    expected_post_daily_pnl: Decimal
    expected_daily_pnl_baseline: Decimal
    expected_daily_pnl_date: str
    expected_position_cost_basis: Optional[Decimal]
    expected_pre_position_mark_price: Optional[Decimal]
    expected_pre_position_source_settlement_id: Optional[str]
    terminal_status: TerminalOrderStatus
    fill_price: Optional[Decimal]
    outcome_at: datetime
    fill_execution_id: Optional[str] = None
    fill_commission_minor: Optional[int] = None
    fill_commission_currency: Optional[str] = None
    fill_commission_source: Optional[str] = None
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != MODEL_VERSION:
            raise ValidationError(f"schema_version must be exactly {MODEL_VERSION}")
        _strict_text(
            self.execution_domain_scope,
            "execution_domain_scope",
            pattern=_SCOPE_RE,
        )
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        _strict_internal_id(self.claim_id, "claim_id", "claim")
        if type(self.claim_sequence) is not int or self.claim_sequence <= 0:
            raise ValidationError("claim_sequence must be a positive integer")
        _strict_hash(
            self.submission_descriptor_fingerprint,
            "submission_descriptor_fingerprint",
        )
        _strict_hash(self.protective_quote_fingerprint, "protective_quote_fingerprint")
        try:
            quote_payload = json.loads(self.protective_quote_payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError("protective_quote_payload is malformed") from exc
        if (
            not isinstance(quote_payload, dict)
            or json.dumps(
                quote_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            != self.protective_quote_payload
        ):
            raise ValidationError("protective_quote_payload is not canonical")
        if sha256_text(self.protective_quote_payload) != self.protective_quote_fingerprint:
            raise ValidationError("protective quote payload fingerprint does not match")
        required_quote_fields = {
            "con_id",
            "portfolio_id",
            "price",
            "receipt_monotonic",
            "receipt_order",
            "source",
            "source_event_id",
            "source_timestamp",
            "symbol",
            "transport_generation",
        }
        if set(quote_payload) != required_quote_fields:
            raise ValidationError("protective quote payload fields are incomplete")
        if (
            quote_payload["portfolio_id"] != self.portfolio_id
            or quote_payload["symbol"] != self.symbol
            or quote_payload["con_id"] != self.con_id
            or quote_payload["source"] != "live-broker"
        ):
            raise ValidationError("protective quote payload identity does not match settlement")
        try:
            quote_price = Decimal(quote_payload["price"])
            receipt_monotonic = float.fromhex(quote_payload["receipt_monotonic"])
            source_timestamp = _parse_protective_quote_timestamp(quote_payload["source_timestamp"])
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ValidationError("protective quote payload values are malformed") from exc
        _strict_decimal(quote_price, "protective quote price", positive=True)
        if str(quote_price) != quote_payload["price"]:
            raise ValidationError("protective quote price is not canonical")
        if (
            type(receipt_monotonic) is not float
            or not receipt_monotonic >= 0.0
            or not receipt_monotonic < float("inf")
            or type(quote_payload["receipt_order"]) is not int
            or quote_payload["receipt_order"] <= 0
            or type(quote_payload["transport_generation"]) is not str
            or not quote_payload["transport_generation"]
            or source_timestamp.tzinfo is None
            or source_timestamp.utcoffset() is None
        ):
            raise ValidationError("protective quote payload lineage is malformed")
        _strict_text(self.order_ref, "order_ref", max_length=128)
        _strict_enum(self.side, OrderSide, "side")
        if self.side not in {OrderSide.SELL, OrderSide.BUY_TO_COVER}:
            raise ValidationError("paper settlement side must reduce exposure")
        requested = _strict_integral_decimal(self.requested_quantity, "requested_quantity")
        filled = _strict_integral_decimal(self.filled_quantity, "filled_quantity")
        remaining = _strict_integral_decimal(self.remaining_quantity, "remaining_quantity")
        if requested <= 0 or filled < 0 or remaining < 0:
            raise ValidationError("terminal quantities are outside the permitted range")
        if _exact_decimal_add(filled, remaining, "terminal quantity") != requested:
            raise ValidationError("filled and remaining quantities must equal requested quantity")
        pre_position = _strict_integral_decimal(
            self.expected_pre_position_quantity,
            "expected_pre_position_quantity",
        )
        post_position = _strict_integral_decimal(
            self.expected_post_position_quantity,
            "expected_post_position_quantity",
        )
        pre_aggregate = _strict_integral_decimal(
            self.expected_pre_aggregate_quantity,
            "expected_pre_aggregate_quantity",
        )
        post_aggregate = _strict_integral_decimal(
            self.expected_post_aggregate_quantity,
            "expected_post_aggregate_quantity",
        )
        signed_fill = _signed_fill(self.side, filled)
        if _exact_decimal_add(pre_position, signed_fill, "post position") != post_position:
            raise ValidationError("post position does not match the exact terminal fill")
        if _exact_decimal_add(pre_aggregate, signed_fill, "post aggregate") != post_aggregate:
            raise ValidationError("post aggregate does not match the exact terminal fill")
        if self.side is OrderSide.SELL and (pre_position <= 0 or post_position < 0):
            raise ValidationError("SELL settlement would not be a long-position reduction")
        if self.side is OrderSide.BUY_TO_COVER and (pre_position >= 0 or post_position > 0):
            raise ValidationError("BUY_TO_COVER settlement would not be a short reduction")
        account_state = PaperAccountSettlementState(
            portfolio_id=self.portfolio_id,
            cash=self.expected_pre_cash,
            realized_pnl=self.expected_pre_realized_pnl,
            daily_pnl=self.expected_pre_daily_pnl,
            daily_pnl_baseline=self.expected_daily_pnl_baseline,
            daily_pnl_date=self.expected_daily_pnl_date,
            position_cost_basis=self.expected_position_cost_basis,
            position_mark_price=self.expected_pre_position_mark_price,
            position_source_settlement_id=(self.expected_pre_position_source_settlement_id),
        )
        (
            expected_post_cash,
            expected_post_realized_pnl,
            expected_post_daily_pnl,
        ) = account_state.post_values(
            side=self.side,
            filled_quantity=filled,
            fill_price=self.fill_price,
            protective_mark_price=quote_price,
            pre_position_quantity=pre_position,
            commission_minor=(
                0 if self.fill_commission_minor is None else self.fill_commission_minor
            ),
        )
        _strict_decimal(self.expected_post_cash, "expected_post_cash")
        _strict_decimal(
            self.expected_post_realized_pnl,
            "expected_post_realized_pnl",
        )
        _strict_decimal(self.expected_post_daily_pnl, "expected_post_daily_pnl")
        _strict_decimal(
            self.expected_daily_pnl_baseline,
            "expected_daily_pnl_baseline",
        )
        if self.expected_post_cash != expected_post_cash:
            raise ValidationError("post cash does not match the exact terminal fill")
        if self.expected_post_realized_pnl != expected_post_realized_pnl:
            raise ValidationError("post realized P&L does not match the exact cost basis")
        if self.expected_post_daily_pnl != expected_post_daily_pnl:
            raise ValidationError(
                "post daily P&L does not match the exact mark-to-fill exposure replacement"
            )
        _strict_enum(self.terminal_status, TerminalOrderStatus, "terminal_status")
        if self.terminal_status is TerminalOrderStatus.FILLED:
            if filled != requested or remaining != 0:
                raise ValidationError("FILLED settlement must contain the complete fill")
            if self.fill_price is None:
                raise ValidationError("FILLED settlement requires an exact fill price")
            _strict_decimal(self.fill_price, "fill_price", positive=True)
            if (
                type(self.fill_execution_id) is not str
                or _LOCAL_PAPER_EXECUTION_ID_RE.fullmatch(self.fill_execution_id) is None
            ):
                raise ValidationError("fill_execution_id is malformed")
            if (
                type(self.fill_commission_minor) is not int
                or abs(self.fill_commission_minor) > 1_000_000_000_000
            ):
                raise ValidationError("FILLED settlement requires exact commission minor units")
            if self.fill_commission_currency != "USD":
                raise ValidationError("FILLED settlement commission currency must be USD")
            if self.fill_commission_source != "LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1":
                raise ValidationError("FILLED settlement commission source is not authoritative")
        elif self.terminal_status in {
            TerminalOrderStatus.CANCELLED,
            TerminalOrderStatus.REJECTED,
            TerminalOrderStatus.EXPIRED,
            TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
        }:
            if filled != 0 or remaining != requested:
                raise ValidationError("unfilled terminal settlement has inconsistent quantities")
            if self.fill_price is not None:
                raise ValidationError("unfilled terminal settlement cannot contain a fill price")
            if any(
                value is not None
                for value in (
                    self.fill_execution_id,
                    self.fill_commission_minor,
                    self.fill_commission_currency,
                    self.fill_commission_source,
                )
            ):
                raise ValidationError("unfilled terminal settlement cannot contain fill evidence")
        else:  # pragma: no cover - enum exhaustiveness guard
            raise ValidationError("unsupported paper terminal status")
        object.__setattr__(self, "outcome_at", _strict_utc(self.outcome_at, "outcome_at"))

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "claim_id": self.claim_id,
                "claim_sequence": self.claim_sequence,
                "con_id": self.con_id,
                "execution_domain_scope": self.execution_domain_scope,
                "expected_post_aggregate_quantity": self.expected_post_aggregate_quantity,
                "expected_post_cash": self.expected_post_cash,
                "expected_post_daily_pnl": self.expected_post_daily_pnl,
                "expected_post_position_quantity": self.expected_post_position_quantity,
                "expected_post_realized_pnl": self.expected_post_realized_pnl,
                "expected_pre_aggregate_quantity": self.expected_pre_aggregate_quantity,
                "expected_pre_cash": self.expected_pre_cash,
                "expected_pre_daily_pnl": self.expected_pre_daily_pnl,
                "expected_daily_pnl_baseline": self.expected_daily_pnl_baseline,
                "expected_daily_pnl_date": self.expected_daily_pnl_date,
                "expected_pre_position_quantity": self.expected_pre_position_quantity,
                "expected_pre_realized_pnl": self.expected_pre_realized_pnl,
                "expected_position_cost_basis": self.expected_position_cost_basis,
                "expected_pre_position_mark_price": self.expected_pre_position_mark_price,
                "expected_pre_position_source_settlement_id": (
                    self.expected_pre_position_source_settlement_id
                ),
                "fill_price": self.fill_price,
                "fill_execution_id": self.fill_execution_id,
                "fill_commission_minor": self.fill_commission_minor,
                "fill_commission_currency": self.fill_commission_currency,
                "fill_commission_source": self.fill_commission_source,
                "filled_quantity": self.filled_quantity,
                "order_ref": self.order_ref,
                "outcome_at": self.outcome_at,
                "portfolio_id": self.portfolio_id,
                "protective_quote_fingerprint": self.protective_quote_fingerprint,
                "protective_quote_payload": self.protective_quote_payload,
                "remaining_quantity": self.remaining_quantity,
                "requested_quantity": self.requested_quantity,
                "reservation_id": self.reservation_id,
                "schema_version": self.schema_version,
                "side": self.side,
                "submission_descriptor_fingerprint": (self.submission_descriptor_fingerprint),
                "symbol": self.symbol,
                "terminal_status": self.terminal_status,
            }
        )

    def fingerprint(self) -> str:
        return sha256_text(self.canonical_payload())

    @property
    def protective_mark_price(self) -> Decimal:
        """Return the exact authenticated mark embedded in quote evidence."""

        payload = json.loads(self.protective_quote_payload)
        return parse_fixed_decimal(payload["price"], "protective quote price")

    @property
    def protective_mark_timestamp(self) -> datetime:
        """Return the authenticated broker-event timestamp for the mark."""

        payload = json.loads(self.protective_quote_payload)
        try:
            return _parse_protective_quote_timestamp(payload["source_timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("protective quote timestamp is malformed") from exc

    @classmethod
    def from_canonical_payload(cls, payload_json: str) -> "PaperTerminalSettlementRequest":
        try:
            payload = json.loads(payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PaperTerminalSettlementError("stored settlement request is malformed") from exc
        if not isinstance(payload, dict) or canonical_json(payload) != payload_json:
            raise PaperTerminalSettlementError("stored settlement request is not canonical")
        try:
            return cls(
                execution_domain_scope=payload["execution_domain_scope"],
                account_scope=payload["account_scope"],
                portfolio_id=payload["portfolio_id"],
                con_id=payload["con_id"],
                symbol=payload["symbol"],
                reservation_id=payload["reservation_id"],
                claim_id=payload["claim_id"],
                claim_sequence=payload["claim_sequence"],
                submission_descriptor_fingerprint=payload["submission_descriptor_fingerprint"],
                protective_quote_fingerprint=payload["protective_quote_fingerprint"],
                protective_quote_payload=payload["protective_quote_payload"],
                order_ref=payload["order_ref"],
                side=OrderSide(payload["side"]),
                requested_quantity=parse_fixed_decimal(
                    payload["requested_quantity"], "requested_quantity"
                ),
                filled_quantity=parse_fixed_decimal(payload["filled_quantity"], "filled_quantity"),
                remaining_quantity=parse_fixed_decimal(
                    payload["remaining_quantity"], "remaining_quantity"
                ),
                expected_pre_position_quantity=parse_fixed_decimal(
                    payload["expected_pre_position_quantity"],
                    "expected_pre_position_quantity",
                ),
                expected_post_position_quantity=parse_fixed_decimal(
                    payload["expected_post_position_quantity"],
                    "expected_post_position_quantity",
                ),
                expected_pre_aggregate_quantity=parse_fixed_decimal(
                    payload["expected_pre_aggregate_quantity"],
                    "expected_pre_aggregate_quantity",
                ),
                expected_post_aggregate_quantity=parse_fixed_decimal(
                    payload["expected_post_aggregate_quantity"],
                    "expected_post_aggregate_quantity",
                ),
                expected_pre_cash=parse_fixed_decimal(
                    payload["expected_pre_cash"], "expected_pre_cash"
                ),
                expected_post_cash=parse_fixed_decimal(
                    payload["expected_post_cash"], "expected_post_cash"
                ),
                expected_pre_daily_pnl=parse_fixed_decimal(
                    payload["expected_pre_daily_pnl"],
                    "expected_pre_daily_pnl",
                ),
                expected_post_daily_pnl=parse_fixed_decimal(
                    payload["expected_post_daily_pnl"],
                    "expected_post_daily_pnl",
                ),
                expected_daily_pnl_baseline=parse_fixed_decimal(
                    payload["expected_daily_pnl_baseline"],
                    "expected_daily_pnl_baseline",
                ),
                expected_daily_pnl_date=payload["expected_daily_pnl_date"],
                expected_pre_realized_pnl=parse_fixed_decimal(
                    payload["expected_pre_realized_pnl"],
                    "expected_pre_realized_pnl",
                ),
                expected_post_realized_pnl=parse_fixed_decimal(
                    payload["expected_post_realized_pnl"],
                    "expected_post_realized_pnl",
                ),
                expected_position_cost_basis=(
                    None
                    if payload["expected_position_cost_basis"] is None
                    else parse_fixed_decimal(
                        payload["expected_position_cost_basis"],
                        "expected_position_cost_basis",
                    )
                ),
                expected_pre_position_mark_price=(
                    None
                    if payload["expected_pre_position_mark_price"] is None
                    else parse_fixed_decimal(
                        payload["expected_pre_position_mark_price"],
                        "expected_pre_position_mark_price",
                    )
                ),
                expected_pre_position_source_settlement_id=payload[
                    "expected_pre_position_source_settlement_id"
                ],
                terminal_status=TerminalOrderStatus(payload["terminal_status"]),
                fill_price=(
                    None
                    if payload["fill_price"] is None
                    else parse_fixed_decimal(payload["fill_price"], "fill_price")
                ),
                outcome_at=parse_utc_text(payload["outcome_at"], "outcome_at"),
                fill_execution_id=payload["fill_execution_id"],
                fill_commission_minor=payload["fill_commission_minor"],
                fill_commission_currency=payload["fill_commission_currency"],
                fill_commission_source=payload["fill_commission_source"],
                schema_version=payload["schema_version"],
            )
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            raise PaperTerminalSettlementError("stored settlement request is invalid") from exc


_RECEIPT_PRODUCER_MARKER = object()
_RECEIPT_REGISTRY_LOCK = threading.Lock()
_RECEIPT_REGISTRY: Dict[
    int, Tuple["weakref.ReferenceType[PaperTerminalSettlementReceipt]", str]
] = {}


@dataclass(frozen=True)
class PaperTerminalSettlementReceipt:
    """Database-produced proof of one committed local terminal settlement."""

    settlement_id: str
    request: PaperTerminalSettlementRequest
    trade_id: Optional[int]
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    committed_at: datetime
    schema_version: int = MODEL_VERSION
    _producer_marker: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._producer_marker is not _RECEIPT_PRODUCER_MARKER:
            raise PaperTerminalSettlementError(
                "paper settlement receipt was not created by the trusted ledger producer"
            )
        if not isinstance(self.settlement_id, str) or not _SETTLEMENT_ID_RE.fullmatch(
            self.settlement_id
        ):
            raise PaperTerminalSettlementError("settlement_id is malformed")
        if type(self.request) is not PaperTerminalSettlementRequest:
            raise PaperTerminalSettlementError("receipt request has an invalid type")
        if self.trade_id is not None and (type(self.trade_id) is not int or self.trade_id <= 0):
            raise PaperTerminalSettlementError("trade_id must be positive when present")
        if self.request.filled_quantity > 0 and self.trade_id is None:
            raise PaperTerminalSettlementError("filled settlement must identify its trade row")
        if self.request.filled_quantity == 0 and self.trade_id is not None:
            raise PaperTerminalSettlementError("unfilled settlement cannot identify a trade row")
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise PaperTerminalSettlementError("database_path must be absolute")
        _strict_database_identity(self.database_identity)
        for field_name in ("database_device", "database_inode"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise PaperTerminalSettlementError(f"{field_name} is malformed")
        object.__setattr__(
            self,
            "committed_at",
            _strict_utc(self.committed_at, "committed_at"),
        )
        if self.committed_at < self.request.outcome_at:
            raise PaperTerminalSettlementError("settlement predates its terminal outcome")
        if type(self.schema_version) is not int or self.schema_version != MODEL_VERSION:
            raise PaperTerminalSettlementError("receipt schema version is unsupported")

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "committed_at": self.committed_at,
                "database_device": self.database_device,
                "database_identity": self.database_identity,
                "database_inode": self.database_inode,
                "database_path": self.database_path,
                "request_fingerprint": self.request.fingerprint(),
                "schema_version": self.schema_version,
                "settlement_id": self.settlement_id,
                "trade_id": self.trade_id,
            }
        )

    def fingerprint(self) -> str:
        return sha256_text(self.canonical_payload())

    @property
    def pre_cash(self) -> Decimal:
        return self.request.expected_pre_cash

    @property
    def post_cash(self) -> Decimal:
        return self.request.expected_post_cash

    @property
    def pre_realized_pnl(self) -> Decimal:
        return self.request.expected_pre_realized_pnl

    @property
    def post_realized_pnl(self) -> Decimal:
        return self.request.expected_post_realized_pnl

    @property
    def pre_daily_pnl(self) -> Decimal:
        return self.request.expected_pre_daily_pnl

    @property
    def post_daily_pnl(self) -> Decimal:
        return self.request.expected_post_daily_pnl

    @property
    def daily_pnl_baseline(self) -> Decimal:
        return self.request.expected_daily_pnl_baseline

    @property
    def daily_pnl_date(self) -> str:
        return self.request.expected_daily_pnl_date

    def to_safety_evidence(self) -> LocalPaperTerminalEvidence:
        request = self.request
        return LocalPaperTerminalEvidence(
            execution_domain_scope=request.execution_domain_scope,
            account_scope=request.account_scope,
            portfolio_id=request.portfolio_id,
            con_id=request.con_id,
            symbol=request.symbol,
            reservation_id=request.reservation_id,
            claim_id=request.claim_id,
            claim_sequence=request.claim_sequence,
            submission_descriptor_fingerprint=request.submission_descriptor_fingerprint,
            protective_quote_fingerprint=request.protective_quote_fingerprint,
            order_ref=request.order_ref,
            settlement_id=self.settlement_id,
            settlement_request_fingerprint=request.fingerprint(),
            settlement_receipt_fingerprint=self.fingerprint(),
            database_path=self.database_path,
            database_identity=self.database_identity,
            database_device=self.database_device,
            database_inode=self.database_inode,
            committed_at=self.committed_at,
            terminal_status=request.terminal_status,
            filled_quantity=request.filled_quantity,
            remaining_quantity=request.remaining_quantity,
            pre_position_quantity=request.expected_pre_position_quantity,
            final_position_quantity=request.expected_post_position_quantity,
            pre_aggregate_quantity=request.expected_pre_aggregate_quantity,
            final_aggregate_quantity=request.expected_post_aggregate_quantity,
            source="LOCAL_PAPER_SETTLEMENT_LEDGER",
        )


def _receipt_digest(receipt: PaperTerminalSettlementReceipt) -> str:
    return hashlib.sha256(receipt.canonical_payload().encode("utf-8")).hexdigest()


def _produce_paper_terminal_settlement_receipt(
    *,
    settlement_id: str,
    request: PaperTerminalSettlementRequest,
    trade_id: Optional[int],
    database_path: str,
    database_identity: str,
    database_device: int,
    database_inode: int,
    committed_at: datetime,
    schema_version: int = MODEL_VERSION,
) -> PaperTerminalSettlementReceipt:
    """Mint a receipt for use by the authoritative database implementation only."""

    receipt = PaperTerminalSettlementReceipt(
        settlement_id=settlement_id,
        request=request,
        trade_id=trade_id,
        database_path=database_path,
        database_identity=database_identity,
        database_device=database_device,
        database_inode=database_inode,
        committed_at=committed_at,
        schema_version=schema_version,
        _producer_marker=_RECEIPT_PRODUCER_MARKER,
    )
    receipt_id = id(receipt)

    def discard(reference: "weakref.ReferenceType[PaperTerminalSettlementReceipt]") -> None:
        with _RECEIPT_REGISTRY_LOCK:
            registered = _RECEIPT_REGISTRY.get(receipt_id)
            if registered is not None and registered[0] is reference:
                _RECEIPT_REGISTRY.pop(receipt_id, None)

    reference = weakref.ref(receipt, discard)
    with _RECEIPT_REGISTRY_LOCK:
        _RECEIPT_REGISTRY[receipt_id] = (reference, _receipt_digest(receipt))
    return receipt


def assert_producer_owned_paper_terminal_settlement_receipt(
    receipt: PaperTerminalSettlementReceipt,
) -> None:
    """Reject reconstructed, copied, altered, or expired receipt identities."""

    if type(receipt) is not PaperTerminalSettlementReceipt:
        raise PaperTerminalSettlementError("receipt must be the exact trusted producer type")
    with _RECEIPT_REGISTRY_LOCK:
        registered = _RECEIPT_REGISTRY.get(id(receipt))
    if (
        registered is None
        or registered[0]() is not receipt
        or not hmac.compare_digest(registered[1], _receipt_digest(receipt))
    ):
        raise PaperTerminalSettlementError(
            "paper settlement receipt is not registered producer-owned evidence"
        )

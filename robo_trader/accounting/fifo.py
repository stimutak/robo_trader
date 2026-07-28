"""Deterministic, exact FIFO projection for the dormant PR4 ledger.

The projector records one complete fill, its commission, derived lot openings
and matches, and a chained position snapshot in one SQLite transaction.  It
contains no runtime wiring and cannot migrate a production database.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
from typing import Optional, Sequence

from .fifo_fixture_migration import _assert_no_temp_fifo_objects, assert_fifo_accounting_schema

_ID_PATTERNS = {
    "epoch_id": re.compile(r"^fepoch-[0-9a-f]{32}$"),
    "fill_id": re.compile(r"^ffill-[0-9a-f]{32}$"),
    "commission_id": re.compile(r"^fcomm-[0-9a-f]{32}$"),
}
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9._-]{0,31}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")
_MAX_INPUT_DECIMAL_DIGITS = 38
_MAX_INPUT_DECIMAL_SCALE = 18
_MAX_DERIVED_DECIMAL_DIGITS = 76
_MAX_DERIVED_DECIMAL_SCALE = 36
_MAX_COMMISSION_MINOR = 1_000_000_000_000


class FifoAccountingError(RuntimeError):
    """Base class for exact FIFO accounting failures."""


class FifoAccountingValidationError(FifoAccountingError):
    """Input or persisted exact accounting data is malformed."""


class FifoAccountingConflict(FifoAccountingError):
    """An idempotency identity is bound to different fill data."""


class FifoAccountingOrderingError(FifoAccountingError):
    """A fill does not extend the epoch's deterministic event order."""


class FillSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


def _text(value: object, field: str, pattern: re.Pattern[str] = _SCOPE) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise FifoAccountingValidationError(f"{field} is malformed")
    return value


def _identifier(value: object, field: str) -> str:
    return _text(value, field, _ID_PATTERNS[field])


def _utc(value: object, field: str) -> datetime:
    if type(value) is not datetime:
        raise FifoAccountingValidationError(f"{field} must be a datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise FifoAccountingValidationError(f"{field} must be timezone-aware UTC")
    if offset.total_seconds() != 0:
        raise FifoAccountingValidationError(f"{field} must be UTC")
    return value.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    return _utc(value, "timestamp").isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: object, field: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise FifoAccountingValidationError(f"{field} is not canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise FifoAccountingValidationError(f"{field} is not a UTC timestamp") from exc
    if _utc_text(parsed) != value:
        raise FifoAccountingValidationError(f"{field} is not canonical UTC text")
    return parsed


def _decimal(value: object, field: str, *, positive: bool = False) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise FifoAccountingValidationError(f"{field} must be a finite Decimal")
    sign, digits, decimal_exponent = value.as_tuple()
    del sign
    exponent = int(decimal_exponent)
    if len(digits) > _MAX_DERIVED_DECIMAL_DIGITS or exponent < -_MAX_DERIVED_DECIMAL_SCALE:
        raise FifoAccountingValidationError(f"{field} exceeds exact-ledger precision")
    if positive and value <= 0:
        raise FifoAccountingValidationError(f"{field} must be positive")
    return value


def _input_decimal(value: object, field: str, *, positive: bool = False) -> Decimal:
    exact = _decimal(value, field, positive=positive)
    _, coefficient, decimal_exponent = exact.as_tuple()
    exponent = int(decimal_exponent)
    significant_digits = list(coefficient)
    while len(significant_digits) > 1 and significant_digits[-1] == 0:
        significant_digits.pop()
        exponent += 1
    integer_digits = max(len(significant_digits) + exponent, 0)
    fractional_digits = max(-exponent, 0)
    expanded_digits = integer_digits + fractional_digits
    if expanded_digits > _MAX_INPUT_DECIMAL_DIGITS or exponent < -_MAX_INPUT_DECIMAL_SCALE:
        raise FifoAccountingValidationError(f"{field} exceeds input precision")
    return exact


def _decimal_text(value: Decimal) -> str:
    _decimal(value, "decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _parse_decimal(value: object, field: str, *, positive: bool = False) -> Decimal:
    if type(value) is not str or not value or "e" in value.lower() or value.startswith("+"):
        raise FifoAccountingValidationError(f"{field} is not canonical decimal text")
    if re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", value) is None:
        raise FifoAccountingValidationError(f"{field} is not canonical decimal text")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise FifoAccountingValidationError(f"{field} is not decimal text") from exc
    _decimal(parsed, field, positive=positive)
    if _decimal_text(parsed) != value:
        raise FifoAccountingValidationError(f"{field} is not canonical decimal text")
    return parsed


def _add(left: Decimal, right: Decimal, field: str) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        result = left + right
    return _decimal(result, field)


def _subtract(left: Decimal, right: Decimal, field: str) -> Decimal:
    return _add(left, right.copy_negate(), field)


def _multiply(left: Decimal, right: Decimal, field: str) -> Decimal:
    with localcontext() as context:
        context.prec = 96
        result = left * right
    return _decimal(result, field)


def _money_from_minor(value: int) -> Decimal:
    if type(value) is not int or abs(value) > _MAX_COMMISSION_MINOR:
        raise FifoAccountingValidationError("commission_minor is outside the allowed range")
    magnitude = str(abs(value))
    return Decimal((1 if value < 0 else 0, tuple(int(digit) for digit in magnitude), -2))


def _stored_int(value: object, field: str) -> int:
    """Read an SQLite INTEGER without silently truncating REAL or TEXT storage."""

    if type(value) is not int:
        raise FifoAccountingValidationError(f"{field} is not stored as an exact integer")
    return value


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _derived_id(prefix: str, *parts: object) -> str:
    value = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:32]}"


def _quantity_atoms(values: Sequence[Decimal]) -> list[int]:
    if not values:
        return []
    scale = max(max(-int(value.as_tuple().exponent), 0) for value in values)
    atoms: list[int] = []
    with localcontext() as context:
        context.prec = 96
        factor = Decimal(10) ** scale
        for value in values:
            exact = value * factor
            if exact != exact.to_integral_value():
                raise FifoAccountingValidationError(
                    "quantity could not be converted to exact atoms"
                )
            atoms.append(int(exact))
    return atoms


def _allocate_minor(total_minor: int, quantities: Sequence[Decimal]) -> list[int]:
    """Allocate every minor unit exactly using deterministic largest remainder."""

    _money_from_minor(total_minor)
    if not quantities:
        if total_minor != 0:
            raise FifoAccountingValidationError("commission has no fill segment")
        return []
    for quantity in quantities:
        _decimal(quantity, "allocation quantity", positive=True)
    weights = _quantity_atoms(quantities)
    total_weight = sum(weights)
    magnitude = abs(total_minor)
    quotients = [magnitude * weight // total_weight for weight in weights]
    remainders = [magnitude * weight % total_weight for weight in weights]
    residual = magnitude - sum(quotients)
    order = sorted(range(len(weights)), key=lambda index: (-remainders[index], index))
    for index in order[:residual]:
        quotients[index] += 1
    sign = -1 if total_minor < 0 else 1
    allocations = [sign * value for value in quotients]
    if sum(allocations) != total_minor:
        raise AssertionError("commission allocation did not conserve minor units")
    return allocations


@dataclass(frozen=True, slots=True)
class AccountingEpoch:
    epoch_id: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    source_fingerprint: str
    effective_at: datetime
    created_at: datetime
    origin_kind: str = "EMPTY_LEDGER"
    schema_version: int = 1

    def __post_init__(self) -> None:
        _identifier(self.epoch_id, "epoch_id")
        _text(self.execution_domain_scope, "execution_domain_scope")
        _text(self.account_scope, "account_scope")
        _text(self.portfolio_id, "portfolio_id")
        if self.origin_kind != "EMPTY_LEDGER":
            raise FifoAccountingValidationError(
                "PR4A can create only EMPTY_LEDGER epochs; legacy adoption belongs to PR4B"
            )
        if (
            type(self.source_fingerprint) is not str
            or _HASH.fullmatch(self.source_fingerprint) is None
        ):
            raise FifoAccountingValidationError("source_fingerprint is malformed")
        _utc(self.effective_at, "effective_at")
        _utc(self.created_at, "created_at")
        if self.created_at < self.effective_at:
            raise FifoAccountingValidationError("created_at precedes effective_at")
        if self.schema_version != 1:
            raise FifoAccountingValidationError("unsupported epoch schema version")

    def payload(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "created_at": _utc_text(self.created_at),
            "effective_at": _utc_text(self.effective_at),
            "epoch_id": self.epoch_id,
            "execution_domain_scope": self.execution_domain_scope,
            "origin_kind": self.origin_kind,
            "portfolio_id": self.portfolio_id,
            "schema_version": self.schema_version,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class FillEvent:
    epoch_id: str
    fill_id: str
    commission_id: str
    event_sequence: int
    execution_id: str
    idempotency_key: str
    con_id: int
    symbol: str
    side: FillSide
    quantity: Decimal
    price: Decimal
    commission_minor: int
    occurred_at: datetime
    recorded_at: datetime

    def __post_init__(self) -> None:
        _identifier(self.epoch_id, "epoch_id")
        _identifier(self.fill_id, "fill_id")
        _identifier(self.commission_id, "commission_id")
        if type(self.event_sequence) is not int or self.event_sequence <= 0:
            raise FifoAccountingValidationError("event_sequence must be a positive integer")
        _text(self.execution_id, "execution_id")
        _text(self.idempotency_key, "idempotency_key")
        if type(self.con_id) is not int or self.con_id <= 0:
            raise FifoAccountingValidationError("con_id must be a positive integer")
        _text(self.symbol, "symbol", _SYMBOL)
        if type(self.side) is not FillSide:
            raise FifoAccountingValidationError("side must be FillSide")
        _input_decimal(self.quantity, "quantity", positive=True)
        _input_decimal(self.price, "price", positive=True)
        _money_from_minor(self.commission_minor)
        _utc(self.occurred_at, "occurred_at")
        _utc(self.recorded_at, "recorded_at")
        if self.recorded_at < self.occurred_at:
            raise FifoAccountingValidationError("recorded_at precedes occurred_at")

    def payload(self) -> dict[str, object]:
        return {
            "commission_id": self.commission_id,
            "commission_minor": self.commission_minor,
            "con_id": self.con_id,
            "epoch_id": self.epoch_id,
            "event_sequence": self.event_sequence,
            "execution_id": self.execution_id,
            "fill_id": self.fill_id,
            "idempotency_key": self.idempotency_key,
            "occurred_at": _utc_text(self.occurred_at),
            "price": _decimal_text(self.price),
            "quantity": _decimal_text(self.quantity),
            "recorded_at": _utc_text(self.recorded_at),
            "side": self.side.value,
            "symbol": self.symbol,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.payload())


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    snapshot_id: str
    epoch_id: str
    source_fill_id: str
    event_sequence: int
    con_id: int
    symbol: str
    signed_quantity: Decimal
    open_cost: Optional[Decimal]
    open_lot_count: int
    cumulative_realized_pnl: Decimal
    cumulative_commission_minor: int
    previous_snapshot_id: Optional[str]
    previous_state_fingerprint: Optional[str]
    state_fingerprint: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class FillResult:
    fill_id: str
    opened_lot_id: Optional[str]
    match_ids: tuple[str, ...]
    snapshot: PositionSnapshot
    replayed: bool


@dataclass(frozen=True, slots=True)
class _OpenLot:
    lot_id: str
    opened_sequence: int
    direction: str
    opened_quantity: Decimal
    remaining_quantity: Decimal
    open_price: Decimal
    opening_commission_minor: int
    allocated_opening_commission_minor: int


@dataclass(slots=True)
class _ProjectionLot:
    lot_id: str
    direction: str
    remaining_quantity: Decimal
    open_price: Decimal


class FifoLedger:
    """Append exact fill events to a previously verified FIFO ledger."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        allow_other_objects: bool = False,
    ) -> None:
        if type(connection) is not sqlite3.Connection:
            raise TypeError("connection must be sqlite3.Connection")
        assert_fifo_accounting_schema(
            connection,
            allow_other_objects=allow_other_objects,
        )
        self._connection = connection

    def create_epoch(self, epoch: AccountingEpoch) -> AccountingEpoch:
        _assert_no_temp_fifo_objects(self._connection)
        if type(epoch) is not AccountingEpoch:
            raise TypeError("epoch must be AccountingEpoch")
        if self._connection.in_transaction:
            raise FifoAccountingError("epoch creation requires an idle connection")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            by_id = self._connection.execute(
                "SELECT * FROM fifo_accounting_epochs WHERE epoch_id = ?",
                (epoch.epoch_id,),
            ).fetchone()
            by_scope = self._connection.execute(
                """
                SELECT * FROM fifo_accounting_epochs
                WHERE execution_domain_scope = ? AND account_scope = ? AND portfolio_id = ?
                """,
                (epoch.execution_domain_scope, epoch.account_scope, epoch.portfolio_id),
            ).fetchone()
            by_fingerprint = self._connection.execute(
                "SELECT * FROM fifo_accounting_epochs WHERE source_fingerprint = ?",
                (epoch.source_fingerprint,),
            ).fetchone()
            existing = [row for row in (by_id, by_scope, by_fingerprint) if row is not None]
            if existing:
                if any(tuple(row) != tuple(existing[0]) for row in existing[1:]):
                    raise FifoAccountingConflict("epoch identities resolve to different records")
                if self._epoch_payload_from_row(existing[0]) != epoch.payload():
                    raise FifoAccountingConflict("epoch identity is bound to different data")
                self._connection.rollback()
                return epoch
            self._connection.execute(
                """
                INSERT INTO fifo_accounting_epochs(
                    epoch_id, schema_version, execution_domain_scope, account_scope,
                    portfolio_id, origin_kind, source_fingerprint, effective_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    epoch.epoch_id,
                    epoch.schema_version,
                    epoch.execution_domain_scope,
                    epoch.account_scope,
                    epoch.portfolio_id,
                    epoch.origin_kind,
                    epoch.source_fingerprint,
                    _utc_text(epoch.effective_at),
                    _utc_text(epoch.created_at),
                ),
            )
            self._connection.commit()
        except BaseException:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise
        return epoch

    @staticmethod
    def _epoch_payload_from_row(row: Sequence[object]) -> dict[str, object]:
        return {
            "account_scope": row[3],
            "created_at": row[8],
            "effective_at": row[7],
            "epoch_id": row[0],
            "execution_domain_scope": row[2],
            "origin_kind": row[5],
            "portfolio_id": row[4],
            "schema_version": row[1],
            "source_fingerprint": row[6],
        }

    def record_fill(self, event: FillEvent) -> FillResult:
        _assert_no_temp_fifo_objects(self._connection)
        if type(event) is not FillEvent:
            raise TypeError("event must be FillEvent")
        if self._connection.in_transaction:
            raise FifoAccountingError("fill recording requires an idle connection")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            effective_at = self._require_epoch(event.epoch_id)
            if event.occurred_at < effective_at:
                raise FifoAccountingOrderingError("fill event time precedes epoch effective_at")
            replay = self._existing_fill(event)
            if replay is not None:
                self._connection.rollback()
                return replay
            self._assert_next_event(event)
            self._assert_instrument_identity(event)
            self._connection.execute(
                """
                INSERT INTO fifo_fills(
                    fill_id, epoch_id, event_sequence, execution_id, idempotency_key,
                    con_id, symbol, side, quantity_text, price_text, occurred_at,
                    recorded_at, payload_fingerprint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.fill_id,
                    event.epoch_id,
                    event.event_sequence,
                    event.execution_id,
                    event.idempotency_key,
                    event.con_id,
                    event.symbol,
                    event.side.value,
                    _decimal_text(event.quantity),
                    _decimal_text(event.price),
                    _utc_text(event.occurred_at),
                    _utc_text(event.recorded_at),
                    event.fingerprint(),
                ),
            )
            self._connection.execute(
                """
                INSERT INTO fifo_commissions(
                    commission_id, epoch_id, fill_id, amount_minor, currency,
                    minor_unit_exponent, recorded_at
                ) VALUES (?, ?, ?, ?, 'USD', 2, ?)
                """,
                (
                    event.commission_id,
                    event.epoch_id,
                    event.fill_id,
                    event.commission_minor,
                    _utc_text(event.recorded_at),
                ),
            )
            result = self._project_fill(event)
            self._connection.commit()
            return result
        except BaseException:
            if self._connection.in_transaction:
                self._connection.rollback()
            raise

    def verify_epoch_integrity(self, epoch_id: str) -> None:
        """Recompute immutable relationships and hash-chain evidence for an epoch."""

        _assert_no_temp_fifo_objects(self._connection)
        _identifier(epoch_id, "epoch_id")
        effective_at = self._require_epoch(epoch_id)
        fill_count, commission_count, snapshot_count = self._connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM fifo_fills WHERE epoch_id = ?),
                (SELECT COUNT(*) FROM fifo_commissions WHERE epoch_id = ?),
                (SELECT COUNT(*) FROM fifo_position_snapshots WHERE epoch_id = ?)
            """,
            (epoch_id, epoch_id, epoch_id),
        ).fetchone()
        if not (
            _stored_int(fill_count, "fill count")
            == _stored_int(commission_count, "commission count")
            == _stored_int(snapshot_count, "snapshot count")
        ):
            raise FifoAccountingValidationError(
                "every fill must have exactly one commission and one snapshot"
            )
        rows = self._connection.execute(
            """
            SELECT f.fill_id, c.commission_id, f.event_sequence, f.execution_id,
                   f.idempotency_key, f.con_id, f.symbol, f.side, f.quantity_text,
                   f.price_text, c.amount_minor, f.occurred_at, f.recorded_at,
                   f.payload_fingerprint, c.recorded_at
            FROM fifo_fills f
            JOIN fifo_commissions c ON c.epoch_id = f.epoch_id AND c.fill_id = f.fill_id
            WHERE f.epoch_id = ? ORDER BY f.event_sequence
            """,
            (epoch_id,),
        ).fetchall()
        prior_time: Optional[datetime] = effective_at
        events: list[FillEvent] = []
        for expected_sequence, row in enumerate(rows, start=1):
            event = FillEvent(
                epoch_id=epoch_id,
                fill_id=str(row[0]),
                commission_id=str(row[1]),
                event_sequence=_stored_int(row[2], "fill event_sequence"),
                execution_id=str(row[3]),
                idempotency_key=str(row[4]),
                con_id=_stored_int(row[5], "fill con_id"),
                symbol=str(row[6]),
                side=FillSide(str(row[7])),
                quantity=_parse_decimal(row[8], "fill quantity", positive=True),
                price=_parse_decimal(row[9], "fill price", positive=True),
                commission_minor=_stored_int(row[10], "commission amount_minor"),
                occurred_at=_parse_utc(row[11], "fill occurred_at"),
                recorded_at=_parse_utc(row[12], "fill recorded_at"),
            )
            if event.event_sequence != expected_sequence:
                raise FifoAccountingValidationError("fill sequence is not contiguous")
            if prior_time is not None and event.occurred_at < prior_time:
                raise FifoAccountingValidationError("fill event time is not monotonic")
            prior_time = event.occurred_at
            events.append(event)
            if event.fingerprint() != row[13]:
                raise FifoAccountingValidationError("fill payload fingerprint mismatch")
            if _parse_utc(row[14], "commission recorded_at") != event.recorded_at:
                raise FifoAccountingValidationError("commission does not match its source fill")
            allocations = self._connection.execute(
                """
                SELECT match_ordinal, matched_quantity_text,
                       closing_commission_minor, 0 AS is_opening
                FROM fifo_lot_matches
                WHERE epoch_id = ? AND closing_fill_id = ?
                UNION ALL
                SELECT 1000000000 + lot_ordinal, opened_quantity_text,
                       opening_commission_minor, 1 AS is_opening
                FROM fifo_lot_openings
                WHERE epoch_id = ? AND opening_fill_id = ?
                ORDER BY match_ordinal
                """,
                (epoch_id, event.fill_id, epoch_id, event.fill_id),
            ).fetchall()
            exact_allocations = _allocate_minor(
                event.commission_minor,
                [
                    _parse_decimal(allocation[1], "commission segment", positive=True)
                    for allocation in allocations
                ],
            )
            allocated_quantity = Decimal("0")
            for allocation in allocations:
                allocated_quantity = _add(
                    allocated_quantity,
                    _parse_decimal(allocation[1], "commission segment", positive=True),
                    "allocated fill quantity",
                )
            if allocated_quantity != event.quantity:
                raise FifoAccountingValidationError("fill quantity allocation is incomplete")
            stored_allocations = [
                _stored_int(allocation[2], "commission allocation") for allocation in allocations
            ]
            if stored_allocations != exact_allocations:
                raise FifoAccountingValidationError(
                    "fill commission allocation is not deterministic"
                )
            if sum(stored_allocations) != event.commission_minor:
                raise FifoAccountingValidationError("fill commission allocation is incomplete")

        self._verify_fifo_projection_structure(events)

        lot_rows = self._connection.execute(
            """
            SELECT lot_id, direction, opened_quantity_text, open_price_text,
                   opening_commission_minor
            FROM fifo_lot_openings WHERE epoch_id = ?
            """,
            (epoch_id,),
        ).fetchall()
        for lot_id, direction, opened_text, open_price_text, commission_minor in lot_rows:
            opened = _parse_decimal(opened_text, "opened quantity", positive=True)
            open_price = _parse_decimal(open_price_text, "open price", positive=True)
            matches = self._connection.execute(
                """
                SELECT matched_quantity_text, opening_price_text, closing_price_text,
                       opening_commission_minor, closing_commission_minor,
                       gross_pnl_text, realized_pnl_text
                FROM fifo_lot_matches m
                JOIN fifo_fills f ON f.fill_id = m.closing_fill_id
                WHERE m.epoch_id = ? AND m.opening_lot_id = ?
                ORDER BY f.event_sequence, m.match_ordinal
                """,
                (epoch_id, lot_id),
            ).fetchall()
            matched = Decimal("0")
            allocated_opening = 0
            for match in matches:
                quantity = _parse_decimal(match[0], "matched quantity", positive=True)
                if _parse_decimal(match[1], "recorded open price", positive=True) != open_price:
                    raise FifoAccountingValidationError("lot match open price diverges")
                close_price = _parse_decimal(match[2], "closing price", positive=True)
                opening_allocation = _stored_int(match[3], "opening commission allocation")
                closing_allocation = _stored_int(match[4], "closing commission allocation")
                lot_state = _OpenLot(
                    lot_id=str(lot_id),
                    opened_sequence=0,
                    direction=str(direction),
                    opened_quantity=opened,
                    remaining_quantity=_subtract(opened, matched, "remaining lot quantity"),
                    open_price=open_price,
                    opening_commission_minor=_stored_int(
                        commission_minor, "lot opening commission"
                    ),
                    allocated_opening_commission_minor=allocated_opening,
                )
                if self._opening_commission_for_match(lot_state, quantity) != opening_allocation:
                    raise FifoAccountingValidationError(
                        "opening commission allocation is not deterministic"
                    )
                per_share = (
                    _subtract(close_price, open_price, "long P&L per share")
                    if direction == "LONG"
                    else _subtract(open_price, close_price, "short P&L per share")
                )
                gross = _multiply(per_share, quantity, "gross realized P&L")
                realized = _subtract(
                    _subtract(
                        gross,
                        _money_from_minor(opening_allocation),
                        "realized P&L",
                    ),
                    _money_from_minor(closing_allocation),
                    "realized P&L",
                )
                if gross != _parse_decimal(match[5], "gross P&L"):
                    raise FifoAccountingValidationError("lot match gross P&L diverges")
                if realized != _parse_decimal(match[6], "realized P&L"):
                    raise FifoAccountingValidationError("lot match realized P&L diverges")
                matched = _add(matched, quantity, "matched quantity")
                allocated_opening += opening_allocation
            if matched > opened:
                raise FifoAccountingValidationError("lot is over-matched")
            if matched == opened and allocated_opening != _stored_int(
                commission_minor, "lot opening commission"
            ):
                raise FifoAccountingValidationError(
                    "closed lot opening commission is not fully allocated"
                )

        previous_by_asset: dict[tuple[int, str], tuple[str, str]] = {}
        snapshot_by_fill: dict[str, Sequence[object]] = {}
        snapshots = self._connection.execute(
            """
            SELECT snapshot_id, source_fill_id, event_sequence, con_id, symbol,
                   signed_quantity_text, open_cost_text, open_lot_count,
                   cumulative_realized_pnl_text, cumulative_commission_minor,
                   previous_snapshot_id, previous_state_fingerprint, state_fingerprint,
                   created_at
            FROM fifo_position_snapshots WHERE epoch_id = ? ORDER BY event_sequence
            """,
            (epoch_id,),
        ).fetchall()
        if len(snapshots) != len(rows):
            raise FifoAccountingValidationError("every fill must have exactly one snapshot")
        events_by_fill = {event.fill_id: event for event in events}
        for row in snapshots:
            source_event = events_by_fill.get(str(row[1]))
            if source_event is None or (
                _stored_int(row[2], "snapshot event_sequence"),
                _stored_int(row[3], "snapshot con_id"),
                str(row[4]),
                _parse_utc(row[13], "snapshot created_at"),
            ) != (
                source_event.event_sequence,
                source_event.con_id,
                source_event.symbol,
                source_event.recorded_at,
            ):
                raise FifoAccountingValidationError("snapshot does not match its source fill")
            asset = (_stored_int(row[3], "snapshot con_id"), str(row[4]))
            previous = previous_by_asset.get(asset)
            expected_previous_id = None if previous is None else previous[0]
            expected_previous_hash = None if previous is None else previous[1]
            if row[10] != expected_previous_id or row[11] != expected_previous_hash:
                raise FifoAccountingValidationError("snapshot chain is discontinuous")
            payload = {
                "con_id": _stored_int(row[3], "snapshot con_id"),
                "cumulative_commission_minor": _stored_int(
                    row[9], "snapshot cumulative commission"
                ),
                "cumulative_realized_pnl": _decimal_text(
                    _parse_decimal(row[8], "cumulative realized P&L")
                ),
                "epoch_id": epoch_id,
                "event_sequence": _stored_int(row[2], "snapshot event_sequence"),
                "open_cost": (
                    None
                    if row[6] is None
                    else _decimal_text(_parse_decimal(row[6], "open cost", positive=True))
                ),
                "open_lot_count": _stored_int(row[7], "snapshot open_lot_count"),
                "previous_snapshot_id": expected_previous_id,
                "previous_state_fingerprint": expected_previous_hash,
                "signed_quantity": _decimal_text(_parse_decimal(row[5], "signed quantity")),
                "snapshot_id": str(row[0]),
                "source_fill_id": str(row[1]),
                "symbol": str(row[4]),
            }
            if _fingerprint(payload) != row[12]:
                raise FifoAccountingValidationError("snapshot state fingerprint mismatch")
            previous_by_asset[asset] = (str(row[0]), str(row[12]))
            snapshot_by_fill[str(row[1])] = row

        for event in events:
            asset = (event.con_id, event.symbol)
            lots = self._open_lots(event, through_sequence=event.event_sequence)
            expected_quantity = Decimal("0")
            expected_open_cost = Decimal("0")
            for lot in lots:
                sign = Decimal("1") if lot.direction == "LONG" else Decimal("-1")
                expected_quantity = _add(
                    expected_quantity,
                    _multiply(sign, lot.remaining_quantity, "signed lot quantity"),
                    "signed position quantity",
                )
                expected_open_cost = _add(
                    expected_open_cost,
                    _multiply(lot.open_price, lot.remaining_quantity, "open lot cost"),
                    "open position cost",
                )
            realized_rows = self._connection.execute(
                """
                SELECT m.realized_pnl_text
                FROM fifo_lot_matches m
                JOIN fifo_fills f ON f.fill_id = m.closing_fill_id
                WHERE m.epoch_id = ? AND f.con_id = ? AND f.symbol = ?
                  AND f.event_sequence <= ?
                """,
                (epoch_id, asset[0], asset[1], event.event_sequence),
            ).fetchall()
            expected_realized = Decimal("0")
            for realized_row in realized_rows:
                expected_realized = _add(
                    expected_realized,
                    _parse_decimal(realized_row[0], "realized P&L"),
                    "cumulative realized P&L",
                )
            commission_rows = self._connection.execute(
                """
                SELECT c.amount_minor
                FROM fifo_commissions c JOIN fifo_fills f ON f.fill_id = c.fill_id
                WHERE c.epoch_id = ? AND f.con_id = ? AND f.symbol = ?
                  AND f.event_sequence <= ?
                """,
                (epoch_id, asset[0], asset[1], event.event_sequence),
            ).fetchall()
            expected_commission = sum(
                _stored_int(commission_row[0], "commission amount_minor")
                for commission_row in commission_rows
            )
            snapshot = snapshot_by_fill[event.fill_id]
            expected_open_cost_text = (
                None if expected_quantity == 0 else _decimal_text(expected_open_cost)
            )
            if (
                _parse_decimal(snapshot[5], "snapshot signed quantity") != expected_quantity
                or snapshot[6] != expected_open_cost_text
                or _stored_int(snapshot[7], "snapshot open_lot_count") != len(lots)
                or _parse_decimal(snapshot[8], "snapshot realized P&L") != expected_realized
                or _stored_int(snapshot[9], "snapshot cumulative commission") != expected_commission
            ):
                raise FifoAccountingValidationError(
                    "position snapshot diverges from immutable events"
                )

    def _verify_fifo_projection_structure(self, events: Sequence[FillEvent]) -> None:
        active_by_asset: dict[tuple[int, str], list[_ProjectionLot]] = {}
        if events:
            epoch_id = events[0].epoch_id
            opening_rows = self._connection.execute(
                """
                SELECT lot_id, con_id, symbol, direction, opened_quantity_text,
                       open_price_text
                FROM fifo_lot_openings
                WHERE epoch_id = ? AND opening_balance_id IS NOT NULL
                ORDER BY con_id, symbol, lot_id
                """,
                (epoch_id,),
            ).fetchall()
            for row in opening_rows:
                asset = (_stored_int(row[1], "opening con_id"), str(row[2]))
                active_by_asset.setdefault(asset, []).append(
                    _ProjectionLot(
                        lot_id=str(row[0]),
                        direction=str(row[3]),
                        remaining_quantity=_parse_decimal(
                            row[4], "opening quantity", positive=True
                        ),
                        open_price=_parse_decimal(row[5], "opening price", positive=True),
                    )
                )
        for event in events:
            asset = (event.con_id, event.symbol)
            active = active_by_asset.setdefault(asset, [])
            fill_direction = "LONG" if event.side is FillSide.BUY else "SHORT"
            opposite = "SHORT" if fill_direction == "LONG" else "LONG"
            remaining = event.quantity
            expected_matches: list[tuple[_ProjectionLot, Decimal]] = []
            if active and active[0].direction == opposite:
                for lot in active:
                    if remaining == 0:
                        break
                    quantity = min(remaining, lot.remaining_quantity)
                    expected_matches.append((lot, quantity))
                    remaining = _subtract(remaining, quantity, "verified unmatched fill quantity")

            matches = self._connection.execute(
                """
                SELECT match_id, opening_lot_id, match_ordinal, matched_quantity_text,
                       opening_price_text, closing_price_text, matched_at
                FROM fifo_lot_matches
                WHERE epoch_id = ? AND closing_fill_id = ?
                ORDER BY match_ordinal
                """,
                (event.epoch_id, event.fill_id),
            ).fetchall()
            if len(matches) != len(expected_matches):
                raise FifoAccountingValidationError("FIFO projection structure diverges")
            for ordinal, (row, (lot, quantity)) in enumerate(zip(matches, expected_matches)):
                expected = (
                    _derived_id("fmatch", event.epoch_id, event.fill_id, ordinal),
                    lot.lot_id,
                    ordinal,
                    _decimal_text(quantity),
                    _decimal_text(lot.open_price),
                    _decimal_text(event.price),
                    _utc_text(event.occurred_at),
                )
                if tuple(row) != expected:
                    raise FifoAccountingValidationError("FIFO projection structure diverges")
                lot.remaining_quantity = _subtract(
                    lot.remaining_quantity,
                    quantity,
                    "verified remaining lot quantity",
                )
            active[:] = [lot for lot in active if lot.remaining_quantity > 0]

            openings = self._connection.execute(
                """
                SELECT lot_id, lot_ordinal, con_id, symbol, direction,
                       opened_quantity_text, open_price_text, opened_sequence, opened_at
                FROM fifo_lot_openings
                WHERE epoch_id = ? AND opening_fill_id = ?
                ORDER BY lot_ordinal
                """,
                (event.epoch_id, event.fill_id),
            ).fetchall()
            if remaining == 0:
                if openings:
                    raise FifoAccountingValidationError("FIFO projection structure diverges")
                continue
            expected_lot_id = _derived_id("flot", event.epoch_id, event.fill_id, 0)
            expected_opening = (
                expected_lot_id,
                0,
                event.con_id,
                event.symbol,
                fill_direction,
                _decimal_text(remaining),
                _decimal_text(event.price),
                event.event_sequence,
                _utc_text(event.occurred_at),
            )
            if len(openings) != 1 or tuple(openings[0]) != expected_opening:
                raise FifoAccountingValidationError("FIFO projection structure diverges")
            active.append(
                _ProjectionLot(
                    lot_id=expected_lot_id,
                    direction=fill_direction,
                    remaining_quantity=remaining,
                    open_price=event.price,
                )
            )

    def _require_epoch(self, epoch_id: str) -> datetime:
        row = self._connection.execute(
            "SELECT origin_kind, effective_at FROM fifo_accounting_epochs WHERE epoch_id = ?",
            (epoch_id,),
        ).fetchone()
        if row is None:
            raise FifoAccountingValidationError("unknown accounting epoch")
        if row[0] == "LEGACY_AGGREGATE_OPENING_BALANCE":
            self._verify_legacy_epoch_shape(epoch_id)
        elif row[0] != "EMPTY_LEDGER":
            raise FifoAccountingValidationError("accounting epoch origin is unsupported")
        return _parse_utc(row[1], "epoch effective_at")

    def _verify_legacy_epoch_shape(self, epoch_id: str) -> None:
        lineage_rows = self._connection.execute(
            """
            SELECT e.source_fingerprint,l.candidate_fingerprint,e.effective_at
            FROM fifo_accounting_epochs e
            JOIN fifo_legacy_bootstrap_lineage l ON l.epoch_id=e.epoch_id
            WHERE e.epoch_id=?
            """,
            (epoch_id,),
        ).fetchall()
        baseline_count = self._connection.execute(
            "SELECT COUNT(*) FROM fifo_epoch_account_baselines WHERE epoch_id = ?",
            (epoch_id,),
        ).fetchone()
        if (
            len(lineage_rows) != 1
            or lineage_rows[0][0] != lineage_rows[0][1]
            or baseline_count != (1,)
        ):
            raise FifoAccountingValidationError(
                "legacy epoch lacks one sealed lineage and account baseline"
            )
        effective_at = _parse_utc(lineage_rows[0][2], "legacy epoch effective_at")
        balances = self._connection.execute(
            """
            SELECT b.opening_balance_id,b.con_id,b.symbol,b.direction,
                   b.opened_quantity_text,b.cost_basis_text,l.opening_balance_id,
                   l.con_id,l.symbol,l.direction,l.opened_quantity_text,l.open_price_text,
                   l.opening_commission_minor,l.opened_sequence,l.opened_at,
                   b.mark_observed_at
            FROM fifo_opening_balances b
            LEFT JOIN fifo_lot_openings l
              ON l.epoch_id=b.epoch_id AND l.opening_balance_id=b.opening_balance_id
            WHERE b.epoch_id=? ORDER BY b.symbol
            """,
            (epoch_id,),
        ).fetchall()
        lot_count = self._connection.execute(
            """
            SELECT COUNT(*) FROM fifo_lot_openings
            WHERE epoch_id=? AND opening_balance_id IS NOT NULL
            """,
            (epoch_id,),
        ).fetchone()
        if lot_count != (len(balances),):
            raise FifoAccountingValidationError(
                "legacy opening balances do not map one-to-one to opening lots"
            )
        for row in balances:
            if row[6] is None or tuple(row[1:6]) != tuple(row[7:12]):
                raise FifoAccountingValidationError(
                    "legacy opening balance differs from its exact opening lot"
                )
            if (
                _stored_int(row[12], "legacy opening commission") != 0
                or _stored_int(row[13], "legacy opened_sequence") != 0
                or _parse_utc(row[14], "legacy opened_at") != effective_at
                or _parse_utc(row[15], "legacy mark_observed_at") > effective_at
            ):
                raise FifoAccountingValidationError(
                    "legacy opening lot must not reconstruct pre-epoch commissions"
                )

    def _existing_fill(self, event: FillEvent) -> Optional[FillResult]:
        rows = self._connection.execute(
            """
            SELECT DISTINCT fill_id FROM fifo_fills
            WHERE epoch_id = ? AND (
                fill_id = ? OR execution_id = ? OR idempotency_key = ? OR event_sequence = ?
            )
            """,
            (
                event.epoch_id,
                event.fill_id,
                event.execution_id,
                event.idempotency_key,
                event.event_sequence,
            ),
        ).fetchall()
        commission_owner = self._connection.execute(
            "SELECT fill_id FROM fifo_commissions WHERE commission_id = ?",
            (event.commission_id,),
        ).fetchone()
        identities = {str(row[0]) for row in rows}
        if commission_owner is not None:
            identities.add(str(commission_owner[0]))
        if not identities:
            return None
        if identities != {event.fill_id}:
            raise FifoAccountingConflict("fill identities resolve to different records")
        stored = self._connection.execute(
            """
            SELECT f.epoch_id, f.fill_id, c.commission_id, f.event_sequence,
                   f.execution_id, f.idempotency_key, f.con_id, f.symbol, f.side,
                   f.quantity_text, f.price_text, c.amount_minor, f.occurred_at,
                   f.recorded_at, f.payload_fingerprint
            FROM fifo_fills f JOIN fifo_commissions c ON c.fill_id = f.fill_id
            WHERE f.fill_id = ?
            """,
            (event.fill_id,),
        ).fetchone()
        if stored is None:
            raise FifoAccountingConflict("existing fill is incomplete")
        expected = (
            event.epoch_id,
            event.fill_id,
            event.commission_id,
            event.event_sequence,
            event.execution_id,
            event.idempotency_key,
            event.con_id,
            event.symbol,
            event.side.value,
            _decimal_text(event.quantity),
            _decimal_text(event.price),
            event.commission_minor,
            _utc_text(event.occurred_at),
            _utc_text(event.recorded_at),
            event.fingerprint(),
        )
        if tuple(stored) != expected:
            raise FifoAccountingConflict("fill identity is bound to different data")
        return self._load_result(event.fill_id, replayed=True)

    def _assert_next_event(self, event: FillEvent) -> None:
        row = self._connection.execute(
            """
            SELECT event_sequence, occurred_at FROM fifo_fills
            WHERE epoch_id = ? ORDER BY event_sequence DESC LIMIT 1
            """,
            (event.epoch_id,),
        ).fetchone()
        expected = 1 if row is None else _stored_int(row[0], "fill event_sequence") + 1
        if event.event_sequence != expected:
            raise FifoAccountingOrderingError(f"event_sequence must extend the epoch at {expected}")
        if row is not None and event.occurred_at < _parse_utc(row[1], "prior occurred_at"):
            raise FifoAccountingOrderingError("fill event time precedes the prior sequence")

    def _assert_instrument_identity(self, event: FillEvent) -> None:
        rows = self._connection.execute(
            """
            SELECT con_id,symbol FROM fifo_fills
            WHERE epoch_id = ? AND (con_id = ? OR symbol = ?)
            UNION
            SELECT con_id,symbol FROM fifo_opening_balances
            WHERE epoch_id = ? AND (con_id = ? OR symbol = ?)
            """,
            (
                event.epoch_id,
                event.con_id,
                event.symbol,
                event.epoch_id,
                event.con_id,
                event.symbol,
            ),
        ).fetchall()
        if any(
            (_stored_int(row[0], "fill con_id"), str(row[1])) != (event.con_id, event.symbol)
            for row in rows
        ):
            raise FifoAccountingConflict("contract identifier and symbol binding changed")

    def _open_lots(
        self,
        event: FillEvent,
        *,
        through_sequence: Optional[int] = None,
    ) -> list[_OpenLot]:
        rows = self._connection.execute(
            """
            SELECT l.lot_id, l.opened_sequence, l.direction, l.opened_quantity_text,
                   l.open_price_text, l.opening_commission_minor
            FROM fifo_lot_openings l
            WHERE l.epoch_id = ? AND l.con_id = ? AND l.symbol = ?
              AND (? IS NULL OR l.opened_sequence <= ?)
            ORDER BY l.opened_sequence, l.lot_id
            """,
            (
                event.epoch_id,
                event.con_id,
                event.symbol,
                through_sequence,
                through_sequence,
            ),
        ).fetchall()
        result: list[_OpenLot] = []
        for row in rows:
            opened = _parse_decimal(row[3], "opened_quantity", positive=True)
            # SQLite numeric aggregation would be inexact. Recompute from text.
            matched_rows = self._connection.execute(
                """
                SELECT m.matched_quantity_text, m.opening_commission_minor
                FROM fifo_lot_matches m
                JOIN fifo_fills f ON f.fill_id = m.closing_fill_id
                WHERE m.epoch_id = ? AND m.opening_lot_id = ?
                  AND (? IS NULL OR f.event_sequence <= ?)
                ORDER BY f.event_sequence, m.match_ordinal
                """,
                (event.epoch_id, row[0], through_sequence, through_sequence),
            ).fetchall()
            matched = Decimal("0")
            allocated = 0
            for matched_text, allocated_minor in matched_rows:
                matched = _add(
                    matched,
                    _parse_decimal(matched_text, "matched_quantity", positive=True),
                    "matched quantity",
                )
                allocated += _stored_int(allocated_minor, "opening commission allocation")
            remaining = _subtract(opened, matched, "remaining lot quantity")
            if remaining < 0:
                raise FifoAccountingValidationError("persisted lot is over-matched")
            if remaining == 0:
                if allocated != _stored_int(row[5], "lot opening commission"):
                    raise FifoAccountingValidationError(
                        "closed lot did not allocate its complete opening commission"
                    )
                continue
            result.append(
                _OpenLot(
                    lot_id=str(row[0]),
                    opened_sequence=_stored_int(row[1], "lot opened_sequence"),
                    direction=str(row[2]),
                    opened_quantity=opened,
                    remaining_quantity=remaining,
                    open_price=_parse_decimal(row[4], "open_price", positive=True),
                    opening_commission_minor=_stored_int(row[5], "lot opening commission"),
                    allocated_opening_commission_minor=allocated,
                )
            )
        directions = {lot.direction for lot in result}
        if len(directions) > 1:
            raise FifoAccountingValidationError("persisted open lots cross zero")
        return result

    @staticmethod
    def _opening_commission_for_match(lot: _OpenLot, quantity: Decimal) -> int:
        if quantity == lot.remaining_quantity:
            return lot.opening_commission_minor - lot.allocated_opening_commission_minor
        consumed_before = _subtract(
            lot.opened_quantity,
            lot.remaining_quantity,
            "consumed opening quantity",
        )
        consumed_after = _add(consumed_before, quantity, "consumed opening quantity")
        before_atoms, after_atoms, total_atoms = _quantity_atoms(
            [consumed_before, consumed_after, lot.opened_quantity]
        )
        magnitude = abs(lot.opening_commission_minor)
        sign = -1 if lot.opening_commission_minor < 0 else 1
        target_before = sign * (magnitude * before_atoms // total_atoms)
        target_after = sign * (magnitude * after_atoms // total_atoms)
        allocation = target_after - target_before
        if lot.allocated_opening_commission_minor != target_before:
            raise FifoAccountingValidationError(
                "persisted opening commission allocation is not deterministic"
            )
        return allocation

    def _project_fill(self, event: FillEvent) -> FillResult:
        lots = self._open_lots(event)
        fill_direction = "LONG" if event.side is FillSide.BUY else "SHORT"
        opposite = "SHORT" if fill_direction == "LONG" else "LONG"
        remaining = event.quantity
        match_specs: list[tuple[_OpenLot, Decimal]] = []
        if lots and lots[0].direction == opposite:
            for lot in lots:
                if remaining == 0:
                    break
                quantity = min(remaining, lot.remaining_quantity)
                match_specs.append((lot, quantity))
                remaining = _subtract(remaining, quantity, "unmatched fill quantity")
        segment_quantities = [quantity for _, quantity in match_specs]
        if remaining > 0:
            segment_quantities.append(remaining)
        commission_allocations = _allocate_minor(event.commission_minor, segment_quantities)

        match_ids: list[str] = []
        realized_delta = Decimal("0")
        for ordinal, ((lot, quantity), closing_commission) in enumerate(
            zip(match_specs, commission_allocations)
        ):
            opening_commission = self._opening_commission_for_match(lot, quantity)
            per_share = (
                _subtract(event.price, lot.open_price, "long P&L per share")
                if lot.direction == "LONG"
                else _subtract(lot.open_price, event.price, "short P&L per share")
            )
            gross = _multiply(per_share, quantity, "gross realized P&L")
            realized = _subtract(
                _subtract(
                    gross,
                    _money_from_minor(opening_commission),
                    "net realized P&L",
                ),
                _money_from_minor(closing_commission),
                "net realized P&L",
            )
            match_id = _derived_id("fmatch", event.epoch_id, event.fill_id, ordinal)
            self._connection.execute(
                """
                INSERT INTO fifo_lot_matches(
                    match_id, epoch_id, closing_fill_id, opening_lot_id,
                    match_ordinal, matched_quantity_text, opening_price_text,
                    closing_price_text, opening_commission_minor,
                    closing_commission_minor, gross_pnl_text, realized_pnl_text,
                    matched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    match_id,
                    event.epoch_id,
                    event.fill_id,
                    lot.lot_id,
                    ordinal,
                    _decimal_text(quantity),
                    _decimal_text(lot.open_price),
                    _decimal_text(event.price),
                    opening_commission,
                    closing_commission,
                    _decimal_text(gross),
                    _decimal_text(realized),
                    _utc_text(event.occurred_at),
                ),
            )
            match_ids.append(match_id)
            realized_delta = _add(realized_delta, realized, "fill realized P&L")

        opened_lot_id: Optional[str] = None
        if remaining > 0:
            opening_commission = commission_allocations[-1]
            opened_lot_id = _derived_id("flot", event.epoch_id, event.fill_id, 0)
            self._connection.execute(
                """
                INSERT INTO fifo_lot_openings(
                    lot_id, epoch_id, opening_fill_id, opening_balance_id,
                    lot_ordinal, con_id, symbol,
                    direction, opened_quantity_text, open_price_text,
                    opening_commission_minor, opened_sequence, opened_at
                ) VALUES (?, ?, ?, NULL, 0, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    opened_lot_id,
                    event.epoch_id,
                    event.fill_id,
                    event.con_id,
                    event.symbol,
                    fill_direction,
                    _decimal_text(remaining),
                    _decimal_text(event.price),
                    opening_commission,
                    event.event_sequence,
                    _utc_text(event.occurred_at),
                ),
            )
        snapshot = self._append_snapshot(event, realized_delta)
        return FillResult(
            fill_id=event.fill_id,
            opened_lot_id=opened_lot_id,
            match_ids=tuple(match_ids),
            snapshot=snapshot,
            replayed=False,
        )

    def _append_snapshot(self, event: FillEvent, realized_delta: Decimal) -> PositionSnapshot:
        lots = self._open_lots(event)
        signed_quantity = Decimal("0")
        open_cost = Decimal("0")
        for lot in lots:
            sign = Decimal("1") if lot.direction == "LONG" else Decimal("-1")
            signed_quantity = _add(
                signed_quantity,
                _multiply(sign, lot.remaining_quantity, "signed lot quantity"),
                "signed position quantity",
            )
            open_cost = _add(
                open_cost,
                _multiply(lot.open_price, lot.remaining_quantity, "open lot cost"),
                "open position cost",
            )
        exact_open_cost = None if signed_quantity == 0 else open_cost

        previous = self._connection.execute(
            """
            SELECT snapshot_id, cumulative_realized_pnl_text,
                   cumulative_commission_minor, state_fingerprint
            FROM fifo_position_snapshots
            WHERE epoch_id = ? AND con_id = ? AND symbol = ?
            ORDER BY event_sequence DESC LIMIT 1
            """,
            (event.epoch_id, event.con_id, event.symbol),
        ).fetchone()
        previous_id = None if previous is None else str(previous[0])
        previous_realized = (
            Decimal("0")
            if previous is None
            else _parse_decimal(previous[1], "cumulative realized P&L")
        )
        cumulative_realized = _add(previous_realized, realized_delta, "cumulative realized P&L")
        previous_commission = 0 if previous is None else int(previous[2])
        cumulative_commission = previous_commission + event.commission_minor
        previous_fingerprint = None if previous is None else str(previous[3])
        snapshot_id = _derived_id("fsnap", event.epoch_id, event.fill_id)
        payload = {
            "con_id": event.con_id,
            "cumulative_commission_minor": cumulative_commission,
            "cumulative_realized_pnl": _decimal_text(cumulative_realized),
            "epoch_id": event.epoch_id,
            "event_sequence": event.event_sequence,
            "open_cost": None if exact_open_cost is None else _decimal_text(exact_open_cost),
            "open_lot_count": len(lots),
            "previous_snapshot_id": previous_id,
            "previous_state_fingerprint": previous_fingerprint,
            "signed_quantity": _decimal_text(signed_quantity),
            "snapshot_id": snapshot_id,
            "source_fill_id": event.fill_id,
            "symbol": event.symbol,
        }
        state_fingerprint = _fingerprint(payload)
        self._connection.execute(
            """
            INSERT INTO fifo_position_snapshots(
                snapshot_id, epoch_id, source_fill_id, event_sequence, con_id,
                symbol, signed_quantity_text, open_cost_text,
                open_lot_count, cumulative_realized_pnl_text,
                cumulative_commission_minor, previous_snapshot_id,
                previous_state_fingerprint, state_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                event.epoch_id,
                event.fill_id,
                event.event_sequence,
                event.con_id,
                event.symbol,
                _decimal_text(signed_quantity),
                None if exact_open_cost is None else _decimal_text(exact_open_cost),
                len(lots),
                _decimal_text(cumulative_realized),
                cumulative_commission,
                previous_id,
                previous_fingerprint,
                state_fingerprint,
                _utc_text(event.recorded_at),
            ),
        )
        return PositionSnapshot(
            snapshot_id=snapshot_id,
            epoch_id=event.epoch_id,
            source_fill_id=event.fill_id,
            event_sequence=event.event_sequence,
            con_id=event.con_id,
            symbol=event.symbol,
            signed_quantity=signed_quantity,
            open_cost=exact_open_cost,
            open_lot_count=len(lots),
            cumulative_realized_pnl=cumulative_realized,
            cumulative_commission_minor=cumulative_commission,
            previous_snapshot_id=previous_id,
            previous_state_fingerprint=previous_fingerprint,
            state_fingerprint=state_fingerprint,
            created_at=event.recorded_at,
        )

    def _load_result(self, fill_id: str, *, replayed: bool) -> FillResult:
        opening = self._connection.execute(
            "SELECT lot_id FROM fifo_lot_openings WHERE opening_fill_id = ?", (fill_id,)
        ).fetchone()
        matches = self._connection.execute(
            """
            SELECT match_id FROM fifo_lot_matches
            WHERE closing_fill_id = ? ORDER BY match_ordinal
            """,
            (fill_id,),
        ).fetchall()
        row = self._connection.execute(
            """
            SELECT snapshot_id, epoch_id, source_fill_id, event_sequence, con_id,
                   symbol, signed_quantity_text, open_cost_text, open_lot_count,
                   cumulative_realized_pnl_text, cumulative_commission_minor,
                   previous_snapshot_id, previous_state_fingerprint,
                   state_fingerprint, created_at
            FROM fifo_position_snapshots WHERE source_fill_id = ?
            """,
            (fill_id,),
        ).fetchone()
        if row is None:
            raise FifoAccountingConflict("existing fill has no derived snapshot")
        snapshot = PositionSnapshot(
            snapshot_id=str(row[0]),
            epoch_id=str(row[1]),
            source_fill_id=str(row[2]),
            event_sequence=_stored_int(row[3], "snapshot event_sequence"),
            con_id=_stored_int(row[4], "snapshot con_id"),
            symbol=str(row[5]),
            signed_quantity=_parse_decimal(row[6], "signed quantity"),
            open_cost=(
                None if row[7] is None else _parse_decimal(row[7], "open cost basis", positive=True)
            ),
            open_lot_count=_stored_int(row[8], "snapshot open_lot_count"),
            cumulative_realized_pnl=_parse_decimal(row[9], "cumulative realized P&L"),
            cumulative_commission_minor=_stored_int(row[10], "snapshot cumulative commission"),
            previous_snapshot_id=None if row[11] is None else str(row[11]),
            previous_state_fingerprint=None if row[12] is None else str(row[12]),
            state_fingerprint=str(row[13]),
            created_at=_parse_utc(row[14], "snapshot created_at"),
        )
        return FillResult(
            fill_id=fill_id,
            opened_lot_id=None if opening is None else str(opening[0]),
            match_ids=tuple(str(match[0]) for match in matches),
            snapshot=snapshot,
            replayed=replayed,
        )

"""Transactional runtime bridge into the exact FIFO ledger.

The bridge accepts only explicit local-paper fill evidence.  It assigns the
next gap-free epoch sequence while the caller holds ``BEGIN IMMEDIATE``, writes
the immutable FIFO event without committing, verifies the complete epoch, and
returns exact values for compatibility projections.  It never creates an
accounting epoch and never infers a fill from an order or position change.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from typing import Optional

import aiosqlite

from .fifo import (
    FifoAccountingConflict,
    FifoAccountingError,
    FifoAccountingValidationError,
    FifoLedger,
    FillEvent,
    FillResult,
    FillSide,
    _decimal_text,
    _parse_decimal,
    _stored_int,
)

LOCAL_PAPER_COMMISSION_SOURCE = "LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1"
_SCOPE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9._-]{0,31}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class FifoRuntimeSettlementError(FifoAccountingError):
    """Runtime fill evidence cannot be projected without ambiguity."""


def reduction_side_to_fifo(value: object) -> FillSide:
    """Map the only two exposure-reducing order sides to FIFO directions."""

    if value == "SELL":
        return FillSide.SELL
    if value == "BUY_TO_COVER":
        return FillSide.BUY
    raise FifoRuntimeSettlementError("runtime FIFO accepts only reduction order sides")


def _strict_scope(value: object, field: str, pattern: re.Pattern[str] = _SCOPE) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise FifoRuntimeSettlementError(f"{field} is malformed")
    return value


def _strict_utc(value: object, field: str) -> datetime:
    if type(value) is not datetime:
        raise FifoRuntimeSettlementError(f"{field} must be a datetime")
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None or offset.total_seconds() != 0:
        raise FifoRuntimeSettlementError(f"{field} must be timezone-aware UTC")
    return value.astimezone(timezone.utc)


def _derived_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _runtime_epoch_id(
    connection: sqlite3.Connection,
    evidence: "RuntimePaperFillEvidence",
) -> str:
    epochs = connection.execute(
        """
        SELECT epoch_id FROM fifo_accounting_epochs
        WHERE execution_domain_scope=? AND account_scope=? AND portfolio_id=?
        """,
        (
            evidence.execution_domain_scope,
            evidence.account_scope,
            evidence.portfolio_id,
        ),
    ).fetchall()
    if len(epochs) != 1:
        raise FifoRuntimeSettlementError("runtime fill requires exactly one sealed FIFO epoch")
    return str(epochs[0][0])


@dataclass(frozen=True, slots=True)
class RuntimePaperFillEvidence:
    """One producer-observed paper execution and its exact commission report."""

    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    side: FillSide
    quantity: Decimal
    price: Decimal
    execution_id: str
    idempotency_key: str
    commission_minor: int
    commission_currency: str
    commission_source: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        _strict_scope(self.execution_domain_scope, "execution_domain_scope")
        _strict_scope(self.account_scope, "account_scope")
        _strict_scope(self.portfolio_id, "portfolio_id")
        if type(self.con_id) is not int or self.con_id <= 0:
            raise FifoRuntimeSettlementError("con_id must be a positive integer")
        _strict_scope(self.symbol, "symbol", _SYMBOL)
        if type(self.side) is not FillSide:
            raise FifoRuntimeSettlementError("side must be FillSide")
        FillEvent(
            epoch_id="fepoch-" + ("0" * 32),
            fill_id="ffill-" + ("0" * 32),
            commission_id="fcomm-" + ("0" * 32),
            event_sequence=1,
            execution_id=self.execution_id,
            idempotency_key=self.idempotency_key,
            con_id=self.con_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            price=self.price,
            commission_minor=self.commission_minor,
            occurred_at=_strict_utc(self.occurred_at, "occurred_at"),
            recorded_at=_strict_utc(self.occurred_at, "occurred_at"),
        )
        if self.commission_currency != "USD":
            raise FifoRuntimeSettlementError("runtime FIFO supports only exact USD commissions")
        if self.commission_source != LOCAL_PAPER_COMMISSION_SOURCE:
            raise FifoRuntimeSettlementError("paper commission source is not authoritative")
        if type(self.idempotency_key) is not str or _HASH.fullmatch(self.idempotency_key) is None:
            raise FifoRuntimeSettlementError("idempotency_key must be a SHA-256 fingerprint")


@dataclass(frozen=True, slots=True)
class FifoRuntimeProjection:
    """Exact post-fill values produced from immutable FIFO state."""

    epoch_id: str
    fill_id: str
    event_sequence: int
    signed_quantity: Decimal
    open_cost: Optional[Decimal]
    average_cost: Optional[Decimal]
    fill_realized_pnl: Decimal
    epoch_realized_pnl: Decimal
    baseline_realized_pnl: Decimal
    total_realized_pnl: Decimal
    cumulative_commission_minor: int
    state_fingerprint: str
    replayed: bool


def _projection_from_result(
    connection: sqlite3.Connection,
    *,
    epoch_id: str,
    fill_id: str,
    event_sequence: int,
    result: FillResult,
) -> FifoRuntimeProjection:
    snapshot = result.snapshot
    realized_rows = connection.execute(
        """
        SELECT realized_pnl_text FROM fifo_lot_matches
        WHERE epoch_id=? AND closing_fill_id=? ORDER BY match_ordinal
        """,
        (epoch_id, fill_id),
    ).fetchall()
    with localcontext() as context:
        context.prec = 96
        fill_realized = Decimal("0")
        for row in realized_rows:
            fill_realized += _parse_decimal(row[0], "runtime fill realized P&L")
    epoch_realized_rows = connection.execute(
        """
        SELECT m.realized_pnl_text
        FROM fifo_lot_matches AS m
        JOIN fifo_fills AS f
          ON f.epoch_id=m.epoch_id AND f.fill_id=m.closing_fill_id
        WHERE m.epoch_id=? AND f.event_sequence<=?
        ORDER BY f.event_sequence,m.match_ordinal
        """,
        (epoch_id, event_sequence),
    ).fetchall()
    with localcontext() as context:
        context.prec = 96
        epoch_realized = Decimal("0")
        for row in epoch_realized_rows:
            epoch_realized += _parse_decimal(row[0], "FIFO epoch realized P&L")
    baseline_row = connection.execute(
        "SELECT realized_pnl_text FROM fifo_epoch_account_baselines WHERE epoch_id=?",
        (epoch_id,),
    ).fetchone()
    baseline_realized = (
        Decimal("0")
        if baseline_row is None
        else _parse_decimal(baseline_row[0], "FIFO epoch realized baseline")
    )
    average_cost: Optional[Decimal] = None
    if snapshot.signed_quantity != 0:
        if snapshot.open_cost is None:
            raise FifoAccountingValidationError("nonzero FIFO position has no open cost")
        with localcontext() as context:
            context.prec = 96
            average_cost = snapshot.open_cost / abs(snapshot.signed_quantity)
        _decimal_text(average_cost)
    elif snapshot.open_cost is not None:
        raise FifoAccountingValidationError("flat FIFO position retained open cost")
    with localcontext() as context:
        context.prec = 96
        total_realized = baseline_realized + epoch_realized
    _decimal_text(total_realized)
    return FifoRuntimeProjection(
        epoch_id=epoch_id,
        fill_id=fill_id,
        event_sequence=event_sequence,
        signed_quantity=snapshot.signed_quantity,
        open_cost=snapshot.open_cost,
        average_cost=average_cost,
        fill_realized_pnl=fill_realized,
        epoch_realized_pnl=epoch_realized,
        baseline_realized_pnl=baseline_realized,
        total_realized_pnl=total_realized,
        cumulative_commission_minor=snapshot.cumulative_commission_minor,
        state_fingerprint=snapshot.state_fingerprint,
        replayed=result.replayed,
    )


def append_runtime_fill_in_transaction(
    connection: sqlite3.Connection,
    evidence: RuntimePaperFillEvidence,
) -> FifoRuntimeProjection:
    """Append or replay one fill inside the caller's active transaction."""

    if type(connection) is not sqlite3.Connection:
        raise TypeError("connection must be sqlite3.Connection")
    if type(evidence) is not RuntimePaperFillEvidence:
        raise TypeError("evidence must be RuntimePaperFillEvidence")
    if not connection.in_transaction:
        raise FifoRuntimeSettlementError("runtime FIFO append requires an active transaction")

    epoch_id = _runtime_epoch_id(connection, evidence)
    fill_id = _derived_id(
        "ffill",
        epoch_id,
        evidence.execution_id,
        evidence.idempotency_key,
    )
    commission_id = _derived_id("fcomm", epoch_id, fill_id)

    identities = connection.execute(
        """
        SELECT fill_id,event_sequence FROM fifo_fills
        WHERE epoch_id=? AND (execution_id=? OR idempotency_key=? OR fill_id=?)
        """,
        (epoch_id, evidence.execution_id, evidence.idempotency_key, fill_id),
    ).fetchall()
    identity_fill_ids = {str(row[0]) for row in identities}
    if len(identity_fill_ids) > 1:
        raise FifoAccountingConflict("runtime fill identities resolve to different events")
    if identities:
        if identity_fill_ids != {fill_id}:
            raise FifoAccountingConflict("runtime fill identity is already bound elsewhere")
        sequences = {_stored_int(row[1], "runtime fill event_sequence") for row in identities}
        if len(sequences) != 1:
            raise FifoAccountingConflict("runtime fill identities have inconsistent sequence")
        event_sequence = sequences.pop()
    else:
        prior = connection.execute(
            "SELECT MAX(event_sequence) FROM fifo_fills WHERE epoch_id=?",
            (epoch_id,),
        ).fetchone()
        if prior is None or prior[0] is None:
            event_sequence = 1
        else:
            event_sequence = _stored_int(prior[0], "prior event_sequence") + 1

    event = FillEvent(
        epoch_id=epoch_id,
        fill_id=fill_id,
        commission_id=commission_id,
        event_sequence=event_sequence,
        execution_id=evidence.execution_id,
        idempotency_key=evidence.idempotency_key,
        con_id=evidence.con_id,
        symbol=evidence.symbol,
        side=evidence.side,
        quantity=evidence.quantity,
        price=evidence.price,
        commission_minor=evidence.commission_minor,
        occurred_at=evidence.occurred_at,
        # The producer observation is stable across crash replay.  Database
        # commit time is recorded by the settlement outbox/link separately.
        recorded_at=evidence.occurred_at,
    )
    ledger = FifoLedger(connection, allow_other_objects=True)
    result = ledger.record_fill_in_transaction(event)
    ledger.verify_epoch_integrity(epoch_id)

    return _projection_from_result(
        connection,
        epoch_id=epoch_id,
        fill_id=fill_id,
        event_sequence=event_sequence,
        result=result,
    )


def verify_runtime_fill_in_transaction(
    connection: sqlite3.Connection,
    evidence: RuntimePaperFillEvidence,
) -> FifoRuntimeProjection:
    """Authenticate one already-persisted runtime fill without writing."""

    if type(connection) is not sqlite3.Connection:
        raise TypeError("connection must be sqlite3.Connection")
    if type(evidence) is not RuntimePaperFillEvidence:
        raise TypeError("evidence must be RuntimePaperFillEvidence")
    if not connection.in_transaction:
        raise FifoRuntimeSettlementError("runtime FIFO verification requires an active transaction")
    epoch_id = _runtime_epoch_id(connection, evidence)
    fill_id = _derived_id(
        "ffill",
        epoch_id,
        evidence.execution_id,
        evidence.idempotency_key,
    )
    rows = connection.execute(
        """
        SELECT event_sequence FROM fifo_fills
        WHERE epoch_id=? AND fill_id=? AND execution_id=? AND idempotency_key=?
        """,
        (epoch_id, fill_id, evidence.execution_id, evidence.idempotency_key),
    ).fetchall()
    if len(rows) != 1:
        raise FifoRuntimeSettlementError("runtime FIFO fill is absent or ambiguous")
    event_sequence = _stored_int(rows[0][0], "runtime fill event_sequence")
    event = FillEvent(
        epoch_id=epoch_id,
        fill_id=fill_id,
        commission_id=_derived_id("fcomm", epoch_id, fill_id),
        event_sequence=event_sequence,
        execution_id=evidence.execution_id,
        idempotency_key=evidence.idempotency_key,
        con_id=evidence.con_id,
        symbol=evidence.symbol,
        side=evidence.side,
        quantity=evidence.quantity,
        price=evidence.price,
        commission_minor=evidence.commission_minor,
        occurred_at=evidence.occurred_at,
        recorded_at=evidence.occurred_at,
    )
    ledger = FifoLedger(connection, allow_other_objects=True)
    result = ledger.record_fill_in_transaction(event)
    if not result.replayed:
        raise FifoRuntimeSettlementError("runtime FIFO verification attempted an append")
    ledger.verify_epoch_integrity(epoch_id)
    return _projection_from_result(
        connection,
        epoch_id=epoch_id,
        fill_id=fill_id,
        event_sequence=event_sequence,
        result=result,
    )


async def append_runtime_fill_on_aiosqlite_worker(
    connection: aiosqlite.Connection,
    evidence: RuntimePaperFillEvidence,
) -> FifoRuntimeProjection:
    """Run the synchronous projector on aiosqlite's owning worker thread.

    ``sqlite3.Connection`` objects may only be used by the thread that owns
    them.  aiosqlite intentionally funnels operations through ``_execute``;
    this narrow bridge keeps the complete FIFO append on that same worker and
    inside the already-active outer transaction.
    """

    if type(connection) is not aiosqlite.Connection:
        raise TypeError("connection must be aiosqlite.Connection")
    if not connection.in_transaction:
        raise FifoRuntimeSettlementError("runtime FIFO append requires an active transaction")

    def project() -> FifoRuntimeProjection:
        raw_connection = connection._conn  # type: ignore[attr-defined]
        return append_runtime_fill_in_transaction(raw_connection, evidence)

    return await connection._execute(project)  # type: ignore[attr-defined]

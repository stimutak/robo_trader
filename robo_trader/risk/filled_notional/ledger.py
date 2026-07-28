"""Authenticated, append-only daily gross executed-fill notional accounting.

Only actual broker executions belong here. Submitted, cancelled, rejected, and
otherwise zero-fill order events cannot be represented by :class:`ExecutedFill`.

Durability boundary
-------------------
The SQLite ledger is checked against an atomic HMAC-protected anchor file in a
different directory. The caller must obtain the 32-byte-or-longer HMAC key from
an independent secret authority; the key is never stored in the database or
anchor. The anchor binds the ledger UUID, database device/inode, fill and
quarantine heads/counts, and the latest append-only checkpoint. Each fill also
contains an authenticated cumulative count and total for its accounting scope
and New York trading date. Startup streams and audits the complete history;
later operations validate bounded checkpoint, tail, and indexed scope state.
This detects ledger rollback, replacement, tail deletion, and forged totals.

An HMAC does not prove freshness: an attacker who replays an older valid
database and matching older valid anchor can pass all local checks while the
HMAC key remains unchanged. Authoritative use therefore requires the caller's
independent monotonic verifier to reject states older than the last accepted
state. That verifier must keep its state outside the database/anchor failure
domain (for example, in an independently operated monotonic service).

The package is dormant: it grants no order, broker, runner, or startup authority.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import stat
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    Rounded,
    localcontext,
)
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, Optional
from zoneinfo import ZoneInfo

from robo_trader.safety.sqlite_identity import (
    SQLiteIdentityError,
    SQLitePathBinding,
    lexical_path_preserving_leaf,
    sqlite_connection_file_identity,
)

_NEW_YORK = ZoneInfo("America/New_York")
_ZERO_HASH = "0" * 64
_SCHEMA_VERSION = 3
_ANCHOR_VERSION = 2
_MAX_DAILY_FILL_ROWS = 100_000
_IDENTIFIER_RE = re.compile(r"[\x21-\x7e]{1,128}\Z")
_CURRENCY_RE = re.compile(r"[A-Z]{3}\Z")
_LEDGER_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")

_DENIED_SQLITE_ACTIONS = frozenset(
    action
    for name in (
        "SQLITE_ALTER_TABLE",
        "SQLITE_ANALYZE",
        "SQLITE_ATTACH",
        "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE",
        "SQLITE_CREATE_TEMP_INDEX",
        "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER",
        "SQLITE_CREATE_TEMP_VIEW",
        "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW",
        "SQLITE_CREATE_VTABLE",
        "SQLITE_DELETE",
        "SQLITE_DETACH",
        "SQLITE_DROP_INDEX",
        "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX",
        "SQLITE_DROP_TEMP_TABLE",
        "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW",
        "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW",
        "SQLITE_DROP_VTABLE",
        "SQLITE_REINDEX",
        "SQLITE_UPDATE",
    )
    if (action := getattr(sqlite3, name, None)) is not None
)

_SCHEMA_TABLE_SQL = """
CREATE TABLE daily_filled_notional_schema (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = 3),
    ledger_id TEXT NOT NULL CHECK (length(ledger_id) = 32)
)
"""

_RECORDS_TABLE_SQL = """
CREATE TABLE daily_filled_notional_records (
    sequence INTEGER PRIMARY KEY,
    account_id TEXT NOT NULL,
    portfolio_id TEXT NOT NULL,
    broker_execution_id TEXT NOT NULL,
    side TEXT NOT NULL CHECK (
        side IN ('BUY', 'SELL', 'SELL_SHORT', 'BUY_TO_COVER')
    ),
    quantity_text TEXT NOT NULL,
    price_text TEXT NOT NULL,
    currency TEXT NOT NULL,
    executed_at_utc TEXT NOT NULL,
    trading_date TEXT NOT NULL,
    notional_text TEXT NOT NULL,
    scope_fill_count INTEGER NOT NULL CHECK (scope_fill_count > 0),
    scope_total_text TEXT NOT NULL,
    previous_hash TEXT NOT NULL CHECK (length(previous_hash) = 64),
    record_hash TEXT NOT NULL CHECK (length(record_hash) = 64)
)
"""

_CONFLICTS_TABLE_SQL = """
CREATE TABLE daily_filled_notional_conflicts (
    sequence INTEGER PRIMARY KEY,
    account_id TEXT NOT NULL,
    broker_execution_id TEXT NOT NULL,
    existing_portfolio_id TEXT NOT NULL,
    claimed_portfolio_id TEXT NOT NULL,
    existing_record_hash TEXT NOT NULL CHECK (length(existing_record_hash) = 64),
    claimed_fill_json TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    previous_hash TEXT NOT NULL CHECK (length(previous_hash) = 64),
    conflict_hash TEXT NOT NULL CHECK (length(conflict_hash) = 64)
)
"""

_CHECKPOINTS_TABLE_SQL = """
CREATE TABLE daily_filled_notional_checkpoints (
    event_sequence INTEGER PRIMARY KEY,
    ledger_id TEXT NOT NULL CHECK (length(ledger_id) = 32),
    database_device INTEGER NOT NULL CHECK (database_device >= 0),
    database_inode INTEGER NOT NULL CHECK (database_inode >= 0),
    fill_count INTEGER NOT NULL CHECK (fill_count >= 0),
    fill_head TEXT NOT NULL CHECK (length(fill_head) = 64),
    conflict_count INTEGER NOT NULL CHECK (conflict_count >= 0),
    conflict_head TEXT NOT NULL CHECK (length(conflict_head) = 64),
    previous_checkpoint_hash TEXT NOT NULL CHECK (length(previous_checkpoint_hash) = 64),
    checkpoint_hash TEXT NOT NULL CHECK (length(checkpoint_hash) = 64)
)
"""

_UNIQUE_EXECUTION_SQL = """
CREATE UNIQUE INDEX daily_filled_notional_execution_identity
ON daily_filled_notional_records(account_id, broker_execution_id)
"""

_SCOPE_DATE_SQL = """
CREATE INDEX daily_filled_notional_scope_date
ON daily_filled_notional_records(
    account_id, portfolio_id, currency, trading_date, sequence DESC
)
"""

_TRIGGER_SQL = {
    "daily_filled_notional_records_insert_guard": """
        CREATE TRIGGER daily_filled_notional_records_insert_guard
        BEFORE INSERT ON daily_filled_notional_records
        WHEN NEW.sequence != COALESCE(
                 (SELECT MAX(sequence) + 1 FROM daily_filled_notional_records), 1
             )
          OR NEW.previous_hash != COALESCE(
                 (
                     SELECT record_hash
                     FROM daily_filled_notional_records
                     ORDER BY sequence DESC LIMIT 1
                 ), '0000000000000000000000000000000000000000000000000000000000000000'
             )
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional records must append to the chain');
        END
    """,
    "daily_filled_notional_records_no_update": """
        CREATE TRIGGER daily_filled_notional_records_no_update
        BEFORE UPDATE ON daily_filled_notional_records
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional records are append-only');
        END
    """,
    "daily_filled_notional_records_no_delete": """
        CREATE TRIGGER daily_filled_notional_records_no_delete
        BEFORE DELETE ON daily_filled_notional_records
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional records are append-only');
        END
    """,
    "daily_filled_notional_conflicts_insert_guard": """
        CREATE TRIGGER daily_filled_notional_conflicts_insert_guard
        BEFORE INSERT ON daily_filled_notional_conflicts
        WHEN NEW.sequence != COALESCE(
                 (SELECT MAX(sequence) + 1 FROM daily_filled_notional_conflicts), 1
             )
          OR NEW.previous_hash != COALESCE(
                 (
                     SELECT conflict_hash
                     FROM daily_filled_notional_conflicts
                     ORDER BY sequence DESC LIMIT 1
                 ), '0000000000000000000000000000000000000000000000000000000000000000'
             )
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional conflicts must append to the chain');
        END
    """,
    "daily_filled_notional_conflicts_no_update": """
        CREATE TRIGGER daily_filled_notional_conflicts_no_update
        BEFORE UPDATE ON daily_filled_notional_conflicts
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional conflicts are append-only');
        END
    """,
    "daily_filled_notional_conflicts_no_delete": """
        CREATE TRIGGER daily_filled_notional_conflicts_no_delete
        BEFORE DELETE ON daily_filled_notional_conflicts
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional conflicts are append-only');
        END
    """,
    "daily_filled_notional_schema_no_update": """
        CREATE TRIGGER daily_filled_notional_schema_no_update
        BEFORE UPDATE ON daily_filled_notional_schema
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional schema is immutable');
        END
    """,
    "daily_filled_notional_schema_no_delete": """
        CREATE TRIGGER daily_filled_notional_schema_no_delete
        BEFORE DELETE ON daily_filled_notional_schema
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional schema is immutable');
        END
    """,
    "daily_filled_notional_schema_insert_guard": """
        CREATE TRIGGER daily_filled_notional_schema_insert_guard
        BEFORE INSERT ON daily_filled_notional_schema
        WHEN EXISTS (SELECT 1 FROM daily_filled_notional_schema)
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional schema is immutable');
        END
    """,
    "daily_filled_notional_checkpoints_insert_guard": """
        CREATE TRIGGER daily_filled_notional_checkpoints_insert_guard
        BEFORE INSERT ON daily_filled_notional_checkpoints
        WHEN NEW.event_sequence != COALESCE(
                 (SELECT MAX(event_sequence) + 1 FROM daily_filled_notional_checkpoints), 0
             )
          OR NEW.previous_checkpoint_hash != COALESCE(
                 (
                     SELECT checkpoint_hash
                     FROM daily_filled_notional_checkpoints
                     ORDER BY event_sequence DESC LIMIT 1
                 ), '0000000000000000000000000000000000000000000000000000000000000000'
             )
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional checkpoints must append to the chain');
        END
    """,
    "daily_filled_notional_checkpoints_no_update": """
        CREATE TRIGGER daily_filled_notional_checkpoints_no_update
        BEFORE UPDATE ON daily_filled_notional_checkpoints
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional checkpoints are append-only');
        END
    """,
    "daily_filled_notional_checkpoints_no_delete": """
        CREATE TRIGGER daily_filled_notional_checkpoints_no_delete
        BEFORE DELETE ON daily_filled_notional_checkpoints
        BEGIN
            SELECT RAISE(ABORT, 'filled-notional checkpoints are append-only');
        END
    """,
}


class FilledNotionalError(RuntimeError):
    """Base error for the filled-notional safety boundary."""


class FilledNotionalUnavailable(FilledNotionalError):
    """The durable total cannot be established safely."""


class FilledNotionalIntegrityError(FilledNotionalUnavailable):
    """The ledger, anchor, schema, or authenticated chain is invalid."""


class FilledNotionalConflict(FilledNotionalUnavailable):
    """Durable conflicting evidence requires operator review."""


class FilledNotionalMigrationRequired(FilledNotionalUnavailable):
    """A preserved older schema requires an explicit reviewed migration."""


class _PendingAnchorInProgress(RuntimeError):
    """A valid pending transition may belong to another active writer."""


class FillSide(str, Enum):
    """Economic direction of an actual executed fill."""

    BUY = "BUY"
    SELL = "SELL"
    SELL_SHORT = "SELL_SHORT"
    BUY_TO_COVER = "BUY_TO_COVER"


@dataclass(frozen=True, slots=True)
class ExecutedFill:
    """Immutable evidence for one actual broker execution (including partials)."""

    broker_execution_id: str
    side: FillSide
    quantity: Decimal
    price: Decimal
    currency: str
    executed_at: datetime


@dataclass(frozen=True, slots=True)
class FillAccountingResult:
    recorded: bool
    trading_date: date
    fill_notional: Decimal
    gross_filled_notional: Decimal


@dataclass(frozen=True, slots=True)
class ConflictEvidence:
    sequence: int
    account_id: str
    broker_execution_id: str
    existing_portfolio_id: str
    claimed_portfolio_id: str
    claimed_fill_json: str
    observed_at_utc: str
    conflict_hash: str


@dataclass(frozen=True, slots=True)
class MonotonicLedgerState:
    """State that an independent monotonic authority must verify and advance."""

    ledger_id: str
    fill_count: int
    fill_head: str
    conflict_count: int
    conflict_head: str


@dataclass(frozen=True, slots=True)
class _CanonicalFill:
    broker_execution_id: str
    side: str
    quantity_text: str
    price_text: str
    currency: str
    executed_at_utc: str
    trading_date: str
    notional_text: str

    @property
    def notional(self) -> Decimal:
        return Decimal(self.notional_text)

    def as_dict(self) -> dict[str, str]:
        return {
            "broker_execution_id": self.broker_execution_id,
            "currency": self.currency,
            "executed_at_utc": self.executed_at_utc,
            "notional_text": self.notional_text,
            "price_text": self.price_text,
            "quantity_text": self.quantity_text,
            "side": self.side,
            "trading_date": self.trading_date,
        }


@dataclass(frozen=True, slots=True)
class _LedgerState:
    ledger_id: str
    database_device: int
    database_inode: int
    fill_count: int
    fill_head: str
    conflict_count: int
    conflict_head: str
    checkpoint_sequence: int
    checkpoint_head: str


@dataclass(frozen=True, slots=True)
class _Anchor:
    state: _LedgerState
    pending_state: Optional[_LedgerState]
    mac: str


def _normalized_sql(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    return re.sub(r"\s+", " ", value.strip().rstrip(";").lower())


def _validate_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise FilledNotionalError(
            f"{field_name} must be 1-128 non-whitespace printable ASCII characters"
        )
    return value


@contextmanager
def _isolated_decimal_context() -> Iterator[Context]:
    context = Context(prec=MAX_PREC, Emax=MAX_EMAX, Emin=MIN_EMIN, clamp=0)
    context.traps[Inexact] = True
    context.traps[Rounded] = True
    context.clear_flags()
    with localcontext(context) as active:
        yield active


def _canonical_positive_decimal(
    value: object,
    field_name: str,
    *,
    max_digits: int = 38,
    max_abs_exponent: int = 18,
) -> str:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise FilledNotionalError(f"{field_name} must be a finite positive Decimal")
    decimal_tuple = value.as_tuple()
    if (
        len(decimal_tuple.digits) > max_digits
        or not isinstance(decimal_tuple.exponent, int)
        or abs(decimal_tuple.exponent) > max_abs_exponent
    ):
        raise FilledNotionalError(f"{field_name} exceeds the exact-decimal storage bound")
    text = format(value, "f")
    if len(text) > max_digits + max_abs_exponent + 2:
        raise FilledNotionalError(f"{field_name} exceeds the exact-decimal storage bound")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _canonical_nonzero_decimal_magnitude(value: object, field_name: str) -> str:
    if type(value) is not Decimal or not value.is_finite() or value == 0:
        raise FilledNotionalError(f"{field_name} must be a finite non-zero Decimal")
    return _canonical_positive_decimal(value.copy_abs(), field_name)


def _exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    with _isolated_decimal_context() as context:
        return context.multiply(left, right)


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with _isolated_decimal_context() as context:
        return context.add(left, right)


def _parse_canonical_positive_decimal(
    value: object,
    field_name: str,
    *,
    max_digits: int = 76,
    max_abs_exponent: int = 36,
) -> Decimal:
    if type(value) is not str or not value:
        raise FilledNotionalIntegrityError(f"stored {field_name} is not canonical decimal text")
    try:
        with _isolated_decimal_context() as context:
            parsed = context.create_decimal(value)
        canonical = _canonical_positive_decimal(
            parsed,
            field_name,
            max_digits=max_digits,
            max_abs_exponent=max_abs_exponent,
        )
    except (DecimalException, FilledNotionalError) as exc:
        raise FilledNotionalIntegrityError(f"stored {field_name} is invalid") from exc
    if canonical != value:
        raise FilledNotionalIntegrityError(f"stored {field_name} is not canonical decimal text")
    return parsed


def _canonical_utc(value: object) -> tuple[str, str]:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise FilledNotionalError("timestamp must be an aware datetime")
    instant = value.astimezone(timezone.utc)
    utc_text = instant.isoformat(timespec="microseconds").replace("+00:00", "Z")
    trading_day = instant.astimezone(_NEW_YORK).date().isoformat()
    return utc_text, trading_day


def _parse_canonical_utc(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise FilledNotionalIntegrityError("stored timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise FilledNotionalIntegrityError("stored timestamp is malformed") from exc
    canonical, _ = _canonical_utc(parsed)
    if canonical != value:
        raise FilledNotionalIntegrityError("stored timestamp is not canonical UTC")
    return parsed


def _canonical_fill(fill: object) -> _CanonicalFill:
    if type(fill) is not ExecutedFill:
        raise FilledNotionalError("only exact ExecutedFill evidence can be recorded")
    execution_id = _validate_identifier(fill.broker_execution_id, "broker_execution_id")
    if type(fill.side) is not FillSide:
        raise FilledNotionalError("side must be a FillSide")
    quantity_text = _canonical_nonzero_decimal_magnitude(fill.quantity, "quantity")
    price_text = _canonical_positive_decimal(fill.price, "price")
    if type(fill.currency) is not str or _CURRENCY_RE.fullmatch(fill.currency) is None:
        raise FilledNotionalError("currency must be a three-letter uppercase code")
    executed_at_utc, trading_day = _canonical_utc(fill.executed_at)
    notional = _exact_multiply(Decimal(quantity_text), Decimal(price_text))
    notional_text = _canonical_positive_decimal(
        notional,
        "notional",
        max_digits=76,
        max_abs_exponent=36,
    )
    return _CanonicalFill(
        broker_execution_id=execution_id,
        side=fill.side.value,
        quantity_text=quantity_text,
        price_text=price_text,
        currency=fill.currency,
        executed_at_utc=executed_at_utc,
        trading_date=trading_day,
        notional_text=notional_text,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _keyed_hash(key: bytes, payload: object) -> str:
    return hmac.new(key, _canonical_json(payload).encode("ascii"), hashlib.sha256).hexdigest()


def _fill_hash(
    key: bytes,
    ledger_id: str,
    sequence: int,
    account_id: str,
    portfolio_id: str,
    fill: _CanonicalFill,
    scope_fill_count: int,
    scope_total_text: str,
    previous_hash: str,
) -> str:
    return _keyed_hash(
        key,
        {
            "account_id": account_id,
            "fill": fill.as_dict(),
            "ledger_id": ledger_id,
            "portfolio_id": portfolio_id,
            "previous_hash": previous_hash,
            "scope_fill_count": scope_fill_count,
            "scope_total_text": scope_total_text,
            "sequence": sequence,
        },
    )


def _conflict_hash(key: bytes, ledger_id: str, payload: dict[str, object]) -> str:
    return _keyed_hash(key, {"conflict": payload, "ledger_id": ledger_id})


def _checkpoint_hash(
    key: bytes,
    *,
    ledger_id: str,
    database_device: int,
    database_inode: int,
    fill_count: int,
    fill_head: str,
    conflict_count: int,
    conflict_head: str,
    event_sequence: int,
    previous_checkpoint_hash: str,
) -> str:
    return _keyed_hash(
        key,
        {
            "checkpoint": {
                "conflict_count": conflict_count,
                "conflict_head": conflict_head,
                "database_device": database_device,
                "database_inode": database_inode,
                "event_sequence": event_sequence,
                "fill_count": fill_count,
                "fill_head": fill_head,
                "ledger_id": ledger_id,
                "previous_checkpoint_hash": previous_checkpoint_hash,
            }
        },
    )


def _append_only_authorizer(
    action: int,
    argument_one: Optional[str],
    argument_two: Optional[str],
    database_name: Optional[str],
    trigger_name: Optional[str],
) -> int:
    del argument_two, database_name, trigger_name
    if action in _DENIED_SQLITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and argument_one not in {
        "integrity_check",
        "schema_version",
    }:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class DailyFilledNotional:
    """Scope-bound durable gross-filled-notional accounting."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        anchor_path: Path | str,
        anchor_key: bytes,
        monotonic_verifier: Callable[[MonotonicLedgerState], bool],
        account_id: str,
        portfolio_id: str,
        currency: str = "USD",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        _review_only: bool = False,
    ) -> None:
        self._path = lexical_path_preserving_leaf(database_path)
        self._anchor_path = lexical_path_preserving_leaf(anchor_path)
        self._key = self._validate_anchor_key(anchor_key)
        if not callable(monotonic_verifier):
            raise FilledNotionalUnavailable(
                "independent monotonic verifier is required for authoritative accounting"
            )
        self._monotonic_verifier = monotonic_verifier
        self._validate_independent_anchor_path()
        self._account_id = _validate_identifier(account_id, "account_id")
        self._portfolio_id = _validate_identifier(portfolio_id, "portfolio_id")
        if type(currency) is not str or _CURRENCY_RE.fullmatch(currency) is None:
            raise FilledNotionalError("currency must be a three-letter uppercase code")
        if not callable(clock):
            raise FilledNotionalError("clock must be callable")
        self._currency = currency
        self._clock = clock
        self._failed_reason: Optional[str] = None
        self._review_only = _review_only

        try:
            created = self._initialize_if_missing()
            if not created:
                recovery_required = self._preflight_existing_schema()
            else:
                recovery_required = False
            with self._anchor_transition_lock():
                if recovery_required:
                    self._recover_hot_journal()
                with self._connection(readonly=True) as connection:
                    connection.execute("BEGIN")
                    state = self._validate_ledger(connection)
                    if created:
                        anchor = self._create_initial_anchor(state)
                    else:
                        anchor = self._reconcile_anchor(state)
                    self._verify_monotonic_state(state)
                    self._ledger_id = state.ledger_id
                    current_day = self._current_trading_date()
                    total = self._total_for_date(connection, current_day)
                    anchor = self._require_exact_anchor(state)
                    self._verify_monotonic_state(state)
                    connection.commit()
        except FilledNotionalError:
            raise
        except DecimalException as exc:
            raise FilledNotionalUnavailable(
                "filled-notional decimal restoration failed closed"
            ) from exc
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            raise FilledNotionalUnavailable("filled-notional ledger startup failed closed") from exc

        self._ledger_id = state.ledger_id
        self._state = state
        self._anchor = anchor
        self._restored_trading_date = current_day
        self._restored_total = total
        if state.conflict_count and not _review_only:
            self._latch_failure("durable conflicting execution evidence requires review")
            raise FilledNotionalConflict(self._failed_reason)

    @staticmethod
    def _validate_anchor_key(value: object) -> bytes:
        if type(value) is not bytes or len(value) < 32:
            raise FilledNotionalError("anchor_key must be at least 32 exact bytes")
        return value

    def _validate_independent_anchor_path(self) -> None:
        if self._anchor_path == self._path:
            raise FilledNotionalError("anchor path must differ from the SQLite ledger")
        if self._anchor_path.parent == self._path.parent:
            raise FilledNotionalError("anchor must be stored in a separate protected directory")
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor: Optional[int] = None
        try:
            descriptor = os.open(self._anchor_path.parent, flags)
            metadata = os.fstat(descriptor)
            path_metadata = os.lstat(self._anchor_path.parent)
        except OSError as exc:
            raise FilledNotionalError("anchor directory must already exist") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_mode & 0o022
            or (metadata.st_dev, metadata.st_ino) != (path_metadata.st_dev, path_metadata.st_ino)
        ):
            raise FilledNotionalError("anchor directory is not safely bindable")
        self._anchor_name = self._anchor_path.name
        self._anchor_lock_name = f".{self._anchor_name}.lock"
        self._anchor_directory_device = metadata.st_dev
        self._anchor_directory_inode = metadata.st_ino

    @contextmanager
    def _anchor_directory_descriptor(self) -> Iterator[int]:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        descriptor: Optional[int] = None
        expected = (self._anchor_directory_device, self._anchor_directory_inode)
        try:
            descriptor = os.open(self._anchor_path.parent, flags)
            metadata = os.fstat(descriptor)
            path_metadata = os.lstat(self._anchor_path.parent)
            if (metadata.st_dev, metadata.st_ino) != expected or (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ) != expected:
                raise FilledNotionalIntegrityError("anchor directory identity changed")
        except FilledNotionalError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise FilledNotionalIntegrityError(
                "anchor directory cannot be accessed safely"
            ) from exc
        try:
            yield descriptor
            metadata = os.fstat(descriptor)
            path_metadata = os.lstat(self._anchor_path.parent)
            if (metadata.st_dev, metadata.st_ino) != expected or (
                path_metadata.st_dev,
                path_metadata.st_ino,
            ) != expected:
                raise FilledNotionalIntegrityError(
                    "anchor directory identity changed during operation"
                )
        finally:
            if descriptor is not None:
                os.close(descriptor)

    @contextmanager
    def _anchor_transition_lock(self) -> Iterator[None]:
        """Serialize database commits through stable-anchor publication."""

        descriptor: Optional[int] = None
        with self._anchor_directory_descriptor() as directory_descriptor:
            try:
                descriptor = os.open(
                    self._anchor_lock_name,
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_descriptor,
                )
                metadata = os.fstat(descriptor)
                path_metadata = os.stat(
                    self._anchor_lock_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_mode & 0o077
                    or metadata.st_nlink != 1
                    or (metadata.st_dev, metadata.st_ino)
                    != (path_metadata.st_dev, path_metadata.st_ino)
                ):
                    raise FilledNotionalIntegrityError(
                        "anchor transition lock is not safely bindable"
                    )
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            except FilledNotionalError:
                if descriptor is not None:
                    os.close(descriptor)
                raise
            except OSError as exc:
                if descriptor is not None:
                    os.close(descriptor)
                raise FilledNotionalIntegrityError(
                    "anchor transition lock cannot be used safely"
                ) from exc
            try:
                yield
                final_metadata = os.fstat(descriptor)
                final_path_metadata = os.stat(
                    self._anchor_lock_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if final_metadata.st_nlink != 1 or (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                ) != (final_path_metadata.st_dev, final_path_metadata.st_ino):
                    raise FilledNotionalIntegrityError("anchor transition lock identity changed")
            finally:
                if descriptor is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        os.close(descriptor)

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def anchor_path(self) -> Path:
        return self._anchor_path

    @property
    def restored_trading_date(self) -> date:
        return self._restored_trading_date

    @property
    def restored_gross_filled_notional(self) -> Decimal:
        return self._restored_total

    def record_fill(self, fill: ExecutedFill) -> FillAccountingResult:
        """Append one execution or accept an exact account-level replay."""

        self._require_available()
        try:
            canonical = _canonical_fill(fill)
            observed_at_utc, _ = _canonical_utc(self._clock())
        except DecimalException as exc:
            self._latch_failure("decimal accounting failed")
            raise FilledNotionalUnavailable("filled-notional decimal failed closed") from exc
        if canonical.currency != self._currency:
            raise FilledNotionalError("fill currency does not match the service scope")

        conflict: Optional[FilledNotionalConflict] = None
        try:
            with self._anchor_transition_lock():
                with self._connection(readonly=False) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    before = self._validate_checkpointed_state(connection)
                    anchor = self._reconcile_anchor(before)
                    self._assert_no_conflicts(before)
                    self._verify_monotonic_state(before)
                    existing = connection.execute(
                        """
                        SELECT portfolio_id, side, quantity_text, price_text, currency,
                               executed_at_utc, trading_date, notional_text, record_hash
                        FROM daily_filled_notional_records
                        WHERE account_id = ? AND broker_execution_id = ?
                        """,
                        (self._account_id, canonical.broker_execution_id),
                    ).fetchone()
                    recorded = existing is None
                    mutated = recorded
                    raw_after = before
                    if existing is not None:
                        stored = tuple(str(value) for value in existing[:8])
                        expected = (
                            self._portfolio_id,
                            canonical.side,
                            canonical.quantity_text,
                            canonical.price_text,
                            canonical.currency,
                            canonical.executed_at_utc,
                            canonical.trading_date,
                            canonical.notional_text,
                        )
                        if stored != expected:
                            conflict_head = self._append_conflict(
                                connection,
                                before,
                                canonical,
                                existing_portfolio_id=str(existing[0]),
                                existing_record_hash=str(existing[8]),
                                observed_at_utc=observed_at_utc,
                            )
                            conflict = FilledNotionalConflict(
                                "broker execution identity has durable conflicting evidence"
                            )
                            raw_after = replace(
                                before,
                                conflict_count=before.conflict_count + 1,
                                conflict_head=conflict_head,
                            )
                            mutated = True
                    else:
                        prior_count, prior_total = self._scope_total_for_date(
                            connection, date.fromisoformat(canonical.trading_date)
                        )
                        if prior_count >= _MAX_DAILY_FILL_ROWS:
                            raise FilledNotionalUnavailable(
                                "daily fill count exceeds the bounded accounting limit"
                            )
                        scope_fill_count = prior_count + 1
                        scope_total = _exact_add(prior_total, canonical.notional)
                        scope_total_text = _canonical_positive_decimal(
                            scope_total,
                            "scope_total",
                            max_digits=96,
                            max_abs_exponent=54,
                        )
                        fill_head = self._append_fill(
                            connection,
                            before,
                            canonical,
                            scope_fill_count=scope_fill_count,
                            scope_total_text=scope_total_text,
                        )
                        raw_after = replace(
                            before,
                            fill_count=before.fill_count + 1,
                            fill_head=fill_head,
                        )
                    after = (
                        self._append_checkpoint(connection, before, raw_after)
                        if mutated
                        else before
                    )
                    trading_day = date.fromisoformat(canonical.trading_date)
                    total = self._total_for_date(connection, trading_day)
                    if mutated:
                        pending = self._replace_anchor(
                            self._anchor_for_state(before, pending_state=after),
                            expected=anchor,
                        )
                        self._commit_database(connection)
                        self._anchor = self._replace_anchor(
                            self._anchor_for_state(after),
                            expected=pending,
                        )
                        self._anchor = self._require_exact_anchor(after)
                        self._verify_monotonic_state(after)
                    else:
                        self._commit_database(connection)
                        self._anchor = self._require_exact_anchor(before)
                        self._verify_monotonic_state(before)
        except FilledNotionalIntegrityError as exc:
            self._latch_failure(str(exc))
            raise
        except DecimalException as exc:
            self._latch_failure("decimal accounting failed")
            raise FilledNotionalUnavailable("filled-notional decimal failed closed") from exc
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            self._latch_failure("filled-notional ledger write failed")
            raise FilledNotionalUnavailable("filled-notional ledger write failed closed") from exc

        if conflict is not None:
            self._latch_failure(str(conflict))
            raise conflict
        return FillAccountingResult(
            recorded=recorded,
            trading_date=trading_day,
            fill_notional=canonical.notional,
            gross_filled_notional=total,
        )

    def current_gross_filled_notional(self, *, as_of: Optional[datetime] = None) -> Decimal:
        """Return a read-only exact total for the containing New York date."""

        self._require_available()
        try:
            trading_day = self._trading_date(as_of if as_of is not None else self._clock())
            pending_attempts = 0
            while True:
                try:
                    with self._connection(readonly=True) as connection:
                        connection.execute("BEGIN")
                        state = self._validate_checkpointed_state(connection)
                        self._anchor = self._require_exact_anchor(state)
                        self._assert_no_conflicts(state)
                        self._verify_monotonic_state(state)
                        total = self._total_for_date(connection, trading_day)
                        self._anchor = self._require_exact_anchor(state)
                        self._verify_monotonic_state(state)
                        connection.commit()
                        self._state = state
                        return total
                except _PendingAnchorInProgress:
                    pending_attempts += 1
                    if pending_attempts > 3:
                        raise FilledNotionalUnavailable(
                            "pending anchor writer did not reach a stable state"
                        )
                    try:
                        self._resolve_pending_anchor()
                    except sqlite3.OperationalError as exc:
                        if "locked" not in str(exc).lower():
                            raise
                        if pending_attempts == 3:
                            raise FilledNotionalUnavailable(
                                "pending anchor writer remained busy"
                            ) from exc
        except FilledNotionalIntegrityError as exc:
            self._latch_failure(str(exc))
            raise
        except DecimalException as exc:
            self._latch_failure("decimal accounting failed")
            raise FilledNotionalUnavailable("filled-notional decimal failed closed") from exc
        except FilledNotionalError:
            raise
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            self._latch_failure("filled-notional ledger read failed")
            raise FilledNotionalUnavailable("filled-notional ledger read failed closed") from exc

    @classmethod
    def review_quarantine(
        cls,
        database_path: Path | str,
        *,
        anchor_path: Path | str,
        anchor_key: bytes,
        monotonic_verifier: Callable[[MonotonicLedgerState], bool],
    ) -> tuple[ConflictEvidence, ...]:
        """Authenticate and return durable conflict markers without clearing them."""

        reviewer = cls(
            database_path,
            anchor_path=anchor_path,
            anchor_key=anchor_key,
            monotonic_verifier=monotonic_verifier,
            account_id="__review__",
            portfolio_id="__review__",
            _review_only=True,
        )
        try:
            with reviewer._connection(readonly=True) as connection:
                connection.execute("BEGIN")
                state = reviewer._validate_checkpointed_state(connection)
                reviewer._require_exact_anchor(state)
                reviewer._verify_monotonic_state(state)
                rows = connection.execute(
                    "SELECT * FROM daily_filled_notional_conflicts ORDER BY sequence"
                ).fetchall()
                evidence = tuple(
                    ConflictEvidence(
                        sequence=int(row["sequence"]),
                        account_id=str(row["account_id"]),
                        broker_execution_id=str(row["broker_execution_id"]),
                        existing_portfolio_id=str(row["existing_portfolio_id"]),
                        claimed_portfolio_id=str(row["claimed_portfolio_id"]),
                        claimed_fill_json=str(row["claimed_fill_json"]),
                        observed_at_utc=str(row["observed_at_utc"]),
                        conflict_hash=str(row["conflict_hash"]),
                    )
                    for row in rows
                )
                reviewer._require_exact_anchor(state)
                reviewer._verify_monotonic_state(state)
                connection.commit()
                return evidence
        finally:
            reviewer._latch_failure("review-only instance cannot perform accounting")

    def _append_fill(
        self,
        connection: sqlite3.Connection,
        state: _LedgerState,
        fill: _CanonicalFill,
        *,
        scope_fill_count: int,
        scope_total_text: str,
    ) -> str:
        sequence = state.fill_count + 1
        digest = _fill_hash(
            self._key,
            state.ledger_id,
            sequence,
            self._account_id,
            self._portfolio_id,
            fill,
            scope_fill_count,
            scope_total_text,
            state.fill_head,
        )
        connection.execute(
            """
            INSERT INTO daily_filled_notional_records (
                sequence, account_id, portfolio_id, broker_execution_id,
                side, quantity_text, price_text, currency, executed_at_utc,
                trading_date, notional_text, scope_fill_count, scope_total_text,
                previous_hash, record_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                self._account_id,
                self._portfolio_id,
                fill.broker_execution_id,
                fill.side,
                fill.quantity_text,
                fill.price_text,
                fill.currency,
                fill.executed_at_utc,
                fill.trading_date,
                fill.notional_text,
                scope_fill_count,
                scope_total_text,
                state.fill_head,
                digest,
            ),
        )
        return digest

    def _append_conflict(
        self,
        connection: sqlite3.Connection,
        state: _LedgerState,
        fill: _CanonicalFill,
        *,
        existing_portfolio_id: str,
        existing_record_hash: str,
        observed_at_utc: str,
    ) -> str:
        sequence = state.conflict_count + 1
        claimed_fill_json = _canonical_json(fill.as_dict())
        payload: dict[str, object] = {
            "account_id": self._account_id,
            "broker_execution_id": fill.broker_execution_id,
            "claimed_fill_json": claimed_fill_json,
            "claimed_portfolio_id": self._portfolio_id,
            "existing_portfolio_id": existing_portfolio_id,
            "existing_record_hash": existing_record_hash,
            "observed_at_utc": observed_at_utc,
            "previous_hash": state.conflict_head,
            "sequence": sequence,
        }
        digest = _conflict_hash(self._key, state.ledger_id, payload)
        connection.execute(
            """
            INSERT INTO daily_filled_notional_conflicts (
                sequence, account_id, broker_execution_id,
                existing_portfolio_id, claimed_portfolio_id,
                existing_record_hash, claimed_fill_json, observed_at_utc,
                previous_hash, conflict_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                self._account_id,
                fill.broker_execution_id,
                existing_portfolio_id,
                self._portfolio_id,
                existing_record_hash,
                claimed_fill_json,
                observed_at_utc,
                state.conflict_head,
                digest,
            ),
        )
        return digest

    def _append_checkpoint(
        self,
        connection: sqlite3.Connection,
        before: _LedgerState,
        after: _LedgerState,
    ) -> _LedgerState:
        event_sequence = before.checkpoint_sequence + 1
        if event_sequence != after.fill_count + after.conflict_count:
            raise FilledNotionalIntegrityError("checkpoint event sequence is invalid")
        digest = _checkpoint_hash(
            self._key,
            ledger_id=after.ledger_id,
            database_device=after.database_device,
            database_inode=after.database_inode,
            fill_count=after.fill_count,
            fill_head=after.fill_head,
            conflict_count=after.conflict_count,
            conflict_head=after.conflict_head,
            event_sequence=event_sequence,
            previous_checkpoint_hash=before.checkpoint_head,
        )
        connection.execute(
            """
            INSERT INTO daily_filled_notional_checkpoints (
                event_sequence, ledger_id, database_device, database_inode,
                fill_count, fill_head, conflict_count, conflict_head,
                previous_checkpoint_hash, checkpoint_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_sequence,
                after.ledger_id,
                after.database_device,
                after.database_inode,
                after.fill_count,
                after.fill_head,
                after.conflict_count,
                after.conflict_head,
                before.checkpoint_head,
                digest,
            ),
        )
        return replace(
            after,
            checkpoint_sequence=event_sequence,
            checkpoint_head=digest,
        )

    @staticmethod
    def _commit_database(connection: sqlite3.Connection) -> None:
        connection.commit()

    def _require_available(self) -> None:
        if self._review_only:
            raise FilledNotionalUnavailable("review-only instance cannot perform accounting")
        if self._failed_reason is not None:
            raise FilledNotionalUnavailable(
                f"filled-notional accounting is latched unavailable: {self._failed_reason}"
            )

    def _assert_no_conflicts(self, state: _LedgerState) -> None:
        if state.conflict_count:
            self._latch_failure("durable conflicting execution evidence requires review")
            raise FilledNotionalConflict(self._failed_reason)

    def _verify_monotonic_state(self, state: _LedgerState) -> None:
        candidate = MonotonicLedgerState(
            ledger_id=state.ledger_id,
            fill_count=state.fill_count,
            fill_head=state.fill_head,
            conflict_count=state.conflict_count,
            conflict_head=state.conflict_head,
        )
        try:
            accepted = self._monotonic_verifier(candidate)
        except Exception as exc:
            raise FilledNotionalUnavailable(
                "independent monotonic verification failed closed"
            ) from exc
        if accepted is not True:
            raise FilledNotionalIntegrityError(
                "independent monotonic authority rejected ledger state"
            )

    def _latch_failure(self, reason: str) -> None:
        self._failed_reason = reason

    def _current_trading_date(self) -> date:
        return self._trading_date(self._clock())

    @staticmethod
    def _trading_date(instant: object) -> date:
        _, trading_day = _canonical_utc(instant)
        return date.fromisoformat(trading_day)

    def _initialize_if_missing(self) -> bool:
        try:
            os.lstat(self._path)
            return False
        except FileNotFoundError:
            pass

        binding: Optional[SQLitePathBinding] = None
        connection: Optional[sqlite3.Connection] = None
        try:
            binding = SQLitePathBinding.open_for_initialization(self._path, create=True)
            connection = sqlite3.connect(
                self._path.as_uri() + "?mode=rw",
                uri=True,
                timeout=1.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(_SCHEMA_TABLE_SQL)
            connection.execute(_RECORDS_TABLE_SQL)
            connection.execute(_CONFLICTS_TABLE_SQL)
            connection.execute(_CHECKPOINTS_TABLE_SQL)
            connection.execute(_UNIQUE_EXECUTION_SQL)
            connection.execute(_SCOPE_DATE_SQL)
            for statement in _TRIGGER_SQL.values():
                connection.execute(statement)
            ledger_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO daily_filled_notional_schema VALUES (1, ?, ?)",
                (_SCHEMA_VERSION, ledger_id),
            )
            initial_checkpoint = _checkpoint_hash(
                self._key,
                ledger_id=ledger_id,
                database_device=bound.device,
                database_inode=bound.inode,
                fill_count=0,
                fill_head=_ZERO_HASH,
                conflict_count=0,
                conflict_head=_ZERO_HASH,
                event_sequence=0,
                previous_checkpoint_hash=_ZERO_HASH,
            )
            connection.execute(
                """
                INSERT INTO daily_filled_notional_checkpoints VALUES (
                    0, ?, ?, ?, 0, ?, 0, ?, ?, ?
                )
                """,
                (
                    ledger_id,
                    bound.device,
                    bound.inode,
                    _ZERO_HASH,
                    _ZERO_HASH,
                    _ZERO_HASH,
                    initial_checkpoint,
                ),
            )
            connection.commit()
            bound.assert_connection_identity(sqlite_connection_file_identity(connection))
            return True
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()
            if binding is not None:
                binding.close()

    def _recover_hot_journal(self) -> None:
        """Force identity-bound SQLite recovery before any read-only open."""

        with self._connection(readonly=False) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("PRAGMA schema_version").fetchone()
            connection.execute("SELECT count(*) FROM sqlite_master").fetchone()
            connection.commit()

    def _reject_wal_sidecars(self) -> None:
        """Reject WAL state before SQLite can open or mutate WAL/SHM artifacts."""

        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{self._path}{suffix}")
            try:
                os.lstat(sidecar)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise FilledNotionalUnavailable(
                    "SQLite sidecar identity cannot be inspected safely"
                ) from exc
            raise FilledNotionalUnavailable(
                "WAL/SHM sidecars are unsupported; preserve them for reviewed recovery"
            )

    def _validate_existing_rollback_journal(self) -> None:
        """Ensure a rollback journal cannot redirect mutations through another inode name."""

        journal = Path(f"{self._path}-journal")
        try:
            metadata = os.lstat(journal)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise FilledNotionalUnavailable(
                "SQLite rollback-journal identity cannot be inspected safely"
            ) from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise FilledNotionalUnavailable(
                "SQLite rollback journal is not an exclusive regular file"
            )
        try:
            anchor = self._load_anchor()
            database_metadata = os.lstat(self._path)
        except (FilledNotionalError, OSError) as exc:
            raise FilledNotionalUnavailable(
                "hot journal lacks an authenticated current-schema recovery anchor"
            ) from exc
        if (
            stat.S_ISLNK(database_metadata.st_mode)
            or not stat.S_ISREG(database_metadata.st_mode)
            or anchor.state.database_device != database_metadata.st_dev
            or anchor.state.database_inode != database_metadata.st_ino
        ):
            raise FilledNotionalUnavailable(
                "hot journal recovery anchor does not bind this database"
            )

    @staticmethod
    def _require_rollback_journal_header(binding: SQLitePathBinding) -> None:
        try:
            header = os.pread(binding.guardian_file_descriptor, 20, 0)
        except OSError as exc:
            raise FilledNotionalUnavailable(
                "SQLite journal mode cannot be inspected safely"
            ) from exc
        if len(header) != 20 or header[:16] != b"SQLite format 3\x00":
            raise FilledNotionalUnavailable(
                "existing schema cannot be detected without SQLite recovery"
            )
        if header[18:20] != b"\x01\x01":
            raise FilledNotionalUnavailable(
                "SQLite ledger must use rollback-journal mode; WAL is unsupported"
            )

    def _preflight_existing_schema(self) -> bool:
        """Identify legacy state without permitting SQLite journal recovery."""

        self._reject_wal_sidecars()
        journal = Path(f"{self._path}-journal")
        try:
            journal_metadata = os.lstat(journal)
        except FileNotFoundError:
            journal_metadata = None
        if journal_metadata is not None:
            if (
                stat.S_ISLNK(journal_metadata.st_mode)
                or not stat.S_ISREG(journal_metadata.st_mode)
                or journal_metadata.st_nlink != 1
            ):
                raise FilledNotionalUnavailable(
                    "SQLite journal prevents safe read-only schema detection"
                )
            try:
                anchor = self._load_anchor()
                database_metadata = os.lstat(self._path)
            except (FilledNotionalError, OSError) as exc:
                raise FilledNotionalUnavailable(
                    "hot journal lacks an authenticated current-schema recovery anchor"
                ) from exc
            if (
                anchor.state.database_device != database_metadata.st_dev
                or anchor.state.database_inode != database_metadata.st_ino
            ):
                raise FilledNotionalUnavailable(
                    "hot journal recovery anchor does not bind this database"
                )
            return True
        try:
            with self._connection(readonly=True, immutable=True) as connection:
                self._schema_version(connection)
        except FilledNotionalMigrationRequired:
            raise
        except FilledNotionalError:
            raise
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            raise FilledNotionalUnavailable(
                "existing schema cannot be detected without SQLite recovery"
            ) from exc
        return False

    @contextmanager
    def _connection(
        self,
        *,
        readonly: bool,
        immutable: bool = False,
    ) -> Iterator[sqlite3.Connection]:
        binding: Optional[SQLitePathBinding] = None
        connection: Optional[sqlite3.Connection] = None
        try:
            binding = SQLitePathBinding.open_for_initialization(self._path, create=False)
            self._require_rollback_journal_header(binding)
            self._reject_wal_sidecars()
            self._validate_existing_rollback_journal()
            mode = "ro" if readonly else "rw"
            immutable_query = "&immutable=1" if immutable else ""
            connection = sqlite3.connect(
                self._path.as_uri() + f"?mode={mode}{immutable_query}",
                uri=True,
                timeout=1.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
            journal_mode = connection.execute("PRAGMA journal_mode").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "delete":
                raise FilledNotionalUnavailable("SQLite ledger journal mode changed while opening")
            connection.execute("PRAGMA busy_timeout=1000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.set_authorizer(_append_only_authorizer)
            bound.assert_connection_identity(sqlite_connection_file_identity(connection))
            yield connection
            bound.assert_connection_identity(sqlite_connection_file_identity(connection))
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()
            if binding is not None:
                binding.close()

    def _schema_version(self, connection: sqlite3.Connection) -> int:
        table = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'daily_filled_notional_schema'"
        ).fetchone()
        if table is None:
            raise FilledNotionalIntegrityError("filled-notional schema table is missing")
        try:
            rows = connection.execute(
                "SELECT schema_version FROM daily_filled_notional_schema"
            ).fetchall()
        except sqlite3.Error as exc:
            raise FilledNotionalIntegrityError(
                "filled-notional schema version is unreadable"
            ) from exc
        if len(rows) != 1 or type(rows[0][0]) is not int:
            raise FilledNotionalIntegrityError("filled-notional schema version is invalid")
        version = int(rows[0][0])
        if version < _SCHEMA_VERSION:
            raise FilledNotionalMigrationRequired(
                f"filled-notional schema v{version} requires reviewed copy migration to v3"
            )
        if version > _SCHEMA_VERSION:
            raise FilledNotionalIntegrityError(
                f"unsupported future filled-notional schema v{version}"
            )
        return version

    def _validate_schema(self, connection: sqlite3.Connection) -> str:
        self._schema_version(connection)
        expected_sql = {
            ("table", "daily_filled_notional_schema"): _SCHEMA_TABLE_SQL,
            ("table", "daily_filled_notional_records"): _RECORDS_TABLE_SQL,
            ("table", "daily_filled_notional_conflicts"): _CONFLICTS_TABLE_SQL,
            ("table", "daily_filled_notional_checkpoints"): _CHECKPOINTS_TABLE_SQL,
            ("index", "daily_filled_notional_execution_identity"): _UNIQUE_EXECUTION_SQL,
            ("index", "daily_filled_notional_scope_date"): _SCOPE_DATE_SQL,
            **{("trigger", name): sql for name, sql in _TRIGGER_SQL.items()},
        }
        rows = connection.execute("""
            SELECT type, name, sql FROM sqlite_master
            WHERE name LIKE 'daily_filled_notional_%'
            ORDER BY type, name
            """).fetchall()
        actual_sql = {(str(row[0]), str(row[1])): str(row[2]) for row in rows}
        if set(actual_sql) != set(expected_sql):
            raise FilledNotionalIntegrityError("filled-notional schema objects do not match")
        for key, statement in expected_sql.items():
            if _normalized_sql(actual_sql[key]) != _normalized_sql(statement):
                raise FilledNotionalIntegrityError(
                    f"filled-notional schema definition changed: {key[1]}"
                )
        schema_rows = connection.execute(
            "SELECT singleton, schema_version, ledger_id FROM daily_filled_notional_schema"
        ).fetchall()
        if len(schema_rows) != 1 or tuple(schema_rows[0][:2]) != (1, _SCHEMA_VERSION):
            raise FilledNotionalIntegrityError("filled-notional schema singleton is invalid")
        ledger_id = str(schema_rows[0][2])
        if _LEDGER_ID_RE.fullmatch(ledger_id) is None:
            raise FilledNotionalIntegrityError("filled-notional ledger identity is invalid")
        return ledger_id

    def _validate_ledger(self, connection: sqlite3.Connection) -> _LedgerState:
        integrity = connection.execute("PRAGMA integrity_check")
        first_integrity = integrity.fetchone()
        if (
            first_integrity is None
            or str(first_integrity[0]) != "ok"
            or integrity.fetchone() is not None
        ):
            raise FilledNotionalIntegrityError("SQLite integrity_check failed")
        ledger_id = self._validate_schema(connection)

        fill_count, fill_head = self._validate_fills(connection, ledger_id)
        conflict_count, conflict_head = self._validate_conflicts(connection, ledger_id)
        metadata = os.lstat(self._path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FilledNotionalIntegrityError("ledger path identity is invalid")
        checkpoint_sequence, checkpoint_head = self._validate_checkpoints(
            connection,
            ledger_id=ledger_id,
            database_device=metadata.st_dev,
            database_inode=metadata.st_ino,
            fill_count=fill_count,
            fill_head=fill_head,
            conflict_count=conflict_count,
            conflict_head=conflict_head,
        )
        return _LedgerState(
            ledger_id=ledger_id,
            database_device=metadata.st_dev,
            database_inode=metadata.st_ino,
            fill_count=fill_count,
            fill_head=fill_head,
            conflict_count=conflict_count,
            conflict_head=conflict_head,
            checkpoint_sequence=checkpoint_sequence,
            checkpoint_head=checkpoint_head,
        )

    def _validate_checkpointed_state(self, connection: sqlite3.Connection) -> _LedgerState:
        """Validate bounded authenticated state after the startup full audit."""

        ledger_id = self._validate_schema(connection)
        row = connection.execute(
            "SELECT * FROM daily_filled_notional_checkpoints "
            "ORDER BY event_sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise FilledNotionalIntegrityError("checkpoint state is missing")
        integer_names = (
            "event_sequence",
            "database_device",
            "database_inode",
            "fill_count",
            "conflict_count",
        )
        if any(type(row[name]) is not int or row[name] < 0 for name in integer_names):
            raise FilledNotionalIntegrityError("checkpoint counters are invalid")
        checkpoint_ledger_id = str(row["ledger_id"])
        database_device = int(row["database_device"])
        database_inode = int(row["database_inode"])
        fill_count = int(row["fill_count"])
        fill_head = str(row["fill_head"])
        conflict_count = int(row["conflict_count"])
        conflict_head = str(row["conflict_head"])
        event_sequence = int(row["event_sequence"])
        previous_checkpoint_hash = str(row["previous_checkpoint_hash"])
        if (
            checkpoint_ledger_id != ledger_id
            or event_sequence != fill_count + conflict_count
            or conflict_count > 1
            or _HASH_RE.fullmatch(fill_head) is None
            or _HASH_RE.fullmatch(conflict_head) is None
            or _HASH_RE.fullmatch(previous_checkpoint_hash) is None
        ):
            raise FilledNotionalIntegrityError("checkpoint state is inconsistent")
        expected_hash = _checkpoint_hash(
            self._key,
            ledger_id=checkpoint_ledger_id,
            database_device=database_device,
            database_inode=database_inode,
            fill_count=fill_count,
            fill_head=fill_head,
            conflict_count=conflict_count,
            conflict_head=conflict_head,
            event_sequence=event_sequence,
            previous_checkpoint_hash=previous_checkpoint_hash,
        )
        checkpoint_head = str(row["checkpoint_hash"])
        if not hmac.compare_digest(checkpoint_head, expected_hash):
            raise FilledNotionalIntegrityError("checkpoint HMAC is invalid")
        metadata = os.lstat(self._path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != (database_device, database_inode)
        ):
            raise FilledNotionalIntegrityError("checkpoint database identity changed")
        fill_tail = connection.execute(
            "SELECT sequence, record_hash FROM daily_filled_notional_records "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        conflict_tail = connection.execute(
            "SELECT sequence, conflict_hash FROM daily_filled_notional_conflicts "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if fill_count == 0:
            fill_matches = fill_tail is None and fill_head == _ZERO_HASH
        else:
            fill_matches = (
                fill_tail is not None
                and type(fill_tail[0]) is int
                and int(fill_tail[0]) == fill_count
                and str(fill_tail[1]) == fill_head
            )
        if conflict_count == 0:
            conflict_matches = conflict_tail is None and conflict_head == _ZERO_HASH
        else:
            conflict_matches = (
                conflict_tail is not None
                and type(conflict_tail[0]) is int
                and int(conflict_tail[0]) == conflict_count
                and str(conflict_tail[1]) == conflict_head
            )
        if not fill_matches or not conflict_matches:
            raise FilledNotionalIntegrityError("checkpoint tails do not match ledger rows")
        return _LedgerState(
            ledger_id=ledger_id,
            database_device=database_device,
            database_inode=database_inode,
            fill_count=fill_count,
            fill_head=fill_head,
            conflict_count=conflict_count,
            conflict_head=conflict_head,
            checkpoint_sequence=event_sequence,
            checkpoint_head=checkpoint_head,
        )

    def _validate_fills(self, connection: sqlite3.Connection, ledger_id: str) -> tuple[int, str]:
        previous_hash = _ZERO_HASH
        expected_sequence = 1
        scope_states: dict[tuple[str, str, str, str], tuple[int, Decimal]] = {}
        for row in connection.execute(
            "SELECT * FROM daily_filled_notional_records ORDER BY sequence"
        ):
            if int(row["sequence"]) != expected_sequence:
                raise FilledNotionalIntegrityError("filled-notional sequence is not contiguous")
            account_id = _validate_stored_identifier(row["account_id"], "account_id")
            portfolio_id = _validate_stored_identifier(row["portfolio_id"], "portfolio_id")
            execution_id = _validate_stored_identifier(
                row["broker_execution_id"], "broker_execution_id"
            )
            side = str(row["side"])
            if side not in {candidate.value for candidate in FillSide}:
                raise FilledNotionalIntegrityError("stored fill side is invalid")
            quantity = _parse_canonical_positive_decimal(
                row["quantity_text"], "quantity", max_digits=38, max_abs_exponent=18
            )
            price = _parse_canonical_positive_decimal(
                row["price_text"], "price", max_digits=38, max_abs_exponent=18
            )
            notional = _parse_canonical_positive_decimal(row["notional_text"], "notional")
            currency = str(row["currency"])
            if _CURRENCY_RE.fullmatch(currency) is None:
                raise FilledNotionalIntegrityError("stored currency is invalid")
            executed_at = _parse_canonical_utc(row["executed_at_utc"])
            _, derived_day = _canonical_utc(executed_at)
            if str(row["trading_date"]) != derived_day:
                raise FilledNotionalIntegrityError("stored New York trading date is invalid")
            if _exact_multiply(quantity, price) != notional:
                raise FilledNotionalIntegrityError("stored fill notional is inconsistent")
            if str(row["previous_hash"]) != previous_hash:
                raise FilledNotionalIntegrityError("filled-notional hash chain is broken")
            canonical = _CanonicalFill(
                broker_execution_id=execution_id,
                side=side,
                quantity_text=str(row["quantity_text"]),
                price_text=str(row["price_text"]),
                currency=currency,
                executed_at_utc=str(row["executed_at_utc"]),
                trading_date=str(row["trading_date"]),
                notional_text=str(row["notional_text"]),
            )
            scope_key = (account_id, portfolio_id, currency, str(row["trading_date"]))
            prior_scope_count, prior_scope_total = scope_states.get(scope_key, (0, Decimal("0")))
            scope_fill_count = int(row["scope_fill_count"])
            if (
                type(row["scope_fill_count"]) is not int
                or scope_fill_count != prior_scope_count + 1
                or scope_fill_count > _MAX_DAILY_FILL_ROWS
            ):
                raise FilledNotionalIntegrityError("stored scope fill count is invalid")
            scope_total = _parse_canonical_positive_decimal(
                row["scope_total_text"],
                "scope total",
                max_digits=96,
                max_abs_exponent=54,
            )
            expected_scope_total = _exact_add(prior_scope_total, notional)
            if scope_total != expected_scope_total:
                raise FilledNotionalIntegrityError("stored scope total is inconsistent")
            expected_hash = _fill_hash(
                self._key,
                ledger_id,
                expected_sequence,
                account_id,
                portfolio_id,
                canonical,
                scope_fill_count,
                str(row["scope_total_text"]),
                previous_hash,
            )
            stored_hash = str(row["record_hash"])
            if not hmac.compare_digest(stored_hash, expected_hash):
                raise FilledNotionalIntegrityError("filled-notional record HMAC is invalid")
            previous_hash = stored_hash
            scope_states[scope_key] = (scope_fill_count, scope_total)
            expected_sequence += 1
        return expected_sequence - 1, previous_hash

    def _validate_conflicts(
        self, connection: sqlite3.Connection, ledger_id: str
    ) -> tuple[int, str]:
        previous_hash = _ZERO_HASH
        expected_sequence = 1
        for row in connection.execute(
            "SELECT * FROM daily_filled_notional_conflicts ORDER BY sequence"
        ):
            if int(row["sequence"]) != expected_sequence:
                raise FilledNotionalIntegrityError("conflict sequence is not contiguous")
            payload: dict[str, object] = {
                "account_id": _validate_stored_identifier(row["account_id"], "account_id"),
                "broker_execution_id": _validate_stored_identifier(
                    row["broker_execution_id"], "broker_execution_id"
                ),
                "claimed_fill_json": str(row["claimed_fill_json"]),
                "claimed_portfolio_id": _validate_stored_identifier(
                    row["claimed_portfolio_id"], "claimed_portfolio_id"
                ),
                "existing_portfolio_id": _validate_stored_identifier(
                    row["existing_portfolio_id"], "existing_portfolio_id"
                ),
                "existing_record_hash": str(row["existing_record_hash"]),
                "observed_at_utc": str(row["observed_at_utc"]),
                "previous_hash": str(row["previous_hash"]),
                "sequence": expected_sequence,
            }
            if _HASH_RE.fullmatch(str(payload["existing_record_hash"])) is None:
                raise FilledNotionalIntegrityError("conflict existing record hash is invalid")
            try:
                claimed = json.loads(str(payload["claimed_fill_json"]))
            except (TypeError, json.JSONDecodeError) as exc:
                raise FilledNotionalIntegrityError("conflict claimed fill is malformed") from exc
            if _canonical_json(claimed) != payload["claimed_fill_json"]:
                raise FilledNotionalIntegrityError("conflict claimed fill is not canonical")
            _parse_canonical_utc(payload["observed_at_utc"])
            if payload["previous_hash"] != previous_hash:
                raise FilledNotionalIntegrityError("conflict hash chain is broken")
            expected_hash = _conflict_hash(self._key, ledger_id, payload)
            stored_hash = str(row["conflict_hash"])
            if not hmac.compare_digest(stored_hash, expected_hash):
                raise FilledNotionalIntegrityError("conflict record HMAC is invalid")
            previous_hash = stored_hash
            expected_sequence += 1
        return expected_sequence - 1, previous_hash

    def _validate_checkpoints(
        self,
        connection: sqlite3.Connection,
        *,
        ledger_id: str,
        database_device: int,
        database_inode: int,
        fill_count: int,
        fill_head: str,
        conflict_count: int,
        conflict_head: str,
    ) -> tuple[int, str]:
        expected_sequence = 0
        previous_hash = _ZERO_HASH
        final_row: Optional[sqlite3.Row] = None
        for row in connection.execute(
            "SELECT * FROM daily_filled_notional_checkpoints ORDER BY event_sequence"
        ):
            if int(row["event_sequence"]) != expected_sequence:
                raise FilledNotionalIntegrityError("checkpoint sequence is not contiguous")
            if str(row["ledger_id"]) != ledger_id:
                raise FilledNotionalIntegrityError("checkpoint ledger identity changed")
            checkpoint_database_device = int(row["database_device"])
            checkpoint_database_inode = int(row["database_inode"])
            checkpoint_fill_count = int(row["fill_count"])
            checkpoint_fill_head = str(row["fill_head"])
            checkpoint_conflict_count = int(row["conflict_count"])
            checkpoint_conflict_head = str(row["conflict_head"])
            if str(row["previous_checkpoint_hash"]) != previous_hash:
                raise FilledNotionalIntegrityError("checkpoint hash chain is broken")
            expected_hash = _checkpoint_hash(
                self._key,
                ledger_id=ledger_id,
                database_device=checkpoint_database_device,
                database_inode=checkpoint_database_inode,
                fill_count=checkpoint_fill_count,
                fill_head=checkpoint_fill_head,
                conflict_count=checkpoint_conflict_count,
                conflict_head=checkpoint_conflict_head,
                event_sequence=expected_sequence,
                previous_checkpoint_hash=previous_hash,
            )
            stored_hash = str(row["checkpoint_hash"])
            if not hmac.compare_digest(stored_hash, expected_hash):
                raise FilledNotionalIntegrityError("checkpoint HMAC is invalid")
            previous_hash = stored_hash
            expected_sequence += 1
            final_row = row
        if final_row is None:
            raise FilledNotionalIntegrityError("initial checkpoint is missing")
        final_values = (
            int(final_row["database_device"]),
            int(final_row["database_inode"]),
            int(final_row["fill_count"]),
            str(final_row["fill_head"]),
            int(final_row["conflict_count"]),
            str(final_row["conflict_head"]),
        )
        if final_values != (
            database_device,
            database_inode,
            fill_count,
            fill_head,
            conflict_count,
            conflict_head,
        ):
            raise FilledNotionalIntegrityError(
                "checkpoint does not match audited ledger state; rollback, tail deletion, "
                "or tamper detected"
            )
        if expected_sequence - 1 != fill_count + conflict_count:
            raise FilledNotionalIntegrityError("checkpoint event count is inconsistent")
        return expected_sequence - 1, previous_hash

    @staticmethod
    def _state_payload(state: _LedgerState) -> dict[str, object]:
        return {
            "checkpoint_head": state.checkpoint_head,
            "checkpoint_sequence": state.checkpoint_sequence,
            "conflict_count": state.conflict_count,
            "conflict_head": state.conflict_head,
            "database_device": state.database_device,
            "database_inode": state.database_inode,
            "fill_count": state.fill_count,
            "fill_head": state.fill_head,
            "ledger_id": state.ledger_id,
        }

    def _unsigned_anchor_payload(
        self,
        state: _LedgerState,
        pending_state: Optional[_LedgerState],
    ) -> dict[str, object]:
        return {
            "anchor_version": _ANCHOR_VERSION,
            "pending_state": (
                None if pending_state is None else self._state_payload(pending_state)
            ),
            "state": self._state_payload(state),
        }

    def _anchor_for_state(
        self,
        state: _LedgerState,
        *,
        pending_state: Optional[_LedgerState] = None,
    ) -> _Anchor:
        unsigned = self._unsigned_anchor_payload(state, pending_state)
        return _Anchor(
            state=state,
            pending_state=pending_state,
            mac=_keyed_hash(self._key, unsigned),
        )

    def _create_initial_anchor(self, state: _LedgerState) -> _Anchor:
        with self._anchor_directory_descriptor() as directory_descriptor:
            try:
                os.stat(
                    self._anchor_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                anchor = self._anchor_for_state(state)
                return self._write_anchor_file(anchor, must_not_exist=True)
        raise FilledNotionalIntegrityError("new ledger anchor path already exists")

    def _load_anchor(self) -> _Anchor:
        descriptor: Optional[int] = None
        try:
            with self._anchor_directory_descriptor() as directory_descriptor:
                metadata = os.stat(
                    self._anchor_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise FilledNotionalIntegrityError("anchor must be a non-symlink regular file")
                if metadata.st_mode & 0o077:
                    raise FilledNotionalIntegrityError("anchor permissions must be owner-only")
                descriptor = os.open(
                    self._anchor_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=directory_descriptor,
                )
                opened_metadata = os.fstat(descriptor)
                if (metadata.st_dev, metadata.st_ino) != (
                    opened_metadata.st_dev,
                    opened_metadata.st_ino,
                ):
                    raise FilledNotionalIntegrityError("anchor identity changed while opening")
                chunks: list[bytes] = []
                size = 0
                while True:
                    chunk = os.read(descriptor, 8192)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > 65536:
                        raise FilledNotionalIntegrityError("anchor file is too large")
                    chunks.append(chunk)
                final_metadata = os.stat(
                    self._anchor_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (metadata.st_dev, metadata.st_ino) != (
                    final_metadata.st_dev,
                    final_metadata.st_ino,
                ):
                    raise FilledNotionalIntegrityError("anchor identity changed while reading")
                raw = b"".join(chunks)
            decoded = json.loads(raw.decode("ascii"))
        except FilledNotionalError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FilledNotionalIntegrityError("anchor cannot be read safely") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if type(decoded) is not dict or set(decoded) != {
            "anchor_version",
            "mac",
            "pending_state",
            "state",
        }:
            raise FilledNotionalIntegrityError("anchor shape is invalid")
        unsigned = {key: decoded[key] for key in decoded if key != "mac"}
        if unsigned["anchor_version"] != _ANCHOR_VERSION:
            raise FilledNotionalIntegrityError("anchor version is unsupported")
        mac = str(decoded["mac"])
        expected_mac = _keyed_hash(self._key, unsigned)
        if _HASH_RE.fullmatch(mac) is None or not hmac.compare_digest(mac, expected_mac):
            raise FilledNotionalIntegrityError("anchor HMAC is invalid")
        state = self._parse_anchor_state(unsigned["state"])
        pending_value = unsigned["pending_state"]
        pending_state = None if pending_value is None else self._parse_anchor_state(pending_value)
        return _Anchor(state=state, pending_state=pending_state, mac=mac)

    @staticmethod
    def _parse_anchor_state(value: object) -> _LedgerState:
        names = {
            "checkpoint_head",
            "checkpoint_sequence",
            "conflict_count",
            "conflict_head",
            "database_device",
            "database_inode",
            "fill_count",
            "fill_head",
            "ledger_id",
        }
        if type(value) is not dict or set(value) != names:
            raise FilledNotionalIntegrityError("anchor state shape is invalid")
        if any(
            type(value[name]) is not int or value[name] < 0
            for name in (
                "checkpoint_sequence",
                "conflict_count",
                "database_device",
                "database_inode",
                "fill_count",
            )
        ):
            raise FilledNotionalIntegrityError("anchor counters or identity are invalid")
        if _LEDGER_ID_RE.fullmatch(str(value["ledger_id"])) is None or any(
            _HASH_RE.fullmatch(str(value[name])) is None
            for name in ("checkpoint_head", "fill_head", "conflict_head")
        ):
            raise FilledNotionalIntegrityError("anchor identifiers are invalid")
        return _LedgerState(
            ledger_id=str(value["ledger_id"]),
            database_device=int(value["database_device"]),
            database_inode=int(value["database_inode"]),
            fill_count=int(value["fill_count"]),
            fill_head=str(value["fill_head"]),
            conflict_count=int(value["conflict_count"]),
            conflict_head=str(value["conflict_head"]),
            checkpoint_sequence=int(value["checkpoint_sequence"]),
            checkpoint_head=str(value["checkpoint_head"]),
        )

    def _reconcile_anchor(self, state: _LedgerState) -> _Anchor:
        anchor = self._load_anchor()
        anchored = anchor.state
        if (
            anchored.ledger_id != state.ledger_id
            or anchored.database_device != state.database_device
            or anchored.database_inode != state.database_inode
        ):
            raise FilledNotionalIntegrityError("anchor does not bind this ledger identity")
        pending = anchor.pending_state
        if pending is None and anchored == state:
            return anchor
        if pending is None:
            raise FilledNotionalIntegrityError(
                "ledger rollback, tail deletion, or stable-anchor rollback detected"
            )
        if (
            pending.ledger_id != anchored.ledger_id
            or pending.database_device != anchored.database_device
            or pending.database_inode != anchored.database_inode
        ):
            raise FilledNotionalIntegrityError("pending anchor identity is invalid")
        if state == anchored:
            return self._replace_anchor(self._anchor_for_state(anchored), expected=anchor)
        if state == pending:
            return self._replace_anchor(self._anchor_for_state(pending), expected=anchor)
        raise FilledNotionalIntegrityError("database does not match either atomic anchor state")

    def _require_exact_anchor(self, state: _LedgerState) -> _Anchor:
        anchor = self._load_anchor()
        pending = anchor.pending_state
        if pending is not None:
            if (
                pending.ledger_id != anchor.state.ledger_id
                or pending.database_device != anchor.state.database_device
                or pending.database_inode != anchor.state.database_inode
                or state not in (anchor.state, pending)
            ):
                raise FilledNotionalIntegrityError(
                    "pending anchor does not authenticate current ledger state"
                )
            raise _PendingAnchorInProgress
        if anchor.state != state:
            raise FilledNotionalIntegrityError("ledger and durable anchor differ")
        return anchor

    def _resolve_pending_anchor(self) -> None:
        """Serialize behind a writer and reconcile only authenticated endpoints."""

        with self._anchor_transition_lock():
            with self._connection(readonly=False) as connection:
                connection.execute("BEGIN IMMEDIATE")
                state = self._validate_ledger(connection)
                self._anchor = self._reconcile_anchor(state)
                connection.commit()

    def _replace_anchor(self, desired: _Anchor, *, expected: _Anchor) -> _Anchor:
        current = self._load_anchor()
        if current == desired:
            return current
        if current != expected:
            raise FilledNotionalIntegrityError("anchor changed concurrently")
        return self._write_anchor_file(desired, must_not_exist=False)

    def _write_anchor_file(self, anchor: _Anchor, *, must_not_exist: bool) -> _Anchor:
        payload = self._unsigned_anchor_payload(anchor.state, anchor.pending_state)
        payload["mac"] = anchor.mac
        encoded = (_canonical_json(payload) + "\n").encode("ascii")
        temporary_name = f".{self._anchor_name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        descriptor: Optional[int] = None
        with self._anchor_directory_descriptor() as directory_descriptor:
            try:
                if must_not_exist:
                    try:
                        os.stat(
                            self._anchor_name,
                            dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    except FileNotFoundError:
                        pass
                    else:
                        raise FilledNotionalIntegrityError("anchor already exists")
                descriptor = os.open(
                    temporary_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=directory_descriptor,
                )
                view = memoryview(encoded)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("anchor write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                if must_not_exist:
                    try:
                        os.link(
                            temporary_name,
                            self._anchor_name,
                            src_dir_fd=directory_descriptor,
                            dst_dir_fd=directory_descriptor,
                            follow_symlinks=False,
                        )
                    except FileExistsError as exc:
                        raise FilledNotionalIntegrityError("anchor already exists") from exc
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
                else:
                    os.replace(
                        temporary_name,
                        self._anchor_name,
                        src_dir_fd=directory_descriptor,
                        dst_dir_fd=directory_descriptor,
                    )
                os.fsync(directory_descriptor)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                try:
                    os.unlink(temporary_name, dir_fd=directory_descriptor)
                except FileNotFoundError:
                    pass
        loaded = self._load_anchor()
        if loaded != anchor:
            raise FilledNotionalIntegrityError("atomic anchor verification failed")
        return loaded

    def _scope_total_for_date(
        self, connection: sqlite3.Connection, trading_day: date
    ) -> tuple[int, Decimal]:
        row = connection.execute(
            """
            SELECT * FROM daily_filled_notional_records
            WHERE account_id = ? AND portfolio_id = ?
              AND currency = ? AND trading_date = ?
            ORDER BY sequence DESC LIMIT 1
            """,
            (
                self._account_id,
                self._portfolio_id,
                self._currency,
                trading_day.isoformat(),
            ),
        ).fetchone()
        if row is None:
            return 0, Decimal("0")
        sequence = int(row["sequence"])
        scope_fill_count = int(row["scope_fill_count"])
        if (
            type(row["sequence"]) is not int
            or type(row["scope_fill_count"]) is not int
            or sequence < 1
            or scope_fill_count < 1
            or scope_fill_count > _MAX_DAILY_FILL_ROWS
        ):
            raise FilledNotionalIntegrityError("stored scope fill count is invalid")
        account_id = _validate_stored_identifier(row["account_id"], "account_id")
        portfolio_id = _validate_stored_identifier(row["portfolio_id"], "portfolio_id")
        execution_id = _validate_stored_identifier(
            row["broker_execution_id"], "broker_execution_id"
        )
        canonical = _CanonicalFill(
            broker_execution_id=execution_id,
            side=str(row["side"]),
            quantity_text=str(row["quantity_text"]),
            price_text=str(row["price_text"]),
            currency=str(row["currency"]),
            executed_at_utc=str(row["executed_at_utc"]),
            trading_date=str(row["trading_date"]),
            notional_text=str(row["notional_text"]),
        )
        scope_total_text = str(row["scope_total_text"])
        total = _parse_canonical_positive_decimal(
            scope_total_text,
            "scope total",
            max_digits=96,
            max_abs_exponent=54,
        )
        expected_hash = _fill_hash(
            self._key,
            self._ledger_id,
            sequence,
            account_id,
            portfolio_id,
            canonical,
            scope_fill_count,
            scope_total_text,
            str(row["previous_hash"]),
        )
        if not hmac.compare_digest(str(row["record_hash"]), expected_hash):
            raise FilledNotionalIntegrityError("scope-total record HMAC is invalid")
        return scope_fill_count, total

    def _total_for_date(self, connection: sqlite3.Connection, trading_day: date) -> Decimal:
        return self._scope_total_for_date(connection, trading_day)[1]


def _validate_stored_identifier(value: object, field_name: str) -> str:
    try:
        return _validate_identifier(value, field_name)
    except FilledNotionalError as exc:
        raise FilledNotionalIntegrityError(f"stored {field_name} is invalid") from exc

"""Append-only accounting for daily gross executed-fill notional.

Only broker execution evidence belongs in this ledger.  An order being
submitted, accepted, cancelled, or rejected is not an execution and cannot be
represented by :class:`ExecutedFill`.

The service is bound to one broker account and portfolio.  Each partial fill is
recorded independently under its immutable broker execution identifier.  Exact
replays are idempotent; a reuse of that identifier with different evidence is a
fatal conflict.  Monetary arithmetic is performed entirely with ``Decimal``
and persisted as canonical decimal text.

This module grants no order or process-start authority and is not wired into
the trading runner.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import MAX_PREC, Decimal, Inexact, Rounded, localcontext
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
_SCHEMA_VERSION = 1
_IDENTIFIER_RE = re.compile(r"[\x21-\x7e]{1,128}\Z")
_CURRENCY_RE = re.compile(r"[A-Z]{3}\Z")
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
    schema_version INTEGER NOT NULL CHECK (schema_version = 1)
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
    previous_hash TEXT NOT NULL CHECK (length(previous_hash) = 64),
    record_hash TEXT NOT NULL CHECK (length(record_hash) = 64)
)
"""

_UNIQUE_EXECUTION_SQL = """
CREATE UNIQUE INDEX daily_filled_notional_execution_identity
ON daily_filled_notional_records(account_id, portfolio_id, broker_execution_id)
"""

_SCOPE_DATE_SQL = """
CREATE INDEX daily_filled_notional_scope_date
ON daily_filled_notional_records(account_id, portfolio_id, currency, trading_date)
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
}


class FilledNotionalError(RuntimeError):
    """Base error for the filled-notional safety boundary."""


class FilledNotionalUnavailable(FilledNotionalError):
    """The durable total cannot be established safely."""


class FilledNotionalIntegrityError(FilledNotionalUnavailable):
    """The ledger schema or immutable record chain is invalid."""


class FilledNotionalConflict(FilledNotionalUnavailable):
    """One immutable execution identifier has conflicting evidence."""


class FillSide(str, Enum):
    """Economic direction of an executed fill.

    Direction never changes gross-notional arithmetic: buys, sells, shorts,
    and covers all add the absolute executed quantity multiplied by price.
    """

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
    """Result of appending or replaying one execution."""

    recorded: bool
    trading_date: date
    fill_notional: Decimal
    gross_filled_notional: Decimal


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
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical_nonzero_decimal_magnitude(value: object, field_name: str) -> str:
    if type(value) is not Decimal or not value.is_finite() or value == 0:
        raise FilledNotionalError(f"{field_name} must be a finite non-zero Decimal")
    return _canonical_positive_decimal(value.copy_abs(), field_name)


def _exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = MAX_PREC
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        return left * right


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
        parsed = Decimal(value)
    except Exception as exc:
        raise FilledNotionalIntegrityError(f"stored {field_name} is malformed") from exc
    try:
        canonical = _canonical_positive_decimal(
            parsed,
            field_name,
            max_digits=max_digits,
            max_abs_exponent=max_abs_exponent,
        )
    except FilledNotionalError as exc:
        raise FilledNotionalIntegrityError(f"stored {field_name} is invalid") from exc
    if canonical != value:
        raise FilledNotionalIntegrityError(f"stored {field_name} is not canonical decimal text")
    return parsed


def _canonical_utc(value: object) -> tuple[str, str]:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise FilledNotionalError("executed_at must be an aware datetime")
    instant = value.astimezone(timezone.utc)
    utc_text = instant.isoformat(timespec="microseconds").replace("+00:00", "Z")
    trading_day = instant.astimezone(_NEW_YORK).date().isoformat()
    return utc_text, trading_day


def _parse_canonical_utc(value: object) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise FilledNotionalIntegrityError("stored execution timestamp is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise FilledNotionalIntegrityError("stored execution timestamp is malformed") from exc
    canonical, _ = _canonical_utc(parsed)
    if canonical != value:
        raise FilledNotionalIntegrityError("stored execution timestamp is not canonical UTC")
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


def _record_hash(
    sequence: int,
    account_id: str,
    portfolio_id: str,
    fill: _CanonicalFill,
    previous_hash: str,
) -> str:
    payload = {
        "account_id": account_id,
        "broker_execution_id": fill.broker_execution_id,
        "currency": fill.currency,
        "executed_at_utc": fill.executed_at_utc,
        "notional_text": fill.notional_text,
        "portfolio_id": portfolio_id,
        "previous_hash": previous_hash,
        "price_text": fill.price_text,
        "quantity_text": fill.quantity_text,
        "sequence": sequence,
        "side": fill.side,
        "trading_date": fill.trading_date,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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
    if action == sqlite3.SQLITE_PRAGMA and argument_one not in {"integrity_check"}:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


class DailyFilledNotional:
    """Scope-bound, durable accounting for gross executed-fill notional."""

    def __init__(
        self,
        database_path: Path | str,
        *,
        account_id: str,
        portfolio_id: str,
        currency: str = "USD",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._path = lexical_path_preserving_leaf(database_path)
        self._account_id = _validate_identifier(account_id, "account_id")
        self._portfolio_id = _validate_identifier(portfolio_id, "portfolio_id")
        if type(currency) is not str or _CURRENCY_RE.fullmatch(currency) is None:
            raise FilledNotionalError("currency must be a three-letter uppercase code")
        if not callable(clock):
            raise FilledNotionalError("clock must be callable")
        self._currency = currency
        self._clock = clock
        self._failed_reason: Optional[str] = None
        self._restored_trading_date: date
        self._restored_total: Decimal

        try:
            self._initialize_if_missing()
            current_day = self._current_trading_date()
            with self._connection(readonly=True) as connection:
                self._validate_ledger(connection)
                total = self._total_for_date(connection, current_day)
        except FilledNotionalError:
            raise
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            raise FilledNotionalUnavailable("filled-notional ledger startup failed closed") from exc
        self._restored_trading_date = current_day
        self._restored_total = total

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def restored_trading_date(self) -> date:
        """Trading date restored during construction."""

        return self._restored_trading_date

    @property
    def restored_gross_filled_notional(self) -> Decimal:
        """Exact total restored from durable records during construction."""

        return self._restored_total

    def record_fill(self, fill: ExecutedFill) -> FillAccountingResult:
        """Append one actual execution, or accept an exact replay idempotently."""

        self._require_available()
        canonical = _canonical_fill(fill)
        if canonical.currency != self._currency:
            raise FilledNotionalError("fill currency does not match the service scope")

        try:
            with self._connection(readonly=False) as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate_ledger(connection)
                existing = connection.execute(
                    """
                    SELECT side, quantity_text, price_text, currency,
                           executed_at_utc, trading_date, notional_text
                    FROM daily_filled_notional_records
                    WHERE account_id = ? AND portfolio_id = ?
                      AND broker_execution_id = ?
                    """,
                    (
                        self._account_id,
                        self._portfolio_id,
                        canonical.broker_execution_id,
                    ),
                ).fetchone()
                recorded = existing is None
                if existing is not None:
                    stored = tuple(str(value) for value in existing)
                    expected = (
                        canonical.side,
                        canonical.quantity_text,
                        canonical.price_text,
                        canonical.currency,
                        canonical.executed_at_utc,
                        canonical.trading_date,
                        canonical.notional_text,
                    )
                    if stored != expected:
                        raise FilledNotionalConflict(
                            "broker execution identity has conflicting immutable evidence"
                        )
                else:
                    tail = connection.execute("""
                        SELECT sequence, record_hash
                        FROM daily_filled_notional_records
                        ORDER BY sequence DESC LIMIT 1
                        """).fetchone()
                    sequence = 1 if tail is None else int(tail[0]) + 1
                    previous_hash = _ZERO_HASH if tail is None else str(tail[1])
                    digest = _record_hash(
                        sequence,
                        self._account_id,
                        self._portfolio_id,
                        canonical,
                        previous_hash,
                    )
                    connection.execute(
                        """
                        INSERT INTO daily_filled_notional_records (
                            sequence, account_id, portfolio_id, broker_execution_id,
                            side, quantity_text, price_text, currency,
                            executed_at_utc, trading_date, notional_text,
                            previous_hash, record_hash
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sequence,
                            self._account_id,
                            self._portfolio_id,
                            canonical.broker_execution_id,
                            canonical.side,
                            canonical.quantity_text,
                            canonical.price_text,
                            canonical.currency,
                            canonical.executed_at_utc,
                            canonical.trading_date,
                            canonical.notional_text,
                            previous_hash,
                            digest,
                        ),
                    )
                trading_day = date.fromisoformat(canonical.trading_date)
                total = self._total_for_date(connection, trading_day)
                connection.commit()
        except FilledNotionalConflict as exc:
            self._latch_failure(str(exc))
            raise
        except FilledNotionalIntegrityError as exc:
            self._latch_failure(str(exc))
            raise
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            self._latch_failure("filled-notional ledger write failed")
            raise FilledNotionalUnavailable("filled-notional ledger write failed closed") from exc

        return FillAccountingResult(
            recorded=recorded,
            trading_date=trading_day,
            fill_notional=canonical.notional,
            gross_filled_notional=total,
        )

    def current_gross_filled_notional(self, *, as_of: Optional[datetime] = None) -> Decimal:
        """Read the exact total for the New York date containing ``as_of``.

        This API performs no database mutation.  Omitting ``as_of`` uses the
        injected clock, which also makes midnight and DST behavior testable.
        """

        self._require_available()
        try:
            trading_day = self._trading_date(as_of if as_of is not None else self._clock())
            with self._connection(readonly=True) as connection:
                self._validate_ledger(connection)
                total = self._total_for_date(connection, trading_day)
        except FilledNotionalIntegrityError as exc:
            self._latch_failure(str(exc))
            raise
        except FilledNotionalError:
            raise
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            self._latch_failure("filled-notional ledger read failed")
            raise FilledNotionalUnavailable("filled-notional ledger read failed closed") from exc
        return total

    def _require_available(self) -> None:
        if self._failed_reason is not None:
            raise FilledNotionalUnavailable(
                f"filled-notional accounting is latched unavailable: {self._failed_reason}"
            )

    def _latch_failure(self, reason: str) -> None:
        self._failed_reason = reason

    def _current_trading_date(self) -> date:
        return self._trading_date(self._clock())

    @staticmethod
    def _trading_date(instant: object) -> date:
        _, trading_day = _canonical_utc(instant)
        return date.fromisoformat(trading_day)

    def _initialize_if_missing(self) -> None:
        try:
            os.lstat(self._path)
            create = False
        except FileNotFoundError:
            create = True

        binding: Optional[SQLitePathBinding] = None
        connection: Optional[sqlite3.Connection] = None
        try:
            binding = SQLitePathBinding.open_for_initialization(self._path, create=create)
            connection = sqlite3.connect(
                self._path.as_uri() + "?mode=rw",
                uri=True,
                timeout=1.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
            if create:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(_SCHEMA_TABLE_SQL)
                connection.execute(_RECORDS_TABLE_SQL)
                connection.execute(_UNIQUE_EXECUTION_SQL)
                connection.execute(_SCOPE_DATE_SQL)
                for statement in _TRIGGER_SQL.values():
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO daily_filled_notional_schema VALUES (1, ?)",
                    (_SCHEMA_VERSION,),
                )
                connection.commit()
            bound.assert_connection_identity(sqlite_connection_file_identity(connection))
        finally:
            if connection is not None:
                if connection.in_transaction:
                    connection.rollback()
                connection.close()
            if binding is not None:
                binding.close()

    @contextmanager
    def _connection(self, *, readonly: bool) -> Iterator[sqlite3.Connection]:
        binding: Optional[SQLitePathBinding] = None
        connection: Optional[sqlite3.Connection] = None
        try:
            binding = SQLitePathBinding.open_for_initialization(self._path, create=False)
            mode = "ro" if readonly else "rw"
            connection = sqlite3.connect(
                self._path.as_uri() + f"?mode={mode}",
                uri=True,
                timeout=1.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
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

    def _validate_ledger(self, connection: sqlite3.Connection) -> None:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        if len(integrity) != 1 or str(integrity[0][0]) != "ok":
            raise FilledNotionalIntegrityError("SQLite integrity_check failed")

        expected_sql = {
            ("table", "daily_filled_notional_schema"): _SCHEMA_TABLE_SQL,
            ("table", "daily_filled_notional_records"): _RECORDS_TABLE_SQL,
            ("index", "daily_filled_notional_execution_identity"): _UNIQUE_EXECUTION_SQL,
            ("index", "daily_filled_notional_scope_date"): _SCOPE_DATE_SQL,
            **{("trigger", name): sql for name, sql in _TRIGGER_SQL.items()},
        }
        rows = connection.execute("""
            SELECT type, name, sql
            FROM sqlite_master
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
            "SELECT singleton, schema_version FROM daily_filled_notional_schema"
        ).fetchall()
        if [tuple(row) for row in schema_rows] != [(1, _SCHEMA_VERSION)]:
            raise FilledNotionalIntegrityError("filled-notional schema version is invalid")

        previous_hash = _ZERO_HASH
        expected_sequence = 1
        records = connection.execute(
            "SELECT * FROM daily_filled_notional_records ORDER BY sequence"
        ).fetchall()
        identities: set[tuple[str, str, str]] = set()
        for row in records:
            if int(row["sequence"]) != expected_sequence:
                raise FilledNotionalIntegrityError(
                    "filled-notional record sequence is not contiguous"
                )
            account_id = _validate_stored_identifier(row["account_id"], "account_id")
            portfolio_id = _validate_stored_identifier(row["portfolio_id"], "portfolio_id")
            execution_id = _validate_stored_identifier(
                row["broker_execution_id"], "broker_execution_id"
            )
            identity = (account_id, portfolio_id, execution_id)
            if identity in identities:
                raise FilledNotionalIntegrityError("duplicate broker execution identity in ledger")
            identities.add(identity)
            if str(row["side"]) not in {side.value for side in FillSide}:
                raise FilledNotionalIntegrityError("stored fill side is invalid")
            quantity = _parse_canonical_positive_decimal(
                row["quantity_text"],
                "quantity",
                max_digits=38,
                max_abs_exponent=18,
            )
            price = _parse_canonical_positive_decimal(
                row["price_text"],
                "price",
                max_digits=38,
                max_abs_exponent=18,
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
                raise FilledNotionalIntegrityError("stored fill notional does not match quantity")
            if str(row["previous_hash"]) != previous_hash:
                raise FilledNotionalIntegrityError("filled-notional hash chain is broken")
            canonical = _CanonicalFill(
                broker_execution_id=execution_id,
                side=str(row["side"]),
                quantity_text=str(row["quantity_text"]),
                price_text=str(row["price_text"]),
                currency=currency,
                executed_at_utc=str(row["executed_at_utc"]),
                trading_date=str(row["trading_date"]),
                notional_text=str(row["notional_text"]),
            )
            expected_hash = _record_hash(
                expected_sequence,
                account_id,
                portfolio_id,
                canonical,
                previous_hash,
            )
            stored_hash = str(row["record_hash"])
            if not hmac.compare_digest(stored_hash, expected_hash):
                raise FilledNotionalIntegrityError("filled-notional record hash is invalid")
            previous_hash = stored_hash
            expected_sequence += 1

    def _total_for_date(self, connection: sqlite3.Connection, trading_day: date) -> Decimal:
        rows = connection.execute(
            """
            SELECT notional_text
            FROM daily_filled_notional_records
            WHERE account_id = ? AND portfolio_id = ?
              AND currency = ? AND trading_date = ?
            ORDER BY sequence
            """,
            (
                self._account_id,
                self._portfolio_id,
                self._currency,
                trading_day.isoformat(),
            ),
        ).fetchall()
        with localcontext() as context:
            context.prec = MAX_PREC
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            return sum(
                (_parse_canonical_positive_decimal(row[0], "notional") for row in rows),
                Decimal("0"),
            )


def _validate_stored_identifier(value: object, field_name: str) -> str:
    try:
        return _validate_identifier(value, field_name)
    except FilledNotionalError as exc:
        raise FilledNotionalIntegrityError(f"stored {field_name} is invalid") from exc

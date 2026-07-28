"""Producer-owned bootstrap reconciliation over verified broker and SQLite evidence.

The public stage accepts only a core-verified broker evidence envelope.  It
collects the local ledger itself from one WAL-visible, read-only SQLite
transaction, evaluates the PR5 paper-simulator policy, and hands a one-shot
unsigned result to a receiver that must independently claim producer ownership.

IBKR remains diagnostic while the local paper simulator is the execution
authority.  Consequently IBKR must prove zero positions and zero open orders;
valid local simulator positions are retained as bootstrap mark identities and
are never compared for equality with IBKR positions.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import stat
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Protocol, SupportsIndex, TypeVar

from robo_trader.config import PAPER_ONLY_EXECUTION_SOURCE, RuntimeContract
from robo_trader.runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from robo_trader.safety.sqlite_identity import (
    SQLiteIdentityError,
    SQLitePathBinding,
    sqlite_connection_file_identity,
)

from .domain import (
    DOMAIN_SCHEMA_VERSION,
    BrokerCollectionKind,
    ExecutionDomainScope,
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
    _account_scope,
    _schema_version,
    _timestamp,
    canonical_json,
    canonical_timestamp,
    fingerprint,
)
from .errors import LedgerSafetyError
from .ledger import ImmutableLedgerReader, validate_portfolio_ids
from .policy import (
    ReconciliationCoverage,
    ReconciliationStatus,
    evaluate_paper_simulator_reconciliation,
)

BOOTSTRAP_RECONCILIATION_STATUS = "BOOTSTRAP_EVIDENCE_COMPLETE"
BOOTSTRAP_RECONCILIATION_MAX_AGE = timedelta(seconds=30)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_FINGERPRINT = re.compile(r"^[0-9a-f]{16,64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_BROKER_SNAPSHOT_ID = re.compile(r"^broker-reconciliation-v1-[0-9a-f]{64}$")
_BROKER_VERDICT_ID = re.compile(r"^reconciliation-verdict-v1-[0-9a-f]{64}$")
_COLLECTION_EVIDENCE_ID = re.compile(r"^broker-collection-v1-[0-9a-f]{64}$")

_LEDGER_PRODUCER_MARKER = object()
_RESULT_PRODUCER_MARKER = object()
_LEDGER_DIGEST_KEY = os.urandom(32)
_RESULT_DIGEST_KEY = os.urandom(32)
_OWNERSHIP_LOCK = threading.RLock()
_LEDGER_REGISTRY: dict[int, tuple["_CollectedLedgerEvidence", str]] = {}
_RESULT_REGISTRY: dict[str, tuple["UnsignedBootstrapReconciliation", str]] = {}
_CLAIMED_RESULT_NONCES: set[str] = set()


class BootstrapReconciliationBlocked(ReconciliationDomainError):
    """No bootstrap reconciliation result may be emitted."""


class VerifiedBrokerEvidenceEnvelope(Protocol):
    """Core verifier-owned, one-shot broker evidence expected by this stage."""

    snapshot: NormalizedBrokerSnapshot
    snapshot_id: str
    snapshot_hash: str
    artifact_hash: str
    runtime_fingerprint: str
    account_scope: str
    receipt_id: str
    public_key_fingerprint: str
    issued_at: datetime
    expires_at: datetime


def _hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed")
    return value


def _safe_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or value != value.strip() or not _SAFE_ID.fullmatch(value):
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed")
    return value


def _exact_pattern(value: object, field_name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or value != value.strip() or not pattern.fullmatch(value):
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed")
    return value


def _exact_nonnegative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise BootstrapReconciliationBlocked(f"{field_name} must be a nonnegative integer")
    return value


def _canonical_portfolios(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if type(values) is not tuple or any(type(value) is not str for value in values):
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed")
    try:
        normalized = validate_portfolio_ids(values)
    except Exception as exc:
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed") from exc
    if values != normalized:
        raise BootstrapReconciliationBlocked(f"{field_name} must be unique and sorted")
    return normalized


def _canonical_position_identities(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple:
        raise BootstrapReconciliationBlocked("local position identities are malformed")
    normalized: list[tuple[str, str]] = []
    for value in values:
        if type(value) is not tuple or len(value) != 2:
            raise BootstrapReconciliationBlocked("local position identities are malformed")
        portfolio_id, symbol = value
        portfolio = _canonical_portfolios((portfolio_id,), "position portfolio_id")[0]
        if type(symbol) is not str or symbol != symbol.upper() or not _SYMBOL.fullmatch(symbol):
            raise BootstrapReconciliationBlocked("local position symbol is malformed")
        normalized.append((portfolio, symbol))
    ordered = tuple(sorted(normalized))
    if tuple(values) != ordered or len(ordered) != len(set(ordered)):
        raise BootstrapReconciliationBlocked("local position identities must be unique and sorted")
    return ordered


def _utc_clock_value(clock: Callable[[], datetime], label: str) -> datetime:
    try:
        value = clock()
    except Exception as exc:
        raise BootstrapReconciliationBlocked(f"{label} is unavailable") from exc
    return _timestamp(value, label)


def _system_clock() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class _CollectedLedgerEvidence:
    """Private producer-owned result of one live SQLite read transaction."""

    runtime_fingerprint: str
    account_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    portfolio_ids: tuple[str, ...]
    active_portfolio_ids: tuple[str, ...]
    position_identities: tuple[tuple[str, str], ...]
    legacy_snapshot_hash: str
    observed_at: datetime
    data_version: int
    wal_fingerprint: tuple[str, int, int, int, str]
    coverage: ReconciliationCoverage
    _producer_marker: object = field(repr=False, compare=False)
    _producer_digest: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _LEDGER_PRODUCER_MARKER:
            raise BootstrapReconciliationBlocked(
                "ledger evidence must come from the internal WAL-visible collector"
            )


def _ledger_digest(evidence: _CollectedLedgerEvidence) -> str:
    payload = {
        "account_scope": evidence.account_scope,
        "active_portfolio_ids": list(evidence.active_portfolio_ids),
        "coverage": evidence.coverage.canonical_dict(),
        "data_version": evidence.data_version,
        "database_device": evidence.database_device,
        "database_identity": evidence.database_identity,
        "database_inode": evidence.database_inode,
        "database_path": evidence.database_path,
        "legacy_snapshot_hash": evidence.legacy_snapshot_hash,
        "observed_at": canonical_timestamp(evidence.observed_at),
        "portfolio_ids": list(evidence.portfolio_ids),
        "position_identities": [list(value) for value in evidence.position_identities],
        "runtime_fingerprint": evidence.runtime_fingerprint,
        "wal_fingerprint": list(evidence.wal_fingerprint),
    }
    return hmac.new(
        _LEDGER_DIGEST_KEY,
        canonical_json(payload).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _register_ledger_evidence(evidence: _CollectedLedgerEvidence) -> None:
    digest = _ledger_digest(evidence)
    object.__setattr__(evidence, "_producer_digest", digest)
    with _OWNERSHIP_LOCK:
        _LEDGER_REGISTRY[id(evidence)] = (evidence, digest)


def _assert_collector_owned_ledger_evidence(
    evidence: object,
) -> _CollectedLedgerEvidence:
    if type(evidence) is not _CollectedLedgerEvidence:
        raise BootstrapReconciliationBlocked("ledger evidence is not collector-owned")
    with _OWNERSHIP_LOCK:
        registered = _LEDGER_REGISTRY.get(id(evidence))
    if (
        registered is None
        or registered[0] is not evidence
        or evidence._producer_marker is not _LEDGER_PRODUCER_MARKER
        or not hmac.compare_digest(registered[1], evidence._producer_digest)
        or not hmac.compare_digest(evidence._producer_digest, _ledger_digest(evidence))
    ):
        raise BootstrapReconciliationBlocked("ledger evidence is not collector-owned")
    return evidence


def _read_only_authorizer(
    action: int,
    argument1: str | None,
    argument2: str | None,
    database: str | None,
    source: str | None,
) -> int:
    del argument2, database, source
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
    }
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and str(argument1).casefold() in {
        "data_version",
        "integrity_check",
        "table_info",
    }:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _wal_fingerprint(database_path: Path) -> tuple[str, int, int, int, str]:
    """Fingerprint WAL bytes so a WAL-only commit cannot hide behind main-file stat."""

    path = Path(f"{database_path}-wal")
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return ("absent", 0, 0, 0, "")
    except OSError as exc:
        raise BootstrapReconciliationBlocked("ledger WAL cannot be inspected") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise BootstrapReconciliationBlocked("ledger WAL is not a regular file")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.lstat(path)
    except OSError as exc:
        raise BootstrapReconciliationBlocked("ledger WAL cannot be read safely") from exc
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in stable):
        raise BootstrapReconciliationBlocked("ledger WAL changed during collection")
    return ("regular", after.st_dev, after.st_ino, after.st_size, digest.hexdigest())


def _canonical_legacy_rows(
    account_rows: tuple[sqlite3.Row, ...],
    position_rows: tuple[sqlite3.Row, ...],
    trade_rows: tuple[sqlite3.Row, ...],
    equity_rows: tuple[sqlite3.Row, ...],
) -> str:
    payload = {
        "account": [list(row) for row in account_rows],
        "equity_history": [list(row) for row in equity_rows],
        "positions": [list(row) for row in position_rows],
        "trades": [list(row) for row in trade_rows],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _strict_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise BootstrapReconciliationBlocked(f"ledger {field_name} is not finite")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BootstrapReconciliationBlocked(f"ledger {field_name} is not finite") from exc
    if not parsed.is_finite():
        raise BootstrapReconciliationBlocked(f"ledger {field_name} is not finite")
    return parsed


def _strict_ledger_timestamp(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value or any(ord(character) < 32 for character in value):
        raise BootstrapReconciliationBlocked(f"ledger {field_name} is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise BootstrapReconciliationBlocked(f"ledger {field_name} is malformed") from exc


def _validate_legacy_rows(
    *,
    portfolios: tuple[str, ...],
    account_rows: tuple[sqlite3.Row, ...],
    position_rows: tuple[sqlite3.Row, ...],
    trade_rows: tuple[sqlite3.Row, ...],
    equity_rows: tuple[sqlite3.Row, ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    known = _canonical_portfolios(portfolios, "ledger portfolio IDs")
    known_set = set(known)
    if {str(row[0]) for row in account_rows} != known_set:
        raise BootstrapReconciliationBlocked(
            "every ledger portfolio requires exactly one account projection"
        )
    position_identities: list[tuple[str, str]] = []
    for row in position_rows:
        portfolio_id, symbol, quantity, average_cost = row[0], row[1], row[2], row[3]
        if (
            type(portfolio_id) is not str
            or portfolio_id not in known_set
            or type(symbol) is not str
            or symbol != symbol.upper()
            or not _SYMBOL.fullmatch(symbol)
            or type(quantity) is not int
            or quantity == 0
            or _strict_decimal(average_cost, "position average cost") < 0
        ):
            raise BootstrapReconciliationBlocked("ledger position evidence is malformed")
        if row[4] is not None:
            _strict_decimal(row[4], "position market price")
        _strict_ledger_timestamp(row[5], "position timestamp")
        position_identities.append((portfolio_id, symbol))
    for row in trade_rows:
        portfolio_id, symbol, side, quantity, price = row[1], row[2], row[3], row[4], row[5]
        if (
            type(portfolio_id) is not str
            or portfolio_id not in known_set
            or type(symbol) is not str
            or symbol != symbol.upper()
            or not _SYMBOL.fullmatch(symbol)
            or side not in {"BUY", "SELL", "BUY_TO_COVER", "SELL_SHORT"}
            or type(quantity) is not int
            or quantity <= 0
            or _strict_decimal(price, "trade price") < 0
        ):
            raise BootstrapReconciliationBlocked("ledger trade evidence is malformed")
        for index, field_name in (
            (6, "trade notional"),
            (7, "trade slippage"),
            (8, "trade commission"),
        ):
            if row[index] is not None:
                _strict_decimal(row[index], field_name)
        if row[9] is not None:
            _strict_decimal(row[9], "trade pnl")
        _strict_ledger_timestamp(row[10], "trade timestamp")
    for rows, label in ((account_rows, "account"), (equity_rows, "equity history")):
        for row in rows:
            if type(row[0 if label == "account" else 1]) is not str:
                raise BootstrapReconciliationBlocked(f"ledger {label} evidence is malformed")
            portfolio_id = str(row[0 if label == "account" else 1])
            if portfolio_id not in known_set:
                raise BootstrapReconciliationBlocked(f"ledger {label} is outside portfolio scope")
            numeric_indexes = range(1, 6) if label == "account" else range(3, 8)
            for index in numeric_indexes:
                if row[index] is not None:
                    _strict_decimal(row[index], f"{label} financial value")
            _strict_ledger_timestamp(
                row[6 if label == "account" else 8],
                f"{label} timestamp",
            )
    identities = _canonical_position_identities(tuple(sorted(position_identities)))
    active = tuple(sorted({portfolio_id for portfolio_id, _ in identities}))
    return active, identities


@contextmanager
def _collect_wal_visible_ledger(
    runtime: RuntimeContract,
    *,
    observed_at: datetime,
) -> Iterator[_CollectedLedgerEvidence]:
    """Hold one WAL-visible read snapshot through receiver ownership claim."""

    database_path = Path(runtime.database_path)
    binding: SQLitePathBinding | None = None
    connection: sqlite3.Connection | None = None
    evidence: _CollectedLedgerEvidence | None = None
    try:
        binding = SQLitePathBinding.open_readonly(database_path)
        connection = sqlite3.connect(
            database_path.as_uri() + "?mode=ro",
            uri=True,
            timeout=1.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
            raise BootstrapReconciliationBlocked("SQLite query-only mode could not be proven")
        connection_binding = binding.bind_sqlite_connection(
            sqlite_connection_file_identity(connection)
        )
        connection_binding.assert_connection_identity(sqlite_connection_file_identity(connection))
        data_version = int(connection.execute("PRAGMA data_version").fetchone()[0])
        connection.set_authorizer(_read_only_authorizer)
        connection.execute("BEGIN")
        ImmutableLedgerReader._validate_schema(connection)
        required_tables = {"account", "equity_history", "portfolios", "positions", "trades"}
        actual_tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not required_tables.issubset(actual_tables):
            raise BootstrapReconciliationBlocked("legacy database schema is incomplete")
        portfolio_rows = tuple(
            connection.execute("SELECT id FROM portfolios ORDER BY id").fetchall()
        )
        account_rows = tuple(
            connection.execute(
                "SELECT portfolio_id,cash,equity,daily_pnl,realized_pnl,"
                "unrealized_pnl,timestamp FROM account ORDER BY portfolio_id"
            ).fetchall()
        )
        position_rows = tuple(
            connection.execute(
                "SELECT portfolio_id,symbol,quantity,avg_cost,market_price,timestamp "
                "FROM positions WHERE quantity <> 0 ORDER BY portfolio_id,symbol"
            ).fetchall()
        )
        trade_rows = tuple(
            connection.execute(
                "SELECT id,portfolio_id,symbol,side,quantity,price,notional,slippage,"
                "commission,pnl,timestamp FROM trades ORDER BY id"
            ).fetchall()
        )
        equity_rows = tuple(
            connection.execute(
                "SELECT id,portfolio_id,date,equity,cash,positions_value,realized_pnl,"
                "unrealized_pnl,timestamp FROM equity_history ORDER BY id"
            ).fetchall()
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or tuple(integrity) != ("ok",):
            raise BootstrapReconciliationBlocked("legacy database failed integrity_check")
        portfolios = tuple(str(row[0]) for row in portfolio_rows)
        active, position_identities = _validate_legacy_rows(
            portfolios=portfolios,
            account_rows=account_rows,
            position_rows=position_rows,
            trade_rows=trade_rows,
            equity_rows=equity_rows,
        )
        legacy_payload = _canonical_legacy_rows(
            account_rows,
            position_rows,
            trade_rows,
            equity_rows,
        )
        coverage = ReconciliationCoverage(
            broker_account=True,
            broker_positions=True,
            broker_open_orders=True,
            broker_completed_orders=True,
            broker_executions=True,
            broker_commissions=True,
            ledger_positions=True,
            ledger_orders=True,
            ledger_executions=True,
            ledger_cash=True,
        )
        evidence = _CollectedLedgerEvidence(
            runtime_fingerprint=runtime.fingerprint,
            account_scope=_account_scope(runtime.safety_account_scope),
            database_path=str(database_path),
            database_identity=runtime.database_identity,
            database_device=binding.device,
            database_inode=binding.inode,
            portfolio_ids=_canonical_portfolios(portfolios, "ledger portfolio IDs"),
            active_portfolio_ids=active,
            position_identities=position_identities,
            legacy_snapshot_hash=hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest(),
            observed_at=observed_at,
            data_version=data_version,
            wal_fingerprint=_wal_fingerprint(database_path),
            coverage=coverage,
            _producer_marker=_LEDGER_PRODUCER_MARKER,
        )
        _register_ledger_evidence(evidence)
        connection_binding.assert_connection_identity(sqlite_connection_file_identity(connection))
        binding.assert_path_identity()
        yield evidence
        _assert_collector_owned_ledger_evidence(evidence)
        if int(connection.execute("PRAGMA data_version").fetchone()[0]) != data_version:
            raise BootstrapReconciliationBlocked(
                "ledger changed while reconciliation evidence was being consumed"
            )
        if _wal_fingerprint(database_path) != evidence.wal_fingerprint:
            raise BootstrapReconciliationBlocked(
                "ledger WAL changed while reconciliation evidence was being consumed"
            )
        connection_binding.assert_connection_identity(sqlite_connection_file_identity(connection))
        binding.assert_path_identity()
    except (sqlite3.Error, SQLiteIdentityError, LedgerSafetyError) as exc:
        raise BootstrapReconciliationBlocked(
            "WAL-visible immutable ledger collection failed"
        ) from exc
    finally:
        if evidence is not None:
            with _OWNERSHIP_LOCK:
                _LEDGER_REGISTRY.pop(id(evidence), None)
        if connection is not None:
            try:
                connection.rollback()
            finally:
                connection.close()
        if binding is not None:
            binding.close()


@dataclass(frozen=True, slots=True)
class UnsignedBootstrapReconciliation:
    """Canonical one-shot producer result with no startup or mutation authority."""

    generated_at: datetime
    runtime_fingerprint: str
    account_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    portfolio_ids: tuple[str, ...]
    local_position_identities: tuple[tuple[str, str], ...]
    legacy_snapshot_hash: str
    broker_snapshot_id: str
    broker_snapshot_hash: str
    broker_artifact_hash: str
    broker_collection_evidence_ids: tuple[str, ...]
    broker_receipt_id: str
    broker_public_key_fingerprint: str
    broker_verdict_id: str
    broker_verdict_hash: str
    comparison_coverage: ReconciliationCoverage
    reconciliation_status: ReconciliationStatus
    broker_positions_count: int
    broker_open_orders_count: int
    _producer_marker: object = field(repr=False, compare=False)
    _producer_nonce: str = field(repr=False, compare=False)
    _producer_digest: str = field(default="", repr=False, compare=False)
    managed_account_count: int = 1
    status: str = BOOTSTRAP_RECONCILIATION_STATUS
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self._producer_marker is not _RESULT_PRODUCER_MARKER:
            raise BootstrapReconciliationBlocked(
                "unsigned reconciliation must come from the reconciliation producer"
            )
        _schema_version(self.schema_version, "unsigned bootstrap reconciliation")
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))
        if not _RUNTIME_FINGERPRINT.fullmatch(self.runtime_fingerprint):
            raise BootstrapReconciliationBlocked("runtime_fingerprint is malformed")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise BootstrapReconciliationBlocked("database_path must be absolute and lexical")
        _safe_id(self.database_identity, "database_identity")
        for field_name in (
            "database_device",
            "database_inode",
            "broker_positions_count",
            "broker_open_orders_count",
            "managed_account_count",
        ):
            _exact_nonnegative_int(getattr(self, field_name), field_name)
        if (
            self.database_inode == 0
            or self.broker_positions_count != 0
            or self.broker_open_orders_count != 0
            or self.managed_account_count != 1
        ):
            raise BootstrapReconciliationBlocked(
                "unsigned result does not prove one zero-exposure paper account"
            )
        _canonical_portfolios(self.portfolio_ids, "portfolio_ids")
        _canonical_position_identities(self.local_position_identities)
        _hash(self.legacy_snapshot_hash, "legacy_snapshot_hash")
        _hash(self.broker_snapshot_hash, "broker_snapshot_hash")
        _hash(self.broker_artifact_hash, "broker_artifact_hash")
        _hash(self.broker_public_key_fingerprint, "broker_public_key_fingerprint")
        _hash(self.broker_verdict_hash, "broker_verdict_hash")
        _exact_pattern(self.broker_snapshot_id, "broker_snapshot_id", _BROKER_SNAPSHOT_ID)
        _exact_pattern(self.broker_verdict_id, "broker_verdict_id", _BROKER_VERDICT_ID)
        _safe_id(self.broker_receipt_id, "broker_receipt_id")
        evidence_ids = tuple(self.broker_collection_evidence_ids)
        if (
            len(evidence_ids) != len(BrokerCollectionKind)
            or tuple(sorted(evidence_ids)) != evidence_ids
            or len(set(evidence_ids)) != len(evidence_ids)
            or any(not _COLLECTION_EVIDENCE_ID.fullmatch(value) for value in evidence_ids)
        ):
            raise BootstrapReconciliationBlocked("broker collection evidence is incomplete")
        if type(self.comparison_coverage) is not ReconciliationCoverage:
            raise BootstrapReconciliationBlocked("comparison coverage is malformed")
        if not self.comparison_coverage.complete:
            raise BootstrapReconciliationBlocked("comparison coverage is incomplete")
        if self.reconciliation_status not in {
            ReconciliationStatus.PASSED,
            ReconciliationStatus.DEGRADED,
        }:
            raise BootstrapReconciliationBlocked("reconciliation result is quarantined")
        if self.status != BOOTSTRAP_RECONCILIATION_STATUS:
            raise BootstrapReconciliationBlocked("bootstrap reconciliation status is invalid")
        if not _HEX_64.fullmatch(self._producer_nonce):
            raise BootstrapReconciliationBlocked("producer nonce is malformed")

    def __copy__(self) -> "UnsignedBootstrapReconciliation":
        raise TypeError("producer-owned reconciliation result cannot be copied")

    def __deepcopy__(self, memo: object) -> "UnsignedBootstrapReconciliation":
        del memo
        raise TypeError("producer-owned reconciliation result cannot be copied")

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("producer-owned reconciliation result cannot be pickled")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        del protocol
        raise TypeError("producer-owned reconciliation result cannot be pickled")

    @property
    def local_simulator_positions_count(self) -> int:
        return len(self.local_position_identities)

    @property
    def mutated_state(self) -> bool:
        return False

    @property
    def authorizes_startup(self) -> bool:
        return False

    @property
    def execution_domain_scope(self) -> str:
        return PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE

    def binding_dict(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "authorizes_startup": False,
            "broker_collection_evidence_ids": list(self.broker_collection_evidence_ids),
            "broker_artifact_hash": self.broker_artifact_hash,
            "broker_open_orders_count": self.broker_open_orders_count,
            "broker_positions_count": self.broker_positions_count,
            "broker_public_key_fingerprint": self.broker_public_key_fingerprint,
            "broker_receipt_id": self.broker_receipt_id,
            "broker_snapshot_hash": self.broker_snapshot_hash,
            "broker_snapshot_id": self.broker_snapshot_id,
            "broker_verdict_hash": self.broker_verdict_hash,
            "broker_verdict_id": self.broker_verdict_id,
            "comparison_coverage": self.comparison_coverage.canonical_dict(),
            "database_device": self.database_device,
            "database_identity": self.database_identity,
            "database_inode": self.database_inode,
            "database_path": self.database_path,
            "execution_domain_scope": self.execution_domain_scope,
            "generated_at": canonical_timestamp(self.generated_at),
            "legacy_snapshot_hash": self.legacy_snapshot_hash,
            "local_position_identities": [list(value) for value in self.local_position_identities],
            "local_simulator_positions_count": self.local_simulator_positions_count,
            "managed_account_count": self.managed_account_count,
            "mutated_state": False,
            "portfolio_ids": list(self.portfolio_ids),
            "reconciliation_status": self.reconciliation_status.value,
            "runtime_fingerprint": self.runtime_fingerprint,
            "schema_version": self.schema_version,
            "status": self.status,
        }

    @property
    def snapshot_id(self) -> str:
        return fingerprint("bootstrap-reconciliation-v1", self.binding_dict())

    def canonical_dict(self) -> dict[str, object]:
        return {**self.binding_dict(), "snapshot_id": self.snapshot_id}

    def canonical_payload(self) -> str:
        return canonical_json(self.canonical_dict())


def _result_digest(result: UnsignedBootstrapReconciliation) -> str:
    return hmac.new(
        _RESULT_DIGEST_KEY,
        result.canonical_payload().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _register_result(result: UnsignedBootstrapReconciliation) -> None:
    digest = _result_digest(result)
    object.__setattr__(result, "_producer_digest", digest)
    with _OWNERSHIP_LOCK:
        if result._producer_nonce in _RESULT_REGISTRY:
            raise BootstrapReconciliationBlocked("producer nonce collision")
        _RESULT_REGISTRY[result._producer_nonce] = (result, digest)


def assert_and_consume_producer_owned_bootstrap_reconciliation(
    result: object,
) -> UnsignedBootstrapReconciliation:
    """One-shot receiver assertion; direct construction, copies, and replay fail."""

    if type(result) is not UnsignedBootstrapReconciliation:
        raise BootstrapReconciliationBlocked("reconciliation result is not producer-owned")
    with _OWNERSHIP_LOCK:
        registered = _RESULT_REGISTRY.get(result._producer_nonce)
        if (
            registered is None
            or registered[0] is not result
            or result._producer_marker is not _RESULT_PRODUCER_MARKER
            or not hmac.compare_digest(registered[1], result._producer_digest)
            or not hmac.compare_digest(result._producer_digest, _result_digest(result))
        ):
            raise BootstrapReconciliationBlocked("reconciliation result is not producer-owned")
        _RESULT_REGISTRY.pop(result._producer_nonce, None)
        _CLAIMED_RESULT_NONCES.add(result._producer_nonce)
    return result


ReceiverResult = TypeVar("ReceiverResult", covariant=True)


class BootstrapReconciliationReceiver(Protocol, Generic[ReceiverResult]):
    """Receiver must claim the exact result before signing or persistence."""

    def receive_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> ReceiverResult:
        """Consume one producer-owned result exactly once."""


@dataclass(frozen=True, slots=True)
class BootstrapReconciliationDelivery(Generic[ReceiverResult]):
    """Non-signable handoff needed by core to collect position marks next."""

    receiver_result: ReceiverResult
    local_position_identities: tuple[tuple[str, str], ...]


def _validate_runtime(runtime: object) -> RuntimeContract:
    if type(runtime) is not RuntimeContract:
        raise BootstrapReconciliationBlocked("producer requires an exact RuntimeContract")
    if (
        runtime.execution_mode != "paper"
        or runtime.execution_source != PAPER_ONLY_EXECUTION_SOURCE
        or runtime.state_namespace != "paper"
        or runtime.account_type != "paper"
        or runtime.ibkr_port != 4002
        or runtime.ibkr_readonly is not True
        or runtime.safety_execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
    ):
        raise BootstrapReconciliationBlocked("runtime is not sealed paper/read-only topology")
    host = runtime.ibkr_host.casefold()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if host not in {"localhost", "localhost."} and not (
        address is not None and address.is_loopback
    ):
        raise BootstrapReconciliationBlocked("runtime broker host is not loopback")
    if not isinstance(runtime.safety_account_scope, str):
        raise BootstrapReconciliationBlocked("runtime account scope is unavailable")
    _account_scope(runtime.safety_account_scope)
    database_path = Path(runtime.database_path)
    if not database_path.is_absolute() or str(database_path) != runtime.database_path:
        raise BootstrapReconciliationBlocked("runtime database path must be absolute and lexical")
    return runtime


def _consume_verified_broker_evidence(
    envelope: object,
) -> VerifiedBrokerEvidenceEnvelope:
    """Late import avoids a receiver/producer import cycle; there is no fallback."""

    try:
        from robo_trader.bootstrap_evidence_receivers import (
            assert_and_consume_verified_broker_evidence,
        )
    except (ImportError, AttributeError) as exc:
        raise BootstrapReconciliationBlocked(
            "verified broker evidence authority is unavailable"
        ) from exc
    try:
        return assert_and_consume_verified_broker_evidence(envelope)
    except Exception as exc:
        raise BootstrapReconciliationBlocked("broker evidence is not verifier-owned") from exc


def _validate_verified_broker_envelope(
    envelope: VerifiedBrokerEvidenceEnvelope,
    runtime: RuntimeContract,
    checked_at: datetime,
) -> NormalizedBrokerSnapshot:
    snapshot = envelope.snapshot
    if type(snapshot) is not NormalizedBrokerSnapshot:
        raise BootstrapReconciliationBlocked("verified broker envelope is not normalized")
    canonical_hash = hashlib.sha256(snapshot.canonical_payload().encode("utf-8")).hexdigest()
    _hash(envelope.artifact_hash, "broker artifact_hash")
    issued_at = _timestamp(envelope.issued_at, "broker receipt issued_at")
    expires_at = _timestamp(envelope.expires_at, "broker receipt expires_at")
    if (
        envelope.snapshot_id != snapshot.snapshot_id
        or _hash(envelope.snapshot_hash, "broker snapshot_hash") != canonical_hash
        or envelope.runtime_fingerprint != runtime.fingerprint
        or envelope.account_scope != runtime.safety_account_scope
        or snapshot.account.account_scope != runtime.safety_account_scope
        or snapshot.account.account_alias != runtime.account_alias
        or not snapshot.retrieved_at <= issued_at <= checked_at <= expires_at
        or checked_at - issued_at > BOOTSTRAP_RECONCILIATION_MAX_AGE
    ):
        raise BootstrapReconciliationBlocked(
            "verified broker evidence is stale or outside the runtime binding"
        )
    _safe_id(envelope.receipt_id, "broker receipt_id")
    _hash(envelope.public_key_fingerprint, "broker public_key_fingerprint")
    return snapshot


def _produce(
    verified_broker_evidence: object,
    runtime_contract: RuntimeContract,
    receiver: BootstrapReconciliationReceiver[ReceiverResult],
    *,
    clock: Callable[[], datetime],
    broker_consumer: Callable[[object], VerifiedBrokerEvidenceEnvelope],
) -> BootstrapReconciliationDelivery[ReceiverResult]:
    runtime = _validate_runtime(runtime_contract)
    envelope = broker_consumer(verified_broker_evidence)
    collected: _CollectedLedgerEvidence | None = None
    result: UnsignedBootstrapReconciliation | None = None
    position_identities: tuple[tuple[str, str], ...] = ()
    try:
        observed_at = _utc_clock_value(clock, "reconciliation producer clock")
        snapshot = _validate_verified_broker_envelope(envelope, runtime, observed_at)
        with _collect_wal_visible_ledger(runtime, observed_at=observed_at) as raw_evidence:
            collected = _assert_collector_owned_ledger_evidence(raw_evidence)
            checked_at = _utc_clock_value(clock, "reconciliation producer clock")
            if checked_at - observed_at > BOOTSTRAP_RECONCILIATION_MAX_AGE:
                raise BootstrapReconciliationBlocked("local ledger evidence became stale")
            _validate_verified_broker_envelope(envelope, runtime, checked_at)
            verdict = evaluate_paper_simulator_reconciliation(
                snapshot,
                collected.coverage,
                expected_account_scope=runtime.safety_account_scope,
                now=checked_at,
                max_age_seconds=BOOTSTRAP_RECONCILIATION_MAX_AGE.total_seconds(),
            )
            if verdict.quarantine_required:
                raise BootstrapReconciliationBlocked(
                    "reconciliation contains stale, incomplete, unknown, or material differences"
                )
            if snapshot.positions or snapshot.open_orders:
                raise BootstrapReconciliationBlocked(
                    "IBKR diagnostic account does not have zero exposure and open orders"
                )
            evidence_by_kind = {item.collection: item for item in snapshot.collection_evidence}
            if not snapshot.completeness.complete or set(evidence_by_kind) != set(
                BrokerCollectionKind
            ):
                raise BootstrapReconciliationBlocked("broker collection evidence is incomplete")
            result = UnsignedBootstrapReconciliation(
                generated_at=checked_at,
                runtime_fingerprint=runtime.fingerprint,
                account_scope=runtime.safety_account_scope,
                database_path=collected.database_path,
                database_identity=collected.database_identity,
                database_device=collected.database_device,
                database_inode=collected.database_inode,
                portfolio_ids=collected.portfolio_ids,
                local_position_identities=collected.position_identities,
                legacy_snapshot_hash=collected.legacy_snapshot_hash,
                broker_snapshot_id=snapshot.snapshot_id,
                broker_snapshot_hash=envelope.snapshot_hash,
                broker_artifact_hash=envelope.artifact_hash,
                broker_collection_evidence_ids=tuple(
                    sorted(item.evidence_id for item in snapshot.collection_evidence)
                ),
                broker_receipt_id=envelope.receipt_id,
                broker_public_key_fingerprint=envelope.public_key_fingerprint,
                broker_verdict_id=verdict.verdict_id,
                broker_verdict_hash=hashlib.sha256(
                    verdict.canonical_payload().encode("utf-8")
                ).hexdigest(),
                comparison_coverage=collected.coverage,
                reconciliation_status=verdict.status,
                broker_positions_count=len(snapshot.positions),
                broker_open_orders_count=len(snapshot.open_orders),
                _producer_marker=_RESULT_PRODUCER_MARKER,
                _producer_nonce=secrets.token_hex(32),
            )
            _register_result(result)
            position_identities = collected.position_identities
        if result is None:  # pragma: no cover - defensive invariant
            raise BootstrapReconciliationBlocked("reconciliation result was not produced")
        capability = getattr(receiver, "receive_unsigned_bootstrap_reconciliation", None)
        if not callable(capability):
            raise BootstrapReconciliationBlocked(
                "bootstrap reconciliation receiver capability is unavailable"
            )
        receiver_result = capability(result)
        with _OWNERSHIP_LOCK:
            if result._producer_nonce not in _CLAIMED_RESULT_NONCES:
                raise BootstrapReconciliationBlocked(
                    "receiver did not independently claim reconciliation ownership"
                )
            _CLAIMED_RESULT_NONCES.remove(result._producer_nonce)
        return BootstrapReconciliationDelivery(
            receiver_result=receiver_result,
            local_position_identities=position_identities,
        )
    finally:
        if result is not None:
            with _OWNERSHIP_LOCK:
                _RESULT_REGISTRY.pop(result._producer_nonce, None)
                _CLAIMED_RESULT_NONCES.discard(result._producer_nonce)


def produce_bootstrap_reconciliation(
    verified_broker_evidence: object,
    runtime_contract: RuntimeContract,
    receiver: BootstrapReconciliationReceiver[ReceiverResult],
) -> BootstrapReconciliationDelivery[ReceiverResult]:
    """Produce from internal wall time and fixed freshness policy only."""

    return _produce(
        verified_broker_evidence,
        runtime_contract,
        receiver,
        clock=_system_clock,
        broker_consumer=_consume_verified_broker_evidence,
    )


@dataclass(frozen=True, slots=True)
class _TestProducerCapability:
    clock: Callable[[], datetime]
    broker_consumer: Callable[[object], VerifiedBrokerEvidenceEnvelope]
    _marker: object = field(repr=False)


_TEST_CAPABILITY_MARKER = object()


def _issue_test_producer_capability(
    *,
    clock: Callable[[], datetime],
    broker_consumer: Callable[[object], VerifiedBrokerEvidenceEnvelope],
) -> _TestProducerCapability:
    """Private fixture seam; production has no caller-controlled clock/verifier."""

    return _TestProducerCapability(
        clock=clock,
        broker_consumer=broker_consumer,
        _marker=_TEST_CAPABILITY_MARKER,
    )


def _produce_bootstrap_reconciliation_for_test(
    verified_broker_evidence: object,
    runtime_contract: RuntimeContract,
    receiver: BootstrapReconciliationReceiver[ReceiverResult],
    capability: _TestProducerCapability,
) -> BootstrapReconciliationDelivery[ReceiverResult]:
    if (
        type(capability) is not _TestProducerCapability
        or capability._marker is not _TEST_CAPABILITY_MARKER
    ):
        raise BootstrapReconciliationBlocked("test producer capability is invalid")
    return _produce(
        verified_broker_evidence,
        runtime_contract,
        receiver,
        clock=capability.clock,
        broker_consumer=capability.broker_consumer,
    )

"""Exact, append-only bootstrap records for a legacy paper-simulator ledger.

The legacy SQLite projections use ``REAL`` and cannot become financial
authority merely by passing through ``Decimal(str(value))``.  A bootstrap is
therefore an explicit, fingerprinted accounting epoch.  Its values are
operator-reviewed inputs bound to a read-only broker snapshot proving that the
separate IBKR paper account has no exposure or open orders.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import sqlite3
import stat
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .safety.models import decimal_to_fixed, utc_to_text
from .safety.sqlite_identity import (
    SQLiteIdentityError,
    SQLitePathBinding,
    lexical_path_preserving_leaf,
    sqlite_connection_file_identity,
)

BOOTSTRAP_SCHEMA_VERSION = 1
BOOTSTRAP_ID_PREFIX = "pboot-"
MAX_MARK_AGE = timedelta(minutes=5)
MAX_EVIDENCE_BYTES = 2 * 1024 * 1024
RECONCILIATION_EVIDENCE_STATUS = "BOOTSTRAP_EVIDENCE_COMPLETE"
MARK_EVIDENCE_SOURCE = "pr3-validated-market-data-v1"
_EVIDENCE_PRODUCER_MARKER = object()

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_BOOTSTRAP_ID = re.compile(r"^pboot-[0-9a-f]{32}$")
_ACCOUNT_SCOPE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class ExactStateBootstrapError(ValueError):
    """A candidate cannot safely establish a sealed accounting epoch."""


@dataclass(frozen=True, slots=True)
class ExactBootstrapMarkEvidence:
    """One verified PR3 mark artifact, retained without mutable JSON state."""

    portfolio_id: str
    symbol: str
    price: Decimal
    observed_at: datetime
    source_event_id: str
    con_id: int
    artifact_path: str
    artifact_hash: str


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapEvidence:
    """Cross-linked offline evidence verified from regular owner-only files."""

    reconciliation_snapshot_id: str
    reconciliation_report_hash: str
    broker_snapshot_hash: str
    legacy_snapshot_hash: str
    runtime_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    portfolio_ids: tuple[str, ...]
    broker_observed_at: datetime
    reconciliation_generated_at: datetime
    broker_position_count: int
    broker_open_order_count: int
    marks: tuple[ExactBootstrapMarkEvidence, ...]
    _producer_marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _EVIDENCE_PRODUCER_MARKER:
            raise ExactStateBootstrapError(
                "bootstrap evidence must come from the verified artifact loader"
            )


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapBackupReceipt:
    """Immutable facts that core re-verifies before beginning the bootstrap."""

    schema_version: int
    created_at: datetime
    candidate_fingerprint: str
    source_path: str
    source_device: int
    source_inode: int
    backup_path: str
    backup_device: int
    backup_inode: int
    integrity_check: str
    source_snapshot_hash: str
    row_counts: tuple[tuple[str, int], ...]
    table_hashes: tuple[tuple[str, str], ...]
    backup_content_hash: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ExactStateBootstrapError("backup receipt schema is unsupported")
        object.__setattr__(self, "created_at", _utc(self.created_at, "backup created_at"))
        for value, label in (
            (self.source_path, "backup source_path"),
            (self.backup_path, "backup backup_path"),
        ):
            path = Path(value)
            if not path.is_absolute() or str(path) != value:
                raise ExactStateBootstrapError(f"{label} must be absolute and lexical")
        for value, label in (
            (self.source_device, "backup source_device"),
            (self.source_inode, "backup source_inode"),
            (self.backup_device, "backup backup_device"),
            (self.backup_inode, "backup backup_inode"),
        ):
            if type(value) is not int or value < 0:
                raise ExactStateBootstrapError(f"{label} is invalid")
        _hash(self.candidate_fingerprint, "backup candidate_fingerprint")
        _hash(self.source_snapshot_hash, "backup source_snapshot_hash")
        _hash(self.backup_content_hash, "backup_content_hash")
        if self.integrity_check != "ok":
            raise ExactStateBootstrapError("backup integrity_check is not ok")
        required_tables = {"account", "equity_history", "positions", "trades"}
        row_names = tuple(name for name, _ in self.row_counts)
        hash_names = tuple(name for name, _ in self.table_hashes)
        if (
            type(self.row_counts) is not tuple
            or type(self.table_hashes) is not tuple
            or row_names != tuple(sorted(set(row_names)))
            or hash_names != row_names
            or not required_tables.issubset(row_names)
            or any(type(count) is not int or count < 0 for _, count in self.row_counts)
            or any(not isinstance(name, str) or not _SAFE_ID.fullmatch(name) for name in row_names)
            or any(_HEX_64.fullmatch(digest) is None for _, digest in self.table_hashes)
        ):
            raise ExactStateBootstrapError("backup table evidence is malformed")


def _decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    if type(value) not in {Decimal, str}:
        raise ExactStateBootstrapError(f"{label} is not an exact decimal")
    try:
        exact = value if type(value) is Decimal else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExactStateBootstrapError(f"{label} is not an exact decimal") from exc
    if not exact.is_finite() or (positive and exact <= 0):
        raise ExactStateBootstrapError(f"{label} is outside its valid range")
    # Reject noncanonical textual inputs instead of silently normalizing them.
    if isinstance(value, str) and decimal_to_fixed(exact) != value:
        raise ExactStateBootstrapError(f"{label} is not canonical")
    return exact


def _utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExactStateBootstrapError(f"{label} is invalid") from exc
    else:
        raise ExactStateBootstrapError(f"{label} is invalid")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ExactStateBootstrapError(f"{label} must be timezone-aware")
    return result.astimezone(timezone.utc)


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ExactStateBootstrapError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ExactStateBootstrapError(f"{label} is malformed")
    return value


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ExactStateBootstrapError(f"{label} fields are incomplete or unknown")


def _verified_regular_file_bytes(path: Path, label: str) -> tuple[bytes, os.stat_result]:
    """Read one immutable-by-identity, owner-only artifact without following its leaf."""

    protected = Path(path)
    if not protected.is_absolute():
        raise ExactStateBootstrapError(f"{label} path must be absolute")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ExactStateBootstrapError("platform cannot enforce no-follow evidence reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor: int | None = None
    try:
        descriptor = os.open(protected, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExactStateBootstrapError(f"{label} must be a regular file")
        getuid = getattr(os, "getuid", None)
        if getuid is not None and before.st_uid != getuid():
            raise ExactStateBootstrapError(f"{label} must be owned by the current user")
        if stat.S_IMODE(before.st_mode) & 0o077:
            raise ExactStateBootstrapError(f"{label} permissions must be owner-only")
        if before.st_size > MAX_EVIDENCE_BYTES:
            raise ExactStateBootstrapError(f"{label} exceeds the evidence size limit")
        chunks: list[bytes] = []
        remaining = MAX_EVIDENCE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > MAX_EVIDENCE_BYTES:
            raise ExactStateBootstrapError(f"{label} exceeds the evidence size limit")
        after = os.fstat(descriptor)
        current = os.stat(protected, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ExactStateBootstrapError(f"{label} changed while it was read")
        return payload, after
    except (OSError, UnicodeError) as exc:
        raise ExactStateBootstrapError(f"{label} cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _verified_json(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    payload, _ = _verified_regular_file_bytes(path, label)
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExactStateBootstrapError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping):
        raise ExactStateBootstrapError(f"{label} must contain a JSON object")
    return raw, hashlib.sha256(payload).hexdigest()


def verified_file_sha256(path: Path, label: str = "file") -> tuple[str, os.stat_result]:
    """Return the exact-byte hash and descriptor identity of an owner-only file."""

    payload, metadata = _verified_regular_file_bytes(path, label)
    return hashlib.sha256(payload).hexdigest(), metadata


def _runtime_contract_values(runtime_contract: object) -> tuple[Path, str, str, str]:
    from .config import RuntimeContract

    if type(runtime_contract) is not RuntimeContract:
        raise ExactStateBootstrapError("evidence requires an exact RuntimeContract")
    path = lexical_path_preserving_leaf(runtime_contract.database_path)
    if (
        runtime_contract.execution_mode != "paper"
        or runtime_contract.execution_source != "paper_simulator"
        or runtime_contract.state_namespace != "paper"
        or runtime_contract.account_type != "paper"
        or runtime_contract.ibkr_port != 4002
        or runtime_contract.ibkr_readonly is not True
        or runtime_contract.safety_execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not isinstance(runtime_contract.safety_account_scope, str)
        or not _ACCOUNT_SCOPE.fullmatch(runtime_contract.safety_account_scope)
        or len(set(runtime_contract.safety_account_scope.removeprefix("acct_v1_"))) == 1
    ):
        raise ExactStateBootstrapError("runtime contract is not sealed paper/read-only topology")
    host = str(runtime_contract.ibkr_host).casefold()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if host not in {"localhost", "localhost."} and not (
        address is not None and address.is_loopback
    ):
        raise ExactStateBootstrapError("runtime contract broker host is not loopback")
    return (
        path,
        runtime_contract.database_identity,
        runtime_contract.fingerprint,
        runtime_contract.safety_account_scope,
    )


def load_exact_state_bootstrap_evidence(
    *,
    reconciliation_path: Path,
    broker_snapshot_path: Path,
    protective_mark_paths: Sequence[Path],
    expected_runtime_contract: object,
) -> ExactStateBootstrapEvidence:
    """Load and cross-check the actual offline artifacts used by a bootstrap."""

    database_path, database_identity, runtime_fingerprint, account_scope = _runtime_contract_values(
        expected_runtime_contract
    )
    try:
        database_metadata = os.lstat(database_path)
    except OSError as exc:
        raise ExactStateBootstrapError("runtime database identity cannot be inspected") from exc
    if stat.S_ISLNK(database_metadata.st_mode) or not stat.S_ISREG(database_metadata.st_mode):
        raise ExactStateBootstrapError("runtime database must be a non-symlink regular file")

    broker, broker_hash = _verified_json(Path(broker_snapshot_path), "broker snapshot")
    _exact_keys(
        broker,
        {
            "schema_version",
            "snapshot_id",
            "observed_at",
            "broker_time_before",
            "broker_time_after",
            "runtime_fingerprint",
            "execution_domain_scope",
            "account_scope",
            "broker_host",
            "broker_port",
            "read_only",
            "managed_account_count",
            "positions",
            "open_orders",
        },
        "broker snapshot",
    )
    if broker["schema_version"] != 1:
        raise ExactStateBootstrapError("broker snapshot schema is unsupported")
    broker_observed_at = _utc(broker["observed_at"], "broker observed_at")
    broker_time_before = _utc(broker["broker_time_before"], "broker_time_before")
    broker_time_after = _utc(broker["broker_time_after"], "broker_time_after")
    if not broker_time_before <= broker_observed_at <= broker_time_after:
        raise ExactStateBootstrapError("broker snapshot time bounds are inconsistent")
    if (
        broker["runtime_fingerprint"] != runtime_fingerprint
        or broker["execution_domain_scope"] != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not hmac.compare_digest(str(broker["account_scope"]), account_scope)
        or str(broker["broker_host"]).casefold()
        != str(expected_runtime_contract.ibkr_host).casefold()
        or broker["broker_port"] != 4002
        or broker["read_only"] is not True
        or broker["managed_account_count"] != 1
    ):
        raise ExactStateBootstrapError("broker snapshot is not bound to runtime evidence")
    if broker["positions"] != [] or broker["open_orders"] != []:
        raise ExactStateBootstrapError("broker snapshot does not prove zero paper exposure")
    host = str(broker["broker_host"]).casefold()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if host not in {"localhost", "localhost."} and not (
        address is not None and address.is_loopback
    ):
        raise ExactStateBootstrapError("broker snapshot is not bound to loopback")
    broker_snapshot_id = _safe_id(broker["snapshot_id"], "broker snapshot_id")

    reconciliation, reconciliation_hash = _verified_json(
        Path(reconciliation_path), "reconciliation report"
    )
    _exact_keys(
        reconciliation,
        {
            "schema_version",
            "snapshot_id",
            "generated_at",
            "runtime_fingerprint",
            "execution_domain_scope",
            "account_scope",
            "database_path",
            "database_identity",
            "database_device",
            "database_inode",
            "portfolio_ids",
            "legacy_snapshot_hash",
            "broker_snapshot_id",
            "broker_snapshot_hash",
            "status",
            "authorizes_startup",
            "mutated_state",
            "managed_account_count",
            "broker_positions_count",
            "broker_open_orders_count",
        },
        "reconciliation report",
    )
    portfolio_ids = reconciliation["portfolio_ids"]
    if (
        reconciliation["schema_version"] != 1
        or reconciliation["status"] != RECONCILIATION_EVIDENCE_STATUS
        or reconciliation["authorizes_startup"] is not False
        or reconciliation["mutated_state"] is not False
        or reconciliation["managed_account_count"] != 1
        or reconciliation["broker_positions_count"] != 0
        or reconciliation["broker_open_orders_count"] != 0
        or reconciliation["runtime_fingerprint"] != runtime_fingerprint
        or reconciliation["execution_domain_scope"] != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not hmac.compare_digest(str(reconciliation["account_scope"]), account_scope)
        or reconciliation["database_path"] != str(database_path)
        or reconciliation["database_identity"] != database_identity
        or reconciliation["database_device"] != database_metadata.st_dev
        or reconciliation["database_inode"] != database_metadata.st_ino
        or reconciliation["broker_snapshot_id"] != broker_snapshot_id
        or not hmac.compare_digest(str(reconciliation["broker_snapshot_hash"]), broker_hash)
        or not isinstance(portfolio_ids, list)
        or not portfolio_ids
        or any(not isinstance(value, str) for value in portfolio_ids)
        or portfolio_ids != sorted(set(portfolio_ids))
    ):
        raise ExactStateBootstrapError("reconciliation report is not bound to runtime evidence")
    generated_at = _utc(reconciliation["generated_at"], "reconciliation generated_at")
    legacy_snapshot_hash = _hash(
        reconciliation["legacy_snapshot_hash"], "reconciliation legacy_snapshot_hash"
    )
    reconciliation_snapshot_id = _safe_id(
        reconciliation["snapshot_id"], "reconciliation snapshot_id"
    )

    if not isinstance(protective_mark_paths, Sequence) or isinstance(
        protective_mark_paths, (str, bytes)
    ):
        raise ExactStateBootstrapError("protective mark paths must be a sequence")
    marks: list[ExactBootstrapMarkEvidence] = []
    for mark_path in protective_mark_paths:
        mark, mark_hash = _verified_json(Path(mark_path), "protective mark")
        _exact_keys(
            mark,
            {
                "schema_version",
                "portfolio_id",
                "symbol",
                "price_text",
                "observed_at",
                "source",
                "source_event_id",
                "con_id",
                "runtime_fingerprint",
                "execution_domain_scope",
                "account_scope",
            },
            "protective mark",
        )
        if (
            mark["schema_version"] != 1
            or mark["source"] != MARK_EVIDENCE_SOURCE
            or mark["runtime_fingerprint"] != runtime_fingerprint
            or mark["execution_domain_scope"] != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
            or not hmac.compare_digest(str(mark["account_scope"]), account_scope)
            or type(mark["con_id"]) is not int
            or mark["con_id"] <= 0
        ):
            raise ExactStateBootstrapError("protective mark lacks PR3/runtime lineage")
        portfolio_id = _safe_id(mark["portfolio_id"], "mark portfolio_id")
        symbol = str(mark["symbol"]).strip().upper()
        if mark["symbol"] != symbol or not _SYMBOL.fullmatch(symbol):
            raise ExactStateBootstrapError("protective mark symbol is malformed")
        marks.append(
            ExactBootstrapMarkEvidence(
                portfolio_id=portfolio_id,
                symbol=symbol,
                price=_decimal(mark["price_text"], "mark price", positive=True),
                observed_at=_utc(mark["observed_at"], "mark observed_at"),
                source_event_id=_safe_id(mark["source_event_id"], "mark source_event_id"),
                con_id=mark["con_id"],
                artifact_path=str(Path(mark_path)),
                artifact_hash=mark_hash,
            )
        )
    mark_keys = [(mark.portfolio_id, mark.symbol) for mark in marks]
    if not marks or mark_keys != sorted(mark_keys) or len(mark_keys) != len(set(mark_keys)):
        raise ExactStateBootstrapError("protective marks must be nonempty, unique, and sorted")

    return ExactStateBootstrapEvidence(
        reconciliation_snapshot_id=reconciliation_snapshot_id,
        reconciliation_report_hash=reconciliation_hash,
        broker_snapshot_hash=broker_hash,
        legacy_snapshot_hash=legacy_snapshot_hash,
        runtime_fingerprint=runtime_fingerprint,
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=account_scope,
        database_path=str(database_path),
        database_identity=database_identity,
        database_device=database_metadata.st_dev,
        database_inode=database_metadata.st_ino,
        portfolio_ids=tuple(portfolio_ids),
        broker_observed_at=broker_observed_at,
        reconciliation_generated_at=generated_at,
        broker_position_count=0,
        broker_open_order_count=0,
        marks=tuple(marks),
        _producer_marker=_EVIDENCE_PRODUCER_MARKER,
    )


@dataclass(frozen=True, slots=True)
class ExactBootstrapPosition:
    symbol: str
    quantity: int
    cost_basis: Decimal
    mark_price: Decimal
    mark_observed_at: datetime
    mark_evidence_fingerprint: str

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not _SYMBOL.fullmatch(symbol):
            raise ExactStateBootstrapError("bootstrap position symbol is malformed")
        if isinstance(self.quantity, bool) or type(self.quantity) is not int or self.quantity == 0:
            raise ExactStateBootstrapError("bootstrap position quantity must be a nonzero integer")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self, "cost_basis", _decimal(self.cost_basis, "cost_basis", positive=True)
        )
        object.__setattr__(
            self, "mark_price", _decimal(self.mark_price, "mark_price", positive=True)
        )
        object.__setattr__(
            self,
            "mark_observed_at",
            _utc(self.mark_observed_at, "mark_observed_at"),
        )
        object.__setattr__(
            self,
            "mark_evidence_fingerprint",
            _hash(self.mark_evidence_fingerprint, "mark_evidence_fingerprint"),
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "cost_basis_text": decimal_to_fixed(self.cost_basis),
            "mark_evidence_fingerprint": self.mark_evidence_fingerprint,
            "mark_observed_at": utc_to_text(self.mark_observed_at),
            "mark_price_text": decimal_to_fixed(self.mark_price),
            "quantity": self.quantity,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class ExactBootstrapAccount:
    cash: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    daily_pnl_baseline: Decimal
    daily_pnl_date: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash", _decimal(self.cash, "cash"))
        object.__setattr__(self, "realized_pnl", _decimal(self.realized_pnl, "realized_pnl"))
        object.__setattr__(self, "daily_pnl", _decimal(self.daily_pnl, "daily_pnl"))
        object.__setattr__(
            self,
            "daily_pnl_baseline",
            _decimal(self.daily_pnl_baseline, "daily_pnl_baseline"),
        )
        if type(self.daily_pnl_date) is not date:
            raise ExactStateBootstrapError("daily_pnl_date must be an exact date")

    def public_dict(self) -> dict[str, str]:
        return {
            "cash_text": decimal_to_fixed(self.cash),
            "daily_pnl_baseline_text": decimal_to_fixed(self.daily_pnl_baseline),
            "daily_pnl_date": self.daily_pnl_date.isoformat(),
            "daily_pnl_text": decimal_to_fixed(self.daily_pnl),
            "realized_pnl_text": decimal_to_fixed(self.realized_pnl),
        }


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapCandidate:
    bootstrap_id: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    database_path: str
    database_identity: str
    reconciliation_snapshot_id: str
    reconciliation_report_hash: str
    broker_snapshot_hash: str
    legacy_snapshot_hash: str
    broker_position_count: int
    broker_open_order_count: int
    effective_at: datetime
    account: ExactBootstrapAccount
    positions: tuple[ExactBootstrapPosition, ...]
    schema_version: int = BOOTSTRAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BOOTSTRAP_SCHEMA_VERSION:
            raise ExactStateBootstrapError("unsupported bootstrap schema version")
        if not _BOOTSTRAP_ID.fullmatch(self.bootstrap_id):
            raise ExactStateBootstrapError("bootstrap_id is malformed")
        if self.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE:
            raise ExactStateBootstrapError("bootstrap is not bound to paper-simulator-v1")
        if not _ACCOUNT_SCOPE.fullmatch(self.account_scope):
            raise ExactStateBootstrapError("account_scope is malformed")
        _safe_id(self.portfolio_id, "portfolio_id")
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise ExactStateBootstrapError("database_path must be absolute and lexical")
        _safe_id(self.database_identity, "database_identity")
        _safe_id(self.reconciliation_snapshot_id, "reconciliation_snapshot_id")
        for value, label in (
            (self.reconciliation_report_hash, "reconciliation_report_hash"),
            (self.broker_snapshot_hash, "broker_snapshot_hash"),
            (self.legacy_snapshot_hash, "legacy_snapshot_hash"),
        ):
            _hash(value, label)
        if (
            type(self.broker_position_count) is not int
            or type(self.broker_open_order_count) is not int
            or self.broker_position_count != 0
            or self.broker_open_order_count != 0
        ):
            raise ExactStateBootstrapError(
                "IBKR paper account must have zero exposure and zero open orders"
            )
        effective_at = _utc(self.effective_at, "effective_at")
        object.__setattr__(self, "effective_at", effective_at)
        positions = tuple(self.positions)
        if not positions:
            raise ExactStateBootstrapError("bootstrap must describe every nonzero legacy position")
        if any(type(position) is not ExactBootstrapPosition for position in positions):
            raise ExactStateBootstrapError("bootstrap positions are malformed")
        symbols = [position.symbol for position in positions]
        if len(symbols) != len(set(symbols)) or symbols != sorted(symbols):
            raise ExactStateBootstrapError("bootstrap positions must be unique and sorted")
        for position in positions:
            age = effective_at - position.mark_observed_at
            if age < timedelta(0) or age > MAX_MARK_AGE:
                raise ExactStateBootstrapError("bootstrap protective mark is future or stale")
        object.__setattr__(self, "positions", positions)
        with localcontext() as context:
            context.prec = 64
            unrealized = sum(
                ((position.mark_price - position.cost_basis) * Decimal(position.quantity))
                for position in positions
            )
            expected_daily_pnl = (
                self.account.realized_pnl + unrealized - self.account.daily_pnl_baseline
            )
        if self.account.daily_pnl != expected_daily_pnl:
            raise ExactStateBootstrapError(
                "daily_pnl does not reconcile realized, marked unrealized, and baseline"
            )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account": self.account.public_dict(),
            "account_scope": self.account_scope,
            "bootstrap_id": self.bootstrap_id,
            "broker_open_order_count": self.broker_open_order_count,
            "broker_position_count": self.broker_position_count,
            "broker_snapshot_hash": self.broker_snapshot_hash,
            "database_identity": self.database_identity,
            "database_path": self.database_path,
            "effective_at": utc_to_text(self.effective_at),
            "execution_domain_scope": self.execution_domain_scope,
            "legacy_snapshot_hash": self.legacy_snapshot_hash,
            "portfolio_id": self.portfolio_id,
            "positions": [position.public_dict() for position in self.positions],
            "reconciliation_report_hash": self.reconciliation_report_hash,
            "reconciliation_snapshot_id": self.reconciliation_snapshot_id,
            "schema_version": self.schema_version,
        }

    def canonical_payload(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExactStateBootstrapCandidate":
        if not isinstance(raw, Mapping):
            raise ExactStateBootstrapError("bootstrap document must be an object")
        expected = {
            "account",
            "account_scope",
            "bootstrap_id",
            "broker_open_order_count",
            "broker_position_count",
            "broker_snapshot_hash",
            "database_identity",
            "database_path",
            "effective_at",
            "execution_domain_scope",
            "legacy_snapshot_hash",
            "portfolio_id",
            "positions",
            "reconciliation_report_hash",
            "reconciliation_snapshot_id",
            "schema_version",
        }
        if set(raw) != expected:
            raise ExactStateBootstrapError("bootstrap document fields are incomplete or unknown")
        account_raw = raw["account"]
        if not isinstance(account_raw, Mapping):
            raise ExactStateBootstrapError("bootstrap account is malformed")
        try:
            daily_pnl_date = date.fromisoformat(str(account_raw.get("daily_pnl_date")))
        except ValueError as exc:
            raise ExactStateBootstrapError("daily_pnl_date is invalid") from exc
        account = ExactBootstrapAccount(
            cash=account_raw.get("cash_text"),
            realized_pnl=account_raw.get("realized_pnl_text"),
            daily_pnl=account_raw.get("daily_pnl_text"),
            daily_pnl_baseline=account_raw.get("daily_pnl_baseline_text"),
            daily_pnl_date=daily_pnl_date,
        )
        positions_raw = raw["positions"]
        if not isinstance(positions_raw, list):
            raise ExactStateBootstrapError("bootstrap positions must be a list")
        positions = tuple(
            ExactBootstrapPosition(
                symbol=item["symbol"],
                quantity=item["quantity"],
                cost_basis=item["cost_basis_text"],
                mark_price=item["mark_price_text"],
                mark_observed_at=item["mark_observed_at"],
                mark_evidence_fingerprint=item["mark_evidence_fingerprint"],
            )
            for item in positions_raw
            if isinstance(item, Mapping)
        )
        if len(positions) != len(positions_raw):
            raise ExactStateBootstrapError("bootstrap position item is malformed")
        return cls(
            bootstrap_id=raw["bootstrap_id"],
            execution_domain_scope=raw["execution_domain_scope"],
            account_scope=raw["account_scope"],
            portfolio_id=raw["portfolio_id"],
            database_path=raw["database_path"],
            database_identity=raw["database_identity"],
            reconciliation_snapshot_id=raw["reconciliation_snapshot_id"],
            reconciliation_report_hash=raw["reconciliation_report_hash"],
            broker_snapshot_hash=raw["broker_snapshot_hash"],
            legacy_snapshot_hash=raw["legacy_snapshot_hash"],
            broker_position_count=raw["broker_position_count"],
            broker_open_order_count=raw["broker_open_order_count"],
            effective_at=raw["effective_at"],
            account=account,
            positions=positions,
            schema_version=raw["schema_version"],
        )


def assert_exact_state_bootstrap_evidence(
    candidate: ExactStateBootstrapCandidate,
    evidence: ExactStateBootstrapEvidence,
    runtime_contract: object,
) -> None:
    """Require every candidate claim to match independently loaded evidence."""

    if type(candidate) is not ExactStateBootstrapCandidate:
        raise ExactStateBootstrapError("candidate has the wrong type")
    if type(evidence) is not ExactStateBootstrapEvidence:
        raise ExactStateBootstrapError("bootstrap evidence has the wrong type")
    database_path, database_identity, runtime_fingerprint, account_scope = _runtime_contract_values(
        runtime_contract
    )
    if (
        candidate.database_path != str(database_path)
        or candidate.database_identity != database_identity
        or candidate.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not hmac.compare_digest(candidate.account_scope, account_scope)
        or evidence.database_path != str(database_path)
        or evidence.database_identity != database_identity
        or evidence.runtime_fingerprint != runtime_fingerprint
        or evidence.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not hmac.compare_digest(evidence.account_scope, account_scope)
        or candidate.portfolio_id not in evidence.portfolio_ids
        or candidate.reconciliation_snapshot_id != evidence.reconciliation_snapshot_id
        or not hmac.compare_digest(
            candidate.reconciliation_report_hash,
            evidence.reconciliation_report_hash,
        )
        or not hmac.compare_digest(candidate.broker_snapshot_hash, evidence.broker_snapshot_hash)
        or not hmac.compare_digest(candidate.legacy_snapshot_hash, evidence.legacy_snapshot_hash)
        or candidate.broker_position_count != evidence.broker_position_count
        or candidate.broker_open_order_count != evidence.broker_open_order_count
    ):
        raise ExactStateBootstrapError("candidate does not match verified bootstrap evidence")
    for observed, label in (
        (evidence.broker_observed_at, "broker snapshot"),
        (evidence.reconciliation_generated_at, "reconciliation report"),
    ):
        age = candidate.effective_at - observed
        if age < timedelta(0) or age > MAX_MARK_AGE:
            raise ExactStateBootstrapError(f"{label} is future or stale")
    evidence_marks = {(mark.portfolio_id, mark.symbol): mark for mark in evidence.marks}
    candidate_keys = {(candidate.portfolio_id, position.symbol) for position in candidate.positions}
    if set(evidence_marks) != candidate_keys:
        raise ExactStateBootstrapError("verified marks do not cover the candidate positions")
    for position in candidate.positions:
        mark = evidence_marks[(candidate.portfolio_id, position.symbol)]
        if (
            mark.price != position.mark_price
            or mark.observed_at != position.mark_observed_at
            or not hmac.compare_digest(
                mark.artifact_hash,
                position.mark_evidence_fingerprint,
            )
        ):
            raise ExactStateBootstrapError(
                f"candidate mark does not match verified evidence for {position.symbol}"
            )


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapReceipt:
    bootstrap_id: str
    candidate_fingerprint: str
    operator_action_id: str
    database_device: int
    database_inode: int
    committed_at: datetime


def _canonical_legacy_rows(
    account_rows: Iterable[sqlite3.Row],
    position_rows: Iterable[sqlite3.Row],
    trade_rows: Iterable[sqlite3.Row],
    equity_history_rows: Iterable[sqlite3.Row],
) -> str:
    payload = {
        "account": [list(row) for row in account_rows],
        "equity_history": [list(row) for row in equity_history_rows],
        "positions": [list(row) for row in position_rows],
        "trades": [list(row) for row in trade_rows],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def sqlite_table_evidence(
    connection: sqlite3.Connection,
) -> tuple[tuple[tuple[str, int], ...], tuple[tuple[str, str], ...]]:
    """Hash every application table and row in stable table/column order."""

    tables = connection.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    row_counts: list[tuple[str, int]] = []
    table_hashes: list[tuple[str, str]] = []
    for (table,) in tables:
        if not isinstance(table, str) or not _SAFE_ID.fullmatch(table):
            raise ExactStateBootstrapError("backup table name is malformed")
        quoted = '"' + table.replace('"', '""') + '"'
        columns = connection.execute(f"PRAGMA table_info({quoted})").fetchall()
        order = ",".join(str(index + 1) for index in range(len(columns)))
        query = f"SELECT * FROM {quoted}"
        if order:
            query += f" ORDER BY {order}"
        digest = hashlib.sha256()
        count = 0
        for row in connection.execute(query):
            values: list[object] = []
            for value in row:
                if isinstance(value, bytes):
                    values.append({"blob_hex": value.hex()})
                elif value is None or type(value) in {str, int, float}:
                    values.append(value)
                else:
                    raise ExactStateBootstrapError("backup contains an unsupported SQLite value")
            encoded = json.dumps(
                values,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            count += 1
        row_counts.append((table, count))
        table_hashes.append((table, digest.hexdigest()))
    required = {"account", "equity_history", "positions", "trades"}
    if not required.issubset(name for name, _ in row_counts):
        raise ExactStateBootstrapError("backup is missing required legacy ledger tables")
    return tuple(row_counts), tuple(table_hashes)


def inspect_legacy_state(database_path: Path) -> dict[str, object]:
    """Read a WAL-aware, identity-bound snapshot of every material legacy row."""

    path = lexical_path_preserving_leaf(Path(database_path))
    binding: SQLitePathBinding | None = None
    connection: sqlite3.Connection | None = None
    try:
        binding = SQLitePathBinding.open_for_initialization(path, create=False)
        binding.assert_path_identity()
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection_binding = binding.bind_sqlite_connection(
            sqlite_connection_file_identity(connection)
        )
        connection_binding.assert_connection_identity(sqlite_connection_file_identity(connection))
        connection.execute("BEGIN")
        required = {"account", "equity_history", "positions", "trades", "portfolios"}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not required.issubset(tables):
            raise ExactStateBootstrapError("legacy database schema is incomplete")
        account_rows = connection.execute(
            "SELECT portfolio_id,cash,equity,daily_pnl,realized_pnl,unrealized_pnl,timestamp "
            "FROM account ORDER BY portfolio_id"
        ).fetchall()
        position_rows = connection.execute(
            "SELECT portfolio_id,symbol,quantity,avg_cost,market_price,timestamp "
            "FROM positions WHERE quantity <> 0 ORDER BY portfolio_id,symbol"
        ).fetchall()
        trade_rows = connection.execute(
            "SELECT id,portfolio_id,symbol,side,quantity,price,notional,slippage,"
            "commission,pnl,timestamp FROM trades ORDER BY id"
        ).fetchall()
        equity_history_rows = connection.execute(
            "SELECT id,portfolio_id,date,equity,cash,positions_value,realized_pnl,"
            "unrealized_pnl,timestamp FROM equity_history ORDER BY id"
        ).fetchall()
        payload = _canonical_legacy_rows(
            account_rows,
            position_rows,
            trade_rows,
            equity_history_rows,
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or tuple(integrity) != ("ok",):
            raise ExactStateBootstrapError("legacy database failed SQLite integrity_check")
        row_counts = {
            "account": len(account_rows),
            "equity_history": len(equity_history_rows),
            "positions": len(position_rows),
            "trades": len(trade_rows),
        }
        connection_binding.assert_connection_identity(sqlite_connection_file_identity(connection))
        binding.assert_path_identity()
        connection.rollback()
        return {
            "account_rows": [dict(row) for row in account_rows],
            "position_rows": [dict(row) for row in position_rows],
            "snapshot_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "trade_count": len(trade_rows),
            "row_counts": row_counts,
            "database_device": binding.device,
            "database_inode": binding.inode,
        }
    except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
        raise ExactStateBootstrapError("legacy database cannot be inspected safely") from exc
    finally:
        if connection is not None:
            connection.close()
        if binding is not None:
            binding.close()

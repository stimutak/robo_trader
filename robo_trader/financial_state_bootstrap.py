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

from .bootstrap_evidence_auth import (
    AUTH_SUFFIX,
    AuthenticatedEvidenceReceipt,
    BootstrapEvidenceAuthenticationError,
    bootstrap_evidence_trust_public_dict,
    verify_receipt,
)
from .reconciliation.domain import IBKR_READ_ONLY_SCOPE, canonical_json, fingerprint
from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .safety.journal import SafetyJournal
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
_EVIDENCE_OBJECT_KEY = os.urandom(32)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_BOOTSTRAP_ID = re.compile(r"^pboot-[0-9a-f]{32}$")
_ACCOUNT_SCOPE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_PRINTABLE_EVENT_ID = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")


class ExactStateBootstrapError(ValueError):
    """A candidate cannot safely establish a sealed accounting epoch."""


class ExactStateSafetyJournalGuard:
    """Hold the exact reviewed journal against writes through bootstrap commit."""

    __slots__ = (
        "_binding",
        "_connection",
        "_connection_binding",
        "_released",
        "_snapshot",
    )

    def __init__(
        self,
        *,
        binding: SQLitePathBinding,
        connection: sqlite3.Connection,
        connection_binding: SQLitePathBinding,
        snapshot: tuple[tuple[int, str], ...],
    ) -> None:
        self._binding = binding
        self._connection = connection
        self._connection_binding = connection_binding
        self._snapshot = snapshot
        self._released = False

    def assert_unchanged(self) -> None:
        """Revalidate path, descriptor, and exact chain while write lock is held."""

        if self._released:
            raise ExactStateBootstrapError("safety journal guard was already released")
        try:
            self._connection_binding.assert_connection_identity(
                sqlite_connection_file_identity(self._connection)
            )
            self._binding.assert_path_identity()
            current = tuple(
                (int(row[0]), str(row[1]))
                for row in self._connection.execute(
                    "SELECT sequence,chain_hash FROM safety_journal_events ORDER BY sequence"
                ).fetchall()
            )
        except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
            raise ExactStateBootstrapError(
                "safety journal changed inside the bootstrap transaction boundary"
            ) from exc
        if current != self._snapshot:
            raise ExactStateBootstrapError(
                "safety journal changed inside the bootstrap transaction boundary"
            )

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        try:
            if self._connection.in_transaction:
                self._connection.rollback()
        finally:
            self._connection.close()
            self._binding.close()


class ExactStateBootstrapCommittedBackupInvalid(ExactStateBootstrapError):
    """The bootstrap committed, but its required rollback backup became invalid."""

    status = "COMMITTED_BACKUP_INVALID"
    mutated_state = True
    safe_retry = False

    def __init__(
        self,
        *,
        bootstrap_id: str,
        candidate_fingerprint: str,
        detail: str,
    ) -> None:
        self.bootstrap_id = bootstrap_id
        self.candidate_fingerprint = candidate_fingerprint
        self.detail = detail
        super().__init__(
            "bootstrap committed but rollback backup is invalid; no safe retry is permitted"
        )


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
    bundle_id: str
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
    safety_journal_path: str
    safety_journal_identity: str
    safety_journal_device: int
    safety_journal_inode: int
    safety_journal_last_sequence: int
    safety_journal_last_chain_hash: str
    terminal_settlement_count: int
    terminal_fill_count: int
    portfolio_ids: tuple[str, ...]
    broker_observed_at: datetime
    reconciliation_generated_at: datetime
    broker_position_count: int
    broker_open_order_count: int
    marks: tuple[ExactBootstrapMarkEvidence, ...]
    authentication_receipts: tuple[AuthenticatedEvidenceReceipt, ...]
    _producer_marker: object = field(repr=False, compare=False)
    _producer_digest: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _EVIDENCE_PRODUCER_MARKER:
            raise ExactStateBootstrapError(
                "bootstrap evidence must come from the verified artifact loader"
            )


def _evidence_object_digest(evidence: ExactStateBootstrapEvidence) -> str:
    payload = {
        "account_scope": evidence.account_scope,
        "authentication_receipts": [
            {
                "account_scope": receipt.account_scope,
                "artifact_kind": receipt.artifact_kind,
                "artifact_sha256": receipt.artifact_sha256,
                "expires_at": utc_to_text(receipt.expires_at),
                "issued_at": utc_to_text(receipt.issued_at),
                "producer_id": receipt.producer_id,
                "public_key_fingerprint": receipt.public_key_fingerprint,
                "receipt_id": receipt.receipt_id,
                "runtime_fingerprint": receipt.runtime_fingerprint,
            }
            for receipt in evidence.authentication_receipts
        ],
        "broker_observed_at": utc_to_text(evidence.broker_observed_at),
        "broker_open_order_count": evidence.broker_open_order_count,
        "broker_position_count": evidence.broker_position_count,
        "broker_snapshot_hash": evidence.broker_snapshot_hash,
        "bundle_id": evidence.bundle_id,
        "database_device": evidence.database_device,
        "database_identity": evidence.database_identity,
        "database_inode": evidence.database_inode,
        "database_path": evidence.database_path,
        "execution_domain_scope": evidence.execution_domain_scope,
        "legacy_snapshot_hash": evidence.legacy_snapshot_hash,
        "marks": [
            {
                "artifact_hash": mark.artifact_hash,
                "artifact_path": mark.artifact_path,
                "con_id": mark.con_id,
                "observed_at": utc_to_text(mark.observed_at),
                "portfolio_id": mark.portfolio_id,
                "price": decimal_to_fixed(mark.price),
                "source_event_id": mark.source_event_id,
                "symbol": mark.symbol,
            }
            for mark in evidence.marks
        ],
        "portfolio_ids": list(evidence.portfolio_ids),
        "reconciliation_generated_at": utc_to_text(evidence.reconciliation_generated_at),
        "reconciliation_report_hash": evidence.reconciliation_report_hash,
        "reconciliation_snapshot_id": evidence.reconciliation_snapshot_id,
        "runtime_fingerprint": evidence.runtime_fingerprint,
        "safety_journal_device": evidence.safety_journal_device,
        "safety_journal_identity": evidence.safety_journal_identity,
        "safety_journal_inode": evidence.safety_journal_inode,
        "safety_journal_last_chain_hash": evidence.safety_journal_last_chain_hash,
        "safety_journal_last_sequence": evidence.safety_journal_last_sequence,
        "safety_journal_path": evidence.safety_journal_path,
        "terminal_fill_count": evidence.terminal_fill_count,
        "terminal_settlement_count": evidence.terminal_settlement_count,
    }
    return hmac.new(
        _EVIDENCE_OBJECT_KEY,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


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
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ExactStateBootstrapError("backup receipt schema is unsupported")
        object.__setattr__(self, "created_at", _utc(self.created_at, "backup created_at"))
        for value, label in (
            (self.source_path, "backup source_path"),
            (self.backup_path, "backup backup_path"),
        ):
            path = Path(value)
            if not path.is_absolute() or str(path) != value:
                raise ExactStateBootstrapError(f"{label} must be absolute and lexical")
        for numeric_value, label in (
            (self.source_device, "backup source_device"),
            (self.source_inode, "backup source_inode"),
            (self.backup_device, "backup backup_device"),
            (self.backup_inode, "backup backup_inode"),
        ):
            if type(numeric_value) is not int or numeric_value < 0:
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


def _printable_event_id(value: object, label: str) -> str:
    if type(value) is not str or value != value.strip() or not _PRINTABLE_EVENT_ID.fullmatch(value):
        raise ExactStateBootstrapError(f"{label} is malformed")
    return value


def _exact_keys(raw: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise ExactStateBootstrapError(f"{label} fields are incomplete or unknown")


def _json_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise ExactStateBootstrapError(f"{label} must be a JSON string")
    return value


def _json_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExactStateBootstrapError(f"{label} must be a JSON integer")
    return value


def _json_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ExactStateBootstrapError(f"{label} must be a JSON boolean")
    return value


def _assert_fresh_against_wall_clock(observed_at: datetime, label: str) -> None:
    age = datetime.now(timezone.utc) - observed_at
    if age < timedelta(0) or age > MAX_MARK_AGE:
        raise ExactStateBootstrapError(f"{label} is future or stale against wall clock")


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
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
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
            or before.st_nlink != 1
            or after.st_nlink != 1
            or current.st_nlink != 1
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


def _verified_canonical_json(path: Path, label: str) -> tuple[Mapping[str, Any], str]:
    payload, _ = _verified_regular_file_bytes(path, label)
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExactStateBootstrapError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(raw, Mapping) or canonical_json(raw).encode("utf-8") != payload:
        raise ExactStateBootstrapError(f"{label} is not canonical producer JSON")
    return raw, hashlib.sha256(payload).hexdigest()


def _verify_artifact_authentication(
    *,
    artifact_path: Path,
    artifact_kind: str,
    artifact_hash: str,
    runtime_fingerprint: str,
    account_scope: str,
) -> AuthenticatedEvidenceReceipt:
    """Verify a detached producer receipt; plain JSON never grants authority."""

    receipt_path = artifact_path.with_name(artifact_path.name + AUTH_SUFFIX)
    receipt, _ = _verified_json(receipt_path, f"{artifact_kind} authentication receipt")
    try:
        return verify_receipt(
            raw=receipt,
            artifact_kind=artifact_kind,
            artifact_sha256=artifact_hash,
            runtime_fingerprint=runtime_fingerprint,
            account_scope=account_scope,
        )
    except BootstrapEvidenceAuthenticationError as exc:
        raise ExactStateBootstrapError(str(exc)) from exc


def verified_file_sha256(path: Path, label: str = "file") -> tuple[str, os.stat_result]:
    """Return the exact-byte hash and descriptor identity of an owner-only file."""

    payload, metadata = _verified_regular_file_bytes(path, label)
    return hashlib.sha256(payload).hexdigest(), metadata


def _assert_authentication_receipts_unconsumed(
    database_path: Path,
    receipts: Sequence[AuthenticatedEvidenceReceipt],
) -> None:
    """Read the append-only source ledger and reject a consumed nonce."""

    binding: SQLitePathBinding | None = None
    connection: sqlite3.Connection | None = None
    try:
        binding = SQLitePathBinding.open_for_initialization(database_path, create=False)
        connection = sqlite3.connect(database_path.as_uri() + "?mode=ro", uri=True)
        bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
        bound.assert_connection_identity(sqlite_connection_file_identity(connection))
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='exact_bootstrap_evidence_consumptions'"
        ).fetchone()
        if exists is None:
            return
        for receipt in receipts:
            consumed = connection.execute(
                "SELECT 1 FROM exact_bootstrap_evidence_consumptions WHERE receipt_id=?",
                (receipt.receipt_id,),
            ).fetchone()
            if consumed is not None:
                raise ExactStateBootstrapError("bootstrap evidence receipt replay is forbidden")
        bound.assert_connection_identity(sqlite_connection_file_identity(connection))
        binding.assert_path_identity()
    except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
        raise ExactStateBootstrapError(
            "bootstrap evidence consumption ledger cannot be checked safely"
        ) from exc
    finally:
        if connection is not None:
            connection.close()
        if binding is not None:
            binding.close()


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

    try:
        bootstrap_evidence_trust_public_dict()
    except BootstrapEvidenceAuthenticationError as exc:
        raise ExactStateBootstrapError(str(exc)) from exc
    database_path, database_identity, runtime_fingerprint, account_scope = _runtime_contract_values(
        expected_runtime_contract
    )
    try:
        database_metadata = os.lstat(database_path)
    except OSError as exc:
        raise ExactStateBootstrapError("runtime database identity cannot be inspected") from exc
    if (
        stat.S_ISLNK(database_metadata.st_mode)
        or not stat.S_ISREG(database_metadata.st_mode)
        or database_metadata.st_nlink != 1
    ):
        raise ExactStateBootstrapError(
            "runtime database must be a single-link non-symlink regular file"
        )

    broker, broker_hash = _verified_canonical_json(Path(broker_snapshot_path), "broker snapshot")
    _exact_keys(
        broker,
        {
            "completed_order_collection_scope",
            "execution_collection_scope",
            "purpose",
            "snapshot",
        },
        "broker producer result",
    )
    completed_scope = broker["completed_order_collection_scope"]
    execution_scope = broker["execution_collection_scope"]
    if type(completed_scope) is not dict or type(execution_scope) is not dict:
        raise ExactStateBootstrapError("broker collection scope is malformed")
    _exact_keys(
        completed_scope,
        {
            "api_method",
            "api_only",
            "broker_time_after",
            "broker_time_before",
            "full_history",
            "kind",
            "client_scope",
            "request_count",
            "request_completed_at",
            "request_started_at",
            "retention_scope",
            "stability_check",
            "verification_completed_at",
            "verification_started_at",
        },
        "broker completed-order scope",
    )
    _exact_keys(
        execution_scope,
        {
            "commission_scope",
            "end_at",
            "full_history",
            "kind",
            "retention_scope",
            "start_at",
        },
        "broker execution scope",
    )
    if (
        _json_string(broker["purpose"], "broker producer purpose") != "bootstrap-broker-signing-v1"
        or _json_string(completed_scope["kind"], "completed-order scope kind")
        != "ibkr_current_retained_completed_orders"
        or _json_string(completed_scope["api_method"], "completed-order API method")
        != "reqCompletedOrders"
        or _json_string(completed_scope["retention_scope"], "completed-order retention scope")
        != "current_tws_or_gateway_retained_set"
        or _json_int(completed_scope["request_count"], "completed-order request_count") != 2
        or _json_string(completed_scope["stability_check"], "completed-order stability_check")
        != "identical_second_read"
        or _json_bool(completed_scope["api_only"], "completed-order api_only") is not False
        or _json_string(completed_scope["client_scope"], "completed-order client_scope")
        != "api_and_manual_orders_visible_to_current_tws_session"
        or _json_bool(completed_scope["full_history"], "completed-order full_history") is not False
        or _json_string(execution_scope["kind"], "execution scope kind")
        != "broker_date_since_midnight"
        or _json_string(execution_scope["retention_scope"], "execution retention scope")
        != "ibkr_gateway_broker_date_since_midnight"
        or _json_string(execution_scope["commission_scope"], "execution commission scope")
        != "matching_callbacks_for_returned_executions"
        or _json_bool(execution_scope["full_history"], "execution full_history") is not False
    ):
        raise ExactStateBootstrapError("broker collection scope is not bounded/current")
    scope_times = [
        _utc(completed_scope[field], f"completed-order {field}")
        for field in (
            "request_started_at",
            "request_completed_at",
            "verification_started_at",
            "verification_completed_at",
        )
    ]
    if scope_times != sorted(scope_times):
        raise ExactStateBootstrapError("broker completed-order scope is time-inconsistent")
    scope_broker_time_before = _utc(
        completed_scope["broker_time_before"], "completed-order broker_time_before"
    )
    scope_broker_time_after = _utc(
        completed_scope["broker_time_after"], "completed-order broker_time_after"
    )
    execution_start = _utc(execution_scope["start_at"], "execution scope start_at")
    execution_end = _utc(execution_scope["end_at"], "execution scope end_at")
    if (
        scope_times[-1] - scope_times[0] > timedelta(seconds=60)
        or scope_times[0] < scope_broker_time_before - timedelta(seconds=120)
        or scope_times[-1] > scope_broker_time_after + timedelta(seconds=120)
        or execution_start
        != scope_broker_time_before.replace(hour=0, minute=0, second=0, microsecond=0)
        or not scope_broker_time_before <= execution_end <= scope_broker_time_after
    ):
        raise ExactStateBootstrapError("broker collection scope is unbounded")
    broker = broker["snapshot"]
    if type(broker) is not dict:
        raise ExactStateBootstrapError("broker normalized snapshot is malformed")
    _exact_keys(
        broker,
        {
            "account",
            "completeness",
            "collection_evidence",
            "executions",
            "observed_from",
            "observed_through",
            "orders",
            "positions",
            "retrieved_at",
            "schema_version",
            "source_scope",
        },
        "broker snapshot",
    )
    account = broker["account"]
    completeness = broker["completeness"]
    collections = broker["collection_evidence"]
    if type(account) is not dict or type(completeness) is not dict:
        raise ExactStateBootstrapError("broker account/completeness evidence is malformed")
    _exact_keys(
        account,
        {
            "account_alias",
            "account_scope",
            "account_type",
            "base_currency",
            "buying_power",
            "observed_at",
            "schema_version",
            "source_scope",
            "total_cash",
        },
        "broker account",
    )
    _exact_keys(
        completeness,
        {
            "account",
            "positions",
            "open_orders",
            "completed_orders",
            "executions",
            "commissions",
        },
        "broker completeness",
    )
    expected_collections = {
        "positions",
        "open_orders",
        "completed_orders",
        "executions",
        "commissions",
    }
    if type(collections) is not list or len(collections) != len(expected_collections):
        raise ExactStateBootstrapError("broker collection evidence is incomplete")
    collection_names: set[str] = set()
    broker_collection_ids: list[str] = []
    collection_observed_times: list[datetime] = []
    for collection in collections:
        if type(collection) is not dict:
            raise ExactStateBootstrapError("broker collection evidence is malformed")
        _exact_keys(
            collection,
            {
                "account_scope",
                "collection",
                "evidence_id",
                "observed_at",
                "result_count",
                "schema_version",
                "source_scope",
            },
            "broker collection evidence",
        )
        name = _json_string(collection["collection"], "broker collection")
        collection_names.add(name)
        evidence_id = _json_string(collection["evidence_id"], "broker collection evidence_id")
        if not re.fullmatch(r"broker-collection-v1-[0-9a-f]{64}", evidence_id):
            raise ExactStateBootstrapError("broker collection evidence identity is malformed")
        broker_collection_ids.append(evidence_id)
        collection_observed_times.append(
            _utc(collection["observed_at"], "broker collection observed_at")
        )
        if (
            not hmac.compare_digest(
                _json_string(collection["account_scope"], "broker collection account_scope"),
                account_scope,
            )
            or _json_int(collection["result_count"], "broker collection result_count") != 0
            or _json_int(collection["schema_version"], "broker collection schema_version") != 1
            or _json_string(collection["source_scope"], "broker collection source_scope")
            != IBKR_READ_ONLY_SCOPE
        ):
            raise ExactStateBootstrapError("broker collection evidence is not zero/read-only")
    broker_observed_at = _utc(broker["retrieved_at"], "broker retrieved_at")
    broker_time_before = _utc(broker["observed_from"], "broker observed_from")
    broker_time_after = _utc(broker["observed_through"], "broker observed_through")
    account_observed_at = _utc(account["observed_at"], "broker account observed_at")
    if (
        scope_broker_time_before != broker_time_before
        or scope_broker_time_after != broker_time_after
        or not broker_time_before <= account_observed_at <= broker_time_after <= broker_observed_at
        or any(
            not broker_time_before <= observed_at <= broker_time_after
            for observed_at in collection_observed_times
        )
    ):
        raise ExactStateBootstrapError("broker snapshot time bounds are inconsistent")
    _assert_fresh_against_wall_clock(broker_observed_at, "broker snapshot")
    if (
        _json_int(broker["schema_version"], "broker schema_version") != 1
        or _json_string(broker["source_scope"], "broker source_scope") != IBKR_READ_ONLY_SCOPE
        or _json_int(account["schema_version"], "broker account schema_version") != 1
        or _json_string(account["source_scope"], "broker account source_scope")
        != IBKR_READ_ONLY_SCOPE
        or _json_string(account["account_type"], "broker account_type") != "paper"
        or _json_string(account["account_alias"], "broker account_alias")
        != getattr(expected_runtime_contract, "account_alias", None)
        or not hmac.compare_digest(
            _json_string(account["account_scope"], "broker account_scope"), account_scope
        )
        or set(completeness) != expected_collections | {"account"}
        or any(
            _json_bool(value, "broker completeness flag") is not True
            for value in completeness.values()
        )
        or collection_names != expected_collections
        or len(broker_collection_ids) != len(set(broker_collection_ids))
    ):
        raise ExactStateBootstrapError("broker snapshot is not bound to runtime evidence")
    if (
        not re.fullmatch(
            r"[A-Z]{3}", _json_string(account["base_currency"], "broker base_currency")
        )
        or _decimal(account["buying_power"], "broker buying_power") < 0
    ):
        raise ExactStateBootstrapError("broker account financial evidence is malformed")
    _decimal(account["total_cash"], "broker total_cash")
    if (
        type(broker["positions"]) is not list
        or type(broker["orders"]) is not list
        or type(broker["executions"]) is not list
        or broker["positions"] != []
        or broker["orders"] != []
        or broker["executions"] != []
    ):
        raise ExactStateBootstrapError("broker snapshot does not prove zero paper exposure")
    broker_snapshot_id = fingerprint("broker-reconciliation-v1", broker)
    broker_snapshot_hash = hashlib.sha256(canonical_json(broker).encode("utf-8")).hexdigest()
    broker_authenticated_at = _verify_artifact_authentication(
        artifact_path=Path(broker_snapshot_path),
        artifact_kind="broker_snapshot",
        artifact_hash=broker_hash,
        runtime_fingerprint=runtime_fingerprint,
        account_scope=account_scope,
    )

    reconciliation, reconciliation_hash = _verified_canonical_json(
        Path(reconciliation_path), "reconciliation report"
    )
    _exact_keys(
        reconciliation,
        {
            "schema_version",
            "snapshot_id",
            "bundle_id",
            "generated_at",
            "runtime_fingerprint",
            "execution_domain_scope",
            "account_scope",
            "database_path",
            "database_identity",
            "database_device",
            "database_inode",
            "safety_journal_path",
            "safety_journal_identity",
            "safety_journal_device",
            "safety_journal_inode",
            "safety_journal_last_sequence",
            "safety_journal_last_chain_hash",
            "terminal_settlement_count",
            "terminal_fill_count",
            "portfolio_ids",
            "legacy_snapshot_hash",
            "broker_snapshot_id",
            "broker_snapshot_hash",
            "broker_artifact_hash",
            "broker_collection_evidence_ids",
            "broker_receipt_id",
            "broker_public_key_fingerprint",
            "broker_verdict_id",
            "broker_verdict_hash",
            "comparison_coverage",
            "reconciliation_status",
            "local_simulator_positions_count",
            "local_position_identities",
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
    bundle_id = _json_string(reconciliation["bundle_id"], "reconciliation bundle_id")
    if not re.fullmatch(r"bootstrap-evidence-bundle-v1-[0-9a-f]{64}", bundle_id):
        raise ExactStateBootstrapError("reconciliation bundle identity is malformed")
    coverage = reconciliation["comparison_coverage"]
    collection_ids = reconciliation["broker_collection_evidence_ids"]
    local_position_identities = reconciliation["local_position_identities"]
    safety_journal_path = _json_string(
        reconciliation["safety_journal_path"],
        "reconciliation safety_journal_path",
    )
    safety_journal_identity = _json_string(
        reconciliation["safety_journal_identity"],
        "reconciliation safety_journal_identity",
    )
    expected_journal_path = getattr(expected_runtime_contract, "safety_journal_path", None)
    expected_journal_identity = getattr(
        expected_runtime_contract,
        "safety_journal_identity",
        None,
    )
    if (
        type(expected_journal_path) is not str
        or type(expected_journal_identity) is not str
        or safety_journal_path != expected_journal_path
        or safety_journal_identity != expected_journal_identity
    ):
        raise ExactStateBootstrapError("reconciliation safety journal is not bound to the runtime")
    try:
        safety_journal_metadata = os.lstat(safety_journal_path)
    except OSError as exc:
        raise ExactStateBootstrapError(
            "runtime safety journal identity cannot be inspected"
        ) from exc
    if (
        stat.S_ISLNK(safety_journal_metadata.st_mode)
        or not stat.S_ISREG(safety_journal_metadata.st_mode)
        or safety_journal_metadata.st_nlink != 1
    ):
        raise ExactStateBootstrapError(
            "runtime safety journal must be a single-link non-symlink regular file"
        )
    journal_sequence = _json_int(
        reconciliation["safety_journal_last_sequence"],
        "reconciliation safety journal last sequence",
    )
    journal_chain_hash = _hash(
        reconciliation["safety_journal_last_chain_hash"],
        "reconciliation safety journal last chain hash",
    )
    terminal_settlement_count = _json_int(
        reconciliation["terminal_settlement_count"],
        "reconciliation terminal settlement count",
    )
    terminal_fill_count = _json_int(
        reconciliation["terminal_fill_count"],
        "reconciliation terminal fill count",
    )
    try:
        journal_state = SafetyJournal(Path(safety_journal_path)).replay_and_bind_runtime_path(
            expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
            expected_account_scope=account_scope,
        )
    except Exception as exc:
        raise ExactStateBootstrapError("runtime safety journal cannot be verified") from exc
    if type(local_position_identities) is not list:
        raise ExactStateBootstrapError("reconciliation local position identities are malformed")
    normalized_position_identities: list[tuple[str, str]] = []
    for identity in local_position_identities:
        if type(identity) is not list or len(identity) != 2:
            raise ExactStateBootstrapError("reconciliation local position identities are malformed")
        portfolio_id = _safe_id(identity[0], "reconciliation position portfolio_id")
        symbol = _json_string(identity[1], "reconciliation position symbol")
        if symbol != symbol.upper() or not _SYMBOL.fullmatch(symbol):
            raise ExactStateBootstrapError("reconciliation local position identities are malformed")
        normalized_position_identities.append((portfolio_id, symbol))
    expected_coverage = {
        "broker_account",
        "broker_positions",
        "broker_open_orders",
        "broker_completed_orders",
        "broker_executions",
        "broker_commissions",
        "ledger_positions",
        "ledger_orders",
        "ledger_executions",
        "ledger_cash",
    }
    if (
        _json_int(reconciliation["schema_version"], "reconciliation schema_version") != 1
        or _json_string(reconciliation["status"], "reconciliation status")
        != RECONCILIATION_EVIDENCE_STATUS
        or _json_bool(reconciliation["authorizes_startup"], "reconciliation authorizes_startup")
        is not False
        or _json_bool(reconciliation["mutated_state"], "reconciliation mutated_state") is not False
        or _json_int(
            reconciliation["managed_account_count"], "reconciliation managed_account_count"
        )
        != 1
        or _json_int(
            reconciliation["broker_positions_count"], "reconciliation broker_positions_count"
        )
        != 0
        or _json_int(
            reconciliation["broker_open_orders_count"],
            "reconciliation broker_open_orders_count",
        )
        != 0
        or _json_int(
            reconciliation["local_simulator_positions_count"],
            "reconciliation local simulator positions count",
        )
        != len(normalized_position_identities)
        or normalized_position_identities != sorted(set(normalized_position_identities))
        or type(coverage) is not dict
        or set(coverage) != expected_coverage
        or any(
            _json_bool(value, "reconciliation coverage flag") is not True
            for value in coverage.values()
        )
        or _json_string(
            reconciliation["reconciliation_status"],
            "reconciliation status",
        )
        not in {"passed", "degraded"}
        or type(collection_ids) is not list
        or len(collection_ids) != 5
        or any(type(value) is not str for value in collection_ids)
        or collection_ids != sorted(set(collection_ids))
        or collection_ids != sorted(broker_collection_ids)
        or _json_string(reconciliation["runtime_fingerprint"], "reconciliation runtime_fingerprint")
        != runtime_fingerprint
        or _json_string(
            reconciliation["execution_domain_scope"],
            "reconciliation execution_domain_scope",
        )
        != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or not hmac.compare_digest(
            _json_string(reconciliation["account_scope"], "reconciliation account_scope"),
            account_scope,
        )
        or _json_string(reconciliation["database_path"], "reconciliation database_path")
        != str(database_path)
        or _json_string(reconciliation["database_identity"], "reconciliation database_identity")
        != database_identity
        or _json_int(reconciliation["database_device"], "reconciliation database_device")
        != database_metadata.st_dev
        or _json_int(reconciliation["database_inode"], "reconciliation database_inode")
        != database_metadata.st_ino
        or _json_int(
            reconciliation["safety_journal_device"],
            "reconciliation safety journal device",
        )
        != safety_journal_metadata.st_dev
        or _json_int(
            reconciliation["safety_journal_inode"],
            "reconciliation safety journal inode",
        )
        != safety_journal_metadata.st_ino
        or journal_sequence != journal_state.last_sequence
        or not hmac.compare_digest(journal_chain_hash, journal_state.last_chain_hash)
        or terminal_fill_count > terminal_settlement_count
        or _json_string(reconciliation["broker_snapshot_id"], "reconciliation broker_snapshot_id")
        != broker_snapshot_id
        or not hmac.compare_digest(
            _hash(reconciliation["broker_snapshot_hash"], "reconciliation broker_snapshot_hash"),
            broker_snapshot_hash,
        )
        or not hmac.compare_digest(
            _hash(reconciliation["broker_artifact_hash"], "reconciliation broker_artifact_hash"),
            broker_hash,
        )
        or _safe_id(reconciliation["broker_receipt_id"], "reconciliation broker receipt_id")
        != broker_authenticated_at.receipt_id
        or not hmac.compare_digest(
            _hash(
                reconciliation["broker_public_key_fingerprint"],
                "reconciliation broker public key fingerprint",
            ),
            broker_authenticated_at.public_key_fingerprint,
        )
        or type(portfolio_ids) is not list
        or not portfolio_ids
        or any(type(value) is not str for value in portfolio_ids)
        or portfolio_ids != sorted(set(portfolio_ids))
        or any(
            portfolio_id not in portfolio_ids for portfolio_id, _ in normalized_position_identities
        )
    ):
        raise ExactStateBootstrapError("reconciliation report is not bound to runtime evidence")
    generated_at = _utc(reconciliation["generated_at"], "reconciliation generated_at")
    _assert_fresh_against_wall_clock(generated_at, "reconciliation report")
    legacy_snapshot_hash = _hash(
        reconciliation["legacy_snapshot_hash"], "reconciliation legacy_snapshot_hash"
    )
    reconciliation_snapshot_id = _safe_id(
        reconciliation["snapshot_id"], "reconciliation snapshot_id"
    )
    if not re.fullmatch(
        r"reconciliation-verdict-v1-[0-9a-f]{64}",
        _json_string(reconciliation["broker_verdict_id"], "broker verdict_id"),
    ) or not _HEX_64.fullmatch(
        _json_string(reconciliation["broker_verdict_hash"], "broker verdict_hash")
    ):
        raise ExactStateBootstrapError("reconciliation broker verdict binding is malformed")
    reconciliation_binding = dict(reconciliation)
    reconciliation_binding.pop("snapshot_id")
    if reconciliation_snapshot_id != fingerprint(
        "bootstrap-reconciliation-v1", reconciliation_binding
    ):
        raise ExactStateBootstrapError(
            "reconciliation snapshot identity is not bound to its canonical payload"
        )
    reconciliation_authenticated_at = _verify_artifact_authentication(
        artifact_path=Path(reconciliation_path),
        artifact_kind="reconciliation_report",
        artifact_hash=reconciliation_hash,
        runtime_fingerprint=runtime_fingerprint,
        account_scope=account_scope,
    )

    if not isinstance(protective_mark_paths, Sequence) or isinstance(
        protective_mark_paths, (str, bytes)
    ):
        raise ExactStateBootstrapError("protective mark paths must be a sequence")
    marks: list[ExactBootstrapMarkEvidence] = []
    authentication_receipts = [broker_authenticated_at, reconciliation_authenticated_at]
    for mark_path in protective_mark_paths:
        mark, mark_hash = _verified_canonical_json(Path(mark_path), "protective mark")
        _exact_keys(
            mark,
            {
                "schema_version",
                "bundle_id",
                "reconciliation_snapshot_id",
                "broker_snapshot_id",
                "broker_snapshot_hash",
                "broker_artifact_hash",
                "broker_receipt_id",
                "broker_public_key_fingerprint",
                "portfolio_id",
                "symbol",
                "price_text",
                "observed_at",
                "source",
                "source_event_id",
                "con_id",
                "transport_generation",
                "protective_quote_id",
                "protective_quote_source",
                "runtime_fingerprint",
                "execution_domain_scope",
                "account_scope",
                "database_identity",
                "database_device",
                "database_inode",
                "authorizes_startup",
                "mutated_state",
            },
            "protective mark",
        )
        if (
            _json_int(mark["schema_version"], "mark schema_version") != 1
            or _json_string(mark["source"], "mark source") != MARK_EVIDENCE_SOURCE
            or _json_bool(mark["authorizes_startup"], "mark authorizes_startup") is not False
            or _json_bool(mark["mutated_state"], "mark mutated_state") is not False
            or _json_string(mark["protective_quote_source"], "mark protective_quote_source")
            != "live-broker"
            or _json_string(mark["runtime_fingerprint"], "mark runtime_fingerprint")
            != runtime_fingerprint
            or _json_string(mark["bundle_id"], "mark bundle_id") != bundle_id
            or _json_string(
                mark["reconciliation_snapshot_id"],
                "mark reconciliation_snapshot_id",
            )
            != reconciliation_snapshot_id
            or _json_string(mark["broker_snapshot_id"], "mark broker_snapshot_id")
            != broker_snapshot_id
            or not hmac.compare_digest(
                _hash(mark["broker_snapshot_hash"], "mark broker_snapshot_hash"),
                broker_snapshot_hash,
            )
            or not hmac.compare_digest(
                _hash(mark["broker_artifact_hash"], "mark broker_artifact_hash"),
                broker_hash,
            )
            or _safe_id(mark["broker_receipt_id"], "mark broker receipt_id")
            != broker_authenticated_at.receipt_id
            or not hmac.compare_digest(
                _hash(
                    mark["broker_public_key_fingerprint"],
                    "mark broker public key fingerprint",
                ),
                broker_authenticated_at.public_key_fingerprint,
            )
            or _json_string(mark["execution_domain_scope"], "mark execution_domain_scope")
            != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
            or not hmac.compare_digest(
                _json_string(mark["account_scope"], "mark account_scope"), account_scope
            )
            or _json_int(mark["con_id"], "mark con_id", minimum=1) <= 0
            or _json_string(mark["database_identity"], "mark database_identity")
            != database_identity
            or _json_int(mark["database_device"], "mark database_device")
            != database_metadata.st_dev
            or _json_int(mark["database_inode"], "mark database_inode") != database_metadata.st_ino
        ):
            raise ExactStateBootstrapError("protective mark lacks PR3/runtime lineage")
        portfolio_id = _safe_id(mark["portfolio_id"], "mark portfolio_id")
        raw_symbol = _json_string(mark["symbol"], "mark symbol")
        symbol = raw_symbol.strip().upper()
        if raw_symbol != symbol or not _SYMBOL.fullmatch(symbol):
            raise ExactStateBootstrapError("protective mark symbol is malformed")
        mark_observed_at = _utc(mark["observed_at"], "mark observed_at")
        _assert_fresh_against_wall_clock(mark_observed_at, "protective mark")
        protective_quote_id = _json_string(mark["protective_quote_id"], "mark protective_quote_id")
        if not re.fullmatch(r"quote:v1:[0-9a-f]{64}", protective_quote_id):
            raise ExactStateBootstrapError("protective mark quote identity is malformed")
        _printable_event_id(mark["transport_generation"], "mark transport_generation")
        marks.append(
            ExactBootstrapMarkEvidence(
                portfolio_id=portfolio_id,
                symbol=symbol,
                price=_decimal(mark["price_text"], "mark price", positive=True),
                observed_at=mark_observed_at,
                source_event_id=_printable_event_id(
                    mark["source_event_id"], "mark source_event_id"
                ),
                con_id=mark["con_id"],
                artifact_path=str(Path(mark_path)),
                artifact_hash=mark_hash,
            )
        )
        authentication_receipts.append(
            _verify_artifact_authentication(
                artifact_path=Path(mark_path),
                artifact_kind="protective_mark",
                artifact_hash=mark_hash,
                runtime_fingerprint=runtime_fingerprint,
                account_scope=account_scope,
            )
        )
    mark_keys = [(mark.portfolio_id, mark.symbol) for mark in marks]
    if mark_keys != sorted(mark_keys) or len(mark_keys) != len(set(mark_keys)):
        raise ExactStateBootstrapError("protective marks must be unique and sorted")
    if mark_keys != normalized_position_identities:
        raise ExactStateBootstrapError(
            "protective marks do not exactly cover reconciled local positions"
        )
    _assert_authentication_receipts_unconsumed(database_path, authentication_receipts)

    evidence = ExactStateBootstrapEvidence(
        reconciliation_snapshot_id=reconciliation_snapshot_id,
        bundle_id=bundle_id,
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
        safety_journal_path=safety_journal_path,
        safety_journal_identity=safety_journal_identity,
        safety_journal_device=safety_journal_metadata.st_dev,
        safety_journal_inode=safety_journal_metadata.st_ino,
        safety_journal_last_sequence=journal_sequence,
        safety_journal_last_chain_hash=journal_chain_hash,
        terminal_settlement_count=terminal_settlement_count,
        terminal_fill_count=terminal_fill_count,
        portfolio_ids=tuple(portfolio_ids),
        broker_observed_at=broker_observed_at,
        reconciliation_generated_at=generated_at,
        broker_position_count=0,
        broker_open_order_count=0,
        marks=tuple(marks),
        authentication_receipts=tuple(authentication_receipts),
        _producer_marker=_EVIDENCE_PRODUCER_MARKER,
    )
    object.__setattr__(evidence, "_producer_digest", _evidence_object_digest(evidence))
    return evidence


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
        if type(self.schema_version) is not int or self.schema_version != BOOTSTRAP_SCHEMA_VERSION:
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
        _exact_keys(
            account_raw,
            {
                "cash_text",
                "realized_pnl_text",
                "daily_pnl_text",
                "daily_pnl_baseline_text",
                "daily_pnl_date",
            },
            "bootstrap account",
        )
        try:
            daily_pnl_date = date.fromisoformat(
                _json_string(account_raw["daily_pnl_date"], "daily_pnl_date")
            )
        except ValueError as exc:
            raise ExactStateBootstrapError("daily_pnl_date is invalid") from exc
        account = ExactBootstrapAccount(
            cash=account_raw["cash_text"],
            realized_pnl=account_raw["realized_pnl_text"],
            daily_pnl=account_raw["daily_pnl_text"],
            daily_pnl_baseline=account_raw["daily_pnl_baseline_text"],
            daily_pnl_date=daily_pnl_date,
        )
        positions_raw = raw["positions"]
        if not isinstance(positions_raw, list):
            raise ExactStateBootstrapError("bootstrap positions must be a list")
        positions_list: list[ExactBootstrapPosition] = []
        for item in positions_raw:
            if not isinstance(item, Mapping):
                raise ExactStateBootstrapError("bootstrap position item is malformed")
            _exact_keys(
                item,
                {
                    "symbol",
                    "quantity",
                    "cost_basis_text",
                    "mark_price_text",
                    "mark_observed_at",
                    "mark_evidence_fingerprint",
                },
                "bootstrap position",
            )
            positions_list.append(
                ExactBootstrapPosition(
                    symbol=item["symbol"],
                    quantity=item["quantity"],
                    cost_basis=item["cost_basis_text"],
                    mark_price=item["mark_price_text"],
                    mark_observed_at=item["mark_observed_at"],
                    mark_evidence_fingerprint=item["mark_evidence_fingerprint"],
                )
            )
        positions = tuple(positions_list)
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


def acquire_exact_state_safety_journal_guard(
    evidence: ExactStateBootstrapEvidence,
    runtime_contract: object,
) -> ExactStateSafetyJournalGuard:
    """Replay and write-lock the reviewed journal until bootstrap finishes."""

    if type(evidence) is not ExactStateBootstrapEvidence:
        raise ExactStateBootstrapError("bootstrap evidence has the wrong type")
    journal_path_value = getattr(runtime_contract, "safety_journal_path", None)
    journal_identity = getattr(runtime_contract, "safety_journal_identity", None)
    execution_scope = getattr(runtime_contract, "safety_execution_domain_scope", None)
    account_scope = getattr(runtime_contract, "safety_account_scope", None)
    if (
        type(journal_path_value) is not str
        or journal_path_value != evidence.safety_journal_path
        or type(journal_identity) is not str
        or journal_identity != evidence.safety_journal_identity
        or execution_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
        or type(account_scope) is not str
    ):
        raise ExactStateBootstrapError("runtime safety journal does not match reviewed evidence")
    journal_path = Path(journal_path_value)
    binding: SQLitePathBinding | None = None
    connection: sqlite3.Connection | None = None
    try:
        binding = SQLitePathBinding.open_for_initialization(journal_path, create=False)
        if (binding.device, binding.inode) != (
            evidence.safety_journal_device,
            evidence.safety_journal_inode,
        ):
            raise ExactStateBootstrapError("runtime safety journal identity changed")
        connection = sqlite3.connect(
            journal_path.as_uri() + "?mode=rw",
            uri=True,
            timeout=1.0,
            isolation_level=None,
        )
        connection.execute("PRAGMA busy_timeout=1000")
        connection_binding = binding.bind_sqlite_connection(
            sqlite_connection_file_identity(connection)
        )
        connection.execute("BEGIN IMMEDIATE")
        replay = SafetyJournal(journal_path).replay_and_bind_runtime_path(
            expected_execution_domain_scope=execution_scope,
            expected_account_scope=account_scope,
        )
        snapshot = tuple(
            (int(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT sequence,chain_hash FROM safety_journal_events ORDER BY sequence"
            ).fetchall()
        )
        if (
            replay.last_sequence != evidence.safety_journal_last_sequence
            or not hmac.compare_digest(
                replay.last_chain_hash,
                evidence.safety_journal_last_chain_hash,
            )
            or len(snapshot) != replay.last_sequence
            or (snapshot and snapshot[-1] != (replay.last_sequence, replay.last_chain_hash))
            or (not snapshot and replay.last_chain_hash != "0" * 64)
        ):
            raise ExactStateBootstrapError("runtime safety journal changed after review")
        guard = ExactStateSafetyJournalGuard(
            binding=binding,
            connection=connection,
            connection_binding=connection_binding,
            snapshot=snapshot,
        )
        guard.assert_unchanged()
        binding = None
        connection = None
        return guard
    except (OSError, sqlite3.Error, SQLiteIdentityError) as exc:
        raise ExactStateBootstrapError(
            "runtime safety journal cannot be locked and revalidated"
        ) from exc
    finally:
        if connection is not None:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
        if binding is not None:
            binding.close()


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
    if evidence._producer_marker is not _EVIDENCE_PRODUCER_MARKER or not hmac.compare_digest(
        evidence._producer_digest,
        _evidence_object_digest(evidence),
    ):
        raise ExactStateBootstrapError("bootstrap evidence is not producer-owned")
    database_path, database_identity, runtime_fingerprint, account_scope = _runtime_contract_values(
        runtime_contract
    )
    try:
        database_metadata = os.lstat(database_path)
    except OSError as exc:
        raise ExactStateBootstrapError("runtime database cannot be revalidated") from exc
    if (
        not stat.S_ISREG(database_metadata.st_mode)
        or stat.S_ISLNK(database_metadata.st_mode)
        or database_metadata.st_nlink != 1
        or (database_metadata.st_dev, database_metadata.st_ino)
        != (evidence.database_device, evidence.database_inode)
    ):
        raise ExactStateBootstrapError("runtime database identity or link count changed")
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
        _assert_fresh_against_wall_clock(observed, label)
    receipt_ids = [receipt.receipt_id for receipt in evidence.authentication_receipts]
    if len(receipt_ids) != len(set(receipt_ids)):
        raise ExactStateBootstrapError("bootstrap evidence receipt nonce was reused")
    for receipt in evidence.authentication_receipts:
        _assert_fresh_against_wall_clock(receipt.issued_at, "evidence authentication receipt")
        if receipt.expires_at < datetime.now(timezone.utc):
            raise ExactStateBootstrapError("evidence authentication receipt expired before apply")
    evidence_marks = {(mark.portfolio_id, mark.symbol): mark for mark in evidence.marks}
    candidate_keys = {(candidate.portfolio_id, position.symbol) for position in candidate.positions}
    if set(evidence_marks) != candidate_keys:
        raise ExactStateBootstrapError("verified marks do not cover the candidate positions")
    for position in candidate.positions:
        mark = evidence_marks[(candidate.portfolio_id, position.symbol)]
        _assert_fresh_against_wall_clock(mark.observed_at, f"protective mark for {position.symbol}")
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

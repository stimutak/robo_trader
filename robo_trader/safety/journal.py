"""Crash-safe, corruption-evident append-only safety journal.

The SHA-256 chain detects mutation, deletion inside the retained prefix, and
reordering. Like any hash chain whose external head is not independently
anchored, it is not proof against replacement by a valid older database suffix.

Construction and import are inert. ``initialize`` is explicit, and every write
uses a fresh SQLite connection plus ``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import sqlite3
import stat
import sys
import sysconfig
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import _sqlite3

from .models import (
    MODEL_VERSION,
    SAFETY_MAX_EVIDENCE_AGE_SECONDS,
    DecisionOutcome,
    ExposureEvidence,
    GateContext,
    JournalEvent,
    JournalEventType,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioAllocationEvidence,
    ReconciliationEvidence,
    ReconciliationStatus,
    ReplayReservation,
    ReplayState,
    Reservation,
    RiskEffect,
    SafetyDecision,
    SubmissionClaim,
    SubmissionDescriptor,
    SubmissionPermit,
    TerminalOrderStatus,
    TimeInForce,
    TransportState,
    ValidationError,
    _exact_decimal_add,
    _exact_decimal_subtract,
    _strict_account_scope,
    _strict_internal_id,
    _strict_text,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
    utc_to_text,
)
from .policy import evaluate_reduce_only

JOURNAL_SCHEMA_VERSION = 1
_ZERO_HASH = "0" * 64
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SQLITE_FCNTL_FILE_POINTER = 7
_SQLITE_FCNTL_VFS_POINTER = 27
_SQLITE_OK = 0
_SQLITE_C_API = None
_SQLITE_UNIX_VFS_NAMES = frozenset(
    {
        b"unix",
        b"unix-afp",
        b"unix-dotfile",
        b"unix-excl",
        b"unix-flock",
        b"unix-nfs",
        b"unix-none",
        b"unix-posix",
        b"unix-proxy",
    }
)


class _CPythonSQLiteConnectionHead(ctypes.Structure):
    """Stable leading fields of CPython's pysqlite Connection on supported versions."""

    _fields_ = (
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("db", ctypes.c_void_p),
    )


class _SQLiteVFSHead(ctypes.Structure):
    _fields_ = (
        ("i_version", ctypes.c_int),
        ("os_file_size", ctypes.c_int),
        ("max_pathname", ctypes.c_int),
        ("next_vfs", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
    )


class _UnixSQLiteFileHead(ctypes.Structure):
    """Leading fields of SQLite's default unix VFS ``unixFile``."""

    _fields_ = (
        ("methods", ctypes.c_void_p),
        ("vfs", ctypes.c_void_p),
        ("inode", ctypes.c_void_p),
        ("file_descriptor", ctypes.c_int),
    )


class JournalError(RuntimeError):
    pass


class JournalNotInitialized(JournalError):
    pass


class JournalIntegrityError(JournalError):
    pass


class IdempotencyConflict(JournalError):
    pass


class ReservationConflict(JournalError):
    pass


class StateTransitionError(JournalError):
    pass


def _sqlite_c_api():
    """Load the SQLite C API used by CPython's active ``_sqlite3`` extension."""

    global _SQLITE_C_API
    if sys.implementation.name != "cpython" or not ((3, 10) <= sys.version_info[:2] <= (3, 14)):
        raise JournalIntegrityError(
            "SQLite inode binding requires supported CPython 3.10 through 3.14"
        )
    if (
        sysconfig.get_config_var("Py_GIL_DISABLED")
        or sysconfig.get_config_var("Py_TRACE_REFS")
        or "t" in getattr(sys, "abiflags", "")
        or hasattr(sys, "getobjects")
    ):
        raise JournalIntegrityError(
            "SQLite inode binding rejects free-threaded or trace-reference CPython"
        )
    if _SQLITE_C_API is not None:
        return _SQLITE_C_API
    try:
        api = ctypes.PyDLL(_sqlite3.__file__)
        api.sqlite3_db_filename.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
        api.sqlite3_db_filename.restype = ctypes.c_char_p
        api.sqlite3_file_control.argtypes = (
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
        )
        api.sqlite3_file_control.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise JournalIntegrityError(
            "active SQLite library cannot prove database-file identity"
        ) from exc
    _SQLITE_C_API = api
    return api


def _sqlite_connection_file_identity(
    connection: sqlite3.Connection,
) -> Tuple[int, Tuple[int, int]]:
    """Return the descriptor and inode owned by SQLite's default unix VFS."""

    if type(connection) is not sqlite3.Connection:
        raise JournalIntegrityError("journal connection must be an exact sqlite3.Connection")
    api = _sqlite_c_api()
    pointer = _CPythonSQLiteConnectionHead.from_address(id(connection)).db
    if not pointer:
        raise JournalIntegrityError("journal connection has no active SQLite handle")
    filename = api.sqlite3_db_filename(pointer, b"main")
    if not filename:
        raise JournalIntegrityError("journal connection has no main database filename")
    vfs_pointer = ctypes.c_void_p()
    result = api.sqlite3_file_control(
        pointer,
        b"main",
        _SQLITE_FCNTL_VFS_POINTER,
        ctypes.byref(vfs_pointer),
    )
    if result != _SQLITE_OK or not vfs_pointer.value:
        raise JournalIntegrityError("SQLite cannot identify the journal VFS")
    vfs = _SQLiteVFSHead.from_address(vfs_pointer.value)
    if (
        not vfs.name
        or vfs.name not in _SQLITE_UNIX_VFS_NAMES
        or vfs.os_file_size < ctypes.sizeof(_UnixSQLiteFileHead)
    ):
        raise JournalIntegrityError("safety journal requires SQLite's default unix VFS")
    file_pointer = ctypes.c_void_p()
    result = api.sqlite3_file_control(
        pointer,
        b"main",
        _SQLITE_FCNTL_FILE_POINTER,
        ctypes.byref(file_pointer),
    )
    if result != _SQLITE_OK or not file_pointer.value:
        raise JournalIntegrityError("SQLite cannot expose its opened journal file")
    sqlite_file = _UnixSQLiteFileHead.from_address(file_pointer.value)
    if (
        not sqlite_file.methods
        or sqlite_file.vfs != vfs_pointer.value
        or sqlite_file.file_descriptor < 0
    ):
        raise JournalIntegrityError("SQLite returned an invalid unix journal file")
    try:
        file_stat = os.fstat(sqlite_file.file_descriptor)
    except OSError as exc:
        raise JournalIntegrityError("SQLite journal descriptor is not open") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise JournalIntegrityError("SQLite journal descriptor is not a regular file")
    return sqlite_file.file_descriptor, (file_stat.st_dev, file_stat.st_ino)


class _RejectedAuthorization:
    """A denial that must commit before its public exception is raised."""

    __slots__ = ("error",)

    def __init__(self, error: JournalError) -> None:
        self.error = error


FaultHook = Callable[[str, JournalEvent], None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class _PathBinding:
    file_descriptor: int
    sqlite_file_descriptor: int
    device: int
    inode: int


class SafetyJournal:
    """Dedicated safety database with one-shot atomic submission authorization."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        busy_timeout_ms: int = 5_000,
        clock: Optional[Clock] = None,
        fault_hook: Optional[FaultHook] = None,
    ) -> None:
        if not isinstance(database_path, (str, Path)):
            raise TypeError("database_path must be an explicit filesystem path")
        raw = str(database_path)
        if not raw or raw == ":memory:" or raw.startswith("file:"):
            raise ValueError("database_path must be a dedicated on-disk path")
        if type(busy_timeout_ms) is not int or not 1 <= busy_timeout_ms <= 60_000:
            raise ValueError("busy_timeout_ms must be in [1, 60000]")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if fault_hook is not None and not callable(fault_hook):
            raise TypeError("fault_hook must be callable")
        # Keep the final component lexical so every open can reject symlinks.
        # ``Path.resolve`` would follow an existing link before our lstat and
        # O_NOFOLLOW checks had a chance to validate it.
        expanded_path = Path(database_path).expanduser()
        self._path = expanded_path.parent.resolve(strict=False) / expanded_path.name
        self._busy_timeout_ms = busy_timeout_ms
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._fault_hook = fault_hook
        self._permit_lock = threading.Lock()
        self._issued_permits: Dict[int, Tuple[SubmissionPermit, str, SubmissionDescriptor]] = {}
        self._path_bindings: Dict[sqlite3.Connection, _PathBinding] = {}

    @property
    def database_path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        """Create the immutable schema; parent directories are never fabricated."""

        statements = (
            """
            CREATE TABLE IF NOT EXISTS safety_schema_version (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                version INTEGER NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS safety_journal_events (
                sequence INTEGER PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                execution_domain_scope TEXT NOT NULL,
                account_scope TEXT NOT NULL,
                portfolio_id TEXT NOT NULL,
                con_id INTEGER NOT NULL CHECK (con_id > 0),
                intent_fingerprint TEXT NOT NULL,
                claim_id TEXT,
                payload_json TEXT NOT NULL,
                previous_chain_hash TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                chain_hash TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS safety_journal_idempotency_idx
            ON safety_journal_events (idempotency_key, sequence)
            """,
            """
            CREATE INDEX IF NOT EXISTS safety_journal_account_contract_idx
            ON safety_journal_events
               (account_scope, con_id, sequence)
            """,
            """
            CREATE INDEX IF NOT EXISTS safety_journal_portfolio_contract_idx
            ON safety_journal_events
               (account_scope, portfolio_id, con_id, sequence)
            """,
            """
            CREATE TRIGGER IF NOT EXISTS safety_schema_version_no_update
            BEFORE UPDATE ON safety_schema_version
            BEGIN SELECT RAISE(ABORT, 'safety schema version is immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS safety_schema_version_no_delete
            BEFORE DELETE ON safety_schema_version
            BEGIN SELECT RAISE(ABORT, 'safety schema version is immutable'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS safety_journal_no_update
            BEFORE UPDATE ON safety_journal_events
            BEGIN SELECT RAISE(ABORT, 'safety journal is append-only'); END
            """,
            """
            CREATE TRIGGER IF NOT EXISTS safety_journal_no_delete
            BEFORE DELETE ON safety_journal_events
            BEGIN SELECT RAISE(ABORT, 'safety journal is append-only'); END
            """,
        )
        self._assert_existing_path_is_dedicated()
        connection = self._connect(create=True)
        try:
            self._assert_connection_path_identity(connection)
            connection.execute("BEGIN IMMEDIATE")
            self._assert_connection_path_identity(connection)
            existing_tables = {row[0] for row in connection.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """).fetchall()}
            unexpected = existing_tables - {
                "safety_schema_version",
                "safety_journal_events",
            }
            if unexpected:
                raise JournalIntegrityError(
                    "dedicated safety database contains unrelated tables: "
                    + ", ".join(sorted(unexpected))
                )
            for statement in statements:
                connection.execute(statement)
            row = connection.execute(
                "SELECT version FROM safety_schema_version WHERE singleton = 1"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO safety_schema_version(singleton, version) VALUES (1, ?)",
                    (JOURNAL_SCHEMA_VERSION,),
                )
            elif row != (JOURNAL_SCHEMA_VERSION,):
                raise JournalIntegrityError(f"unsupported schema version {row!r}")
            self._validate_schema(connection)
            self._assert_connection_path_identity(connection)
            connection.commit()
            self._assert_connection_path_identity(connection)
        except BaseException:
            connection.rollback()
            raise
        finally:
            self._close_connection(connection)

    def _assert_existing_path_is_dedicated(self) -> None:
        """Inspect an existing path read-only before WAL or chmod can mutate it."""

        try:
            self._path_identity()
        except FileNotFoundError:
            return
        connection = sqlite3.connect(
            f"{self._path.as_uri()}?mode=ro",
            uri=True,
            timeout=self._busy_timeout_ms / 1000,
            isolation_level=None,
        )
        try:
            existing_tables = {row[0] for row in connection.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """).fetchall()}
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError(
                "existing journal path is not a readable dedicated SQLite database"
            ) from exc
        finally:
            connection.close()
        unexpected = existing_tables - {
            "safety_schema_version",
            "safety_journal_events",
        }
        if unexpected:
            raise JournalIntegrityError(
                "dedicated safety database contains unrelated tables: "
                + ", ".join(sorted(unexpected))
            )

    def authorize_submission(
        self,
        idempotency_key: str,
        intent: OrderIntent,
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
        descriptor: SubmissionDescriptor,
    ) -> Tuple[Reservation, SubmissionClaim, Optional[SubmissionPermit]]:
        """Atomically reserve scopes and commit one ``SUBMISSION_STARTED``.

        A newly committed call receives an ephemeral permit. Exact replays
        receive durable history and ``None``—never renewed submission authority.
        """

        key = _strict_text(idempotency_key, "idempotency_key", max_length=128)
        if type(intent) is not OrderIntent:
            raise TypeError("intent must be OrderIntent")
        if type(exposure) is not ExposureEvidence:
            raise TypeError("exposure must be ExposureEvidence")
        if type(allocation) is not PortfolioAllocationEvidence:
            raise TypeError("allocation must be PortfolioAllocationEvidence")
        if type(gates) is not GateContext:
            raise TypeError("gates must be GateContext")
        if type(descriptor) is not SubmissionDescriptor:
            raise TypeError("descriptor must be SubmissionDescriptor")
        intent = replace(intent)
        exposure = replace(exposure)
        allocation = replace(allocation)
        gates = replace(gates)
        descriptor = replace(descriptor)
        intent_fingerprint = intent.fingerprint()
        descriptor_fingerprint = descriptor.fingerprint()
        context = self._authorization_context(exposure, allocation, gates)
        context["descriptor"] = json.loads(descriptor.canonical_payload())
        context_fingerprint = sha256_text(canonical_json(context))
        descriptor_mismatch = (
            descriptor.execution_domain_scope != intent.execution_domain_scope
            or descriptor.account_scope != intent.account_scope
            or descriptor.con_id != intent.con_id
            or descriptor.side is not intent.side
            or descriptor.quantity != intent.quantity
        )

        def operation(connection: sqlite3.Connection):
            at = self._event_time()
            state = self._replay_connection(connection)
            binding = self._decision_binding(state.events, key)
            existing = self._find_reservation(state.reservations, key)
            if binding is not None:
                if (
                    binding["intent_fingerprint"] != intent_fingerprint
                    or binding["authorization_context_fingerprint"] != context_fingerprint
                ):
                    raise IdempotencyConflict(
                        "idempotency key is bound to different authorization evidence"
                    )
                if existing is None:
                    raise StateTransitionError(
                        "idempotency key already records a non-authorizing decision"
                    )
                if existing.intent_fingerprint != intent_fingerprint:
                    raise IdempotencyConflict("idempotency key is bound to a different intent")
                if existing.submission_descriptor_fingerprint != descriptor_fingerprint:
                    raise IdempotencyConflict(
                        "idempotency key is bound to different submission terms"
                    )
                return (
                    self._reservation_model(existing, newly_acquired=False),
                    self._claim_model(existing, granted=False),
                    False,
                )

            def reject(
                reason_code: str,
                error: JournalError,
                *,
                risk_effect: RiskEffect = RiskEffect.UNKNOWN,
            ) -> _RejectedAuthorization:
                decision = SafetyDecision(
                    outcome=DecisionOutcome.DENY,
                    risk_effect=risk_effect,
                    reason_codes=(reason_code,),
                    current_quantity=None,
                    computed_target_quantity=None,
                    intent_fingerprint=intent_fingerprint,
                )
                decision_payload = json.loads(decision.canonical_payload())
                binding = self._decision_binding(state.events, key)
                if binding is None:
                    self._append_decision(
                        connection,
                        key,
                        intent,
                        decision,
                        context,
                        context_fingerprint,
                        at,
                    )
                elif (
                    binding["intent_fingerprint"] != intent_fingerprint
                    or binding["decision"] != decision_payload
                    or binding["authorization_context_fingerprint"] != context_fingerprint
                ):
                    raise IdempotencyConflict("idempotency key is bound to another evaluation")
                return _RejectedAuthorization(error)

            if descriptor_mismatch:
                return reject(
                    "DESCRIPTOR_MISMATCH",
                    StateTransitionError("descriptor does not exactly match authorized intent"),
                )
            locked_decision = self._evaluate_at_authorization(
                intent, exposure, allocation, gates, at
            )
            if (
                locked_decision.outcome is DecisionOutcome.ALLOW
                and self._snapshots_not_newer_than_terminal(
                    state, intent, exposure, allocation, gates
                )
            ):
                locked_decision = SafetyDecision(
                    outcome=DecisionOutcome.DENY,
                    risk_effect=locked_decision.risk_effect,
                    reason_codes=("SNAPSHOTS_NOT_NEWER_THAN_TERMINAL",),
                    current_quantity=locked_decision.current_quantity,
                    computed_target_quantity=locked_decision.computed_target_quantity,
                    intent_fingerprint=locked_decision.intent_fingerprint,
                )
            if locked_decision.outcome is not DecisionOutcome.ALLOW:
                binding = self._decision_binding(state.events, key)
                decision_payload = json.loads(locked_decision.canonical_payload())
                if binding is None:
                    self._append_decision(
                        connection,
                        key,
                        intent,
                        locked_decision,
                        context,
                        context_fingerprint,
                        at,
                    )
                elif (
                    binding["intent_fingerprint"] != intent_fingerprint
                    or binding["decision"] != decision_payload
                    or binding["authorization_context_fingerprint"] != context_fingerprint
                ):
                    raise IdempotencyConflict("idempotency key is bound to another evaluation")
                return None, locked_decision, False
            for active in state.active_reservations:
                if active.account_scope == intent.account_scope and active.con_id == intent.con_id:
                    scope = (
                        "account+portfolio+conId"
                        if active.portfolio_id == intent.portfolio_id
                        else "account+conId"
                    )
                    return reject(
                        "RESERVATION_CONFLICT",
                        ReservationConflict(f"active {scope} reservation exists"),
                        risk_effect=locked_decision.risk_effect,
                    )
            for event in state.events:
                if event.event_type is not JournalEventType.SAFETY_DECISION:
                    continue
                prior = json.loads(event.payload_json)
                if prior["decision"]["outcome"] == DecisionOutcome.ALLOW.value and (
                    prior["authorization_context_fingerprint"] == context_fingerprint
                    or self._context_snapshot_keys(prior["authorization_context"])
                    & self._context_snapshot_keys(context)
                ):
                    return reject(
                        "SNAPSHOT_ALREADY_CONSUMED",
                        StateTransitionError(
                            "authoritative evidence snapshot was already consumed"
                        ),
                        risk_effect=locked_decision.risk_effect,
                    )

            reservation_id = f"res-{uuid.uuid4().hex}"
            self._append_decision(
                connection,
                key,
                intent,
                locked_decision,
                context,
                context_fingerprint,
                at,
            )
            reservation_event = self._append(
                connection,
                JournalEventType.RESERVATION_ACQUIRED,
                at,
                key,
                intent.execution_domain_scope,
                intent.account_scope,
                intent.portfolio_id,
                intent.con_id,
                intent_fingerprint,
                None,
                {
                    "intent": dict(intent.authorization_payload()),
                    "intent_fingerprint": intent_fingerprint,
                    "reservation_id": reservation_id,
                },
            )
            claim_id = f"claim-{uuid.uuid4().hex}"
            claim_event = self._append(
                connection,
                JournalEventType.SUBMISSION_STARTED,
                at,
                key,
                intent.execution_domain_scope,
                intent.account_scope,
                intent.portfolio_id,
                intent.con_id,
                intent_fingerprint,
                claim_id,
                {
                    "claim_id": claim_id,
                    "descriptor": json.loads(descriptor.canonical_payload()),
                    "submission_descriptor_fingerprint": descriptor_fingerprint,
                    "reservation_id": reservation_id,
                    "reservation_sequence": reservation_event.sequence,
                },
            )
            reservation = Reservation(
                reservation_id=reservation_id,
                idempotency_key=key,
                intent_fingerprint=intent_fingerprint,
                execution_domain_scope=intent.execution_domain_scope,
                account_scope=intent.account_scope,
                portfolio_id=intent.portfolio_id,
                con_id=intent.con_id,
                sequence=reservation_event.sequence,
                acquired_at=reservation_event.occurred_at,
                newly_acquired=True,
            )
            claim = SubmissionClaim(
                claim_id=claim_id,
                reservation_id=reservation_id,
                reservation_sequence=reservation_event.sequence,
                idempotency_key=key,
                submission_descriptor_fingerprint=descriptor_fingerprint,
                execution_domain_scope=intent.execution_domain_scope,
                account_scope=intent.account_scope,
                portfolio_id=intent.portfolio_id,
                con_id=intent.con_id,
                order_ref=descriptor.order_ref,
                sequence=claim_event.sequence,
                claimed_at=claim_event.occurred_at,
                granted=True,
            )
            return reservation, claim, True

        with self._permit_lock:
            result, committed_identity = self._write_transaction(
                operation,
                include_path_identity=True,
            )
            if isinstance(result, _RejectedAuthorization):
                raise result.error
            reservation, claim, issue_permit = result
            if reservation is None:
                raise StateTransitionError(
                    "policy denied submission: " + ", ".join(claim.reason_codes)
                )
            self._assert_path_identity(committed_identity)
            permit = SubmissionPermit._issue(claim.claim_id) if issue_permit else None
            if permit is not None:
                self._issued_permits[id(permit)] = (permit, claim.claim_id, descriptor)
            return reservation, claim, permit

    acquire_and_claim = authorize_submission

    def consume_submission_permit(self, permit: SubmissionPermit) -> SubmissionDescriptor:
        """Durably dispatch a live permit before returning its exact terms.

        The ``OUTCOME_UNKNOWN`` append is serialized with release and unknown
        transitions by ``BEGIN IMMEDIATE`` across every journal instance.
        """

        if type(permit) is not SubmissionPermit:
            raise TypeError("permit must be SubmissionPermit")
        with self._permit_lock:
            registered = self._issued_permits.get(id(permit))
            if (
                registered is None
                or registered[0] is not permit
                or registered[1] != permit.claim_id
            ):
                raise StateTransitionError("permit was not issued by this live journal instance")

            def operation(connection: sqlite3.Connection) -> None:
                at = self._event_time()
                state = self._replay_connection(connection)
                reservation = next(
                    (item for item in state.reservations if item.claim_id == permit.claim_id),
                    None,
                )
                if reservation is None:
                    raise StateTransitionError("permit claim has no reservation")
                if reservation.released:
                    raise StateTransitionError("permit reservation is already terminal")
                if reservation.outcome_unknown:
                    raise StateTransitionError(
                        "permit claim is already dispatched or outcome-unknown"
                    )
                self._append(
                    connection,
                    JournalEventType.OUTCOME_UNKNOWN,
                    at,
                    reservation.idempotency_key,
                    reservation.execution_domain_scope,
                    reservation.account_scope,
                    reservation.portfolio_id,
                    reservation.con_id,
                    reservation.intent_fingerprint,
                    reservation.claim_id,
                    {
                        "claim_id": reservation.claim_id,
                        "dispatch_state": "PERMIT_CONSUMED",
                        "reservation_id": reservation.reservation_id,
                    },
                )

            _, committed_identity = self._write_transaction(
                operation,
                include_path_identity=True,
            )
            self._assert_path_identity(committed_identity)
            self._issued_permits.pop(id(permit), None)
            permit._mark_consumed()
            return registered[2]

    def _invalidate_claim_permits_locked(self, claim_id: Optional[str]) -> None:
        if claim_id is None:
            return
        stale = [
            permit_id
            for permit_id, (_, registered_claim_id, _) in self._issued_permits.items()
            if registered_claim_id == claim_id
        ]
        for permit_id in stale:
            self._issued_permits.pop(permit_id, None)

    def record_rejection(
        self,
        idempotency_key: str,
        intent: OrderIntent,
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
    ) -> JournalEvent:
        """Durably record a denied decision without creating authority."""

        key = _strict_text(idempotency_key, "idempotency_key", max_length=128)
        if type(intent) is not OrderIntent:
            raise TypeError("intent must be OrderIntent")
        if type(exposure) is not ExposureEvidence:
            raise TypeError("exposure must be ExposureEvidence")
        if type(allocation) is not PortfolioAllocationEvidence:
            raise TypeError("allocation must be PortfolioAllocationEvidence")
        if type(gates) is not GateContext:
            raise TypeError("gates must be GateContext")
        intent = replace(intent)
        exposure = replace(exposure)
        allocation = replace(allocation)
        gates = replace(gates)
        fingerprint = intent.fingerprint()
        context = self._authorization_context(exposure, allocation, gates)
        context_fingerprint = sha256_text(canonical_json(context))
        return self._record_decision(
            key, intent, exposure, allocation, gates, context, context_fingerprint
        )

    def _record_decision(
        self,
        key: str,
        intent: OrderIntent,
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
        context: dict,
        context_fingerprint: str,
    ) -> JournalEvent:
        fingerprint = intent.fingerprint()

        def operation(connection: sqlite3.Connection):
            at = self._event_time()
            decision = self._evaluate_at_authorization(intent, exposure, allocation, gates, at)
            if decision.outcome is DecisionOutcome.ALLOW:
                raise StateTransitionError("a matching DENY decision is required")
            decision_payload = json.loads(decision.canonical_payload())
            state = self._replay_connection(connection)
            binding = self._decision_binding(state.events, key)
            if binding is not None:
                if binding["intent_fingerprint"] != fingerprint:
                    raise IdempotencyConflict("idempotency key is bound to a different intent")
                if binding["decision"] != decision_payload:
                    raise IdempotencyConflict("idempotency key is bound to a different decision")
                if binding["authorization_context_fingerprint"] != context_fingerprint:
                    raise IdempotencyConflict("idempotency key is bound to different evidence")
                return next(
                    event
                    for event in state.events
                    if event.idempotency_key == key
                    and event.event_type is JournalEventType.SAFETY_DECISION
                )
            return self._append_decision(
                connection,
                key,
                intent,
                decision,
                context,
                context_fingerprint,
                at,
            )

        return self._write_transaction(operation)

    def mark_outcome_unknown(
        self,
        idempotency_key: str,
        intent_fingerprint: str,
    ) -> ReplayReservation:
        key = _strict_text(idempotency_key, "idempotency_key", max_length=128)
        self._validate_hash(intent_fingerprint, "intent_fingerprint")

        def operation(connection: sqlite3.Connection):
            at = self._event_time()
            state = self._replay_connection(connection)
            reservation = self._require_reservation(state, key, intent_fingerprint)
            if reservation.released:
                raise StateTransitionError("reservation is already terminal")
            if reservation.outcome_unknown:
                return reservation
            event = self._append(
                connection,
                JournalEventType.OUTCOME_UNKNOWN,
                at,
                key,
                reservation.execution_domain_scope,
                reservation.account_scope,
                reservation.portfolio_id,
                reservation.con_id,
                intent_fingerprint,
                reservation.claim_id,
                {
                    "claim_id": reservation.claim_id,
                    "reservation_id": reservation.reservation_id,
                },
            )
            return replace(
                reservation,
                outcome_unknown=True,
                quarantined=True,
            )

        with self._permit_lock:
            result = self._write_transaction(operation)
            self._invalidate_claim_permits_locked(result.claim_id)
            return result

    def release_after_reconciliation(
        self,
        idempotency_key: str,
        intent_fingerprint: str,
        evidence: ReconciliationEvidence,
    ) -> ReplayReservation:
        key = _strict_text(idempotency_key, "idempotency_key", max_length=128)
        self._validate_hash(intent_fingerprint, "intent_fingerprint")
        if type(evidence) is not ReconciliationEvidence:
            raise TypeError("evidence must be ReconciliationEvidence")
        evidence = replace(evidence)

        def operation(connection: sqlite3.Connection):
            at = self._event_time()
            state = self._replay_connection(connection)
            reservation = self._require_reservation(state, key, intent_fingerprint)
            if reservation.released:
                return reservation
            failures = []
            exact_pairs = (
                (
                    evidence.execution_domain_scope,
                    reservation.execution_domain_scope,
                    "execution domain",
                ),
                (evidence.account_scope, reservation.account_scope, "account"),
                (evidence.portfolio_id, reservation.portfolio_id, "portfolio"),
                (evidence.con_id, reservation.con_id, "conId"),
                (evidence.symbol, reservation.symbol, "symbol metadata"),
                (evidence.reservation_id, reservation.reservation_id, "reservation"),
                (evidence.claim_id, reservation.claim_id, "claim"),
                (evidence.claim_sequence, reservation.claim_sequence, "claim sequence"),
                (
                    evidence.submission_descriptor_fingerprint,
                    reservation.submission_descriptor_fingerprint,
                    "submission descriptor",
                ),
                (evidence.order_ref, reservation.order_ref, "order reference"),
            )
            failures.extend(
                f"{label} mismatch" for actual, expected, label in exact_pairs if actual != expected
            )
            if reservation.claim_time is None:
                failures.append("claim time missing")
            else:
                if evidence.observed_at <= reservation.claim_time:
                    failures.append("terminal order evidence must be strictly after claim")
                if evidence.position_observed_at <= reservation.claim_time:
                    failures.append("position evidence must be strictly after claim")
                if evidence.observed_at > at:
                    failures.append("terminal order evidence is future-dated")
                if evidence.position_observed_at > at:
                    failures.append("position evidence is future-dated")
                if (at - evidence.observed_at).total_seconds() > evidence.max_evidence_age_seconds:
                    failures.append("terminal order evidence is stale")
                if (
                    at - evidence.position_observed_at
                ).total_seconds() > evidence.max_evidence_age_seconds:
                    failures.append("position evidence is stale")
            if evidence.status is not ReconciliationStatus.PASSED:
                failures.append("reconciliation did not pass")
            if evidence.transport_state is not TransportState.CONNECTED:
                failures.append("transport is not certainly connected")
            if not evidence.open_orders_complete:
                failures.append("open-order evidence is incomplete")
            if not evidence.open_orders_all_clients:
                failures.append("open-order evidence excludes clients")
            if not evidence.open_orders_snapshot_stable:
                failures.append("open-order snapshot is unstable")
            if evidence.active_order_count != 0:
                failures.append("an active broker order still exists")
            if evidence.has_offsetting_allocations:
                failures.append("offsetting allocations exist")
            if evidence.aggregate_allocated_quantity != evidence.account_position_quantity:
                failures.append("aggregate allocation does not reconcile")
            signed_full = (
                reservation.quantity
                if reservation.side is OrderSide.BUY_TO_COVER
                else reservation.quantity.copy_negate()
            )
            signed_fill = (
                evidence.filled_quantity
                if reservation.side is OrderSide.BUY_TO_COVER
                else (
                    evidence.filled_quantity
                    if evidence.filled_quantity.is_zero()
                    else evidence.filled_quantity.copy_negate()
                )
            )
            try:
                initial_account = _exact_decimal_subtract(
                    reservation.target_quantity,
                    signed_full,
                    "initial account quantity",
                )
                initial_portfolio = _exact_decimal_subtract(
                    reservation.portfolio_target_quantity,
                    signed_full,
                    "initial portfolio quantity",
                )
                expected_account = _exact_decimal_add(
                    initial_account,
                    signed_fill,
                    "expected account quantity",
                )
                expected_portfolio = _exact_decimal_add(
                    initial_portfolio,
                    signed_fill,
                    "expected portfolio quantity",
                )
                terminal_quantity = _exact_decimal_add(
                    evidence.filled_quantity,
                    evidence.remaining_quantity,
                    "terminal claimed quantity",
                )
            except ValidationError as exc:
                failures.append(f"reconciliation arithmetic invalid: {exc}")
                expected_account = None
                expected_portfolio = None
                terminal_quantity = None
            if evidence.account_position_quantity != expected_account:
                failures.append("account position does not equal authorized target")
            if evidence.portfolio_position_quantity != expected_portfolio:
                failures.append("portfolio position does not equal authorized target")
            if terminal_quantity != reservation.quantity:
                failures.append("terminal quantities do not equal claimed quantity")
            if (
                evidence.terminal_order_status is TerminalOrderStatus.FILLED
                and evidence.remaining_quantity != 0
            ):
                failures.append("filled order has remaining quantity")
            if evidence.terminal_order_status in {
                TerminalOrderStatus.REJECTED,
                TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
            } and (
                evidence.filled_quantity != 0 or evidence.remaining_quantity != reservation.quantity
            ):
                failures.append("unsubmitted/rejected order quantities are inconsistent")
            dispatch_time = self._permit_consumed_event_time(state, reservation)
            if evidence.terminal_order_status is TerminalOrderStatus.NO_SUBMISSION_CONFIRMED:
                if dispatch_time is not None:
                    failures.append("no-submission evidence cannot release dispatched authority")
            elif dispatch_time is None:
                failures.append("terminal broker outcome lacks durable permit dispatch")
            elif (
                evidence.observed_at <= dispatch_time
                or evidence.position_observed_at <= dispatch_time
            ):
                failures.append("terminal evidence must strictly postdate permit dispatch")
            if failures:
                raise StateTransitionError("; ".join(failures))

            event = self._append(
                connection,
                JournalEventType.TERMINAL_RECONCILED,
                at,
                key,
                reservation.execution_domain_scope,
                reservation.account_scope,
                reservation.portfolio_id,
                reservation.con_id,
                intent_fingerprint,
                reservation.claim_id,
                {
                    "evidence": json.loads(evidence.canonical_payload()),
                    "reservation_id": reservation.reservation_id,
                },
            )
            return replace(
                reservation,
                quarantined=False,
                released=True,
                terminal_sequence=event.sequence,
            )

        with self._permit_lock:
            result = self._write_transaction(operation)
            self._invalidate_claim_permits_locked(result.claim_id)
            return result

    release_reservation = release_after_reconciliation

    def replay(self) -> ReplayState:
        """Replay through a read-only connection; never returns a permit."""

        try:
            expected_identity = self._path_identity()
        except FileNotFoundError:
            raise JournalNotInitialized("safety journal has not been initialized")
        connection = self._open_bound_connection(
            f"{self._path.as_uri()}?mode=ro",
            uri=True,
            expected_identity=expected_identity,
            writable=False,
        )
        try:
            self._assert_connection_path_identity(connection)
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA query_only = ON")
            connection.set_authorizer(self._read_only_authorizer)
            result = connection.execute("PRAGMA integrity_check").fetchone()
            if result != ("ok",):
                raise JournalIntegrityError(f"SQLite integrity check failed: {result!r}")
            state = self._replay_connection(connection)
            self._assert_connection_path_identity(connection)
            return state
        finally:
            self._close_connection(connection)

    verify_integrity = replay

    def _connect(self, *, create: bool = False) -> sqlite3.Connection:
        try:
            pre_connect_identity = self._path_identity()
        except FileNotFoundError:
            if not create:
                raise JournalNotInitialized("safety journal has not been initialized") from None
            flags = (
                os.O_CREAT
                | os.O_EXCL
                | os.O_RDWR
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            try:
                file_descriptor = os.open(self._path, flags, 0o600)
            except FileExistsError as exc:
                raise JournalIntegrityError(
                    "safety journal path appeared during initialization"
                ) from exc
            except OSError as exc:
                raise JournalIntegrityError(
                    "cannot atomically create the safety journal path"
                ) from exc
            try:
                file_stat = os.fstat(file_descriptor)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise JournalIntegrityError("new safety journal path is not a regular file")
                pre_connect_identity = (file_stat.st_dev, file_stat.st_ino)
            finally:
                os.close(file_descriptor)
        if create:
            database = str(self._path)
            uri = False
        else:
            database = f"{self._path.as_uri()}?mode=rw"
            uri = True
        try:
            connection = self._open_bound_connection(
                database,
                uri=uri,
                expected_identity=pre_connect_identity,
                writable=True,
            )
        except (OSError, sqlite3.DatabaseError) as exc:
            raise JournalIntegrityError(
                "safety journal path is not a writable SQLite database"
            ) from exc
        try:
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            connection.execute("PRAGMA foreign_keys = ON")
            self._validate_connection_before_mutation(
                connection,
                allow_empty=create,
            )
            self._assert_connection_path_identity(connection)
            connection.execute("PRAGMA journal_mode = WAL")
            self._assert_connection_path_identity(connection)
            connection.execute("PRAGMA synchronous = FULL")
            self._assert_connection_path_identity(connection)
            self._harden_journal_permissions(connection)
            self._assert_connection_path_identity(connection)
            return connection
        except sqlite3.DatabaseError as exc:
            self._close_connection(connection)
            raise JournalIntegrityError(
                "safety journal path is not the expected SQLite journal"
            ) from exc
        except BaseException:
            self._close_connection(connection)
            raise

    def _path_identity(self) -> Tuple[int, int]:
        path_stat = os.lstat(self._path)
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
            raise JournalIntegrityError("safety journal path must be a non-symlink regular file")
        return path_stat.st_dev, path_stat.st_ino

    def _bind_connection_to_path(
        self,
        connection: sqlite3.Connection,
        expected_identity: Tuple[int, int],
        guardian_file_descriptor: int,
    ) -> None:
        guardian_stat = os.fstat(guardian_file_descriptor)
        guardian_identity = (guardian_stat.st_dev, guardian_stat.st_ino)
        sqlite_file_descriptor, sqlite_identity = _sqlite_connection_file_identity(connection)
        path_identity = self._path_identity()
        if (
            not stat.S_ISREG(guardian_stat.st_mode)
            or guardian_identity != expected_identity
            or sqlite_identity != expected_identity
            or path_identity != expected_identity
        ):
            raise JournalIntegrityError("safety journal path identity changed while opening")
        self._path_bindings[connection] = _PathBinding(
            file_descriptor=guardian_file_descriptor,
            sqlite_file_descriptor=sqlite_file_descriptor,
            device=expected_identity[0],
            inode=expected_identity[1],
        )

    def _open_bound_connection(
        self,
        database: str,
        *,
        uri: bool,
        expected_identity: Tuple[int, int],
        writable: bool,
    ) -> sqlite3.Connection:
        flags = (
            (os.O_RDWR if writable else os.O_RDONLY)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            guardian_file_descriptor = os.open(self._path, flags)
        except OSError as exc:
            raise JournalIntegrityError(
                "cannot bind the opened journal to its filesystem path"
            ) from exc
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(
                database,
                uri=uri,
                timeout=self._busy_timeout_ms / 1000,
                isolation_level=None,
            )
            self._bind_connection_to_path(
                connection,
                expected_identity,
                guardian_file_descriptor,
            )
        except BaseException:
            binding = self._path_bindings.pop(connection, None) if connection is not None else None
            try:
                if connection is not None:
                    connection.close()
            finally:
                os.close(
                    binding.file_descriptor if binding is not None else guardian_file_descriptor
                )
            raise
        return connection

    def _assert_connection_path_identity(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        binding = self._path_bindings.get(connection)
        if binding is None:
            raise JournalIntegrityError("journal connection lacks a path binding")
        try:
            guardian_stat = os.fstat(binding.file_descriptor)
            sqlite_file_descriptor, sqlite_identity = _sqlite_connection_file_identity(connection)
            path_identity = self._path_identity()
        except (OSError, JournalIntegrityError) as exc:
            raise JournalIntegrityError(
                "safety journal path identity is no longer authoritative"
            ) from exc
        identity = (binding.device, binding.inode)
        if (
            not stat.S_ISREG(guardian_stat.st_mode)
            or (guardian_stat.st_dev, guardian_stat.st_ino) != identity
            or sqlite_file_descriptor != binding.sqlite_file_descriptor
            or sqlite_identity != identity
            or path_identity != identity
        ):
            raise JournalIntegrityError(
                "safety journal path identity changed during the transaction"
            )

    def _assert_path_identity(self, expected: Tuple[int, int]) -> None:
        try:
            actual = self._path_identity()
        except (OSError, JournalIntegrityError) as exc:
            raise JournalIntegrityError(
                "safety journal path identity is no longer authoritative"
            ) from exc
        if actual != expected:
            raise JournalIntegrityError(
                "safety journal path identity changed before authority return"
            )

    def _close_connection(self, connection: sqlite3.Connection) -> None:
        binding = self._path_bindings.pop(connection, None)
        try:
            connection.close()
        finally:
            if binding is not None:
                os.close(binding.file_descriptor)

    def _validate_connection_before_mutation(
        self,
        connection: sqlite3.Connection,
        *,
        allow_empty: bool,
    ) -> None:
        """Validate the opened file itself before WAL or chmod can mutate it."""

        try:
            existing_tables = {row[0] for row in connection.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                    """).fetchall()}
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError(
                "opened journal is not a readable dedicated SQLite database"
            ) from exc
        expected_tables = {
            "safety_schema_version",
            "safety_journal_events",
        }
        if allow_empty and not existing_tables:
            return
        unexpected = existing_tables - expected_tables
        if unexpected:
            raise JournalIntegrityError(
                "dedicated safety database contains unrelated tables: "
                + ", ".join(sorted(unexpected))
            )
        if existing_tables != expected_tables:
            raise JournalIntegrityError(
                "dedicated safety database has an incomplete journal schema"
            )
        try:
            self._validate_schema(connection)
            schema = connection.execute(
                "SELECT singleton, version FROM safety_schema_version"
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError("opened journal has an unreadable safety schema") from exc
        if schema != [(1, JOURNAL_SCHEMA_VERSION)]:
            raise JournalIntegrityError("invalid safety schema version")

    def _harden_journal_permissions(self, connection: sqlite3.Connection) -> None:
        binding = self._path_bindings[connection]
        os.fchmod(binding.file_descriptor, 0o600)
        for path in (Path(f"{self._path}-wal"), Path(f"{self._path}-shm")):
            self._assert_connection_path_identity(connection)
            flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                file_descriptor = os.open(path, flags)
            except FileNotFoundError:
                continue
            try:
                file_stat = os.fstat(file_descriptor)
                path_stat = os.lstat(path)
                if (
                    stat.S_ISLNK(path_stat.st_mode)
                    or not stat.S_ISREG(file_stat.st_mode)
                    or (file_stat.st_dev, file_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino)
                ):
                    raise JournalIntegrityError(
                        "journal companion path identity is not authoritative"
                    )
                os.fchmod(file_descriptor, 0o600)
            finally:
                os.close(file_descriptor)

    def _write_transaction(self, operation, *, include_path_identity: bool = False):
        connection = self._connect()
        try:
            binding = self._path_bindings[connection]
            committed_identity = (binding.device, binding.inode)
            self._assert_connection_path_identity(connection)
            connection.execute("BEGIN IMMEDIATE")
            self._assert_connection_path_identity(connection)
            before = self._last_event(connection)
            result = operation(connection)
            after = self._last_event(connection)
            if (
                after is not None
                and (before is None or after.sequence != before.sequence)
                and self._fault_hook is not None
            ):
                self._fault_hook("BEFORE_COMMIT", after)
            self._assert_connection_path_identity(connection)
            connection.commit()
            self._assert_connection_path_identity(connection)
            if include_path_identity:
                return result, committed_identity
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            self._close_connection(connection)

    def _append(
        self,
        connection: sqlite3.Connection,
        event_type: JournalEventType,
        occurred_at: datetime,
        idempotency_key: str,
        execution_domain_scope: str,
        account_scope: str,
        portfolio_id: str,
        con_id: int,
        intent_fingerprint: str,
        claim_id: Optional[str],
        payload: dict,
    ) -> JournalEvent:
        previous = self._last_event(connection)
        if previous is not None and occurred_at < previous.occurred_at:
            raise StateTransitionError("journal event time cannot move backward")
        sequence = 1 if previous is None else previous.sequence + 1
        previous_hash = _ZERO_HASH if previous is None else previous.chain_hash
        payload_json = canonical_json(payload)
        payload_hash = sha256_text(payload_json)
        occurred_text = utc_to_text(occurred_at)
        chain_hash = self._chain_hash(
            sequence,
            event_type,
            occurred_text,
            idempotency_key,
            execution_domain_scope,
            account_scope,
            portfolio_id,
            con_id,
            intent_fingerprint,
            claim_id,
            payload_hash,
            previous_hash,
            MODEL_VERSION,
        )
        event_id = f"evt-{uuid.uuid4().hex}"
        connection.execute(
            """
            INSERT INTO safety_journal_events (
                sequence, event_id, event_type, occurred_at, idempotency_key,
                execution_domain_scope, account_scope, portfolio_id, con_id,
                intent_fingerprint, claim_id, payload_json, previous_chain_hash,
                payload_hash, chain_hash, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                event_id,
                event_type.value,
                occurred_text,
                idempotency_key,
                execution_domain_scope,
                account_scope,
                portfolio_id,
                con_id,
                intent_fingerprint,
                claim_id,
                payload_json,
                previous_hash,
                payload_hash,
                chain_hash,
                MODEL_VERSION,
            ),
        )
        event = JournalEvent(
            sequence=sequence,
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            idempotency_key=idempotency_key,
            execution_domain_scope=execution_domain_scope,
            account_scope=account_scope,
            portfolio_id=portfolio_id,
            con_id=con_id,
            intent_fingerprint=intent_fingerprint,
            claim_id=claim_id,
            payload_json=payload_json,
            previous_chain_hash=previous_hash,
            payload_hash=payload_hash,
            chain_hash=chain_hash,
        )
        if self._fault_hook is not None:
            self._fault_hook("AFTER_APPEND", event)
        return event

    def _append_decision(
        self,
        connection: sqlite3.Connection,
        key: str,
        intent: OrderIntent,
        decision: SafetyDecision,
        context: dict,
        context_fingerprint: str,
        at: datetime,
    ) -> JournalEvent:
        return self._append(
            connection,
            JournalEventType.SAFETY_DECISION,
            at,
            key,
            intent.execution_domain_scope,
            intent.account_scope,
            intent.portfolio_id,
            intent.con_id,
            intent.fingerprint(),
            None,
            {
                "authorization_context": context,
                "authorization_context_fingerprint": context_fingerprint,
                "decision": json.loads(decision.canonical_payload()),
                "intent_fingerprint": intent.fingerprint(),
            },
        )

    def _replay_connection(self, connection: sqlite3.Connection) -> ReplayState:
        self._validate_schema(connection)
        try:
            schema = connection.execute(
                "SELECT singleton, version FROM safety_schema_version"
            ).fetchall()
            rows = connection.execute("""
                SELECT sequence, event_id, event_type, occurred_at,
                       idempotency_key, execution_domain_scope, account_scope,
                       portfolio_id, con_id, intent_fingerprint, claim_id,
                       payload_json, previous_chain_hash, payload_hash,
                       chain_hash, schema_version
                FROM safety_journal_events ORDER BY sequence
                """).fetchall()
        except sqlite3.DatabaseError as exc:
            raise JournalIntegrityError("missing or unreadable safety schema") from exc
        if schema != [(1, JOURNAL_SCHEMA_VERSION)]:
            raise JournalIntegrityError("invalid safety schema version")

        events = []
        reservations: Dict[str, ReplayReservation] = {}
        decisions: Dict[str, dict] = {}
        dispatch_times: Dict[str, datetime] = {}
        previous_hash = _ZERO_HASH
        for expected_sequence, row in enumerate(rows, start=1):
            event = self._event_from_row(row)
            if event.sequence != expected_sequence:
                raise JournalIntegrityError("journal sequence is not contiguous")
            if event.previous_chain_hash != previous_hash:
                raise JournalIntegrityError("chain predecessor mismatch")
            if sha256_text(event.payload_json) != event.payload_hash:
                raise JournalIntegrityError("payload hash mismatch")
            expected_hash = self._chain_hash(
                event.sequence,
                event.event_type,
                utc_to_text(event.occurred_at),
                event.idempotency_key,
                event.execution_domain_scope,
                event.account_scope,
                event.portfolio_id,
                event.con_id,
                event.intent_fingerprint,
                event.claim_id,
                event.payload_hash,
                event.previous_chain_hash,
                event.schema_version,
            )
            if event.chain_hash != expected_hash:
                raise JournalIntegrityError("chain hash mismatch")
            try:
                payload = json.loads(event.payload_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise JournalIntegrityError("payload is not valid JSON") from exc
            if not isinstance(payload, dict) or canonical_json(payload) != event.payload_json:
                raise JournalIntegrityError("payload is not canonical JSON")
            self._apply_event(
                reservations,
                decisions,
                dispatch_times,
                event,
                payload,
            )
            events.append(event)
            previous_hash = event.chain_hash

        ordered = tuple(sorted(reservations.values(), key=lambda item: item.acquired_sequence))
        # A committed reservation must always include its claim in the same
        # transaction. Anything else is corruption, not reclaimable authority.
        if any(item.claim_id is None for item in ordered):
            raise JournalIntegrityError("committed reservation lacks atomic submission claim")
        for key, decision in decisions.items():
            has_reservation = key in reservations
            if (decision["outcome"] == DecisionOutcome.ALLOW.value) != has_reservation:
                raise JournalIntegrityError(
                    "decision/reservation authorization transition mismatch"
                )
        active = tuple(item for item in ordered if not item.released)
        self._validate_active_scopes(active)
        quarantined = tuple(item for item in active if item.quarantined)
        return ReplayState(
            last_sequence=len(events),
            last_chain_hash=previous_hash,
            events=tuple(events),
            reservations=ordered,
            active_reservations=active,
            quarantined_reservations=quarantined,
        )

    @staticmethod
    def _read_only_authorizer(
        action: int,
        arg1: Optional[str],
        arg2: Optional[str],
        database_name: Optional[str],
        trigger_name: Optional[str],
    ) -> int:
        del arg1, arg2, database_name, trigger_name
        allowed = {
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_PRAGMA,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_SELECT,
        }
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        rows = connection.execute("""
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            """).fetchall()
        actual = {(row[0], row[1], row[2]) for row in rows}
        expected = {
            ("table", "safety_schema_version", "safety_schema_version"),
            ("table", "safety_journal_events", "safety_journal_events"),
            (
                "index",
                "safety_journal_idempotency_idx",
                "safety_journal_events",
            ),
            (
                "index",
                "safety_journal_account_contract_idx",
                "safety_journal_events",
            ),
            (
                "index",
                "safety_journal_portfolio_contract_idx",
                "safety_journal_events",
            ),
            (
                "trigger",
                "safety_schema_version_no_update",
                "safety_schema_version",
            ),
            (
                "trigger",
                "safety_schema_version_no_delete",
                "safety_schema_version",
            ),
            ("trigger", "safety_journal_no_update", "safety_journal_events"),
            ("trigger", "safety_journal_no_delete", "safety_journal_events"),
        }
        if actual != expected:
            raise JournalIntegrityError("journal sqlite_master objects do not match schema")
        expected_sql = {
            "safety_schema_version": """
                CREATE TABLE safety_schema_version (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    version INTEGER NOT NULL
                )
            """,
            "safety_journal_events": """
                CREATE TABLE safety_journal_events (
                    sequence INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    execution_domain_scope TEXT NOT NULL,
                    account_scope TEXT NOT NULL,
                    portfolio_id TEXT NOT NULL,
                    con_id INTEGER NOT NULL CHECK (con_id > 0),
                    intent_fingerprint TEXT NOT NULL,
                    claim_id TEXT,
                    payload_json TEXT NOT NULL,
                    previous_chain_hash TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    chain_hash TEXT NOT NULL UNIQUE,
                    schema_version INTEGER NOT NULL
                )
            """,
            "safety_journal_idempotency_idx": """
                CREATE INDEX safety_journal_idempotency_idx
                ON safety_journal_events (idempotency_key, sequence)
            """,
            "safety_journal_account_contract_idx": """
                CREATE INDEX safety_journal_account_contract_idx
                ON safety_journal_events
                (account_scope, con_id, sequence)
            """,
            "safety_journal_portfolio_contract_idx": """
                CREATE INDEX safety_journal_portfolio_contract_idx
                ON safety_journal_events
                (account_scope, portfolio_id, con_id, sequence)
            """,
            "safety_schema_version_no_update": """
                CREATE TRIGGER safety_schema_version_no_update
                BEFORE UPDATE ON safety_schema_version
                BEGIN SELECT RAISE(ABORT, 'safety schema version is immutable'); END
            """,
            "safety_schema_version_no_delete": """
                CREATE TRIGGER safety_schema_version_no_delete
                BEFORE DELETE ON safety_schema_version
                BEGIN SELECT RAISE(ABORT, 'safety schema version is immutable'); END
            """,
            "safety_journal_no_update": """
                CREATE TRIGGER safety_journal_no_update
                BEFORE UPDATE ON safety_journal_events
                BEGIN SELECT RAISE(ABORT, 'safety journal is append-only'); END
            """,
            "safety_journal_no_delete": """
                CREATE TRIGGER safety_journal_no_delete
                BEFORE DELETE ON safety_journal_events
                BEGIN SELECT RAISE(ABORT, 'safety journal is append-only'); END
            """,
        }

        def normalize(sql: str) -> str:
            return " ".join(sql.lower().split())

        for _, name, _, sql in rows:
            if sql is None or normalize(sql) != normalize(expected_sql[name]):
                raise JournalIntegrityError(
                    f"journal sqlite_master definition for {name} does not match schema"
                )

        event_columns = tuple(
            (row[1], row[2], row[3], row[5])
            for row in connection.execute("PRAGMA table_info(safety_journal_events)").fetchall()
        )
        expected_event_columns = (
            ("sequence", "INTEGER", 0, 1),
            ("event_id", "TEXT", 1, 0),
            ("event_type", "TEXT", 1, 0),
            ("occurred_at", "TEXT", 1, 0),
            ("idempotency_key", "TEXT", 1, 0),
            ("execution_domain_scope", "TEXT", 1, 0),
            ("account_scope", "TEXT", 1, 0),
            ("portfolio_id", "TEXT", 1, 0),
            ("con_id", "INTEGER", 1, 0),
            ("intent_fingerprint", "TEXT", 1, 0),
            ("claim_id", "TEXT", 0, 0),
            ("payload_json", "TEXT", 1, 0),
            ("previous_chain_hash", "TEXT", 1, 0),
            ("payload_hash", "TEXT", 1, 0),
            ("chain_hash", "TEXT", 1, 0),
            ("schema_version", "INTEGER", 1, 0),
        )
        schema_columns = tuple(
            (row[1], row[2], row[3], row[5])
            for row in connection.execute("PRAGMA table_info(safety_schema_version)").fetchall()
        )
        if event_columns != expected_event_columns or schema_columns != (
            ("singleton", "INTEGER", 0, 1),
            ("version", "INTEGER", 1, 0),
        ):
            raise JournalIntegrityError("journal table definitions do not match schema")

        expected_indexes = {
            "safety_journal_idempotency_idx": ("idempotency_key", "sequence"),
            "safety_journal_account_contract_idx": (
                "account_scope",
                "con_id",
                "sequence",
            ),
            "safety_journal_portfolio_contract_idx": (
                "account_scope",
                "portfolio_id",
                "con_id",
                "sequence",
            ),
        }
        for name, columns in expected_indexes.items():
            actual_columns = tuple(
                row[2] for row in connection.execute(f"PRAGMA index_info({name})").fetchall()
            )
            if actual_columns != columns:
                raise JournalIntegrityError(f"journal index {name} does not match schema")

        trigger_sql = {
            row[1]: " ".join((row[3] or "").lower().split()) for row in rows if row[0] == "trigger"
        }
        trigger_requirements = {
            "safety_schema_version_no_update": (
                "before update on safety_schema_version",
                "safety schema version is immutable",
            ),
            "safety_schema_version_no_delete": (
                "before delete on safety_schema_version",
                "safety schema version is immutable",
            ),
            "safety_journal_no_update": (
                "before update on safety_journal_events",
                "safety journal is append-only",
            ),
            "safety_journal_no_delete": (
                "before delete on safety_journal_events",
                "safety journal is append-only",
            ),
        }
        for name, fragments in trigger_requirements.items():
            sql = trigger_sql.get(name, "")
            if any(fragment not in sql for fragment in fragments):
                raise JournalIntegrityError(f"journal trigger {name} does not match schema")

    def _apply_event(
        self,
        reservations: Dict[str, ReplayReservation],
        decisions: Dict[str, dict],
        dispatch_times: Dict[str, datetime],
        event: JournalEvent,
        payload: dict,
    ) -> None:
        current = reservations.get(event.idempotency_key)
        if event.event_type is JournalEventType.SAFETY_DECISION:
            if event.idempotency_key in decisions or current is not None:
                raise JournalIntegrityError("duplicate safety decision")
            decision = payload.get("decision")
            context = payload.get("authorization_context")
            context_fingerprint = payload.get("authorization_context_fingerprint")
            if (
                not isinstance(decision, dict)
                or payload.get("intent_fingerprint") != event.intent_fingerprint
                or decision.get("intent_fingerprint") != event.intent_fingerprint
                or decision.get("outcome")
                not in {DecisionOutcome.ALLOW.value, DecisionOutcome.DENY.value}
                or not isinstance(decision.get("reason_codes"), list)
                or not decision["reason_codes"]
                or not isinstance(context, dict)
                or not isinstance(context_fingerprint, str)
                or not _HASH_RE.fullmatch(context_fingerprint)
                or sha256_text(canonical_json(context)) != context_fingerprint
            ):
                raise JournalIntegrityError("invalid safety decision payload")
            try:
                decision_model = SafetyDecision(
                    outcome=DecisionOutcome(decision["outcome"]),
                    risk_effect=RiskEffect(decision["risk_effect"]),
                    reason_codes=tuple(decision["reason_codes"]),
                    current_quantity=(
                        None
                        if decision["current_quantity"] is None
                        else parse_fixed_decimal(decision["current_quantity"], "current_quantity")
                    ),
                    computed_target_quantity=(
                        None
                        if decision["computed_target_quantity"] is None
                        else parse_fixed_decimal(
                            decision["computed_target_quantity"],
                            "computed_target_quantity",
                        )
                    ),
                    intent_fingerprint=decision["intent_fingerprint"],
                    schema_version=decision["schema_version"],
                )
            except (KeyError, ValueError, TypeError, ValidationError) as exc:
                raise JournalIntegrityError("invalid safety decision model") from exc
            if json.loads(decision_model.canonical_payload()) != decision:
                raise JournalIntegrityError("noncanonical safety decision")
            decisions[event.idempotency_key] = {
                "authorization_context_fingerprint": context_fingerprint,
                "decision": decision,
                "intent_fingerprint": event.intent_fingerprint,
                "outcome": decision["outcome"],
            }
            return
        if event.event_type is JournalEventType.RESERVATION_ACQUIRED:
            if current is not None:
                raise JournalIntegrityError("duplicate reservation for idempotency key")
            decision = decisions.get(event.idempotency_key)
            if decision is None or decision["outcome"] != DecisionOutcome.ALLOW.value:
                raise JournalIntegrityError("reservation lacks an ALLOW decision")
            intent = payload.get("intent")
            required = {
                "account_scope",
                "account_current_quantity",
                "con_id",
                "created_at",
                "execution_domain_scope",
                "portfolio_id",
                "portfolio_current_quantity",
                "portfolio_target_quantity",
                "quantity",
                "reason",
                "reduce_only",
                "schema_version",
                "side",
                "strategy",
                "symbol",
                "target_quantity",
            }
            if not isinstance(intent, dict) or set(intent) != required:
                raise JournalIntegrityError("reservation intent fields are incomplete")
            try:
                intent_model = OrderIntent(
                    execution_domain_scope=intent["execution_domain_scope"],
                    account_scope=intent["account_scope"],
                    portfolio_id=intent["portfolio_id"],
                    con_id=intent["con_id"],
                    symbol=intent["symbol"],
                    side=OrderSide(intent["side"]),
                    quantity=parse_fixed_decimal(intent["quantity"], "quantity"),
                    account_current_quantity=parse_fixed_decimal(
                        intent["account_current_quantity"],
                        "account_current_quantity",
                    ),
                    target_quantity=parse_fixed_decimal(
                        intent["target_quantity"], "target_quantity"
                    ),
                    portfolio_current_quantity=parse_fixed_decimal(
                        intent["portfolio_current_quantity"],
                        "portfolio_current_quantity",
                    ),
                    portfolio_target_quantity=parse_fixed_decimal(
                        intent["portfolio_target_quantity"],
                        "portfolio_target_quantity",
                    ),
                    created_at=parse_utc_text(intent["created_at"], "created_at"),
                    reduce_only=intent["reduce_only"],
                    reason=intent["reason"],
                    strategy=intent["strategy"],
                    schema_version=intent["schema_version"],
                )
            except (KeyError, ValueError, TypeError, ValidationError) as exc:
                raise JournalIntegrityError("invalid persisted order intent") from exc
            if payload.get("intent_fingerprint") != event.intent_fingerprint:
                raise JournalIntegrityError("reservation fingerprint mismatch")
            if intent_model.fingerprint() != event.intent_fingerprint:
                raise JournalIntegrityError("invalid intent fingerprint")
            if (
                intent["execution_domain_scope"] != event.execution_domain_scope
                or intent["account_scope"] != event.account_scope
                or intent["portfolio_id"] != event.portfolio_id
                or intent["con_id"] != event.con_id
            ):
                raise JournalIntegrityError("reservation columns mismatch intent")
            reservation_id = payload.get("reservation_id")
            _strict_internal_id(reservation_id, "reservation_id", "res")
            reservations[event.idempotency_key] = ReplayReservation(
                reservation_id=reservation_id,
                idempotency_key=event.idempotency_key,
                intent_fingerprint=event.intent_fingerprint,
                execution_domain_scope=event.execution_domain_scope,
                account_scope=event.account_scope,
                portfolio_id=event.portfolio_id,
                con_id=event.con_id,
                symbol=intent_model.symbol,
                side=intent_model.side,
                quantity=intent_model.quantity,
                target_quantity=intent_model.target_quantity,
                portfolio_target_quantity=intent_model.portfolio_target_quantity,
                acquired_at=event.occurred_at,
                acquired_sequence=event.sequence,
                claim_id=None,
                reservation_sequence=event.sequence,
                submission_descriptor_fingerprint=None,
                order_ref=None,
                claim_sequence=None,
                claim_time=None,
                outcome_unknown=False,
                quarantined=True,
                released=False,
                terminal_sequence=None,
            )
            return

        if current is None:
            raise JournalIntegrityError("transition lacks reservation")
        if (
            current.intent_fingerprint != event.intent_fingerprint
            or current.execution_domain_scope != event.execution_domain_scope
            or current.account_scope != event.account_scope
            or current.portfolio_id != event.portfolio_id
            or current.con_id != event.con_id
        ):
            raise JournalIntegrityError("transition changed immutable reservation scope")
        if current.released:
            raise JournalIntegrityError("event exists after terminal release")
        if payload.get("reservation_id") != current.reservation_id:
            raise JournalIntegrityError("transition reservation mismatch")

        if event.event_type is JournalEventType.SUBMISSION_STARTED:
            if current.claim_id is not None:
                raise JournalIntegrityError("submission authority granted more than once")
            descriptor = payload.get("descriptor")
            descriptor_fingerprint = payload.get("submission_descriptor_fingerprint")
            if not isinstance(descriptor, dict):
                raise JournalIntegrityError("claim lacks descriptor")
            try:
                descriptor_model = SubmissionDescriptor(
                    execution_domain_scope=descriptor["execution_domain_scope"],
                    account_scope=descriptor["account_scope"],
                    con_id=descriptor["con_id"],
                    side=OrderSide(descriptor["side"]),
                    quantity=parse_fixed_decimal(descriptor["quantity"], "quantity"),
                    order_type=OrderType(descriptor["order_type"]),
                    limit_price=(
                        None
                        if descriptor["limit_price"] is None
                        else parse_fixed_decimal(descriptor["limit_price"], "limit_price")
                    ),
                    stop_price=(
                        None
                        if descriptor["stop_price"] is None
                        else parse_fixed_decimal(descriptor["stop_price"], "stop_price")
                    ),
                    time_in_force=TimeInForce(descriptor["time_in_force"]),
                    outside_regular_hours=descriptor["outside_regular_hours"],
                    order_ref=descriptor["order_ref"],
                    attempt_number=descriptor["attempt_number"],
                    slice_count=descriptor["slice_count"],
                    bracket=descriptor["bracket"],
                    schema_version=descriptor["schema_version"],
                )
            except (KeyError, ValueError, TypeError, ValidationError) as exc:
                raise JournalIntegrityError("invalid persisted submission descriptor") from exc
            if descriptor_model.fingerprint() != descriptor_fingerprint:
                raise JournalIntegrityError("descriptor fingerprint mismatch")
            if (
                descriptor_model.execution_domain_scope != current.execution_domain_scope
                or descriptor_model.account_scope != current.account_scope
                or descriptor_model.con_id != current.con_id
                or descriptor_model.side is not current.side
                or descriptor_model.quantity != current.quantity
            ):
                raise JournalIntegrityError("descriptor changed authorized terms")
            if payload.get("reservation_sequence") != current.acquired_sequence:
                raise JournalIntegrityError("claim reservation sequence mismatch")
            if event.claim_id is None or payload.get("claim_id") != event.claim_id:
                raise JournalIntegrityError("claim correlation mismatch")
            order_ref = descriptor_model.order_ref
            reservations[event.idempotency_key] = replace(
                current,
                claim_id=event.claim_id,
                submission_descriptor_fingerprint=descriptor_fingerprint,
                order_ref=order_ref,
                claim_sequence=event.sequence,
                claim_time=event.occurred_at,
                quarantined=True,
            )
        elif event.event_type is JournalEventType.OUTCOME_UNKNOWN:
            if current.claim_id is None or event.claim_id != current.claim_id:
                raise JournalIntegrityError("unknown outcome lacks exact claim")
            if current.outcome_unknown:
                raise JournalIntegrityError("duplicate unknown outcome")
            expected_keys = {"claim_id", "reservation_id"}
            if "dispatch_state" in payload:
                expected_keys.add("dispatch_state")
                if payload["dispatch_state"] != "PERMIT_CONSUMED":
                    raise JournalIntegrityError("invalid durable dispatch state")
                if event.idempotency_key in dispatch_times:
                    raise JournalIntegrityError("permit was durably dispatched more than once")
                dispatch_times[event.idempotency_key] = event.occurred_at
            if set(payload) != expected_keys:
                raise JournalIntegrityError("unknown-outcome payload fields are invalid")
            reservations[event.idempotency_key] = replace(
                current, outcome_unknown=True, quarantined=True
            )
        elif event.event_type is JournalEventType.TERMINAL_RECONCILED:
            if current.claim_id is None or event.claim_id != current.claim_id:
                raise JournalIntegrityError("terminal event lacks exact claim")
            evidence = payload.get("evidence")
            if not isinstance(evidence, dict):
                raise JournalIntegrityError("terminal event lacks evidence")
            exact = (
                evidence.get("execution_domain_scope") == current.execution_domain_scope
                and evidence.get("account_scope") == current.account_scope
                and evidence.get("portfolio_id") == current.portfolio_id
                and evidence.get("con_id") == current.con_id
                and evidence.get("reservation_id") == current.reservation_id
                and evidence.get("claim_id") == current.claim_id
                and evidence.get("claim_sequence") == current.claim_sequence
                and evidence.get("submission_descriptor_fingerprint")
                == current.submission_descriptor_fingerprint
                and evidence.get("order_ref") == current.order_ref
                and evidence.get("symbol") == current.symbol
            )
            if not exact:
                raise JournalIntegrityError("terminal evidence correlation mismatch")
            observed = parse_utc_text(evidence.get("observed_at"), "observed_at")
            position_observed = parse_utc_text(
                evidence.get("position_observed_at"), "position_observed_at"
            )
            if (
                current.claim_time is None
                or observed <= current.claim_time
                or position_observed <= current.claim_time
                or observed > event.occurred_at
                or position_observed > event.occurred_at
            ):
                raise JournalIntegrityError("terminal evidence time is invalid")
            max_age = evidence.get("max_evidence_age_seconds")
            if (
                type(max_age) is not int
                or not 0 <= max_age <= SAFETY_MAX_EVIDENCE_AGE_SECONDS
                or (event.occurred_at - observed).total_seconds() > max_age
                or (event.occurred_at - position_observed).total_seconds() > max_age
            ):
                raise JournalIntegrityError("terminal evidence is stale")
            filled = parse_fixed_decimal(evidence.get("filled_quantity"), "filled_quantity")
            remaining = parse_fixed_decimal(
                evidence.get("remaining_quantity"),
                "remaining_quantity",
            )
            try:
                terminal_quantity = _exact_decimal_add(
                    filled,
                    remaining,
                    "terminal claimed quantity",
                )
            except ValidationError as exc:
                raise JournalIntegrityError("terminal quantity arithmetic is invalid") from exc
            authoritative = (
                evidence.get("status") == ReconciliationStatus.PASSED.value
                and evidence.get("transport_state") == TransportState.CONNECTED.value
                and evidence.get("open_orders_complete") is True
                and evidence.get("open_orders_all_clients") is True
                and evidence.get("open_orders_snapshot_stable") is True
                and evidence.get("active_order_count") == 0
                and evidence.get("has_offsetting_allocations") is False
                and evidence.get("terminal_order_status")
                in {item.value for item in TerminalOrderStatus}
                and parse_fixed_decimal(
                    evidence.get("aggregate_allocated_quantity"),
                    "aggregate_allocated_quantity",
                )
                == parse_fixed_decimal(
                    evidence.get("account_position_quantity"),
                    "account_position_quantity",
                )
                and terminal_quantity == current.quantity
            )
            if not authoritative:
                raise JournalIntegrityError("terminal evidence is not authoritative")
            dispatch_time = dispatch_times.get(event.idempotency_key)
            terminal_status = evidence.get("terminal_order_status")
            if terminal_status == TerminalOrderStatus.NO_SUBMISSION_CONFIRMED.value:
                if dispatch_time is not None:
                    raise JournalIntegrityError("no-submission terminal follows durable dispatch")
            elif dispatch_time is None:
                raise JournalIntegrityError("terminal broker outcome lacks durable permit dispatch")
            elif observed <= dispatch_time or position_observed <= dispatch_time:
                raise JournalIntegrityError("terminal evidence does not strictly postdate dispatch")
            signed_full = (
                current.quantity
                if current.side is OrderSide.BUY_TO_COVER
                else current.quantity.copy_negate()
            )
            signed_fill = (
                filled
                if current.side is OrderSide.BUY_TO_COVER or filled.is_zero()
                else filled.copy_negate()
            )
            try:
                initial_account = _exact_decimal_subtract(
                    current.target_quantity,
                    signed_full,
                    "initial account quantity",
                )
                initial_portfolio = _exact_decimal_subtract(
                    current.portfolio_target_quantity,
                    signed_full,
                    "initial portfolio quantity",
                )
                expected_account = _exact_decimal_add(
                    initial_account,
                    signed_fill,
                    "expected account quantity",
                )
                expected_portfolio = _exact_decimal_add(
                    initial_portfolio,
                    signed_fill,
                    "expected portfolio quantity",
                )
            except ValidationError as exc:
                raise JournalIntegrityError("terminal position arithmetic is invalid") from exc
            if (
                parse_fixed_decimal(
                    evidence.get("account_position_quantity"),
                    "account_position_quantity",
                )
                != expected_account
                or parse_fixed_decimal(
                    evidence.get("portfolio_position_quantity"),
                    "portfolio_position_quantity",
                )
                != expected_portfolio
            ):
                raise JournalIntegrityError("terminal positions do not match exact claimed fill")
            if (
                evidence.get("terminal_order_status") == TerminalOrderStatus.FILLED.value
                and parse_fixed_decimal(evidence.get("remaining_quantity"), "remaining_quantity")
                != 0
            ):
                raise JournalIntegrityError("filled order has remaining quantity")
            if evidence.get("terminal_order_status") in {
                TerminalOrderStatus.REJECTED.value,
                TerminalOrderStatus.NO_SUBMISSION_CONFIRMED.value,
            } and (
                parse_fixed_decimal(evidence.get("filled_quantity"), "filled_quantity") != 0
                or parse_fixed_decimal(evidence.get("remaining_quantity"), "remaining_quantity")
                != current.quantity
            ):
                raise JournalIntegrityError(
                    "unsubmitted/rejected order quantities are inconsistent"
                )
            reservations[event.idempotency_key] = replace(
                current,
                quarantined=False,
                released=True,
                terminal_sequence=event.sequence,
            )
        else:  # pragma: no cover
            raise JournalIntegrityError("unknown transition")

    def _event_from_row(self, row: tuple) -> JournalEvent:
        try:
            (
                sequence,
                event_id,
                event_type,
                occurred_at,
                idempotency_key,
                execution_domain_scope,
                account_scope,
                portfolio_id,
                con_id,
                intent_fingerprint,
                claim_id,
                payload_json,
                previous_chain_hash,
                payload_hash,
                chain_hash,
                schema_version,
            ) = row
            if type(sequence) is not int or sequence <= 0:
                raise JournalIntegrityError("invalid sequence")
            if type(con_id) is not int or con_id <= 0:
                raise JournalIntegrityError("invalid conId")
            if schema_version != MODEL_VERSION:
                raise JournalIntegrityError("unsupported event model version")
            event_type = JournalEventType(event_type)
            occurred_at = parse_utc_text(occurred_at, "occurred_at")
            for value, field in (
                (idempotency_key, "idempotency_key"),
                (execution_domain_scope, "execution_domain_scope"),
                (portfolio_id, "portfolio_id"),
            ):
                _strict_text(value, field, max_length=128)
            _strict_internal_id(event_id, "event_id", "evt")
            _strict_account_scope(account_scope)
            for value, field in (
                (intent_fingerprint, "intent_fingerprint"),
                (previous_chain_hash, "previous_chain_hash"),
                (payload_hash, "payload_hash"),
                (chain_hash, "chain_hash"),
            ):
                self._validate_hash(value, field)
            if claim_id is not None:
                _strict_internal_id(claim_id, "claim_id", "claim")
            if not isinstance(payload_json, str):
                raise JournalIntegrityError("payload must be text")
            return JournalEvent(
                sequence=sequence,
                event_id=event_id,
                event_type=event_type,
                occurred_at=occurred_at,
                idempotency_key=idempotency_key,
                execution_domain_scope=execution_domain_scope,
                account_scope=account_scope,
                portfolio_id=portfolio_id,
                con_id=con_id,
                intent_fingerprint=intent_fingerprint,
                claim_id=claim_id,
                payload_json=payload_json,
                previous_chain_hash=previous_chain_hash,
                payload_hash=payload_hash,
                chain_hash=chain_hash,
                schema_version=schema_version,
            )
        except (ValueError, TypeError, ValidationError) as exc:
            raise JournalIntegrityError("invalid journal event") from exc

    def _last_event(self, connection: sqlite3.Connection) -> Optional[JournalEvent]:
        row = connection.execute("""
            SELECT sequence, event_id, event_type, occurred_at,
                   idempotency_key, execution_domain_scope, account_scope,
                   portfolio_id, con_id, intent_fingerprint, claim_id,
                   payload_json, previous_chain_hash, payload_hash,
                   chain_hash, schema_version
            FROM safety_journal_events ORDER BY sequence DESC LIMIT 1
            """).fetchone()
        return None if row is None else self._event_from_row(row)

    @staticmethod
    def _find_reservation(
        reservations: Iterable[ReplayReservation], key: str
    ) -> Optional[ReplayReservation]:
        return next((item for item in reservations if item.idempotency_key == key), None)

    def _require_reservation(
        self, state: ReplayState, key: str, fingerprint: str
    ) -> ReplayReservation:
        result = self._find_reservation(state.reservations, key)
        if result is None:
            raise StateTransitionError("idempotency key has no reservation")
        if result.intent_fingerprint != fingerprint:
            raise IdempotencyConflict("idempotency key is bound to another intent")
        return result

    @staticmethod
    def _permit_consumed_event_time(
        state: ReplayState, reservation: ReplayReservation
    ) -> Optional[datetime]:
        for event in reversed(state.events):
            if (
                event.idempotency_key == reservation.idempotency_key
                and event.event_type is JournalEventType.OUTCOME_UNKNOWN
            ):
                payload = json.loads(event.payload_json)
                if payload.get("dispatch_state") == "PERMIT_CONSUMED":
                    return event.occurred_at
        return None

    @staticmethod
    def _decision_binding(events: Iterable[JournalEvent], key: str) -> Optional[dict]:
        for event in events:
            if (
                event.idempotency_key == key
                and event.event_type is JournalEventType.SAFETY_DECISION
            ):
                payload = json.loads(event.payload_json)
                return {
                    "authorization_context_fingerprint": payload[
                        "authorization_context_fingerprint"
                    ],
                    "decision": payload["decision"],
                    "intent_fingerprint": payload["intent_fingerprint"],
                }
        return None

    @staticmethod
    def _authorization_context(
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
    ) -> dict:
        return {
            "allocation": json.loads(allocation.canonical_payload()),
            "exposure": json.loads(exposure.canonical_payload()),
            "gates": json.loads(gates.canonical_payload()),
        }

    @staticmethod
    def _context_snapshot_keys(context: dict) -> set[tuple]:
        try:
            exposure = context["exposure"]
            allocation = context["allocation"]
            gates = context["gates"]
            common = (
                exposure["account_scope"],
                exposure["con_id"],
            )
            return {
                ("exposure", *common, exposure["snapshot_id"]),
                (
                    "allocation",
                    *common,
                    allocation["portfolio_id"],
                    allocation["snapshot_id"],
                ),
                ("open_orders", *common, gates["open_orders_snapshot_id"]),
            }
        except (KeyError, TypeError) as exc:
            raise JournalIntegrityError(
                "authorization context snapshot identifiers are incomplete"
            ) from exc

    @staticmethod
    def _snapshots_not_newer_than_terminal(
        state: ReplayState,
        intent: OrderIntent,
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
    ) -> bool:
        for event in reversed(state.events):
            if event.event_type is not JournalEventType.TERMINAL_RECONCILED:
                continue
            if event.account_scope != intent.account_scope or event.con_id != intent.con_id:
                continue
            evidence = json.loads(event.payload_json)["evidence"]
            terminal_position_time = parse_utc_text(
                evidence["position_observed_at"], "position_observed_at"
            )
            terminal_order_time = parse_utc_text(evidence["observed_at"], "observed_at")
            return (
                exposure.observed_at <= terminal_position_time
                or allocation.observed_at <= terminal_position_time
                or gates.open_orders_observed_at <= terminal_order_time
            )
        return False

    @staticmethod
    def _evaluate_at_authorization(
        intent: OrderIntent,
        exposure: ExposureEvidence,
        allocation: PortfolioAllocationEvidence,
        gates: GateContext,
        authorization_time: datetime,
    ) -> SafetyDecision:
        decision = evaluate_reduce_only(intent, exposure, allocation, gates)
        if decision.outcome is DecisionOutcome.DENY:
            return decision
        if gates.evaluated_at > authorization_time:
            return SafetyDecision(
                outcome=DecisionOutcome.DENY,
                risk_effect=decision.risk_effect,
                reason_codes=("FUTURE_AUTHORIZATION_CONTEXT",),
                current_quantity=decision.current_quantity,
                computed_target_quantity=decision.computed_target_quantity,
                intent_fingerprint=decision.intent_fingerprint,
            )
        if (
            authorization_time - gates.evaluated_at
        ).total_seconds() > gates.max_evidence_age_seconds:
            return SafetyDecision(
                outcome=DecisionOutcome.DENY,
                risk_effect=decision.risk_effect,
                reason_codes=("STALE_AUTHORIZATION_CONTEXT",),
                current_quantity=decision.current_quantity,
                computed_target_quantity=decision.computed_target_quantity,
                intent_fingerprint=decision.intent_fingerprint,
            )
        return decision

    @staticmethod
    def _reservation_model(value: ReplayReservation, *, newly_acquired: bool) -> Reservation:
        return Reservation(
            reservation_id=value.reservation_id,
            idempotency_key=value.idempotency_key,
            intent_fingerprint=value.intent_fingerprint,
            execution_domain_scope=value.execution_domain_scope,
            account_scope=value.account_scope,
            portfolio_id=value.portfolio_id,
            con_id=value.con_id,
            sequence=value.acquired_sequence,
            acquired_at=value.acquired_at,
            newly_acquired=newly_acquired,
        )

    @staticmethod
    def _claim_model(value: ReplayReservation, *, granted: bool) -> SubmissionClaim:
        if (
            value.claim_id is None
            or value.claim_sequence is None
            or value.claim_time is None
            or value.submission_descriptor_fingerprint is None
            or value.order_ref is None
        ):
            raise JournalIntegrityError("reservation lacks durable claim")
        return SubmissionClaim(
            claim_id=value.claim_id,
            reservation_id=value.reservation_id,
            reservation_sequence=value.acquired_sequence,
            idempotency_key=value.idempotency_key,
            submission_descriptor_fingerprint=value.submission_descriptor_fingerprint,
            execution_domain_scope=value.execution_domain_scope,
            account_scope=value.account_scope,
            portfolio_id=value.portfolio_id,
            con_id=value.con_id,
            order_ref=value.order_ref,
            sequence=value.claim_sequence,
            claimed_at=value.claim_time,
            granted=granted,
        )

    @staticmethod
    def _validate_active_scopes(active: Tuple[ReplayReservation, ...]) -> None:
        account_contract = set()
        portfolio_contract = set()
        for item in active:
            account_key = (
                item.account_scope,
                item.con_id,
            )
            portfolio_key = (
                item.account_scope,
                item.portfolio_id,
                item.con_id,
            )
            if account_key in account_contract or portfolio_key in portfolio_contract:
                raise JournalIntegrityError("overlapping active reservation scopes")
            account_contract.add(account_key)
            portfolio_contract.add(portfolio_key)

    def _event_time(self, supplied: Optional[datetime] = None) -> datetime:
        value = self._clock() if supplied is None else supplied
        return parse_utc_text(utc_to_text(value), "occurred_at")

    @staticmethod
    def _validate_hash(value: object, field: str) -> None:
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            raise ValidationError(f"{field} must be a lowercase SHA-256 digest")

    @staticmethod
    def _chain_hash(
        sequence: int,
        event_type: JournalEventType,
        occurred_at: str,
        idempotency_key: str,
        execution_domain_scope: str,
        account_scope: str,
        portfolio_id: str,
        con_id: int,
        intent_fingerprint: str,
        claim_id: Optional[str],
        payload_hash: str,
        previous_hash: str,
        schema_version: int,
    ) -> str:
        return sha256_text(
            canonical_json(
                {
                    "account_scope": account_scope,
                    "claim_id": claim_id,
                    "con_id": con_id,
                    "event_type": event_type,
                    "execution_domain_scope": execution_domain_scope,
                    "idempotency_key": idempotency_key,
                    "intent_fingerprint": intent_fingerprint,
                    "occurred_at": occurred_at,
                    "payload_hash": payload_hash,
                    "portfolio_id": portfolio_id,
                    "previous_chain_hash": previous_hash,
                    "schema_version": schema_version,
                    "sequence": sequence,
                }
            )
        )

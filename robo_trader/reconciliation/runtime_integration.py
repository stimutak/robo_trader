"""Production-only runtime integration for signed broker reconciliation.

This module owns the diagnostic read-only provider, produces one signed
broker/reconciliation/mark bundle per trigger, binds it to the exact SQLite
inode, and exposes only sanitized operational status.  It never repairs local
or broker state and never grants order authority.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import secrets
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import aiosqlite

from robo_trader.bootstrap_evidence_receivers import (
    BootstrapEvidenceReceiverSet,
    SealedBootstrapEvidenceArtifact,
    create_bootstrap_evidence_receivers,
)
from robo_trader.bootstrap_mark_producer import (
    collect_and_produce_bootstrap_protective_mark,
    create_runtime_bound_mark_only_producer,
)
from robo_trader.config import RuntimeContract
from robo_trader.database_migrations import assert_exact_state_schema
from robo_trader.financial_state_bootstrap import (
    ExactStateBootstrapCandidate,
    load_exact_state_bootstrap_evidence,
)
from robo_trader.reconciliation_migrations import assert_reconciliation_schema
from robo_trader.safety.sqlite_identity import SQLitePathBinding, lexical_path_preserving_leaf

from .bootstrap_producer import produce_bootstrap_reconciliation
from .ibkr_adapter import (
    IBKRDiagnosticSnapshotProvider,
    assert_factory_owned_protective_quote_source,
    await_cleanup_required,
    build_diagnostic_provider,
)
from .identity import RuntimeSafetyContext, assert_validated_runtime_safety_context
from .persistence import ReconciliationPersistence
from .runtime_evidence import (
    VerifiedRuntimeReconciliationEvidence,
    bind_verified_runtime_reconciliation_evidence,
)
from .service import ReconciliationService, ReconciliationServiceOutcome

_CAPABILITY_DIRECTORY_ENV = "RT_RECONCILIATION_SIGNING_CAPABILITY_DIR"
_EVIDENCE_ROOT_ENV = "RT_RECONCILIATION_EVIDENCE_ROOT"
_STATUS_PATH_ENV = "RT_RECONCILIATION_STATUS_PATH"
_STATUS_FILENAME = "reconciliation_runtime_status.json"
_STATUS_FIELDS = {
    "schema_version",
    "owner_binding",
    "state",
    "trigger",
    "completed_at",
    "eligible_until",
    "entry_eligible",
    "quarantined",
    "run_id",
    "snapshot_id",
}


class RuntimeReconciliationIntegrationError(RuntimeError):
    """Runtime reconciliation cannot establish a safe entry boundary."""


def _absolute_lexical_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeReconciliationIntegrationError(f"{label} is not configured")
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        raise RuntimeReconciliationIntegrationError(f"{label} must be an absolute lexical path")
    return path


def _require_private_directory(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise RuntimeReconciliationIntegrationError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RuntimeReconciliationIntegrationError(
            f"{label} must be an owner-only non-symlink directory"
        )


def runtime_reconciliation_status_path(
    runtime_contract: RuntimeContract,
    environment: Mapping[str, str],
) -> Path:
    configured = environment.get(_STATUS_PATH_ENV, "").strip()
    if configured:
        return _absolute_lexical_path(configured, "reconciliation status path")
    return Path(runtime_contract.database_path).parent / _STATUS_FILENAME


def _assert_status_path_is_unprotected(
    status_path: Path,
    *,
    runtime_contract: RuntimeContract,
    capability_directory: Path,
    evidence_root: Path,
) -> None:
    """Reject status targets that could replace protected runtime artifacts."""

    protected_files: set[Path] = set()
    for configured in (
        runtime_contract.database_path,
        runtime_contract.safety_journal_path,
    ):
        if not isinstance(configured, str) or not configured:
            continue
        protected = lexical_path_preserving_leaf(configured)
        protected_files.update(
            {
                protected,
                Path(f"{protected}-wal"),
                Path(f"{protected}-shm"),
                Path(f"{protected}-journal"),
            }
        )
    target = lexical_path_preserving_leaf(status_path)
    protected_directories = (
        lexical_path_preserving_leaf(capability_directory),
        lexical_path_preserving_leaf(evidence_root),
    )
    if target in protected_files or any(
        target == directory or directory in target.parents for directory in protected_directories
    ):
        raise RuntimeReconciliationIntegrationError(
            "reconciliation status path overlaps protected runtime state"
        )


def _status_owner_binding(runtime: RuntimeContract, status_path: Path) -> str:
    payload = (
        "robotrader-reconciliation-status-v1\0" + runtime.fingerprint + "\0" + str(status_path)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_status_owner_binding(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


async def assert_runtime_bootstrap_ready(runtime_context: RuntimeSafetyContext) -> None:
    """Read-only proof that every portfolio has complete exact bootstrap lineage."""

    context = assert_validated_runtime_safety_context(runtime_context)
    contract = context.runtime_contract
    if type(contract) is not RuntimeContract:
        raise RuntimeReconciliationIntegrationError("exact runtime contract is required")
    database_path = Path(contract.database_path)
    binding: SQLitePathBinding | None = None
    connection: aiosqlite.Connection | None = None
    try:
        binding = SQLitePathBinding.open_readonly(database_path)
        uri = f"file:{binding.path.as_posix()}?mode=ro"
        connection = await aiosqlite.connect(uri, uri=True, isolation_level=None)
        await connection.execute("PRAGMA query_only = ON")
        await connection.execute("PRAGMA foreign_keys = ON")
        await assert_exact_state_schema(connection)
        await assert_reconciliation_schema(connection)
        binding.assert_path_identity()

        cursor = await connection.execute("""
            SELECT p.id, b.bootstrap_id, b.execution_domain_scope, b.account_scope,
                   b.database_path, b.database_identity, b.database_device,
                   b.database_inode, a.origin_bootstrap_id,
                   b.candidate_payload_json, b.broker_snapshot_hash,
                   b.reconciliation_report_hash
            FROM portfolios AS p
            LEFT JOIN paper_state_bootstraps AS b ON b.portfolio_id = p.id
            LEFT JOIN paper_account_settlement_state AS a ON a.portfolio_id = p.id
            ORDER BY p.id
            """)
        portfolio_rows = await cursor.fetchall()
        if not portfolio_rows:
            raise RuntimeReconciliationIntegrationError("runtime portfolio state is unavailable")
        bootstrap_ids: dict[str, str] = {}
        expected_receipts: dict[str, dict[str, list[str]]] = {}
        for row in portfolio_rows:
            portfolio_id, bootstrap_id = row[0], row[1]
            if (
                not isinstance(portfolio_id, str)
                or not isinstance(bootstrap_id, str)
                or row[2] != contract.safety_execution_domain_scope
                or row[3] != contract.safety_account_scope
                or row[4] != contract.database_path
                or row[5] != contract.database_identity
                or (row[6], row[7]) != (binding.device, binding.inode)
                or row[8] != bootstrap_id
            ):
                raise RuntimeReconciliationIntegrationError(
                    "exact bootstrap lineage is missing or mismatched"
                )
            bootstrap_ids[portfolio_id] = bootstrap_id
            try:
                candidate_raw = json.loads(row[9])
                candidate = ExactStateBootstrapCandidate.from_mapping(candidate_raw)
            except Exception as exc:
                raise RuntimeReconciliationIntegrationError(
                    "exact bootstrap candidate is missing or malformed"
                ) from exc
            if (
                candidate.bootstrap_id != bootstrap_id
                or candidate.portfolio_id != portfolio_id
                or candidate.database_path != contract.database_path
                or candidate.database_identity != contract.database_identity
            ):
                raise RuntimeReconciliationIntegrationError(
                    "exact bootstrap candidate lineage is mismatched"
                )
            expected_receipts[bootstrap_id] = {
                "broker_snapshot": [str(row[10])],
                "reconciliation_report": [str(row[11])],
                "protective_mark": [
                    position.mark_evidence_fingerprint for position in candidate.positions
                ],
            }

        cursor = await connection.execute("""
            SELECT p.portfolio_id, p.symbol, s.cost_basis_text, s.mark_price_text,
                   s.origin_bootstrap_id
            FROM positions AS p
            LEFT JOIN paper_position_settlement_state AS s
              ON s.portfolio_id = p.portfolio_id AND s.symbol = p.symbol
            WHERE p.quantity <> 0
            ORDER BY p.portfolio_id, p.symbol
            """)
        position_rows = await cursor.fetchall()
        for row in position_rows:
            if (
                row[0] not in bootstrap_ids
                or not isinstance(row[2], str)
                or not row[2]
                or not isinstance(row[3], str)
                or not row[3]
                or row[4] != bootstrap_ids[row[0]]
            ):
                raise RuntimeReconciliationIntegrationError(
                    "exact position bootstrap state is missing or partial"
                )

        cursor = await connection.execute("""
            SELECT bootstrap_id, artifact_kind, artifact_sha256,
                   runtime_fingerprint, account_scope
            FROM exact_bootstrap_evidence_consumptions
            ORDER BY bootstrap_id, artifact_kind, receipt_id
            """)
        receipt_rows = await cursor.fetchall()
        observed_receipts: dict[str, dict[str, list[str]]] = {
            bootstrap_id: {
                "broker_snapshot": [],
                "reconciliation_report": [],
                "protective_mark": [],
            }
            for bootstrap_id in bootstrap_ids.values()
        }
        for (
            bootstrap_id,
            artifact_kind,
            artifact_hash,
            runtime_fingerprint,
            account_scope,
        ) in receipt_rows:
            if (
                bootstrap_id not in observed_receipts
                or artifact_kind not in observed_receipts[bootstrap_id]
                or runtime_fingerprint != contract.fingerprint
                or account_scope != contract.safety_account_scope
            ):
                raise RuntimeReconciliationIntegrationError(
                    "exact bootstrap authentication receipt lineage is mismatched"
                )
            observed_receipts[bootstrap_id][artifact_kind].append(artifact_hash)
        if any(
            {kind: sorted(hashes) for kind, hashes in observed_receipts[bootstrap_id].items()}
            != {kind: sorted(hashes) for kind, hashes in expected_receipts[bootstrap_id].items()}
            for bootstrap_id in expected_receipts
        ):
            raise RuntimeReconciliationIntegrationError(
                "exact bootstrap authentication receipts are incomplete"
            )
        binding.assert_path_identity()
    except RuntimeReconciliationIntegrationError:
        raise
    except Exception as exc:
        raise RuntimeReconciliationIntegrationError(
            "runtime database or exact bootstrap schema is unavailable"
        ) from exc
    finally:
        if connection is not None:
            await connection.close()
        if binding is not None:
            binding.close()


class ProductionRuntimeEvidenceSource:
    """Collect one fresh, signed, broker-bound generation per service trigger."""

    def __init__(
        self,
        *,
        runtime_context: RuntimeSafetyContext,
        provider: IBKRDiagnosticSnapshotProvider,
        capability_directory: Path,
        evidence_root: Path,
    ) -> None:
        self._runtime_context = assert_validated_runtime_safety_context(runtime_context)
        self._provider = provider
        self._capability_directory = capability_directory
        self._evidence_root = evidence_root
        self._closed = False

    async def collect_verified_evidence(
        self,
        *,
        max_age_seconds: float,
    ) -> VerifiedRuntimeReconciliationEvidence:
        if self._closed:
            raise RuntimeReconciliationIntegrationError("reconciliation evidence source is closed")
        if (
            not isinstance(max_age_seconds, (int, float))
            or isinstance(max_age_seconds, bool)
            or not math.isfinite(float(max_age_seconds))
            or float(max_age_seconds) <= 0
        ):
            raise RuntimeReconciliationIntegrationError("evidence freshness bound is invalid")
        context = assert_validated_runtime_safety_context(self._runtime_context)
        runtime = context.runtime_contract
        if type(runtime) is not RuntimeContract:
            raise RuntimeReconciliationIntegrationError("exact runtime contract is required")
        _require_private_directory(self._capability_directory, "signing capability directory")
        _require_private_directory(self._evidence_root, "reconciliation evidence root")
        output = self._evidence_root / ("runtime-reconciliation-" + secrets.token_hex(24))
        receivers: BootstrapEvidenceReceiverSet | None = None
        published = False
        try:
            receivers = create_bootstrap_evidence_receivers(
                runtime_contract=runtime,
                capability_directory=self._capability_directory,
                output_directory=output,
            )
            broker_envelope = await self._provider.produce_normalized_snapshot(
                receiver=receivers.broker_snapshot,
                max_age_seconds=float(max_age_seconds),
            )
            broker_artifact = receivers.broker_artifact
            delivery = produce_bootstrap_reconciliation(
                broker_envelope,
                runtime,
                receivers.reconciliation_report,
            )
            reconciliation_artifact = delivery.receiver_result
            if type(reconciliation_artifact) is not SealedBootstrapEvidenceArtifact:
                raise RuntimeReconciliationIntegrationError(
                    "reconciliation receiver returned invalid evidence"
                )
            quote_source = self._provider.issue_protective_quote_source(runtime_contract=runtime)
            quote_identity = assert_factory_owned_protective_quote_source(
                quote_source,
                runtime_contract=runtime,
            )
            mark_artifacts: list[SealedBootstrapEvidenceArtifact] = []
            for portfolio_id, symbol in delivery.local_position_identities:
                discovery = await quote_source.get_protective_quotes(
                    (symbol,),
                    active_symbols=(symbol,),
                )
                if type(discovery) is not tuple or len(discovery) != 1:
                    raise RuntimeReconciliationIntegrationError(
                        "protective quote evidence is incomplete"
                    )
                quote = discovery[0]
                if (
                    quote.symbol != symbol
                    or quote.transport_generation != quote_identity.transport_generation
                ):
                    raise RuntimeReconciliationIntegrationError(
                        "protective quote generation changed during collection"
                    )
                producer = create_runtime_bound_mark_only_producer(
                    runtime,
                    portfolio_id=portfolio_id,
                )
                mark = await collect_and_produce_bootstrap_protective_mark(
                    quote_source,
                    producer,
                    runtime,
                    receivers.protective_mark,
                    expected_portfolio_id=portfolio_id,
                    expected_symbol=symbol,
                    expected_con_id=quote.con_id,
                    expected_transport_generation=quote_identity.transport_generation,
                )
                if type(mark) is not SealedBootstrapEvidenceArtifact:
                    raise RuntimeReconciliationIntegrationError(
                        "protective mark receiver returned invalid evidence"
                    )
                mark_artifacts.append(mark)
            expected_marks = set(delivery.local_position_identities)
            receivers.assert_complete(expected_marks)
            receivers.publish_complete_bundle(expected_marks)
            published = True
            broker_path = receivers.published_artifact_path(broker_artifact)
            reconciliation_path = receivers.published_artifact_path(reconciliation_artifact)
            mark_paths = tuple(receivers.published_artifact_path(mark) for mark in mark_artifacts)
            exact_state = load_exact_state_bootstrap_evidence(
                reconciliation_path=reconciliation_path,
                broker_snapshot_path=broker_path,
                protective_mark_paths=mark_paths,
                expected_runtime_contract=runtime,
            )
            return bind_verified_runtime_reconciliation_evidence(
                delivery.verified_broker_evidence,
                exact_state,
                context,
                receivers.protective_mark,
            )
        except BaseException:
            if receivers is not None and not published:
                receivers.discard_unpublished_bundle()
            raise
        finally:
            if receivers is not None:
                receivers.close()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        cancellation_received = await await_cleanup_required(self._provider.close())
        if cancellation_received:
            raise asyncio.CancelledError


class RuntimeReconciliationController:
    """Fail-closed service facade plus sanitized cross-process status."""

    def __init__(
        self,
        service: ReconciliationService,
        *,
        status_path: Path,
        status_owner_binding: str | None = None,
    ) -> None:
        self._service = service
        self._status_path = status_path
        self._status_owner_binding = (
            status_owner_binding
            if status_owner_binding is not None
            else hashlib.sha256(secrets.token_bytes(32)).hexdigest()
        )
        if not _valid_status_owner_binding(self._status_owner_binding):
            raise RuntimeReconciliationIntegrationError(
                "reconciliation status owner binding is malformed"
            )
        self._status_available = True

    @property
    def service(self) -> ReconciliationService:
        return self._service

    def entry_eligible(self) -> bool:
        eligible = self._status_available and self._service.entry_eligible()
        if not eligible:
            self._publish_quarantine_best_effort()
        return eligible

    async def reconcile_startup(self) -> ReconciliationServiceOutcome:
        return await self._run(self._service.reconcile_startup)

    async def reconcile_reconnect(self) -> ReconciliationServiceOutcome:
        return await self._run(self._service.reconcile_reconnect)

    async def reconcile_periodic_if_due(self) -> ReconciliationServiceOutcome | None:
        try:
            outcome = await self._service.reconcile_periodic_if_due()
            if outcome is not None:
                self._publish_outcome(outcome)
            return outcome
        except BaseException:
            self._publish_quarantine_best_effort()
            raise

    async def _run(self, operation) -> ReconciliationServiceOutcome:
        try:
            outcome = await operation()
            self._publish_outcome(outcome)
            return outcome
        except BaseException:
            self._publish_quarantine_best_effort()
            raise

    async def close(self) -> None:
        try:
            await self._service.close()
        finally:
            self._publish_quarantine_best_effort(state="closed")

    def _publish_outcome(self, outcome: ReconciliationServiceOutcome) -> None:
        payload = {
            "schema_version": 1,
            "owner_binding": self._status_owner_binding,
            "state": outcome.state.value,
            "trigger": outcome.trigger.value,
            "completed_at": outcome.completed_at.isoformat(),
            "eligible_until": outcome.eligible_until.isoformat(),
            "entry_eligible": bool(outcome.entry_eligible),
            "quarantined": not bool(outcome.entry_eligible),
            "run_id": outcome.persisted.run_id,
            "snapshot_id": outcome.persisted.snapshot_id,
        }
        try:
            _write_status(
                self._status_path,
                payload,
                owner_binding=self._status_owner_binding,
            )
        except Exception as exc:
            self._status_available = False
            raise RuntimeReconciliationIntegrationError(
                "reconciliation status publication failed closed"
            ) from exc
        self._status_available = True

    def _publish_quarantine_best_effort(self, *, state: str = "quarantined") -> None:
        self._status_available = False
        payload = {
            "schema_version": 1,
            "owner_binding": self._status_owner_binding,
            "state": state,
            "trigger": None,
            "completed_at": None,
            "eligible_until": None,
            "entry_eligible": False,
            "quarantined": True,
            "run_id": None,
            "snapshot_id": None,
        }
        try:
            _write_status(
                self._status_path,
                payload,
                owner_binding=self._status_owner_binding,
            )
        except Exception:
            pass


def _read_owned_status_descriptor(descriptor: int, owner_binding: str) -> None:
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_size > 16 * 1024
    ):
        raise RuntimeReconciliationIntegrationError(
            "existing reconciliation status artifact is not owner-bound"
        )
    encoded = os.read(descriptor, 16 * 1024 + 1)
    if len(encoded) != metadata.st_size:
        raise RuntimeReconciliationIntegrationError(
            "existing reconciliation status artifact changed while inspected"
        )
    try:
        prior = json.loads(encoded.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeReconciliationIntegrationError(
            "existing reconciliation status artifact is malformed"
        ) from exc
    if (
        type(prior) is not dict
        or set(prior) != _STATUS_FIELDS
        or prior.get("schema_version") != 1
        or prior.get("owner_binding") != owner_binding
        or not isinstance(prior.get("state"), str)
        or not prior.get("state")
        or (prior.get("trigger") is not None and not isinstance(prior.get("trigger"), str))
        or type(prior.get("entry_eligible")) is not bool
        or type(prior.get("quarantined")) is not bool
        or prior.get("entry_eligible") is prior.get("quarantined")
        or any(
            value is not None and not isinstance(value, str)
            for value in (
                prior.get("completed_at"),
                prior.get("eligible_until"),
                prior.get("run_id"),
                prior.get("snapshot_id"),
            )
        )
    ):
        raise RuntimeReconciliationIntegrationError(
            "existing reconciliation status artifact belongs to another owner"
        )


def _write_status(
    path: Path,
    payload: dict[str, object],
    *,
    owner_binding: str | None = None,
) -> None:
    binding = owner_binding if owner_binding is not None else payload.get("owner_binding")
    if not _valid_status_owner_binding(binding):
        raise RuntimeReconciliationIntegrationError(
            "reconciliation status owner binding is malformed"
        )
    if set(payload) != _STATUS_FIELDS or payload.get("owner_binding") != binding:
        raise RuntimeReconciliationIntegrationError("reconciliation status payload is malformed")
    parent = path.parent
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    metadata = os.fstat(parent_descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        os.close(parent_descriptor)
        raise RuntimeReconciliationIntegrationError("status directory is unavailable")
    temporary_name = f".{path.name}.stage-{secrets.token_hex(16)}"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    descriptor = -1
    prior_descriptor = -1
    prior_identity: tuple[int, int] | None = None
    try:
        try:
            prior_descriptor = os.open(
                path.name,
                os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            pass
        else:
            _read_owned_status_descriptor(prior_descriptor, binding)
            prior_metadata = os.fstat(prior_descriptor)
            prior_identity = (prior_metadata.st_dev, prior_metadata.st_ino)
            os.lseek(prior_descriptor, 0, os.SEEK_SET)
            os.ftruncate(prior_descriptor, 0)
            offset = 0
            while offset < len(encoded):
                written = os.write(prior_descriptor, encoded[offset:])
                if written <= 0:
                    raise RuntimeReconciliationIntegrationError(
                        "reconciliation status write did not complete"
                    )
                offset += written
            os.fsync(prior_descriptor)
            current = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (current.st_dev, current.st_ino) != prior_identity:
                raise RuntimeReconciliationIntegrationError(
                    "existing reconciliation status artifact changed during publication"
                )
            os.fsync(parent_descriptor)
            return
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_descriptor,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise RuntimeReconciliationIntegrationError(
                    "reconciliation status write did not complete"
                )
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if prior_descriptor >= 0:
            os.close(prior_descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent_descriptor)


def read_runtime_reconciliation_status(path: Path) -> dict[str, object]:
    """Read only the fixed, sanitized status surface; uncertainty quarantines."""

    unavailable = {
        "state": "unavailable",
        "trigger": None,
        "completed_at": None,
        "eligible_until": None,
        "entry_eligible": False,
        "quarantined": True,
        "age_seconds": None,
    }
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > 16 * 1024
        ):
            return unavailable
        encoded = os.read(descriptor, 16 * 1024 + 1)
        if len(encoded) != metadata.st_size:
            return unavailable
        payload = json.loads(encoded.decode("ascii"))
        if type(payload) is not dict or set(payload) != _STATUS_FIELDS:
            return unavailable
        if payload["schema_version"] != 1 or not _valid_status_owner_binding(
            payload["owner_binding"]
        ):
            return unavailable
        entry_eligible = payload["entry_eligible"]
        quarantined = payload["quarantined"]
        if type(entry_eligible) is not bool or type(quarantined) is not bool:
            return unavailable
        if entry_eligible is quarantined:
            return unavailable
        completed_at = payload["completed_at"]
        eligible_until = payload["eligible_until"]
        age_seconds = None
        now = datetime.now(timezone.utc)
        if isinstance(completed_at, str):
            observed = datetime.fromisoformat(completed_at.replace("Z", "+00:00"))
            if observed.tzinfo is None:
                return unavailable
            observed = observed.astimezone(timezone.utc)
            if observed > now:
                return unavailable
            age_seconds = max(
                0.0,
                (now - observed).total_seconds(),
            )
        elif completed_at is not None:
            return unavailable
        if isinstance(eligible_until, str):
            expires = datetime.fromisoformat(eligible_until.replace("Z", "+00:00"))
            if expires.tzinfo is None:
                return unavailable
            expires = expires.astimezone(timezone.utc)
            if not isinstance(completed_at, str) or expires < observed:
                return unavailable
        elif eligible_until is not None:
            return unavailable
        if entry_eligible and (
            not isinstance(completed_at, str)
            or not isinstance(eligible_until, str)
            or now > expires
        ):
            return unavailable
        return {
            "state": payload["state"],
            "trigger": payload["trigger"],
            "completed_at": completed_at,
            "eligible_until": eligible_until,
            "entry_eligible": entry_eligible,
            "quarantined": quarantined,
            "age_seconds": None if age_seconds is None else round(age_seconds, 3),
        }
    except Exception:
        return unavailable
    finally:
        if descriptor >= 0:
            os.close(descriptor)


async def build_runtime_reconciliation_controller(
    runtime_context: RuntimeSafetyContext,
) -> tuple[RuntimeReconciliationController, IBKRDiagnosticSnapshotProvider]:
    """Build the production source only after exact-state readiness is proven."""

    context = assert_validated_runtime_safety_context(runtime_context)
    runtime = context.runtime_contract
    if type(runtime) is not RuntimeContract:
        raise RuntimeReconciliationIntegrationError("exact runtime contract is required")
    environment = os.environ
    capability_directory = _absolute_lexical_path(
        environment.get(_CAPABILITY_DIRECTORY_ENV),
        "signing capability directory",
    )
    evidence_root = _absolute_lexical_path(
        environment.get(_EVIDENCE_ROOT_ENV),
        "reconciliation evidence root",
    )
    _require_private_directory(capability_directory, "signing capability directory")
    _require_private_directory(evidence_root, "reconciliation evidence root")
    status_path = runtime_reconciliation_status_path(runtime, environment)
    _assert_status_path_is_unprotected(
        status_path,
        runtime_contract=runtime,
        capability_directory=capability_directory,
        evidence_root=evidence_root,
    )
    await assert_runtime_bootstrap_ready(context)
    provider = await build_diagnostic_provider(context)
    try:
        source = ProductionRuntimeEvidenceSource(
            runtime_context=context,
            provider=provider,
            capability_directory=capability_directory,
            evidence_root=evidence_root,
        )
        service = ReconciliationService(
            evidence_source=source,
            persistence=ReconciliationPersistence(Path(runtime.database_path)),
            expected_account_scope=runtime.safety_account_scope,
        )
        controller = RuntimeReconciliationController(
            service,
            status_path=status_path,
            status_owner_binding=_status_owner_binding(runtime, status_path),
        )
        return controller, provider
    except BaseException:
        cleanup_cancelled = await await_cleanup_required(provider.close())
        if cleanup_cancelled:
            raise asyncio.CancelledError
        raise

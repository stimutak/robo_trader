"""Producer-coupled PR5 reconciliation evidence for exact-state bootstrap.

This module is deliberately not a JSON/artifact loader and has no signing
surface.  A trusted local collector supplies one typed, immutable ledger
observation; this stage binds it to the validated runtime and one complete
``NormalizedBrokerSnapshot`` before delivering an unsigned typed result to the
single narrow receiver capability.

IBKR remains diagnostic in the current paper-simulator execution domain.  Its
positions are therefore required to be empty, rather than compared for
equality with valid local simulator positions.
"""

from __future__ import annotations

import hashlib
import ipaddress
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generic, Iterable, Protocol, TypeVar

from robo_trader.config import PAPER_ONLY_EXECUTION_SOURCE, RuntimeContract
from robo_trader.runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE

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
from .ledger import validate_portfolio_ids
from .policy import (
    ExpectedTimingLagProof,
    ReconciliationCoverage,
    ReconciliationDifference,
    ReconciliationStatus,
    evaluate_paper_simulator_reconciliation,
)

BOOTSTRAP_RECONCILIATION_STATUS = "BOOTSTRAP_EVIDENCE_COMPLETE"

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_FINGERPRINT = re.compile(r"^[0-9a-f]{16,64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_BROKER_SNAPSHOT_ID = re.compile(r"^broker-reconciliation-v1-[0-9a-f]{64}$")
_BROKER_VERDICT_ID = re.compile(r"^reconciliation-verdict-v1-[0-9a-f]{64}$")
_COLLECTION_EVIDENCE_ID = re.compile(r"^broker-collection-v1-[0-9a-f]{64}$")


class BootstrapReconciliationBlocked(ReconciliationDomainError):
    """No bootstrap reconciliation result may be emitted."""


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


def _canonical_portfolios(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    original = tuple(values)
    try:
        normalized = validate_portfolio_ids(original)
    except Exception as exc:
        raise BootstrapReconciliationBlocked(f"{field_name} is malformed") from exc
    if original != normalized:
        raise BootstrapReconciliationBlocked(f"{field_name} must be unique and sorted")
    return normalized


@dataclass(frozen=True, slots=True)
class BootstrapLedgerEvidence:
    """Immutable output expected from the future trusted local-ledger collector.

    ``portfolio_ids`` is the intended bootstrap set, ``known_portfolio_ids`` is
    every portfolio present in the ledger, and ``covered_portfolio_ids`` is the
    set actually examined across positions, orders, executions, and cash.  A
    bootstrap result is possible only when all three sets are exactly equal.
    """

    runtime_fingerprint: str
    execution_domain_scope: ExecutionDomainScope
    account_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    database_size: int
    database_mtime_ns: int
    database_ctime_ns: int
    portfolio_ids: tuple[str, ...]
    known_portfolio_ids: tuple[str, ...]
    active_portfolio_ids: tuple[str, ...]
    covered_portfolio_ids: tuple[str, ...]
    local_simulator_positions_count: int
    legacy_snapshot_hash: str
    observed_at: datetime
    coverage: ReconciliationCoverage
    differences: tuple[ReconciliationDifference, ...] = ()
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...] = ()
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "bootstrap ledger evidence")
        if self.execution_domain_scope is not ExecutionDomainScope.PAPER_SIMULATOR:
            raise BootstrapReconciliationBlocked(
                "bootstrap ledger evidence is not in the paper-simulator domain"
            )
        if not isinstance(self.runtime_fingerprint, str) or not _RUNTIME_FINGERPRINT.fullmatch(
            self.runtime_fingerprint
        ):
            raise BootstrapReconciliationBlocked("runtime_fingerprint is malformed")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise BootstrapReconciliationBlocked("database_path must be absolute and lexical")
        object.__setattr__(
            self,
            "database_identity",
            _safe_id(self.database_identity, "database_identity"),
        )
        for field_name in (
            "database_device",
            "database_inode",
            "database_size",
            "database_mtime_ns",
            "database_ctime_ns",
            "local_simulator_positions_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _exact_nonnegative_int(getattr(self, field_name), field_name),
            )
        if self.database_inode == 0:
            raise BootstrapReconciliationBlocked("database_inode must identify a real file")
        for field_name in (
            "portfolio_ids",
            "known_portfolio_ids",
            "active_portfolio_ids",
            "covered_portfolio_ids",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_portfolios(getattr(self, field_name), field_name),
            )
        if not set(self.active_portfolio_ids).issubset(self.known_portfolio_ids):
            raise BootstrapReconciliationBlocked(
                "active portfolio evidence is outside the known ledger portfolio set"
            )
        object.__setattr__(
            self,
            "legacy_snapshot_hash",
            _hash(self.legacy_snapshot_hash, "legacy_snapshot_hash"),
        )
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "observed_at"))
        if type(self.coverage) is not ReconciliationCoverage:
            raise BootstrapReconciliationBlocked("ledger comparison coverage is malformed")
        differences = tuple(self.differences)
        if any(type(value) is not ReconciliationDifference for value in differences):
            raise BootstrapReconciliationBlocked("ledger differences are not normalized")
        if len({value.identity for value in differences}) != len(differences):
            raise BootstrapReconciliationBlocked("ledger differences contain duplicates")
        object.__setattr__(self, "differences", differences)
        proofs = tuple(self.timing_lag_proofs)
        if any(type(value) is not ExpectedTimingLagProof for value in proofs):
            raise BootstrapReconciliationBlocked("timing-lag proofs are not normalized")
        if len({value.binding_key for value in proofs}) != len(proofs):
            raise BootstrapReconciliationBlocked("timing-lag proofs contain duplicates")
        object.__setattr__(self, "timing_lag_proofs", proofs)


@dataclass(frozen=True, slots=True)
class UnsignedBootstrapReconciliation:
    """Canonical unsigned result; it never grants startup or mutation authority."""

    generated_at: datetime
    runtime_fingerprint: str
    account_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    portfolio_ids: tuple[str, ...]
    legacy_snapshot_hash: str
    broker_snapshot_id: str
    broker_snapshot_hash: str
    broker_collection_evidence_ids: tuple[str, ...]
    broker_verdict_id: str
    broker_verdict_hash: str
    comparison_coverage: ReconciliationCoverage
    reconciliation_status: ReconciliationStatus
    local_simulator_positions_count: int
    broker_positions_count: int
    broker_open_orders_count: int
    managed_account_count: int = 1
    status: str = BOOTSTRAP_RECONCILIATION_STATUS
    schema_version: int = DOMAIN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "unsigned bootstrap reconciliation")
        object.__setattr__(self, "generated_at", _timestamp(self.generated_at, "generated_at"))
        if not isinstance(self.runtime_fingerprint, str) or not _RUNTIME_FINGERPRINT.fullmatch(
            self.runtime_fingerprint
        ):
            raise BootstrapReconciliationBlocked("runtime_fingerprint is malformed")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise BootstrapReconciliationBlocked("database_path must be absolute and lexical")
        object.__setattr__(
            self,
            "database_identity",
            _safe_id(self.database_identity, "database_identity"),
        )
        for field_name in (
            "database_device",
            "database_inode",
            "local_simulator_positions_count",
            "broker_positions_count",
            "broker_open_orders_count",
            "managed_account_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _exact_nonnegative_int(getattr(self, field_name), field_name),
            )
        if (
            self.database_inode == 0
            or self.broker_positions_count != 0
            or self.broker_open_orders_count != 0
            or self.managed_account_count != 1
        ):
            raise BootstrapReconciliationBlocked(
                "unsigned bootstrap result does not prove one zero-exposure paper account"
            )
        object.__setattr__(
            self,
            "portfolio_ids",
            _canonical_portfolios(self.portfolio_ids, "portfolio_ids"),
        )
        for field_name in (
            "legacy_snapshot_hash",
            "broker_snapshot_hash",
            "broker_verdict_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _hash(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "broker_snapshot_id",
            _exact_pattern(
                self.broker_snapshot_id,
                "broker_snapshot_id",
                _BROKER_SNAPSHOT_ID,
            ),
        )
        object.__setattr__(
            self,
            "broker_verdict_id",
            _exact_pattern(
                self.broker_verdict_id,
                "broker_verdict_id",
                _BROKER_VERDICT_ID,
            ),
        )
        evidence_ids = tuple(self.broker_collection_evidence_ids)
        if (
            len(evidence_ids) != len(BrokerCollectionKind)
            or len(set(evidence_ids)) != len(evidence_ids)
            or tuple(sorted(evidence_ids)) != evidence_ids
            or any(
                _exact_pattern(
                    value,
                    "broker collection evidence_id",
                    _COLLECTION_EVIDENCE_ID,
                )
                != value
                for value in evidence_ids
            )
        ):
            raise BootstrapReconciliationBlocked(
                "broker collection evidence identities are incomplete or noncanonical"
            )
        object.__setattr__(self, "broker_collection_evidence_ids", evidence_ids)
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
            "broker_open_orders_count": self.broker_open_orders_count,
            "broker_positions_count": self.broker_positions_count,
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


ReceiverResult = TypeVar("ReceiverResult", covariant=True)


class BootstrapReconciliationReceiver(Protocol, Generic[ReceiverResult]):
    """Only capability invoked after a complete result has been produced.

    Core can later bind this to its producer-specific trust handoff.  It is not
    a generic artifact writer or signer and receives no key or caller-supplied
    path.
    """

    def receive_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> ReceiverResult:
        """Consume one validated unsigned result."""


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
    return runtime


def _database_metadata(path: Path) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BootstrapReconciliationBlocked("runtime database cannot be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BootstrapReconciliationBlocked(
            "runtime database must be a single-link non-symlink regular file"
        )
    return metadata


def _assert_database_binding(
    evidence: BootstrapLedgerEvidence,
    runtime_path: Path,
) -> None:
    metadata = _database_metadata(runtime_path)
    expected = (
        evidence.database_device,
        evidence.database_inode,
        evidence.database_size,
        evidence.database_mtime_ns,
        evidence.database_ctime_ns,
    )
    observed = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if observed != expected:
        raise BootstrapReconciliationBlocked(
            "runtime database changed after immutable ledger collection"
        )


def produce_bootstrap_reconciliation(
    snapshot: NormalizedBrokerSnapshot,
    ledger_evidence: BootstrapLedgerEvidence,
    runtime_contract: RuntimeContract,
    receiver: BootstrapReconciliationReceiver[ReceiverResult],
    *,
    now: datetime,
    max_age_seconds: float = 30.0,
) -> ReceiverResult:
    """Deliver one unsigned bootstrap result, or fail before receiver invocation."""

    if type(snapshot) is not NormalizedBrokerSnapshot:
        raise BootstrapReconciliationBlocked("producer requires a normalized broker snapshot")
    if type(ledger_evidence) is not BootstrapLedgerEvidence:
        raise BootstrapReconciliationBlocked("producer requires typed immutable ledger evidence")
    runtime = _validate_runtime(runtime_contract)
    checked_at = _timestamp(now, "producer clock")
    if (
        isinstance(max_age_seconds, bool)
        or not isinstance(max_age_seconds, (int, float))
        or not math.isfinite(float(max_age_seconds))
        or max_age_seconds <= 0
    ):
        raise BootstrapReconciliationBlocked("freshness bound must be finite and positive")

    runtime_path = Path(runtime.database_path)
    if not runtime_path.is_absolute() or str(runtime_path) != runtime.database_path:
        raise BootstrapReconciliationBlocked("runtime database path must be absolute and lexical")
    runtime_scope = runtime.safety_account_scope
    if (
        ledger_evidence.runtime_fingerprint != runtime.fingerprint
        or ledger_evidence.execution_domain_scope is not ExecutionDomainScope.PAPER_SIMULATOR
        or ledger_evidence.account_scope != runtime_scope
        or ledger_evidence.database_path != str(runtime_path)
        or ledger_evidence.database_identity != runtime.database_identity
    ):
        raise BootstrapReconciliationBlocked(
            "immutable ledger evidence is not bound to the validated runtime"
        )
    if (
        snapshot.account.account_scope != runtime_scope
        or snapshot.account.account_alias != runtime.account_alias
    ):
        raise BootstrapReconciliationBlocked(
            "broker snapshot account identity does not match the validated runtime"
        )
    if (
        ledger_evidence.portfolio_ids != ledger_evidence.known_portfolio_ids
        or ledger_evidence.covered_portfolio_ids != ledger_evidence.known_portfolio_ids
        or not set(ledger_evidence.active_portfolio_ids).issubset(
            ledger_evidence.covered_portfolio_ids
        )
    ):
        raise BootstrapReconciliationBlocked("local ledger portfolio coverage is incomplete")
    if not ledger_evidence.coverage.complete:
        raise BootstrapReconciliationBlocked("local ledger comparison coverage is incomplete")
    local_age = checked_at - ledger_evidence.observed_at
    if local_age < timedelta(0) or local_age > timedelta(seconds=float(max_age_seconds)):
        raise BootstrapReconciliationBlocked("immutable local ledger evidence is future or stale")

    _assert_database_binding(ledger_evidence, runtime_path)
    verdict = evaluate_paper_simulator_reconciliation(
        snapshot,
        ledger_evidence.coverage,
        ledger_evidence.differences,
        ledger_evidence.timing_lag_proofs,
        expected_account_scope=runtime_scope,
        now=checked_at,
        max_age_seconds=max_age_seconds,
    )
    if verdict.quarantine_required:
        raise BootstrapReconciliationBlocked(
            "reconciliation contains stale, incomplete, unknown, or material differences"
        )
    if snapshot.positions or snapshot.open_orders:
        # The evaluator already quarantines these.  Keep this explicit invariant
        # adjacent to result construction so future policy changes cannot turn
        # IBKR exposure into simulator-ledger equality evidence.
        raise BootstrapReconciliationBlocked(
            "IBKR diagnostic account does not have zero exposure and open orders"
        )
    evidence_by_kind = {item.collection: item for item in snapshot.collection_evidence}
    if not snapshot.completeness.complete or set(evidence_by_kind) != set(BrokerCollectionKind):
        raise BootstrapReconciliationBlocked("broker collection evidence is incomplete")

    broker_payload = snapshot.canonical_payload().encode("utf-8")
    verdict_payload = verdict.canonical_payload().encode("utf-8")
    result = UnsignedBootstrapReconciliation(
        generated_at=checked_at,
        runtime_fingerprint=runtime.fingerprint,
        account_scope=runtime_scope,
        database_path=str(runtime_path),
        database_identity=runtime.database_identity,
        database_device=ledger_evidence.database_device,
        database_inode=ledger_evidence.database_inode,
        portfolio_ids=ledger_evidence.covered_portfolio_ids,
        legacy_snapshot_hash=ledger_evidence.legacy_snapshot_hash,
        broker_snapshot_id=snapshot.snapshot_id,
        broker_snapshot_hash=hashlib.sha256(broker_payload).hexdigest(),
        broker_collection_evidence_ids=tuple(
            sorted(item.evidence_id for item in snapshot.collection_evidence)
        ),
        broker_verdict_id=verdict.verdict_id,
        broker_verdict_hash=hashlib.sha256(verdict_payload).hexdigest(),
        comparison_coverage=ledger_evidence.coverage,
        reconciliation_status=verdict.status,
        local_simulator_positions_count=ledger_evidence.local_simulator_positions_count,
        broker_positions_count=len(snapshot.positions),
        broker_open_orders_count=len(snapshot.open_orders),
    )
    _assert_database_binding(ledger_evidence, runtime_path)
    capability = getattr(receiver, "receive_unsigned_bootstrap_reconciliation", None)
    if not callable(capability):
        raise BootstrapReconciliationBlocked(
            "bootstrap reconciliation receiver capability is unavailable"
        )
    return capability(result)

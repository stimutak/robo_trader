"""Exact one-shot binding for runtime reconciliation evidence."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import SupportsIndex

from robo_trader.bootstrap_evidence_receivers import (
    ProtectiveMarkBundleIdentity,
    VerifiedBrokerEvidenceEnvelope,
    assert_and_consume_verified_broker_evidence,
    assert_protective_mark_receiver_capability,
)
from robo_trader.config import RuntimeContract
from robo_trader.financial_state_bootstrap import (
    ExactStateBootstrapEvidence,
    assert_exact_state_runtime_sources_unchanged,
    assert_verified_exact_state_reconciliation_evidence,
)
from robo_trader.safety.sqlite_identity import SQLitePathBinding

from .domain import (
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
    _timestamp,
    canonical_json,
    canonical_timestamp,
)
from .identity import RuntimeSafetyContext, assert_validated_runtime_safety_context
from .policy import (
    ExpectedTimingLagProof,
    ReconciliationCoverage,
    ReconciliationDifference,
)

_CAPABILITY_MARKER = object()
_CAPABILITY_KEY = secrets.token_bytes(32)
_CAPABILITY_LOCK = threading.Lock()
_CAPABILITIES: dict[
    int,
    tuple[
        weakref.ReferenceType["VerifiedRuntimeReconciliationEvidence"],
        str,
        ExactStateBootstrapEvidence,
        str,
        bool,
    ],
] = {}
_COMPARISON_CONSUMPTION_LOCK = threading.Lock()
_MAX_CONSUMED_COMPARISON_LINEAGES = 1024
_CONSUMED_COMPARISON_LINEAGES: dict[tuple[str, str, str], datetime] = {}
_COMPARISON_LAST_CLOCK: datetime | None = None


def _comparison_clock() -> datetime:
    return datetime.now(timezone.utc)


class RuntimeReconciliationEvidenceError(ReconciliationDomainError):
    """Core evidence is forged, cross-bound, stale, or attached to another database."""


def _hash(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeReconciliationEvidenceError(f"{field_name} is malformed")
    return value


def _strict_text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise RuntimeReconciliationEvidenceError(f"{field_name} is malformed")
    return value


@dataclass(frozen=True, repr=False)
class VerifiedRuntimeReconciliationEvidence:
    """One exact core-authenticated broker generation and runtime ledger binding."""

    # Python 3.10 predates dataclass(weakref_slot=True). Explicit slots keep the
    # capability immutable, dictionary-free, and weak-referenceable throughout
    # the supported 3.10+ interpreter range.
    __slots__ = (
        "snapshot",
        "snapshot_id",
        "snapshot_hash",
        "comparison_coverage",
        "differences",
        "timing_lag_proofs",
        "bundle_id",
        "runtime_fingerprint",
        "account_scope",
        "account_alias",
        "database_path",
        "database_identity",
        "database_device",
        "database_inode",
        "broker_artifact_hash",
        "broker_receipt_id",
        "broker_public_key_fingerprint",
        "broker_evidence_expires_at",
        "reconciliation_snapshot_id",
        "reconciliation_artifact_path",
        "reconciliation_artifact_hash",
        "reconciliation_receipt_id",
        "reconciliation_public_key_fingerprint",
        "reconciliation_signature_ed25519",
        "reconciliation_evidence_issued_at",
        "reconciliation_evidence_expires_at",
        "issued_at",
        "expires_at",
        "_runtime_context",
        "_exact_state_evidence",
        "_marker",
        "__weakref__",
    )

    snapshot: NormalizedBrokerSnapshot
    snapshot_id: str
    snapshot_hash: str
    comparison_coverage: ReconciliationCoverage
    differences: tuple[ReconciliationDifference, ...]
    timing_lag_proofs: tuple[ExpectedTimingLagProof, ...]
    bundle_id: str
    runtime_fingerprint: str
    account_scope: str
    account_alias: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    broker_artifact_hash: str
    broker_receipt_id: str
    broker_public_key_fingerprint: str
    broker_evidence_expires_at: datetime
    reconciliation_snapshot_id: str
    reconciliation_artifact_path: str
    reconciliation_artifact_hash: str
    reconciliation_receipt_id: str
    reconciliation_public_key_fingerprint: str
    reconciliation_signature_ed25519: str
    reconciliation_evidence_issued_at: datetime
    reconciliation_evidence_expires_at: datetime
    issued_at: datetime
    expires_at: datetime
    _runtime_context: RuntimeSafetyContext
    _exact_state_evidence: ExactStateBootstrapEvidence
    _marker: object

    def __post_init__(self) -> None:
        if self._marker is not _CAPABILITY_MARKER:
            raise RuntimeReconciliationEvidenceError(
                "runtime reconciliation evidence is factory-only"
            )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_alias": self.account_alias,
            "account_scope": self.account_scope,
            "broker_artifact_hash": self.broker_artifact_hash,
            "broker_public_key_fingerprint": self.broker_public_key_fingerprint,
            "broker_receipt_id": self.broker_receipt_id,
            "broker_evidence_expires_at": canonical_timestamp(self.broker_evidence_expires_at),
            "bundle_id": self.bundle_id,
            "comparison_coverage": self.comparison_coverage.canonical_dict(),
            "database_device": self.database_device,
            "database_identity": self.database_identity,
            "database_inode": self.database_inode,
            "database_path": self.database_path,
            "expires_at": canonical_timestamp(self.expires_at),
            "issued_at": canonical_timestamp(self.issued_at),
            "differences": [value.canonical_dict() for value in self.differences],
            "timing_lag_proofs": [value.canonical_dict() for value in self.timing_lag_proofs],
            "reconciliation_artifact_hash": self.reconciliation_artifact_hash,
            "reconciliation_artifact_path": self.reconciliation_artifact_path,
            "reconciliation_evidence_expires_at": canonical_timestamp(
                self.reconciliation_evidence_expires_at
            ),
            "reconciliation_evidence_issued_at": canonical_timestamp(
                self.reconciliation_evidence_issued_at
            ),
            "reconciliation_public_key_fingerprint": (self.reconciliation_public_key_fingerprint),
            "reconciliation_receipt_id": self.reconciliation_receipt_id,
            "reconciliation_signature_ed25519": self.reconciliation_signature_ed25519,
            "reconciliation_snapshot_id": self.reconciliation_snapshot_id,
            "runtime_fingerprint": self.runtime_fingerprint,
            "snapshot_hash": self.snapshot_hash,
            "snapshot_id": self.snapshot_id,
        }

    def __copy__(self) -> "VerifiedRuntimeReconciliationEvidence":
        raise TypeError("runtime reconciliation evidence cannot be copied")

    def __deepcopy__(self, memo: object) -> "VerifiedRuntimeReconciliationEvidence":
        del memo
        raise TypeError("runtime reconciliation evidence cannot be copied")

    def __reduce__(self) -> str:
        raise TypeError("runtime reconciliation evidence cannot be pickled")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str:
        del protocol
        raise TypeError("runtime reconciliation evidence cannot be pickled")


def _exact_state_source_binding(
    evidence: VerifiedRuntimeReconciliationEvidence,
) -> tuple[ExactStateBootstrapEvidence, str]:
    source = evidence._exact_state_evidence
    if type(source) is not ExactStateBootstrapEvidence:
        raise RuntimeReconciliationEvidenceError(
            "runtime reconciliation exact-state source is missing or substituted"
        )
    try:
        producer_digest = _hash(source._producer_digest, "exact-state producer digest")
    except (AttributeError, RuntimeReconciliationEvidenceError) as exc:
        raise RuntimeReconciliationEvidenceError(
            "runtime reconciliation exact-state source is not producer-sealed"
        ) from exc
    return source, producer_digest


def _digest(evidence: VerifiedRuntimeReconciliationEvidence) -> str:
    payload = canonical_json(evidence.canonical_dict())
    source, source_digest = _exact_state_source_binding(evidence)
    source_binding = f"{id(source):x}:{source_digest}".encode("ascii")
    return hmac.new(
        _CAPABILITY_KEY,
        payload.encode("utf-8")
        + b"\0"
        + evidence.snapshot.canonical_payload().encode("utf-8")
        + b"\0"
        + source_binding,
        hashlib.sha256,
    ).hexdigest()


def _register(
    evidence: VerifiedRuntimeReconciliationEvidence,
) -> VerifiedRuntimeReconciliationEvidence:
    object_id = id(evidence)
    source, source_digest = _exact_state_source_binding(evidence)

    def discard(reference: weakref.ReferenceType[VerifiedRuntimeReconciliationEvidence]) -> None:
        with _CAPABILITY_LOCK:
            current = _CAPABILITIES.get(object_id)
            if current is not None and current[0] is reference:
                _CAPABILITIES.pop(object_id, None)

    reference = weakref.ref(evidence, discard)
    with _CAPABILITY_LOCK:
        if object_id in _CAPABILITIES:
            raise RuntimeReconciliationEvidenceError("runtime evidence registration collided")
        _CAPABILITIES[object_id] = (
            reference,
            _digest(evidence),
            source,
            source_digest,
            False,
        )
    return evidence


def _database_binding(runtime_contract: RuntimeContract) -> SQLitePathBinding:
    path = Path(runtime_contract.database_path)
    if not path.is_absolute() or str(path) != runtime_contract.database_path:
        raise RuntimeReconciliationEvidenceError("runtime database path is not absolute lexical")
    try:
        return SQLitePathBinding.open_readonly(path)
    except Exception as exc:
        raise RuntimeReconciliationEvidenceError(
            "runtime database identity cannot be bound"
        ) from exc


def _assert_consumed_source_binding(
    evidence: VerifiedRuntimeReconciliationEvidence,
    *,
    changed_message: str,
) -> ExactStateBootstrapEvidence:
    try:
        source, source_digest = _exact_state_source_binding(evidence)
        current_digest = _digest(evidence)
    except RuntimeReconciliationEvidenceError as exc:
        raise RuntimeReconciliationEvidenceError(changed_message) from exc
    with _CAPABILITY_LOCK:
        registered = _CAPABILITIES.get(id(evidence))
    if (
        registered is None
        or registered[0]() is not evidence
        or evidence._marker is not _CAPABILITY_MARKER
        or not registered[4]
        or not hmac.compare_digest(registered[1], current_digest)
        or registered[2] is not source
        or not hmac.compare_digest(registered[3], source_digest)
    ):
        raise RuntimeReconciliationEvidenceError(changed_message)
    return source


def _consume_comparison_lineage(
    lineage: tuple[str, str, str],
    *,
    expires_at: datetime,
) -> None:
    global _COMPARISON_LAST_CLOCK

    checked_at = _timestamp(_comparison_clock(), "comparison replay clock")
    expiry = _timestamp(expires_at, "comparison replay expires_at")
    with _COMPARISON_CONSUMPTION_LOCK:
        if _COMPARISON_LAST_CLOCK is not None and checked_at < _COMPARISON_LAST_CLOCK:
            raise RuntimeReconciliationEvidenceError("comparison replay clock moved backwards")
        _COMPARISON_LAST_CLOCK = checked_at
        expired = tuple(
            key
            for key, cached_expiry in _CONSUMED_COMPARISON_LINEAGES.items()
            if cached_expiry < checked_at
        )
        for key in expired:
            _CONSUMED_COMPARISON_LINEAGES.pop(key, None)
        if expiry < checked_at:
            raise RuntimeReconciliationEvidenceError("signed reconciliation comparison expired")
        if lineage in _CONSUMED_COMPARISON_LINEAGES:
            raise RuntimeReconciliationEvidenceError(
                "signed reconciliation comparison was already consumed"
            )
        if len(_CONSUMED_COMPARISON_LINEAGES) >= _MAX_CONSUMED_COMPARISON_LINEAGES:
            raise RuntimeReconciliationEvidenceError(
                "signed reconciliation replay cache is at its safety bound"
            )
        _CONSUMED_COMPARISON_LINEAGES[lineage] = expiry


def bind_verified_runtime_reconciliation_evidence(
    verified_broker_evidence: object,
    verified_exact_state_evidence: object,
    runtime_context: object,
    protective_mark_receiver: object,
) -> VerifiedRuntimeReconciliationEvidence:
    """Consume core broker evidence and bind it to one exact runtime/database bundle."""

    try:
        context = assert_validated_runtime_safety_context(runtime_context)
    except Exception as exc:
        raise RuntimeReconciliationEvidenceError(
            "runtime safety context is not core-validated"
        ) from exc
    contract = context.runtime_contract
    if type(contract) is not RuntimeContract:
        raise RuntimeReconciliationEvidenceError("exact RuntimeContract is required")
    try:
        broker = assert_and_consume_verified_broker_evidence(verified_broker_evidence)
        exact_state = assert_verified_exact_state_reconciliation_evidence(
            verified_exact_state_evidence,
            contract,
        )
        bundle = assert_protective_mark_receiver_capability(
            protective_mark_receiver,
            runtime_contract=contract,
        )
    except Exception as exc:
        raise RuntimeReconciliationEvidenceError(
            "runtime reconciliation evidence is not core-authenticated"
        ) from exc
    if (
        type(broker) is not VerifiedBrokerEvidenceEnvelope
        or type(bundle) is not ProtectiveMarkBundleIdentity
        or type(exact_state) is not ExactStateBootstrapEvidence
    ):
        raise RuntimeReconciliationEvidenceError("core evidence types are not exact")

    reconciliation_receipts = tuple(
        receipt
        for receipt in exact_state.authentication_receipts
        if receipt.artifact_kind == "reconciliation_report"
    )
    if len(reconciliation_receipts) != 1:
        raise RuntimeReconciliationEvidenceError(
            "signed reconciliation receipt lineage is incomplete"
        )
    reconciliation_receipt = reconciliation_receipts[0]
    snapshot = broker.snapshot
    snapshot_hash = hashlib.sha256(snapshot.canonical_payload().encode("utf-8")).hexdigest()
    if (
        broker.snapshot_id != snapshot.snapshot_id
        or not hmac.compare_digest(broker.snapshot_hash, snapshot_hash)
        or broker.runtime_fingerprint != contract.fingerprint
        or broker.account_scope != contract.safety_account_scope
        or broker.account_scope != snapshot.account.account_scope
        or snapshot.account.account_alias != context.account_alias
        or bundle.runtime_fingerprint != contract.fingerprint
        or bundle.account_scope != broker.account_scope
        or bundle.database_identity != contract.database_identity
        or bundle.broker_snapshot_id != broker.snapshot_id
        or not hmac.compare_digest(bundle.broker_snapshot_hash, broker.snapshot_hash)
        or not hmac.compare_digest(bundle.broker_artifact_hash, broker.artifact_hash)
        or bundle.broker_receipt_id != broker.receipt_id
        or not hmac.compare_digest(
            bundle.broker_public_key_fingerprint,
            broker.public_key_fingerprint,
        )
        or type(exact_state.reconciliation_coverage) is not ReconciliationCoverage
        or exact_state.reconciliation_snapshot_id != bundle.reconciliation_snapshot_id
        or exact_state.bundle_id != bundle.bundle_id
        or exact_state.runtime_fingerprint != contract.fingerprint
        or exact_state.account_scope != broker.account_scope
        or exact_state.database_path != contract.database_path
        or exact_state.database_identity != contract.database_identity
        or (exact_state.database_device, exact_state.database_inode)
        != (bundle.database_device, bundle.database_inode)
        or exact_state.broker_snapshot_id != broker.snapshot_id
        or not hmac.compare_digest(exact_state.broker_snapshot_hash, broker.artifact_hash)
        or reconciliation_receipt.artifact_sha256 != exact_state.reconciliation_report_hash
        or reconciliation_receipt.runtime_fingerprint != contract.fingerprint
        or reconciliation_receipt.account_scope != broker.account_scope
    ):
        raise RuntimeReconciliationEvidenceError("core evidence bindings disagree")
    issued_at = _timestamp(broker.issued_at, "broker evidence issued_at")
    broker_expires_at = _timestamp(broker.expires_at, "broker evidence expires_at")
    reconciliation_issued_at = _timestamp(
        reconciliation_receipt.issued_at,
        "reconciliation evidence issued_at",
    )
    reconciliation_expires_at = _timestamp(
        reconciliation_receipt.expires_at,
        "reconciliation evidence expires_at",
    )
    expires_at = min(broker_expires_at, reconciliation_expires_at)
    if (
        not snapshot.retrieved_at <= issued_at <= broker_expires_at
        or not exact_state.reconciliation_generated_at
        <= reconciliation_issued_at
        <= reconciliation_expires_at
    ):
        raise RuntimeReconciliationEvidenceError("broker evidence chronology is invalid")
    binding = _database_binding(contract)
    try:
        if (binding.device, binding.inode) != (
            bundle.database_device,
            bundle.database_inode,
        ):
            raise RuntimeReconciliationEvidenceError("core bundle belongs to a replaced database")
    finally:
        binding.close()

    for value, field_name in (
        (broker.artifact_hash, "broker artifact hash"),
        (broker.public_key_fingerprint, "broker public-key fingerprint"),
        (broker.snapshot_hash, "broker snapshot hash"),
        (exact_state.reconciliation_report_hash, "reconciliation artifact hash"),
        (
            reconciliation_receipt.public_key_fingerprint,
            "reconciliation public-key fingerprint",
        ),
    ):
        _hash(value, field_name)
    for value, field_name in (
        (bundle.bundle_id, "bundle_id"),
        (broker.receipt_id, "broker receipt_id"),
        (contract.database_identity, "database_identity"),
        (contract.fingerprint, "runtime_fingerprint"),
        (exact_state.reconciliation_snapshot_id, "reconciliation_snapshot_id"),
        (reconciliation_receipt.receipt_id, "reconciliation receipt_id"),
        (reconciliation_receipt.signature_ed25519, "reconciliation signature"),
    ):
        _strict_text(value, field_name)
    if type(bundle.database_device) is not int or type(bundle.database_inode) is not int:
        raise RuntimeReconciliationEvidenceError("database identity numbers are malformed")

    _consume_comparison_lineage(
        (
            exact_state.reconciliation_snapshot_id,
            reconciliation_receipt.receipt_id,
            reconciliation_receipt.signature_ed25519,
        ),
        expires_at=reconciliation_expires_at,
    )

    return _register(
        VerifiedRuntimeReconciliationEvidence(
            snapshot=snapshot,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot_hash,
            comparison_coverage=exact_state.reconciliation_coverage,
            differences=exact_state.reconciliation_differences,
            timing_lag_proofs=exact_state.reconciliation_timing_lag_proofs,
            bundle_id=bundle.bundle_id,
            runtime_fingerprint=contract.fingerprint,
            account_scope=broker.account_scope,
            account_alias=snapshot.account.account_alias,
            database_path=contract.database_path,
            database_identity=contract.database_identity,
            database_device=bundle.database_device,
            database_inode=bundle.database_inode,
            broker_artifact_hash=broker.artifact_hash,
            broker_receipt_id=broker.receipt_id,
            broker_public_key_fingerprint=broker.public_key_fingerprint,
            broker_evidence_expires_at=broker_expires_at,
            reconciliation_snapshot_id=exact_state.reconciliation_snapshot_id,
            reconciliation_artifact_path=exact_state.reconciliation_artifact_path,
            reconciliation_artifact_hash=exact_state.reconciliation_report_hash,
            reconciliation_receipt_id=reconciliation_receipt.receipt_id,
            reconciliation_public_key_fingerprint=(reconciliation_receipt.public_key_fingerprint),
            reconciliation_signature_ed25519=reconciliation_receipt.signature_ed25519,
            reconciliation_evidence_issued_at=reconciliation_issued_at,
            reconciliation_evidence_expires_at=reconciliation_expires_at,
            issued_at=issued_at,
            expires_at=expires_at,
            _runtime_context=context,
            _exact_state_evidence=exact_state,
            _marker=_CAPABILITY_MARKER,
        )
    )


def assert_runtime_reconciliation_evidence_sources_current(
    evidence: VerifiedRuntimeReconciliationEvidence,
) -> None:
    """Revalidate one consumed capability's sealed sources and database inode."""

    if type(evidence) is not VerifiedRuntimeReconciliationEvidence:
        raise RuntimeReconciliationEvidenceError(
            "exact verified runtime reconciliation evidence is required"
        )
    source = _assert_consumed_source_binding(
        evidence,
        changed_message="consumed runtime reconciliation binding changed",
    )
    try:
        context = assert_validated_runtime_safety_context(evidence._runtime_context)
    except Exception as exc:
        raise RuntimeReconciliationEvidenceError("runtime context changed after binding") from exc
    contract = context.runtime_contract
    if (
        type(contract) is not RuntimeContract
        or contract.fingerprint != evidence.runtime_fingerprint
        or contract.database_identity != evidence.database_identity
        or contract.database_path != evidence.database_path
        or contract.safety_account_scope != evidence.account_scope
    ):
        raise RuntimeReconciliationEvidenceError("runtime binding changed after production")
    try:
        assert_exact_state_runtime_sources_unchanged(source, contract)
    except Exception as exc:
        raise RuntimeReconciliationEvidenceError(
            "signed reconciliation source state changed before comparison"
        ) from exc
    binding = _database_binding(contract)
    try:
        if (binding.device, binding.inode) != (
            evidence.database_device,
            evidence.database_inode,
        ):
            raise RuntimeReconciliationEvidenceError(
                "runtime database was replaced after evidence production"
            )
    finally:
        binding.close()
    _assert_consumed_source_binding(
        evidence,
        changed_message=("consumed runtime reconciliation binding changed during revalidation"),
    )


def assert_and_consume_verified_runtime_reconciliation_evidence(
    evidence: object,
) -> VerifiedRuntimeReconciliationEvidence:
    """Consume one exact capability and revalidate all sealed runtime sources."""

    if type(evidence) is not VerifiedRuntimeReconciliationEvidence:
        raise RuntimeReconciliationEvidenceError(
            "exact verified runtime reconciliation evidence is required"
        )
    try:
        source, source_digest = _exact_state_source_binding(evidence)
        current_digest = _digest(evidence)
    except RuntimeReconciliationEvidenceError as exc:
        raise RuntimeReconciliationEvidenceError(
            "runtime reconciliation evidence is forged, changed, or already consumed"
        ) from exc
    with _CAPABILITY_LOCK:
        registered = _CAPABILITIES.get(id(evidence))
        if (
            registered is None
            or registered[0]() is not evidence
            or evidence._marker is not _CAPABILITY_MARKER
            or registered[4]
            or not hmac.compare_digest(registered[1], current_digest)
            or registered[2] is not source
            or not hmac.compare_digest(registered[3], source_digest)
        ):
            raise RuntimeReconciliationEvidenceError(
                "runtime reconciliation evidence is forged, changed, or already consumed"
            )
        _CAPABILITIES[id(evidence)] = (
            registered[0],
            registered[1],
            registered[2],
            registered[3],
            True,
        )
    assert_runtime_reconciliation_evidence_sources_current(evidence)
    return evidence


__all__ = [
    "RuntimeReconciliationEvidenceError",
    "VerifiedRuntimeReconciliationEvidence",
    "assert_and_consume_verified_runtime_reconciliation_evidence",
    "assert_runtime_reconciliation_evidence_sources_current",
    "bind_verified_runtime_reconciliation_evidence",
]

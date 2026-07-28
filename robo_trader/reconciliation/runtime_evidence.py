"""Exact one-shot binding for runtime reconciliation evidence."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from robo_trader.bootstrap_evidence_receivers import (
    ProtectiveMarkBundleIdentity,
    VerifiedBrokerEvidenceEnvelope,
    assert_and_consume_verified_broker_evidence,
    assert_protective_mark_receiver_capability,
)
from robo_trader.config import RuntimeContract
from robo_trader.safety.sqlite_identity import SQLitePathBinding

from .domain import (
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
    _timestamp,
    canonical_json,
    canonical_timestamp,
)
from .identity import RuntimeSafetyContext, assert_validated_runtime_safety_context

_CAPABILITY_MARKER = object()
_CAPABILITY_KEY = secrets.token_bytes(32)
_CAPABILITY_LOCK = threading.Lock()
_CAPABILITIES: dict[
    int,
    tuple[weakref.ReferenceType["VerifiedRuntimeReconciliationEvidence"], str],
] = {}


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


@dataclass(frozen=True, slots=True, weakref_slot=True, repr=False)
class VerifiedRuntimeReconciliationEvidence:
    """One exact core-authenticated broker generation and runtime ledger binding."""

    snapshot: NormalizedBrokerSnapshot
    snapshot_id: str
    snapshot_hash: str
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
    issued_at: datetime
    expires_at: datetime
    _runtime_context: RuntimeSafetyContext = field(repr=False, compare=False)
    _marker: object = field(repr=False, compare=False)

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
            "bundle_id": self.bundle_id,
            "database_device": self.database_device,
            "database_identity": self.database_identity,
            "database_inode": self.database_inode,
            "database_path": self.database_path,
            "expires_at": canonical_timestamp(self.expires_at),
            "issued_at": canonical_timestamp(self.issued_at),
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


def _digest(evidence: VerifiedRuntimeReconciliationEvidence) -> str:
    payload = canonical_json(evidence.canonical_dict())
    return hmac.new(
        _CAPABILITY_KEY,
        payload.encode("utf-8") + b"\0" + evidence.snapshot.canonical_payload().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _register(
    evidence: VerifiedRuntimeReconciliationEvidence,
) -> VerifiedRuntimeReconciliationEvidence:
    object_id = id(evidence)

    def discard(reference: weakref.ReferenceType[VerifiedRuntimeReconciliationEvidence]) -> None:
        with _CAPABILITY_LOCK:
            current = _CAPABILITIES.get(object_id)
            if current is not None and current[0] is reference:
                _CAPABILITIES.pop(object_id, None)

    reference = weakref.ref(evidence, discard)
    with _CAPABILITY_LOCK:
        if object_id in _CAPABILITIES:
            raise RuntimeReconciliationEvidenceError("runtime evidence registration collided")
        _CAPABILITIES[object_id] = (reference, _digest(evidence))
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


def bind_verified_runtime_reconciliation_evidence(
    verified_broker_evidence: object,
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
    ):
        raise RuntimeReconciliationEvidenceError("core evidence types are not exact")

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
    ):
        raise RuntimeReconciliationEvidenceError("core evidence bindings disagree")
    issued_at = _timestamp(broker.issued_at, "broker evidence issued_at")
    expires_at = _timestamp(broker.expires_at, "broker evidence expires_at")
    if not snapshot.retrieved_at <= issued_at <= expires_at:
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
    ):
        _hash(value, field_name)
    for value, field_name in (
        (bundle.bundle_id, "bundle_id"),
        (broker.receipt_id, "broker receipt_id"),
        (contract.database_identity, "database_identity"),
        (contract.fingerprint, "runtime_fingerprint"),
    ):
        _strict_text(value, field_name)
    if type(bundle.database_device) is not int or type(bundle.database_inode) is not int:
        raise RuntimeReconciliationEvidenceError("database identity numbers are malformed")

    return _register(
        VerifiedRuntimeReconciliationEvidence(
            snapshot=snapshot,
            snapshot_id=snapshot.snapshot_id,
            snapshot_hash=snapshot_hash,
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
            issued_at=issued_at,
            expires_at=expires_at,
            _runtime_context=context,
            _marker=_CAPABILITY_MARKER,
        )
    )


def assert_and_consume_verified_runtime_reconciliation_evidence(
    evidence: object,
) -> VerifiedRuntimeReconciliationEvidence:
    """Consume one exact capability and revalidate its runtime database inode."""

    if type(evidence) is not VerifiedRuntimeReconciliationEvidence:
        raise RuntimeReconciliationEvidenceError(
            "exact verified runtime reconciliation evidence is required"
        )
    with _CAPABILITY_LOCK:
        registered = _CAPABILITIES.pop(id(evidence), None)
    if (
        registered is None
        or registered[0]() is not evidence
        or evidence._marker is not _CAPABILITY_MARKER
        or not hmac.compare_digest(registered[1], _digest(evidence))
    ):
        raise RuntimeReconciliationEvidenceError(
            "runtime reconciliation evidence is forged, changed, or already consumed"
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
    return evidence


__all__ = [
    "RuntimeReconciliationEvidenceError",
    "VerifiedRuntimeReconciliationEvidence",
    "assert_and_consume_verified_runtime_reconciliation_evidence",
    "bind_verified_runtime_reconciliation_evidence",
]

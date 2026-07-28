"""Opaque, producer-typed signing receivers for bootstrap evidence.

Each receiver owns exactly one externally provisioned Ed25519 capability and
accepts exactly one immutable producer result class.  No receiver accepts an
artifact path, JSON mapping, payload bytes, or caller-selected artifact kind.

Receiver objects never retain a raw private key or expose a generic signing
callable. Keys live only in this trusted-core module's bound authority registry
and are released after the one-shot bundle. This prevents callback code from
extracting keys by walking receiver attributes; it does not claim isolation
from malicious code already executing with arbitrary access to module globals
inside this Python interpreter. Process isolation is the boundary for that
stronger threat model.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, SupportsIndex

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .bootstrap_evidence_auth import (
    AUTH_SCHEMA_VERSION,
    AUTH_SUFFIX,
    MAX_RECEIPT_LIFETIME,
    AuthenticatedEvidenceReceipt,
    bootstrap_evidence_trust_public_dict,
    ed25519_public_key_fingerprint,
    receipt_signature_payload,
    verify_receipt,
)
from .bootstrap_mark_producer import (
    UnsignedBootstrapProtectiveMark,
    assert_producer_owned_unsigned_bootstrap_protective_mark,
)
from .config import RuntimeContract
from .reconciliation.bootstrap_producer import (
    UnsignedBootstrapReconciliation,
    assert_and_consume_producer_owned_bootstrap_reconciliation,
)
from .reconciliation.domain import NormalizedBrokerSnapshot
from .reconciliation.ibkr_adapter import (
    BrokerSnapshotProducerResult,
    assert_producer_owned_broker_snapshot_result,
)

BROKER_PRIVATE_KEY_FILENAME = "broker_snapshot_ed25519_private.pem"
RECONCILIATION_PRIVATE_KEY_FILENAME = "reconciliation_report_ed25519_private.pem"
PROTECTIVE_MARK_PRIVATE_KEY_FILENAME = "protective_mark_ed25519_private.pem"

_BROKER_ARTIFACT_FILENAME = "broker_snapshot.json"
_RECONCILIATION_ARTIFACT_FILENAME = "reconciliation_report.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_FACTORY_TOKEN = object()
_VERIFIED_BROKER_MARKER = object()
_VERIFIED_BROKER_REGISTRY_KEY = secrets.token_bytes(32)
_VERIFIED_BROKER_REGISTRY_LOCK = threading.Lock()
_RECONCILIATION_RECEIVER_REGISTRY_LOCK = threading.Lock()
_RECONCILIATION_RECEIVER_REGISTRY: weakref.WeakValueDictionary[
    int, "ReconciliationEvidenceReceiver"
] = weakref.WeakValueDictionary()
_PROTECTIVE_MARK_RECEIVER_REGISTRY_LOCK = threading.Lock()
_PROTECTIVE_MARK_RECEIVER_REGISTRY: weakref.WeakValueDictionary[
    int, "ProtectiveMarkEvidenceReceiver"
] = weakref.WeakValueDictionary()
_RECONCILIATION_STAGE_MARKER = object()
_RECONCILIATION_STAGE_REGISTRY_LOCK = threading.Lock()
_RECONCILIATION_STAGE_REGISTRY: dict[int, "_StagedReconciliationEvidence"] = {}
_PROTECTIVE_MARK_STAGE_MARKER = object()
_PROTECTIVE_MARK_STAGE_REGISTRY_LOCK = threading.Lock()
_PROTECTIVE_MARK_STAGE_REGISTRY: dict[int, "_StagedProtectiveMarkEvidence"] = {}
_SIGNING_AUTHORITY_MARKER = object()
_SIGNING_AUTHORITY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class _SigningAuthorityHandle:
    nonce: str
    marker: object = field(repr=False, compare=False)


@dataclass(slots=True)
class _SigningAuthority:
    key: Ed25519PrivateKey = field(repr=False)
    artifact_kind: str
    producer_id: str
    runtime_fingerprint: str
    account_scope: str
    bundle_id: str
    signed_object_ids: set[str] = field(default_factory=set, repr=False)


_SIGNING_AUTHORITIES: dict[str, _SigningAuthority] = {}


class BootstrapEvidenceReceiverError(ValueError):
    """A signing capability or typed producer handoff is unsafe."""


@dataclass(frozen=True, slots=True)
class SealedBootstrapEvidenceArtifact:
    artifact_kind: str
    artifact_path: Path
    authentication_receipt_path: Path
    artifact_sha256: str
    producer_object_id: str


@dataclass(frozen=True, slots=True)
class ReconciliationBundleIdentity:
    receiver_type: type[object]
    bundle_id: str
    runtime_fingerprint: str
    account_scope: str
    database_identity: str


@dataclass(frozen=True, slots=True)
class ProtectiveMarkBundleIdentity:
    receiver_type: type[object]
    bundle_id: str
    reconciliation_snapshot_id: str
    broker_snapshot_id: str
    broker_snapshot_hash: str
    broker_artifact_hash: str
    broker_receipt_id: str
    broker_public_key_fingerprint: str
    runtime_fingerprint: str
    account_scope: str
    database_identity: str
    database_device: int
    database_inode: int


@dataclass(frozen=True, repr=False)
class VerifiedBrokerEvidenceEnvelope:
    """One-shot core-owned broker authority for reconciliation only."""

    __slots__ = (
        "snapshot",
        "snapshot_id",
        "snapshot_hash",
        "runtime_fingerprint",
        "account_scope",
        "receipt_id",
        "public_key_fingerprint",
        "artifact_hash",
        "issued_at",
        "expires_at",
        "_marker",
        "__weakref__",
    )

    snapshot: NormalizedBrokerSnapshot
    snapshot_id: str
    snapshot_hash: str
    runtime_fingerprint: str
    account_scope: str
    receipt_id: str
    public_key_fingerprint: str
    artifact_hash: str
    issued_at: datetime
    expires_at: datetime
    _marker: object

    def __post_init__(self) -> None:
        if type(self.snapshot) is not NormalizedBrokerSnapshot:
            raise BootstrapEvidenceReceiverError("verified broker snapshot is not normalized")
        if self._marker is not _VERIFIED_BROKER_MARKER:
            raise BootstrapEvidenceReceiverError("verified broker envelope is factory-only")

    def __copy__(self) -> "VerifiedBrokerEvidenceEnvelope":
        raise TypeError("verified broker envelope cannot be copied")

    def __deepcopy__(self, memo: object) -> "VerifiedBrokerEvidenceEnvelope":
        raise TypeError("verified broker envelope cannot be copied")

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("verified broker envelope cannot be pickled")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        del protocol
        raise TypeError("verified broker envelope cannot be pickled")


_VerifiedBrokerRegistryEntry = tuple[
    weakref.ReferenceType[VerifiedBrokerEvidenceEnvelope],
    str,
]
_VERIFIED_BROKER_REGISTRY: dict[int, _VerifiedBrokerRegistryEntry] = {}


def _verified_broker_digest(envelope: VerifiedBrokerEvidenceEnvelope) -> str:
    payload = {
        "account_scope": envelope.account_scope,
        "artifact_hash": envelope.artifact_hash,
        "expires_at": _utc_text(envelope.expires_at),
        "issued_at": _utc_text(envelope.issued_at),
        "public_key_fingerprint": envelope.public_key_fingerprint,
        "receipt_id": envelope.receipt_id,
        "runtime_fingerprint": envelope.runtime_fingerprint,
        "snapshot_hash": envelope.snapshot_hash,
        "snapshot_id": envelope.snapshot_id,
    }
    return hashlib.sha256(
        _VERIFIED_BROKER_REGISTRY_KEY
        + json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        + envelope.snapshot.canonical_payload().encode("utf-8")
    ).hexdigest()


def _register_verified_broker_envelope(
    envelope: VerifiedBrokerEvidenceEnvelope,
) -> VerifiedBrokerEvidenceEnvelope:
    object_id = id(envelope)

    def discard(reference: weakref.ReferenceType[VerifiedBrokerEvidenceEnvelope]) -> None:
        with _VERIFIED_BROKER_REGISTRY_LOCK:
            current = _VERIFIED_BROKER_REGISTRY.get(object_id)
            if current is not None and current[0] is reference:
                _VERIFIED_BROKER_REGISTRY.pop(object_id, None)

    reference = weakref.ref(envelope, discard)
    with _VERIFIED_BROKER_REGISTRY_LOCK:
        _VERIFIED_BROKER_REGISTRY[object_id] = (
            reference,
            _verified_broker_digest(envelope),
        )
    return envelope


def assert_and_consume_verified_broker_evidence(
    envelope: VerifiedBrokerEvidenceEnvelope,
) -> VerifiedBrokerEvidenceEnvelope:
    """Consume one exact verifier-owned envelope for reconciliation."""

    if type(envelope) is not VerifiedBrokerEvidenceEnvelope:
        raise BootstrapEvidenceReceiverError("exact verified broker envelope is required")
    digest = _verified_broker_digest(envelope)
    with _VERIFIED_BROKER_REGISTRY_LOCK:
        registered = _VERIFIED_BROKER_REGISTRY.pop(id(envelope), None)
        if (
            registered is None
            or registered[0]() is not envelope
            or envelope._marker is not _VERIFIED_BROKER_MARKER
            or not secrets.compare_digest(registered[1], digest)
        ):
            raise BootstrapEvidenceReceiverError(
                "verified broker envelope is forged, changed, or already consumed"
            )
    return envelope


@dataclass(slots=True)
class _BundleBindings:
    bundle_id: str
    runtime_fingerprint: str
    account_scope: str
    database_identity: str
    broker_snapshot_id: str | None = None
    broker_snapshot_hash: str | None = None
    broker_artifact_hash: str | None = None
    broker_artifact: SealedBootstrapEvidenceArtifact | None = None
    broker_receipt_id: str | None = None
    broker_public_key_fingerprint: str | None = None
    reconciliation_snapshot_id: str | None = None
    reconciliation_portfolio_ids: tuple[str, ...] = ()
    database_device: int | None = None
    database_inode: int | None = None
    safety_journal_path: str | None = None
    safety_journal_identity: str | None = None
    safety_journal_device: int | None = None
    safety_journal_inode: int | None = None
    safety_journal_last_sequence: int | None = None
    safety_journal_last_chain_hash: str | None = None
    terminal_settlement_count: int | None = None
    terminal_fill_count: int | None = None
    local_simulator_positions_count: int | None = None
    broker_positions_count: int | None = None
    broker_open_orders_count: int | None = None
    marks: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _StagedReconciliationEvidence:
    receiver: "ReconciliationEvidenceReceiver" = field(repr=False, compare=False)
    artifact: SealedBootstrapEvidenceArtifact
    staged_artifact_path: Path
    staged_receipt_path: Path
    final_artifact_path: Path
    final_receipt_path: Path
    portfolio_ids: tuple[str, ...]
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
    local_simulator_positions_count: int
    broker_positions_count: int
    broker_open_orders_count: int
    marker: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _StagedProtectiveMarkEvidence:
    receiver: "ProtectiveMarkEvidenceReceiver" = field(repr=False, compare=False)
    artifact: SealedBootstrapEvidenceArtifact
    identity: tuple[str, str]
    staged_artifact_path: Path
    staged_receipt_path: Path
    final_artifact_path: Path
    final_receipt_path: Path
    marker: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class BootstrapEvidenceReceiverSet:
    broker_snapshot: "BrokerSnapshotEvidenceReceiver"
    reconciliation_report: "ReconciliationEvidenceReceiver"
    protective_mark: "ProtectiveMarkEvidenceReceiver"
    _state: _BundleBindings = field(repr=False)

    def assert_complete(self, expected_marks: set[tuple[str, str]]) -> None:
        if (
            self._state.broker_snapshot_id is None
            or self._state.reconciliation_snapshot_id is None
            or self._state.marks != expected_marks
        ):
            raise BootstrapEvidenceReceiverError(
                "evidence bundle is incomplete for the reconciled ledger positions"
            )

    def close(self) -> None:
        """Release all private signing capabilities after the one-shot bundle."""

        self.broker_snapshot._release_capability()
        self.reconciliation_report._release_capability()
        self.protective_mark._release_capability()

    @property
    def broker_artifact(self) -> SealedBootstrapEvidenceArtifact:
        artifact = self._state.broker_artifact
        if artifact is None:
            raise BootstrapEvidenceReceiverError("broker artifact is unavailable")
        return artifact


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _register_signing_authority(
    *,
    key: Ed25519PrivateKey,
    artifact_kind: str,
    producer_id: str,
    runtime_fingerprint: str,
    account_scope: str,
    bundle_id: str,
) -> _SigningAuthorityHandle:
    nonce = secrets.token_hex(32)
    handle = _SigningAuthorityHandle(nonce=nonce, marker=_SIGNING_AUTHORITY_MARKER)
    authority = _SigningAuthority(
        key=key,
        artifact_kind=artifact_kind,
        producer_id=producer_id,
        runtime_fingerprint=runtime_fingerprint,
        account_scope=account_scope,
        bundle_id=bundle_id,
    )
    with _SIGNING_AUTHORITY_LOCK:
        if nonce in _SIGNING_AUTHORITIES:  # pragma: no cover - random collision invariant
            raise BootstrapEvidenceReceiverError("signing authority nonce collision")
        _SIGNING_AUTHORITIES[nonce] = authority
    return handle


def _sign_bound_artifact_receipt(
    handle: _SigningAuthorityHandle,
    *,
    artifact_kind: str,
    artifact_sha256: str,
    producer_object_id: str,
    runtime_fingerprint: str,
    account_scope: str,
    bundle_id: str,
    now: datetime,
) -> tuple[dict[str, object], AuthenticatedEvidenceReceipt]:
    """Perform one artifact-bound signing operation inside trusted core."""

    if (
        type(handle) is not _SigningAuthorityHandle
        or handle.marker is not _SIGNING_AUTHORITY_MARKER
    ):
        raise BootstrapEvidenceReceiverError("signing authority handle is invalid")
    with _SIGNING_AUTHORITY_LOCK:
        authority = _SIGNING_AUTHORITIES.get(handle.nonce)
        if (
            authority is None
            or authority.artifact_kind != artifact_kind
            or authority.runtime_fingerprint != runtime_fingerprint
            or authority.account_scope != account_scope
            or authority.bundle_id != bundle_id
            or producer_object_id in authority.signed_object_ids
        ):
            raise BootstrapEvidenceReceiverError(
                "signing authority is absent, replayed, or outside its bundle binding"
            )
        values: dict[str, object] = {
            "schema_version": AUTH_SCHEMA_VERSION,
            "receipt_id": "bevr-v2-" + secrets.token_hex(32),
            "artifact_kind": artifact_kind,
            "producer_id": authority.producer_id,
            "artifact_sha256": artifact_sha256,
            "runtime_fingerprint": runtime_fingerprint,
            "account_scope": account_scope,
            "issued_at": _utc_text(now),
            "expires_at": _utc_text(now + MAX_RECEIPT_LIFETIME),
            "public_key_fingerprint": ed25519_public_key_fingerprint(authority.key.public_key()),
        }
        values["signature_ed25519"] = base64.b64encode(
            authority.key.sign(receipt_signature_payload(values))
        ).decode("ascii")
        authentication = verify_receipt(
            raw=values,
            artifact_kind=artifact_kind,
            artifact_sha256=artifact_sha256,
            runtime_fingerprint=runtime_fingerprint,
            account_scope=account_scope,
            now=now,
        )
        authority.signed_object_ids.add(producer_object_id)
    return values, authentication


def _release_signing_authority(handle: _SigningAuthorityHandle | None) -> None:
    if type(handle) is not _SigningAuthorityHandle:
        return
    with _SIGNING_AUTHORITY_LOCK:
        _SIGNING_AUTHORITIES.pop(handle.nonce, None)


def _safe_private_key(
    capability_directory: Path,
    filename: str,
    artifact_kind: str,
    expected_fingerprint: str,
) -> Ed25519PrivateKey:
    directory = Path(capability_directory)
    try:
        metadata = os.lstat(directory)
        resolved_directory = directory.resolve(strict=True)
    except OSError as exc:
        raise BootstrapEvidenceReceiverError(
            "bootstrap signing capability directory cannot be inspected"
        ) from exc
    if (
        not directory.is_absolute()
        or resolved_directory != directory
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or directory == _PROJECT_ROOT
        or _PROJECT_ROOT in directory.parents
    ):
        raise BootstrapEvidenceReceiverError(
            "bootstrap signing capabilities must be an external owner-only directory"
        )
    key_path = directory / filename
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(key_path, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size > 4096
        ):
            raise BootstrapEvidenceReceiverError(
                f"{artifact_kind} signing capability is not a sealed owner key"
            )
        payload = os.read(descriptor, 4097)
        after = os.fstat(descriptor)
        current = os.lstat(key_path)
        if (
            len(payload) > 4096
            or os.read(descriptor, 1)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
        ):
            raise BootstrapEvidenceReceiverError(
                f"{artifact_kind} signing capability changed while read"
            )
    except OSError as exc:
        raise BootstrapEvidenceReceiverError(
            f"{artifact_kind} signing capability cannot be read safely"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise BootstrapEvidenceReceiverError(
            f"{artifact_kind} signing capability is invalid"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise BootstrapEvidenceReceiverError(f"{artifact_kind} signing capability must be Ed25519")
    if ed25519_public_key_fingerprint(key.public_key()) != expected_fingerprint:
        raise BootstrapEvidenceReceiverError(
            f"{artifact_kind} signing capability does not match the pinned trust root"
        )
    return key


def _prepare_output_directory(path: Path, capability_directory: Path) -> Path:
    output = Path(path)
    if (
        not output.is_absolute()
        or output == capability_directory
        or capability_directory in output.parents
        or output.parent.resolve(strict=True) / output.name != output
    ):
        raise BootstrapEvidenceReceiverError("evidence output directory is unsafe")
    try:
        os.mkdir(output, 0o700)
    except OSError as exc:
        raise BootstrapEvidenceReceiverError(
            "evidence output directory must be new and exclusive"
        ) from exc
    metadata = os.lstat(output)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise BootstrapEvidenceReceiverError("evidence output directory is not owner-only")
    return output


def _write_new_sealed_file(path: Path, payload: bytes) -> None:
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise BootstrapEvidenceReceiverError("bootstrap evidence artifact is too large")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise BootstrapEvidenceReceiverError("bootstrap evidence write was partial")
            written += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        metadata = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o400
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
        ):
            raise BootstrapEvidenceReceiverError(
                "bootstrap evidence artifact is not a sealed owner file"
            )
    except OSError as exc:
        raise BootstrapEvidenceReceiverError(
            "bootstrap evidence artifact must be a new exclusive file"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _canonical_payload_bytes(payload: str) -> bytes:
    if type(payload) is not str:
        raise BootstrapEvidenceReceiverError("producer canonical payload must be text")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BootstrapEvidenceReceiverError("producer canonical payload is invalid JSON") from exc
    expected = json.dumps(parsed, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    if expected != payload:
        raise BootstrapEvidenceReceiverError("producer payload is not canonical JSON")
    return payload.encode("utf-8")


class BrokerSnapshotEvidenceReceiver:
    __slots__ = ("__authority", "__clock", "__output", "__runtime", "__state")

    def __init__(
        self,
        token: object,
        runtime: RuntimeContract,
        output: Path,
        state: _BundleBindings,
        authority: _SigningAuthorityHandle,
        clock: Callable[[], datetime],
    ) -> None:
        if token is not _FACTORY_TOKEN:
            raise BootstrapEvidenceReceiverError("broker receiver is factory-only")
        self.__runtime = runtime
        self.__output = output
        self.__state = state
        self.__authority: _SigningAuthorityHandle | None = authority
        self.__clock = clock

    def receive_broker_snapshot_producer_result(
        self,
        result: BrokerSnapshotProducerResult,
    ) -> VerifiedBrokerEvidenceEnvelope:
        if self.__authority is None:
            raise BootstrapEvidenceReceiverError("broker signing capability was released")
        # This one-shot producer claim must happen before any field or payload
        # is read, hashed, persisted, or signed.
        result = assert_producer_owned_broker_snapshot_result(result, receiver=self)
        if type(result) is not BrokerSnapshotProducerResult:
            raise BootstrapEvidenceReceiverError(
                "broker receiver requires BrokerSnapshotProducerResult"
            )
        if (
            result.snapshot.account.account_scope != self.__state.account_scope
            or result.snapshot.account.account_alias != self.__runtime.account_alias
            or self.__state.broker_snapshot_id is not None
        ):
            raise BootstrapEvidenceReceiverError(
                "broker producer result is outside this evidence bundle"
            )
        payload = _canonical_payload_bytes(result.canonical_payload)
        artifact_hash = hashlib.sha256(payload).hexdigest()
        artifact_path = self.__output / _BROKER_ARTIFACT_FILENAME
        _write_new_sealed_file(artifact_path, payload)
        now = self.__clock().astimezone(timezone.utc)
        values, authentication = _sign_bound_artifact_receipt(
            self.__authority,
            artifact_kind="broker_snapshot",
            artifact_sha256=artifact_hash,
            producer_object_id=result.snapshot_id,
            runtime_fingerprint=self.__state.runtime_fingerprint,
            account_scope=self.__state.account_scope,
            bundle_id=self.__state.bundle_id,
            now=now,
        )
        receipt_path = artifact_path.with_name(artifact_path.name + AUTH_SUFFIX)
        _write_new_sealed_file(
            receipt_path,
            json.dumps(values, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            ),
        )
        self.__state.broker_snapshot_id = result.snapshot_id
        snapshot_hash = hashlib.sha256(
            result.snapshot.canonical_payload().encode("utf-8")
        ).hexdigest()
        self.__state.broker_snapshot_hash = snapshot_hash
        self.__state.broker_artifact_hash = artifact_hash
        self.__state.broker_receipt_id = authentication.receipt_id
        self.__state.broker_public_key_fingerprint = authentication.public_key_fingerprint
        self.__state.broker_artifact = SealedBootstrapEvidenceArtifact(
            artifact_kind="broker_snapshot",
            artifact_path=artifact_path,
            authentication_receipt_path=receipt_path,
            artifact_sha256=artifact_hash,
            producer_object_id=result.snapshot_id,
        )
        return _register_verified_broker_envelope(
            VerifiedBrokerEvidenceEnvelope(
                snapshot=result.snapshot,
                snapshot_id=result.snapshot_id,
                snapshot_hash=snapshot_hash,
                runtime_fingerprint=self.__state.runtime_fingerprint,
                account_scope=self.__state.account_scope,
                receipt_id=authentication.receipt_id,
                public_key_fingerprint=authentication.public_key_fingerprint,
                artifact_hash=artifact_hash,
                issued_at=authentication.issued_at,
                expires_at=authentication.expires_at,
                _marker=_VERIFIED_BROKER_MARKER,
            )
        )

    def _release_capability(self) -> None:
        _release_signing_authority(self.__authority)
        self.__authority = None


class ReconciliationEvidenceReceiver:
    __slots__ = ("__authority", "__clock", "__output", "__state", "__weakref__")

    def __init__(
        self,
        token: object,
        output: Path,
        state: _BundleBindings,
        authority: _SigningAuthorityHandle,
        clock: Callable[[], datetime],
    ) -> None:
        if token is not _FACTORY_TOKEN:
            raise BootstrapEvidenceReceiverError("reconciliation receiver is factory-only")
        self.__output = output
        self.__state = state
        self.__authority: _SigningAuthorityHandle | None = authority
        self.__clock = clock

    def stage_unsigned_bootstrap_reconciliation(
        self,
        result: UnsignedBootstrapReconciliation,
    ) -> object:
        if self.__authority is None:
            raise BootstrapEvidenceReceiverError("reconciliation signing capability was released")
        # Claim the producer-owned one-shot result before reading any field,
        # hashing its payload, or creating unpublished staging files.
        result = assert_and_consume_producer_owned_bootstrap_reconciliation(result)
        if type(result) is not UnsignedBootstrapReconciliation:
            raise BootstrapEvidenceReceiverError(
                "reconciliation receiver requires UnsignedBootstrapReconciliation"
            )
        if (
            result.bundle_id != self.__state.bundle_id
            or result.runtime_fingerprint != self.__state.runtime_fingerprint
            or result.account_scope != self.__state.account_scope
            or result.database_identity != self.__state.database_identity
            or result.broker_snapshot_id != self.__state.broker_snapshot_id
            or result.broker_snapshot_hash != self.__state.broker_snapshot_hash
            or result.broker_artifact_hash != self.__state.broker_artifact_hash
            or result.safety_journal_path != self.__state.safety_journal_path
            or result.safety_journal_identity != self.__state.safety_journal_identity
            or self.__state.reconciliation_snapshot_id is not None
        ):
            raise BootstrapEvidenceReceiverError(
                "reconciliation result is not cross-bound to the broker bundle"
            )
        payload = _canonical_payload_bytes(result.canonical_payload())
        artifact_hash = hashlib.sha256(payload).hexdigest()
        final_artifact_path = self.__output / _RECONCILIATION_ARTIFACT_FILENAME
        final_receipt_path = final_artifact_path.with_name(final_artifact_path.name + AUTH_SUFFIX)
        stage_nonce = secrets.token_hex(32)
        staged_artifact_path = self.__output / (
            f".{_RECONCILIATION_ARTIFACT_FILENAME}.stage-{stage_nonce}"
        )
        staged_receipt_path = self.__output / (
            f".{_RECONCILIATION_ARTIFACT_FILENAME}{AUTH_SUFFIX}.stage-{stage_nonce}"
        )
        if final_artifact_path.exists() or final_receipt_path.exists():
            raise BootstrapEvidenceReceiverError("reconciliation evidence was already published")
        now = self.__clock().astimezone(timezone.utc)
        values, _authentication = _sign_bound_artifact_receipt(
            self.__authority,
            artifact_kind="reconciliation_report",
            artifact_sha256=artifact_hash,
            producer_object_id=result.snapshot_id,
            runtime_fingerprint=self.__state.runtime_fingerprint,
            account_scope=self.__state.account_scope,
            bundle_id=self.__state.bundle_id,
            now=now,
        )
        receipt_payload = json.dumps(
            values,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        artifact = SealedBootstrapEvidenceArtifact(
            artifact_kind="reconciliation_report",
            artifact_path=final_artifact_path,
            authentication_receipt_path=final_receipt_path,
            artifact_sha256=artifact_hash,
            producer_object_id=result.snapshot_id,
        )
        stage = _StagedReconciliationEvidence(
            receiver=self,
            artifact=artifact,
            staged_artifact_path=staged_artifact_path,
            staged_receipt_path=staged_receipt_path,
            final_artifact_path=final_artifact_path,
            final_receipt_path=final_receipt_path,
            portfolio_ids=result.portfolio_ids,
            database_device=result.database_device,
            database_inode=result.database_inode,
            safety_journal_path=result.safety_journal_path,
            safety_journal_identity=result.safety_journal_identity,
            safety_journal_device=result.safety_journal_device,
            safety_journal_inode=result.safety_journal_inode,
            safety_journal_last_sequence=result.safety_journal_last_sequence,
            safety_journal_last_chain_hash=result.safety_journal_last_chain_hash,
            terminal_settlement_count=result.terminal_settlement_count,
            terminal_fill_count=result.terminal_fill_count,
            local_simulator_positions_count=result.local_simulator_positions_count,
            broker_positions_count=result.broker_positions_count,
            broker_open_orders_count=result.broker_open_orders_count,
            marker=_RECONCILIATION_STAGE_MARKER,
        )
        try:
            _write_new_sealed_file(staged_artifact_path, payload)
            _write_new_sealed_file(staged_receipt_path, receipt_payload)
            with _RECONCILIATION_STAGE_REGISTRY_LOCK:
                _RECONCILIATION_STAGE_REGISTRY[id(stage)] = stage
        except BaseException:
            for path in (staged_receipt_path, staged_artifact_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return stage

    def commit_staged_bootstrap_reconciliation(
        self,
        stage: object,
    ) -> SealedBootstrapEvidenceArtifact:
        checked = self.__consume_stage(stage)
        artifact_published = False
        try:
            if checked.final_artifact_path.exists() or checked.final_receipt_path.exists():
                raise BootstrapEvidenceReceiverError(
                    "reconciliation evidence publication target already exists"
                )
            checked.staged_artifact_path.rename(checked.final_artifact_path)
            artifact_published = True
            checked.staged_receipt_path.rename(checked.final_receipt_path)
        except BaseException:
            if artifact_published and checked.final_artifact_path.exists():
                try:
                    checked.final_artifact_path.rename(checked.staged_artifact_path)
                except OSError as exc:
                    with _RECONCILIATION_STAGE_REGISTRY_LOCK:
                        _RECONCILIATION_STAGE_REGISTRY[id(checked)] = checked
                    raise BootstrapEvidenceReceiverError(
                        "reconciliation publication rollback failed closed"
                    ) from exc
            with _RECONCILIATION_STAGE_REGISTRY_LOCK:
                _RECONCILIATION_STAGE_REGISTRY[id(checked)] = checked
            raise
        self.__state.reconciliation_snapshot_id = checked.artifact.producer_object_id
        self.__state.reconciliation_portfolio_ids = checked.portfolio_ids
        self.__state.database_device = checked.database_device
        self.__state.database_inode = checked.database_inode
        self.__state.safety_journal_path = checked.safety_journal_path
        self.__state.safety_journal_identity = checked.safety_journal_identity
        self.__state.safety_journal_device = checked.safety_journal_device
        self.__state.safety_journal_inode = checked.safety_journal_inode
        self.__state.safety_journal_last_sequence = checked.safety_journal_last_sequence
        self.__state.safety_journal_last_chain_hash = checked.safety_journal_last_chain_hash
        self.__state.terminal_settlement_count = checked.terminal_settlement_count
        self.__state.terminal_fill_count = checked.terminal_fill_count
        self.__state.local_simulator_positions_count = checked.local_simulator_positions_count
        self.__state.broker_positions_count = checked.broker_positions_count
        self.__state.broker_open_orders_count = checked.broker_open_orders_count
        return checked.artifact

    def abort_staged_bootstrap_reconciliation(self, stage: object) -> None:
        checked = self.__consume_stage(stage)
        failures: list[OSError] = []
        for path in (checked.staged_receipt_path, checked.staged_artifact_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(exc)
        if failures:
            raise BootstrapEvidenceReceiverError(
                "unpublished reconciliation stage could not be removed"
            ) from failures[0]

    def __consume_stage(self, stage: object) -> _StagedReconciliationEvidence:
        if type(stage) is not _StagedReconciliationEvidence:
            raise BootstrapEvidenceReceiverError("reconciliation stage is not core-owned")
        with _RECONCILIATION_STAGE_REGISTRY_LOCK:
            registered = _RECONCILIATION_STAGE_REGISTRY.pop(id(stage), None)
        if (
            registered is not stage
            or stage.receiver is not self
            or stage.marker is not _RECONCILIATION_STAGE_MARKER
        ):
            raise BootstrapEvidenceReceiverError(
                "reconciliation stage is forged, cross-receiver, or already consumed"
            )
        return stage

    def _release_capability(self) -> None:
        with _RECONCILIATION_RECEIVER_REGISTRY_LOCK:
            registered = _RECONCILIATION_RECEIVER_REGISTRY.get(id(self))
            if registered is self:
                _RECONCILIATION_RECEIVER_REGISTRY.pop(id(self), None)
        _release_signing_authority(self.__authority)
        self.__authority = None

    def _bundle_identity(self) -> ReconciliationBundleIdentity:
        return ReconciliationBundleIdentity(
            receiver_type=type(self),
            bundle_id=self.__state.bundle_id,
            runtime_fingerprint=self.__state.runtime_fingerprint,
            account_scope=self.__state.account_scope,
            database_identity=self.__state.database_identity,
        )


def assert_reconciliation_receiver_capability(
    receiver: object,
) -> ReconciliationBundleIdentity:
    """Authenticate one exact receiver created by the core capability factory."""

    if type(receiver) is not ReconciliationEvidenceReceiver:
        raise BootstrapEvidenceReceiverError(
            "exact factory-issued reconciliation receiver is required"
        )
    with _RECONCILIATION_RECEIVER_REGISTRY_LOCK:
        registered = _RECONCILIATION_RECEIVER_REGISTRY.get(id(receiver))
    if registered is not receiver:
        raise BootstrapEvidenceReceiverError(
            "reconciliation receiver was not issued by the core capability factory"
        )
    return receiver._bundle_identity()


class ProtectiveMarkEvidenceReceiver:
    __slots__ = ("__authority", "__clock", "__output", "__state", "__weakref__")

    def __init__(
        self,
        token: object,
        output: Path,
        state: _BundleBindings,
        authority: _SigningAuthorityHandle,
        clock: Callable[[], datetime],
    ) -> None:
        if token is not _FACTORY_TOKEN:
            raise BootstrapEvidenceReceiverError("protective mark receiver is factory-only")
        self.__output = output
        self.__state = state
        self.__authority: _SigningAuthorityHandle | None = authority
        self.__clock = clock

    def stage_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> object:
        if self.__authority is None:
            raise BootstrapEvidenceReceiverError("mark signing capability was released")
        # Producer ownership is receiver-bound and one-shot.  Consume it
        # before reading any result field or creating any artifact.
        result = assert_producer_owned_unsigned_bootstrap_protective_mark(
            result,
            receiver=self,
        )
        if type(result) is not UnsignedBootstrapProtectiveMark:
            raise BootstrapEvidenceReceiverError(
                "mark receiver requires UnsignedBootstrapProtectiveMark"
            )
        identity = (result.portfolio_id, result.symbol)
        if (
            self.__state.reconciliation_snapshot_id is None
            or result.bundle_id != self.__state.bundle_id
            or result.reconciliation_snapshot_id != self.__state.reconciliation_snapshot_id
            or result.broker_snapshot_id != self.__state.broker_snapshot_id
            or result.broker_snapshot_hash != self.__state.broker_snapshot_hash
            or result.broker_artifact_hash != self.__state.broker_artifact_hash
            or result.broker_receipt_id != self.__state.broker_receipt_id
            or result.broker_public_key_fingerprint != self.__state.broker_public_key_fingerprint
            or result.runtime_fingerprint != self.__state.runtime_fingerprint
            or result.account_scope != self.__state.account_scope
            or result.database_identity != self.__state.database_identity
            or result.database_device != self.__state.database_device
            or result.database_inode != self.__state.database_inode
            or result.portfolio_id not in self.__state.reconciliation_portfolio_ids
            or identity in self.__state.marks
        ):
            raise BootstrapEvidenceReceiverError(
                "protective mark is not cross-bound to the reconciled bundle"
            )
        payload = _canonical_payload_bytes(result.canonical_payload())
        artifact_hash = hashlib.sha256(payload).hexdigest()
        final_artifact_path = self.__output / (
            f"protective_mark-{result.portfolio_id}-{result.symbol}.json"
        )
        final_receipt_path = final_artifact_path.with_name(final_artifact_path.name + AUTH_SUFFIX)
        stage_nonce = secrets.token_hex(32)
        staged_artifact_path = self.__output / (f".{final_artifact_path.name}.stage-{stage_nonce}")
        staged_receipt_path = self.__output / (f".{final_receipt_path.name}.stage-{stage_nonce}")
        if final_artifact_path.exists() or final_receipt_path.exists():
            raise BootstrapEvidenceReceiverError("protective mark was already published")
        now = self.__clock().astimezone(timezone.utc)
        values, _authentication = _sign_bound_artifact_receipt(
            self.__authority,
            artifact_kind="protective_mark",
            artifact_sha256=artifact_hash,
            producer_object_id=result.protective_quote_id,
            runtime_fingerprint=self.__state.runtime_fingerprint,
            account_scope=self.__state.account_scope,
            bundle_id=self.__state.bundle_id,
            now=now,
        )
        receipt_payload = json.dumps(
            values,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        artifact = SealedBootstrapEvidenceArtifact(
            artifact_kind="protective_mark",
            artifact_path=final_artifact_path,
            authentication_receipt_path=final_receipt_path,
            artifact_sha256=artifact_hash,
            producer_object_id=result.protective_quote_id,
        )
        stage = _StagedProtectiveMarkEvidence(
            receiver=self,
            artifact=artifact,
            identity=identity,
            staged_artifact_path=staged_artifact_path,
            staged_receipt_path=staged_receipt_path,
            final_artifact_path=final_artifact_path,
            final_receipt_path=final_receipt_path,
            marker=_PROTECTIVE_MARK_STAGE_MARKER,
        )
        try:
            _write_new_sealed_file(staged_artifact_path, payload)
            _write_new_sealed_file(staged_receipt_path, receipt_payload)
            with _PROTECTIVE_MARK_STAGE_REGISTRY_LOCK:
                _PROTECTIVE_MARK_STAGE_REGISTRY[id(stage)] = stage
        except BaseException:
            for path in (staged_receipt_path, staged_artifact_path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise
        return stage

    def commit_staged_bootstrap_protective_mark(
        self,
        stage: object,
    ) -> SealedBootstrapEvidenceArtifact:
        checked = self.__consume_stage(stage)
        artifact_published = False
        try:
            if checked.final_artifact_path.exists() or checked.final_receipt_path.exists():
                raise BootstrapEvidenceReceiverError(
                    "protective mark publication target already exists"
                )
            checked.staged_artifact_path.rename(checked.final_artifact_path)
            artifact_published = True
            checked.staged_receipt_path.rename(checked.final_receipt_path)
        except BaseException:
            if artifact_published and checked.final_artifact_path.exists():
                try:
                    checked.final_artifact_path.rename(checked.staged_artifact_path)
                except OSError as exc:
                    with _PROTECTIVE_MARK_STAGE_REGISTRY_LOCK:
                        _PROTECTIVE_MARK_STAGE_REGISTRY[id(checked)] = checked
                    raise BootstrapEvidenceReceiverError(
                        "protective mark publication rollback failed closed"
                    ) from exc
            with _PROTECTIVE_MARK_STAGE_REGISTRY_LOCK:
                _PROTECTIVE_MARK_STAGE_REGISTRY[id(checked)] = checked
            raise
        self.__state.marks.add(checked.identity)
        return checked.artifact

    def abort_staged_bootstrap_protective_mark(self, stage: object) -> None:
        checked = self.__consume_stage(stage)
        failures: list[OSError] = []
        for path in (checked.staged_receipt_path, checked.staged_artifact_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                failures.append(exc)
        if failures:
            raise BootstrapEvidenceReceiverError(
                "unpublished protective mark stage could not be removed"
            ) from failures[0]

    def __consume_stage(self, stage: object) -> _StagedProtectiveMarkEvidence:
        if type(stage) is not _StagedProtectiveMarkEvidence:
            raise BootstrapEvidenceReceiverError("protective mark stage is not core-owned")
        with _PROTECTIVE_MARK_STAGE_REGISTRY_LOCK:
            registered = _PROTECTIVE_MARK_STAGE_REGISTRY.pop(id(stage), None)
        if (
            registered is not stage
            or stage.receiver is not self
            or stage.marker is not _PROTECTIVE_MARK_STAGE_MARKER
        ):
            raise BootstrapEvidenceReceiverError(
                "protective mark stage is forged, cross-receiver, or already consumed"
            )
        return stage

    def _release_capability(self) -> None:
        with _PROTECTIVE_MARK_RECEIVER_REGISTRY_LOCK:
            registered = _PROTECTIVE_MARK_RECEIVER_REGISTRY.get(id(self))
            if registered is self:
                _PROTECTIVE_MARK_RECEIVER_REGISTRY.pop(id(self), None)
        _release_signing_authority(self.__authority)
        self.__authority = None

    def _bundle_identity(self) -> ProtectiveMarkBundleIdentity:
        state = self.__state
        if (
            state.reconciliation_snapshot_id is None
            or state.broker_snapshot_id is None
            or state.broker_snapshot_hash is None
            or state.broker_artifact_hash is None
            or state.broker_receipt_id is None
            or state.broker_public_key_fingerprint is None
            or state.database_device is None
            or state.database_inode is None
        ):
            raise BootstrapEvidenceReceiverError(
                "protective mark receiver lacks reconciled bundle lineage"
            )
        return ProtectiveMarkBundleIdentity(
            receiver_type=type(self),
            bundle_id=state.bundle_id,
            reconciliation_snapshot_id=state.reconciliation_snapshot_id,
            broker_snapshot_id=state.broker_snapshot_id,
            broker_snapshot_hash=state.broker_snapshot_hash,
            broker_artifact_hash=state.broker_artifact_hash,
            broker_receipt_id=state.broker_receipt_id,
            broker_public_key_fingerprint=state.broker_public_key_fingerprint,
            runtime_fingerprint=state.runtime_fingerprint,
            account_scope=state.account_scope,
            database_identity=state.database_identity,
            database_device=state.database_device,
            database_inode=state.database_inode,
        )


def assert_protective_mark_receiver_capability(
    receiver: object,
    *,
    runtime_contract: object,
) -> ProtectiveMarkBundleIdentity:
    """Authenticate and describe one exact live core mark receiver."""

    if type(receiver) is not ProtectiveMarkEvidenceReceiver:
        raise BootstrapEvidenceReceiverError(
            "exact factory-issued protective mark receiver is required"
        )
    with _PROTECTIVE_MARK_RECEIVER_REGISTRY_LOCK:
        registered = _PROTECTIVE_MARK_RECEIVER_REGISTRY.get(id(receiver))
    state = receiver._bundle_identity()
    if (
        registered is not receiver
        or type(runtime_contract) is not RuntimeContract
        or runtime_contract.fingerprint != state.runtime_fingerprint
        or runtime_contract.safety_account_scope != state.account_scope
        or runtime_contract.database_identity != state.database_identity
    ):
        raise BootstrapEvidenceReceiverError(
            "protective mark receiver is unavailable or outside its reconciled runtime"
        )
    return state


def create_bootstrap_evidence_receivers(
    *,
    runtime_contract: RuntimeContract,
    capability_directory: Path,
    output_directory: Path,
) -> BootstrapEvidenceReceiverSet:
    """Resolve three isolated capabilities and create one new evidence bundle."""

    if type(runtime_contract) is not RuntimeContract:
        raise BootstrapEvidenceReceiverError("receiver factory requires RuntimeContract")
    if (
        runtime_contract.execution_mode != "paper"
        or runtime_contract.execution_source != "paper_simulator"
        or runtime_contract.ibkr_readonly is not True
        or runtime_contract.state_namespace != "paper"
        or not isinstance(runtime_contract.safety_account_scope, str)
    ):
        raise BootstrapEvidenceReceiverError("receiver runtime is not sealed paper/read-only")
    capability_directory = Path(capability_directory)
    trust = bootstrap_evidence_trust_public_dict()
    fingerprints = trust["public_key_fingerprints"]
    if not isinstance(fingerprints, dict):  # pragma: no cover - verifier invariant
        raise BootstrapEvidenceReceiverError("bootstrap trust manifest is malformed")
    broker_key = _safe_private_key(
        capability_directory,
        BROKER_PRIVATE_KEY_FILENAME,
        "broker_snapshot",
        str(fingerprints["broker_snapshot"]),
    )
    reconciliation_key = _safe_private_key(
        capability_directory,
        RECONCILIATION_PRIVATE_KEY_FILENAME,
        "reconciliation_report",
        str(fingerprints["reconciliation_report"]),
    )
    mark_key = _safe_private_key(
        capability_directory,
        PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
        "protective_mark",
        str(fingerprints["protective_mark"]),
    )
    if (
        len(
            {
                ed25519_public_key_fingerprint(broker_key.public_key()),
                ed25519_public_key_fingerprint(reconciliation_key.public_key()),
                ed25519_public_key_fingerprint(mark_key.public_key()),
            }
        )
        != 3
    ):
        raise BootstrapEvidenceReceiverError("signing capabilities must be pairwise distinct")
    output = _prepare_output_directory(Path(output_directory), capability_directory)
    bundle_id = "bootstrap-evidence-bundle-v1-" + secrets.token_hex(32)
    state = _BundleBindings(
        bundle_id=bundle_id,
        runtime_fingerprint=runtime_contract.fingerprint,
        account_scope=runtime_contract.safety_account_scope,
        database_identity=runtime_contract.database_identity,
        safety_journal_path=runtime_contract.safety_journal_path,
        safety_journal_identity=runtime_contract.safety_journal_identity,
    )

    def clock() -> datetime:
        return datetime.now(timezone.utc)

    authorities: list[_SigningAuthorityHandle] = []
    try:
        broker_authority = _register_signing_authority(
            key=broker_key,
            artifact_kind="broker_snapshot",
            producer_id="robotrader-broker-snapshot-producer-v1",
            runtime_fingerprint=state.runtime_fingerprint,
            account_scope=state.account_scope,
            bundle_id=bundle_id,
        )
        authorities.append(broker_authority)
        reconciliation_authority = _register_signing_authority(
            key=reconciliation_key,
            artifact_kind="reconciliation_report",
            producer_id="robotrader-reconciliation-producer-v1",
            runtime_fingerprint=state.runtime_fingerprint,
            account_scope=state.account_scope,
            bundle_id=bundle_id,
        )
        authorities.append(reconciliation_authority)
        mark_authority = _register_signing_authority(
            key=mark_key,
            artifact_kind="protective_mark",
            producer_id="robotrader-protective-mark-producer-v1",
            runtime_fingerprint=state.runtime_fingerprint,
            account_scope=state.account_scope,
            bundle_id=bundle_id,
        )
        authorities.append(mark_authority)
        reconciliation_receiver = ReconciliationEvidenceReceiver(
            _FACTORY_TOKEN,
            output,
            state,
            reconciliation_authority,
            clock,
        )
        mark_receiver = ProtectiveMarkEvidenceReceiver(
            _FACTORY_TOKEN,
            output,
            state,
            mark_authority,
            clock,
        )
        with _RECONCILIATION_RECEIVER_REGISTRY_LOCK:
            _RECONCILIATION_RECEIVER_REGISTRY[id(reconciliation_receiver)] = reconciliation_receiver
        with _PROTECTIVE_MARK_RECEIVER_REGISTRY_LOCK:
            _PROTECTIVE_MARK_RECEIVER_REGISTRY[id(mark_receiver)] = mark_receiver
        return BootstrapEvidenceReceiverSet(
            broker_snapshot=BrokerSnapshotEvidenceReceiver(
                _FACTORY_TOKEN,
                runtime_contract,
                output,
                state,
                broker_authority,
                clock,
            ),
            reconciliation_report=reconciliation_receiver,
            protective_mark=mark_receiver,
            _state=state,
        )
    except BaseException:
        for authority in authorities:
            _release_signing_authority(authority)
        raise

"""Verification-only trust boundary for exact-state evidence receipts.

The bootstrap consumer has three immutable, pairwise-distinct verification
roots.  It deliberately contains no signing API and rejects both verification-
root overrides and signing-key capability in its environment.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

AUTH_SUFFIX = ".auth.json"
AUTH_SCHEMA_VERSION = 3
_LEGACY_AUTH_SCHEMA_VERSION = 2
MAX_RECEIPT_LIFETIME = timedelta(minutes=5)
_AUTH_DOMAIN_V2 = b"robotrader-exact-state-bootstrap-evidence-ed25519-v2\0"
_AUTH_DOMAIN_V3 = b"robotrader-exact-state-bootstrap-evidence-ed25519-v3\0"
_RECEIPT_ID = "bevr-v2-"
_KINDS = {
    "broker_snapshot": "robotrader-broker-snapshot-producer-v1",
    "reconciliation_report": "robotrader-reconciliation-producer-v1",
    "protective_mark": "robotrader-protective-mark-producer-v1",
}
_REJECTED_PUBLIC_KEY_ENV = frozenset(
    {
        "BOOTSTRAP_BROKER_EVIDENCE_PUBLIC_KEY_PATH",
        "BOOTSTRAP_RECONCILIATION_EVIDENCE_PUBLIC_KEY_PATH",
        "BOOTSTRAP_MARK_EVIDENCE_PUBLIC_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_PUBLIC_KEY_PATH",
    }
)
FORBIDDEN_SIGNING_ENV = frozenset(
    {
        "BOOTSTRAP_BROKER_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_RECONCILIATION_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_MARK_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_SIGNING_KEY",
    }
)
_TRUST_ROOT = Path(__file__).with_name("bootstrap_evidence_trust")
_PINNED_PUBLIC_KEY_PATHS: dict[str, Path] = {
    "broker_snapshot": (_TRUST_ROOT / "broker_snapshot_ed25519_public.pem"),
    "reconciliation_report": (_TRUST_ROOT / "reconciliation_report_ed25519_public.pem"),
    "protective_mark": _TRUST_ROOT / "protective_mark_ed25519_public.pem",
}
_TRUST_MANIFEST_PATH = _TRUST_ROOT / "manifest.json"
_MAX_FILE_BYTES = 2 * 1024 * 1024


class BootstrapEvidenceAuthenticationError(ValueError):
    """An evidence receipt or producer key is unsafe or invalid."""


@dataclass(frozen=True, slots=True)
class AuthenticatedEvidenceReceipt:
    receipt_id: str
    artifact_kind: str
    producer_id: str
    artifact_sha256: str
    runtime_fingerprint: str
    account_scope: str
    publication_directory: str
    publication_nonce: str
    issued_at: datetime
    expires_at: datetime
    public_key_fingerprint: str
    signature_ed25519: str


def _safe_file_bytes(
    path: Path,
    label: str,
    *,
    allow_public_read: bool = False,
) -> bytes:
    protected = Path(path)
    if (
        not protected.is_absolute()
        or protected.parent.resolve(strict=True) / protected.name != protected
        or not hasattr(os, "O_NOFOLLOW")
    ):
        raise BootstrapEvidenceAuthenticationError(f"{label} path is not safely absolute")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(protected, flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or (stat.S_IMODE(before.st_mode) & (0o022 if allow_public_read else 0o077))
            or before.st_size > _MAX_FILE_BYTES
        ):
            raise BootstrapEvidenceAuthenticationError(f"{label} is not a sealed owner file")
        payload = os.read(descriptor, _MAX_FILE_BYTES + 1)
        if len(payload) > _MAX_FILE_BYTES or os.read(descriptor, 1):
            raise BootstrapEvidenceAuthenticationError(f"{label} is too large")
        after = os.fstat(descriptor)
        current = os.lstat(protected)
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or after.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise BootstrapEvidenceAuthenticationError(f"{label} changed while read")
        return payload
    except OSError as exc:
        raise BootstrapEvidenceAuthenticationError(f"{label} cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: object, label: str) -> datetime:
    if type(value) is not str:
        raise BootstrapEvidenceAuthenticationError(f"{label} must be a JSON string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapEvidenceAuthenticationError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BootstrapEvidenceAuthenticationError(f"{label} is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def receipt_signature_payload(values: Mapping[str, object]) -> bytes:
    fields: tuple[str, ...] = (
        "schema_version",
        "receipt_id",
        "artifact_kind",
        "producer_id",
        "artifact_sha256",
        "runtime_fingerprint",
        "account_scope",
        "issued_at",
        "expires_at",
        "public_key_fingerprint",
    )
    if "publication_directory" in values or "publication_nonce" in values:
        fields += ("publication_directory", "publication_nonce")
    signed = {key: values[key] for key in fields}
    kind = signed["artifact_kind"]
    if type(kind) is not str:
        raise BootstrapEvidenceAuthenticationError("receipt artifact_kind is invalid")
    domain = (
        _AUTH_DOMAIN_V3 if values.get("schema_version") == AUTH_SCHEMA_VERSION else _AUTH_DOMAIN_V2
    )
    return (
        domain
        + kind.encode("ascii")
        + b"\0"
        + json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def ed25519_public_key_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _load_public_key(path: Path) -> Ed25519PublicKey:
    payload = _safe_file_bytes(path, "pinned producer public key", allow_public_read=True)
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise BootstrapEvidenceAuthenticationError("producer public key is invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise BootstrapEvidenceAuthenticationError("producer public key must be Ed25519")
    return key


def _trust_manifest() -> dict[str, object]:
    payload = _safe_file_bytes(
        _TRUST_MANIFEST_PATH,
        "bootstrap evidence trust manifest",
        allow_public_read=True,
    )
    try:
        manifest = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapEvidenceAuthenticationError("trust manifest is invalid") from exc
    expected_fields = {
        "schema_version",
        "producer_ids",
        "public_key_fingerprints",
        "trust_set_digest",
    }
    if (
        type(manifest) is not dict
        or set(manifest) != expected_fields
        or manifest["schema_version"] != 1
        or manifest["producer_ids"] != _KINDS
        or type(manifest["public_key_fingerprints"]) is not dict
        or set(manifest["public_key_fingerprints"]) != set(_KINDS)
        or type(manifest["trust_set_digest"]) is not str
    ):
        raise BootstrapEvidenceAuthenticationError("trust manifest fields are invalid")
    canonical = json.dumps(
        {
            "producer_ids": manifest["producer_ids"],
            "public_key_fingerprints": manifest["public_key_fingerprints"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, manifest["trust_set_digest"]):
        raise BootstrapEvidenceAuthenticationError(
            "bootstrap evidence trust-set manifest digest is invalid"
        )
    return manifest


def _pinned_public_keys(
    manifest: dict[str, object] | None = None,
) -> dict[str, Ed25519PublicKey]:
    """Load the tracked trust set and reject every environment override surface."""

    rejected = _REJECTED_PUBLIC_KEY_ENV | FORBIDDEN_SIGNING_ENV
    present = sorted(name for name in rejected if name in os.environ)
    if present:
        raise BootstrapEvidenceAuthenticationError(
            "bootstrap consumer refuses evidence trust/signing overrides: " + ",".join(present)
        )
    keys: dict[str, Ed25519PublicKey] = {}
    fingerprints: dict[str, str] = {}
    manifest = _trust_manifest() if manifest is None else manifest
    expected_fingerprints = manifest["public_key_fingerprints"]
    if type(expected_fingerprints) is not dict:  # pragma: no cover - checked above
        raise BootstrapEvidenceAuthenticationError("trust manifest fingerprints are invalid")
    for kind, path in _PINNED_PUBLIC_KEY_PATHS.items():
        expected_fingerprint = expected_fingerprints[kind]
        if type(expected_fingerprint) is not str:
            raise BootstrapEvidenceAuthenticationError(
                f"tracked {kind} verification fingerprint is malformed"
            )
        key = _load_public_key(path)
        actual_fingerprint = ed25519_public_key_fingerprint(key)
        if not hmac.compare_digest(actual_fingerprint, expected_fingerprint):
            raise BootstrapEvidenceAuthenticationError(
                f"tracked {kind} verification key does not match its immutable fingerprint"
            )
        fingerprints[kind] = actual_fingerprint
        keys[kind] = key
    if len(set(fingerprints.values())) != len(_KINDS):
        raise BootstrapEvidenceAuthenticationError(
            "bootstrap evidence producers must use three distinct public keys"
        )
    return keys


def bootstrap_evidence_trust_public_dict() -> dict[str, object]:
    """Return reviewable trust facts included in every RuntimeContract fingerprint."""

    manifest = _trust_manifest()
    keys = _pinned_public_keys(manifest)
    fingerprints = {kind: ed25519_public_key_fingerprint(keys[kind]) for kind in sorted(_KINDS)}
    return {
        "schema_version": 1,
        "producer_ids": {kind: _KINDS[kind] for kind in sorted(_KINDS)},
        "public_key_fingerprints": fingerprints,
        "trust_set_digest": manifest["trust_set_digest"],
    }


def verify_receipt(
    *,
    raw: Mapping[str, object],
    artifact_kind: str,
    artifact_sha256: str,
    runtime_fingerprint: str,
    account_scope: str,
    publication_directory: str | None = None,
    publication_nonce: str | None = None,
    now: datetime | None = None,
) -> AuthenticatedEvidenceReceipt:
    expected_fields = {
        "schema_version",
        "receipt_id",
        "artifact_kind",
        "producer_id",
        "artifact_sha256",
        "runtime_fingerprint",
        "account_scope",
        "issued_at",
        "expires_at",
        "public_key_fingerprint",
        "signature_ed25519",
    }
    if publication_directory is not None:
        expected_fields |= {"publication_directory", "publication_nonce"}
    if set(raw) != expected_fields or type(raw.get("schema_version")) is not int:
        raise BootstrapEvidenceAuthenticationError("authentication receipt fields are invalid")
    expected_schema_version = (
        AUTH_SCHEMA_VERSION if publication_directory is not None else _LEGACY_AUTH_SCHEMA_VERSION
    )
    if raw["schema_version"] != expected_schema_version:
        raise BootstrapEvidenceAuthenticationError("authentication receipt schema is unsupported")
    for field in expected_fields - {"schema_version"}:
        if type(raw[field]) is not str:
            raise BootstrapEvidenceAuthenticationError(f"receipt {field} must be a JSON string")
    if (
        artifact_kind not in _KINDS
        or raw["artifact_kind"] != artifact_kind
        or raw["producer_id"] != _KINDS[artifact_kind]
        or raw["artifact_sha256"] != artifact_sha256
        or raw["runtime_fingerprint"] != runtime_fingerprint
        or raw["account_scope"] != account_scope
        or (
            publication_directory is not None
            and raw["publication_directory"] != publication_directory
        )
        or (publication_nonce is not None and raw.get("publication_nonce") != publication_nonce)
        or (
            publication_directory is not None
            and not re.fullmatch(
                r"bootstrap-publication-v1-[0-9a-f]{64}",
                str(raw["publication_nonce"]),
            )
        )
        or not str(raw["receipt_id"]).startswith(_RECEIPT_ID)
        or len(str(raw["receipt_id"])) != len(_RECEIPT_ID) + 64
    ):
        raise BootstrapEvidenceAuthenticationError(
            "authentication receipt is bound to the wrong producer, kind, or artifact"
        )
    issued_at = _parse_utc(raw["issued_at"], "receipt issued_at")
    expires_at = _parse_utc(raw["expires_at"], "receipt expires_at")
    observed_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if (
        issued_at > observed_now
        or expires_at < observed_now
        or expires_at <= issued_at
        or expires_at - issued_at > MAX_RECEIPT_LIFETIME
    ):
        raise BootstrapEvidenceAuthenticationError("authentication receipt is expired or future")
    public_key = _pinned_public_keys()[artifact_kind]
    fingerprint = ed25519_public_key_fingerprint(public_key)
    if raw["public_key_fingerprint"] != fingerprint:
        raise BootstrapEvidenceAuthenticationError("authentication receipt uses the wrong key")
    try:
        signature = base64.b64decode(str(raw["signature_ed25519"]), validate=True)
        public_key.verify(signature, receipt_signature_payload(raw))
    except (InvalidSignature, ValueError) as exc:
        raise BootstrapEvidenceAuthenticationError("authentication signature is invalid") from exc
    return AuthenticatedEvidenceReceipt(
        receipt_id=str(raw["receipt_id"]),
        artifact_kind=artifact_kind,
        producer_id=str(raw["producer_id"]),
        artifact_sha256=artifact_sha256,
        runtime_fingerprint=runtime_fingerprint,
        account_scope=account_scope,
        publication_directory=publication_directory or "",
        publication_nonce=str(raw.get("publication_nonce", "")),
        issued_at=issued_at,
        expires_at=expires_at,
        public_key_fingerprint=fingerprint,
        signature_ed25519=str(raw["signature_ed25519"]),
    )

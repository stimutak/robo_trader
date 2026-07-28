"""Producer-only Ed25519 receipts for exact-state bootstrap evidence.

The bootstrap consumer receives only three pinned public keys.  Private key
paths are accepted solely by the producer entry points in this module and the
standalone producer CLI; the bootstrap loader refuses to run if any signing
key capability is advertised in its environment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

AUTH_SUFFIX = ".auth.json"
AUTH_SCHEMA_VERSION = 2
MAX_RECEIPT_LIFETIME = timedelta(minutes=5)
_AUTH_DOMAIN = b"robotrader-exact-state-bootstrap-evidence-ed25519-v2\0"
_RECEIPT_ID = "bevr-v2-"
_KINDS = {
    "broker_snapshot": "robotrader-broker-snapshot-producer-v1",
    "reconciliation_report": "robotrader-reconciliation-producer-v1",
    "protective_mark": "robotrader-protective-mark-producer-v1",
}
PUBLIC_KEY_ENV = {
    "broker_snapshot": "BOOTSTRAP_BROKER_EVIDENCE_PUBLIC_KEY_PATH",
    "reconciliation_report": "BOOTSTRAP_RECONCILIATION_EVIDENCE_PUBLIC_KEY_PATH",
    "protective_mark": "BOOTSTRAP_MARK_EVIDENCE_PUBLIC_KEY_PATH",
}
FORBIDDEN_SIGNING_ENV = frozenset(
    {
        "BOOTSTRAP_BROKER_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_RECONCILIATION_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_MARK_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_PRIVATE_KEY_PATH",
        "BOOTSTRAP_EVIDENCE_SIGNING_KEY",
    }
)
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
    issued_at: datetime
    expires_at: datetime
    public_key_fingerprint: str


def _safe_file_bytes(path: Path, label: str, *, exact_mode: int | None = None) -> bytes:
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
            or stat.S_IMODE(before.st_mode) & 0o077
            or (exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode)
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


def _receipt_payload(values: Mapping[str, object]) -> bytes:
    signed = {
        key: values[key]
        for key in (
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
    }
    kind = signed["artifact_kind"]
    if type(kind) is not str:
        raise BootstrapEvidenceAuthenticationError("receipt artifact_kind is invalid")
    return (
        _AUTH_DOMAIN
        + kind.encode("ascii")
        + b"\0"
        + json.dumps(signed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _public_fingerprint(public_key: Ed25519PublicKey) -> str:
    raw = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    payload = _safe_file_bytes(path, "producer private key", exact_mode=0o400)
    try:
        key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as exc:
        raise BootstrapEvidenceAuthenticationError("producer private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise BootstrapEvidenceAuthenticationError("producer private key must be Ed25519")
    return key


def _load_public_key(path: Path) -> Ed25519PublicKey:
    payload = _safe_file_bytes(path, "producer public key", exact_mode=0o400)
    try:
        key = serialization.load_pem_public_key(payload)
    except (TypeError, ValueError) as exc:
        raise BootstrapEvidenceAuthenticationError("producer public key is invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise BootstrapEvidenceAuthenticationError("producer public key must be Ed25519")
    return key


def public_key_paths_from_consumer_environment() -> dict[str, Path]:
    """Return pinned verification keys while refusing signing capability."""

    present = sorted(name for name in FORBIDDEN_SIGNING_ENV if os.environ.get(name))
    if present:
        raise BootstrapEvidenceAuthenticationError(
            "bootstrap consumer refuses producer signing-key presence: " + ",".join(present)
        )
    paths: dict[str, Path] = {}
    fingerprints: dict[str, str] = {}
    for kind, name in PUBLIC_KEY_ENV.items():
        raw = os.environ.get(name, "")
        path = Path(raw)
        if not raw or not path.is_absolute():
            raise BootstrapEvidenceAuthenticationError(f"{name} must be an absolute public key")
        fingerprints[kind] = _public_fingerprint(_load_public_key(path))
        paths[kind] = path
    if len(set(fingerprints.values())) != len(_KINDS):
        raise BootstrapEvidenceAuthenticationError(
            "bootstrap evidence producers must use three distinct public keys"
        )
    return paths


def _write_receipt(path: Path, payload: bytes) -> None:
    if not path.is_absolute() or path.parent.resolve(strict=True) / path.name != path:
        raise BootstrapEvidenceAuthenticationError("authentication receipt path is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        if os.write(descriptor, payload) != len(payload):
            raise BootstrapEvidenceAuthenticationError("authentication receipt write was partial")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        metadata = os.fstat(descriptor)
        current = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
            or stat.S_IMODE(current.st_mode) != 0o400
        ):
            raise BootstrapEvidenceAuthenticationError("authentication receipt is not single-link")
    except OSError as exc:
        raise BootstrapEvidenceAuthenticationError(
            "authentication receipt must be a new exclusive file"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _emit_receipt(
    *,
    artifact_path: Path,
    private_key_path: Path,
    artifact_kind: str,
    runtime_fingerprint: str,
    account_scope: str,
    issued_at: datetime | None = None,
    lifetime: timedelta = MAX_RECEIPT_LIFETIME,
) -> Path:
    if artifact_kind not in _KINDS:
        raise BootstrapEvidenceAuthenticationError("artifact kind has no authorized producer")
    if lifetime <= timedelta(0) or lifetime > MAX_RECEIPT_LIFETIME:
        raise BootstrapEvidenceAuthenticationError("receipt lifetime is outside its safety bound")
    artifact = _safe_file_bytes(Path(artifact_path), f"{artifact_kind} artifact")
    key = _load_private_key(Path(private_key_path))
    now = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    values: dict[str, object] = {
        "schema_version": AUTH_SCHEMA_VERSION,
        "receipt_id": _RECEIPT_ID + secrets.token_hex(32),
        "artifact_kind": artifact_kind,
        "producer_id": _KINDS[artifact_kind],
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "runtime_fingerprint": runtime_fingerprint,
        "account_scope": account_scope,
        "issued_at": _utc_text(now),
        "expires_at": _utc_text(now + lifetime),
        "public_key_fingerprint": _public_fingerprint(key.public_key()),
    }
    signature = key.sign(_receipt_payload(values))
    values["signature_ed25519"] = base64.b64encode(signature).decode("ascii")
    receipt_path = Path(artifact_path).with_name(Path(artifact_path).name + AUTH_SUFFIX)
    _write_receipt(
        receipt_path,
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return receipt_path


def emit_broker_snapshot_receipt(**kwargs: object) -> Path:
    return _emit_receipt(artifact_kind="broker_snapshot", **kwargs)


def emit_reconciliation_report_receipt(**kwargs: object) -> Path:
    return _emit_receipt(artifact_kind="reconciliation_report", **kwargs)


def emit_protective_mark_receipt(**kwargs: object) -> Path:
    return _emit_receipt(artifact_kind="protective_mark", **kwargs)


def verify_receipt(
    *,
    raw: Mapping[str, object],
    artifact_kind: str,
    artifact_sha256: str,
    runtime_fingerprint: str,
    account_scope: str,
    public_key_path: Path,
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
    if set(raw) != expected_fields or type(raw.get("schema_version")) is not int:
        raise BootstrapEvidenceAuthenticationError("authentication receipt fields are invalid")
    if raw["schema_version"] != AUTH_SCHEMA_VERSION:
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
    public_key = _load_public_key(Path(public_key_path))
    fingerprint = _public_fingerprint(public_key)
    if raw["public_key_fingerprint"] != fingerprint:
        raise BootstrapEvidenceAuthenticationError("authentication receipt uses the wrong key")
    try:
        signature = base64.b64decode(str(raw["signature_ed25519"]), validate=True)
        public_key.verify(signature, _receipt_payload(raw))
    except (InvalidSignature, ValueError) as exc:
        raise BootstrapEvidenceAuthenticationError("authentication signature is invalid") from exc
    return AuthenticatedEvidenceReceipt(
        receipt_id=str(raw["receipt_id"]),
        artifact_kind=artifact_kind,
        producer_id=str(raw["producer_id"]),
        artifact_sha256=artifact_sha256,
        runtime_fingerprint=runtime_fingerprint,
        account_scope=account_scope,
        issued_at=issued_at,
        expires_at=expires_at,
        public_key_fingerprint=fingerprint,
    )

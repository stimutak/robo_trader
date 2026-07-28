#!/usr/bin/env python3
"""One-time provisioning for isolated bootstrap evidence signing capabilities.

This command creates three distinct owner-only private keys outside the source
tree and updates only the tracked public trust roots and manifest.  It never
prints, exports, or stores private-key paths in environment configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.bootstrap_evidence_auth import (  # noqa: E402
    ed25519_public_key_fingerprint,
)

CONFIRMATION = "PROVISION-THREE-ISOLATED-BOOTSTRAP-EVIDENCE-KEYS"
BROKER_PRIVATE_KEY_FILENAME = "broker_snapshot_ed25519_private.pem"
RECONCILIATION_PRIVATE_KEY_FILENAME = "reconciliation_report_ed25519_private.pem"
PROTECTIVE_MARK_PRIVATE_KEY_FILENAME = "protective_mark_ed25519_private.pem"
_KINDS = {
    "broker_snapshot": (
        BROKER_PRIVATE_KEY_FILENAME,
        "broker_snapshot_ed25519_public.pem",
        "robotrader-broker-snapshot-producer-v1",
    ),
    "reconciliation_report": (
        RECONCILIATION_PRIVATE_KEY_FILENAME,
        "reconciliation_report_ed25519_public.pem",
        "robotrader-reconciliation-producer-v1",
    ),
    "protective_mark": (
        PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
        "protective_mark_ed25519_public.pem",
        "robotrader-protective-mark-producer-v1",
    ),
}


class BootstrapKeyProvisioningError(ValueError):
    """Provisioning target or write sequence is unsafe."""


def _write_new_private_key(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        if os.write(descriptor, payload) != len(payload):
            raise BootstrapKeyProvisioningError("private-key write was partial")
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
        ):
            raise BootstrapKeyProvisioningError("private key was not sealed owner-readonly")
    except OSError as exc:
        raise BootstrapKeyProvisioningError("private key could not be created exclusively") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _atomic_public_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.provision-{os.getpid()}")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        if os.write(descriptor, payload) != len(payload):
            raise BootstrapKeyProvisioningError("public trust write was partial")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except OSError as exc:
        raise BootstrapKeyProvisioningError("public trust root could not be updated") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def provision_bootstrap_evidence_keys(
    *,
    project_root: Path,
    capability_directory: Path,
    confirmation: str,
) -> dict[str, object]:
    """Provision three keys and update the tracked public trust manifest."""

    if confirmation != CONFIRMATION:
        raise BootstrapKeyProvisioningError("exact provisioning confirmation is required")
    project = Path(project_root).resolve(strict=True)
    trust_root = project / "robo_trader" / "bootstrap_evidence_trust"
    try:
        trust_metadata = os.lstat(trust_root)
    except OSError as exc:
        raise BootstrapKeyProvisioningError("tracked bootstrap trust directory is missing") from exc
    if (
        stat.S_ISLNK(trust_metadata.st_mode)
        or not stat.S_ISDIR(trust_metadata.st_mode)
        or trust_metadata.st_uid != os.geteuid()
        or trust_root.resolve(strict=True) != trust_root
    ):
        raise BootstrapKeyProvisioningError("tracked bootstrap trust directory is missing")
    capability = Path(capability_directory)
    if (
        not capability.is_absolute()
        or capability == project
        or project in capability.parents
        or capability.parent.resolve(strict=True) / capability.name != capability
    ):
        raise BootstrapKeyProvisioningError(
            "private capabilities must target a new directory outside the repository"
        )
    try:
        os.mkdir(capability, 0o700)
    except OSError as exc:
        raise BootstrapKeyProvisioningError(
            "capability directory must be new and exclusively creatable"
        ) from exc
    directory_metadata = os.lstat(capability)
    if (
        not stat.S_ISDIR(directory_metadata.st_mode)
        or stat.S_ISLNK(directory_metadata.st_mode)
        or directory_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(directory_metadata.st_mode) != 0o700
    ):
        raise BootstrapKeyProvisioningError("capability directory is not owner-only")

    public_payloads: dict[str, bytes] = {}
    fingerprints: dict[str, str] = {}
    for kind, (private_filename, _public_filename, _producer_id) in _KINDS.items():
        key = Ed25519PrivateKey.generate()
        private_payload = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        _write_new_private_key(capability / private_filename, private_payload)
        public_payloads[kind] = key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        fingerprints[kind] = ed25519_public_key_fingerprint(key.public_key())
    if len(set(fingerprints.values())) != 3:
        raise BootstrapKeyProvisioningError("generated signing capabilities are not distinct")

    producer_ids = {kind: values[2] for kind, values in _KINDS.items()}
    canonical = json.dumps(
        {
            "producer_ids": producer_ids,
            "public_key_fingerprints": fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = {
        "producer_ids": producer_ids,
        "public_key_fingerprints": fingerprints,
        "schema_version": 1,
        "trust_set_digest": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    for kind, (_private_filename, public_filename, _producer_id) in _KINDS.items():
        _atomic_public_write(trust_root / public_filename, public_payloads[kind])
    _atomic_public_write(
        trust_root / "manifest.json",
        (
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8"),
    )
    return {
        "authorizes_startup": False,
        "private_keys_exported": False,
        "schema_version": 1,
        "status": "PROVISIONED_REVIEW_AND_COMMIT_PUBLIC_TRUST_ROOTS",
        "trust_set_digest": manifest["trust_set_digest"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--capability-directory", type=Path, required=True)
    parser.add_argument("--confirm", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report = provision_bootstrap_evidence_keys(
            project_root=args.project_root,
            capability_directory=args.capability_directory,
            confirmation=args.confirm,
        )
    except (BootstrapKeyProvisioningError, OSError) as exc:
        print(
            json.dumps(
                {
                    "authorizes_startup": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "schema_version": 1,
                    "status": "BLOCKED",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

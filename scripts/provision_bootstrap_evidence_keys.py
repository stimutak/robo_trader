#!/usr/bin/env python3
"""One-time provisioning for isolated bootstrap evidence signing capabilities.

This command creates three distinct owner-only private keys outside the source
tree and updates only the tracked public trust roots and manifest.  It never
prints, exports, or stores private-key paths in environment configuration.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

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
_PUBLIC_FILENAMES = tuple(values[1] for values in _KINDS.values())
_TRUST_FILENAMES = frozenset((*_PUBLIC_FILENAMES, "manifest.json"))
_PRIVATE_FILENAMES = tuple(values[0] for values in _KINDS.values())
_RENAME_EXCHANGE = 0x00000002


class BootstrapKeyProvisioningError(ValueError):
    """Provisioning target or write sequence is unsafe."""


def _write_all(descriptor: int, payload: bytes, *, label: str) -> None:
    written = 0
    while written < len(payload):
        count = os.write(descriptor, payload[written:])
        if count <= 0:
            raise BootstrapKeyProvisioningError(f"{label} write was partial")
        written += count


def _open_directory(path: Path, *, label: str) -> int:
    descriptor: int | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        current = os.lstat(path)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise BootstrapKeyProvisioningError(f"{label} cannot be opened safely") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
    ):
        os.close(descriptor)
        raise BootstrapKeyProvisioningError(f"{label} identity is unsafe")
    return descriptor


def _open_child_directory(parent_descriptor: int, name: str, *, label: str) -> int:
    descriptor: int | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
        metadata = os.fstat(descriptor)
        current = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise BootstrapKeyProvisioningError(f"{label} cannot be opened safely") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
    ):
        os.close(descriptor)
        raise BootstrapKeyProvisioningError(f"{label} identity is unsafe")
    return descriptor


def _write_new_private_key(directory_descriptor: int, filename: str, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        _write_all(descriptor, payload, label="private-key")
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
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


def _write_new_public_file(directory_descriptor: int, filename: str, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        _write_all(descriptor, payload, label="public trust")
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise BootstrapKeyProvisioningError("public trust file was not sealed read-only")
    except OSError as exc:
        raise BootstrapKeyProvisioningError("public trust file could not be staged") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _safe_public_file_bytes(directory_descriptor: int, filename: str) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_descriptor,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022
            or before.st_size > 64 * 1024
        ):
            raise BootstrapKeyProvisioningError("public trust file is unsafe")
        payload = os.read(descriptor, 64 * 1024 + 1)
        after = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            len(payload) > 64 * 1024
            or os.read(descriptor, 1)
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (current.st_dev, current.st_ino)
            or current.st_nlink != 1
        ):
            raise BootstrapKeyProvisioningError("public trust file changed while validated")
        return payload
    except OSError as exc:
        raise BootstrapKeyProvisioningError("public trust file cannot be read safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _validate_complete_trust_set(directory_descriptor: int) -> dict[str, object]:
    try:
        entries = frozenset(os.listdir(directory_descriptor))
    except OSError as exc:
        raise BootstrapKeyProvisioningError("public trust directory cannot be listed") from exc
    if entries != _TRUST_FILENAMES:
        raise BootstrapKeyProvisioningError("public trust directory is not a complete exact set")
    try:
        manifest = json.loads(
            _safe_public_file_bytes(directory_descriptor, "manifest.json").decode("utf-8")
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapKeyProvisioningError("public trust manifest is invalid") from exc
    producer_ids = {kind: values[2] for kind, values in _KINDS.items()}
    if (
        type(manifest) is not dict
        or set(manifest)
        != {"producer_ids", "public_key_fingerprints", "schema_version", "trust_set_digest"}
        or manifest["schema_version"] != 1
        or manifest["producer_ids"] != producer_ids
        or type(manifest["public_key_fingerprints"]) is not dict
        or set(manifest["public_key_fingerprints"]) != set(_KINDS)
        or type(manifest["trust_set_digest"]) is not str
    ):
        raise BootstrapKeyProvisioningError("public trust manifest fields are invalid")
    actual_fingerprints: dict[str, str] = {}
    for kind, (_private_filename, public_filename, _producer_id) in _KINDS.items():
        try:
            key = serialization.load_pem_public_key(
                _safe_public_file_bytes(directory_descriptor, public_filename)
            )
        except (TypeError, ValueError) as exc:
            raise BootstrapKeyProvisioningError("public trust key is invalid") from exc
        if not isinstance(key, Ed25519PublicKey):
            raise BootstrapKeyProvisioningError("public trust key must be Ed25519")
        actual_fingerprints[kind] = ed25519_public_key_fingerprint(key)
    if actual_fingerprints != manifest["public_key_fingerprints"] or len(
        set(actual_fingerprints.values())
    ) != len(_KINDS):
        raise BootstrapKeyProvisioningError("public trust keys do not match the manifest")
    canonical = json.dumps(
        {
            "producer_ids": producer_ids,
            "public_key_fingerprints": actual_fingerprints,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    if not secrets.compare_digest(
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        manifest["trust_set_digest"],
    ):
        raise BootstrapKeyProvisioningError("public trust manifest digest is invalid")
    return manifest


def _atomic_exchange_directories(
    parent_descriptor: int,
    left_name: str,
    right_name: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    left = os.fsencode(left_name)
    right = os.fsencode(right_name)
    if sys.platform == "darwin":
        operation = getattr(library, "renameatx_np", None)
        if operation is None:
            raise BootstrapKeyProvisioningError("atomic directory exchange is unavailable")
        operation.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        operation.restype = ctypes.c_int
        result = operation(
            parent_descriptor,
            left,
            parent_descriptor,
            right,
            _RENAME_EXCHANGE,
        )
    elif sys.platform.startswith("linux"):
        operation = getattr(library, "renameat2", None)
        if operation is None:
            raise BootstrapKeyProvisioningError("atomic directory exchange is unavailable")
        operation.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        operation.restype = ctypes.c_int
        result = operation(
            parent_descriptor,
            left,
            parent_descriptor,
            right,
            _RENAME_EXCHANGE,
        )
    else:
        raise BootstrapKeyProvisioningError("atomic directory exchange is unsupported")
    if result != 0:
        error_number = ctypes.get_errno()
        raise BootstrapKeyProvisioningError("atomic public trust exchange failed") from OSError(
            error_number,
            os.strerror(error_number),
        )


def _remove_exact_directory(
    parent_descriptor: int,
    name: str,
    allowed_entries: frozenset[str],
    *,
    expected_identity: tuple[int, int],
) -> bool:
    descriptor: int | None = None
    try:
        descriptor = _open_child_directory(
            parent_descriptor,
            name,
            label="transaction directory",
        )
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != expected_identity:
            return False
        entries = frozenset(os.listdir(descriptor))
        if not entries.issubset(allowed_entries):
            return False
        for filename in entries:
            metadata = os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                return False
        for filename in entries:
            os.unlink(filename, dir_fd=descriptor)
        os.fsync(descriptor)
        current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != expected_identity:
            return False
        os.close(descriptor)
        descriptor = None
        os.rmdir(name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True
    except (OSError, BootstrapKeyProvisioningError):
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _provisioning_fault(boundary: str) -> None:
    """No-op production hook used only for deterministic transaction tests."""

    del boundary


def provision_bootstrap_evidence_keys(
    *,
    project_root: Path,
    capability_directory: Path,
    confirmation: str,
) -> dict[str, object]:
    """Provision three keys and atomically replace the public trust set."""

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
    trust_parent_descriptor: int | None = None
    trust_descriptor: int | None = None
    capability_parent_descriptor: int | None = None
    capability_descriptor: int | None = None
    stage_descriptor: int | None = None
    stage_name: str | None = None
    capability_identity: tuple[int, int] | None = None
    stage_identity: tuple[int, int] | None = None
    old_trust_identity: tuple[int, int] | None = None
    capability_created = False
    exchanged = False
    committed = False
    old_trust_entries: frozenset[str] = frozenset()
    manifest: dict[str, object] | None = None
    try:
        trust_parent_descriptor = _open_directory(
            trust_root.parent,
            label="public trust parent",
        )
        try:
            fcntl.flock(
                trust_parent_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except OSError as exc:
            raise BootstrapKeyProvisioningError(
                "another public trust provisioning transaction is active"
            ) from exc
        trust_descriptor = _open_child_directory(
            trust_parent_descriptor,
            trust_root.name,
            label="public trust directory",
        )
        old_trust_metadata = os.fstat(trust_descriptor)
        old_trust_identity = (old_trust_metadata.st_dev, old_trust_metadata.st_ino)
        if (
            old_trust_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(old_trust_metadata.st_mode) & 0o022
        ):
            raise BootstrapKeyProvisioningError("public trust directory permissions are unsafe")
        old_trust_entries = frozenset(os.listdir(trust_descriptor))
        if old_trust_entries:
            _validate_complete_trust_set(trust_descriptor)

        capability_parent_descriptor = _open_directory(
            capability.parent.resolve(strict=True),
            label="capability parent",
        )
        try:
            os.mkdir(
                capability.name,
                0o700,
                dir_fd=capability_parent_descriptor,
            )
        except OSError as exc:
            raise BootstrapKeyProvisioningError(
                "capability directory must be new and exclusively creatable"
            ) from exc
        capability_created = True
        capability_metadata = os.stat(
            capability.name,
            dir_fd=capability_parent_descriptor,
            follow_symlinks=False,
        )
        capability_identity = (capability_metadata.st_dev, capability_metadata.st_ino)
        if (
            not stat.S_ISDIR(capability_metadata.st_mode)
            or capability_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(capability_metadata.st_mode) != 0o700
        ):
            raise BootstrapKeyProvisioningError("capability directory is not owner-only")
        capability_descriptor = _open_child_directory(
            capability_parent_descriptor,
            capability.name,
            label="capability directory",
        )
        capability_metadata = os.fstat(capability_descriptor)
        if (capability_metadata.st_dev, capability_metadata.st_ino) != capability_identity:
            raise BootstrapKeyProvisioningError("capability directory identity changed")
        os.fsync(capability_parent_descriptor)
        _provisioning_fault("AFTER_CAPABILITY_DIRECTORY_CREATED")

        public_payloads: dict[str, bytes] = {}
        fingerprints: dict[str, str] = {}
        for kind, (private_filename, _public_filename, _producer_id) in _KINDS.items():
            key = Ed25519PrivateKey.generate()
            private_payload = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            _write_new_private_key(
                capability_descriptor,
                private_filename,
                private_payload,
            )
            public_payloads[kind] = key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            fingerprints[kind] = ed25519_public_key_fingerprint(key.public_key())
        if len(set(fingerprints.values())) != len(_KINDS):
            raise BootstrapKeyProvisioningError("generated signing capabilities are not distinct")
        os.fsync(capability_descriptor)
        os.fsync(capability_parent_descriptor)
        _provisioning_fault("AFTER_PRIVATE_KEYS_WRITTEN")

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

        for _attempt in range(32):
            candidate = f".{trust_root.name}.provision-{os.getpid()}-{secrets.token_hex(8)}"
            try:
                os.mkdir(
                    candidate,
                    stat.S_IMODE(old_trust_metadata.st_mode),
                    dir_fd=trust_parent_descriptor,
                )
            except FileExistsError:
                continue
            stage_name = candidate
            break
        if stage_name is None:
            raise BootstrapKeyProvisioningError(
                "public trust staging directory could not be allocated"
            )
        stage_metadata = os.stat(
            stage_name,
            dir_fd=trust_parent_descriptor,
            follow_symlinks=False,
        )
        stage_identity = (stage_metadata.st_dev, stage_metadata.st_ino)
        if not stat.S_ISDIR(stage_metadata.st_mode) or stage_metadata.st_uid != os.geteuid():
            raise BootstrapKeyProvisioningError("public trust staging owner is unsafe")
        stage_descriptor = _open_child_directory(
            trust_parent_descriptor,
            stage_name,
            label="public trust staging directory",
        )
        stage_metadata = os.fstat(stage_descriptor)
        if (stage_metadata.st_dev, stage_metadata.st_ino) != stage_identity:
            raise BootstrapKeyProvisioningError("public trust staging identity changed")
        _provisioning_fault("AFTER_TRUST_STAGE_CREATED")

        for kind, (_private_filename, public_filename, _producer_id) in _KINDS.items():
            _write_new_public_file(
                stage_descriptor,
                public_filename,
                public_payloads[kind],
            )
        _write_new_public_file(
            stage_descriptor,
            "manifest.json",
            (
                json.dumps(
                    manifest,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8"),
        )
        _provisioning_fault("AFTER_TRUST_STAGE_WRITTEN")
        os.fsync(stage_descriptor)
        os.fsync(trust_parent_descriptor)
        _provisioning_fault("AFTER_TRUST_STAGE_SYNCED")
        _validate_complete_trust_set(stage_descriptor)
        _provisioning_fault("AFTER_TRUST_STAGE_VALIDATED")
        _provisioning_fault("BEFORE_TRUST_EXCHANGE")

        _atomic_exchange_directories(
            trust_parent_descriptor,
            stage_name,
            trust_root.name,
        )
        exchanged = True
        _provisioning_fault("AFTER_TRUST_EXCHANGE")
        os.fsync(trust_parent_descriptor)
        _provisioning_fault("AFTER_PARENT_SYNC")

        live_metadata = os.stat(
            trust_root.name,
            dir_fd=trust_parent_descriptor,
            follow_symlinks=False,
        )
        if (live_metadata.st_dev, live_metadata.st_ino) != stage_identity:
            raise BootstrapKeyProvisioningError(
                "public trust path does not identify the validated staged set"
            )
        _validate_complete_trust_set(stage_descriptor)
        current_capability = os.stat(
            capability.name,
            dir_fd=capability_parent_descriptor,
            follow_symlinks=False,
        )
        if (current_capability.st_dev, current_capability.st_ino) != capability_identity:
            raise BootstrapKeyProvisioningError(
                "private capability path changed during provisioning"
            )
        _provisioning_fault("AFTER_LIVE_VALIDATION")
        committed = True

        if trust_descriptor is not None:
            os.close(trust_descriptor)
            trust_descriptor = None
        retired_removed = _remove_exact_directory(
            trust_parent_descriptor,
            stage_name,
            _TRUST_FILENAMES,
            expected_identity=old_trust_identity,
        )
        if not retired_removed:
            raise BootstrapKeyProvisioningError(
                "public trust committed but retired trust cleanup failed"
            )
        return {
            "authorizes_startup": False,
            "private_keys_exported": False,
            "schema_version": 1,
            "status": "PROVISIONED_REVIEW_AND_COMMIT_PUBLIC_TRUST_ROOTS",
            "trust_set_digest": manifest["trust_set_digest"],
        }
    except BaseException as exc:
        rollback_error: BaseException | None = None
        if exchanged and not committed:
            try:
                assert trust_parent_descriptor is not None
                assert stage_name is not None
                assert old_trust_identity is not None
                _atomic_exchange_directories(
                    trust_parent_descriptor,
                    stage_name,
                    trust_root.name,
                )
                os.fsync(trust_parent_descriptor)
                restored_metadata = os.stat(
                    trust_root.name,
                    dir_fd=trust_parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    restored_metadata.st_dev,
                    restored_metadata.st_ino,
                ) != old_trust_identity:
                    raise BootstrapKeyProvisioningError(
                        "public trust rollback restored the wrong directory"
                    )
                if old_trust_entries:
                    assert trust_descriptor is not None
                    _validate_complete_trust_set(trust_descriptor)
                exchanged = False
            except BaseException as rollback_exc:
                rollback_error = rollback_exc

        safe_to_revoke = not exchanged and not committed and rollback_error is None
        cleanup_failed = False
        if safe_to_revoke and stage_name is not None and stage_identity is not None:
            if stage_descriptor is not None:
                os.close(stage_descriptor)
                stage_descriptor = None
            assert trust_parent_descriptor is not None
            cleanup_failed = not _remove_exact_directory(
                trust_parent_descriptor,
                stage_name,
                _TRUST_FILENAMES,
                expected_identity=stage_identity,
            )
        if safe_to_revoke and capability_created and capability_identity is not None:
            if capability_descriptor is not None:
                os.close(capability_descriptor)
                capability_descriptor = None
            assert capability_parent_descriptor is not None
            cleanup_failed = (
                not _remove_exact_directory(
                    capability_parent_descriptor,
                    capability.name,
                    frozenset(_PRIVATE_FILENAMES),
                    expected_identity=capability_identity,
                )
                or cleanup_failed
            )
        if rollback_error is not None:
            raise BootstrapKeyProvisioningError(
                "public trust rollback failed; transaction state requires recovery"
            ) from rollback_error
        if cleanup_failed:
            raise BootstrapKeyProvisioningError(
                "provisioning failed with old trust preserved, but capability cleanup failed"
            ) from exc
        raise
    finally:
        for descriptor in (
            stage_descriptor,
            capability_descriptor,
            capability_parent_descriptor,
            trust_descriptor,
            trust_parent_descriptor,
        ):
            if descriptor is not None:
                os.close(descriptor)


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

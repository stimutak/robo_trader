"""Descriptor-bound SQLite backup, verification, and clean-room restore.

This module is deliberately dormant.  It has no runtime imports, broker access,
startup authority, or operation that replaces an existing database path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from robo_trader.maintenance.models import (
    DatabaseEvidence,
    MaintenanceManifest,
    MigrationDryRunReport,
    MigrationPlan,
    MigrationStep,
    TableEvidence,
)
from robo_trader.safety.sqlite_identity import (
    SQLiteIdentityError,
    SQLitePathBinding,
    lexical_path_preserving_leaf,
    sqlite_connection_file_identity,
)

ProgressHook = Callable[[str, int, int], None]


class SQLiteMaintenanceError(RuntimeError):
    """A maintenance operation failed closed."""


@dataclass(slots=True)
class _TargetReservation:
    """Exclusive reservation for a new database and all SQLite sidecar names."""

    binding: SQLitePathBinding
    companions: dict[Path, tuple[int, int, int]]

    def release(self) -> None:
        error: SQLiteMaintenanceError | None = None
        for path, (descriptor, device, inode) in self.companions.items():
            try:
                descriptor_metadata = os.fstat(descriptor)
                path_metadata = os.lstat(path)
                if (
                    not stat.S_ISREG(descriptor_metadata.st_mode)
                    or descriptor_metadata.st_nlink != 1
                    or descriptor_metadata.st_size != 0
                    or (descriptor_metadata.st_dev, descriptor_metadata.st_ino) != (device, inode)
                    or (path_metadata.st_dev, path_metadata.st_ino) != (device, inode)
                ):
                    raise SQLiteMaintenanceError(
                        "target SQLite sidecar reservation changed during operation"
                    )
                os.unlink(path)
            except (OSError, SQLiteMaintenanceError) as exc:
                error = SQLiteMaintenanceError(
                    "target SQLite sidecar reservation could not be released safely"
                )
                error.__cause__ = exc
            finally:
                os.close(descriptor)
        self.companions.clear()
        if error is not None:
            raise error


class SQLiteMaintenanceService:
    """Operate only on explicit, descriptor-bound SQLite files."""

    def __init__(
        self,
        *,
        progress_hook: ProgressHook | None = None,
        max_copy_seconds: float = 120.0,
    ) -> None:
        if not math.isfinite(max_copy_seconds) or not 1.0 <= max_copy_seconds <= 3600.0:
            raise ValueError("max_copy_seconds must be between 1 and 3600")
        self._progress_hook = progress_hook
        self._max_copy_seconds = max_copy_seconds

    def verify(
        self,
        database_path: Path | str,
        manifest: MaintenanceManifest | None = None,
    ) -> DatabaseEvidence:
        """Verify one existing database without mutating it."""

        database_candidate = _absolute_canonical_path(database_path)
        companions = _safe_companion_identities(database_candidate)
        if companions:
            raise SQLiteMaintenanceError(
                "read-only verification requires a sealed database without SQLite companions"
            )
        binding = self._open_source(database_candidate)
        connection: sqlite3.Connection | None = None
        try:
            connection, binding = self._connect_bound(
                binding, readonly=True, immutable_readonly=True
            )
            evidence = _database_evidence(connection)
            binding.assert_connection_identity(sqlite_connection_file_identity(connection))
            artifact_sha256, artifact_size = _descriptor_digest(binding.guardian_file_descriptor)
            binding.assert_path_identity()
            _assert_companion_identities(binding.path, companions)
            if manifest is not None:
                if (
                    manifest.artifact_sha256 != artifact_sha256
                    or manifest.artifact_size != artifact_size
                    or manifest.evidence != evidence
                ):
                    raise SQLiteMaintenanceError("database does not match the supplied manifest")
            return evidence
        except (sqlite3.Error, SQLiteIdentityError, OSError) as exc:
            raise SQLiteMaintenanceError("SQLite verification failed") from exc
        finally:
            if connection is not None:
                connection.close()
            binding.close()

    def backup(
        self,
        source_path: Path | str,
        target_path: Path | str,
    ) -> MaintenanceManifest:
        """Create and verify a WAL-safe snapshot at an exclusive new path."""

        return self._online_copy(source_path, target_path, operation="backup")

    def restore_clean_room(
        self,
        backup_path: Path | str,
        target_path: Path | str,
        manifest: MaintenanceManifest,
    ) -> MaintenanceManifest:
        """Restore a verified backup only into an exclusive clean-room path."""

        if manifest.operation != "backup":
            raise SQLiteMaintenanceError("clean-room restore requires a backup manifest")
        restored = self._online_copy(
            backup_path,
            target_path,
            operation="restore",
            expected_source_manifest=manifest,
        )
        if restored.evidence != manifest.evidence:
            raise SQLiteMaintenanceError("clean-room restore evidence differs from backup")
        return MaintenanceManifest(
            manifest_version=restored.manifest_version,
            operation="restore",
            created_at=restored.created_at,
            artifact_sha256=restored.artifact_sha256,
            artifact_size=restored.artifact_size,
            evidence=restored.evidence,
            input_artifact_sha256=manifest.artifact_sha256,
        )

    def dry_run_migration(
        self,
        source_path: Path | str,
        synthetic_target_path: Path | str,
        *,
        plan: MigrationPlan,
    ) -> MigrationDryRunReport:
        """Apply a declarative plan transactionally to a synthetic snapshot.

        Callers never receive the SQLite connection. Transaction, ATTACH, and
        DETACH statements are denied so the service retains the rollback
        boundary and the plan cannot reach another database.
        """

        _validate_migration_plan(plan)
        source_before = self.verify(source_path)
        source_before_hash, source_before_size = _path_digest(_absolute_canonical_path(source_path))
        source_manifest = MaintenanceManifest(
            manifest_version=1,
            operation="backup",
            created_at=_utc_now(),
            artifact_sha256=source_before_hash,
            artifact_size=source_before_size,
            evidence=source_before,
        )
        self._online_copy(
            source_path,
            synthetic_target_path,
            operation="backup",
            seal=False,
            expected_source_manifest=source_manifest,
        )
        target = self._open_source(synthetic_target_path, writable=True)
        connection: sqlite3.Connection | None = None
        outcome = "applied_to_synthetic_copy"
        error_code: str | None = None
        try:
            connection, target = self._connect_bound(target, readonly=False)
            before = _database_evidence(connection)
            connection.execute("BEGIN IMMEDIATE")
            connection.set_authorizer(_migration_authorizer)
            try:
                for step in plan.steps:
                    connection.execute(step.sql, step.parameters)
                if plan.target_user_version is not None:
                    connection.execute(f"PRAGMA user_version={plan.target_user_version}")
                interim = _database_evidence(connection)
                if interim.integrity_check != "ok" or interim.foreign_key_violations:
                    raise SQLiteMaintenanceError("migration produced invalid SQLite state")
            except BaseException:
                connection.set_authorizer(None)
                connection.rollback()
                outcome = "rolled_back"
                error_code = "migration_plan_failed"
            else:
                connection.set_authorizer(None)
                connection.commit()
            target.assert_connection_identity(sqlite_connection_file_identity(connection))
        finally:
            if connection is not None:
                connection.close()
            try:
                _seal_readonly(target)
                _fsync_directory(target.path.parent)
            finally:
                target.close()

        after = self.verify(synthetic_target_path)
        source_after = self.verify(source_path)
        source_after_hash, _ = _path_digest(_absolute_canonical_path(source_path))
        source_unchanged = source_before == source_after and source_before_hash == source_after_hash
        if not source_unchanged:
            outcome = "source_changed_fail_closed"
            error_code = "authoritative_source_changed"
        artifact_sha256, _ = _path_digest(_absolute_canonical_path(synthetic_target_path))
        return MigrationDryRunReport(
            report_version=1,
            migration_id=plan.migration_id,
            created_at=_utc_now(),
            outcome=outcome,
            before=before,
            after=after,
            source_unchanged=source_unchanged,
            target_artifact_sha256=artifact_sha256,
            error_code=error_code,
        )

    def write_manifest(
        self,
        manifest: MaintenanceManifest | MigrationDryRunReport,
        target_path: Path | str,
    ) -> None:
        """Write one canonical JSON report to an exclusive owner-only path."""

        path = _absolute_canonical_path(target_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as exc:
            raise SQLiteMaintenanceError("manifest target must be a new exclusive path") from exc
        try:
            payload = (
                json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
                + b"\n"
            )
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            _fsync_directory(path.parent)
            metadata = os.fstat(descriptor)
            current = os.lstat(path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise SQLiteMaintenanceError("manifest identity changed while writing")
        except OSError as exc:
            raise SQLiteMaintenanceError("manifest write failed closed") from exc
        finally:
            os.close(descriptor)

    def load_manifest(self, path: Path | str) -> MaintenanceManifest:
        """Load a bounded, single-link manifest without following its leaf."""

        manifest_path = _absolute_canonical_path(path)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(manifest_path, flags)
        except OSError as exc:
            raise SQLiteMaintenanceError("manifest cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            current = os.lstat(manifest_path)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_size > 4 * 1024 * 1024
                or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise SQLiteMaintenanceError("manifest must be a bounded single-link file")
            payload = _read_exact(descriptor, metadata.st_size)
            after = os.fstat(descriptor)
            current_after = os.lstat(manifest_path)
            stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(metadata, field) != getattr(after, field) for field in stable) or (
                after.st_dev,
                after.st_ino,
            ) != (current_after.st_dev, current_after.st_ino):
                raise SQLiteMaintenanceError("manifest changed while reading")
            raw = json.loads(payload.decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise SQLiteMaintenanceError("manifest must contain a JSON object")
            return MaintenanceManifest.from_mapping(raw)
        except (UnicodeError, json.JSONDecodeError, ValueError, OSError) as exc:
            raise SQLiteMaintenanceError("manifest is invalid") from exc
        finally:
            os.close(descriptor)

    def _online_copy(
        self,
        source_path: Path | str,
        target_path: Path | str,
        *,
        operation: str,
        seal: bool = True,
        expected_source_manifest: MaintenanceManifest | None = None,
    ) -> MaintenanceManifest:
        source_candidate = _absolute_canonical_path(source_path)
        target_candidate = _absolute_canonical_path(target_path)
        if _sqlite_resource_family(source_candidate) & _sqlite_resource_family(target_candidate):
            raise SQLiteMaintenanceError("source and target SQLite resource families overlap")
        source_companions = _safe_companion_identities(source_candidate)
        source = self._open_source(source_candidate)
        source_connection: sqlite3.Connection | None = None
        try:
            source_connection, source = self._connect_bound(
                source,
                readonly=True,
                immutable_readonly=expected_source_manifest is not None,
            )
            source_connection.execute("BEGIN")
            source_evidence = _database_evidence(source_connection)
            source_artifact_sha256, source_artifact_size = _descriptor_digest(
                source.guardian_file_descriptor
            )
            if expected_source_manifest is not None and (
                expected_source_manifest.artifact_sha256 != source_artifact_sha256
                or expected_source_manifest.artifact_size != source_artifact_size
                or expected_source_manifest.evidence != source_evidence
            ):
                raise SQLiteMaintenanceError("copy source does not match the supplied manifest")
            reservation = self._reserve_target(target_candidate)
            target = reservation.binding
        except (sqlite3.Error, SQLiteIdentityError, OSError) as exc:
            if source_connection is not None:
                source_connection.close()
            source.close()
            raise SQLiteMaintenanceError("copy source verification failed closed") from exc
        except BaseException:
            if source_connection is not None:
                source_connection.close()
            source.close()
            raise
        target_connection: sqlite3.Connection | None = None
        succeeded = False
        try:
            if source_connection is None:
                raise SQLiteMaintenanceError("copy source connection is unavailable")
            if (source.device, source.inode) == (target.device, target.inode):
                raise SQLiteMaintenanceError("source and target identities must differ")
            target_connection, target = self._connect_bound(
                target, readonly=False, journal_off=True
            )
            source_db = source_connection
            target_db = target_connection
            copy_started = time.monotonic()

            def progress(_status: int, remaining: int, total: int) -> None:
                if time.monotonic() - copy_started > self._max_copy_seconds:
                    raise SQLiteMaintenanceError("SQLite online copy exceeded its deadline")
                source.assert_connection_identity(sqlite_connection_file_identity(source_db))
                target.assert_connection_identity(sqlite_connection_file_identity(target_db))
                _assert_single_link(source)
                _assert_single_link(target)
                _assert_companion_identities(source.path, source_companions)
                if self._progress_hook is not None:
                    self._progress_hook(operation, remaining, total)

            source_connection.backup(target_connection, pages=64, progress=progress, sleep=0.001)
            target_connection.commit()
            source.assert_connection_identity(sqlite_connection_file_identity(source_connection))
            target.assert_connection_identity(sqlite_connection_file_identity(target_connection))
            _assert_single_link(source)
            _assert_single_link(target)
            _assert_companion_identities(source.path, source_companions)
            evidence = _database_evidence(target_connection)
            final_source_sha256, final_source_size = _descriptor_digest(
                source.guardian_file_descriptor
            )
            if (
                final_source_sha256 != source_artifact_sha256
                or final_source_size != source_artifact_size
            ):
                raise SQLiteMaintenanceError("copy source changed during operation")
            if (
                expected_source_manifest is not None
                and evidence != expected_source_manifest.evidence
            ):
                raise SQLiteMaintenanceError("copied evidence differs from supplied manifest")
            target_connection.close()
            target_connection = None
            if seal:
                _seal_readonly(target)
            else:
                os.fsync(target.guardian_file_descriptor)
                _assert_single_link(target)
            _fsync_directory(target.path.parent)
            artifact_sha256, artifact_size = _descriptor_digest(target.guardian_file_descriptor)
            target.assert_path_identity()
            manifest = MaintenanceManifest(
                manifest_version=1,
                operation=operation,
                created_at=_utc_now(),
                artifact_sha256=artifact_sha256,
                artifact_size=artifact_size,
                evidence=evidence,
            )
            reservation.release()
            if _safe_companion_identities(target.path):
                raise SQLiteMaintenanceError(
                    "target SQLite companions appeared after reservation release"
                )
            _fsync_directory(target.path.parent)
            succeeded = True
            return manifest
        except (sqlite3.Error, SQLiteIdentityError, OSError) as exc:
            raise SQLiteMaintenanceError("SQLite online copy failed closed") from exc
        finally:
            if target_connection is not None:
                target_connection.close()
            if source_connection is not None:
                source_connection.close()
            try:
                if not succeeded:
                    _seal_forensic_target(target)
                if reservation.companions:
                    reservation.release()
            finally:
                source.close()
                target.close()

    @staticmethod
    def _open_source(
        source_path: Path | str,
        *,
        writable: bool = False,
    ) -> SQLitePathBinding:
        path = _absolute_canonical_path(source_path)
        try:
            metadata = os.lstat(path)
        except OSError as exc:
            raise SQLiteMaintenanceError("database source does not exist") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise SQLiteMaintenanceError("database source must be a single-link regular file")
        try:
            binding = SQLitePathBinding.open_readonly(path)
        except SQLiteIdentityError as exc:
            raise SQLiteMaintenanceError("database identity cannot be established") from exc
        if writable and not os.access(path, os.W_OK):
            binding.close()
            raise SQLiteMaintenanceError("synthetic migration target must be writable")
        return binding

    @staticmethod
    def _reserve_target(target_path: Path | str) -> _TargetReservation:
        path = _absolute_canonical_path(target_path)
        try:
            parent = os.lstat(path.parent)
        except OSError as exc:
            raise SQLiteMaintenanceError("target parent does not exist") from exc
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise SQLiteMaintenanceError("target parent must be a non-symlink directory")
        family = _sqlite_resource_family(path)
        for member in family:
            try:
                os.lstat(member)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise SQLiteMaintenanceError("target SQLite family cannot be inspected") from exc
            raise SQLiteMaintenanceError("target must be a new exclusively-created SQLite family")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        companions: dict[Path, tuple[int, int, int]] = {}
        binding: SQLitePathBinding | None = None
        try:
            for suffix in ("-journal", "-shm", "-wal"):
                companion = path.with_name(path.name + suffix)
                descriptor = os.open(companion, flags, 0o400)
                metadata = os.fstat(descriptor)
                companions[companion] = (descriptor, metadata.st_dev, metadata.st_ino)
            binding = SQLitePathBinding.open_for_initialization(path, create=True)
            return _TargetReservation(binding=binding, companions=companions)
        except Exception as exc:
            if binding is not None:
                binding.close()
            reservation = _TargetReservation(binding=binding, companions=companions)  # type: ignore[arg-type]
            try:
                reservation.release()
            except SQLiteMaintenanceError:
                pass
            raise SQLiteMaintenanceError(
                "target must be a new exclusively-created SQLite family"
            ) from exc

    @staticmethod
    def _connect_bound(
        binding: SQLitePathBinding,
        *,
        readonly: bool,
        journal_off: bool = False,
        immutable_readonly: bool = False,
    ) -> tuple[sqlite3.Connection, SQLitePathBinding]:
        mode = "ro" if readonly else "rw"
        immutable = "&immutable=1" if immutable_readonly else ""
        connection = sqlite3.connect(
            binding.path.as_uri() + f"?mode={mode}{immutable}",
            uri=True,
            timeout=10.0,
        )
        try:
            binding = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
            binding.assert_connection_identity(sqlite_connection_file_identity(connection))
            if readonly:
                connection.execute("PRAGMA query_only=ON")
            if journal_off:
                mode_row = connection.execute("PRAGMA journal_mode=OFF").fetchone()
                if mode_row is None or str(mode_row[0]).lower() != "off":
                    raise SQLiteMaintenanceError("copy target journaling could not be disabled")
            connection.execute("PRAGMA foreign_keys=ON")
            return connection, binding
        except BaseException:
            connection.close()
            raise


def _absolute_canonical_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise SQLiteMaintenanceError("maintenance paths must be absolute")
    protected = lexical_path_preserving_leaf(candidate)
    if protected != candidate:
        raise SQLiteMaintenanceError("maintenance path parent must be canonical")
    return protected


def _validate_migration_plan(plan: MigrationPlan) -> None:
    if type(plan) is not MigrationPlan or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", plan.migration_id
    ):
        raise SQLiteMaintenanceError("migration plan has an invalid public identifier")
    if type(plan.steps) is not tuple or not 1 <= len(plan.steps) <= 256:
        raise SQLiteMaintenanceError("migration plan must contain 1-256 ordered steps")
    for step in plan.steps:
        if (
            type(step) is not MigrationStep
            or type(step.sql) is not str
            or not step.sql.strip()
            or len(step.sql.encode("utf-8")) > 128 * 1024
            or type(step.parameters) is not tuple
            or len(step.parameters) > 1024
        ):
            raise SQLiteMaintenanceError("migration plan contains an invalid step")
        for parameter in step.parameters:
            if type(parameter) not in (str, int, float, bytes, type(None)):
                raise SQLiteMaintenanceError("migration step parameter type is unsupported")
            if isinstance(parameter, float) and not math.isfinite(parameter):
                raise SQLiteMaintenanceError("migration step parameter must be finite")
            if isinstance(parameter, (str, bytes)) and len(parameter) > 4 * 1024 * 1024:
                raise SQLiteMaintenanceError("migration step parameter is too large")
    if plan.target_user_version is not None and (
        isinstance(plan.target_user_version, bool)
        or not isinstance(plan.target_user_version, int)
        or not 0 <= plan.target_user_version <= 2_147_483_647
    ):
        raise SQLiteMaintenanceError("migration target user version is invalid")


def _database_evidence(connection: sqlite3.Connection) -> DatabaseEvidence:
    quick = _single_check(connection, "quick_check")
    integrity = _single_check(connection, "integrity_check")
    if quick != "ok" or integrity != "ok":
        raise SQLiteMaintenanceError("database failed SQLite integrity verification")
    foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
    if foreign_key_rows:
        raise SQLiteMaintenanceError("database contains foreign-key violations")

    schema_rows = connection.execute(
        "SELECT type,name,tbl_name,COALESCE(sql,'') FROM sqlite_master "
        "ORDER BY type,name,tbl_name,COALESCE(sql,'')"
    ).fetchall()
    schema_hash = _hash_rows(schema_rows)
    table_names = [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND (name NOT LIKE 'sqlite_%' OR name='sqlite_sequence') "
            "ORDER BY name"
        ).fetchall()
    ]
    tables = tuple(_table_evidence(connection, name) for name in table_names)
    application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    return DatabaseEvidence(
        schema_sha256=schema_hash,
        tables=tables,
        quick_check=quick,
        integrity_check=integrity,
        foreign_key_violations=0,
        application_id=application_id,
        user_version=user_version,
    )


def _single_check(connection: sqlite3.Connection, pragma: str) -> str:
    rows = connection.execute(f"PRAGMA {pragma}").fetchall()
    if rows == [("ok",)]:
        return "ok"
    return "failed"


def _table_evidence(connection: sqlite3.Connection, table_name: str) -> TableEvidence:
    quoted = '"' + table_name.replace('"', '""') + '"'
    # The identifier comes only from sqlite_master and is escaped as a quoted
    # SQLite identifier; DB-API parameters cannot represent identifiers.
    cursor = connection.execute(f"SELECT * FROM {quoted}")  # nosec B608
    row_hashes: list[bytes] = []
    row_count = 0
    for row in cursor:
        row_hashes.append(hashlib.sha256(_encode_row(row)).digest())
        row_count += 1
    row_hashes.sort()
    digest = hashlib.sha256()
    for row_hash in row_hashes:
        digest.update(row_hash)
    return TableEvidence(
        hashlib.sha256(table_name.encode("utf-8")).hexdigest(),
        row_count,
        digest.hexdigest(),
    )


def _hash_rows(rows: Iterable[Sequence[object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        item = _encode_row(row)
        digest.update(len(item).to_bytes(8, "big"))
        digest.update(item)
    return digest.hexdigest()


def _encode_row(row: Sequence[object]) -> bytes:
    return json.dumps(
        [_encode_value(value) for value in row],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _encode_value(value: object) -> object:
    if value is None or isinstance(value, (str, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SQLiteMaintenanceError("database contains a non-finite numeric value")
        return ["float", value.hex()]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    raise SQLiteMaintenanceError("database contains an unsupported SQLite value")


def _migration_authorizer(
    action: int,
    arg1: str | None,
    _arg2: str | None,
    _db: str | None,
    _trigger: str | None,
) -> int:
    denied = {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_TRANSACTION,
    }
    dangerous_pragmas = {
        "data_store_directory",
        "journal_mode",
        "temp_store_directory",
        "wal_checkpoint",
        "writable_schema",
    }
    if action in denied or (
        action == sqlite3.SQLITE_PRAGMA
        and isinstance(arg1, str)
        and arg1.lower() in dangerous_pragmas
    ):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _assert_single_link(binding: SQLitePathBinding) -> None:
    metadata = os.fstat(binding.guardian_file_descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise SQLiteMaintenanceError("database acquired another filesystem link")
    try:
        binding.assert_path_identity()
    except SQLiteIdentityError as exc:
        raise SQLiteMaintenanceError("database path identity changed") from exc


def _seal_readonly(binding: SQLitePathBinding) -> None:
    _assert_single_link(binding)
    os.fsync(binding.guardian_file_descriptor)
    os.fchmod(binding.guardian_file_descriptor, 0o400)
    os.fsync(binding.guardian_file_descriptor)
    _assert_single_link(binding)


def _seal_forensic_target(binding: SQLitePathBinding) -> None:
    try:
        os.fsync(binding.guardian_file_descriptor)
    except OSError:
        pass
    try:
        os.fchmod(binding.guardian_file_descriptor, 0o400)
    except OSError:
        pass
    try:
        os.fsync(binding.guardian_file_descriptor)
        _fsync_directory(binding.path.parent)
    except OSError:
        pass


def _sqlite_resource_family(path: Path) -> frozenset[Path]:
    return frozenset(
        {
            path,
            path.with_name(path.name + "-journal"),
            path.with_name(path.name + "-shm"),
            path.with_name(path.name + "-wal"),
        }
    )


def _safe_companion_identities(path: Path) -> dict[str, tuple[int, int]]:
    identities: dict[str, tuple[int, int]] = {}
    for suffix in ("-journal", "-shm", "-wal"):
        companion = path.with_name(path.name + suffix)
        try:
            metadata = os.lstat(companion)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise SQLiteMaintenanceError("SQLite companion cannot be inspected") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise SQLiteMaintenanceError(
                "SQLite companions must be single-link non-symlink regular files"
            )
        identities[suffix] = (metadata.st_dev, metadata.st_ino)
    return identities


def _assert_companion_identities(
    path: Path,
    expected: Mapping[str, tuple[int, int]],
) -> None:
    current = _safe_companion_identities(path)
    if current != expected:
        raise SQLiteMaintenanceError("SQLite companion identity changed during operation")


def _descriptor_digest(descriptor: int) -> tuple[str, int]:
    metadata = os.fstat(descriptor)
    digest = hashlib.sha256()
    offset = 0
    while offset < metadata.st_size:
        chunk = os.pread(descriptor, min(128 * 1024, metadata.st_size - offset), offset)
        if not chunk:
            raise SQLiteMaintenanceError("database ended while hashing")
        digest.update(chunk)
        offset += len(chunk)
    after = os.fstat(descriptor)
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(metadata, field) != getattr(after, field) for field in stable):
        raise SQLiteMaintenanceError("database changed while hashing")
    return digest.hexdigest(), metadata.st_size


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_digest(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        return _descriptor_digest(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise SQLiteMaintenanceError("manifest write did not make progress")
        offset += written


def _read_exact(descriptor: int, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(128 * 1024, remaining))
        if not chunk:
            raise SQLiteMaintenanceError("manifest ended while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise SQLiteMaintenanceError("manifest grew while reading")
    return b"".join(chunks)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robo_trader.maintenance import (
    MigrationPlan,
    MigrationStep,
    SQLiteMaintenanceError,
    SQLiteMaintenanceService,
)
from robo_trader.maintenance import sqlite_service as sqlite_service_module
from robo_trader.multiuser.migration import (
    LegacyMultiuserMigrationDisabled,
    MultiuserMigration,
)


def _create_multiportfolio_database(path: Path, *, wal: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    if wal:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE portfolios (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE positions (
            portfolio_id TEXT NOT NULL REFERENCES portfolios(id),
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            PRIMARY KEY (portfolio_id, symbol)
        );
        INSERT INTO portfolios VALUES ('alpha', 'Alpha'), ('beta', 'Beta');
        INSERT INTO positions VALUES
            ('alpha', 'AAPL', 3),
            ('beta', 'AAPL', 7),
            ('beta', 'MSFT', 2);
        PRAGMA user_version=7;
        """)
    connection.commit()
    return connection


def test_wal_active_backup_and_clean_room_restore_preserve_multiportfolio_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synthetic-source.db"
    backup = tmp_path / "new-backup.db"
    restored = tmp_path / "new-restore.db"
    writer = _create_multiportfolio_database(source, wal=True)
    try:
        writer.execute("INSERT INTO positions VALUES ('alpha', 'NVDA', 4)")
        writer.commit()
        assert source.with_name(source.name + "-wal").stat().st_size > 0

        service = SQLiteMaintenanceService()
        manifest = service.backup(source, backup)
        restore_manifest = service.restore_clean_room(backup, restored, manifest)

        assert manifest.operation == "backup"
        assert restore_manifest.operation == "restore"
        assert restore_manifest.input_artifact_sha256 == manifest.artifact_sha256
        assert manifest.evidence == restore_manifest.evidence
        assert manifest.authorizes_startup is False
        assert stat.S_IMODE(backup.stat().st_mode) == 0o400
        assert stat.S_IMODE(restored.stat().st_mode) == 0o400
        for database in (backup, restored):
            assert not any(
                database.with_name(database.name + suffix).exists()
                for suffix in ("-journal", "-shm", "-wal")
            )
        with sqlite3.connect(restored) as connection:
            assert connection.execute(
                "SELECT portfolio_id,symbol,quantity FROM positions " "ORDER BY portfolio_id,symbol"
            ).fetchall() == [
                ("alpha", "AAPL", 3),
                ("alpha", "NVDA", 4),
                ("beta", "AAPL", 7),
                ("beta", "MSFT", 2),
            ]
    finally:
        writer.close()


def test_wal_checkpoint_during_online_backup_is_not_treated_as_source_mutation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "backup.db"
    writer = _create_multiportfolio_database(source, wal=True)
    writer.execute("INSERT INTO positions VALUES ('alpha', 'NVDA', 4)")
    writer.commit()
    main_before_checkpoint = source.read_bytes()
    checkpointed = False

    def checkpoint(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal checkpointed
        if not checkpointed:
            checkpointed = True
            writer.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()

    try:
        manifest = SQLiteMaintenanceService(progress_hook=checkpoint).backup(source, target)
    finally:
        writer.close()

    assert checkpointed
    assert source.read_bytes() != main_before_checkpoint
    assert SQLiteMaintenanceService().verify(target) == manifest.evidence


def test_online_copy_rejects_oversized_source_before_materializing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "oversized.db"
    target = tmp_path / "backup.db"
    progress_events: list[tuple[str, int, int]] = []
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE payloads (value BLOB NOT NULL)")
        connection.execute("INSERT INTO payloads(value) VALUES (zeroblob(2000000))")

    with pytest.raises(SQLiteMaintenanceError, match="artifact size limit"):
        SQLiteMaintenanceService(
            max_source_bytes=1024 * 1024,
            progress_hook=lambda *event: progress_events.append(event),
        ).backup(source, target)

    assert progress_events == []
    assert not target.exists()


@pytest.mark.parametrize(
    "invalid_limit",
    [True, 1024 * 1024 - 1, 1024 * 1024 * 1024 + 1],
)
def test_online_copy_source_size_limit_is_bounded(invalid_limit: object) -> None:
    with pytest.raises(ValueError, match="max_source_bytes"):
        SQLiteMaintenanceService(max_source_bytes=invalid_limit)  # type: ignore[arg-type]


def test_cleanly_closed_wal_source_accounts_for_connection_sidecars(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    synthetic = tmp_path / "synthetic.db"
    writer = _create_multiportfolio_database(source, wal=True)
    writer.close()
    companions = tuple(source.with_name(source.name + suffix) for suffix in ("-wal", "-shm"))
    assert not any(path.exists() for path in companions)

    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    report = service.dry_run_migration(
        source,
        synthetic,
        plan=MigrationPlan(
            migration_id="clean-wal-noop",
            steps=(MigrationStep(sql="SELECT 1"),),
        ),
    )

    assert service.verify(backup) == manifest.evidence
    assert report.source_unchanged is True
    assert report.outcome == "applied_to_synthetic_copy"
    for companion in companions:
        if companion.exists():
            metadata = companion.lstat()
            assert stat.S_ISREG(metadata.st_mode)
            assert metadata.st_nlink == 1


def test_companion_free_wal_backup_is_one_atomic_committed_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    writer = _create_multiportfolio_database(source, wal=True)
    writer.execute("CREATE TABLE atomic_rows (id INTEGER PRIMARY KEY, value INTEGER NOT NULL)")
    writer.executemany(
        "INSERT INTO atomic_rows(id, value) VALUES (?, 0)",
        ((row_id,) for row_id in range(30_000)),
    )
    writer.commit()
    writer.close()
    assert not source.with_name(source.name + "-wal").exists()
    update_committed = False

    def commit_atomic_update(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal update_committed
        if update_committed:
            return
        concurrent = sqlite3.connect(source, timeout=10.0)
        try:
            concurrent.execute("BEGIN IMMEDIATE")
            concurrent.execute("UPDATE atomic_rows SET value = 1")
            concurrent.commit()
            concurrent.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
            update_committed = True
        finally:
            concurrent.close()

    manifest = SQLiteMaintenanceService(progress_hook=commit_atomic_update).backup(source, backup)

    assert update_committed
    with sqlite3.connect(backup) as connection:
        committed_states = connection.execute(
            "SELECT value, COUNT(*) FROM atomic_rows GROUP BY value ORDER BY value"
        ).fetchall()
    assert committed_states in ([(0, 30_000)], [(1, 30_000)])
    assert SQLiteMaintenanceService().verify(backup) == manifest.evidence


def test_descriptor_byte_corruption_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "backup.db"
    alternate = tmp_path / "alternate.db"
    _create_multiportfolio_database(source).close()
    alternate_connection = _create_multiportfolio_database(alternate)
    alternate_connection.execute(
        "UPDATE positions SET quantity = 999 WHERE portfolio_id = 'alpha' AND symbol = 'AAPL'"
    )
    alternate_connection.commit()
    alternate_connection.close()
    alternate_image = alternate.read_bytes()
    retained_descriptor: int | None = None
    service = SQLiteMaintenanceService()
    real_stage = service._stage_target
    real_seal = sqlite_service_module._StagedTarget.seal_readonly

    def capture_stage(requested_path):
        nonlocal retained_descriptor
        staged = real_stage(requested_path)
        retained_descriptor = os.dup(staged.guardian_file_descriptor)
        return staged

    def corrupt_after_seal(staged) -> None:
        real_seal(staged)
        assert retained_descriptor is not None
        os.ftruncate(retained_descriptor, len(alternate_image))
        os.pwrite(retained_descriptor, alternate_image, 0)

    monkeypatch.setattr(service, "_stage_target", capture_stage)
    monkeypatch.setattr(
        sqlite_service_module._StagedTarget,
        "seal_readonly",
        corrupt_after_seal,
    )
    try:
        with pytest.raises(
            SQLiteMaintenanceError,
            match="descriptor-bound artifact evidence differs",
        ):
            service.backup(source, target)
    finally:
        if retained_descriptor is not None:
            os.close(retained_descriptor)

    assert not target.exists()


def test_manifest_is_secret_free_portable_and_exclusively_written(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    manifest_path = tmp_path / "manifest.json"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    service.write_manifest(manifest, manifest_path)
    payload = json.loads(manifest_path.read_text())

    serialized = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "credential" not in serialized.lower()
    assert "account" not in serialized.lower()
    assert "positions" not in serialized.lower()
    assert "portfolios" not in serialized.lower()
    assert payload["contains_secrets"] is False
    assert payload["mutated_authoritative_state"] is False
    assert payload["authorizes_startup"] is False
    assert service.load_manifest(manifest_path) == manifest
    with pytest.raises(SQLiteMaintenanceError, match="new exclusive"):
        service.write_manifest(manifest, manifest_path)


def test_restore_manifest_is_complete_before_database_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    target = tmp_path / "restore.db"
    restore_manifest_path = tmp_path / "restore.json"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    backup_manifest = service.backup(source, backup)
    reservation = service.reserve_manifest(restore_manifest_path)
    real_publish = sqlite_service_module._StagedTarget.publish
    inspected = False

    def inspect_manifest_before_publish(staged_target) -> None:
        nonlocal inspected
        payload = json.loads(restore_manifest_path.read_text())
        assert payload["operation"] == "restore"
        assert payload["input_artifact_sha256"] == backup_manifest.artifact_sha256
        assert stat.S_IMODE(restore_manifest_path.stat().st_mode) == 0o400
        inspected = True
        real_publish(staged_target)

    monkeypatch.setattr(
        sqlite_service_module._StagedTarget,
        "publish",
        inspect_manifest_before_publish,
    )
    try:
        restore_manifest = service.restore_clean_room(
            backup,
            target,
            backup_manifest,
            manifest_reservation=reservation,
        )
    finally:
        reservation.close()

    assert inspected
    assert service.load_manifest(restore_manifest_path) == restore_manifest
    assert target.exists()


@pytest.mark.parametrize("operation", ["backup", "restore"])
def test_reserved_manifest_write_failure_prevents_database_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    target = backup
    report_path = tmp_path / f"{operation}.json"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    backup_manifest = None
    if operation == "restore":
        backup_manifest = service.backup(source, backup)
        target = tmp_path / "restore.db"
    reservation = service.reserve_manifest(report_path)

    def fail_manifest_write(_descriptor: int, _payload: bytes) -> None:
        raise OSError("synthetic manifest I/O failure")

    monkeypatch.setattr(sqlite_service_module, "_write_all", fail_manifest_write)
    try:
        with pytest.raises(SQLiteMaintenanceError, match="manifest write failed closed"):
            if operation == "backup":
                service.backup(
                    source,
                    target,
                    manifest_reservation=reservation,
                )
            else:
                assert backup_manifest is not None
                service.restore_clean_room(
                    backup,
                    target,
                    backup_manifest,
                    manifest_reservation=reservation,
                )
    finally:
        reservation.close()

    assert not target.exists()
    assert report_path.exists()
    assert report_path.stat().st_size == 0


@pytest.mark.parametrize("kind", ["source_symlink", "source_hardlink", "target_symlink"])
def test_aliases_fail_closed(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "source.db"
    _create_multiportfolio_database(source).close()
    candidate_source = source
    target = tmp_path / "target.db"
    if kind == "source_symlink":
        candidate_source = tmp_path / "source-link.db"
        candidate_source.symlink_to(source)
    elif kind == "source_hardlink":
        candidate_source = tmp_path / "source-hardlink.db"
        os.link(source, candidate_source)
    else:
        target.symlink_to(source)

    with pytest.raises(SQLiteMaintenanceError):
        SQLiteMaintenanceService().backup(candidate_source, target)


def test_symlinked_wal_companion_is_rejected_before_sqlite_open(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_multiportfolio_database(source).close()
    source.with_name(source.name + "-wal").symlink_to(source)

    with pytest.raises(SQLiteMaintenanceError, match="SQLite companions"):
        SQLiteMaintenanceService().backup(source, target)
    assert not target.exists()


def test_existing_restore_target_is_never_replaced(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    target = tmp_path / "authoritative-looking.db"
    _create_multiportfolio_database(source).close()
    target.write_bytes(b"do-not-replace")
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)

    with pytest.raises(SQLiteMaintenanceError, match="new exclusively-created"):
        service.restore_clean_room(backup, target, manifest)
    assert target.read_bytes() == b"do-not-replace"


@pytest.mark.parametrize("suffix", ["-journal", "-shm", "-wal"])
def test_preexisting_target_sidecar_is_preserved_without_main_creation(
    tmp_path: Path, suffix: str
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    sidecar = target.with_name(target.name + suffix)
    _create_multiportfolio_database(source).close()
    sidecar.write_bytes(b"operator-evidence")

    with pytest.raises(SQLiteMaintenanceError, match="new exclusively-created SQLite family"):
        SQLiteMaintenanceService().backup(source, target)

    assert not target.exists()
    assert sidecar.read_bytes() == b"operator-evidence"


def test_sqlite_uses_private_namespace_and_never_unlinks_public_wal_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    public_wal = target.with_name(target.name + "-wal")
    protected_payload = b"irreplaceable-user-evidence"
    writer = _create_multiportfolio_database(source, wal=True)
    invoked = False

    def inject_public_sidecar(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            public_wal.write_bytes(protected_payload)

    try:
        with pytest.raises(SQLiteMaintenanceError, match="new exclusively-created SQLite family"):
            SQLiteMaintenanceService(progress_hook=inject_public_sidecar).backup(source, target)
    finally:
        writer.close()

    assert invoked
    assert public_wal.read_bytes() == protected_payload
    assert not target.exists()
    forensic_targets = list(tmp_path.glob(f".{target.name}.robo-trader-stage-*"))
    assert forensic_targets == []


def test_staging_symlink_cannot_redirect_sqlite_into_attacker_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    attacker_directory = tmp_path / "attacker"
    attacker_directory.mkdir()
    planted_wal = attacker_directory / "database.db-wal"
    planted_wal.write_bytes(b"irreplaceable-user-data")
    staged_name = f".{target.name}.robo-trader-stage-{'a' * 32}"
    (tmp_path / staged_name).symlink_to(attacker_directory, target_is_directory=True)
    writer = _create_multiportfolio_database(source, wal=True)

    try:
        manifest = SQLiteMaintenanceService().backup(source, target)
    finally:
        writer.close()

    assert planted_wal.read_bytes() == b"irreplaceable-user-data"
    assert SQLiteMaintenanceService().verify(target) == manifest.evidence


def test_no_persistent_staging_path_is_exposed_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    real_stage = service._stage_target
    staged_target = None
    attempted = False
    located_paths: list[Path] = []
    baseline_entries = set(tmp_path.iterdir())

    def capture_stage(requested_path):
        nonlocal staged_target
        staged_target = real_stage(requested_path)
        return staged_target

    def attempt_writable_open(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal attempted
        if not attempted:
            attempted = True
            assert staged_target is not None
            assert os.fstat(staged_target.guardian_file_descriptor).st_nlink == 0
            located_paths.extend(set(tmp_path.iterdir()) - baseline_entries)
            for path in located_paths:
                path.chmod(0o600)
                descriptor = os.open(path, os.O_RDWR)
                os.close(descriptor)

    monkeypatch.setattr(service, "_stage_target", capture_stage)
    service._progress_hook = attempt_writable_open
    manifest = service.backup(source, target)

    assert attempted
    assert located_paths == []
    assert hashlib.sha256(target.read_bytes()).hexdigest() == manifest.artifact_sha256


def test_parent_replacement_during_publication_fsync_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    output_parent = tmp_path / "output"
    parked_parent = tmp_path / "parked-output"
    output_parent.mkdir()
    target = output_parent / "target.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    real_stage = service._stage_target
    real_fsync = sqlite_service_module.os.fsync
    staged_target = None
    replaced = False
    replacement_payload = b"unrelated-user-file"

    def capture_stage(requested_path):
        nonlocal staged_target
        staged_target = real_stage(requested_path)
        return staged_target

    def replace_parent_on_publication_fsync(descriptor: int) -> None:
        nonlocal replaced
        if staged_target is not None and descriptor == staged_target.parent_file_descriptor:
            try:
                os.stat(
                    target.name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if not replaced:
                    replaced = True
                    output_parent.rename(parked_parent)
                    output_parent.mkdir()
                    target.write_bytes(replacement_payload)
        real_fsync(descriptor)

    monkeypatch.setattr(service, "_stage_target", capture_stage)
    monkeypatch.setattr(sqlite_service_module.os, "fsync", replace_parent_on_publication_fsync)

    with pytest.raises(SQLiteMaintenanceError, match="published database identity changed"):
        service.backup(source, target)

    assert replaced
    assert target.read_bytes() == replacement_payload
    assert (parked_parent / target.name).exists()


def test_target_cannot_overlap_source_sqlite_companion_family(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_multiportfolio_database(source).close()
    target = source.with_name(source.name + "-journal")

    with pytest.raises(SQLiteMaintenanceError, match="resource families overlap"):
        SQLiteMaintenanceService().backup(source, target)
    assert not target.exists()


def test_corrupt_backup_is_rejected_before_restore_target_creation(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.db"
    target = tmp_path / "restore.db"
    corrupt.write_bytes(b"not a sqlite database")

    with pytest.raises(SQLiteMaintenanceError):
        SQLiteMaintenanceService().verify(corrupt)
    assert not target.exists()


def test_manifest_detects_backup_corruption_before_clean_room_creation(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    target = tmp_path / "restore.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    backup.chmod(0o600)
    with backup.open("r+b") as stream:
        stream.seek(100)
        stream.write(b"tampered")
    backup.chmod(0o400)

    with pytest.raises(SQLiteMaintenanceError):
        service.restore_clean_room(backup, target, manifest)
    assert not target.exists()


def test_restore_verifies_and_copies_through_one_bound_source_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    replacement = tmp_path / "replacement.db"
    parked = tmp_path / "parked.db"
    target = tmp_path / "restore.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    with sqlite3.connect(replacement) as connection:
        connection.execute("PRAGMA page_size=8192")
    _create_multiportfolio_database(replacement).close()
    with sqlite3.connect(replacement) as connection:
        connection.execute("VACUUM")
        connection.execute(f"PRAGMA schema_version={manifest.evidence.schema_version}")
    assert replacement.read_bytes() != backup.read_bytes()
    assert service.verify(replacement) == manifest.evidence

    real_connect = service._connect_bound
    readonly_backup_opens = 0

    def observe_connect(binding, *, readonly, journal_off=False, immutable_readonly=False):
        nonlocal readonly_backup_opens
        if readonly and binding.path == backup:
            readonly_backup_opens += 1
            if readonly_backup_opens == 2:
                backup.rename(parked)
                replacement.rename(backup)
        return real_connect(
            binding,
            readonly=readonly,
            journal_off=journal_off,
            immutable_readonly=immutable_readonly,
        )

    monkeypatch.setattr(service, "_connect_bound", observe_connect)
    restored = service.restore_clean_room(backup, target, manifest)

    assert readonly_backup_opens == 1
    assert restored.input_artifact_sha256 == manifest.artifact_sha256
    assert not parked.exists()
    assert replacement.exists()


def test_interrupted_backup_preserves_source_and_seals_forensic_target(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "partial.db"
    _create_multiportfolio_database(source).close()
    before = source.read_bytes()

    def interrupt(_operation: str, _remaining: int, _total: int) -> None:
        raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        SQLiteMaintenanceService(progress_hook=interrupt).backup(source, target)

    assert source.read_bytes() == before
    assert not target.exists()
    forensic_targets = list(tmp_path.glob(f".{target.name}.robo-trader-stage-*"))
    assert forensic_targets == []


def test_manifest_with_unknown_field_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    manifest_path = tmp_path / "manifest.json"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    payload = manifest.to_dict()
    payload["broker_token"] = "must-not-be-accepted"
    manifest_path.write_text(json.dumps(payload))

    with pytest.raises(SQLiteMaintenanceError, match="manifest is invalid"):
        service.load_manifest(manifest_path)


def test_target_substitution_during_backup_fails_descriptor_check(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_multiportfolio_database(source).close()
    invoked = False

    def replace_target(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal invoked
        if invoked:
            return
        invoked = True
        target.symlink_to(source)

    with pytest.raises(SQLiteMaintenanceError, match="new exclusively-created"):
        SQLiteMaintenanceService(progress_hook=replace_target).backup(source, target)
    assert source.exists()
    assert target.is_symlink()


def test_anonymous_staging_has_no_path_for_hardlink_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    real_stage = service._stage_target
    staged_target = None
    invoked = False

    def capture_stage(requested_path):
        nonlocal staged_target
        staged_target = real_stage(requested_path)
        return staged_target

    def attempt_link(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            assert staged_target is not None
            assert os.fstat(staged_target.guardian_file_descriptor).st_nlink == 0
            assert list(tmp_path.glob(f".{target.name}.robo-trader-stage-*")) == []

    monkeypatch.setattr(service, "_stage_target", capture_stage)
    service._progress_hook = attempt_link
    service.backup(source, target)
    assert source.exists()
    assert target.exists()


def test_interrupted_restore_preserves_backup_and_seals_target(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    target = tmp_path / "partial-restore.db"
    _create_multiportfolio_database(source).close()
    manifest = SQLiteMaintenanceService().backup(source, backup)
    backup_before = backup.read_bytes()

    def interrupt(operation: str, _remaining: int, _total: int) -> None:
        if operation == "restore":
            raise RuntimeError("synthetic restore interruption")

    with pytest.raises(RuntimeError, match="restore interruption"):
        SQLiteMaintenanceService(progress_hook=interrupt).restore_clean_room(
            backup, target, manifest
        )
    assert backup.read_bytes() == backup_before
    assert not target.exists()
    forensic_targets = list(tmp_path.glob(f".{target.name}.robo-trader-stage-*"))
    assert forensic_targets == []


def test_migration_dry_run_changes_only_synthetic_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    before_bytes = source.read_bytes()

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="synthetic-v8",
            steps=(
                MigrationStep("ALTER TABLE positions ADD COLUMN note TEXT"),
                MigrationStep(
                    "UPDATE positions SET note=? WHERE portfolio_id=?",
                    ("synthetic-only", "alpha"),
                ),
            ),
            target_user_version=8,
        ),
    )

    assert report.outcome == "applied_to_synthetic_copy"
    assert report.source_unchanged is True
    assert report.before != report.after
    assert report.after.user_version == 8
    assert report.authorizes_startup is False
    assert source.read_bytes() == before_bytes
    with sqlite3.connect(source) as connection:
        columns = connection.execute("PRAGMA table_info(positions)").fetchall()
        assert "note" not in {row[1] for row in columns}


def test_migration_dry_run_accepts_active_wal_source(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    writer = _create_multiportfolio_database(source, wal=True)
    writer.execute("INSERT INTO positions VALUES ('alpha', 'NVDA', 4)")
    writer.commit()
    assert source.with_name(source.name + "-wal").exists()

    try:
        report = SQLiteMaintenanceService().dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="active-wal",
                steps=(MigrationStep("ALTER TABLE positions ADD COLUMN note TEXT"),),
            ),
        )
    finally:
        writer.close()

    assert report.outcome == "applied_to_synthetic_copy"
    assert report.source_unchanged is True
    assert report.before != report.after


def test_migration_detects_hidden_rowid_change_with_same_visible_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute("CREATE TABLE rowid_records (value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO rowid_records(value) VALUES (?)",
            (("alpha",), ("beta",)),
        )

    changed = False

    def replace_one_row(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal changed
        if changed:
            return
        with sqlite3.connect(source) as writer:
            writer.execute("DELETE FROM rowid_records WHERE rowid=1")
            writer.execute("INSERT INTO rowid_records(value) VALUES ('alpha')")
        changed = True

    report = SQLiteMaintenanceService(progress_hook=replace_one_row).dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="rowid-source-change",
            steps=(MigrationStep("SELECT 1"),),
        ),
    )

    with sqlite3.connect(source) as connection:
        rows = connection.execute("SELECT rowid,value FROM rowid_records ORDER BY rowid").fetchall()
    assert changed is True
    assert rows == [(2, "beta"), (3, "alpha")]
    assert report.before == report.after
    assert report.source_unchanged is False
    assert report.outcome == "source_changed_fail_closed"
    assert report.error_code == "authoritative_source_changed"


def test_evidence_handles_without_rowid_and_shadowed_rowid_aliases(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "backup.db"
    with sqlite3.connect(source) as connection:
        connection.executescript("""
            CREATE TABLE natural_keys (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            INSERT INTO natural_keys VALUES ('alpha', 'one');
            CREATE TABLE shadowed_aliases (
                rowid INTEGER NOT NULL,
                _rowid_ INTEGER NOT NULL,
                oid INTEGER NOT NULL,
                value TEXT NOT NULL
            );
            INSERT INTO shadowed_aliases VALUES (1, 2, 3, 'visible');
            """)

    service = SQLiteMaintenanceService()
    manifest = service.backup(source, target)

    assert service.verify(target) == manifest.evidence


def test_migration_evidence_includes_persistent_planner_statistics(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.executescript("""
            CREATE TABLE analyzed_rows (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
            CREATE INDEX idx_analyzed_rows_value ON analyzed_rows(value);
            INSERT INTO analyzed_rows VALUES (1, 'alpha'), (2, 'beta'), (3, 'beta');
            ANALYZE;
            """)
        assert connection.execute("SELECT COUNT(*) FROM sqlite_stat1").fetchone() == (1,)

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="planner-statistics-change",
            steps=(
                MigrationStep(
                    "UPDATE sqlite_stat1 SET stat=? WHERE tbl=?",
                    ("999 1", "analyzed_rows"),
                ),
            ),
        ),
    )

    assert report.outcome == "applied_to_synthetic_copy"
    assert report.before != report.after
    assert report.source_unchanged is True
    with sqlite3.connect(target) as connection:
        assert connection.execute(
            "SELECT stat FROM sqlite_stat1 WHERE tbl='analyzed_rows'"
        ).fetchone() == ("999 1",)


def test_migration_plan_deadline_interrupts_and_rolls_back(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE deadline_rows (id INTEGER PRIMARY KEY, marker INTEGER, value TEXT)"
        )
        connection.executemany(
            "INSERT INTO deadline_rows(id, marker, value) VALUES (?, 1, 'before')",
            ((row_id,) for row_id in range(100_000)),
        )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                (
                    "import sys",
                    "from robo_trader.maintenance import MigrationPlan, MigrationStep, SQLiteMaintenanceService",
                    "report = SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(",
                    "    sys.argv[1], sys.argv[2],",
                    "    plan=MigrationPlan(migration_id='bounded-bulk-update', steps=(",
                    "        MigrationStep('UPDATE deadline_rows SET value=? WHERE marker=?', ('after', 1)),",
                    "    )),",
                    ")",
                    "print(report.outcome)",
                    "print(report.error_code)",
                )
            ),
            str(source),
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    # The child process is a hard outer boundary: a lost SQLite interrupt
    # cannot hang the test runner. The progress-specific result proves that
    # SQLite's VM handler, rather than only a post-statement check, stopped it.
    assert child.stdout.strip().splitlines()[-2:] == [
        "rolled_back",
        "migration_progress_deadline_exceeded",
    ]
    with sqlite3.connect(target) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM deadline_rows WHERE value='after'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM deadline_rows WHERE value='before'"
        ).fetchone() == (100_000,)
    with sqlite3.connect(source) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM deadline_rows WHERE value='after'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM deadline_rows WHERE value='before'"
        ).fetchone() == (100_000,)


def test_migration_denies_native_function_that_can_evade_vm_deadline(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="deny-native-allocation",
                steps=(
                    MigrationStep("CREATE TABLE oversized_payload (value BLOB NOT NULL)"),
                    MigrationStep(
                        "INSERT INTO oversized_payload(value) VALUES (randomblob(50000000))"
                    ),
                ),
            ),
        )

    assert not target.exists()


def test_migration_grammar_rejects_unreviewed_sql_shapes_before_copy(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_multiportfolio_database(source).close()
    unsupported = (
        "UPDATE positions SET quantity=?",
        "INSERT INTO positions(portfolio_id,symbol,quantity) SELECT ?,?,?",
        "CREATE TABLE escaped (value TEXT CHECK(value <> ''))",
        'ALTER TABLE "positions" ADD COLUMN note TEXT',
        "SELECT 1 -- comment",
        "DELETE FROM positions WHERE portfolio_id='alpha'",
        "BEGIN",
    )

    for index, sql in enumerate(unsupported):
        target = tmp_path / f"rejected-{index}.db"
        parameters = tuple(1 for _ in range(sql.count("?")))
        with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
            SQLiteMaintenanceService().dry_run_migration(
                source,
                target,
                plan=MigrationPlan(
                    migration_id=f"rejected-shape-{index}",
                    steps=(MigrationStep(sql, parameters),),
                ),
            )
        assert not target.exists()


def test_migration_deadline_checks_every_vm_opcode_between_allocations(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    payload = b"x" * (1024 * 1024)
    expression = "||".join("?1" for _ in range(40))
    started = time.monotonic()

    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="bounded-between-opcodes",
                steps=(MigrationStep(f"SELECT {expression}", (payload,)),),
            ),
        )

    assert time.monotonic() - started < 2.0
    assert not target.exists()


def test_migration_growth_is_capped_by_synthetic_database_page_limit(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    payload = b"x" * 700_000

    report = SQLiteMaintenanceService(max_migration_growth_bytes=1024 * 1024).dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="bounded-synthetic-growth",
            steps=(
                MigrationStep("CREATE TABLE bounded_payload (value BLOB NOT NULL)"),
                MigrationStep("INSERT INTO bounded_payload(value) VALUES (?)", (payload,)),
                MigrationStep("INSERT INTO bounded_payload(value) VALUES (?)", (payload,)),
            ),
        ),
    )

    assert report.outcome == "rolled_back"
    assert report.error_code == "migration_plan_failed"
    assert report.before == report.after
    assert report.source_unchanged is True


def test_migration_temp_schema_writes_are_denied_and_capped(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    payload = b"x" * (3 * 1024 * 1024)

    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService(max_migration_growth_bytes=1024 * 1024).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="deny-temp-growth",
                steps=(
                    MigrationStep("CREATE TEMP TABLE escaped_growth (value BLOB NOT NULL)"),
                    MigrationStep("INSERT INTO escaped_growth(value) VALUES (?)", (payload,)),
                ),
            ),
        )

    assert not target.exists()


def test_migration_rejects_functions_embedded_in_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE guarded ("
            "value INTEGER NOT NULL CHECK(length(randomblob(50000000)) > 0)"
            ")"
        )
    started = time.monotonic()

    with pytest.raises(SQLiteMaintenanceError, match="source schema"):
        SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="reject-schema-function",
                steps=(MigrationStep("INSERT INTO guarded(value) VALUES (?)", (1,)),),
            ),
        )

    assert time.monotonic() - started < 2.0
    assert not target.exists()


def test_migration_screens_virtual_generated_functions_before_row_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE generated_rows ("
            "base INTEGER NOT NULL,"
            "inflated TEXT GENERATED ALWAYS AS (hex(zeroblob(2000000))) VIRTUAL"
            ")"
        )
        connection.execute("INSERT INTO generated_rows(base) VALUES (1)")

    events: list[str] = []
    original_schema_function_calls = sqlite_service_module._schema_function_calls
    original_database_evidence = sqlite_service_module._database_evidence

    def tracked_schema_function_calls(connection: sqlite3.Connection) -> tuple[str, ...]:
        events.append("schema_screen")
        return original_schema_function_calls(connection)

    def tracked_database_evidence(
        connection: sqlite3.Connection,
    ) -> sqlite_service_module.DatabaseEvidence:
        events.append("row_evidence")
        return original_database_evidence(connection)

    monkeypatch.setattr(
        sqlite_service_module,
        "_schema_function_calls",
        tracked_schema_function_calls,
    )
    monkeypatch.setattr(
        sqlite_service_module,
        "_database_evidence",
        tracked_database_evidence,
    )

    with pytest.raises(SQLiteMaintenanceError, match="source schema"):
        SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="screen-before-generated-evidence",
                steps=(
                    MigrationStep(
                        "UPDATE generated_rows SET base=? WHERE base=?",
                        (2, 1),
                    ),
                ),
            ),
        )

    assert events == ["schema_screen"]
    assert not target.exists()


@pytest.mark.parametrize(
    "schema_expression",
    [
        "randomblob/**/(50000000)",
        "randomblob-- split token\n(50000000)",
        "/**/randomblob(50000000)",
        "-- prefix comment\nrandomblob(50000000)",
    ],
)
def test_migration_rejects_commented_schema_function_calls(
    tmp_path: Path,
    schema_expression: str,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE guarded ("
            f"value INTEGER NOT NULL CHECK({schema_expression} IS NOT NULL)"
            ")"
        )
        assert sqlite_service_module._schema_function_calls(connection) == ("randomblob",)

    with pytest.raises(SQLiteMaintenanceError, match="source schema"):
        SQLiteMaintenanceService(max_migration_seconds=0.01).dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="reject-commented-schema-function",
                steps=(MigrationStep("INSERT INTO guarded(value) VALUES (?)", (1,)),),
            ),
        )

    assert not target.exists()


def test_schema_line_comment_continues_through_carriage_return(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE guarded ("
            "value INTEGER NOT NULL CHECK(1 -- note\rrandomblob/**/(50000000)\n)"
            ")"
        )
        assert sqlite_service_module._schema_function_calls(connection) == ()

    report = SQLiteMaintenanceService(max_migration_seconds=1).dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="accept-function-name-inside-line-comment",
            steps=(MigrationStep("INSERT INTO guarded(value) VALUES (?)", (1,)),),
        ),
    )

    assert report.outcome == "applied_to_synthetic_copy"
    assert report.error_code is None
    assert report.before != report.after
    assert report.source_unchanged is True


def test_migration_authorizer_denies_native_pointer_and_virtual_table_actions(
    tmp_path: Path,
) -> None:
    authorizer = sqlite_service_module._migration_authorizer
    assert (
        authorizer(sqlite3.SQLITE_FUNCTION, None, "fts3_tokenizer", None, None)
        == sqlite3.SQLITE_DENY
    )
    assert (
        authorizer(sqlite3.SQLITE_FUNCTION, None, "load_extension", None, None)
        == sqlite3.SQLITE_DENY
    )
    assert (
        authorizer(sqlite3.SQLITE_FUNCTION, None, "randomblob", None, None) == sqlite3.SQLITE_DENY
    )
    assert (
        authorizer(sqlite3.SQLITE_CREATE_VTABLE, "docs", "fts3", "main", None)
        == sqlite3.SQLITE_DENY
    )
    assert (
        authorizer(sqlite3.SQLITE_DROP_VTABLE, "docs", "fts3", "main", None) == sqlite3.SQLITE_DENY
    )

    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService().dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="deny-native-pointer-function",
                steps=(
                    MigrationStep(
                        "SELECT fts3_tokenizer('unsafe', ?)",
                        (b"\x42" * 8,),
                    ),
                ),
            ),
        )

    assert not target.exists()


def test_migration_plan_cannot_change_untracked_schema_cookie(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    assert (
        sqlite_service_module._migration_authorizer(
            sqlite3.SQLITE_PRAGMA,
            "schema_version",
            "999",
            "main",
            None,
        )
        == sqlite3.SQLITE_DENY
    )

    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService().dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="deny-schema-cookie",
                steps=(MigrationStep("PRAGMA schema_version=999"),),
            ),
        )

    assert not target.exists()


def test_migration_evidence_tracks_schema_cookie_after_transient_ddl(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="transient-ddl-schema-cookie",
            steps=(
                MigrationStep("CREATE TABLE transient_table (id INTEGER PRIMARY KEY)"),
                MigrationStep("DROP TABLE transient_table"),
            ),
        ),
    )

    assert report.outcome == "applied_to_synthetic_copy"
    assert report.before.schema_sha256 == report.after.schema_sha256
    assert report.after.schema_version > report.before.schema_version
    assert report.before != report.after


def test_final_source_evidence_failure_prevents_synthetic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()

    def fail_final_evidence(_source_path: Path | str):
        raise SQLiteMaintenanceError("injected final evidence failure")

    monkeypatch.setattr(service, "_live_source_evidence", fail_final_evidence)

    with pytest.raises(SQLiteMaintenanceError, match="final evidence failure"):
        service.dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="final-evidence-failure",
                steps=(MigrationStep("ALTER TABLE positions ADD COLUMN note TEXT"),),
            ),
        )

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.robo-trader-stage-*")) == []


def test_migration_uses_the_copy_connection_without_a_writable_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    real_connect = service._connect_bound
    writable_paths: list[Path] = []
    all_opened_paths: list[Path] = []

    def count_connect(binding, *, readonly, journal_off=False, immutable_readonly=False):
        all_opened_paths.append(binding.path)
        if not readonly:
            writable_paths.append(binding.path)
        return real_connect(
            binding,
            readonly=readonly,
            journal_off=journal_off,
            immutable_readonly=immutable_readonly,
        )

    monkeypatch.setattr(service, "_connect_bound", count_connect)
    report = service.dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="single-bound-target",
            steps=(MigrationStep("ALTER TABLE positions ADD COLUMN note TEXT"),),
        ),
    )

    assert report.outcome == "applied_to_synthetic_copy"
    assert writable_paths == []
    assert target not in all_opened_paths


def test_migration_report_remains_bound_to_manifest_after_target_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    parked = tmp_path / "bound-copy.db"
    replacement = tmp_path / "replacement.db"
    _create_multiportfolio_database(source).close()
    replacement_connection = _create_multiportfolio_database(replacement)
    replacement_connection.execute("INSERT INTO positions VALUES ('alpha', 'TSLA', 99)")
    replacement_connection.commit()
    replacement_connection.close()
    service = SQLiteMaintenanceService()
    real_online_copy = service._online_copy
    captured_manifest = None

    def substitute_after_bound_copy(*args, **kwargs):
        nonlocal captured_manifest
        manifest, result = real_online_copy(*args, **kwargs)
        captured_manifest = manifest
        target.rename(parked)
        replacement.rename(target)
        return manifest, result

    monkeypatch.setattr(service, "_online_copy", substitute_after_bound_copy)
    report = service.dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="bound-final-report",
            steps=(MigrationStep("ALTER TABLE positions ADD COLUMN note TEXT"),),
        ),
    )

    assert captured_manifest is not None
    assert report.after == captured_manifest.evidence
    assert report.target_artifact_sha256 == captured_manifest.artifact_sha256
    assert report.target_artifact_sha256 == hashlib.sha256(parked.read_bytes()).hexdigest()
    assert report.target_artifact_sha256 != hashlib.sha256(target.read_bytes()).hexdigest()


def test_service_transaction_completion_policy_is_python310_compatible() -> None:
    arguments = (sqlite3.SQLITE_TRANSACTION, "COMMIT", None, None, None)
    assert sqlite_service_module._migration_authorizer(*arguments) == sqlite3.SQLITE_DENY
    assert sqlite_service_module._migration_completion_authorizer(*arguments) == sqlite3.SQLITE_OK


def test_interrupted_migration_rolls_back_copy_and_preserves_source(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="synthetic-interruption",
            steps=(
                MigrationStep(
                    "UPDATE positions SET quantity=? WHERE portfolio_id=?",
                    (99, "alpha"),
                ),
                MigrationStep(
                    "UPDATE missing_table SET value=? WHERE id=?",
                    ("unreachable", 1),
                ),
            ),
        ),
    )

    assert report.outcome == "rolled_back"
    assert report.error_code == "migration_plan_failed"
    assert report.before == report.after
    assert report.source_unchanged is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o400


def test_migration_plan_cannot_attach_or_commit(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
        SQLiteMaintenanceService().dry_run_migration(
            source,
            target,
            plan=MigrationPlan(
                migration_id="synthetic-attach",
                steps=(MigrationStep("ATTACH DATABASE ? AS escaped", (str(source),)),),
            ),
        )
    assert not target.exists()


@pytest.mark.parametrize("pragma", ["hard_heap_limit", "soft_heap_limit"])
def test_migration_plan_cannot_change_process_global_heap_limits(
    tmp_path: Path,
    pragma: str,
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    with sqlite3.connect(":memory:") as observer:
        before_limit = observer.execute(f"PRAGMA {pragma}").fetchone()
        with pytest.raises(SQLiteMaintenanceError, match="supported grammar"):
            SQLiteMaintenanceService().dry_run_migration(
                source,
                target,
                plan=MigrationPlan(
                    migration_id=f"deny-{pragma.replace('_', '-')}",
                    steps=(MigrationStep(f"PRAGMA {pragma}=1048576"),),
                ),
            )
        after_limit = observer.execute(f"PRAGMA {pragma}").fetchone()

    assert after_limit == before_limit
    assert not target.exists()


def test_migration_oversized_integer_parameter_rolls_back_with_report(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "dry-run.db"
    _create_multiportfolio_database(source).close()

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="oversized-integer",
            steps=(
                MigrationStep(
                    "UPDATE positions SET quantity=? WHERE portfolio_id=?",
                    (2**100, "alpha"),
                ),
            ),
        ),
    )

    assert report.outcome == "rolled_back"
    assert report.error_code == "migration_plan_failed"
    assert report.before == report.after
    assert report.source_unchanged is True
    assert stat.S_IMODE(target.stat().st_mode) == 0o400


def test_verify_rejects_companions_without_creating_or_rewriting_them(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    _create_multiportfolio_database(source).close()
    wal = source.with_name(source.name + "-wal")
    wal.write_bytes(b"preserve-this-evidence")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(SQLiteMaintenanceError, match="sealed database"):
        SQLiteMaintenanceService().verify(source)

    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.asyncio
async def test_legacy_multiuser_migration_preserves_inspection_and_noop(tmp_path: Path) -> None:
    missing = MultiuserMigration(tmp_path / "missing.db")
    assert await missing.needs_migration() is True
    assert await missing.migrate() is False

    applied_path = tmp_path / "applied.db"
    with sqlite3.connect(applied_path) as connection:
        connection.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, description TEXT)"
        )
        connection.execute("INSERT INTO schema_migrations VALUES (1, 'already applied')")
    applied = MultiuserMigration(applied_path)
    assert await applied.needs_migration() is False
    assert await applied.migrate() is False


@pytest.mark.asyncio
async def test_legacy_multiuser_mutation_entrypoints_are_quarantined(tmp_path: Path) -> None:
    source = tmp_path / "legacy.db"
    with sqlite3.connect(source) as connection:
        connection.execute("CREATE TABLE positions (symbol TEXT)")
        connection.execute("INSERT INTO positions VALUES ('AAPL')")
        connection.execute("CREATE TABLE account (id INTEGER PRIMARY KEY, cash REAL)")
        connection.execute("INSERT INTO account VALUES (1, 1000.0), (2, 1250.0)")
    before = source.read_bytes()
    migration = MultiuserMigration(source)

    with pytest.raises(LegacyMultiuserMigrationDisabled):
        await migration.migrate()
    with pytest.raises(LegacyMultiuserMigrationDisabled):
        await migration._create_backup()
    with pytest.raises(LegacyMultiuserMigrationDisabled):
        await migration._restore_from_backup(tmp_path / "backup.db")
    assert source.read_bytes() == before
    with sqlite3.connect(source) as connection:
        assert connection.execute("SELECT id,cash FROM account ORDER BY id").fetchall() == [
            (1, 1000.0),
            (2, 1250.0),
        ]


def test_cli_backup_verify_and_restore_are_non_authorizing(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    backup_manifest = tmp_path / "backup.json"
    restore = tmp_path / "restore.db"
    restore_manifest = tmp_path / "restore.json"
    _create_multiportfolio_database(source).close()
    script = Path(__file__).resolve().parents[2] / "scripts" / "database_maintenance.py"

    for arguments in (
        [
            "backup",
            "--source",
            str(source),
            "--target",
            str(backup),
            "--manifest",
            str(backup_manifest),
        ],
        [
            "verify",
            "--database",
            str(backup),
            "--manifest",
            str(backup_manifest),
        ],
        [
            "restore-clean-room",
            "--backup",
            str(backup),
            "--backup-manifest",
            str(backup_manifest),
            "--target",
            str(restore),
            "--restore-manifest",
            str(restore_manifest),
        ],
    ):
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout)["authorizes_startup"] is False


def test_cli_reserves_backup_manifest_before_database_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    manifest = tmp_path / "backup.json"
    _create_multiportfolio_database(source).close()
    manifest.write_bytes(b"operator-owned-report")
    script = Path(__file__).resolve().parents[2] / "scripts" / "database_maintenance.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "backup",
            "--source",
            str(source),
            "--target",
            str(backup),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 2
    assert not backup.exists()
    assert manifest.read_bytes() == b"operator-owned-report"


def test_cli_reserves_restore_manifest_before_database_publication(tmp_path: Path) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    backup_manifest = tmp_path / "backup.json"
    target = tmp_path / "restore.db"
    restore_manifest = tmp_path / "restore.json"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    manifest = service.backup(source, backup)
    service.write_manifest(manifest, backup_manifest)
    restore_manifest.write_bytes(b"operator-owned-report")
    script = Path(__file__).resolve().parents[2] / "scripts" / "database_maintenance.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "restore-clean-room",
            "--backup",
            str(backup),
            "--backup-manifest",
            str(backup_manifest),
            "--target",
            str(target),
            "--restore-manifest",
            str(restore_manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 2
    assert not target.exists()
    assert restore_manifest.read_bytes() == b"operator-owned-report"


@pytest.mark.parametrize(
    ("family_owner", "suffix"),
    [
        ("source", "-wal"),
        ("target", ""),
        ("target", "-journal"),
        ("target", "-shm"),
        ("target", "-wal"),
    ],
)
def test_cli_rejects_manifest_in_database_resource_family(
    tmp_path: Path,
    family_owner: str,
    suffix: str,
) -> None:
    source = tmp_path / "source.db"
    backup = tmp_path / "backup.db"
    owner = source if family_owner == "source" else backup
    manifest = owner.with_name(owner.name + suffix)
    _create_multiportfolio_database(source).close()
    script = Path(__file__).resolve().parents[2] / "scripts" / "database_maintenance.py"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "backup",
            "--source",
            str(source),
            "--target",
            str(backup),
            "--manifest",
            str(manifest),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 2
    assert "report path overlaps a SQLite resource family" in completed.stderr
    assert not backup.exists()
    assert not manifest.exists()

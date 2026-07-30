from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    public_wal = target.with_name(target.name + "-wal")
    protected_payload = b"irreplaceable-user-evidence"
    writer = _create_multiportfolio_database(source, wal=True)
    service = SQLiteMaintenanceService()
    real_connect = service._connect_bound
    writable_paths: list[Path] = []

    def observe_connect(binding, *, readonly, journal_off=False, immutable_readonly=False):
        if not readonly:
            writable_paths.append(binding.path)
            public_wal.write_bytes(protected_payload)
        return real_connect(
            binding,
            readonly=readonly,
            journal_off=journal_off,
            immutable_readonly=immutable_readonly,
        )

    monkeypatch.setattr(service, "_connect_bound", observe_connect)
    try:
        with pytest.raises(SQLiteMaintenanceError, match="new exclusively-created SQLite family"):
            service.backup(source, target)
    finally:
        writer.close()

    assert writable_paths
    assert all(path != target and path.parent != target.parent for path in writable_paths)
    assert stat.S_IMODE(writable_paths[0].parent.stat().st_mode) == 0o700
    assert writable_paths[0].parent.name.startswith(f".{target.name}.robo-trader-stage-")
    assert public_wal.read_bytes() == protected_payload
    assert not target.exists()


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
    forensic_targets = list(tmp_path.glob(f".{target.name}.robo-trader-stage-*/database.db"))
    assert len(forensic_targets) == 1
    assert stat.S_IMODE(forensic_targets[0].stat().st_mode) == 0o400


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


def test_hardlink_race_during_backup_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    alias = tmp_path / "target-alias.db"
    _create_multiportfolio_database(source).close()
    service = SQLiteMaintenanceService()
    real_stage = service._stage_target
    staged_path: Path | None = None
    invoked = False

    def capture_stage(requested_path):
        nonlocal staged_path
        staged = real_stage(requested_path)
        staged_path = staged.binding.path
        return staged

    def add_link(_operation: str, _remaining: int, _total: int) -> None:
        nonlocal invoked
        if not invoked:
            invoked = True
            assert staged_path is not None
            os.link(staged_path, alias)

    monkeypatch.setattr(service, "_stage_target", capture_stage)
    with pytest.raises(SQLiteMaintenanceError, match="another filesystem link"):
        service._progress_hook = add_link
        service.backup(source, target)
    assert source.exists()
    assert not target.exists()
    assert alias.exists()


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
    forensic_targets = list(tmp_path.glob(f".{target.name}.robo-trader-stage-*/database.db"))
    assert len(forensic_targets) == 1
    assert stat.S_IMODE(forensic_targets[0].stat().st_mode) == 0o400


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
    assert len(writable_paths) == 1
    assert writable_paths[0] != target
    assert writable_paths[0].parent != target.parent
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
                MigrationStep("DELETE FROM positions"),
                MigrationStep("THIS IS NOT VALID SQL"),
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

    report = SQLiteMaintenanceService().dry_run_migration(
        source,
        target,
        plan=MigrationPlan(
            migration_id="synthetic-attach",
            steps=(MigrationStep("ATTACH DATABASE ? AS escaped", (str(source),)),),
        ),
    )
    assert report.outcome == "rolled_back"
    assert report.before == report.after
    assert report.source_unchanged is True


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

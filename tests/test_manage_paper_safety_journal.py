import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import robo_trader.config as config_module
import scripts.manage_paper_safety_journal as journal_script
from robo_trader.config import _derive_safety_account_scope
from robo_trader.runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from robo_trader.safety import (
    EvidenceStatus,
    ExposureEvidence,
    GateContext,
    JournalIntegrityError,
    OrderIntent,
    OrderSide,
    OrderType,
    PortfolioAllocationEvidence,
    ReconciliationStatus,
    RuntimeStartupBlocked,
    SafetyJournal,
    SubmissionDescriptor,
    TimeInForce,
    TransportState,
)
from scripts.manage_paper_safety_journal import (
    CREATE_CONFIRMATION,
    MIGRATE_CONFIRMATION,
    generate_account_scope,
    initialize_journal,
    migrate_empty_legacy_journal,
    verify_journal,
)


def _env(tmp_path: Path) -> dict[str, str]:
    scope_key = "0123456789abcdef" * 4
    return {
        "EXECUTION_MODE": "paper",
        "TRADING_MODE": "paper",
        "ENVIRONMENT": "dev",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "4002",
        "IBKR_READONLY": "true",
        "IBKR_CLIENT_ID": "123",
        "IBKR_ACCOUNT": "DU_TEST_PAPER",
        "IBKR_APPROVED_ACCOUNTS": "DU_TEST_PAPER",
        "IBKR_ACCOUNT_TYPE": "paper",
        "RT_STATE_NAMESPACE": "paper",
        "RT_DB_PATH": str(tmp_path / "paper-ledger.db"),
        "SAFETY_ACCOUNT_SCOPE_KEY": scope_key,
        "SAFETY_ACCOUNT_SCOPE": _derive_safety_account_scope(scope_key, "DU_TEST_PAPER"),
        "SAFETY_JOURNAL_PATH": str(tmp_path / "paper-safety.db"),
        "MODEL_ARTIFACT_SET": "test-models",
        "BUILD_ID": "test-build",
    }


class _AvailableLifecycleLock:
    def __init__(self) -> None:
        self.acquired = False
        self.released = False

    def acquire(self) -> bool:
        self.acquired = True
        return True

    def release(self) -> None:
        self.released = True


def _allow_migration_lock(monkeypatch) -> _AvailableLifecycleLock:
    lock = _AvailableLifecycleLock()
    monkeypatch.setattr(journal_script, "RuntimeLifecycleLock", lambda: lock)
    return lock


def _legacy_env(
    tmp_path: Path,
    *,
    journal_path: Path | None = None,
) -> dict[str, str]:
    environ = _env(tmp_path)
    environ.pop("SAFETY_ACCOUNT_SCOPE_KEY")
    environ["SAFETY_ACCOUNT_SCOPE"] = "acct_v1_" + ("0123456789abcdef" * 4)
    if journal_path is not None:
        environ["SAFETY_JOURNAL_PATH"] = str(journal_path)
    SafetyJournal(environ["SAFETY_JOURNAL_PATH"]).initialize(
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    )
    return environ


def test_generated_scope_is_opaque_shape_and_unique():
    first_key, first = generate_account_scope("DU_TEST_PAPER")
    second_key, second = generate_account_scope("DU_TEST_PAPER")

    assert len(first_key) == 64
    assert len(second_key) == 64
    assert first.startswith("acct_v1_")
    assert len(first) == len("acct_v1_") + 64
    assert first == _derive_safety_account_scope(first_key, "DU_TEST_PAPER")
    assert second == _derive_safety_account_scope(second_key, "DU_TEST_PAPER")
    assert first != second
    int(first.removeprefix("acct_v1_"), 16)


def test_initialize_requires_exact_confirmation_and_never_creates_on_failure(tmp_path):
    environ = _env(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])

    with pytest.raises(ValueError, match="confirmation"):
        initialize_journal(environ, confirmation="yes")

    assert not journal_path.exists()


def test_initialize_creates_new_empty_journal_and_verify_is_read_only(tmp_path):
    environ = _env(tmp_path)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])

    contract = initialize_journal(environ, confirmation=CREATE_CONFIRMATION)
    original_stat = journal_path.stat()
    verified = verify_journal(environ)

    assert contract.safety_journal_identity == verified.safety_journal_identity
    assert journal_path.stat().st_ino == original_stat.st_ino
    assert journal_path.stat().st_size == original_stat.st_size


def test_initialize_refuses_to_modify_existing_journal(tmp_path):
    environ = _env(tmp_path)
    initialize_journal(environ, confirmation=CREATE_CONFIRMATION)
    journal_path = Path(environ["SAFETY_JOURNAL_PATH"])
    original = journal_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to modify"):
        initialize_journal(environ, confirmation=CREATE_CONFIRMATION)

    assert journal_path.read_bytes() == original


def test_initialize_refuses_missing_parent_directory(tmp_path):
    environ = _env(tmp_path)
    environ["SAFETY_JOURNAL_PATH"] = str(tmp_path / "missing" / "paper-safety.db")

    with pytest.raises(FileNotFoundError, match="parent directory"):
        initialize_journal(environ, confirmation=CREATE_CONFIRMATION)

    assert not (tmp_path / "missing").exists()


def test_empty_legacy_migration_preserves_source_and_creates_bound_target(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "account-bound-safety.db"
    env_file = tmp_path / ".env"
    env_file.write_text("sentinel=unchanged\n")
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    lock = _allow_migration_lock(monkeypatch)

    result = migrate_empty_legacy_journal(
        environ,
        target_path=target,
        confirmation=MIGRATE_CONFIRMATION,
    )

    assert lock.acquired is True
    assert lock.released is True
    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert env_file.read_text() == "sentinel=unchanged\n"
    assert Path(result.safety_journal_path) == target
    assert result.safety_account_scope == _derive_safety_account_scope(
        result.safety_account_scope_key,
        environ["IBKR_ACCOUNT"],
    )
    old_state = SafetyJournal(source).replay(
        expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        expected_account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    )
    new_state = SafetyJournal(target).replay(
        expected_execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        expected_account_scope=result.safety_account_scope,
    )
    assert old_state.events == new_state.events == ()
    receipt = json.loads(result.receipt)
    assert receipt["migration"] == "empty-legacy-paper-safety-journal-v1"
    assert environ["IBKR_ACCOUNT"] not in result.receipt
    assert str(source) not in result.receipt
    assert str(target) not in result.receipt


def test_empty_legacy_migration_cli_emits_only_required_config_and_redacted_receipt(
    tmp_path,
    monkeypatch,
    capsys,
):
    environ = _legacy_env(tmp_path)
    target = tmp_path / "cli-account-bound-safety.db"
    _allow_migration_lock(monkeypatch)
    monkeypatch.setattr(journal_script, "_resolved_environment", lambda: environ)

    result = journal_script.main(
        [
            "migrate-empty-legacy",
            "--target",
            str(target),
            "--confirm",
            MIGRATE_CONFIRMATION,
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "SAFETY_ACCOUNT_SCOPE_KEY=" in output
    assert "SAFETY_ACCOUNT_SCOPE=" in output
    assert f"SAFETY_JOURNAL_PATH={target}" in output
    assert "MIGRATION_RECEIPT=" in output
    assert environ["IBKR_ACCOUNT"] not in output
    assert str(Path(environ["SAFETY_JOURNAL_PATH"])) not in output
    assert "Legacy journal preserved unchanged; .env was not edited." in output


def test_legacy_migration_requires_confirmation_and_stopped_runtime(tmp_path, monkeypatch):
    environ = _legacy_env(tmp_path)
    target = tmp_path / "account-bound-safety.db"

    with pytest.raises(ValueError, match=MIGRATE_CONFIRMATION):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation="yes",
        )
    assert not target.exists()


def test_legacy_migration_rejects_already_account_bound_source(tmp_path, monkeypatch):
    environ = _env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "rotated-account-bound-safety.db"
    SafetyJournal(source).initialize(
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=environ["SAFETY_ACCOUNT_SCOPE"],
    )
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="already account-bound"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()

    monkeypatch.setattr(
        journal_script,
        "RuntimeLifecycleLock",
        lambda: type(
            "UnavailableLock",
            (),
            {"acquire": lambda self: False, "release": lambda self: None},
        )(),
    )
    with pytest.raises(RuntimeError, match="trader must be stopped"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )
    assert not target.exists()


def test_legacy_config_failure_points_to_non_destructive_migration(tmp_path):
    environ = _legacy_env(tmp_path)

    with pytest.raises(config_module.ConfigValidationError) as caught:
        config_module.load_runtime_contract_from_env(environ, project_root=tmp_path)

    message = str(caught.value)
    assert "SAFETY_ACCOUNT_SCOPE_KEY" in message
    assert "migrate-empty-legacy" in message
    assert MIGRATE_CONFIRMATION in message
    assert environ["IBKR_ACCOUNT"] not in message


def test_legacy_migration_rejects_existing_or_symlink_target(tmp_path, monkeypatch):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    source_bytes = source.read_bytes()
    existing = tmp_path / "existing.db"
    existing.write_text("do-not-overwrite")
    linked = tmp_path / "linked.db"
    linked.symlink_to(existing)
    _allow_migration_lock(monkeypatch)

    for target in (source, existing, linked):
        with pytest.raises((ValueError, FileExistsError)):
            migrate_empty_legacy_journal(
                environ,
                target_path=target,
                confirmation=MIGRATE_CONFIRMATION,
            )

    assert source.read_bytes() == source_bytes
    assert existing.read_text() == "do-not-overwrite"
    assert linked.is_symlink()

    source_link = tmp_path / "legacy-link.db"
    source_link.symlink_to(source)
    linked_source_environment = dict(environ, SAFETY_JOURNAL_PATH=str(source_link))
    with pytest.raises(RuntimeError, match="cannot be opened safely"):
        migrate_empty_legacy_journal(
            linked_source_environment,
            target_path=tmp_path / "new-target.db",
            confirmation=MIGRATE_CONFIRMATION,
        )
    assert source_link.is_symlink()
    assert source.read_bytes() == source_bytes


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    [
        ("companion-collision.db-wal", "companion-collision.db"),
        ("companion-collision.db", "companion-collision.db-wal"),
    ],
)
def test_legacy_migration_rejects_source_target_sqlite_companion_collisions(
    tmp_path,
    monkeypatch,
    source_name,
    target_name,
):
    source = tmp_path / source_name
    target = tmp_path / target_name
    environ = _legacy_env(tmp_path, journal_path=source)
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_rejects_mixed_case_companion_alias_on_insensitive_volume(
    tmp_path,
    monkeypatch,
):
    probe = tmp_path / "CaseSensitivityProbe"
    probe.touch()
    case_insensitive = (tmp_path / "casesensitivityprobe").exists()
    probe.unlink()
    if not case_insensitive:
        pytest.skip("mixed-case alias requires a case-insensitive filesystem")

    source = tmp_path / "COLLISION.DB-WAL"
    target = tmp_path / "collision.db"
    environ = _legacy_env(tmp_path, journal_path=source)
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_rejects_unicode_normalization_companion_alias(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "caf\N{LATIN SMALL LETTER E WITH ACUTE}.db-wal"
    target = tmp_path / "cafe\N{COMBINING ACUTE ACCENT}.db"
    environ = _legacy_env(tmp_path, journal_path=source)
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_rejects_allocation_ledger_companion_target(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = Path(f"{environ['RT_DB_PATH']}-wal")
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_uses_default_allocation_ledger_resource_family(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    environ.pop("RT_DB_PATH")
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "trading_data.db-wal"
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    monkeypatch.setattr(journal_script, "PROJECT_ROOT", tmp_path)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_rejects_explicitly_empty_allocation_ledger_path(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    environ["RT_DB_PATH"] = "   "
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "must-not-be-created.db"
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="RT_DB_PATH cannot be empty"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


@pytest.mark.parametrize(
    ("environment_name", "suffix"),
    [
        ("LIVE_RT_DB_PATH", "-wal"),
        ("LIVE_SAFETY_JOURNAL_PATH", "-shm"),
    ],
)
def test_legacy_migration_rejects_live_sqlite_companion_target(
    tmp_path,
    monkeypatch,
    environment_name,
    suffix,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    live_path = tmp_path / f"{environment_name.lower()}.db"
    environ[environment_name] = str(live_path)
    target = Path(f"{live_path}{suffix}")
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    assert not target.exists()
    _allow_migration_lock(monkeypatch)

    with pytest.raises(ValueError, match="resource families must be pairwise disjoint"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert not target.exists()


def test_legacy_migration_rejects_zero_byte_target_raced_after_precheck(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "raced-zero-byte-target.db"
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    raced_identity = None
    original_initialize_new = SafetyJournal.initialize_new
    _allow_migration_lock(monkeypatch)

    def inject_zero_byte_race(self, **kwargs):
        nonlocal raced_identity
        if self.database_path == target:
            target.touch(exist_ok=False)
            raced_identity = (target.stat().st_dev, target.stat().st_ino)
        return original_initialize_new(self, **kwargs)

    monkeypatch.setattr(SafetyJournal, "initialize_new", inject_zero_byte_race)
    with pytest.raises(JournalIntegrityError, match="appeared during exclusive initialization"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert target.read_bytes() == b""
    assert (target.stat().st_dev, target.stat().st_ino) == raced_identity


def test_legacy_migration_rejects_symlink_target_raced_after_precheck(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "raced-symlink-target.db"
    victim = tmp_path / "operator-owned.db"
    victim.write_bytes(b"operator-owned")
    victim_identity = (victim.stat().st_dev, victim.stat().st_ino)
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    original_initialize_new = SafetyJournal.initialize_new
    _allow_migration_lock(monkeypatch)

    def inject_symlink_race(self, **kwargs):
        if self.database_path == target:
            target.symlink_to(victim)
        return original_initialize_new(self, **kwargs)

    monkeypatch.setattr(SafetyJournal, "initialize_new", inject_symlink_race)
    with pytest.raises(JournalIntegrityError, match="appeared during exclusive initialization"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert target.is_symlink()
    assert target.readlink() == victim
    assert victim.read_bytes() == b"operator-owned"
    assert (victim.stat().st_dev, victim.stat().st_ino) == victim_identity


def test_legacy_migration_rejects_target_replacement_after_exclusive_reservation(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "replaced-after-reservation.db"
    displaced = tmp_path / "displaced-reservation.db"
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    replacement_identity = None
    original_initialize = SafetyJournal.initialize
    _allow_migration_lock(monkeypatch)

    def replace_reserved_target(self, **kwargs):
        nonlocal replacement_identity
        if self.database_path == target:
            target.rename(displaced)
            target.touch(exist_ok=False)
            replacement_identity = (target.stat().st_dev, target.stat().st_ino)
        return original_initialize(self, **kwargs)

    monkeypatch.setattr(SafetyJournal, "initialize", replace_reserved_target)
    with pytest.raises(JournalIntegrityError, match="inode replayed at startup"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert target.read_bytes() == b""
    assert (target.stat().st_dev, target.stat().st_ino) == replacement_identity
    assert displaced.read_bytes() == b""


def test_legacy_migration_rejects_source_replacement_before_creation(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "account-bound-safety.db"
    original = source.read_bytes()
    original_assert = journal_script._assert_same_source
    replaced = False
    _allow_migration_lock(monkeypatch)

    def replace_before_assert(path, expected_bytes, expected_identity):
        nonlocal replaced
        if not replaced:
            replaced = True
            archived = source.with_suffix(".original")
            source.rename(archived)
            source.write_bytes(original)
        return original_assert(path, expected_bytes, expected_identity)

    monkeypatch.setattr(journal_script, "_assert_same_source", replace_before_assert)
    with pytest.raises(RuntimeError, match="changed during migration"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert not target.exists()


def test_legacy_migration_interruption_keeps_source_and_leaves_target_fail_closed(
    tmp_path,
    monkeypatch,
):
    environ = _legacy_env(tmp_path)
    source = Path(environ["SAFETY_JOURNAL_PATH"])
    target = tmp_path / "account-bound-safety.db"
    source_bytes = source.read_bytes()
    source_identity = (source.stat().st_dev, source.stat().st_ino)
    original_replay = SafetyJournal.replay_and_bind_runtime_path
    _allow_migration_lock(monkeypatch)

    def interrupt_target_replay(self, **kwargs):
        if self.database_path == target:
            raise RuntimeError("forced interruption after target creation")
        return original_replay(self, **kwargs)

    monkeypatch.setattr(
        SafetyJournal,
        "replay_and_bind_runtime_path",
        interrupt_target_replay,
    )
    with pytest.raises(RuntimeError, match="forced interruption"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )

    assert source.read_bytes() == source_bytes
    assert (source.stat().st_dev, source.stat().st_ino) == source_identity
    assert target.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        migrate_empty_legacy_journal(
            environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )


def test_relative_journal_path_is_anchored_to_project_root(monkeypatch, tmp_path):
    environ = _env(tmp_path)
    project_root = tmp_path / "project"
    project_root.mkdir()
    journal_path = project_root / "data" / "paper-safety.db"
    journal_path.parent.mkdir()
    monkeypatch.setattr(config_module, "_PROJECT_ROOT", project_root)
    monkeypatch.setattr(journal_script, "PROJECT_ROOT", project_root)
    environ["SAFETY_JOURNAL_PATH"] = os.path.relpath(journal_path, project_root)
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    wrong_cwd_path = (unrelated_cwd / environ["SAFETY_JOURNAL_PATH"]).resolve()
    assert wrong_cwd_path != journal_path.resolve()
    monkeypatch.chdir(unrelated_cwd)

    contract = initialize_journal(environ, confirmation=CREATE_CONFIRMATION)
    verified = verify_journal(environ)

    assert Path(contract.safety_journal_path) == journal_path.resolve()
    assert verified.safety_journal_path == contract.safety_journal_path
    assert journal_path.exists()
    assert not wrong_cwd_path.exists()


def test_verify_rejects_configured_symlink_to_valid_same_identity_journal(tmp_path):
    target_env = _env(tmp_path)
    target_path = tmp_path / "target-safety.db"
    target_env["SAFETY_JOURNAL_PATH"] = str(target_path)
    initialize_journal(target_env, confirmation=CREATE_CONFIRMATION)
    original = target_path.read_bytes()

    configured_link = tmp_path / "configured-safety.db"
    configured_link.symlink_to(target_path)
    linked_env = dict(target_env)
    linked_env["SAFETY_JOURNAL_PATH"] = str(configured_link)

    with pytest.raises(JournalIntegrityError, match="non-symlink regular file"):
        verify_journal(linked_env)

    assert configured_link.is_symlink()
    assert target_path.read_bytes() == original


def test_verify_rejects_active_or_quarantined_submission_authority(
    tmp_path,
    monkeypatch,
):
    environ = _env(tmp_path)
    initialize_journal(environ, confirmation=CREATE_CONFIRMATION)
    now = datetime.now(timezone.utc)
    scope = environ["SAFETY_ACCOUNT_SCOPE"]
    domain = "paper-simulator-v1"
    intent = OrderIntent(
        execution_domain_scope=domain,
        account_scope=scope,
        portfolio_id="default",
        con_id=265598,
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        account_current_quantity=Decimal("2"),
        target_quantity=Decimal("1"),
        portfolio_current_quantity=Decimal("2"),
        portfolio_target_quantity=Decimal("1"),
        created_at=now,
        reduce_only=True,
    )
    exposure = ExposureEvidence(
        execution_domain_scope=domain,
        account_scope=scope,
        con_id=265598,
        symbol="AAPL",
        position_quantity=Decimal("2"),
        observed_at=now,
        status=EvidenceStatus.AUTHORITATIVE,
        source="test-account",
        snapshot_id="account-snapshot",
    )
    allocation = PortfolioAllocationEvidence(
        execution_domain_scope=domain,
        account_scope=scope,
        portfolio_id="default",
        con_id=265598,
        symbol="AAPL",
        position_quantity=Decimal("2"),
        aggregate_allocated_quantity=Decimal("2"),
        has_offsetting_allocations=False,
        observed_at=now,
        status=EvidenceStatus.AUTHORITATIVE,
        source="test-allocation",
        snapshot_id="allocation-snapshot",
    )
    gates = GateContext(
        execution_domain_scope=domain,
        account_scope=scope,
        con_id=265598,
        evaluated_at=now,
        max_evidence_age_seconds=30,
        transport_state=TransportState.CONNECTED,
        reconciliation_status=ReconciliationStatus.PASSED,
        open_orders_complete=True,
        open_orders_all_clients=True,
        open_orders_snapshot_stable=True,
        open_orders_observed_at=now,
        open_orders_snapshot_id="orders-snapshot",
        active_order_count=0,
    )
    descriptor = SubmissionDescriptor(
        execution_domain_scope=domain,
        account_scope=scope,
        con_id=265598,
        side=OrderSide.SELL,
        quantity=Decimal("1"),
        order_type=OrderType.MARKET,
        limit_price=None,
        stop_price=None,
        time_in_force=TimeInForce.DAY,
        outside_regular_hours=False,
        order_ref="test-close",
    )
    journal = SafetyJournal(environ["SAFETY_JOURNAL_PATH"], clock=lambda: now)
    journal.authorize_submission(
        "active-test",
        intent,
        exposure,
        allocation,
        gates,
        descriptor,
    )

    with pytest.raises(RuntimeStartupBlocked) as exc_info:
        verify_journal(environ)

    assert "ACTIVE_RESERVATION_AT_STARTUP" in exc_info.value.reason_codes
    assert "QUARANTINED_RESERVATION_AT_STARTUP" in exc_info.value.reason_codes

    original = Path(environ["SAFETY_JOURNAL_PATH"]).read_bytes()
    target = tmp_path / "must-not-exist.db"
    legacy_environ = dict(environ)
    legacy_environ.pop("SAFETY_ACCOUNT_SCOPE_KEY")
    _allow_migration_lock(monkeypatch)
    with pytest.raises(RuntimeError, match="not empty"):
        migrate_empty_legacy_journal(
            legacy_environ,
            target_path=target,
            confirmation=MIGRATE_CONFIRMATION,
        )
    assert Path(environ["SAFETY_JOURNAL_PATH"]).read_bytes() == original
    assert not target.exists()

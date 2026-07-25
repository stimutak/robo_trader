import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import robo_trader.config as config_module
import scripts.manage_paper_safety_journal as journal_script
from robo_trader.config import _derive_safety_account_scope
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
    generate_account_scope,
    initialize_journal,
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


def test_verify_rejects_active_or_quarantined_submission_authority(tmp_path):
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

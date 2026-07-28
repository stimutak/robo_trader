from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from robo_trader.bootstrap_evidence_auth import ed25519_public_key_fingerprint
from scripts import provision_bootstrap_evidence_keys as provisioning
from scripts.provision_bootstrap_evidence_keys import (
    BROKER_PRIVATE_KEY_FILENAME,
    CONFIRMATION,
    PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
    RECONCILIATION_PRIVATE_KEY_FILENAME,
    BootstrapKeyProvisioningError,
    provision_bootstrap_evidence_keys,
)

_TRUST_FILENAMES = {
    "broker_snapshot_ed25519_public.pem",
    "manifest.json",
    "protective_mark_ed25519_public.pem",
    "reconciliation_report_ed25519_public.pem",
}


def _fake_project(path: Path) -> Path:
    trust = path / "robo_trader" / "bootstrap_evidence_trust"
    trust.mkdir(parents=True)
    return path


def _trust_snapshot(project: Path) -> dict[str, bytes]:
    trust = project / "robo_trader" / "bootstrap_evidence_trust"
    assert {path.name for path in trust.iterdir()} == _TRUST_FILENAMES
    return {path.name: path.read_bytes() for path in trust.iterdir()}


def _provision(project: Path, capability: Path) -> dict[str, object]:
    return provision_bootstrap_evidence_keys(
        project_root=project,
        capability_directory=capability,
        confirmation=CONFIRMATION,
    )


def _assert_no_transaction_directory(project: Path) -> None:
    parent = project / "robo_trader"
    assert not list(parent.glob(".bootstrap_evidence_trust.provision-*"))


def test_provisioning_creates_three_external_isolated_capabilities_and_public_roots(
    tmp_path: Path,
) -> None:
    project = _fake_project(tmp_path / "project")
    capability = tmp_path / "capabilities"

    result = _provision(project, capability)

    assert result["authorizes_startup"] is False
    assert result["private_keys_exported"] is False
    assert stat.S_IMODE(capability.stat().st_mode) == 0o700
    filenames = {
        BROKER_PRIVATE_KEY_FILENAME,
        RECONCILIATION_PRIVATE_KEY_FILENAME,
        PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
    }
    private_fingerprints: set[str] = set()
    for filename in filenames:
        path = capability / filename
        assert stat.S_IMODE(path.stat().st_mode) == 0o400
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        assert isinstance(key, Ed25519PrivateKey)
        private_fingerprints.add(ed25519_public_key_fingerprint(key.public_key()))
    assert len(private_fingerprints) == 3
    assert not any(project.rglob("*private*.pem"))

    trust = project / "robo_trader" / "bootstrap_evidence_trust"
    manifest = json.loads((trust / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["public_key_fingerprints"].values()) == private_fingerprints
    canonical = json.dumps(
        {
            "producer_ids": manifest["producer_ids"],
            "public_key_fingerprints": manifest["public_key_fingerprints"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    assert manifest["trust_set_digest"] == hashlib.sha256(canonical.encode()).hexdigest()
    _assert_no_transaction_directory(project)


def test_provisioning_requires_exact_confirmation_before_any_change(tmp_path: Path) -> None:
    project = _fake_project(tmp_path / "project")
    capability = tmp_path / "capabilities"

    with pytest.raises(BootstrapKeyProvisioningError, match="exact provisioning confirmation"):
        provision_bootstrap_evidence_keys(
            project_root=project,
            capability_directory=capability,
            confirmation="yes",
        )

    assert not capability.exists()
    assert list((project / "robo_trader" / "bootstrap_evidence_trust").iterdir()) == []


def test_reprovisioning_replaces_the_complete_trust_set_in_one_transaction(
    tmp_path: Path,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)

    result = _provision(project, tmp_path / "capabilities-new")

    after = _trust_snapshot(project)
    assert after.keys() == before.keys()
    assert all(after[name] != before[name] for name in after)
    manifest = json.loads(after["manifest.json"])
    assert result["trust_set_digest"] == manifest["trust_set_digest"]
    _assert_no_transaction_directory(project)


@pytest.mark.parametrize(
    "boundary",
    [
        "AFTER_CAPABILITY_DIRECTORY_CREATED",
        "AFTER_PRIVATE_KEYS_WRITTEN",
        "AFTER_TRUST_STAGE_CREATED",
        "AFTER_TRUST_STAGE_WRITTEN",
        "AFTER_TRUST_STAGE_SYNCED",
        "AFTER_TRUST_STAGE_VALIDATED",
        "BEFORE_TRUST_EXCHANGE",
        "AFTER_TRUST_EXCHANGE",
        "AFTER_PARENT_SYNC",
        "AFTER_LIVE_VALIDATION",
    ],
)
def test_failure_at_each_transaction_boundary_preserves_the_old_complete_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"

    def fail_at_boundary(current: str) -> None:
        if current == boundary:
            raise RuntimeError(f"injected failure at {boundary}")

    monkeypatch.setattr(provisioning, "_provisioning_fault", fail_at_boundary)

    with pytest.raises(RuntimeError, match=boundary):
        _provision(project, capability)

    assert _trust_snapshot(project) == before
    assert not capability.exists()
    _assert_no_transaction_directory(project)


def test_partial_public_stage_write_never_mutates_the_live_trust_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"
    original_write = provisioning._write_new_public_file
    calls = 0

    def fail_second_write(
        directory_descriptor: int,
        filename: str,
        payload: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected staged write failure")
        original_write(directory_descriptor, filename, payload)

    monkeypatch.setattr(provisioning, "_write_new_public_file", fail_second_write)

    with pytest.raises(OSError, match="injected staged write failure"):
        _provision(project, capability)

    assert _trust_snapshot(project) == before
    assert not capability.exists()
    _assert_no_transaction_directory(project)


def test_private_key_write_failure_revokes_the_partial_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"
    original_write = provisioning._write_new_private_key
    calls = 0

    def fail_second_write(
        directory_descriptor: int,
        filename: str,
        payload: bytes,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected private write failure")
        original_write(directory_descriptor, filename, payload)

    monkeypatch.setattr(provisioning, "_write_new_private_key", fail_second_write)

    with pytest.raises(OSError, match="injected private write failure"):
        _provision(project, capability)

    assert _trust_snapshot(project) == before
    assert not capability.exists()
    _assert_no_transaction_directory(project)


def test_atomic_exchange_failure_preserves_old_trust_and_revokes_new_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"

    def fail_exchange(_parent: int, _left: str, _right: str) -> None:
        raise OSError("injected atomic exchange failure")

    monkeypatch.setattr(provisioning, "_atomic_exchange_directories", fail_exchange)

    with pytest.raises(OSError, match="injected atomic exchange failure"):
        _provision(project, capability)

    assert _trust_snapshot(project) == before
    assert not capability.exists()
    _assert_no_transaction_directory(project)


def test_rollback_primitive_failure_retains_both_recovery_tree_and_active_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"
    original_exchange = provisioning._atomic_exchange_directories
    exchanges = 0

    def fail_rollback(parent: int, left: str, right: str) -> None:
        nonlocal exchanges
        exchanges += 1
        if exchanges == 2:
            raise OSError("injected rollback failure")
        original_exchange(parent, left, right)

    def fail_after_exchange(boundary: str) -> None:
        if boundary == "AFTER_TRUST_EXCHANGE":
            raise RuntimeError("force rollback")

    monkeypatch.setattr(provisioning, "_atomic_exchange_directories", fail_rollback)
    monkeypatch.setattr(provisioning, "_provisioning_fault", fail_after_exchange)

    with pytest.raises(
        BootstrapKeyProvisioningError,
        match="rollback failed; transaction state requires recovery",
    ):
        _provision(project, capability)

    assert capability.is_dir()
    assert _trust_snapshot(project) != before
    recovery_directories = list(
        (project / "robo_trader").glob(".bootstrap_evidence_trust.provision-*")
    )
    assert len(recovery_directories) == 1
    recovery = recovery_directories[0]
    assert {path.name: path.read_bytes() for path in recovery.iterdir()} == before


def test_failed_post_exchange_live_validation_rolls_back_the_exact_old_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _fake_project(tmp_path / "project")
    _provision(project, tmp_path / "capabilities-old")
    before = _trust_snapshot(project)
    capability = tmp_path / "capabilities-new"
    original_validate = provisioning._validate_complete_trust_set
    calls = 0

    def fail_live_validation(directory_descriptor: int) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise BootstrapKeyProvisioningError("injected live validation failure")
        return original_validate(directory_descriptor)

    monkeypatch.setattr(
        provisioning,
        "_validate_complete_trust_set",
        fail_live_validation,
    )

    with pytest.raises(BootstrapKeyProvisioningError, match="live validation failure"):
        _provision(project, capability)

    assert _trust_snapshot(project) == before
    assert not capability.exists()
    _assert_no_transaction_directory(project)

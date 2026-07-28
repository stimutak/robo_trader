from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from robo_trader.bootstrap_evidence_auth import ed25519_public_key_fingerprint
from scripts.provision_bootstrap_evidence_keys import (
    BROKER_PRIVATE_KEY_FILENAME,
    CONFIRMATION,
    PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
    RECONCILIATION_PRIVATE_KEY_FILENAME,
    BootstrapKeyProvisioningError,
    provision_bootstrap_evidence_keys,
)


def _fake_project(path: Path) -> Path:
    trust = path / "robo_trader" / "bootstrap_evidence_trust"
    trust.mkdir(parents=True)
    return path


def test_provisioning_creates_three_external_isolated_capabilities_and_public_roots(
    tmp_path: Path,
) -> None:
    project = _fake_project(tmp_path / "project")
    capability = tmp_path / "capabilities"

    result = provision_bootstrap_evidence_keys(
        project_root=project,
        capability_directory=capability,
        confirmation=CONFIRMATION,
    )

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

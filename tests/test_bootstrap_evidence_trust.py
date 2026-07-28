from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import robo_trader.bootstrap_evidence_auth as auth

_RECEIPTS = {
    "broker_snapshot": {
        "account_scope": "acct_v1_" + "b" * 64,
        "artifact_kind": "broker_snapshot",
        "artifact_sha256": "6684b47d5e0afdbae3f0591e860e6b80274a941b2574c07854903be4c68c4a73",
        "expires_at": "2026-07-28T12:05:00.000000Z",
        "issued_at": "2026-07-28T12:00:00.000000Z",
        "producer_id": "robotrader-broker-snapshot-producer-v1",
        "public_key_fingerprint": "173cdf680bd80fac65515f6c2948315fc672dbfcc54a0b8403b38d3ef4b83958",
        "receipt_id": "bevr-v2-3df4737926b4bed98e8e9fb2f2ab1257e80de48a9b31011a948bb331bc676d3b",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "dfJPRw5vHKI+P2aCqZjSOLBq9G8AsQ7wJN8djMxREPC36aE6RMCGyn4u92gtlY5gRXnJbcLnCFoMSNbsdfnqDA==",
    },
    "reconciliation_report": {
        "account_scope": "acct_v1_" + "b" * 64,
        "artifact_kind": "reconciliation_report",
        "artifact_sha256": "42643b32fa7414877e6f10bd61d75675080f3d5a6373ade000795f75d5a3fd1c",
        "expires_at": "2026-07-28T12:05:00.000000Z",
        "issued_at": "2026-07-28T12:00:00.000000Z",
        "producer_id": "robotrader-reconciliation-producer-v1",
        "public_key_fingerprint": "c5850cd6a7544d672428eb6804c97410f2de39aed9d7bc19e6893c708adcf917",
        "receipt_id": "bevr-v2-94464023895255b02872c2839f95d9f26006d3674d1e1acb65201687c79b507f",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "cvLcvVpMsQqMTuKtUk6ujhGYDQPq53XEGfXSjassySein32ZLGwjJd+6xVyvd9jwoWjA5JXOcMng0EhlI0m9Dg==",
    },
    "protective_mark": {
        "account_scope": "acct_v1_" + "b" * 64,
        "artifact_kind": "protective_mark",
        "artifact_sha256": "a03013dacfa30bac41f413068e91852ea556ce7ed31702e5a9c2df73f20a7dfb",
        "expires_at": "2026-07-28T12:05:00.000000Z",
        "issued_at": "2026-07-28T12:00:00.000000Z",
        "producer_id": "robotrader-protective-mark-producer-v1",
        "public_key_fingerprint": "77f355308d372426229d4b12d13aa582314a71d0c58e34b59aba11e90e535388",
        "receipt_id": "bevr-v2-cdefad2d21d009c1c9787b76c1af1489c50b04adede6592fd48613bd4062ae00",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "uI9J7/SSF8yV5e0Aq6Az/NTcPKvRYcIcAnZCwAdmE82eciGsQRz6ViDnxonG/yGscAOPONyv0fZ377uqmt9dAA==",
    },
}


@pytest.mark.parametrize("artifact_kind", sorted(_RECEIPTS))
def test_static_producer_fixture_verifies_only_with_its_pinned_key(
    artifact_kind: str,
) -> None:
    raw = _RECEIPTS[artifact_kind]
    receipt = auth.verify_receipt(
        raw=raw,
        artifact_kind=artifact_kind,
        artifact_sha256=raw["artifact_sha256"],
        runtime_fingerprint="0123456789abcdef",
        account_scope="acct_v1_" + "b" * 64,
        now=datetime(2026, 7, 28, 12, 1, tzinfo=timezone.utc),
    )
    assert receipt.artifact_kind == artifact_kind
    assert receipt.public_key_fingerprint == raw["public_key_fingerprint"]


def test_cross_kind_and_arbitrary_json_cannot_mint_authority() -> None:
    broker = _RECEIPTS["broker_snapshot"]
    with pytest.raises(auth.BootstrapEvidenceAuthenticationError, match="wrong producer"):
        auth.verify_receipt(
            raw=broker,
            artifact_kind="protective_mark",
            artifact_sha256=broker["artifact_sha256"],
            runtime_fingerprint="0123456789abcdef",
            account_scope="acct_v1_" + "b" * 64,
            now=datetime(2026, 7, 28, 12, 1, tzinfo=timezone.utc),
        )
    forged = dict(broker)
    forged["signature_ed25519"] = "A" * 88
    with pytest.raises(auth.BootstrapEvidenceAuthenticationError, match="signature is invalid"):
        auth.verify_receipt(
            raw=forged,
            artifact_kind="broker_snapshot",
            artifact_sha256=broker["artifact_sha256"],
            runtime_fingerprint="0123456789abcdef",
            account_scope="acct_v1_" + "b" * 64,
            now=datetime(2026, 7, 28, 12, 1, tzinfo=timezone.utc),
        )


def test_verifier_tree_contains_no_private_key_or_generic_signer() -> None:
    package_root = Path(auth.__file__).resolve().parent
    assert not list(package_root.rglob("*private*.pem"))
    assert "BEGIN PRIVATE KEY" not in Path(auth.__file__).read_text()
    assert not (
        package_root.parent / "scripts" / "emit_exact_state_bootstrap_evidence_receipt.py"
    ).exists()
    assert not any(name.startswith("emit_") for name in dir(auth))


def test_environment_cannot_substitute_a_pinned_trust_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOOTSTRAP_BROKER_EVIDENCE_PUBLIC_KEY_PATH", "/tmp/attacker.pem")
    with pytest.raises(auth.BootstrapEvidenceAuthenticationError, match="refuses"):
        auth.bootstrap_evidence_trust_public_dict()

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
        "public_key_fingerprint": "f07b78427afc1c9ac34008de0d22da47636c3f2c04cb34783cbb34caa0c394e2",
        "receipt_id": "bevr-v2-a4984e68e5e5976a3b00e0948fdcca6571d8dccc0d04f630fe587313c7248b35",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "G7iUJCYNksZu7QshVnv3Xe5+r0yjSrIjvzpUv2pbGDbpzvSDeQqkqs4dMBtklHqbDrnD7eNeWyFeYbDrIlraBg==",
    },
    "reconciliation_report": {
        "account_scope": "acct_v1_" + "b" * 64,
        "artifact_kind": "reconciliation_report",
        "artifact_sha256": "42643b32fa7414877e6f10bd61d75675080f3d5a6373ade000795f75d5a3fd1c",
        "expires_at": "2026-07-28T12:05:00.000000Z",
        "issued_at": "2026-07-28T12:00:00.000000Z",
        "producer_id": "robotrader-reconciliation-producer-v1",
        "public_key_fingerprint": "48c087ce23c8bdc95cca3c11c1372631420b2a11f6ef062c126736dd05b59c13",
        "receipt_id": "bevr-v2-428e81227dcd5871d80c5f304ae8e0fdb5cba5e7d9141f8e20cf0b73e9b6d5ee",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "zhRMptI9foTnOGrChTj2OBb6Y+XW2EEJ+EdNvZJ5K4mx97+MXee783O5VqVvA++GN+qtOSrwp6s+1S8hAZV4BQ==",
    },
    "protective_mark": {
        "account_scope": "acct_v1_" + "b" * 64,
        "artifact_kind": "protective_mark",
        "artifact_sha256": "a03013dacfa30bac41f413068e91852ea556ce7ed31702e5a9c2df73f20a7dfb",
        "expires_at": "2026-07-28T12:05:00.000000Z",
        "issued_at": "2026-07-28T12:00:00.000000Z",
        "producer_id": "robotrader-protective-mark-producer-v1",
        "public_key_fingerprint": "fa8963176466eb433593799177bea797cf63d8d2a34858afb48732374b2c5743",
        "receipt_id": "bevr-v2-b9561e4f28c128e0dcaf816c25d3f0747ce042adfe4016fcdf5cabbd2666b718",
        "runtime_fingerprint": "0123456789abcdef",
        "schema_version": 2,
        "signature_ed25519": "gi2aFAbBsGwJN6zagmHJ86eZMP+YErpBporfcLdbAumsxW3q4wh8VJz7BUXby/5xYgHGH+Vh2oCe0+fXiBXSCg==",
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

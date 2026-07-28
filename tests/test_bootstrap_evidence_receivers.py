from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import scripts.produce_exact_state_bootstrap_evidence as evidence_cli
from robo_trader import bootstrap_evidence_auth as auth
from robo_trader.bootstrap_evidence_receivers import (
    BROKER_PRIVATE_KEY_FILENAME,
    PROTECTIVE_MARK_PRIVATE_KEY_FILENAME,
    RECONCILIATION_PRIVATE_KEY_FILENAME,
    BootstrapEvidenceReceiverError,
    ProtectiveMarkBundleIdentity,
    assert_protective_mark_receiver_capability,
    assert_reconciliation_receiver_capability,
    create_bootstrap_evidence_receivers,
)
from robo_trader.config import RuntimeContract
from robo_trader.financial_state_bootstrap import load_exact_state_bootstrap_evidence
from robo_trader.reconciliation import bootstrap_producer as reconciliation_producer
from robo_trader.reconciliation.domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    NormalizedBrokerAccount,
    NormalizedBrokerSnapshot,
)
from robo_trader.reconciliation.ibkr_adapter import (
    CompletedOrderCollectionScope,
    ExecutionCollectionScope,
    IBKRDiagnosticSnapshotProvider,
    _assert_broker_result_registration_consumed,
    _produce_broker_snapshot_result,
    _register_broker_result,
)
from robo_trader.safety.journal import SafetyJournal
from scripts.produce_exact_state_bootstrap_evidence import (
    produce_bootstrap_evidence_bundle,
)

ACCOUNT_SCOPE = "acct_v1_" + "0123456789abcdef" * 4


def _ledger(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE portfolios (id TEXT PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL, quantity INTEGER NOT NULL, avg_cost REAL NOT NULL,
                market_price REAL, timestamp DATETIME
            );
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL, side TEXT NOT NULL, quantity INTEGER NOT NULL,
                price REAL NOT NULL, notional REAL DEFAULT 0, slippage REAL DEFAULT 0,
                commission REAL DEFAULT 0, pnl REAL DEFAULT NULL, timestamp DATETIME
            );
            CREATE TABLE account (
                portfolio_id TEXT PRIMARY KEY, cash REAL NOT NULL, equity REAL NOT NULL,
                daily_pnl REAL DEFAULT 0, realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0, timestamp DATETIME
            );
            CREATE TABLE equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, portfolio_id TEXT NOT NULL,
                date TEXT NOT NULL, equity REAL NOT NULL, cash REAL DEFAULT 0,
                positions_value REAL DEFAULT 0, realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0, timestamp DATETIME
            );
            INSERT INTO portfolios VALUES ('default', 'Default');
            INSERT INTO account VALUES
              ('default', 1000, 1000, 0, 0, 0, '2026-07-28T11:59:00+00:00');
            INSERT INTO equity_history(
                portfolio_id,date,equity,cash,positions_value,realized_pnl,
                unrealized_pnl,timestamp
            ) VALUES
              ('default','2026-07-28',1000,1000,0,0,0,'2026-07-28T11:59:00+00:00');
            """)
    SafetyJournal(path.with_name("safety-journal.db")).initialize(
        execution_domain_scope="paper-simulator-v1",
        account_scope=ACCOUNT_SCOPE,
    )


def _install_test_trust(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    trust_root = root / "trust"
    trust_root.mkdir()
    capability = root / "capabilities"
    capability.mkdir(mode=0o700)
    definitions = {
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
    fingerprints: dict[str, str] = {}
    public_paths: dict[str, Path] = {}
    producers: dict[str, str] = {}
    for kind, (private_name, public_name, producer_id) in definitions.items():
        key = Ed25519PrivateKey.generate()
        private_path = capability / private_name
        private_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        private_path.chmod(0o400)
        public_path = trust_root / public_name
        public_path.write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        )
        public_path.chmod(0o600)
        public_paths[kind] = public_path
        fingerprints[kind] = auth.ed25519_public_key_fingerprint(key.public_key())
        producers[kind] = producer_id
    canonical = json.dumps(
        {"producer_ids": producers, "public_key_fingerprints": fingerprints},
        sort_keys=True,
        separators=(",", ":"),
    )
    manifest = trust_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "producer_ids": producers,
                "public_key_fingerprints": fingerprints,
                "schema_version": 1,
                "trust_set_digest": hashlib.sha256(canonical.encode()).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    monkeypatch.setattr(auth, "_PINNED_PUBLIC_KEY_PATHS", public_paths)
    monkeypatch.setattr(auth, "_TRUST_MANIFEST_PATH", manifest)
    return capability


def _runtime(database: Path) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(database),
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
        safety_account_scope=ACCOUNT_SCOPE,
        safety_execution_domain_scope="paper-simulator-v1",
        safety_journal_path=str(database.with_name("safety-journal.db")),
    )


def _broker_result(now: datetime):
    evidence = tuple(
        BrokerCollectionEvidence(
            account_scope=ACCOUNT_SCOPE,
            collection=kind,
            evidence_id="broker-collection-v1-" + hashlib.sha256(kind.value.encode()).hexdigest(),
            result_count=0,
            observed_at=now,
        )
        for kind in BrokerCollectionKind
    )
    snapshot = NormalizedBrokerSnapshot(
        account=NormalizedBrokerAccount(
            account_scope=ACCOUNT_SCOPE,
            account_alias="***1234",
            account_type="paper",
            base_currency="USD",
            total_cash=Decimal("1000"),
            buying_power=Decimal("4000"),
            observed_at=now,
        ),
        observed_from=now - timedelta(seconds=1),
        observed_through=now,
        retrieved_at=now,
        completeness=BrokerEvidenceCompleteness(
            account=True,
            positions=True,
            open_orders=True,
            completed_orders=True,
            executions=True,
            commissions=True,
        ),
        collection_evidence=evidence,
    )
    scope = CompletedOrderCollectionScope(
        kind="ibkr_current_retained_completed_orders",
        api_method="reqCompletedOrders",
        api_only=False,
        client_scope="api_and_manual_orders_visible_to_current_tws_session",
        request_count=2,
        stability_check="identical_second_read",
        retention_scope="current_tws_or_gateway_retained_set",
        full_history=False,
        request_started_at=now - timedelta(milliseconds=400),
        request_completed_at=now - timedelta(milliseconds=300),
        verification_started_at=now - timedelta(milliseconds=200),
        verification_completed_at=now - timedelta(milliseconds=100),
        broker_time_before=now - timedelta(seconds=1),
        broker_time_after=now,
    )
    execution_scope = ExecutionCollectionScope(
        kind="broker_date_since_midnight",
        start_at=now.replace(hour=0, minute=0, second=0, microsecond=0),
        end_at=now,
        retention_scope="ibkr_gateway_broker_date_since_midnight",
        full_history=False,
        commission_scope="matching_callbacks_for_returned_executions",
    )
    producer = object.__new__(IBKRDiagnosticSnapshotProvider)
    return _produce_broker_snapshot_result(
        producer,
        snapshot=snapshot,
        completed_order_scope=scope,
        execution_scope=execution_scope,
    )


class _Provider:
    def __init__(self, result: object) -> None:
        self.result = result
        self.closed = False
        self.events: list[str] = []

    async def produce_normalized_snapshot(self, *, receiver: object, max_age_seconds: float):
        assert max_age_seconds == 30.0
        self.events.append("snapshot")
        registration = _register_broker_result(self.result, receiver)
        received = receiver.receive_broker_snapshot_producer_result(self.result)
        _assert_broker_result_registration_consumed(self.result, registration)
        return received

    def issue_protective_quote_source(self, *, runtime_contract: RuntimeContract) -> object:
        del runtime_contract
        self.events.append("quote-source")
        return self

    async def close(self) -> None:
        self.events.append("close")
        self.closed = True


@pytest.mark.asyncio
async def test_full_empty_position_evidence_chain_is_typed_and_one_shot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _install_test_trust(monkeypatch, tmp_path)
    database = tmp_path / "ledger.db"
    _ledger(database)
    runtime = _runtime(database)
    receivers = create_bootstrap_evidence_receivers(
        runtime_contract=runtime,
        capability_directory=capability,
        output_directory=tmp_path / "evidence",
    )
    result = _broker_result(datetime.now(timezone.utc) - timedelta(milliseconds=50))
    provider = _Provider(result)

    async def no_marks(**kwargs: object) -> tuple:
        assert kwargs["mark_identities"] == ()
        assert kwargs["quote_source"] is provider
        assert provider.closed is False
        mark_identity = assert_protective_mark_receiver_capability(
            kwargs["receiver"],
            runtime_contract=runtime,
        )
        assert type(mark_identity) is ProtectiveMarkBundleIdentity
        assert mark_identity.reconciliation_snapshot_id.startswith("bootstrap-reconciliation-v1-")
        provider.events.append("marks")
        return ()

    report = await produce_bootstrap_evidence_bundle(
        runtime_contract=runtime,
        snapshot_provider=provider,
        receivers=receivers,
        protective_mark_collector=no_marks,
    )

    assert provider.closed is True
    assert provider.events == ["snapshot", "quote-source", "marks", "close"]
    assert report["authorizes_startup"] is False
    assert report["status"] == "EVIDENCE_COMPLETE_GATE_A_STILL_CLOSED"
    assert (tmp_path / "evidence" / "broker_snapshot.json").stat().st_mode & 0o777 == 0o400
    assert (tmp_path / "evidence" / "reconciliation_report.json").is_file()
    loaded = load_exact_state_bootstrap_evidence(
        reconciliation_path=tmp_path / "evidence" / "reconciliation_report.json",
        broker_snapshot_path=tmp_path / "evidence" / "broker_snapshot.json",
        protective_mark_paths=[],
        expected_runtime_contract=runtime,
    )
    assert loaded.safety_journal_identity == runtime.safety_journal_identity
    assert loaded.terminal_settlement_count == 0
    assert loaded.terminal_fill_count == 0
    with pytest.raises(BootstrapEvidenceReceiverError, match="not issued"):
        assert_reconciliation_receiver_capability(receivers.reconciliation_report)

    with pytest.raises(Exception, match="already consumed|absent|released"):
        receivers.broker_snapshot.receive_broker_snapshot_producer_result(result)


def test_invalid_capability_does_not_create_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _install_test_trust(monkeypatch, tmp_path)
    database = tmp_path / "ledger.db"
    _ledger(database)
    (capability / BROKER_PRIVATE_KEY_FILENAME).chmod(0o600)
    output = tmp_path / "evidence"

    with pytest.raises(BootstrapEvidenceReceiverError, match="sealed owner key"):
        create_bootstrap_evidence_receivers(
            runtime_contract=_runtime(database),
            capability_directory=capability,
            output_directory=output,
        )

    assert not output.exists()


def test_capability_directory_rejects_symlinked_ancestor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    capability = _install_test_trust(monkeypatch, real_root)
    alias_root = tmp_path / "redirected"
    alias_root.symlink_to(real_root, target_is_directory=True)
    database = tmp_path / "ledger.db"
    _ledger(database)

    with pytest.raises(BootstrapEvidenceReceiverError, match="external owner-only"):
        create_bootstrap_evidence_receivers(
            runtime_contract=_runtime(database),
            capability_directory=alias_root / capability.name,
            output_directory=tmp_path / "evidence",
        )


@pytest.mark.asyncio
async def test_run_reaps_provider_once_when_receiver_factory_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "ledger.db"
    _ledger(database)
    runtime = _runtime(database)

    class Provider:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1

    provider = Provider()

    async def build_provider(_context: object) -> Provider:
        return provider

    monkeypatch.setattr(
        evidence_cli,
        "validate_runtime_safety",
        lambda *_args: type("Context", (), {"runtime_contract": runtime})(),
    )
    monkeypatch.setattr(evidence_cli, "build_diagnostic_provider", build_provider)
    monkeypatch.setattr(
        evidence_cli,
        "create_bootstrap_evidence_receivers",
        lambda **_kwargs: (_ for _ in ()).throw(
            BootstrapEvidenceReceiverError("capability rejected")
        ),
    )

    with pytest.raises(BootstrapEvidenceReceiverError, match="capability rejected"):
        await evidence_cli._run(
            argparse.Namespace(
                capability_directory=tmp_path / "capabilities",
                output_directory=tmp_path / "evidence",
            )
        )

    assert provider.close_count == 1


def test_public_receivers_expose_no_raw_key_or_generic_signing_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _install_test_trust(monkeypatch, tmp_path)
    database = tmp_path / "ledger.db"
    _ledger(database)
    receivers = create_bootstrap_evidence_receivers(
        runtime_contract=_runtime(database),
        capability_directory=capability,
        output_directory=tmp_path / "evidence",
    )
    try:
        for receiver in (
            receivers.broker_snapshot,
            receivers.reconciliation_report,
            receivers.protective_mark,
        ):
            exposed: list[object] = []
            for receiver_type in type(receiver).__mro__:
                slots = getattr(receiver_type, "__slots__", ())
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot == "__weakref__":
                        continue
                    attribute = (
                        f"_{receiver_type.__name__}{slot}"
                        if slot.startswith("__") and not slot.endswith("__")
                        else slot
                    )
                    exposed.append(getattr(receiver, attribute))
            assert not any(isinstance(value, Ed25519PrivateKey) for value in exposed)
            assert not hasattr(receiver, "sign")
            assert not hasattr(receiver, "sign_bytes")
            assert not any(
                callable(getattr(receiver, name))
                for name in dir(receiver)
                if name.strip("_") in {"sign", "sign_bytes", "sign_payload"}
            )
    finally:
        receivers.close()


@pytest.mark.asyncio
async def test_run_uses_registered_production_mark_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "ledger.db"
    _ledger(database)
    runtime = _runtime(database)

    class Provider:
        def __init__(self) -> None:
            self.close_count = 0

        async def close(self) -> None:
            self.close_count += 1

    provider = Provider()
    receiver_sentinel = object()

    async def build_provider(_context: object) -> Provider:
        return provider

    async def pipeline(**kwargs: object) -> dict[str, object]:
        assert kwargs["snapshot_provider"] is provider
        assert kwargs["receivers"] is receiver_sentinel
        assert kwargs["protective_mark_collector"] is evidence_cli._PRODUCTION_MARK_COLLECTOR
        await provider.close()
        return {"status": "ok"}

    monkeypatch.setattr(
        evidence_cli,
        "validate_runtime_safety",
        lambda *_args: type("Context", (), {"runtime_contract": runtime})(),
    )
    monkeypatch.setattr(evidence_cli, "build_diagnostic_provider", build_provider)
    monkeypatch.setattr(
        evidence_cli,
        "create_bootstrap_evidence_receivers",
        lambda **_kwargs: receiver_sentinel,
    )
    monkeypatch.setattr(evidence_cli, "produce_bootstrap_evidence_bundle", pipeline)

    report = await evidence_cli._run(
        argparse.Namespace(
            capability_directory=tmp_path / "capabilities",
            output_directory=tmp_path / "evidence",
        )
    )

    assert report == {"status": "ok"}
    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_reconciliation_stage_is_removed_when_post_stage_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _install_test_trust(monkeypatch, tmp_path)
    database = tmp_path / "ledger.db"
    _ledger(database)
    runtime = _runtime(database)
    output = tmp_path / "evidence"
    receivers = create_bootstrap_evidence_receivers(
        runtime_contract=runtime,
        capability_directory=capability,
        output_directory=output,
    )
    provider = _Provider(_broker_result(datetime.now(timezone.utc) - timedelta(milliseconds=50)))

    def reject_after_stage(_session: object) -> None:
        raise reconciliation_producer.BootstrapReconciliationBlocked("ledger changed after stage")

    monkeypatch.setattr(
        reconciliation_producer._LedgerCollectionSession,
        "assert_unchanged_after_receiver_claim",
        reject_after_stage,
    )

    with pytest.raises(
        reconciliation_producer.BootstrapReconciliationBlocked,
        match="ledger changed",
    ):
        await produce_bootstrap_evidence_bundle(
            runtime_contract=runtime,
            snapshot_provider=provider,
            receivers=receivers,
            protective_mark_collector=lambda **_kwargs: (),  # type: ignore[arg-type]
        )

    assert provider.closed is True
    assert not (output / "reconciliation_report.json").exists()
    assert not list(output.glob(".reconciliation_report.json*.stage-*"))


@pytest.mark.asyncio
async def test_mark_cancellation_reaps_provider_and_releases_receivers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capability = _install_test_trust(monkeypatch, tmp_path)
    database = tmp_path / "ledger.db"
    _ledger(database)
    runtime = _runtime(database)
    receivers = create_bootstrap_evidence_receivers(
        runtime_contract=runtime,
        capability_directory=capability,
        output_directory=tmp_path / "evidence",
    )
    provider = _Provider(_broker_result(datetime.now(timezone.utc) - timedelta(milliseconds=50)))

    async def cancelled_marks(**_kwargs: object) -> tuple:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await produce_bootstrap_evidence_bundle(
            runtime_contract=runtime,
            snapshot_provider=provider,
            receivers=receivers,
            protective_mark_collector=cancelled_marks,
        )

    assert provider.closed is True
    with pytest.raises(BootstrapEvidenceReceiverError, match="not issued"):
        assert_reconciliation_receiver_capability(receivers.reconciliation_report)

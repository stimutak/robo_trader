"""Synthetic production-integration tests for PR5 runtime reconciliation."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest

import robo_trader.reconciliation.runtime_integration as integration
from robo_trader.bootstrap_evidence_receivers import SealedBootstrapEvidenceArtifact
from robo_trader.config import RuntimeContract
from robo_trader.reconciliation.runtime_integration import (
    ProductionRuntimeEvidenceSource,
    RuntimeReconciliationController,
    RuntimeReconciliationIntegrationError,
    assert_runtime_bootstrap_ready,
    build_runtime_reconciliation_controller,
    read_runtime_reconciliation_status,
)

ACCOUNT_SCOPE = "acct_v1_0123456789abcdef0123456789abcdef" "fedcba9876543210fedcba9876543210"
STATUS_OWNER_BINDING = "a" * 64
BOOTSTRAP_ID = "pboot-" + "1" * 32


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


class _Service:
    def __init__(self, outcome: object, *, eligible: bool = True) -> None:
        self.outcome = outcome
        self.eligible = eligible
        self.close = AsyncMock()

    def entry_eligible(self) -> bool:
        return self.eligible

    async def reconcile_startup(self):
        return self.outcome

    async def reconcile_reconnect(self):
        return self.outcome

    async def reconcile_periodic_if_due(self):
        return self.outcome


def _outcome(now: datetime) -> object:
    return SimpleNamespace(
        state=SimpleNamespace(value="ready"),
        trigger=SimpleNamespace(value="startup"),
        verdict=SimpleNamespace(checked_at=now),
        completed_at=now,
        eligible_until=now + timedelta(seconds=30),
        entry_eligible=True,
        persisted=SimpleNamespace(run_id="run-safe", snapshot_id="snapshot-safe"),
    )


def _artifact(tmp_path: Path, kind: str) -> SealedBootstrapEvidenceArtifact:
    return SealedBootstrapEvidenceArtifact(
        artifact_kind=kind,
        artifact_path=tmp_path / f"{kind}.json",
        authentication_receipt_path=tmp_path / f"{kind}.receipt.json",
        artifact_sha256="a" * 64,
        producer_object_id=f"producer-{kind}",
    )


@pytest.mark.asyncio
async def test_source_uses_one_broker_generation_through_marks_and_runtime_bind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "ledger.db"
    database.touch()
    runtime = _runtime(database)
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    capability_directory = tmp_path / "capabilities"
    evidence_root = tmp_path / "evidence"
    capability_directory.mkdir(mode=0o700)
    evidence_root.mkdir(mode=0o700)

    broker_envelope = object()
    runtime_handoff = object()
    exact_state = object()
    bound_runtime_evidence = object()
    quotes = {
        "AAPL": SimpleNamespace(
            symbol="AAPL",
            con_id=265598,
            transport_generation="generation-1",
        ),
        "MSFT": SimpleNamespace(
            symbol="MSFT",
            con_id=272093,
            transport_generation="generation-1",
        ),
    }

    async def collect_quote(symbols, *, active_symbols):
        assert active_symbols == ("AAPL", "MSFT")
        return (quotes[symbols[0]],)

    quote_source = SimpleNamespace(
        get_protective_quotes=AsyncMock(side_effect=collect_quote),
    )
    provider = SimpleNamespace(
        produce_normalized_snapshot=AsyncMock(return_value=broker_envelope),
        issue_protective_quote_source=MagicMock(return_value=quote_source),
        close=AsyncMock(),
    )
    broker_artifact = _artifact(tmp_path, "broker_snapshot")
    reconciliation_artifact = _artifact(tmp_path, "reconciliation_report")
    mark_artifact = _artifact(tmp_path, "protective_mark")
    receivers = SimpleNamespace(
        broker_snapshot=object(),
        broker_artifact=broker_artifact,
        reconciliation_report=object(),
        protective_mark=object(),
        assert_complete=MagicMock(),
        unpublished_bundle_size_bytes=MagicMock(return_value=1024),
        publish_complete_bundle=MagicMock(),
        published_artifact_path=MagicMock(
            side_effect=lambda artifact: tmp_path / artifact.artifact_path.name
        ),
        discard_unpublished_bundle=MagicMock(),
        close=MagicMock(),
    )
    delivery = SimpleNamespace(
        receiver_result=reconciliation_artifact,
        local_position_identities=(("default", "AAPL"), ("growth", "MSFT")),
        verified_broker_evidence=runtime_handoff,
    )
    produce = MagicMock(return_value=delivery)
    bind = MagicMock(return_value=bound_runtime_evidence)
    monkeypatch.setattr(
        integration,
        "create_bootstrap_evidence_receivers",
        MagicMock(return_value=receivers),
    )
    monkeypatch.setattr(integration, "produce_bootstrap_reconciliation", produce)
    monkeypatch.setattr(
        integration,
        "assert_factory_owned_protective_quote_source",
        MagicMock(return_value=SimpleNamespace(transport_generation="generation-1")),
    )
    monkeypatch.setattr(
        integration,
        "create_runtime_bound_mark_only_producer",
        MagicMock(return_value=object()),
    )
    collect_mark = AsyncMock(return_value=mark_artifact)
    monkeypatch.setattr(
        integration,
        "collect_and_produce_bootstrap_protective_mark",
        collect_mark,
    )
    monkeypatch.setattr(
        integration,
        "load_exact_state_bootstrap_evidence",
        MagicMock(return_value=exact_state),
    )
    monkeypatch.setattr(
        integration,
        "bind_verified_runtime_reconciliation_evidence",
        bind,
    )
    source = ProductionRuntimeEvidenceSource(
        runtime_context=context,
        provider=provider,
        capability_directory=capability_directory,
        evidence_root=evidence_root,
    )

    observed = await source.collect_verified_evidence(max_age_seconds=30.0)

    assert observed is bound_runtime_evidence
    provider.produce_normalized_snapshot.assert_awaited_once_with(
        receiver=receivers.broker_snapshot,
        max_age_seconds=30.0,
    )
    produce.assert_called_once_with(
        broker_envelope,
        runtime,
        receivers.reconciliation_report,
    )
    provider.issue_protective_quote_source.assert_called_once_with(runtime_contract=runtime)
    assert quote_source.get_protective_quotes.await_args_list == [
        call(("AAPL",), active_symbols=("AAPL", "MSFT")),
        call(("MSFT",), active_symbols=("AAPL", "MSFT")),
    ]
    assert collect_mark.await_args_list[0].kwargs["expected_active_symbols"] == (
        "AAPL",
        "MSFT",
    )
    assert collect_mark.await_args_list[1].kwargs["expected_active_symbols"] == (
        "AAPL",
        "MSFT",
    )
    bind.assert_called_once_with(
        runtime_handoff,
        exact_state,
        context,
        receivers.protective_mark,
    )
    provider.close.assert_not_awaited()
    receivers.close.assert_called_once_with()

    await source.close()
    provider.close.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_bundles", "max_bytes"),
    [(1, 1024 * 1024), (100, 5)],
)
async def test_evidence_retention_ceiling_quarantines_without_deleting_audit_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_bundles: int,
    max_bytes: int,
) -> None:
    database = tmp_path / "ledger.db"
    database.touch()
    runtime = _runtime(database)
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    capability_directory = tmp_path / "capabilities"
    evidence_root = tmp_path / "evidence"
    capability_directory.mkdir(mode=0o700)
    evidence_root.mkdir(mode=0o700)
    bundle = evidence_root / ("runtime-reconciliation-" + "a" * 48)
    bundle.mkdir(mode=0o700)
    artifact = bundle / "bundle_complete.json"
    artifact.write_bytes(b"audit-lineage")
    artifact.chmod(0o400)
    before = artifact.read_bytes()
    provider = SimpleNamespace(
        produce_normalized_snapshot=AsyncMock(),
        close=AsyncMock(),
    )
    source = ProductionRuntimeEvidenceSource(
        runtime_context=context,
        provider=provider,
        capability_directory=capability_directory,
        evidence_root=evidence_root,
        max_published_bundles=max_bundles,
        max_published_bytes=max_bytes,
    )

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="retention ceiling reached",
    ):
        await source.collect_verified_evidence(max_age_seconds=30.0)

    assert artifact.read_bytes() == before
    assert bundle.is_dir()
    provider.produce_normalized_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_publishes_only_sanitized_status_and_age(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    service = _Service(_outcome(now))
    status_path = tmp_path / "status.json"
    controller = RuntimeReconciliationController(service, status_path=status_path)

    await controller.reconcile_startup()
    status = read_runtime_reconciliation_status(status_path)

    assert status == {
        "state": "ready",
        "trigger": "startup",
        "completed_at": now.isoformat(),
        "eligible_until": (now + timedelta(seconds=30)).isoformat(),
        "entry_eligible": True,
        "quarantined": False,
        "age_seconds": pytest.approx(0.0, abs=1.0),
    }
    assert "run-safe" not in str(status)
    assert "snapshot-safe" not in str(status)

    await controller.reconcile_reconnect()
    assert read_runtime_reconciliation_status(status_path)["state"] == "ready"


@pytest.mark.asyncio
async def test_status_publication_failure_blocks_entry_eligibility(tmp_path: Path) -> None:
    service = _Service(_outcome(datetime.now(timezone.utc)))
    controller = RuntimeReconciliationController(
        service,
        status_path=tmp_path / "missing" / "status.json",
    )

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="status publication failed closed",
    ):
        await controller.reconcile_startup()

    assert controller.entry_eligible() is False


def test_missing_or_malformed_status_is_quarantined(tmp_path: Path) -> None:
    expected = read_runtime_reconciliation_status(tmp_path / "missing.json")
    assert expected["state"] == "unavailable"
    assert expected["entry_eligible"] is False
    assert expected["quarantined"] is True

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"state":"ready"}', encoding="ascii")
    assert read_runtime_reconciliation_status(malformed) == expected


@pytest.mark.parametrize(
    "eligible_until",
    [
        (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
        "not-a-timestamp",
        "2026-07-30T12:00:00",
    ],
)
def test_expired_or_malformed_persisted_eligibility_is_quarantined(
    tmp_path: Path,
    eligible_until: str,
) -> None:
    status_path = tmp_path / "status.json"
    integration._write_status(
        status_path,
        {
            "schema_version": 1,
            "owner_binding": STATUS_OWNER_BINDING,
            "state": "ready",
            "trigger": "startup",
            "completed_at": (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),
            "eligible_until": eligible_until,
            "entry_eligible": True,
            "quarantined": False,
            "run_id": "not-exposed",
            "snapshot_id": "not-exposed",
        },
        owner_binding=STATUS_OWNER_BINDING,
    )

    status = read_runtime_reconciliation_status(status_path)

    assert status["state"] == "unavailable"
    assert status["entry_eligible"] is False
    assert status["quarantined"] is True


@pytest.mark.asyncio
async def test_builder_requires_explicit_signing_paths_before_provider_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _runtime(tmp_path / "ledger.db")
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.delenv("RT_RECONCILIATION_SIGNING_CAPABILITY_DIR", raising=False)
    monkeypatch.delenv("RT_RECONCILIATION_EVIDENCE_ROOT", raising=False)
    readiness = AsyncMock()
    provider = AsyncMock()
    monkeypatch.setattr(integration, "assert_runtime_bootstrap_ready", readiness)
    monkeypatch.setattr(integration, "build_diagnostic_provider", provider)

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="signing capability directory is not configured",
    ):
        await build_runtime_reconciliation_controller(context)

    readiness.assert_not_awaited()
    provider.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("protected_target", ["database", "journal", "capability"])
async def test_builder_rejects_protected_status_target_before_overwrite_or_provider_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_target: str,
) -> None:
    database = tmp_path / "ledger.db"
    database.write_bytes(b"ledger-state")
    runtime = _runtime(database)
    journal = Path(runtime.safety_journal_path)
    journal.write_bytes(b"journal-state")
    capability_directory = tmp_path / "capabilities"
    evidence_root = tmp_path / "evidence"
    capability_directory.mkdir(mode=0o700)
    evidence_root.mkdir(mode=0o700)
    capability_file = capability_directory / "broker_snapshot_private.pem"
    capability_file.write_bytes(b"signing-capability")
    targets = {
        "database": database,
        "journal": journal,
        "capability": capability_file,
    }
    target = targets[protected_target]
    before = target.read_bytes()
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.setenv(
        "RT_RECONCILIATION_SIGNING_CAPABILITY_DIR",
        str(capability_directory),
    )
    monkeypatch.setenv("RT_RECONCILIATION_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.setenv("RT_RECONCILIATION_STATUS_PATH", str(target))
    readiness = AsyncMock()
    provider = AsyncMock()
    monkeypatch.setattr(integration, "assert_runtime_bootstrap_ready", readiness)
    monkeypatch.setattr(integration, "build_diagnostic_provider", provider)

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="overlaps protected runtime state",
    ):
        await build_runtime_reconciliation_controller(context)

    assert target.read_bytes() == before
    readiness.assert_not_awaited()
    provider.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("database_state", ["missing", "partial"])
async def test_database_unavailable_or_partial_blocks_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    database_state: str,
) -> None:
    database = tmp_path / "ledger.db"
    if database_state == "partial":
        with sqlite3.connect(database) as connection:
            connection.execute("CREATE TABLE portfolios(id TEXT PRIMARY KEY)")
    runtime = _runtime(database)
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )

    with pytest.raises(RuntimeReconciliationIntegrationError):
        await assert_runtime_bootstrap_ready(context)

    if database_state == "missing":
        assert database.exists() is False
    else:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert tables == {"portfolios"}


def _readiness_database(
    database: Path,
    runtime: RuntimeContract,
    *,
    include_bootstrap: bool,
    include_reconciliation_receipt: bool = True,
    include_position: bool = False,
) -> None:
    with sqlite3.connect(database) as connection:
        now = datetime.now(timezone.utc).replace(microsecond=0)
        candidate = {
            "schema_version": 1,
            "bootstrap_id": BOOTSTRAP_ID,
            "execution_domain_scope": runtime.safety_execution_domain_scope,
            "account_scope": runtime.safety_account_scope,
            "portfolio_id": "default",
            "reconciliation_snapshot_id": "snapshot-1",
            "reconciliation_report_hash": "b" * 64,
            "broker_snapshot_hash": "c" * 64,
            "legacy_snapshot_hash": "d" * 64,
            "database_path": runtime.database_path,
            "database_identity": runtime.database_identity,
            "effective_at": now.isoformat(),
            "broker_position_count": 0,
            "broker_open_order_count": 0,
            "account": {
                "cash_text": "1000",
                "realized_pnl_text": "0",
                "daily_pnl_text": "0",
                "daily_pnl_baseline_text": "0",
                "daily_pnl_date": now.date().isoformat(),
            },
            "positions": (
                [
                    {
                        "con_id": 265598,
                        "symbol": "AAPL",
                        "quantity": 1,
                        "cost_basis_text": "100",
                        "mark_price_text": "110",
                        "mark_observed_at": now.isoformat(),
                        "mark_evidence_fingerprint": "e" * 64,
                    }
                ]
                if include_position
                else []
            ),
        }
        if include_position:
            candidate["account"]["daily_pnl_text"] = "10"  # type: ignore[index]
        connection.executescript("""
            CREATE TABLE portfolios(id TEXT PRIMARY KEY);
            CREATE TABLE paper_state_bootstraps(
                bootstrap_id TEXT PRIMARY KEY,
                execution_domain_scope TEXT,
                account_scope TEXT,
                portfolio_id TEXT,
                database_path TEXT,
                database_identity TEXT,
                database_device INTEGER,
                database_inode INTEGER,
                candidate_payload_json TEXT,
                broker_snapshot_hash TEXT,
                reconciliation_report_hash TEXT
            );
            CREATE TABLE paper_account_settlement_state(
                portfolio_id TEXT PRIMARY KEY,
                origin_bootstrap_id TEXT
            );
            CREATE TABLE positions(
                portfolio_id TEXT,
                symbol TEXT,
                quantity INTEGER
            );
            CREATE TABLE paper_position_settlement_state(
                portfolio_id TEXT,
                symbol TEXT,
                cost_basis_text TEXT,
                mark_price_text TEXT,
                origin_bootstrap_id TEXT
            );
            CREATE TABLE exact_bootstrap_evidence_consumptions(
                receipt_id TEXT,
                bootstrap_id TEXT,
                artifact_kind TEXT,
                artifact_sha256 TEXT,
                runtime_fingerprint TEXT,
                account_scope TEXT
            );
            INSERT INTO portfolios VALUES ('default');
            """)
        if include_bootstrap:
            metadata = database.stat()
            connection.execute(
                "INSERT INTO paper_state_bootstraps VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    BOOTSTRAP_ID,
                    runtime.safety_execution_domain_scope,
                    runtime.safety_account_scope,
                    "default",
                    runtime.database_path,
                    runtime.database_identity,
                    metadata.st_dev,
                    metadata.st_ino,
                    json.dumps(candidate, sort_keys=True, separators=(",", ":")),
                    "c" * 64,
                    "b" * 64,
                ),
            )
            connection.execute(
                "INSERT INTO paper_account_settlement_state VALUES (?,?)",
                ("default", BOOTSTRAP_ID),
            )
            if include_position:
                connection.execute(
                    "INSERT INTO positions VALUES (?,?,?)",
                    ("default", "AAPL", 1),
                )
                connection.execute(
                    "INSERT INTO paper_position_settlement_state VALUES (?,?,?,?,?)",
                    ("default", "AAPL", "100", "110", BOOTSTRAP_ID),
                )
            connection.execute(
                "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?,?,?,?,?)",
                (
                    "receipt-broker",
                    BOOTSTRAP_ID,
                    "broker_snapshot",
                    "c" * 64,
                    runtime.fingerprint,
                    runtime.safety_account_scope,
                ),
            )
            if include_reconciliation_receipt:
                connection.execute(
                    "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?,?,?,?,?)",
                    (
                        "receipt-reconciliation",
                        BOOTSTRAP_ID,
                        "reconciliation_report",
                        "b" * 64,
                        runtime.fingerprint,
                        runtime.safety_account_scope,
                    ),
                )
            if include_position:
                connection.execute(
                    "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?,?,?,?,?)",
                    (
                        "receipt-mark",
                        BOOTSTRAP_ID,
                        "protective_mark",
                        "e" * 64,
                        runtime.fingerprint,
                        runtime.safety_account_scope,
                    ),
                )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("include_bootstrap", "include_reconciliation_receipt", "should_pass"),
    [
        (False, False, False),
        (True, False, False),
        (True, True, True),
    ],
)
async def test_bootstrap_and_receipt_readiness_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_bootstrap: bool,
    include_reconciliation_receipt: bool,
    should_pass: bool,
) -> None:
    database = tmp_path / "ledger.db"
    runtime = _runtime(database)
    _readiness_database(
        database,
        runtime,
        include_bootstrap=include_bootstrap,
        include_reconciliation_receipt=include_reconciliation_receipt,
    )
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    schema_assertion = AsyncMock()
    monkeypatch.setattr(integration, "assert_exact_state_schema", schema_assertion)
    reconciliation_schema_assertion = AsyncMock()
    monkeypatch.setattr(
        integration,
        "assert_reconciliation_schema",
        reconciliation_schema_assertion,
    )

    if should_pass:
        await assert_runtime_bootstrap_ready(context)
    else:
        with pytest.raises(RuntimeReconciliationIntegrationError):
            await assert_runtime_bootstrap_ready(context)

    schema_assertion.assert_awaited_once()
    reconciliation_schema_assertion.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receipt_mutation",
    ["missing", "duplicate", "extra", "wrong-bootstrap"],
)
async def test_position_protective_receipts_require_exact_bootstrap_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_mutation: str,
) -> None:
    database = tmp_path / "ledger.db"
    runtime = _runtime(database)
    _readiness_database(
        database,
        runtime,
        include_bootstrap=True,
        include_position=True,
    )
    with sqlite3.connect(database) as connection:
        if receipt_mutation == "missing":
            connection.execute(
                "DELETE FROM exact_bootstrap_evidence_consumptions "
                "WHERE artifact_kind='protective_mark'"
            )
        elif receipt_mutation == "wrong-bootstrap":
            connection.execute(
                "UPDATE exact_bootstrap_evidence_consumptions SET bootstrap_id=? "
                "WHERE artifact_kind='protective_mark'",
                ("pboot-" + "2" * 32,),
            )
        else:
            connection.execute(
                "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?,?,?,?,?)",
                (
                    "receipt-mark-extra",
                    BOOTSTRAP_ID,
                    "protective_mark",
                    "e" * 64 if receipt_mutation == "duplicate" else "f" * 64,
                    runtime.fingerprint,
                    runtime.safety_account_scope,
                ),
            )
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.setattr(integration, "assert_exact_state_schema", AsyncMock())
    monkeypatch.setattr(integration, "assert_reconciliation_schema", AsyncMock())

    with pytest.raises(RuntimeReconciliationIntegrationError):
        await assert_runtime_bootstrap_ready(context)


@pytest.mark.asyncio
async def test_position_protective_receipt_exact_coverage_is_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "ledger.db"
    runtime = _runtime(database)
    _readiness_database(
        database,
        runtime,
        include_bootstrap=True,
        include_position=True,
    )
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.setattr(integration, "assert_exact_state_schema", AsyncMock())
    monkeypatch.setattr(integration, "assert_reconciliation_schema", AsyncMock())

    await assert_runtime_bootstrap_ready(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("evolution", ["closed-bootstrap-position", "later-entry"])
async def test_bootstrap_receipts_remain_ready_after_valid_position_evolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evolution: str,
) -> None:
    database = tmp_path / "ledger.db"
    runtime = _runtime(database)
    _readiness_database(
        database,
        runtime,
        include_bootstrap=True,
        include_position=evolution == "closed-bootstrap-position",
    )
    with sqlite3.connect(database) as connection:
        if evolution == "closed-bootstrap-position":
            connection.execute("DELETE FROM positions WHERE symbol='AAPL'")
            connection.execute("DELETE FROM paper_position_settlement_state WHERE symbol='AAPL'")
        else:
            connection.execute(
                "INSERT INTO positions VALUES (?,?,?)",
                ("default", "MSFT", 2),
            )
            connection.execute(
                "INSERT INTO paper_position_settlement_state VALUES (?,?,?,?,?)",
                ("default", "MSFT", "200", "210", BOOTSTRAP_ID),
            )
    context = SimpleNamespace(runtime_contract=runtime)
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.setattr(integration, "assert_exact_state_schema", AsyncMock())
    monkeypatch.setattr(integration, "assert_reconciliation_schema", AsyncMock())

    await assert_runtime_bootstrap_ready(context)


@pytest.mark.asyncio
@pytest.mark.parametrize("schema_state", ["absent", "partial"])
async def test_runtime_schema_assertion_is_byte_preserving_and_precedes_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_state: str,
) -> None:
    database = tmp_path / "ledger.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE portfolios(id TEXT PRIMARY KEY)")
        if schema_state == "partial":
            connection.execute("CREATE TABLE rt_schema_migrations(component TEXT, version INTEGER)")
    before = database.read_bytes()
    runtime = _runtime(database)
    context = SimpleNamespace(runtime_contract=runtime)
    capability_directory = tmp_path / "capabilities"
    evidence_root = tmp_path / "evidence"
    capability_directory.mkdir(mode=0o700)
    evidence_root.mkdir(mode=0o700)
    monkeypatch.setenv(
        "RT_RECONCILIATION_SIGNING_CAPABILITY_DIR",
        str(capability_directory),
    )
    monkeypatch.setenv("RT_RECONCILIATION_EVIDENCE_ROOT", str(evidence_root))
    monkeypatch.setattr(
        integration,
        "assert_validated_runtime_safety_context",
        lambda value: value,
    )
    monkeypatch.setattr(integration, "assert_exact_state_schema", AsyncMock())
    provider = AsyncMock()
    monkeypatch.setattr(integration, "build_diagnostic_provider", provider)

    with pytest.raises(RuntimeReconciliationIntegrationError):
        await build_runtime_reconciliation_controller(context)

    assert database.read_bytes() == before
    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    provider.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_kind", ["unrelated", "other-owner"])
async def test_status_publication_never_replaces_unrelated_existing_file(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    status_path = tmp_path / "status.json"
    if existing_kind == "unrelated":
        status_path.write_bytes(b"irreplaceable unrelated state")
    else:
        other_payload = {
            "schema_version": 1,
            "owner_binding": "b" * 64,
            "state": "quarantined",
            "trigger": None,
            "completed_at": None,
            "eligible_until": None,
            "entry_eligible": False,
            "quarantined": True,
            "run_id": None,
            "snapshot_id": None,
        }
        status_path.write_text(
            json.dumps(other_payload, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
    status_path.chmod(0o600)
    before = status_path.read_bytes()
    controller = RuntimeReconciliationController(
        _Service(_outcome(datetime.now(timezone.utc))),
        status_path=status_path,
        status_owner_binding=STATUS_OWNER_BINDING,
    )

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="status publication failed closed",
    ):
        await controller.reconcile_startup()

    assert status_path.read_bytes() == before
    assert controller.entry_eligible() is False


def test_status_republication_failure_preserves_complete_prior_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    first_payload = {
        "schema_version": 1,
        "owner_binding": STATUS_OWNER_BINDING,
        "state": "ready",
        "trigger": "startup",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "eligible_until": (datetime.now(timezone.utc) + timedelta(seconds=30)).isoformat(),
        "entry_eligible": True,
        "quarantined": False,
        "run_id": "run-first",
        "snapshot_id": "snapshot-first",
    }
    integration._write_status(
        status_path,
        first_payload,
        owner_binding=STATUS_OWNER_BINDING,
    )
    prior = status_path.read_bytes()

    def fail_exchange(*args, **kwargs):
        raise OSError("simulated crash boundary")

    monkeypatch.setattr(integration, "_exchange_status_entries", fail_exchange)
    second_payload = dict(first_payload, run_id="run-second", snapshot_id="snapshot-second")

    with pytest.raises(OSError, match="simulated crash boundary"):
        integration._write_status(
            status_path,
            second_payload,
            owner_binding=STATUS_OWNER_BINDING,
        )

    assert status_path.read_bytes() == prior
    assert not list(tmp_path.glob(".status.json.stage-*"))


def test_status_atomic_exchange_restores_raced_unrelated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    payload = {
        "schema_version": 1,
        "owner_binding": STATUS_OWNER_BINDING,
        "state": "quarantined",
        "trigger": None,
        "completed_at": None,
        "eligible_until": None,
        "entry_eligible": False,
        "quarantined": True,
        "run_id": None,
        "snapshot_id": None,
    }
    integration._write_status(
        status_path,
        payload,
        owner_binding=STATUS_OWNER_BINDING,
    )
    held_owned = tmp_path / "held-owned-status.json"
    real_exchange = integration._exchange_status_entries
    calls = 0

    def race_then_exchange(parent_descriptor: int, left: str, right: str) -> None:
        nonlocal calls
        if calls == 0:
            os.rename(
                right,
                held_owned.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            raced = os.open(
                right,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            os.write(raced, b"unrelated-raced-state")
            os.close(raced)
        calls += 1
        real_exchange(parent_descriptor, left, right)

    monkeypatch.setattr(integration, "_exchange_status_entries", race_then_exchange)

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="changed during publication",
    ):
        integration._write_status(
            status_path,
            dict(payload, state="ready"),
            owner_binding=STATUS_OWNER_BINDING,
        )

    assert calls == 2
    assert status_path.read_bytes() == b"unrelated-raced-state"
    assert held_owned.is_file()


def test_status_failed_race_rollback_preserves_displaced_unrelated_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    payload = {
        "schema_version": 1,
        "owner_binding": STATUS_OWNER_BINDING,
        "state": "quarantined",
        "trigger": None,
        "completed_at": None,
        "eligible_until": None,
        "entry_eligible": False,
        "quarantined": True,
        "run_id": None,
        "snapshot_id": None,
    }
    integration._write_status(
        status_path,
        payload,
        owner_binding=STATUS_OWNER_BINDING,
    )
    held_owned = tmp_path / "held-owned-status.json"
    real_exchange = integration._exchange_status_entries
    calls = 0

    def race_then_fail_rollback(parent_descriptor: int, left: str, right: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            os.rename(
                right,
                held_owned.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            raced = os.open(
                right,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_descriptor,
            )
            os.write(raced, b"unrelated-raced-state")
            os.close(raced)
            real_exchange(parent_descriptor, left, right)
            return
        raise OSError("simulated rollback failure")

    monkeypatch.setattr(
        integration,
        "_exchange_status_entries",
        race_then_fail_rollback,
    )

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="displaced inode was preserved",
    ):
        integration._write_status(
            status_path,
            dict(payload, state="ready"),
            owner_binding=STATUS_OWNER_BINDING,
        )

    displaced = list(tmp_path.glob(".status.json.stage-*"))
    assert calls == 2
    assert len(displaced) == 1
    assert displaced[0].read_bytes() == b"unrelated-raced-state"
    assert held_owned.is_file()


def test_first_status_publication_race_never_replaces_unrelated_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status_path = tmp_path / "status.json"
    payload = {
        "schema_version": 1,
        "owner_binding": STATUS_OWNER_BINDING,
        "state": "quarantined",
        "trigger": None,
        "completed_at": None,
        "eligible_until": None,
        "entry_eligible": False,
        "quarantined": True,
        "run_id": None,
        "snapshot_id": None,
    }
    real_publish = integration._publish_status_exclusive

    def race_then_publish(parent_descriptor: int, source: str, target: str) -> None:
        raced = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        os.write(raced, b"unrelated-raced-state")
        os.close(raced)
        real_publish(parent_descriptor, source, target)

    monkeypatch.setattr(integration, "_publish_status_exclusive", race_then_publish)

    with pytest.raises(
        RuntimeReconciliationIntegrationError,
        match="exclusive reconciliation status publication failed",
    ):
        integration._write_status(
            status_path,
            payload,
            owner_binding=STATUS_OWNER_BINDING,
        )

    assert status_path.read_bytes() == b"unrelated-raced-state"
    assert not list(tmp_path.glob(".status.json.stage-*"))

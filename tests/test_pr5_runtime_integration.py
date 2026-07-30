"""Synthetic production-integration tests for PR5 runtime reconciliation."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import robo_trader.reconciliation.runtime_integration as integration
from robo_trader.bootstrap_evidence_receivers import SealedBootstrapEvidenceArtifact
from robo_trader.config import RuntimeContract
from robo_trader.reconciliation.runtime_integration import (
    RuntimeReconciliationController,
    RuntimeReconciliationIntegrationError,
    ProductionRuntimeEvidenceSource,
    assert_runtime_bootstrap_ready,
    build_runtime_reconciliation_controller,
    read_runtime_reconciliation_status,
)

ACCOUNT_SCOPE = "acct_v1_0123456789abcdef0123456789abcdef" "fedcba9876543210fedcba9876543210"


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
    quote = SimpleNamespace(
        symbol="AAPL",
        con_id=265598,
        transport_generation="generation-1",
    )
    quote_source = SimpleNamespace(
        get_protective_quotes=AsyncMock(return_value=(quote,)),
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
        publish_complete_bundle=MagicMock(),
        published_artifact_path=MagicMock(
            side_effect=lambda artifact: tmp_path / artifact.artifact_path.name
        ),
        discard_unpublished_bundle=MagicMock(),
        close=MagicMock(),
    )
    delivery = SimpleNamespace(
        receiver_result=reconciliation_artifact,
        local_position_identities=(("default", "AAPL"),),
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
    quote_source.get_protective_quotes.assert_awaited_once_with(
        ("AAPL",),
        active_symbols=("AAPL",),
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
) -> None:
    with sqlite3.connect(database) as connection:
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
                database_inode INTEGER
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
                bootstrap_id TEXT,
                artifact_kind TEXT
            );
            INSERT INTO portfolios VALUES ('default');
            """)
        if include_bootstrap:
            metadata = database.stat()
            connection.execute(
                "INSERT INTO paper_state_bootstraps VALUES (?,?,?,?,?,?,?,?)",
                (
                    "bootstrap-1",
                    runtime.safety_execution_domain_scope,
                    runtime.safety_account_scope,
                    "default",
                    runtime.database_path,
                    runtime.database_identity,
                    metadata.st_dev,
                    metadata.st_ino,
                ),
            )
            connection.execute(
                "INSERT INTO paper_account_settlement_state VALUES (?,?)",
                ("default", "bootstrap-1"),
            )
            connection.execute(
                "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?)",
                ("bootstrap-1", "broker_snapshot"),
            )
            if include_reconciliation_receipt:
                connection.execute(
                    "INSERT INTO exact_bootstrap_evidence_consumptions VALUES (?,?)",
                    ("bootstrap-1", "reconciliation_report"),
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

    if should_pass:
        await assert_runtime_bootstrap_ready(context)
    else:
        with pytest.raises(RuntimeReconciliationIntegrationError):
            await assert_runtime_bootstrap_ready(context)

    schema_assertion.assert_awaited_once()

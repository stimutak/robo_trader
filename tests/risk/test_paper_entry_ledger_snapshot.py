"""Entry history reconstructed from a genuinely bootstrapped temporary ledger."""

import json
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.execution import LocalPaperExecutionEvidence
from robo_trader.paper_reduction_submitter import (
    LocalPaperOrderStatus,
    LocalPaperOutcomeProvenance,
    LocalPaperTerminalOutcome,
)
from robo_trader.protective_quote_evidence import ProtectiveQuoteSource, _produce_protective_quote
from robo_trader.risk.entry_reservations import reserve_entry_capacity
from robo_trader.risk.paper_ledger_snapshot import (
    PaperRiskLedgerSnapshotError,
    collect_paper_risk_ledger_snapshot,
)
from robo_trader.safety import SafetyJournal
from robo_trader.safety.entry_capacity import claim_entry_capacity
from robo_trader.stop_loss_monitor import StopLossMonitor
from tests import test_pr7_entry_risk_contract as risk_fixture
from tests.test_exact_state_bootstrap import _legacy_database  # noqa: F401
from tests.test_exact_state_bootstrap import (
    _backup_receipt,
    _bootstrap_evidence_keys,
    _candidate_bundle,
)
from tests.test_paper_entry_persistence import _snapshot
from tests.test_paper_entry_storage_replay import _commit


@pytest_asyncio.fixture
async def bootstrapped_entry(tmp_path, monkeypatch, request):
    # Isolate real authority state: these current-time decisions must not advance
    # the process registry past other suites' historical synthetic clocks.
    from robo_trader.risk import entry_contract

    for name, function in zip(
        (
            "_is_capability_seal",
            "_mint_capability",
            "_consume_capability",
            "_transfer_risk_decision",
        ),
        entry_contract._build_capability_authority(),
    ):
        monkeypatch.setattr(entry_contract, name, function)
    flat = getattr(request, "param", "flat") == "flat"
    path = tmp_path / "paper.db"
    if flat:
        database = AsyncTradingDatabase(path, pool_size=1)
        await database.initialize()
        await database.close()
    else:
        _legacy_database(path)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path, flat=flat)
    backup = _backup_receipt(path, tmp_path / "backup.db", candidate)
    database = AsyncTradingDatabase(path, pool_size=1)
    try:
        await database.apply_exact_state_bootstrap_offline_atomic(
            candidate,
            evidence=evidence,
            backup_receipt=backup,
            operator_reason="Test entry recovery from a sealed flat bootstrap.",
            runtime_contract=runtime,
        )
        await database.initialize()
        journal, case = await make_bootstrapped_entry_case(database, runtime, monkeypatch)
        yield (database, runtime, journal), case, candidate
    finally:
        await database.close()


async def make_bootstrapped_entry_case(database, runtime, monkeypatch):
    """Use the real decision/reservation/claim path for another synthetic entry."""
    now = datetime.now(timezone.utc)
    monkeypatch.setattr(risk_fixture, "NOW", now)
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(), risk_manager=None, portfolio_id="default"
    )
    quote = _produce_protective_quote(
        monitor,
        portfolio_id="default",
        symbol="AAPL",
        con_id=265598,
        price=Decimal("333"),
        source_timestamp=now - timedelta(seconds=1),
        receipt_monotonic=time.monotonic(),
        receipt_order=1,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        transport_generation=risk_fixture.ACTIVE_GENERATION,
        source_event_id="entry-ticker",
    )
    decision = risk_fixture._evaluate(
        evaluated_at=now,
        intent=risk_fixture._intent(
            portfolio_id="default",
            intent_id="intent-" + uuid.uuid4().hex,
            created_at=now - timedelta(seconds=5),
        ),
        evidence=risk_fixture._evidence(
            portfolio_id="default",
            quote=risk_fixture._quote(quote_id=quote.quote_id),
            correlation=risk_fixture._correlation(portfolio_id="default"),
            liquidity=risk_fixture._liquidity(portfolio_id="default"),
        ),
    )
    journal = SafetyJournal(runtime.safety_journal_path, clock=lambda: now)
    head = journal.replay()
    reserved = reserve_entry_capacity(
        journal,
        decision,
        sector="Technology",
        expected_head=(head.last_sequence, head.last_chain_hash),
    )
    claim = claim_entry_capacity(
        journal,
        reservation_sequence=reserved.sequence,
        reservation_chain_hash=reserved.chain_hash,
        expected_head=(reserved.sequence, reserved.chain_hash),
    )
    quantity = Decimal(json.loads(reserved.payload_json)["quantity"])
    fill = LocalPaperExecutionEvidence(
        "lpfill-" + uuid.uuid4().hex,
        quantity,
        quote.price,
        0,
        "USD",
        "LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1",
        now,
    )
    outcome = LocalPaperTerminalOutcome(
        json.loads(claim.payload_json)["order_ref"],
        LocalPaperOrderStatus.FILLED,
        quantity,
        quantity,
        Decimal("0"),
        quote.price,
        now,
        LocalPaperOutcomeProvenance.LOCAL_PAPER_EXECUTOR,
        True,
        "filled",
        fill,
    )
    case = dict(
        reservation=reserved,
        claim=claim,
        account=await database.get_paper_account_settlement_state(
            "default", "AAPL", runtime_contract=runtime
        ),
        pre_position_quantity=Decimal("0"),
        pre_aggregate_quantity=Decimal("0"),
        pre_symbol_gross_quantity=Decimal("0"),
        quote=quote,
        quote_producer=monitor,
        outcome=outcome,
    )
    return journal, case


@pytest.mark.asyncio
async def test_new_entry_inherits_account_bootstrap_origin(bootstrapped_entry):
    entry_db, case, candidate = bootstrapped_entry
    await _commit(entry_db, case)
    async with entry_db[0].get_connection() as conn:
        assert await (
            await conn.execute(
                "SELECT origin_bootstrap_id FROM paper_position_settlement_state WHERE symbol='AAPL'"
            )
        ).fetchone() == (candidate.bootstrap_id,)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["FILLED", "REJECTED", "CANCELLED", "EXPIRED"])
async def test_snapshot_replays_entry_from_flat_bootstrap_without_writes(
    bootstrapped_entry, status
):
    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    if status != "FILLED":
        case["outcome"] = replace(
            case["outcome"],
            status=LocalPaperOrderStatus(status),
            filled_quantity=Decimal("0"),
            remaining_quantity=case["outcome"].requested_quantity,
            exact_fill_price=None,
            fill_evidence=None,
        )
    await _commit(entry_db, case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("PRAGMA query_only=ON")
    try:
        snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
        assert snapshot.portfolio_cash == (
            ("default", Decimal("98002" if status == "FILLED" else "100000")),
        )
        assert len(snapshot.fills) == (1 if status == "FILLED" else 0)
        assert [(p.symbol, p.quantity) for p in snapshot.positions] == (
            [("AAPL", Decimal("6"))] if status == "FILLED" else []
        )
        async with database.get_connection() as conn:
            assert not conn.in_transaction
            assert await _snapshot(conn) == before
    finally:
        async with database.get_connection() as conn:
            await conn.execute("PRAGMA query_only=OFF")


@pytest.mark.asyncio
@pytest.mark.parametrize("bootstrapped_entry", ["held"], indirect=True)
async def test_snapshot_replays_reduction_then_entry_across_symbols(bootstrapped_entry):
    from tests.risk.test_paper_ledger_snapshot import _settle

    entry_db, case, candidate = bootstrapped_entry
    database, runtime, _ = entry_db
    await _settle((database, runtime, candidate))
    now = datetime.now(timezone.utc)
    case["outcome"] = replace(
        case["outcome"],
        observed_at=now,
        fill_evidence=replace(case["outcome"].fill_evidence, occurred_at=now),
    )
    case["account"] = await database.get_paper_account_settlement_state(
        "default", "AAPL", runtime_contract=runtime
    )
    await _commit(entry_db, case)
    # Reopen the actual database to exercise restart reconstruction.
    await database.close()
    await database.initialize()
    result = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert result.portfolio_cash == (
        ("default", candidate.account.cash + Decimal("660") - Decimal("1998")),
    )
    assert [(p.symbol, p.quantity) for p in result.positions] == [
        ("AAPL", Decimal("6")),
        ("NVDA", Decimal("7")),
        ("TSLA", Decimal("2")),
    ]
    assert [fill.symbol for fill in result.fills] == ["NVDA", "AAPL"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "table,sql",
    [
        ("trades", "UPDATE trades SET quantity=5 WHERE side='BUY'"),
        ("paper_fifo_settlement_links", "DELETE FROM paper_fifo_settlement_links"),
        (
            "paper_reduction_settlements",
            "UPDATE paper_reduction_settlements SET settlement_kind='REDUCTION'",
        ),
        (
            "paper_account_settlement_state",
            "UPDATE paper_account_settlement_state SET cash_text='100000'",
        ),
        ("positions", "UPDATE positions SET quantity=5"),
        (
            "paper_position_settlement_state",
            "UPDATE paper_position_settlement_state SET origin_bootstrap_id=NULL",
        ),
    ],
)
async def test_snapshot_rejects_damaged_entry_projection(bootstrapped_entry, table, sql):
    from tests.test_paper_entry_storage_replay import _corrupt

    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    await _commit(entry_db, case)
    async with database.get_connection() as conn:
        await _corrupt(conn, table, sql)
    with pytest.raises(PaperRiskLedgerSnapshotError):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_rejects_valid_entry_with_discontinuous_cash_history(bootstrapped_entry):
    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    # The staged record and mutable projections are mutually consistent, but
    # they cannot explain this extra cash from the sealed bootstrap.
    async with database.get_connection() as conn:
        await conn.execute("UPDATE account SET cash=100100")
        await conn.execute("UPDATE paper_account_settlement_state SET cash_text='100100'")
        await conn.commit()
    case["account"] = replace(case["account"], cash=Decimal("100100"))
    await _commit(entry_db, case)
    with pytest.raises(PaperRiskLedgerSnapshotError, match="cash history"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_requires_entry_journal_without_creating_missing_file(
    bootstrapped_entry, tmp_path
):
    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    await _commit(entry_db, case)
    missing = tmp_path / "missing-journal.db"
    with pytest.raises(PaperRiskLedgerSnapshotError):
        await collect_paper_risk_ledger_snapshot(
            database, replace(runtime, safety_journal_path=str(missing))
        )
    assert not missing.exists()

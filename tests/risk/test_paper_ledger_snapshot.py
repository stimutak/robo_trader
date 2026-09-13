"""Coherent, read-only entry ledger evidence from real synthetic bootstrap history."""

from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.risk.paper_ledger_snapshot import (
    PaperRiskLedgerSnapshotError,
    assert_owned_paper_risk_ledger_snapshot,
    collect_paper_risk_ledger_snapshot,
)
from tests.test_exact_state_bootstrap import (
    _backup_receipt,
    _bootstrap_evidence_keys,  # noqa: F401 - test-only trust fixture
    _candidate_bundle,
    _legacy_database,
)


@pytest_asyncio.fixture
async def ledger(tmp_path):
    path = tmp_path / "paper.db"
    _legacy_database(path)
    candidate, evidence, runtime = _candidate_bundle(path, tmp_path)
    backup = _backup_receipt(path, tmp_path / "backup.db", candidate)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.apply_exact_state_bootstrap_offline_atomic(
        candidate,
        evidence=evidence,
        backup_receipt=backup,
        operator_reason="Verify synthetic read-only risk snapshot bootstrap.",
        runtime_contract=runtime,
    )
    await database.initialize()
    try:
        yield database, runtime, candidate
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_snapshot_reconstructs_exact_cash_and_all_positions_without_writes(ledger):
    database, runtime, candidate = ledger
    async with database.get_connection() as connection:
        before = connection.total_changes
    result = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert_owned_paper_risk_ledger_snapshot(result)
    assert result.portfolio_cash == (("default", candidate.account.cash),)
    assert tuple((p.portfolio_id, p.symbol, p.quantity) for p in result.positions) == (
        ("default", "NVDA", Decimal("9")),
        ("default", "TSLA", Decimal("2")),
    )
    assert result.database_identity == runtime.database_identity
    assert not result.fills
    async with database.get_connection() as connection:
        assert connection.total_changes == before
        assert not connection.in_transaction


@pytest.mark.asyncio
async def test_snapshot_rejects_cash_rewrite_with_unchanged_bootstrap_lineage(ledger):
    database, runtime, candidate = ledger
    async with database.get_connection() as connection:
        await connection.execute(
            "UPDATE paper_account_settlement_state SET cash_text='999999' WHERE portfolio_id='default'"
        )
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="cash"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_rejects_quantity_rewrite_with_unchanged_bootstrap_lineage(ledger):
    database, runtime, _ = ledger
    async with database.get_connection() as connection:
        await connection.execute("UPDATE positions SET quantity=8 WHERE symbol='NVDA'")
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="position"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_rejects_new_unbootstrapped_portfolio(ledger):
    database, runtime, _ = ledger
    async with database.get_connection() as connection:
        await connection.execute("INSERT INTO portfolios(id,name) VALUES ('new','New')")
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="bootstrap"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_rejects_replaced_database_path(ledger, tmp_path):
    database, runtime, _ = ledger
    Path(runtime.database_path).rename(tmp_path / "original.db")
    Path(runtime.database_path).write_bytes(b"foreign database")
    with pytest.raises(PaperRiskLedgerSnapshotError, match="identity|replaced"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_snapshot_mutation_invalidates_provenance(ledger):
    database, runtime, _ = ledger
    result = await collect_paper_risk_ledger_snapshot(database, runtime)
    object.__setattr__(result, "portfolio_cash", (("default", Decimal("999999")),))
    with pytest.raises(PaperRiskLedgerSnapshotError):
        assert_owned_paper_risk_ledger_snapshot(result)


async def _settle(ledger, *, filled=True):
    import hashlib
    import json
    from dataclasses import replace
    from datetime import datetime, timezone

    from robo_trader.paper_terminal_settlement import PaperAccountSettlementState
    from robo_trader.safety.models import OrderSide, TerminalOrderStatus, decimal_to_fixed
    from tests.test_pr2b3_terminal_settlement_persistence import _request, _quote_payload

    database, runtime, candidate = ledger
    position = candidate.positions[0]
    quantity = Decimal(2) if filled else Decimal(0)
    price = Decimal("330") if filled else None
    state = PaperAccountSettlementState(
        portfolio_id="default",
        cash=candidate.account.cash,
        realized_pnl=candidate.account.realized_pnl,
        daily_pnl=candidate.account.daily_pnl,
        daily_pnl_baseline=candidate.account.daily_pnl_baseline,
        daily_pnl_date=candidate.account.daily_pnl_date.isoformat(),
        position_cost_basis=position.cost_basis,
        position_mark_price=position.mark_price,
        position_source_settlement_id=None,
    )
    post_cash, post_realized, post_daily = state.post_values(
        side=OrderSide.SELL,
        filled_quantity=quantity,
        fill_price=price,
        protective_mark_price=position.mark_price,
        pre_position_quantity=Decimal(position.quantity),
    )
    quote = json.loads(_quote_payload())
    quote.update(
        portfolio_id="default",
        symbol=position.symbol,
        con_id=position.con_id,
        price=decimal_to_fixed(position.mark_price),
    )
    payload = json.dumps(quote, sort_keys=True, separators=(",", ":"))
    request = replace(
        _request(outcome_at=datetime.now(timezone.utc)),
        portfolio_id="default",
        account_scope=runtime.safety_account_scope,
        symbol=position.symbol,
        con_id=position.con_id,
        protective_quote_payload=payload,
        protective_quote_fingerprint=hashlib.sha256(payload.encode()).hexdigest(),
        filled_quantity=quantity,
        remaining_quantity=Decimal(2) - quantity,
        expected_pre_position_quantity=Decimal(position.quantity),
        expected_post_position_quantity=Decimal(position.quantity) - quantity,
        expected_pre_aggregate_quantity=Decimal(position.quantity),
        expected_post_aggregate_quantity=Decimal(position.quantity) - quantity,
        expected_pre_cash=state.cash,
        expected_post_cash=post_cash,
        expected_pre_realized_pnl=state.realized_pnl,
        expected_post_realized_pnl=post_realized,
        expected_pre_daily_pnl=state.daily_pnl,
        expected_post_daily_pnl=post_daily,
        expected_daily_pnl_baseline=state.daily_pnl_baseline,
        expected_daily_pnl_date=state.daily_pnl_date,
        expected_position_cost_basis=position.cost_basis,
        expected_pre_position_mark_price=position.mark_price,
        terminal_status=TerminalOrderStatus.FILLED if filled else TerminalOrderStatus.REJECTED,
        fill_price=price,
        fill_execution_id="lpfill-" + "8" * 32 if filled else None,
        fill_commission_minor=0 if filled else None,
        fill_commission_currency="USD" if filled else None,
        fill_commission_source="LOCAL_PAPER_EXECUTOR_EXACT_COMMISSION_V1" if filled else None,
    )
    return await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("filled", [True, False])
async def test_snapshot_replays_real_terminal_history_and_fifo(ledger, filled):
    database, runtime, candidate = ledger
    receipt = await _settle(ledger, filled=filled)
    snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert snapshot.portfolio_cash == (("default", receipt.request.expected_post_cash),)
    assert snapshot.positions[0].quantity == receipt.request.expected_post_position_quantity
    assert len(snapshot.fills) == int(filled)
    if filled:
        assert snapshot.fills[0].execution_id == receipt.request.fill_execution_id


@pytest.mark.asyncio
async def test_snapshot_uses_one_transaction_during_concurrent_cash_update(ledger, monkeypatch):
    import aiosqlite
    from robo_trader.risk import paper_ledger_snapshot as module

    database, runtime, candidate = ledger
    original = module._read_validated_bootstrap_candidates

    async def update_after_bootstrap_read(connection, contract, binding):
        result = await original(connection, contract, binding)
        async with aiosqlite.connect(runtime.database_path) as writer:
            await writer.execute("UPDATE paper_account_settlement_state SET cash_text='999999'")
            await writer.commit()
        return result

    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", update_after_bootstrap_read)
    snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert snapshot.portfolio_cash == (("default", candidate.account.cash),)
    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", original)
    with pytest.raises(PaperRiskLedgerSnapshotError, match="cash"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["cash_string", "quantity_string", "position_dict", "timestamp_string"]
)
async def test_snapshot_seal_rejects_type_changing_mutations(ledger, mutation):
    from dataclasses import asdict

    database, runtime, _ = ledger
    result = await collect_paper_risk_ledger_snapshot(database, runtime)
    if mutation == "cash_string":
        object.__setattr__(
            result, "portfolio_cash", tuple((p, str(c)) for p, c in result.portfolio_cash)
        )
    elif mutation == "quantity_string":
        object.__setattr__(result.positions[0], "quantity", str(result.positions[0].quantity))
    elif mutation == "position_dict":
        object.__setattr__(result, "positions", tuple(asdict(p) for p in result.positions))
    else:
        object.__setattr__(result, "observed_at", str(result.observed_at))
    with pytest.raises(PaperRiskLedgerSnapshotError):
        assert_owned_paper_risk_ledger_snapshot(result)


@pytest.mark.asyncio
async def test_missing_compatibility_account_blocks_snapshot(ledger):
    database, runtime, _ = ledger
    async with database.get_connection() as connection:
        await connection.execute("DELETE FROM account")
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="account"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_missing_fifo_terminal_link_blocks_snapshot(ledger):
    database, runtime, _ = ledger
    await _settle(ledger)
    async with database.get_connection() as connection:
        trigger = await (
            await connection.execute(
                "SELECT sql FROM sqlite_master WHERE name='paper_fifo_settlement_links_no_delete'"
            )
        ).fetchone()
        await connection.execute("DROP TRIGGER paper_fifo_settlement_links_no_delete")
        await connection.execute("DELETE FROM paper_fifo_settlement_links")
        await connection.execute(trigger[0])
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="FIFO"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_database_replacement_during_snapshot_blocks_return(ledger, tmp_path, monkeypatch):
    from robo_trader.risk import paper_ledger_snapshot as module

    database, runtime, _ = ledger
    original = module._read_validated_bootstrap_candidates

    async def replace_after_read(connection, contract, binding):
        result = await original(connection, contract, binding)
        Path(runtime.database_path).rename(tmp_path / "original.db")
        Path(runtime.database_path).write_bytes(b"replacement")
        return result

    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", replace_after_read)
    with pytest.raises(PaperRiskLedgerSnapshotError, match="identity"):
        await collect_paper_risk_ledger_snapshot(database, runtime)


@pytest.mark.asyncio
async def test_cancelled_snapshot_releases_transaction_and_requires_pool_recovery(
    ledger, monkeypatch
):
    import asyncio
    from robo_trader.risk import paper_ledger_snapshot as module

    database, runtime, _ = ledger
    original = module._read_validated_bootstrap_candidates
    entered = asyncio.Event()

    async def pause_after_read(connection, contract, binding):
        result = await original(connection, contract, binding)
        entered.set()
        await asyncio.Event().wait()
        return result

    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", pause_after_read)
    task = asyncio.create_task(collect_paper_risk_ledger_snapshot(database, runtime))
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", original)
    from robo_trader.database_async import SafetyDatabasePoolError

    with pytest.raises(SafetyDatabasePoolError):
        async with database.get_connection():
            pytest.fail("cancelled checkout must preserve pool containment")
    await database.ensure_connection()
    async with database.get_connection() as connection:
        assert not connection.in_transaction
    assert_owned_paper_risk_ledger_snapshot(
        await collect_paper_risk_ledger_snapshot(database, runtime)
    )


@pytest.mark.asyncio
async def test_snapshot_rejects_foreign_account(ledger):
    from dataclasses import replace

    database, runtime, _ = ledger
    foreign = replace(runtime, safety_account_scope="acct_v1_" + "f" * 64)
    with pytest.raises(PaperRiskLedgerSnapshotError, match="bootstrap"):
        await collect_paper_risk_ledger_snapshot(database, foreign)


@pytest.mark.asyncio
async def test_snapshot_age_includes_time_spent_validating_history(ledger, monkeypatch):
    from datetime import datetime, timezone
    from robo_trader.risk import paper_ledger_snapshot as module

    database, runtime, _ = ledger
    original = module._read_validated_bootstrap_candidates
    validation_started = None

    async def record_validation_start(connection, contract, binding):
        nonlocal validation_started
        validation_started = datetime.now(timezone.utc)
        return await original(connection, contract, binding)

    monkeypatch.setattr(module, "_read_validated_bootstrap_candidates", record_validation_start)
    snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert snapshot.observed_at <= validation_started


@pytest.mark.asyncio
async def test_snapshot_cannot_hide_unbootstrapped_portfolio_with_temporary_shadow(ledger):
    database, runtime, _ = ledger
    async with database.get_connection() as connection:
        await connection.execute("INSERT INTO main.portfolios(id,name) VALUES ('new','New')")
        await connection.execute(
            "CREATE TEMP TABLE portfolios AS SELECT * FROM main.portfolios WHERE id='default'"
        )
        await connection.commit()
    with pytest.raises(PaperRiskLedgerSnapshotError, match="bootstrap"):
        await collect_paper_risk_ledger_snapshot(database, runtime)

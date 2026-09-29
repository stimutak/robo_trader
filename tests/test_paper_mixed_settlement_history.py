"""Mixed BUY/reduction storage on real temporary bootstrap/FIFO state."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from robo_trader.risk.paper_ledger_snapshot import collect_paper_risk_ledger_snapshot
from tests.risk.test_paper_entry_ledger_snapshot import bootstrapped_entry  # noqa: F401
from tests.test_exact_state_bootstrap import _bootstrap_evidence_keys  # noqa: F401
from tests.test_paper_entry_persistence import _snapshot
from tests.test_paper_entry_storage_replay import _commit, _corrupt
from tests.test_pr2b3_terminal_settlement_persistence import _request, _quote_payload


async def _close_request(database, runtime):
    state = await database.get_paper_account_settlement_state(
        "default", "AAPL", runtime_contract=runtime
    )
    quote = json.loads(_quote_payload())
    quote.update(portfolio_id="default", price="333")
    payload = json.dumps(quote, sort_keys=True, separators=(",", ":"))
    return replace(
        _request(outcome_at=datetime.now(timezone.utc)),
        portfolio_id="default",
        account_scope=runtime.safety_account_scope,
        requested_quantity=Decimal("6"),
        filled_quantity=Decimal("6"),
        expected_pre_position_quantity=Decimal("6"),
        expected_post_position_quantity=Decimal("0"),
        expected_pre_aggregate_quantity=Decimal("6"),
        expected_post_aggregate_quantity=Decimal("0"),
        expected_pre_cash=state.cash,
        expected_post_cash=state.cash + Decimal("2004"),
        expected_pre_realized_pnl=state.realized_pnl,
        expected_post_realized_pnl=state.realized_pnl + Decimal("6"),
        expected_pre_daily_pnl=state.daily_pnl,
        expected_post_daily_pnl=state.daily_pnl + Decimal("6"),
        expected_daily_pnl_baseline=state.daily_pnl_baseline,
        expected_daily_pnl_date=state.daily_pnl_date,
        expected_position_cost_basis=state.position_cost_basis,
        expected_pre_position_mark_price=state.position_mark_price,
        expected_pre_position_source_settlement_id=state.position_source_settlement_id,
        protective_quote_payload=payload,
        protective_quote_fingerprint=hashlib.sha256(payload.encode()).hexdigest(),
        fill_price=Decimal("334"),
    )


@pytest.mark.asyncio
async def test_entry_then_full_reduction_reopens_and_retries_without_reapplying(bootstrapped_entry):
    entry_db, case, _ = bootstrapped_entry
    database, runtime, journal = entry_db
    await _commit(entry_db, case)
    await database.close()
    await database.initialize()
    request = await _close_request(database, runtime)
    receipt = await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)
    await database.close()
    await database.initialize()
    snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert snapshot.portfolio_cash == (("default", Decimal("100006")),)
    assert snapshot.positions == ()
    assert len(snapshot.fills) == 2
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
    again = await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)
    assert again.settlement_id == receipt.settlement_id
    async with database.get_connection() as conn:
        assert await _snapshot(conn) == before
    receipts = [
        r
        async for r in database.iter_paper_terminal_receipts(
            runtime_contract=runtime, journal=journal
        )
    ]
    assert len(receipts) == 2
    # Storage completion cannot release entry capacity: independent daily
    # accounting and receipt-bound journal release are separate obligations.
    assert len(journal.replay().pending_entry_events) == 1


@pytest.mark.asyncio
async def test_damaged_entry_blocks_reduction_before_any_mutation(bootstrapped_entry):
    from robo_trader.paper_terminal_settlement import PaperTerminalSettlementError

    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    await _commit(entry_db, case)
    request = await _close_request(database, runtime)
    async with database.get_connection() as conn:
        await _corrupt(conn, "trades", "UPDATE trades SET quantity=5 WHERE side='BUY'")
        before = await _snapshot(conn)
    with pytest.raises(PaperTerminalSettlementError):
        await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)
    async with database.get_connection() as conn:
        assert await _snapshot(conn) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["AFTER_TRADE_INSERT", "AFTER_FIFO_LINK_INSERT"])
async def test_mixed_reduction_failure_rolls_back_all_tables(bootstrapped_entry, stage):
    entry_db, case, _ = bootstrapped_entry
    database, runtime, _ = entry_db
    await _commit(entry_db, case)
    request = await _close_request(database, runtime)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)

    def fail(at):
        if at == stage:
            raise RuntimeError("mixed reduction fault")

    database._paper_settlement_fault_hook = fail
    with pytest.raises(RuntimeError, match="mixed reduction fault"):
        await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)
    async with database.get_connection() as conn:
        assert await _snapshot(conn) == before
        assert not conn.in_transaction


@pytest.mark.asyncio
async def test_buy_release_sell_buy_reconstructs_after_each_reopen(
    bootstrapped_entry, tmp_path, monkeypatch
):
    from robo_trader.paper_entry_receipt import recover_committed_entry_receipt
    from robo_trader.risk.paper_fill_accounting import PaperFillAccounting
    from robo_trader.safety import SafetyJournal
    from robo_trader.paper_entry_release import release_entry_capacity
    from tests.risk.test_daily_filled_notional import MutableClock, _service
    from tests.risk.test_paper_entry_ledger_snapshot import make_bootstrapped_entry_case

    entry_db, case, candidate = bootstrapped_entry
    database, runtime, _ = entry_db
    record, _ = await _commit(entry_db, case)
    journal = SafetyJournal(runtime.safety_journal_path)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    ledger = _service(
        tmp_path / "daily.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="default",
        clock=MutableClock(datetime.now(timezone.utc)),
    )
    accounting = PaperFillAccounting(runtime, {"default": ledger})
    confirmation = await accounting.confirm_entry_settlement(receipt, database=database)
    state = journal.replay()
    release_entry_capacity(
        journal,
        receipt,
        confirmation,
        database=database,
        accounting=accounting,
        expected_head=(state.last_sequence, state.last_chain_hash),
    )
    await database.close()
    await database.initialize()
    assert (await collect_paper_risk_ledger_snapshot(database, runtime)).positions[
        0
    ].quantity == Decimal("6")
    close = await database.commit_paper_reduction_outcome(
        await _close_request(database, runtime), runtime_contract=runtime
    )
    await accounting.ingest(close)
    await database.close()
    await database.initialize()
    assert (await collect_paper_risk_ledger_snapshot(database, runtime)).positions == ()
    next_journal, next_case = await make_bootstrapped_entry_case(database, runtime, monkeypatch)
    await _commit((database, runtime, next_journal), next_case)
    await database.close()
    await database.initialize()
    snapshot = await collect_paper_risk_ledger_snapshot(database, runtime)
    assert snapshot.portfolio_cash == (("default", Decimal("98008")),)
    assert [(p.symbol, p.quantity) for p in snapshot.positions] == [("AAPL", Decimal("6"))]
    assert len(snapshot.fills) == 3
    result = await accounting.replay(database)
    assert (result.receipts_seen, result.fills_recorded) == (3, 1)
    assert await accounting.current_total("default") == Decimal("6000")
    assert journal.replay().pending_entry_events == (next_case["reservation"],)

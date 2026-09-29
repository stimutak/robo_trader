"""Transaction-engine tests on synthetic ledgers, not runtime entry authority."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.paper_entry_settlement import build_paper_entry_terminal_record
from robo_trader.paper_entry_persistence import stage_entry_settlement
from robo_trader.safety import SafetyJournal
from tests.test_paper_entry_terminal_record import case  # noqa: F401
from tests.test_pr2b3_terminal_settlement_persistence import _runtime_contract
from tests.fifo_runtime_test_support import install_synthetic_fifo_epoch


@pytest_asyncio.fixture
async def entry_db(tmp_path, case):
    from dataclasses import replace
    from decimal import Decimal

    runtime = replace(
        _runtime_contract(tmp_path),
        safety_account_scope=case["claim"].account_scope,
        safety_journal_path=str(tmp_path / "journal.db"),
    )
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as conn:
            await conn.execute(
                "INSERT INTO portfolios (id,name) VALUES ('portfolio-alpha','Entry')"
            )
            await conn.commit()
        await database.update_account(
            Decimal("100000"), Decimal("100000"), portfolio_id="portfolio-alpha"
        )
        async with database.get_connection() as conn:
            await conn.execute(
                "UPDATE paper_account_settlement_state SET daily_pnl_date='2026-07-28' WHERE portfolio_id='portfolio-alpha'"
            )
            await conn.commit()
        await install_synthetic_fifo_epoch(
            database,
            execution_domain_scope=case["claim"].execution_domain_scope,
            account_scope=case["claim"].account_scope,
            portfolio_id="portfolio-alpha",
            con_id=265598,
            symbol="AAPL",
        )
        yield database, runtime, SafetyJournal(tmp_path / "journal.db")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_full_entry_updates_all_projections_in_one_transaction(entry_db, case):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        result = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        await conn.commit()
        assert await (
            await conn.execute(
                "SELECT cash_text,source_settlement_id FROM paper_account_settlement_state WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchone() == ("98002", result.settlement_id)
        assert await (
            await conn.execute(
                "SELECT quantity,avg_cost,market_price FROM positions WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchone() == (6, 333, 333)
        assert await (
            await conn.execute(
                "SELECT side,quantity,price,notional,commission FROM trades WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchall() == [("BUY", 6, 333, 1998, 0)]
        assert await (
            await conn.execute(
                "SELECT settlement_kind,request_payload_json FROM paper_reduction_settlements"
            )
        ).fetchall() == [("ENTRY", record.payload_json)]
        assert await (
            await conn.execute("SELECT COUNT(*) FROM paper_fifo_settlement_links")
        ).fetchone() == (1,)
        assert await (await conn.execute("PRAGMA foreign_key_check")).fetchall() == []


async def _snapshot(conn):
    tables = await (
        await conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    ).fetchall()
    result = {}
    for (table,) in tables:
        quoted = '"' + table.replace('"', '""') + '"'
        rows = await (await conn.execute(f"SELECT * FROM {quoted}")).fetchall()
        result[table] = sorted(rows, key=repr)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        "ENTRY_AFTER_PRESTATE",
        "ENTRY_AFTER_FIFO",
        "ENTRY_AFTER_TRADE",
        "ENTRY_AFTER_TERMINAL",
        "ENTRY_AFTER_POSITION",
        "ENTRY_AFTER_FIFO_LINK",
        "ENTRY_AFTER_ACCOUNT",
    ],
)
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt])
async def test_stage_failure_rolls_back_even_if_outer_caller_commits(
    entry_db, case, stage, exception
):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)

    def fail(at):
        if at == stage:
            raise exception("injected storage failure")

    database._paper_settlement_fault_hook = fail
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(exception, match="injected"):
            await stage_entry_settlement(
                conn,
                record,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=datetime.now(timezone.utc),
            )
        await conn.commit()
        assert await _snapshot(conn) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["REJECTED", "CANCELLED", "EXPIRED"])
async def test_zero_fill_stages_only_terminal_account_lineage(entry_db, case, status):
    from dataclasses import replace
    from decimal import Decimal
    from robo_trader.paper_reduction_submitter import LocalPaperOrderStatus

    database, runtime, journal = entry_db
    case["outcome"] = replace(
        case["outcome"],
        status=LocalPaperOrderStatus(status),
        filled_quantity=Decimal("0"),
        remaining_quantity=case["outcome"].requested_quantity,
        exact_fill_price=None,
        fill_evidence=None,
    )
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        result = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        assert result.trade_id is None
        await conn.commit()
        after = await _snapshot(conn)
        for table in before:
            if table not in {"paper_reduction_settlements", "paper_account_settlement_state"}:
                assert after[table] == before[table], table
        assert await (
            await conn.execute(
                "SELECT cash_text,source_settlement_id FROM paper_account_settlement_state WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchone() == ("100000", result.settlement_id)


@pytest.mark.asyncio
async def test_duplicate_stage_cannot_apply_a_second_fill(entry_db, case):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        original = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        await conn.commit()
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        replay = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        assert replay == original
        await conn.commit()
        assert await _snapshot(conn) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cash", "opposing", "fractional", "metadata", "compatibility"])
async def test_changed_pre_state_rejects_without_mutation(entry_db, case, change):
    from robo_trader.safety.models import ValidationError

    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        if change == "cash":
            await conn.execute(
                "UPDATE paper_account_settlement_state SET cash_text='99999' WHERE portfolio_id='portfolio-alpha'"
            )
        elif change in {"opposing", "fractional"}:
            await conn.execute("INSERT INTO portfolios(id,name) VALUES('opposite','Opposite')")
            await conn.execute(
                "INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost) VALUES('portfolio-alpha','AAPL',?,333)",
                (6 if change == "opposing" else 0.5,),
            )
            await conn.execute(
                "INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost) VALUES('opposite','AAPL',?,333)",
                (-6 if change == "opposing" else -0.5,),
            )
        elif change == "metadata":
            await conn.execute(
                "INSERT INTO paper_position_settlement_state(portfolio_id,symbol,cost_basis_text,mark_price_text,updated_at) VALUES('portfolio-alpha','AAPL','1','1','now')"
            )
        else:
            await conn.execute("UPDATE account SET cash=99999 WHERE portfolio_id='portfolio-alpha'")
        await conn.commit()
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValidationError):
            await stage_entry_settlement(
                conn,
                record,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=datetime.now(timezone.utc),
            )
        await conn.commit()
        assert await _snapshot(conn) == before


@pytest.mark.asyncio
async def test_price_improvement_updates_cash_pnl_and_compatibility_equity(entry_db, case):
    from dataclasses import replace
    from decimal import Decimal, Inexact, Rounded, localcontext

    database, runtime, journal = entry_db
    case["outcome"] = replace(
        case["outcome"],
        exact_fill_price=Decimal("332"),
        fill_evidence=replace(case["outcome"].fill_evidence, exact_fill_price=Decimal("332")),
    )
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        with localcontext() as context:
            context.prec = 2
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            await stage_entry_settlement(
                conn,
                record,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=datetime.now(timezone.utc),
            )
        await conn.commit()
        assert await (
            await conn.execute(
                "SELECT cash_text,daily_pnl_text FROM paper_account_settlement_state WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchone() == ("98008", "6")
        assert await (
            await conn.execute(
                "SELECT cash,equity,daily_pnl,unrealized_pnl FROM account WHERE portfolio_id='portfolio-alpha'"
            )
        ).fetchone() == (98008, 100006, 6, 6)


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_cost", [100, 99])
async def test_closed_position_reentry_checks_retained_metadata(entry_db, case, legacy_cost):
    from dataclasses import replace
    from decimal import Decimal
    from robo_trader.safety.models import ValidationError

    database, runtime, journal = entry_db
    case["account"] = replace(
        case["account"], position_cost_basis=Decimal("100"), position_mark_price=Decimal("101")
    )
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute(
            "INSERT INTO positions(portfolio_id,symbol,quantity,avg_cost,market_price) VALUES('portfolio-alpha','AAPL',0,?,101)",
            (legacy_cost,),
        )
        await conn.execute(
            "INSERT INTO paper_position_settlement_state(portfolio_id,symbol,cost_basis_text,mark_price_text,updated_at) VALUES('portfolio-alpha','AAPL','100','101','now')"
        )
        await conn.commit()
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        if legacy_cost == 99:
            with pytest.raises(ValidationError, match="position compatibility"):
                await stage_entry_settlement(
                    conn,
                    record,
                    database=database,
                    runtime_contract=runtime,
                    journal=journal,
                    committed_at=datetime.now(timezone.utc),
                )
            await conn.commit()
            assert await _snapshot(conn) == before
        else:
            await stage_entry_settlement(
                conn,
                record,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=datetime.now(timezone.utc),
            )
            await conn.commit()
            assert await (
                await conn.execute(
                    "SELECT quantity,avg_cost FROM positions WHERE portfolio_id='portfolio-alpha'"
                )
            ).fetchone() == (6, 333)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "no_transaction",
        "account_scope",
        "writable_broker",
        "future_time",
        "file_identity",
        "missing_journal",
    ],
)
async def test_invalid_storage_context_cannot_mutate(entry_db, case, tmp_path, failure):
    from dataclasses import replace
    from datetime import timedelta
    from robo_trader.safety.models import ValidationError

    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    stamp = datetime.now(timezone.utc)
    if failure == "account_scope":
        runtime = replace(runtime, safety_account_scope="acct_v1_" + "0" * 64)
    elif failure == "writable_broker":
        runtime = replace(runtime, ibkr_readonly=False)
    elif failure == "future_time":
        stamp += timedelta(days=1)
    elif failure == "file_identity":
        device, inode = database._expected_database_file_identity
        database._expected_database_file_identity = (device, inode + 1)
    elif failure == "missing_journal":
        journal = SafetyJournal(tmp_path / "absent.db")
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        if failure != "no_transaction":
            await conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(ValidationError):
            await stage_entry_settlement(
                conn,
                record,
                database=database,
                runtime_contract=runtime,
                journal=journal,
                committed_at=stamp,
            )
        await conn.commit()
        assert await _snapshot(conn) == before
    assert not (tmp_path / "absent.db").exists()


@pytest.mark.asyncio
async def test_path_replacement_during_stage_rolls_back_before_return(entry_db, case, tmp_path):
    import sqlite3
    from robo_trader.safety.sqlite_identity import SQLiteIdentityError

    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    parked = tmp_path / "entry-original.db"

    def replace_path(stage):
        if stage == "ENTRY_AFTER_TERMINAL":
            database.db_path.rename(parked)
            sqlite3.connect(database.db_path).close()

    database._paper_settlement_fault_hook = replace_path
    async with database.get_connection() as conn:
        before = await _snapshot(conn)
        await conn.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(SQLiteIdentityError):
                await stage_entry_settlement(
                    conn,
                    record,
                    database=database,
                    runtime_contract=runtime,
                    journal=journal,
                    committed_at=datetime.now(timezone.utc),
                )
        finally:
            if parked.exists():
                database.db_path.unlink()
                parked.rename(database.db_path)
        await conn.commit()
        assert await _snapshot(conn) == before

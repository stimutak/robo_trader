"""Committed simulator fills replay into isolated durable risk accounting."""

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.risk.filled_notional import FilledNotionalUnavailable
from robo_trader.risk.paper_fill_accounting import PaperFillAccounting, PaperFillAccountingError
from tests.risk.test_daily_filled_notional import MutableClock, _service
from tests.test_pr2b3_terminal_settlement_persistence import _request, _runtime_contract, _seed


@pytest.mark.asyncio
async def test_restart_replays_committed_fill_exactly_once_into_paper_risk_scope(tmp_path):
    runtime = _runtime_contract(tmp_path)
    now = datetime.now(timezone.utc)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        await database.commit_paper_reduction_outcome(
            _request(outcome_at=now - timedelta(seconds=1)), runtime_contract=runtime
        )
        # Simulate loss of process state after durable settlement but before
        # notional ingestion. Recovery reads the actual terminal outbox.
        kwargs = dict(
            account_id="paper-simulator-v1:" + runtime.safety_account_scope,
            portfolio_id="portfolio-a",
            clock=MutableClock(now),
        )
        risk = _service(tmp_path / "risk.db", **kwargs)
        accounting = PaperFillAccounting(runtime, {"portfolio-a": risk})
        first = await accounting.replay(database)
        assert (first.receipts_seen, first.fills_recorded) == (1, 1)
        assert risk.current_gross_filled_notional() == Decimal("202.50")

        restored = _service(tmp_path / "risk.db", **kwargs)
        second = await PaperFillAccounting(runtime, {"portfolio-a": restored}).replay(database)
        assert (second.receipts_seen, second.fills_recorded) == (1, 0)
        assert restored.current_gross_filled_notional() == Decimal("202.50")
        async with database.get_connection() as connection:
            assert not connection.in_transaction
            assert await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone() == (
                1,
            )
    finally:
        await database.close()


def test_paper_accounting_rejects_broker_scoped_risk_ledger(tmp_path):
    runtime = _runtime_contract(tmp_path)
    risk = _service(tmp_path / "broker-risk.db", portfolio_id="portfolio-a")
    with pytest.raises(PaperFillAccountingError, match="scope"):
        PaperFillAccounting(runtime, {"portfolio-a": risk})


@pytest.mark.asyncio
async def test_cancellation_drains_fill_commit_and_replay_remains_idempotent(tmp_path, monkeypatch):
    runtime = _runtime_contract(tmp_path)
    now = datetime.now(timezone.utc)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    release = threading.Event()
    try:
        await _seed(database)
        receipt = await database.commit_paper_reduction_outcome(
            _request(outcome_at=now - timedelta(seconds=1)), runtime_contract=runtime
        )
        risk = _service(
            tmp_path / "risk.db",
            account_id="paper-simulator-v1:" + runtime.safety_account_scope,
            portfolio_id="portfolio-a",
            clock=MutableClock(now),
        )
        accounting = PaperFillAccounting(runtime, {"portfolio-a": risk})
        entered = threading.Event()
        record = accounting._record

        def paused(receipt):
            entered.set()
            assert release.wait(5)
            return record(receipt)

        monkeypatch.setattr(accounting, "_record", paused)
        task = asyncio.create_task(accounting.ingest(receipt))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert risk.current_gross_filled_notional() == Decimal("202.50")
        assert await accounting.ingest(receipt) is False
    finally:
        release.set()
        await database.close()


def test_accounting_rejects_split_account_ledgers(tmp_path):
    runtime = _runtime_contract(tmp_path)
    account_id = "paper-simulator-v1:" + runtime.safety_account_scope
    ledgers = {
        portfolio: _service(
            tmp_path / (portfolio + ".db"), account_id=account_id, portfolio_id=portfolio
        )
        for portfolio in ("portfolio-a", "portfolio-b")
    }
    with pytest.raises(PaperFillAccountingError, match="one account-wide"):
        PaperFillAccounting(runtime, ledgers)


def test_accounting_requires_complete_bootstrapped_portfolio_coverage(tmp_path):
    runtime = _runtime_contract(tmp_path)
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-a",
    )
    accounting = PaperFillAccounting(runtime, {"portfolio-a": risk})
    with pytest.raises(PaperFillAccountingError, match="coverage"):
        accounting.assert_runtime_coverage(runtime, ("portfolio-a", "retired"))


@pytest.mark.asyncio
async def test_current_total_rechecks_durable_ledger_authority(tmp_path):
    runtime = _runtime_contract(tmp_path)
    risk = _service(
        tmp_path / "risk.db",
        account_id="paper-simulator-v1:" + runtime.safety_account_scope,
        portfolio_id="portfolio-a",
    )
    accounting = PaperFillAccounting(runtime, {"portfolio-a": risk})
    assert await accounting.current_total("portfolio-a") == Decimal("0")
    risk._monotonic_verifier = lambda _: False
    with pytest.raises(FilledNotionalUnavailable, match="monotonic"):
        await accounting.current_total("portfolio-a")


@pytest.mark.asyncio
async def test_terminal_ingestion_cannot_redirect_to_replacement_risk_ledger(tmp_path):
    runtime = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        receipt = await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc)), runtime_contract=runtime
        )
        args = dict(
            account_id="paper-simulator-v1:" + runtime.safety_account_scope,
            portfolio_id="portfolio-a",
            clock=MutableClock(datetime.now(timezone.utc)),
        )
        original = _service(tmp_path / "risk.db", **args)
        accounting = PaperFillAccounting(runtime, {"portfolio-a": original})
        replacement = _service(tmp_path / "replacement.db", **args)
        accounting._ledgers["portfolio-a"] = replacement
        with pytest.raises(PaperFillAccountingError, match="binding"):
            await accounting.ingest(receipt)
        assert replacement.current_gross_filled_notional() == 0
    finally:
        await database.close()

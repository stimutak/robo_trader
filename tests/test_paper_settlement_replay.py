"""Read-only, account-bound outbox replay for daily-risk recovery."""

from contextlib import aclosing
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.paper_terminal_settlement import (
    PaperTerminalSettlementError,
    assert_producer_owned_paper_terminal_settlement_receipt,
)
from tests.test_pr2b3_terminal_settlement_persistence import _request, _runtime_contract, _seed


@pytest.mark.asyncio
async def test_replay_recovers_committed_receipt_after_restart_without_writes(tmp_path):
    runtime = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    await _seed(database)
    request = _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    committed = await database.commit_paper_reduction_outcome(request, runtime_contract=runtime)
    await database.close()

    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        async with database.get_connection() as connection:
            before = connection.total_changes
        receipts = [
            receipt
            async for receipt in database.iter_paper_terminal_receipts(runtime_contract=runtime)
        ]
        assert len(receipts) == 1
        assert receipts[0].fingerprint() == committed.fingerprint()
        assert_producer_owned_paper_terminal_settlement_receipt(receipts[0])
        async with database.get_connection() as connection:
            assert connection.total_changes == before
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_replay_rejects_foreign_account_instead_of_hiding_its_receipts(tmp_path):
    runtime = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=runtime,
        )
        foreign = replace(runtime, safety_account_scope="acct_v1_" + "f" * 64)
        with pytest.raises(PaperTerminalSettlementError, match="scope"):
            async for _ in database.iter_paper_terminal_receipts(runtime_contract=foreign):
                pytest.fail("foreign receipt escaped replay")
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_abandoned_replay_releases_snapshot_and_connection(tmp_path):
    runtime = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=runtime,
        )
        async with aclosing(
            database.iter_paper_terminal_receipts(runtime_contract=runtime)
        ) as rows:
            async for _ in rows:
                break
        async with database.get_connection() as connection:
            assert not connection.in_transaction
            assert await (await connection.execute("SELECT COUNT(*) FROM trades")).fetchone() == (
                1,
            )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_replay_rejects_corrupted_receipt_even_with_restored_immutability_trigger(tmp_path):
    runtime = _runtime_contract(tmp_path)
    database = AsyncTradingDatabase(Path(runtime.database_path), pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=runtime,
        )
        async with database.get_connection() as connection:
            trigger = await (
                await connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='paper_reduction_settlements_no_update'"
                )
            ).fetchone()
            await connection.execute("DROP TRIGGER paper_reduction_settlements_no_update")
            await connection.execute(
                "UPDATE paper_reduction_settlements SET receipt_fingerprint=?", ("f" * 64,)
            )
            await connection.execute(trigger[0])
            await connection.commit()
        with pytest.raises(PaperTerminalSettlementError, match="fingerprint"):
            async for _ in database.iter_paper_terminal_receipts(runtime_contract=runtime):
                pytest.fail("corrupted receipt escaped replay")
    finally:
        await database.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_during_replay", [False, True])
async def test_replay_rejects_replaced_database_path(tmp_path, replace_during_replay):
    runtime = _runtime_contract(tmp_path)
    path = Path(runtime.database_path)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    try:
        await _seed(database)
        await database.commit_paper_reduction_outcome(
            _request(outcome_at=datetime.now(timezone.utc) - timedelta(seconds=1)),
            runtime_contract=runtime,
        )

        def replace_path():
            path.rename(tmp_path / "original.db")
            path.write_bytes(b"replacement must not be accepted as the ledger")

        if not replace_during_replay:
            replace_path()
        with pytest.raises(PaperTerminalSettlementError, match="identity|replaced"):
            async with aclosing(
                database.iter_paper_terminal_receipts(runtime_contract=runtime)
            ) as rows:
                async for _ in rows:
                    if replace_during_replay:
                        replace_path()
                    else:
                        pytest.fail("replaced database path escaped replay")
    finally:
        await database.close()

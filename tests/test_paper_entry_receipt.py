"""Only an independent committed read may produce an entry receipt."""

import copy
from datetime import datetime, timezone

import pytest

from robo_trader.paper_entry_persistence import stage_entry_settlement
from robo_trader.paper_entry_receipt import (
    assert_owned_entry_receipt,
    recover_committed_entry_receipt,
)
from robo_trader.paper_entry_settlement import build_paper_entry_terminal_record
from robo_trader.safety.models import ValidationError
from tests.test_paper_entry_persistence import _snapshot, entry_db  # noqa: F401
from tests.test_paper_entry_terminal_record import case  # noqa: F401


@pytest.mark.asyncio
async def test_uncommitted_row_cannot_produce_receipt_then_commit_can(entry_db, case):
    database, runtime, journal = entry_db
    record = build_paper_entry_terminal_record(**case)
    async with database.get_connection() as conn:
        await conn.execute("BEGIN IMMEDIATE")
        staged = await stage_entry_settlement(
            conn,
            record,
            database=database,
            runtime_contract=runtime,
            journal=journal,
            committed_at=datetime.now(timezone.utc),
        )
        with pytest.raises(ValidationError, match="absent"):
            await recover_committed_entry_receipt(
                record, database=database, runtime_contract=runtime, journal=journal
            )
        await conn.commit()
        before = await _snapshot(conn)
        receipt = await recover_committed_entry_receipt(
            record, database=database, runtime_contract=runtime, journal=journal
        )
        assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)
        assert receipt.settlement_id == staged.settlement_id
        assert receipt.fingerprint() == staged.fingerprint
        assert await _snapshot(conn) == before
        with pytest.raises(ValidationError):
            assert_owned_entry_receipt(staged, database=database, runtime_contract=runtime)
        with pytest.raises(ValidationError):
            copy.copy(receipt)
        object.__setattr__(receipt, "trade_id", receipt.trade_id + 1)
        with pytest.raises(ValidationError):
            assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["replace", "deepcopy", "pickle", "record_mutation"])
async def test_receipt_copies_and_mutations_cannot_pass_ownership(entry_db, case, operation):
    import pickle
    from dataclasses import replace

    from tests.test_paper_entry_storage_replay import _commit

    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    if operation == "replace":
        forged = replace(receipt)
        with pytest.raises(ValidationError):
            assert_owned_entry_receipt(forged, database=database, runtime_contract=runtime)
    elif operation == "record_mutation":
        object.__setattr__(receipt.record, "payload_json", "{}")
        with pytest.raises(ValidationError):
            assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)
    else:
        with pytest.raises(ValidationError):
            (copy.deepcopy if operation == "deepcopy" else pickle.dumps)(receipt)


@pytest.mark.asyncio
async def test_recovery_connection_is_sqlite_read_only(entry_db, case, monkeypatch):
    import sqlite3

    import robo_trader.paper_entry_receipt as module
    from tests.test_paper_entry_storage_replay import _commit

    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    original = module.read_entry_settlement
    attempted = []

    async def prove_read_only(connection, *args, **kwargs):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            await connection.execute("UPDATE account SET cash=cash")
        attempted.append(True)
        return await original(connection, *args, **kwargs)

    monkeypatch.setattr(module, "read_entry_settlement", prove_read_only)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    assert attempted == [True]
    assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)


@pytest.mark.asyncio
async def test_receipt_cannot_transfer_to_other_scope_or_replaced_file(entry_db, case, tmp_path):
    from dataclasses import replace

    from tests.test_paper_entry_storage_replay import _commit

    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    with pytest.raises(ValidationError, match="runtime identity"):
        assert_owned_entry_receipt(
            receipt,
            database=database,
            runtime_contract=replace(runtime, safety_account_scope="acct_v1_" + "0" * 64),
        )
    parked = tmp_path / "receipt-source.db"
    database.db_path.rename(parked)
    try:
        database.db_path.touch()
        with pytest.raises(ValidationError, match="file identity"):
            assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)
    finally:
        database.db_path.unlink()
        parked.rename(database.db_path)


@pytest.mark.asyncio
async def test_record_shaped_object_cannot_replace_receipt_record(entry_db, case):
    from types import SimpleNamespace

    from tests.test_paper_entry_storage_replay import _commit

    database, runtime, journal = entry_db
    record, _ = await _commit(entry_db, case)
    receipt = await recover_committed_entry_receipt(
        record, database=database, runtime_contract=runtime, journal=journal
    )
    object.__setattr__(
        receipt,
        "record",
        SimpleNamespace(payload_json=record.payload_json, fingerprint=record.fingerprint),
    )
    with pytest.raises(ValidationError, match="record type"):
        assert_owned_entry_receipt(receipt, database=database, runtime_contract=runtime)

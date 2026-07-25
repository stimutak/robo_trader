"""Adversarial identity binding for authoritative allocation snapshots."""

import asyncio
import hashlib
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import robo_trader.database_async as database_module
from robo_trader.config import RuntimeContract
from robo_trader.database_async import (
    AsyncTradingDatabase,
    SafetyAllocationSnapshot,
    SafetyAllocationSnapshotError,
    SafetyPortfolioAllocation,
    assert_producer_owned_safety_allocation_snapshot,
)
from robo_trader.safety.sqlite_identity import SQLitePathBinding


def _create_ledger(path: Path, *, quantity: int = 7) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL NOT NULL DEFAULT 100000,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                market_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """)
        connection.execute("INSERT INTO portfolios (id, name) VALUES ('default', 'Default')")
        connection.execute(
            """
            INSERT INTO positions
                (portfolio_id, symbol, quantity, avg_cost, market_price)
            VALUES ('default', 'AAPL', ?, 100, 101)
            """,
            (quantity,),
        )


def _table_names(path: Path):
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()


def _runtime_contract(database_path: str) -> RuntimeContract:
    return RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host="127.0.0.1",
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=database_path,
        account_alias="***1234",
        account_type="paper",
        model_artifact_set="test-models",
        build_id="test-build",
        state_namespace="paper",
    )


async def _snapshot(database: AsyncTradingDatabase):
    return await database.get_safety_allocation_snapshot(
        "AAPL",
        runtime_contract=_runtime_contract(str(database.db_path)),
    )


@pytest.mark.asyncio
async def test_snapshot_binds_runtime_identity_path_and_opened_inode(tmp_path):
    path = tmp_path / "paper-ledger.db"
    _create_ledger(path)
    database = AsyncTradingDatabase(path)
    contract = _runtime_contract(str(path))
    metadata = os.lstat(path)

    snapshot = await database.get_safety_allocation_snapshot(
        "AAPL",
        runtime_contract=contract,
    )

    assert snapshot.database_path == str(path)
    assert snapshot.database_identity == contract.database_identity
    assert (snapshot.database_device, snapshot.database_inode) == (
        metadata.st_dev,
        metadata.st_ino,
    )
    assert snapshot.aggregate_allocated_quantity == 7


@pytest.mark.asyncio
async def test_snapshot_requires_one_strict_expected_database_identity(tmp_path):
    path = tmp_path / "paper-ledger.db"
    _create_ledger(path)
    database = AsyncTradingDatabase(path)

    with pytest.raises(SafetyAllocationSnapshotError, match="exact RuntimeContract"):
        await database.get_safety_allocation_snapshot("AAPL")
    with pytest.raises(SafetyAllocationSnapshotError, match="exact RuntimeContract"):
        await database.get_safety_allocation_snapshot(
            "AAPL",
            runtime_contract=SimpleNamespace(
                database_path=str(path),
                database_identity="paper:caller-label",
            ),
        )
    caller_labeled = replace(
        _runtime_contract(str(path)),
        state_namespace="caller-label",
    )
    with pytest.raises(SafetyAllocationSnapshotError, match="validated execution mode"):
        await database.get_safety_allocation_snapshot(
            "AAPL",
            runtime_contract=caller_labeled,
        )
    with pytest.raises(SafetyAllocationSnapshotError, match="expected database path"):
        await database.get_safety_allocation_snapshot(
            "AAPL",
            runtime_contract=_runtime_contract(str(tmp_path / "other.db")),
        )


@pytest.mark.asyncio
async def test_structurally_similar_identity_cannot_label_database(
    tmp_path,
    monkeypatch,
):
    configured = tmp_path / "configured"
    drifted = tmp_path / "drifted"
    configured.mkdir()
    drifted.mkdir()
    _create_ledger(configured / "ledger.db")
    _create_ledger(drifted / "ledger.db", quantity=999)

    monkeypatch.chdir(configured)
    database = AsyncTradingDatabase(Path("ledger.db"))
    with pytest.raises(SafetyAllocationSnapshotError, match="exact RuntimeContract"):
        await database.get_safety_allocation_snapshot(
            "AAPL",
            runtime_contract=SimpleNamespace(
                database_path=str(database.db_path),
                database_identity="paper:caller-label",
                state_namespace="paper",
            ),
        )
    assert database.db_path == configured / "ledger.db"


@pytest.mark.asyncio
async def test_runtime_contract_relative_path_rejects_changed_cwd(tmp_path, monkeypatch):
    configured = tmp_path / "configured"
    drifted = tmp_path / "drifted"
    configured.mkdir()
    drifted.mkdir()
    _create_ledger(configured / "ledger.db")
    _create_ledger(drifted / "ledger.db", quantity=999)

    monkeypatch.chdir(configured)
    database = AsyncTradingDatabase(Path("ledger.db"))
    contract = _runtime_contract("ledger.db")
    original_identity = contract.database_identity
    monkeypatch.chdir(drifted)

    with pytest.raises(SafetyAllocationSnapshotError, match="expected database path"):
        await database.get_safety_allocation_snapshot("AAPL", runtime_contract=contract)
    assert contract.database_identity != original_identity


def test_import_default_database_path_does_not_follow_later_cwd_change(
    tmp_path,
    monkeypatch,
):
    import_default = database_module.DB_PATH

    monkeypatch.chdir(tmp_path)
    database = AsyncTradingDatabase()

    assert database.db_path == import_default


@pytest.mark.asyncio
async def test_initialize_atomically_creates_missing_regular_ledger(tmp_path):
    path = tmp_path / "new-ledger.db"
    database = AsyncTradingDatabase(path, pool_size=1)

    await database.initialize()
    try:
        metadata = os.lstat(path)
        assert not path.is_symlink()
        assert metadata.st_mode & 0o077 == 0
        with sqlite3.connect(path) as connection:
            assert connection.execute("SELECT COUNT(*) FROM portfolios").fetchone() == (1,)
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert database._expected_database_file_identity == (
            metadata.st_dev,
            metadata.st_ino,
        )
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_initialize_rejects_leaf_planted_after_absent_capture(tmp_path, monkeypatch):
    path = tmp_path / "configured.db"
    target = tmp_path / "sentinel.db"
    _create_ledger(target, quantity=41)
    database = AsyncTradingDatabase(path, pool_size=1)
    before_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    before_tables = _table_names(target)
    path.symlink_to(target)
    entered_initializer = False
    original_initializer = database._init_database

    async def record_initializer(guardian):
        nonlocal entered_initializer
        entered_initializer = True
        await original_initializer(guardian)

    monkeypatch.setattr(database, "_init_database", record_initializer)

    with pytest.raises(SafetyAllocationSnapshotError, match="during initialization"):
        await database.initialize()

    assert entered_initializer is False
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before_sha
    assert _table_names(target) == before_tables
    with sqlite3.connect(target) as connection:
        assert connection.execute("SELECT quantity FROM positions").fetchall() == [(41,)]


@pytest.mark.asyncio
async def test_initialize_rejects_replacement_before_any_ledger_mutation(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "configured.db"
    replacement = tmp_path / "replacement.db"
    original_away = tmp_path / "original-away.db"
    _create_ledger(path, quantity=7)
    _create_ledger(replacement, quantity=700)
    database = AsyncTradingDatabase(path, pool_size=1)
    original_before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    replacement_before_sha = hashlib.sha256(replacement.read_bytes()).hexdigest()
    original_before_tables = _table_names(path)
    replacement_before_tables = _table_names(replacement)
    os.replace(path, original_away)
    os.replace(replacement, path)
    entered_initializer = False
    original_initializer = database._init_database

    async def record_initializer(guardian):
        nonlocal entered_initializer
        entered_initializer = True
        await original_initializer(guardian)

    monkeypatch.setattr(database, "_init_database", record_initializer)

    with pytest.raises(SafetyAllocationSnapshotError, match="replaced before initialization"):
        await database.initialize()

    assert entered_initializer is False
    assert hashlib.sha256(original_away.read_bytes()).hexdigest() == original_before_sha
    assert hashlib.sha256(path.read_bytes()).hexdigest() == replacement_before_sha
    assert _table_names(original_away) == original_before_tables
    assert _table_names(path) == replacement_before_tables
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT quantity FROM positions").fetchall() == [(700,)]
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name = 'account'").fetchall()
            == []
        )


@pytest.mark.asyncio
async def test_allocation_snapshot_requires_registered_producer_ownership(tmp_path):
    now = datetime.now(timezone.utc)
    fields = {
        "snapshot_id": "allocation-manual",
        "observed_at": now,
        "symbol": "AAPL",
        "allocations": (
            SafetyPortfolioAllocation(
                portfolio_id="default",
                symbol="AAPL",
                quantity=Decimal("7"),
                updated_at=now,
            ),
        ),
        "aggregate_allocated_quantity": Decimal("7"),
        "has_offsetting_allocations": False,
        "complete": True,
        "database_path": str(tmp_path / "configured.db"),
        "database_identity": "paper:caller-label",
        "database_device": 1,
        "database_inode": 1,
    }
    with pytest.raises(SafetyAllocationSnapshotError, match="trusted ledger producer"):
        SafetyAllocationSnapshot(**fields)

    path = tmp_path / "real.db"
    _create_ledger(path)
    genuine = await _snapshot(AsyncTradingDatabase(path))
    assert_producer_owned_safety_allocation_snapshot(genuine)
    copied = replace(genuine, database_inode=genuine.database_inode + 1)
    with pytest.raises(SafetyAllocationSnapshotError, match="producer-owned"):
        assert_producer_owned_safety_allocation_snapshot(copied)


@pytest.mark.asyncio
async def test_snapshot_rejects_symlinked_final_leaf_without_mutation(tmp_path):
    target = tmp_path / "target.db"
    linked = tmp_path / "configured.db"
    _create_ledger(target)
    linked.symlink_to(target)
    before = target.read_bytes()
    with pytest.raises(SafetyAllocationSnapshotError, match="non-symlink regular file"):
        AsyncTradingDatabase(linked)

    assert linked.is_symlink()
    assert target.read_bytes() == before


@pytest.mark.asyncio
async def test_snapshot_rejects_same_schema_replacement_before_open(tmp_path):
    path = tmp_path / "configured.db"
    replacement = tmp_path / "replacement.db"
    original_away = tmp_path / "original-away.db"
    _create_ledger(path, quantity=7)
    _create_ledger(replacement, quantity=700)
    original_bytes = path.read_bytes()
    database = AsyncTradingDatabase(path)
    os.replace(path, original_away)
    os.replace(replacement, path)

    try:
        with pytest.raises(SafetyAllocationSnapshotError, match="replaced before snapshot"):
            await _snapshot(database)
    finally:
        os.replace(path, replacement)
        os.replace(original_away, path)

    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_snapshot_rejects_same_schema_swap_open_restore(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "configured.db"
    replacement = tmp_path / "replacement.db"
    original_away = tmp_path / "original-away.db"
    replacement_away = tmp_path / "replacement-away.db"
    _create_ledger(path, quantity=7)
    _create_ledger(replacement, quantity=700)
    original_bytes = path.read_bytes()

    original_open = SQLitePathBinding.open_readonly.__func__
    original_identity = database_module.sqlite_connection_file_identity

    def guardian_then_swap(cls, protected_path):
        binding = original_open(cls, protected_path)
        os.replace(path, original_away)
        os.replace(replacement, path)
        return binding

    def inspect_then_restore(connection):
        identity = original_identity(connection)
        os.replace(path, replacement_away)
        os.replace(original_away, path)
        return identity

    monkeypatch.setattr(
        SQLitePathBinding,
        "open_readonly",
        classmethod(guardian_then_swap),
    )
    monkeypatch.setattr(
        database_module,
        "sqlite_connection_file_identity",
        inspect_then_restore,
    )

    with pytest.raises(SafetyAllocationSnapshotError, match="identity cannot be proven"):
        await _snapshot(AsyncTradingDatabase(path))

    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_snapshot_rejects_sqlite_descriptor_number_drift(tmp_path, monkeypatch):
    path = tmp_path / "configured.db"
    _create_ledger(path)
    original_identity = database_module.sqlite_connection_file_identity
    calls = 0

    def changed_descriptor(connection):
        nonlocal calls
        calls += 1
        identity = original_identity(connection)
        if calls == 2:
            return replace(identity, file_descriptor=identity.file_descriptor + 1)
        return identity

    monkeypatch.setattr(
        database_module,
        "sqlite_connection_file_identity",
        changed_descriptor,
    )

    with pytest.raises(SafetyAllocationSnapshotError, match="identity cannot be proven"):
        await _snapshot(AsyncTradingDatabase(path))


@pytest.mark.asyncio
async def test_snapshot_rejects_path_replacement_after_read(tmp_path, monkeypatch):
    path = tmp_path / "configured.db"
    replacement = tmp_path / "replacement.db"
    original_away = tmp_path / "original-away.db"
    _create_ledger(path, quantity=7)
    _create_ledger(replacement, quantity=700)
    original_bytes = path.read_bytes()
    original_identity = database_module.sqlite_connection_file_identity
    calls = 0

    def replace_after_select(connection):
        nonlocal calls
        calls += 1
        identity = original_identity(connection)
        if calls == 3:
            os.replace(path, original_away)
            os.replace(replacement, path)
        return identity

    monkeypatch.setattr(
        database_module,
        "sqlite_connection_file_identity",
        replace_after_select,
    )

    try:
        with pytest.raises(SafetyAllocationSnapshotError, match="identity cannot be proven"):
            await _snapshot(AsyncTradingDatabase(path))
    finally:
        if original_away.exists():
            os.replace(path, replacement)
            os.replace(original_away, path)

    assert path.read_bytes() == original_bytes


@pytest.mark.asyncio
async def test_snapshot_preserves_rows_and_main_file_in_rollback_mode(tmp_path):
    path = tmp_path / "configured.db"
    _create_ledger(path)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    before_metadata = os.stat(path)

    snapshot = await _snapshot(AsyncTradingDatabase(path))

    after_metadata = os.stat(path)
    assert snapshot.aggregate_allocated_quantity == 7
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert after_metadata.st_size == before_metadata.st_size
    assert after_metadata.st_mtime_ns == before_metadata.st_mtime_ns
    assert not Path(f"{path}-journal").exists()
    assert not Path(f"{path}-wal").exists()


@pytest.mark.asyncio
async def test_snapshot_preserves_rows_main_file_and_wal_contents(tmp_path):
    path = tmp_path / "configured.db"
    _create_ledger(path)
    writer = sqlite3.connect(path)
    try:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        writer.execute("UPDATE positions SET quantity = quantity WHERE symbol = 'AAPL'")
        writer.commit()
        database = AsyncTradingDatabase(path)
        before_rows = writer.execute(
            "SELECT portfolio_id, symbol, quantity FROM positions ORDER BY id"
        ).fetchall()
        before_main = path.read_bytes()
        wal_path = Path(f"{path}-wal")
        assert wal_path.exists()
        before_wal = wal_path.read_bytes()

        snapshot = await _snapshot(database)

        assert snapshot.aggregate_allocated_quantity == 7
        assert (
            writer.execute(
                "SELECT portfolio_id, symbol, quantity FROM positions ORDER BY id"
            ).fetchall()
            == before_rows
        )
        assert path.read_bytes() == before_main
        assert wal_path.read_bytes() == before_wal
        # SQLite may update lock-coordination bytes in the -shm file while a
        # WAL reader is active; literal filesystem immutability is not claimed.
    finally:
        writer.close()


@pytest.mark.asyncio
async def test_snapshot_closes_guardian_descriptor(tmp_path, monkeypatch):
    path = tmp_path / "configured.db"
    _create_ledger(path)
    original_open = SQLitePathBinding.open_readonly.__func__
    captured = {}

    def capture_guardian(cls, protected_path):
        binding = original_open(cls, protected_path)
        captured["descriptor"] = binding.guardian_file_descriptor
        return binding

    monkeypatch.setattr(
        SQLitePathBinding,
        "open_readonly",
        classmethod(capture_guardian),
    )

    await _snapshot(AsyncTradingDatabase(path))

    with pytest.raises(OSError):
        os.fstat(captured["descriptor"])


@pytest.mark.asyncio
async def test_database_initialization_cancellation_closes_partial_pool(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled.db")
    cleanup = AsyncMock(wraps=database._close_partial_pool)
    monkeypatch.setattr(database, "_close_partial_pool", cleanup)
    monkeypatch.setattr(
        database,
        "_init_database",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await database.initialize()

    cleanup.assert_awaited_once()
    assert database._pool == []


@pytest.mark.asyncio
async def test_pool_error_replaces_one_connection_with_identity_bound_connection(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "pool-recovery.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="forced pool failure"):
            async with database.get_connection():
                pass

        assert database._available.qsize() == 1
        assert len(database._pool) == 1
        replacement = database._pool[0]
        assert replacement is not original
        expected = database._expected_database_file_identity
        assert expected is not None
        identity = await database._sqlite_descriptor_identity(replacement)
        assert (identity.device, identity.inode) == expected

        async with asyncio.timeout(1):
            async with database.get_connection() as connection:
                assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
    finally:
        await database.close()

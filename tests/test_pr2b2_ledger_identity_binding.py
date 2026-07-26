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
    SafetyDatabasePoolError,
    SafetyPortfolioAllocation,
    assert_producer_owned_safety_allocation_snapshot,
)
from robo_trader.safety.sqlite_identity import SQLiteIdentityError, SQLitePathBinding


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


async def _bounded_await(awaitable, timeout: float):
    """Bound an await using the Python 3.10-compatible asyncio API."""
    return await asyncio.wait_for(awaitable, timeout=timeout)


async def _borrow_connection(database: AsyncTradingDatabase) -> None:
    async with database.get_connection():
        pass


async def _select_one(database: AsyncTradingDatabase):
    async with database.get_connection() as connection:
        return await (await connection.execute("SELECT 1")).fetchone()


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
async def test_initializer_connect_cancellation_is_never_swallowed(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-initializer-connect.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    monkeypatch.setattr(
        database_module.aiosqlite,
        "connect",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await database.initialize()

    assert database._initialized is False
    assert database._pool == []
    assert database._available.qsize() == 0
    assert database._quarantined_connections == []
    assert _table_names(path) == []


@pytest.mark.asyncio
async def test_snapshot_connect_cancellation_is_never_swallowed(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-snapshot-connect.db"
    _create_ledger(path)
    database = AsyncTradingDatabase(path)
    monkeypatch.setattr(
        database_module.aiosqlite,
        "connect",
        AsyncMock(side_effect=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await _snapshot(database)

    assert database._quarantined_connections == []


@pytest.mark.asyncio
async def test_repeated_cancel_during_initializer_close_cannot_leak_raw_handle(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-initializer-close.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE
    original_connect = database_module.aiosqlite.connect
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    captured = {}

    async def tracked_connect(*args, **kwargs):
        connection = await original_connect(*args, **kwargs)
        captured.setdefault("connection", connection)
        return connection

    async def blocked_close(connection):
        if connection is captured.get("connection"):
            close_entered.set()
            await release_close.wait()
        await exact_close(connection)

    monkeypatch.setattr(database_module.aiosqlite, "connect", tracked_connect)
    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", blocked_close)
    initializer = asyncio.create_task(database.initialize())
    await close_entered.wait()
    initializer.cancel()
    await asyncio.sleep(0)
    assert initializer.done() is False
    initializer.cancel()
    await asyncio.sleep(0)
    assert initializer.done() is False
    release_close.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await initializer
        connection = captured["connection"]
        assert database._pool == []
        assert database._quarantined_connections == []
        with pytest.raises(ValueError, match="no active connection"):
            await connection.execute("CREATE TABLE forbidden_initializer (id INTEGER)")
        with sqlite3.connect(path) as reader:
            assert (
                reader.execute(
                    "SELECT name FROM sqlite_master " "WHERE name = 'forbidden_initializer'"
                ).fetchall()
                == []
            )
    finally:
        release_close.set()
        await asyncio.gather(initializer, return_exceptions=True)
        monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)
        await database.close()


@pytest.mark.asyncio
async def test_repeated_cancel_during_snapshot_close_cannot_leak_read_handle(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-snapshot-close.db"
    _create_ledger(path)
    database = AsyncTradingDatabase(path)
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE
    original_connect = database_module.aiosqlite.connect
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    captured = {}

    async def tracked_connect(*args, **kwargs):
        connection = await original_connect(*args, **kwargs)
        captured["connection"] = connection
        return connection

    async def blocked_close(connection):
        close_entered.set()
        await release_close.wait()
        await exact_close(connection)

    monkeypatch.setattr(database_module.aiosqlite, "connect", tracked_connect)
    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", blocked_close)
    snapshot_task = asyncio.create_task(_snapshot(database))
    await close_entered.wait()
    snapshot_task.cancel()
    await asyncio.sleep(0)
    assert snapshot_task.done() is False
    snapshot_task.cancel()
    await asyncio.sleep(0)
    assert snapshot_task.done() is False
    release_close.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await snapshot_task
        connection = captured["connection"]
        assert database._quarantined_connections == []
        with pytest.raises(ValueError, match="no active connection"):
            await connection.execute("SELECT 1")
        with sqlite3.connect(path) as reader:
            assert (
                reader.execute(
                    "SELECT name FROM sqlite_master " "WHERE name = 'forbidden_snapshot'"
                ).fetchall()
                == []
            )
    finally:
        release_close.set()
        await asyncio.gather(snapshot_task, return_exceptions=True)
        monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)


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
        assert database._quarantined_connections == []
        expected = database._expected_database_file_identity
        assert expected is not None
        identity = await database._sqlite_descriptor_identity(replacement)
        assert (identity.device, identity.inode) == expected

        assert await _bounded_await(_select_one(database), 1) == (1,)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_failed_pool_replacement_poison_is_prompt_and_ensure_recovers(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "failed-pool-recovery.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    expected_identity = database._expected_database_file_identity
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    open_replacement = database._open_identity_bound_pool_connection
    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        AsyncMock(side_effect=SafetyAllocationSnapshotError("replacement rejected")),
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="forced pool failure") as caught:
            async with database.get_connection():
                pass
        assert isinstance(caught.value.__cause__, SafetyAllocationSnapshotError)
        assert database._initialized is True
        assert database._pool == []
        assert database._available.qsize() == 0

        with pytest.raises(SafetyDatabasePoolError, match="pool is poisoned"):
            await _bounded_await(_borrow_connection(database), 0.25)

        monkeypatch.setattr(
            database,
            "_open_identity_bound_pool_connection",
            open_replacement,
        )
        await database.ensure_connection()

        assert database._pool_recovery_failure is None
        assert database._expected_database_file_identity == expected_identity
        assert len(database._pool) == 1
        assert database._available.qsize() == 1
        assert await _bounded_await(_select_one(database), 1) == (1,)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_poisoned_pool_ensure_rejects_replaced_ledger_inode(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "replaced-after-pool-failure.db"
    replacement = tmp_path / "replacement.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    original = database._pool[0]
    expected_identity = database._expected_database_file_identity
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        AsyncMock(side_effect=SafetyAllocationSnapshotError("replacement rejected")),
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="forced pool failure"):
            async with database.get_connection():
                pass

        _create_ledger(replacement, quantity=700)
        os.replace(replacement, path)
        with pytest.raises(
            SafetyAllocationSnapshotError,
            match="replaced before initialization",
        ):
            await database.ensure_connection()

        assert database._initialized is False
        assert database._pool == []
        assert database._expected_database_file_identity == expected_identity
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_failed_pool_replacement_wakes_concurrent_waiter(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "waiting-pool-recovery.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    execute_entered = asyncio.Event()
    release_failure = asyncio.Event()

    async def fail_health_query(*_args, **_kwargs):
        execute_entered.set()
        await release_failure.wait()
        raise sqlite3.OperationalError("forced pool failure")

    monkeypatch.setattr(original, "execute", AsyncMock(side_effect=fail_health_query))
    open_replacement = database._open_identity_bound_pool_connection
    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        AsyncMock(side_effect=SafetyAllocationSnapshotError("replacement rejected")),
    )

    async def borrow():
        async with database.get_connection():
            pass

    failing_borrower = asyncio.create_task(borrow())
    await execute_entered.wait()
    queued_borrowers = [asyncio.create_task(borrow()) for _ in range(2)]
    await asyncio.sleep(0)
    release_failure.set()

    async def finish_failed_recovery():
        first_result = (await asyncio.gather(failing_borrower, return_exceptions=True))[0]
        monkeypatch.setattr(
            database,
            "_open_identity_bound_pool_connection",
            open_replacement,
        )
        await database.ensure_connection()
        queued_results = await asyncio.gather(
            *queued_borrowers,
            return_exceptions=True,
        )
        return first_result, queued_results

    try:
        first_result, queued_results = await _bounded_await(finish_failed_recovery(), 1)
        assert isinstance(first_result, sqlite3.OperationalError)
        assert str(first_result) == "forced pool failure"
        assert all(isinstance(result, SafetyDatabasePoolError) for result in queued_results)
        assert all("pool is poisoned" in str(result) for result in queued_results)
        assert database._available.qsize() == 1
    finally:
        for task in (failing_borrower, *queued_borrowers):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            failing_borrower,
            *queued_borrowers,
            return_exceptions=True,
        )
        await database.close()


@pytest.mark.asyncio
async def test_repeated_cancel_after_queue_dequeue_restores_pool_slot(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-pool-wait.db", pool_size=1)
    await database.initialize()
    borrowed_queue = database._available
    generation = database._pool_generation
    pooled_connection = database._pool[0]
    original_get = borrowed_queue.get
    dequeued = asyncio.Event()

    async def observed_get():
        connection = await original_get()
        dequeued.set()
        return connection

    monkeypatch.setattr(borrowed_queue, "get", observed_get)
    borrower = asyncio.create_task(database._wait_for_pool_connection(borrowed_queue, generation))
    await dequeued.wait()
    borrower.cancel()
    asyncio.get_running_loop().call_soon(borrower.cancel)

    try:
        with pytest.raises(asyncio.CancelledError):
            await borrower
        assert database._pool == [pooled_connection]
        assert database._available.qsize() == 1
        async with database.get_connection() as connection:
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
        assert database._available.qsize() == 1
    finally:
        await asyncio.gather(borrower, return_exceptions=True)
        await database.close()


@pytest.mark.asyncio
async def test_repeated_cancel_during_success_handoff_restores_pool_slot(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-pool-handoff.db", pool_size=1)
    await database.initialize()
    pooled_connection = database._pool[0]
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()
    original_cleanup = database._cleanup_pool_waiters
    cleanup_calls = 0

    async def blocked_first_cleanup(*args, **kwargs):
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            cleanup_entered.set()
            await release_cleanup.wait()
        await original_cleanup(*args, **kwargs)

    monkeypatch.setattr(database, "_cleanup_pool_waiters", blocked_first_cleanup)
    borrower = asyncio.create_task(
        database._wait_for_pool_connection(
            database._available,
            database._pool_generation,
        )
    )
    await cleanup_entered.wait()
    borrower.cancel()
    await asyncio.sleep(0)
    assert borrower.done() is False
    borrower.cancel()
    await asyncio.sleep(0)
    assert borrower.done() is False
    release_cleanup.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await borrower
        assert cleanup_calls == 2
        assert database._pool == [pooled_connection]
        assert database._available.qsize() == 1
        async with database.get_connection() as connection:
            assert connection is pooled_connection
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
        assert database._available.qsize() == 1
    finally:
        release_cleanup.set()
        await asyncio.gather(borrower, return_exceptions=True)
        await database.close()


@pytest.mark.asyncio
async def test_close_wakes_waiter_on_retired_pool_generation(tmp_path):
    database = AsyncTradingDatabase(tmp_path / "close-waiter.db", pool_size=1)
    await database.initialize()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()

    async def hold_connection():
        async with database.get_connection():
            holder_entered.set()
            await release_holder.wait()

    async def borrow_connection():
        async with database.get_connection():
            pass

    holder = asyncio.create_task(hold_connection())
    await holder_entered.wait()
    waiter = asyncio.create_task(borrow_connection())
    await asyncio.sleep(0)

    async def close_and_collect_waiter():
        await database.close()
        return (await asyncio.gather(waiter, return_exceptions=True))[0]

    try:
        waiter_result = await _bounded_await(close_and_collect_waiter(), 1)
        assert isinstance(waiter_result, SafetyDatabasePoolError)
        assert "database was closed" in str(waiter_result)
        assert database._closed is True
        assert database._initialized is False
        release_holder.set()
        holder_result = (await asyncio.gather(holder, return_exceptions=True))[0]
        assert holder_result is None
    finally:
        release_holder.set()
        await asyncio.gather(holder, waiter, return_exceptions=True)
        await database.close()


@pytest.mark.asyncio
async def test_concurrent_close_wins_over_ensure_reinitialization(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "close-ensure-race.db", pool_size=1)
    await database.initialize()
    health_entered = asyncio.Event()
    release_health = asyncio.Event()

    async def blocked_unhealthy_check():
        health_entered.set()
        await release_health.wait()
        return False

    monkeypatch.setattr(database, "health_check", blocked_unhealthy_check)
    ensure_task = asyncio.create_task(database.ensure_connection())
    await health_entered.wait()

    async def close_and_finish_ensure():
        await database.close()
        release_health.set()
        await ensure_task

    await _bounded_await(close_and_finish_ensure(), 1)

    assert database._closed is True
    assert database._initialized is False
    assert database._pool == []
    with pytest.raises(RuntimeError, match="Database is closed"):
        async with database.get_connection():
            pass


@pytest.mark.asyncio
async def test_poison_recovery_force_closes_active_old_generation_lease(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "active-lease-recovery.db", pool_size=2)
    await database.initialize()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    held = {}

    async def hold_connection():
        async with database.get_connection() as connection:
            held["connection"] = connection
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold_connection())
    await holder_entered.wait()
    leased_connection = held["connection"]
    idle_connection = next(
        connection for connection in database._pool if connection is not leased_connection
    )
    monkeypatch.setattr(
        idle_connection,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    open_replacement = database._open_identity_bound_pool_connection
    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        AsyncMock(side_effect=SafetyAllocationSnapshotError("replacement rejected")),
    )

    try:
        with pytest.raises(sqlite3.OperationalError, match="forced pool failure"):
            async with database.get_connection():
                pass
        assert leased_connection in database._leased_connections
        with pytest.raises(ValueError, match="no active connection"):
            await leased_connection.execute("SELECT 1")

        monkeypatch.setattr(
            database,
            "_open_identity_bound_pool_connection",
            open_replacement,
        )
        await database.ensure_connection()
        assert database._initialized is True
        assert database._closed is False
        assert len(database._pool) == 2
        assert database._available.qsize() == 2
        release_holder.set()
        holder_result = (await asyncio.gather(holder, return_exceptions=True))[0]
        assert holder_result is None
        assert leased_connection not in database._leased_connections
        assert len(database._pool) == 2
        assert database._available.qsize() == 2
    finally:
        release_holder.set()
        await asyncio.gather(holder, return_exceptions=True)
        await database.close()


@pytest.mark.asyncio
async def test_cancelled_poison_waits_until_active_lease_is_exactly_revoked(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-poison-active-lease.db"
    database = AsyncTradingDatabase(path, pool_size=2)
    await database.initialize()
    holder_entered = asyncio.Event()
    release_holder = asyncio.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    held = {}

    async def hold_connection():
        async with database.get_connection() as connection:
            held["connection"] = connection
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold_connection())
    await holder_entered.wait()
    leased_connection = held["connection"]
    failing_connection = next(
        connection for connection in database._pool if connection is not leased_connection
    )
    monkeypatch.setattr(
        failing_connection,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        AsyncMock(side_effect=SafetyAllocationSnapshotError("replacement rejected")),
    )
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE

    async def blocked_active_close(connection):
        if connection is leased_connection:
            close_entered.set()
            await release_close.wait()
        await exact_close(connection)

    monkeypatch.setattr(
        database_module,
        "_EXACT_AIOSQLITE_CLOSE",
        blocked_active_close,
    )

    async def trigger_poison():
        async with database.get_connection():
            pass

    failing_borrower = asyncio.create_task(trigger_poison())
    await close_entered.wait()
    failing_borrower.cancel()
    await asyncio.sleep(0)
    assert failing_borrower.done() is False
    release_close.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await failing_borrower
        with pytest.raises(ValueError, match="no active connection"):
            await leased_connection.execute("CREATE TABLE forbidden_during_poison (id INTEGER)")
        with sqlite3.connect(path) as reader:
            assert (
                reader.execute(
                    "SELECT name FROM sqlite_master WHERE name = 'forbidden_during_poison'"
                ).fetchall()
                == []
            )
        with pytest.raises(SafetyDatabasePoolError, match="pool is poisoned"):
            await _bounded_await(_borrow_connection(database), 0.25)
    finally:
        release_close.set()
        release_holder.set()
        await asyncio.gather(holder, failing_borrower, return_exceptions=True)
        monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)
        await database.close()


@pytest.mark.asyncio
async def test_close_rejects_post_return_mutation_from_old_lease(tmp_path):
    path = tmp_path / "post-close-mutation.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    holder_entered = asyncio.Event()
    attempt_mutation = asyncio.Event()

    async def mutate_after_close():
        async with database.get_connection() as connection:
            holder_entered.set()
            await attempt_mutation.wait()
            try:
                await connection.execute("CREATE TABLE forbidden_after_close (id INTEGER)")
                await connection.commit()
            except BaseException as error:
                return error
            return None

    holder = asyncio.create_task(mutate_after_close())
    await holder_entered.wait()
    await database.close()
    attempt_mutation.set()
    mutation_result = await holder

    assert isinstance(mutation_result, ValueError)
    assert "no active connection" in str(mutation_result)
    with sqlite3.connect(path) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name = 'forbidden_after_close'"
            ).fetchall()
            == []
        )


@pytest.mark.asyncio
async def test_cancelled_stale_holder_does_not_block_fresh_generation(tmp_path):
    database = AsyncTradingDatabase(tmp_path / "cancelled-stale-holder.db", pool_size=1)
    await database.initialize()
    holder_entered = asyncio.Event()
    keep_holding = asyncio.Event()

    async def hold_connection():
        async with database.get_connection():
            holder_entered.set()
            await keep_holding.wait()

    holder = asyncio.create_task(hold_connection())
    await holder_entered.wait()
    await database.close()
    await database.initialize()
    holder.cancel()

    with pytest.raises(asyncio.CancelledError):
        await holder
    assert database._quarantined_connections == []
    assert database._leased_connections == []
    async with database.get_connection() as connection:
        assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
    await database.close()


@pytest.mark.asyncio
async def test_close_cancellation_is_preserved_after_fail_closed_cleanup(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-close.db", pool_size=1)
    await database.initialize()
    holder_entered = asyncio.Event()
    rollback_entered = asyncio.Event()
    release_holder = asyncio.Event()
    captured = {}

    async def blocked_rollback():
        rollback_entered.set()
        await asyncio.Event().wait()

    async def hold_transaction():
        async with database.get_connection() as connection:
            await connection.execute("BEGIN")
            captured["connection"] = connection
            captured["rollback"] = connection.rollback
            monkeypatch.setattr(connection, "rollback", blocked_rollback)
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold_transaction())
    await holder_entered.wait()
    close_task = asyncio.create_task(database.close())
    await rollback_entered.wait()
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert database._closed is True
    assert database._initialized is False
    assert database._quarantined_connections == []

    release_holder.set()
    holder_result = (await asyncio.gather(holder, return_exceptions=True))[0]
    assert holder_result is None
    assert database._leased_connections == []


@pytest.mark.asyncio
async def test_repeated_cancel_during_leased_close_cannot_escape_revocation(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "cancelled-leased-close.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    holder_entered = asyncio.Event()
    close_entered = asyncio.Event()
    release_close = asyncio.Event()
    release_holder = asyncio.Event()
    captured = {}
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE

    async def blocked_close(connection):
        close_entered.set()
        await release_close.wait()
        await exact_close(connection)

    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", blocked_close)

    async def hold_connection():
        async with database.get_connection() as connection:
            captured["connection"] = connection
            holder_entered.set()
            await release_holder.wait()

    holder = asyncio.create_task(hold_connection())
    await holder_entered.wait()
    close_task = asyncio.create_task(database.close())
    await close_entered.wait()
    close_task.cancel()
    await asyncio.sleep(0)
    assert close_task.done() is False
    close_task.cancel()
    await asyncio.sleep(0)
    assert close_task.done() is False
    release_close.set()

    with pytest.raises(asyncio.CancelledError):
        await close_task
    connection = captured["connection"]
    assert database._closed is True
    assert database._quarantined_connections == []
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("CREATE TABLE forbidden_after_cancel (id INTEGER)")
    with sqlite3.connect(path) as reader:
        assert (
            reader.execute(
                "SELECT name FROM sqlite_master WHERE name = 'forbidden_after_cancel'"
            ).fetchall()
            == []
        )

    release_holder.set()
    holder_result = (await asyncio.gather(holder, return_exceptions=True))[0]
    assert holder_result is None
    assert database._leased_connections == []


@pytest.mark.asyncio
async def test_self_cancelling_instance_close_cannot_bypass_exact_revocation(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "self-cancelling-close.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    connection = database._pool[0]

    async def self_cancelling_close():
        raise asyncio.CancelledError()

    monkeypatch.setattr(connection, "close", self_cancelling_close)
    await database.close()

    assert database._closed is True
    assert database._quarantined_connections == []
    with pytest.raises(ValueError, match="no active connection"):
        await connection.execute("CREATE TABLE forbidden_self_cancel (id INTEGER)")
    with sqlite3.connect(path) as reader:
        assert (
            reader.execute(
                "SELECT name FROM sqlite_master WHERE name = 'forbidden_self_cancel'"
            ).fetchall()
            == []
        )


@pytest.mark.asyncio
async def test_cancelled_pool_replacement_still_latches_prompt_failure(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-pool-recovery.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    replacement_entered = asyncio.Event()

    async def blocked_replacement():
        replacement_entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(database, "_open_identity_bound_pool_connection", blocked_replacement)

    async def borrow():
        async with database.get_connection():
            pass

    try:
        borrower = asyncio.create_task(borrow())
        await replacement_entered.wait()
        borrower.cancel()
        with pytest.raises(asyncio.CancelledError):
            await borrower

        # Exercise the normal context-manager entry path too; the cancelled
        # replacement must have poisoned it before propagating cancellation.
        with pytest.raises(
            SafetyDatabasePoolError,
            match="replacement connection opening was cancelled",
        ):
            await _bounded_await(_borrow_connection(database), 0.25)

        assert database._pool == []
        assert database._available.qsize() == 0
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_task_cancel_during_bad_connection_close_poisons_before_propagating(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-bad-close.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE
    close_entered = asyncio.Event()
    release_close = asyncio.Event()

    async def blocked_close(connection):
        close_entered.set()
        await release_close.wait()
        await exact_close(connection)

    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", blocked_close)

    async def borrow():
        async with database.get_connection():
            pass

    borrower = asyncio.create_task(borrow())
    await close_entered.wait()
    borrower.cancel()
    await asyncio.sleep(0)
    assert borrower.done() is False
    release_close.set()

    try:
        with pytest.raises(asyncio.CancelledError):
            await borrower
        assert database._pool == []
        assert database._quarantined_connections == []
        with pytest.raises(
            SafetyDatabasePoolError,
            match="bad connection cleanup was cancelled",
        ):
            await _bounded_await(_borrow_connection(database), 0.25)
    finally:
        monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)
        await database.close()


@pytest.mark.asyncio
async def test_task_cancel_during_rollback_poisons_before_propagating(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "cancelled-rollback.db", pool_size=1)
    await database.initialize()
    rollback_entered = asyncio.Event()
    captured = {}

    async def blocked_rollback():
        rollback_entered.set()
        await asyncio.Event().wait()

    async def borrow_with_transaction():
        async with database.get_connection() as connection:
            await connection.execute("BEGIN")
            captured["connection"] = connection
            captured["rollback"] = connection.rollback
            monkeypatch.setattr(connection, "rollback", blocked_rollback)

    borrower = asyncio.create_task(borrow_with_transaction())
    await rollback_entered.wait()
    borrower.cancel()

    try:
        with pytest.raises(asyncio.CancelledError):
            await borrower
        connection = captured["connection"]
        assert database._pool == []
        assert database._quarantined_connections == []
        with pytest.raises(
            SafetyDatabasePoolError,
            match="pooled connection rollback was cancelled",
        ):
            await _bounded_await(_borrow_connection(database), 0.25)
    finally:
        connection = captured.get("connection")
        if connection is not None:
            monkeypatch.setattr(connection, "rollback", captured["rollback"])
        await database.close()


@pytest.mark.asyncio
async def test_unresolved_quarantine_blocks_reopen_until_cleanup_succeeds(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "quarantine-gate.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    original_close = original.close
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    monkeypatch.setattr(
        original,
        "close",
        AsyncMock(side_effect=OSError("forced cleanup failure")),
    )
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE

    async def failed_exact_close(_connection):
        raise OSError("forced exact cleanup failure")

    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", failed_exact_close)
    with pytest.raises(
        SafetyDatabasePoolError,
        match="poisoned pool cleanup could not close every connection",
    ):
        async with database.get_connection():
            pass
    assert original in database._quarantined_connections
    with pytest.raises(
        SafetyDatabasePoolError,
        match="quarantined connections remain unresolved",
    ):
        await database.initialize()

    with pytest.raises(
        SafetyDatabasePoolError,
        match="could not resolve every quarantined connection",
    ) as ensure_failure:
        await database.ensure_connection()
    assert isinstance(ensure_failure.value.__cause__, OSError)
    assert database._closed is True
    assert database._initialized is False
    assert original in database._quarantined_connections

    with pytest.raises(
        SafetyDatabasePoolError,
        match="quarantined connections remain unresolved",
    ):
        await database.initialize()
    assert database._pool == []

    monkeypatch.setattr(original, "close", original_close)
    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)
    await database.ensure_connection()

    try:
        assert database._quarantined_connections == []
        assert database._initialized is True
        assert database._closed is False
        assert len(database._pool) == 1
        async with database.get_connection() as connection:
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_partial_initialization_connection_is_quarantined_on_close_failure(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "partial-init-quarantine.db", pool_size=1)
    original_connect = database_module.aiosqlite.connect
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE
    captured = {}

    async def tracked_connect(path):
        connection = await original_connect(path)
        captured["connection"] = connection
        captured["close"] = connection.close
        monkeypatch.setattr(
            connection,
            "close",
            AsyncMock(side_effect=OSError("forced partial cleanup failure")),
        )
        return connection

    monkeypatch.setattr(database, "_init_database", AsyncMock())
    monkeypatch.setattr(database_module.aiosqlite, "connect", tracked_connect)

    async def failed_exact_close(_connection):
        raise OSError("forced exact partial cleanup failure")

    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", failed_exact_close)
    monkeypatch.setattr(
        database,
        "_sqlite_descriptor_identity",
        AsyncMock(side_effect=SQLiteIdentityError("forced identity failure")),
    )

    with pytest.raises(
        SafetyDatabasePoolError,
        match="initialization cleanup could not close every connection",
    ) as failure:
        await database.initialize()
    assert isinstance(failure.value.__cause__, OSError)
    connection = captured["connection"]
    assert connection in database._quarantined_connections
    assert database._initialized is False
    assert database._pool == []

    monkeypatch.setattr(connection, "close", captured["close"])
    monkeypatch.setattr(database_module, "_EXACT_AIOSQLITE_CLOSE", exact_close)
    await database.close()
    assert database._quarantined_connections == []


@pytest.mark.asyncio
async def test_stale_replacement_close_failure_poisons_reinitialized_pool(
    tmp_path,
    monkeypatch,
):
    database = AsyncTradingDatabase(tmp_path / "stale-replacement.db", pool_size=1)
    await database.initialize()
    original = database._pool[0]
    monkeypatch.setattr(
        original,
        "execute",
        AsyncMock(side_effect=sqlite3.OperationalError("forced pool failure")),
    )
    open_replacement = database._open_identity_bound_pool_connection
    replacement_opened = asyncio.Event()
    release_replacement = asyncio.Event()
    captured = {}

    async def blocked_replacement():
        replacement = await open_replacement()
        captured["replacement"] = replacement
        replacement_opened.set()
        await release_replacement.wait()
        return replacement

    monkeypatch.setattr(
        database,
        "_open_identity_bound_pool_connection",
        blocked_replacement,
    )

    async def borrow():
        async with database.get_connection():
            pass

    old_borrower = asyncio.create_task(borrow())
    await replacement_opened.wait()

    try:
        await database.close()
        await database.initialize()
        assert len(database._pool) == 1
        assert database._pool_recovery_failure is None

        release_replacement.set()
        old_result = (await asyncio.gather(old_borrower, return_exceptions=True))[0]
        assert isinstance(old_result, sqlite3.OperationalError)
        assert str(old_result) == "forced pool failure"
        assert database._quarantined_connections == []
        assert database._pool == []

        with pytest.raises(
            SafetyDatabasePoolError,
            match="stale replacement close failed",
        ):
            await _bounded_await(_borrow_connection(database), 0.25)
    finally:
        release_replacement.set()
        await asyncio.gather(old_borrower, return_exceptions=True)
        await database.close()

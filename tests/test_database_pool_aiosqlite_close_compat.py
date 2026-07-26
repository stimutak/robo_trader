"""Compatibility regressions for exact aiosqlite pool cleanup."""

import asyncio

import pytest

from robo_trader import database_async as database_module
from robo_trader.database_async import AsyncTradingDatabase, SafetyDatabasePoolError


@pytest.mark.asyncio
async def test_quarantine_accepts_only_proven_already_closed_aiosqlite_handle(
    tmp_path,
):
    """A handle closed by another generation is resolved under aiosqlite 0.19."""

    connection = await database_module.aiosqlite.connect(tmp_path / "already-closed.db")
    database = AsyncTradingDatabase(tmp_path / "pool-owner.db", pool_size=1)
    database._quarantine_pool_connection(connection)

    # The first pool-owned cleanup is the only authority that can establish
    # identity proof.  A stale generation can then attempt to quarantine the
    # same object again without invoking aiosqlite 0.19's non-idempotent close.
    await database._close_quarantined_connections("initial exact aiosqlite close failed")
    assert connection in database._proven_closed_connections
    database._quarantine_pool_connection(connection)
    assert database._quarantined_connections == []

    await database._close_quarantined_connections(
        "already-closed exact aiosqlite handle remained quarantined"
    )

    assert database._quarantined_connections == []


@pytest.mark.asyncio
async def test_snapshot_owned_cleanup_reuses_concurrent_close_identity_proof(
    tmp_path,
):
    """Snapshot finalization does not double-close after lifecycle cleanup."""

    connection = await database_module.aiosqlite.connect(tmp_path / "snapshot-close-race.db")
    database = AsyncTradingDatabase(tmp_path / "pool-owner.db", pool_size=1)
    database._quarantine_pool_connection(connection)
    lifecycle_close_finished = asyncio.Event()

    async def concurrent_lifecycle_close():
        await database._close_quarantined_connections("concurrent lifecycle close failed")
        lifecycle_close_finished.set()

    async def snapshot_finalizer():
        await lifecycle_close_finished.wait()
        await database._close_owned_connection(
            connection,
            "snapshot finalizer repeated an already-proven close",
        )

    await asyncio.gather(concurrent_lifecycle_close(), snapshot_finalizer())

    assert connection in database._proven_closed_connections
    assert database._quarantined_connections == []


@pytest.mark.asyncio
async def test_stale_poison_closes_only_its_owned_connection(tmp_path):
    """Stale recovery cannot consume a fresh snapshot quarantine entry."""

    database = AsyncTradingDatabase(tmp_path / "stale-poison.db", pool_size=1)
    await database.initialize()
    old_generation = database._pool_generation
    stale_orphan = await database_module.aiosqlite.connect(database.db_path)

    await database.close()
    await database.initialize()
    fresh_generation = database._pool_generation
    fresh_pool_connection = database._pool[0]
    fresh_snapshot = await database_module.aiosqlite.connect(database.db_path)
    database._quarantine_pool_connection(fresh_snapshot)

    try:
        await database._poison_connection_pool(
            SafetyDatabasePoolError("stale borrower cleanup"),
            old_generation,
            "stale borrower cleanup",
            (stale_orphan,),
        )

        assert stale_orphan in database._proven_closed_connections
        assert fresh_snapshot not in database._proven_closed_connections
        assert any(connection is fresh_snapshot for connection in database._quarantined_connections)
        assert await (await fresh_snapshot.execute("SELECT 1")).fetchone() == (1,)
        assert database._pool_generation is fresh_generation
        assert database._pool_recovery_failure is None
        assert database._pool == [fresh_pool_connection]
        async with database.get_connection() as connection:
            assert connection is fresh_pool_connection
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_stale_waiter_cleanup_does_not_poison_reinitialized_pool(
    tmp_path,
):
    """An orphan from a retired queue cannot latch failure on its successor."""

    database = AsyncTradingDatabase(tmp_path / "stale-waiter.db", pool_size=1)
    await database.initialize()
    old_queue = database._available
    old_generation = database._pool_generation
    connection_wait = asyncio.create_task(old_queue.get())
    old_connection = await connection_wait
    failure_wait = asyncio.create_task(old_generation.failure_event.wait())

    await database.close()
    await database.initialize()
    fresh_generation = database._pool_generation
    fresh_connection = database._pool[0]
    fresh_snapshot = await database_module.aiosqlite.connect(database.db_path)
    database._quarantine_pool_connection(fresh_snapshot)

    try:
        await database._cleanup_pool_waiters(
            connection_wait,
            failure_wait,
            old_queue,
            old_generation,
            False,
        )

        assert old_connection in database._proven_closed_connections
        assert database._pool_generation is fresh_generation
        assert database._pool_recovery_failure is None
        assert database._pool == [fresh_connection]
        assert database._available.qsize() == 1
        assert fresh_snapshot not in database._proven_closed_connections
        assert any(connection is fresh_snapshot for connection in database._quarantined_connections)
        assert await (await fresh_snapshot.execute("SELECT 1")).fetchone() == (1,)
        async with database.get_connection() as connection:
            assert connection is fresh_connection
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
        assert database._available.qsize() == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_stale_return_closes_only_retired_generation_connection(
    tmp_path,
    monkeypatch,
):
    """A lifecycle race during rollback cannot poison the successor pool."""

    database = AsyncTradingDatabase(tmp_path / "stale-return.db", pool_size=1)
    await database.initialize()
    old_generation = database._pool_generation
    old_connection = database._pool[0]
    original_rollback = old_connection.rollback
    rollback_started = asyncio.Event()
    release_borrower_rollback = asyncio.Event()
    rollback_calls = 0
    poison_calls = []
    original_poison = database._poison_connection_pool

    async def controlled_rollback():
        nonlocal rollback_calls
        rollback_calls += 1
        if rollback_calls == 1:
            rollback_started.set()
            await release_borrower_rollback.wait()
            return
        await original_rollback()

    async def tracked_poison(
        error,
        expected_generation,
        operation="replacement connection opening",
        owned_connections=(),
    ):
        poison_calls.append((expected_generation, operation, owned_connections))
        await original_poison(
            error,
            expected_generation,
            operation,
            owned_connections,
        )

    monkeypatch.setattr(old_connection, "rollback", controlled_rollback)
    monkeypatch.setattr(database, "_poison_connection_pool", tracked_poison)

    async def borrow_with_transaction():
        async with database.get_connection() as connection:
            await connection.execute("BEGIN")
            assert connection.in_transaction

    borrower = asyncio.create_task(borrow_with_transaction())
    await rollback_started.wait()

    try:
        await database.close()
        await database.initialize()
        fresh_generation = database._pool_generation
        fresh_connection = database._pool[0]

        release_borrower_rollback.set()
        await borrower

        stale_close_calls = [call for call in poison_calls if call[1] == "stale connection close"]
        assert stale_close_calls == [(old_generation, "stale connection close", (old_connection,))]
        assert old_connection in database._proven_closed_connections
        assert database._pool_generation is fresh_generation
        assert database._pool_recovery_failure is None
        assert database._pool == [fresh_connection]
        assert database._available.qsize() == 1
        async with database.get_connection() as connection:
            assert connection is fresh_connection
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
    finally:
        release_borrower_rollback.set()
        await asyncio.gather(borrower, return_exceptions=True)
        await database.close()


@pytest.mark.asyncio
async def test_stale_simultaneous_failure_and_checkout_do_not_poison_new_pool(
    tmp_path,
    monkeypatch,
):
    """Both completed old waiters still retain their captured generation."""

    database = AsyncTradingDatabase(tmp_path / "simultaneous-old-waiters.db", pool_size=1)
    await database.initialize()
    old_queue = database._available
    old_generation = database._pool_generation
    old_connection = database._pool[0]
    fresh = {}
    original_wait = database_module.asyncio.wait

    async def retire_between_wait_and_result(waiters, **_kwargs):
        connection_wait, failure_wait = tuple(waiters)
        await asyncio.sleep(0)
        assert connection_wait.done()
        assert connection_wait.result() is old_connection
        await database.close()
        await failure_wait
        await database.initialize()
        fresh["generation"] = database._pool_generation
        fresh["connection"] = database._pool[0]
        return {connection_wait, failure_wait}, set()

    monkeypatch.setattr(
        database_module.asyncio,
        "wait",
        retire_between_wait_and_result,
    )

    try:
        with pytest.raises(SafetyDatabasePoolError, match="database was closed"):
            await database._wait_for_pool_connection(old_queue, old_generation)
        monkeypatch.setattr(database_module.asyncio, "wait", original_wait)

        assert old_connection in database._proven_closed_connections
        assert database._pool_generation is fresh["generation"]
        assert database._pool_recovery_failure is None
        assert database._pool == [fresh["connection"]]
        assert database._available.qsize() == 1
        async with database.get_connection() as connection:
            assert connection is fresh["connection"]
            assert await (await connection.execute("SELECT 1")).fetchone() == (1,)
        assert database._available.qsize() == 1
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_raised_exact_close_has_no_identity_proof_and_stays_quarantined(
    tmp_path,
    monkeypatch,
):
    """A raised close stays unresolved even when private fields look closed."""

    connection = await database_module.aiosqlite.connect(tmp_path / "active-close-failure.db")
    database = AsyncTradingDatabase(tmp_path / "pool-owner.db", pool_size=1)
    database._quarantine_pool_connection(connection)
    exact_close = database_module._EXACT_AIOSQLITE_CLOSE

    async def close_then_raise(active_connection):
        # Reproduce the unsafe inference trap in aiosqlite 0.19: its private
        # state also says "closed" when exact close raises.  A raised cleanup
        # has no success proof and must stay quarantined regardless.
        await exact_close(active_connection)
        raise OSError("forced close failure after backend revocation")

    monkeypatch.setattr(
        database_module,
        "_EXACT_AIOSQLITE_CLOSE",
        close_then_raise,
    )
    try:
        with pytest.raises(
            SafetyDatabasePoolError,
            match="active exact close failure remained unresolved",
        ) as caught:
            await database._close_quarantined_connections(
                "active exact close failure remained unresolved"
            )

        assert isinstance(caught.value.__cause__, OSError)
        assert vars(connection)["_connection"] is None
        assert vars(connection)["_running"] is False
        assert connection not in database._proven_closed_connections
        assert database._quarantined_connections == [connection]
    finally:
        monkeypatch.setattr(
            database_module,
            "_EXACT_AIOSQLITE_CLOSE",
            exact_close,
        )

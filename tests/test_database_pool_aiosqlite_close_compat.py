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

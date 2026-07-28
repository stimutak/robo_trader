from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from robo_trader.database_async import AsyncTradingDatabase


@pytest.mark.asyncio
async def test_exact_state_migration_is_component_scoped_and_enables_foreign_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fresh.db"
    database = AsyncTradingDatabase(path, pool_size=2)
    await database.initialize()
    try:
        async with database.get_connection() as connection:
            foreign_keys = await connection.execute("PRAGMA foreign_keys")
            assert await foreign_keys.fetchone() == (1,)
            migration = await connection.execute("""
                SELECT component,version FROM rt_schema_migrations
                WHERE component='paper_exact_state'
                """)
            assert await migration.fetchone() == ("paper_exact_state", 1)
            legacy = await connection.execute("SELECT COUNT(*) FROM schema_migrations")
            assert await legacy.fetchone() == (0,)
    finally:
        await database.close()


@pytest.mark.asyncio
async def test_partial_exact_schema_is_completed_without_rewriting_rows(tmp_path: Path) -> None:
    path = tmp_path / "partial.db"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL NOT NULL DEFAULT 100000,
                symbols TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1,
                max_position_pct REAL, max_daily_loss_pct REAL,
                max_open_positions INTEGER, stop_loss_pct REAL,
                trailing_stop_pct REAL, use_trailing_stop INTEGER,
                enabled_strategies TEXT, min_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO portfolios(id,name) VALUES('default','Default');
            CREATE TABLE paper_account_settlement_state (
                portfolio_id TEXT PRIMARY KEY,
                cash_text TEXT NOT NULL,
                realized_pnl_text TEXT NOT NULL,
                daily_pnl_text TEXT NOT NULL,
                daily_pnl_baseline_text TEXT NOT NULL,
                daily_pnl_date TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_settlement_id TEXT
            );
            INSERT INTO paper_account_settlement_state VALUES
              ('default','1','0','0','0','2026-07-28','2026-07-28T12:00:00Z',NULL);
        """)
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(paper_account_settlement_state)")
        }
        assert "origin_bootstrap_id" in columns
        assert connection.execute(
            "SELECT cash_text,realized_pnl_text FROM paper_account_settlement_state"
        ).fetchall() == [("1", "0")]


@pytest.mark.asyncio
async def test_append_only_bootstrap_triggers_reject_update_and_delete(tmp_path: Path) -> None:
    path = tmp_path / "triggers.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()
    # The triggers are schema evidence here; row-level behavior is covered by
    # the applied bootstrap test without fabricating invalid foreign keys.
    async with aiosqlite.connect(path) as connection:
        cursor = await connection.execute("""
            SELECT name FROM sqlite_master
            WHERE type='trigger' AND name IN (
                'paper_state_bootstraps_no_update',
                'paper_state_bootstraps_no_delete',
                'administrator_actions_no_update',
                'administrator_actions_no_delete'
            ) ORDER BY name
            """)
        assert [row[0] for row in await cursor.fetchall()] == [
            "administrator_actions_no_delete",
            "administrator_actions_no_update",
            "paper_state_bootstraps_no_delete",
            "paper_state_bootstraps_no_update",
        ]


@pytest.mark.asyncio
async def test_recorded_migration_does_not_hide_malformed_trigger(tmp_path: Path) -> None:
    path = tmp_path / "malformed-trigger.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()

    with sqlite3.connect(path) as connection:
        connection.executescript("""
            DROP TRIGGER paper_state_bootstraps_no_update;
            CREATE TRIGGER paper_state_bootstraps_no_update
            BEFORE UPDATE ON paper_state_bootstraps
            BEGIN
                SELECT 1;
            END;
        """)
        assert connection.execute(
            "SELECT version FROM rt_schema_migrations WHERE component='paper_exact_state'"
        ).fetchone() == (1,)

    reopened = AsyncTradingDatabase(path, pool_size=1)
    with pytest.raises(RuntimeError, match="trigger .* malformed"):
        await reopened.initialize()
    await reopened.close()


@pytest.mark.asyncio
async def test_recorded_migration_does_not_hide_missing_foreign_key(tmp_path: Path) -> None:
    path = tmp_path / "missing-foreign-key.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executescript("""
            ALTER TABLE paper_account_settlement_state
            RENAME TO paper_account_settlement_state_original;
            CREATE TABLE paper_account_settlement_state (
                portfolio_id TEXT PRIMARY KEY,
                cash_text TEXT NOT NULL,
                realized_pnl_text TEXT NOT NULL,
                daily_pnl_text TEXT NOT NULL,
                daily_pnl_baseline_text TEXT NOT NULL,
                daily_pnl_date TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_settlement_id TEXT,
                origin_bootstrap_id TEXT
            );
            INSERT INTO paper_account_settlement_state
            SELECT * FROM paper_account_settlement_state_original;
            DROP TABLE paper_account_settlement_state_original;
        """)
        assert connection.execute(
            "SELECT version FROM rt_schema_migrations WHERE component='paper_exact_state'"
        ).fetchone() == (1,)

    reopened = AsyncTradingDatabase(path, pool_size=1)
    with pytest.raises(RuntimeError, match="foreign keys are malformed"):
        await reopened.initialize()
    await reopened.close()


@pytest.mark.asyncio
async def test_recorded_migration_does_not_hide_weakened_check_constraint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "weakened-check.db"
    database = AsyncTradingDatabase(path, pool_size=1)
    await database.initialize()
    await database.close()

    with sqlite3.connect(path) as connection:
        schema_version = connection.execute("PRAGMA schema_version").fetchone()[0]
        connection.execute("PRAGMA writable_schema = ON")
        connection.execute("""
            UPDATE sqlite_master
            SET sql = replace(
                sql,
                'CHECK (length(trim(reason)) >= 10)',
                'CHECK (1)'
            )
            WHERE type='table' AND name='administrator_actions'
            """)
        connection.execute("PRAGMA writable_schema = OFF")
        connection.execute(f"PRAGMA schema_version = {schema_version + 1}")
        connection.commit()
        assert connection.execute(
            "SELECT version FROM rt_schema_migrations WHERE component='paper_exact_state'"
        ).fetchone() == (1,)

    reopened = AsyncTradingDatabase(path, pool_size=1)
    with pytest.raises(RuntimeError, match="constraints are malformed"):
        await reopened.initialize()
    await reopened.close()

"""
Database migration for multiuser/multi-portfolio support.

Adds portfolio_id column to positions, trades, account, equity_history, and signals.
Creates new portfolios table. All existing data gets portfolio_id='default'.

This migration is safe and backward-compatible:
- Only runs once (checks migration version)
- Creates backup before modifying tables
- Existing single-portfolio usage continues to work unchanged
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiosqlite

from ..logger import get_logger

logger = get_logger(__name__)

MIGRATION_VERSION = 1  # Increment when adding new migrations


class MultiuserMigration:
    """Handles database schema migration for multi-portfolio support."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def get_migration_version(self, conn: aiosqlite.Connection) -> int:
        """Get current migration version from database."""
        try:
            cursor = await conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
        except Exception:
            # Table doesn't exist yet
            return 0

    async def needs_migration(self) -> bool:
        """Check if the database needs the multiuser migration."""
        async with aiosqlite.connect(self.db_path) as conn:
            current = await self.get_migration_version(conn)
            return current < MIGRATION_VERSION

    def _create_backup(self) -> Optional[Path]:
        """Create a backup of the database before migration.

        Returns:
            Path to backup file, or None if db doesn't exist
        """
        if not self.db_path.exists():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.db_path.with_suffix(f".backup_premultiuser_{timestamp}.db")
        shutil.copy2(self.db_path, backup_path)
        logger.info(f"Created database backup at: {backup_path}")
        return backup_path

    async def migrate(self, default_cash: float = 100_000.0) -> bool:
        """Run the multiuser migration.

        Args:
            default_cash: Starting cash for the 'default' portfolio record

        Returns:
            True if migration was applied, False if already up to date
        """
        if not self.db_path.exists():
            logger.info("No database found - migration will be applied on first init")
            return False

        async with aiosqlite.connect(self.db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=10000")
            await conn.execute("PRAGMA foreign_keys=OFF")

            current_version = await self.get_migration_version(conn)
            if current_version >= MIGRATION_VERSION:
                logger.info(f"Database already at migration version {current_version}, skipping")
                return False

            # Create backup BEFORE any changes
            backup_path = self._create_backup()
            logger.info(f"Starting multiuser migration (v{current_version} → v{MIGRATION_VERSION})")

            try:
                await self._apply_migration_v1(conn, default_cash)
                await conn.commit()
                logger.info("Multiuser migration completed successfully")
                return True

            except Exception as e:
                logger.error(f"Migration failed: {e}")
                await conn.rollback()
                if backup_path and backup_path.exists():
                    logger.info(f"Restoring from backup: {backup_path}")
                    shutil.copy2(backup_path, self.db_path)
                raise

    async def _apply_migration_v1(self, conn: aiosqlite.Connection, default_cash: float):
        """Apply migration version 1: Add multi-portfolio support."""

        # 1. Create schema_migrations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                description TEXT NOT NULL,
                applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Create portfolios table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                starting_cash REAL NOT NULL DEFAULT 100000,
                symbols TEXT NOT NULL DEFAULT '',
                active INTEGER NOT NULL DEFAULT 1,
                max_position_pct REAL,
                max_daily_loss_pct REAL,
                max_open_positions INTEGER,
                stop_loss_pct REAL,
                trailing_stop_pct REAL,
                use_trailing_stop INTEGER,
                enabled_strategies TEXT,
                min_confidence REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 3. Insert 'default' portfolio if not exists
        await conn.execute("""
            INSERT OR IGNORE INTO portfolios (id, name, starting_cash)
            VALUES ('default', 'Default Portfolio', ?)
        """, (default_cash,))

        # 4. Migrate positions table: add portfolio_id, change unique constraint
        await self._migrate_table_add_portfolio_id(
            conn,
            table_name="positions",
            create_sql="""
                CREATE TABLE positions_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_cost REAL NOT NULL,
                    market_price REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_id, symbol)
                )
            """,
            insert_sql="""
                INSERT INTO positions_new (id, portfolio_id, symbol, quantity, avg_cost, market_price, timestamp)
                SELECT id, 'default', symbol, quantity, avg_cost, market_price, timestamp
                FROM positions
            """,
            index_sql=[
                "CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions (portfolio_id)",
            ],
        )

        # 5. Migrate trades table: add portfolio_id
        await self._migrate_table_add_portfolio_id(
            conn,
            table_name="trades",
            create_sql="""
                CREATE TABLE trades_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    notional REAL DEFAULT 0,
                    slippage REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    pnl REAL DEFAULT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            insert_sql="""
                INSERT INTO trades_new (id, portfolio_id, symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp)
                SELECT id, 'default', symbol, side, quantity, price, notional, slippage, commission, pnl, timestamp
                FROM trades
            """,
            index_sql=[
                "CREATE INDEX IF NOT EXISTS idx_trades_portfolio ON trades (portfolio_id)",
                "CREATE INDEX IF NOT EXISTS idx_trades_portfolio_symbol ON trades (portfolio_id, symbol, timestamp DESC)",
            ],
        )

        # 6. Migrate account table: replace id=1 with portfolio_id
        await self._migrate_table_add_portfolio_id(
            conn,
            table_name="account",
            create_sql="""
                CREATE TABLE account_new (
                    portfolio_id TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    equity REAL NOT NULL,
                    daily_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            insert_sql="""
                INSERT INTO account_new (portfolio_id, cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp)
                SELECT 'default', cash, equity, daily_pnl, realized_pnl, unrealized_pnl, timestamp
                FROM account
                WHERE id = 1
            """,
            index_sql=[],
        )

        # 7. Migrate equity_history table: add portfolio_id, change unique constraint
        await self._migrate_table_add_portfolio_id(
            conn,
            table_name="equity_history",
            create_sql="""
                CREATE TABLE equity_history_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    date TEXT NOT NULL,
                    equity REAL NOT NULL,
                    cash REAL DEFAULT 0,
                    positions_value REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(portfolio_id, date)
                )
            """,
            insert_sql="""
                INSERT INTO equity_history_new (id, portfolio_id, date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp)
                SELECT id, 'default', date, equity, cash, positions_value, realized_pnl, unrealized_pnl, timestamp
                FROM equity_history
            """,
            index_sql=[
                "CREATE INDEX IF NOT EXISTS idx_equity_history_portfolio ON equity_history (portfolio_id, date)",
            ],
        )

        # 8. Migrate signals table: add portfolio_id
        await self._migrate_table_add_portfolio_id(
            conn,
            table_name="signals",
            create_sql="""
                CREATE TABLE signals_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL DEFAULT 'default',
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """,
            insert_sql="""
                INSERT INTO signals_new (id, portfolio_id, symbol, strategy, signal_type, strength, metadata, timestamp)
                SELECT id, 'default', symbol, strategy, signal_type, strength, metadata, timestamp
                FROM signals
            """,
            index_sql=[
                "CREATE INDEX IF NOT EXISTS idx_signals_portfolio ON signals (portfolio_id)",
            ],
        )

        # 9. Record migration
        await conn.execute("""
            INSERT INTO schema_migrations (version, description)
            VALUES (?, ?)
        """, (MIGRATION_VERSION, "Add multi-portfolio support: portfolio_id on positions, trades, account, equity_history, signals"))

        logger.info("Migration v1 applied: multi-portfolio schema ready")

    async def _migrate_table_add_portfolio_id(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        create_sql: str,
        insert_sql: str,
        index_sql: List[str],
    ):
        """Migrate a table by recreating it with portfolio_id column.

        Uses the rename-recreate pattern since SQLite doesn't support
        ALTER TABLE ADD CONSTRAINT.
        """
        # Check if table exists
        cursor = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        exists = await cursor.fetchone()

        if not exists:
            # Table doesn't exist yet - create directly with portfolio_id
            await conn.execute(create_sql.replace(f"{table_name}_new", table_name))
            for idx_sql in index_sql:
                await conn.execute(idx_sql)
            logger.info(f"Created new table: {table_name} (with portfolio_id)")
            return

        # Check if already has portfolio_id column
        cursor = await conn.execute(f"PRAGMA table_info({table_name})")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]

        if "portfolio_id" in column_names:
            logger.info(f"Table {table_name} already has portfolio_id column, skipping")
            return

        # Count existing rows for logging
        cursor = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = (await cursor.fetchone())[0]

        # Create new table
        await conn.execute(create_sql)

        # Copy data
        if row_count > 0:
            await conn.execute(insert_sql)

        # Drop old table and rename new one
        await conn.execute(f"DROP TABLE {table_name}")
        await conn.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")

        # Create indexes
        for idx_sql in index_sql:
            await conn.execute(idx_sql)

        logger.info(f"Migrated table {table_name}: {row_count} rows, added portfolio_id column")

"""Quarantined legacy multiuser/multi-portfolio migration.

The former migration copied only the SQLite main file, rewrote financial
tables with foreign keys disabled, collapsed account rows, and attempted
rollback by overwriting the authoritative database.  Read-only version
inspection and the already-applied no-op remain available.  Every mutation,
backup, restore, and table-rewrite entrypoint now fails closed.
"""

from pathlib import Path
from typing import List

import aiosqlite

from ..logger import get_logger

logger = get_logger(__name__)

MIGRATION_VERSION = 1
ALLOWED_MIGRATION_TABLES = frozenset(
    {
        "positions",
        "trades",
        "account",
        "signals",
        "equity_history",
        "market_data",
        "ticks",
        "portfolios",
    }
)

_QUARANTINE_MESSAGE = (
    "legacy multiuser mutation is disabled: use descriptor-bound online backup, "
    "clean-room restore, and a reviewed migration dry-run on a synthetic copy"
)


class LegacyMultiuserMigrationDisabled(RuntimeError):
    """The unsafe legacy migration or overwrite restore was requested."""


class MultiuserMigration:
    """Provide inspection and fail-closed compatibility for the old API."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    async def get_migration_version(self, conn: aiosqlite.Connection) -> int:
        """Read the legacy migration version without modifying schema."""

        try:
            cursor = await conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else 0
        except aiosqlite.OperationalError:
            return 0

    async def needs_migration(self) -> bool:
        """Report whether the quarantined legacy schema would need migration."""

        if not self.db_path.exists():
            return True
        async with aiosqlite.connect(self._readonly_uri(), uri=True) as connection:
            current = await self.get_migration_version(connection)
            return current < MIGRATION_VERSION

    async def _create_backup(self) -> None:
        """Reject the legacy raw-copy backup path."""

        raise LegacyMultiuserMigrationDisabled(_QUARANTINE_MESSAGE)

    async def migrate(self, default_cash: float = 100_000.0) -> bool:
        """No-op if absent/applied; reject every legacy mutation request."""

        del default_cash
        if not self.db_path.exists():
            logger.info("No database found; quarantined migration made no changes")
            return False
        async with aiosqlite.connect(self._readonly_uri(), uri=True) as connection:
            current_version = await self.get_migration_version(connection)
            if current_version >= MIGRATION_VERSION:
                logger.info(
                    "Database already at migration version %s; no changes made",
                    current_version,
                )
                return False
        raise LegacyMultiuserMigrationDisabled(_QUARANTINE_MESSAGE)

    def _readonly_uri(self) -> str:
        path = self.db_path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.as_uri() + "?mode=ro"

    async def _restore_from_backup(self, backup_path: Path) -> None:
        """Reject overwrite-based rollback even when called directly."""

        del backup_path
        raise LegacyMultiuserMigrationDisabled(_QUARANTINE_MESSAGE)

    async def _apply_migration_v1(
        self,
        conn: aiosqlite.Connection,
        default_cash: float,
    ) -> None:
        """Reject the old table-rewrite migration even when called directly."""

        del conn, default_cash
        raise LegacyMultiuserMigrationDisabled(_QUARANTINE_MESSAGE)

    async def _migrate_table_add_portfolio_id(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        create_sql: str,
        insert_sql: str,
        index_sql: List[str],
    ) -> None:
        """Retain validation compatibility, then reject all table rewrites."""

        del conn, create_sql, insert_sql, index_sql
        if table_name not in ALLOWED_MIGRATION_TABLES:
            raise ValueError(
                f"unexpected table_name in migration: {table_name!r}; "
                f"must be one of {sorted(ALLOWED_MIGRATION_TABLES)}"
            )
        raise LegacyMultiuserMigrationDisabled(_QUARANTINE_MESSAGE)

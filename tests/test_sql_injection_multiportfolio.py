"""SQL injection security tests for portfolio_id parameter.

Verifies that all database methods using portfolio_id are safe against
SQL injection attacks, whether through parameterized queries (SQLite ?
placeholders) or input validation (DatabaseValidator.validate_portfolio_id).
"""

import tempfile
from pathlib import Path

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import ValidationError

# SQL injection payloads to test
INJECTION_PAYLOADS = [
    "default' OR '1'='1",
    "default'; DROP TABLE positions; --",
    "default' UNION SELECT * FROM positions WHERE '1'='1",
    "'; DELETE FROM trades; --",
    "default' OR portfolio_id IS NOT NULL --",
]


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
async def db(temp_db):
    """Create an initialized async database with seed data in two portfolios."""
    database = AsyncTradingDatabase(db_path=temp_db)
    await database.initialize()

    # Seed portfolio_a
    await database.update_position("AAPL", 100, 180.0, portfolio_id="portfolio_a")
    await database.record_trade("AAPL", "BUY", 100, 180.0, portfolio_id="portfolio_a")
    await database.update_account(50000, 55000, portfolio_id="portfolio_a")
    await database.save_equity_snapshot(
        55000, 50000, 5000, snapshot_date="2026-02-10", portfolio_id="portfolio_a"
    )
    await database.record_signal("AAPL", "momentum", "BUY", 0.8, portfolio_id="portfolio_a")

    # Seed portfolio_b
    await database.update_position("MSFT", 50, 300.0, portfolio_id="portfolio_b")
    await database.record_trade("MSFT", "BUY", 50, 300.0, portfolio_id="portfolio_b")
    await database.update_account(80000, 82000, portfolio_id="portfolio_b")
    await database.save_equity_snapshot(
        82000, 80000, 2000, snapshot_date="2026-02-10", portfolio_id="portfolio_b"
    )
    await database.record_signal("MSFT", "ml_enhanced", "SELL", 0.7, portfolio_id="portfolio_b")

    yield database
    await database.close()


# ──────────────────────────────────────────────
# SELECT Injection Tests
# ──────────────────────────────────────────────


class TestSelectInjection:
    """Verify that SELECT queries with injected portfolio_id cannot leak data."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_get_positions_injection(self, db, payload):
        """get_positions with injection payload returns empty, not all data."""
        try:
            result = await db.get_positions(portfolio_id=payload)
            assert result == [], f"Injection payload returned data: {result}"
        except (ValidationError, ValueError):
            pass  # Format validation caught it

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_get_position_single_injection(self, db, payload):
        """get_position with injection payload returns None."""
        try:
            result = await db.get_position("AAPL", portfolio_id=payload)
            assert result is None, f"Injection payload returned data: {result}"
        except (ValidationError, ValueError):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_get_recent_trades_injection(self, db, payload):
        """get_recent_trades with injection payload returns empty."""
        try:
            result = await db.get_recent_trades(portfolio_id=payload)
            assert result == [], f"Injection payload returned data: {result}"
        except (ValidationError, ValueError):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_get_account_info_injection(self, db, payload):
        """get_account_info with injection payload returns empty dict."""
        try:
            result = await db.get_account_info(portfolio_id=payload)
            assert result == {}, f"Injection payload returned data: {result}"
        except (ValidationError, ValueError):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_get_equity_history_injection(self, db, payload):
        """get_equity_history with injection payload returns empty."""
        try:
            result = await db.get_equity_history(portfolio_id=payload)
            assert result == [], f"Injection payload returned data: {result}"
        except (ValidationError, ValueError):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_has_recent_buy_trade_injection(self, db, payload):
        """has_recent_buy_trade with injection payload returns False."""
        try:
            result = await db.has_recent_buy_trade("AAPL", 600, portfolio_id=payload)
            assert result is False, f"Injection payload returned True"
        except (ValidationError, ValueError):
            pass

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_has_recent_sell_trade_injection(self, db, payload):
        """has_recent_sell_trade with injection payload returns False."""
        try:
            result = await db.has_recent_sell_trade("AAPL", 600, portfolio_id=payload)
            assert result is False, f"Injection payload returned True"
        except (ValidationError, ValueError):
            pass


# ──────────────────────────────────────────────
# INSERT Injection Tests (DROP TABLE attempts)
# ──────────────────────────────────────────────


class TestInsertInjection:
    """Verify that INSERT operations with injected portfolio_id cannot destroy data."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_record_trade_drop_table(self, db, payload):
        """record_trade with DROP TABLE payload does not destroy tables."""
        import aiosqlite

        try:
            await db.record_trade("AAPL", "BUY", 10, 180.0, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass  # Validation caught it

        # Verify all tables still exist
        async with aiosqlite.connect(db.db_path) as conn:
            for table in ["positions", "trades", "account", "equity_history", "signals"]:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                row = await cursor.fetchone()
                assert row is not None, f"Table '{table}' was destroyed by injection"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_record_signal_drop_table(self, db, payload):
        """record_signal with injection payload does not destroy tables."""
        import aiosqlite

        try:
            await db.record_signal("AAPL", "test", "BUY", 0.5, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            )
            assert await cursor.fetchone() is not None, "signals table destroyed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_save_equity_snapshot_drop_table(self, db, payload):
        """save_equity_snapshot with injection payload does not destroy tables."""
        import aiosqlite

        try:
            await db.save_equity_snapshot(50000, 48000, 2000, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='equity_history'"
            )
            assert await cursor.fetchone() is not None, "equity_history table destroyed"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_update_account_drop_table(self, db, payload):
        """update_account with injection payload does not destroy tables."""
        import aiosqlite

        try:
            await db.update_account(50000, 55000, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        async with aiosqlite.connect(db.db_path) as conn:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='account'"
            )
            assert await cursor.fetchone() is not None, "account table destroyed"


# ──────────────────────────────────────────────
# UPDATE Injection Tests (cross-portfolio leaks)
# ──────────────────────────────────────────────


class TestUpdateInjection:
    """Verify that UPDATE/INSERT with injected portfolio_id cannot modify other portfolios."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_update_position_no_cross_portfolio_leak(self, db, payload):
        """update_position with injection payload does not alter other portfolios."""
        # Record original state
        pos_a_before = await db.get_positions(portfolio_id="portfolio_a")
        pos_b_before = await db.get_positions(portfolio_id="portfolio_b")

        try:
            await db.update_position("AAPL", 999, 1.0, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        # Verify both portfolios unchanged
        pos_a_after = await db.get_positions(portfolio_id="portfolio_a")
        pos_b_after = await db.get_positions(portfolio_id="portfolio_b")

        assert pos_a_after == pos_a_before, "portfolio_a positions changed by injection"
        assert pos_b_after == pos_b_before, "portfolio_b positions changed by injection"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_update_account_no_cross_portfolio_leak(self, db, payload):
        """update_account with injection payload does not alter other portfolios."""
        acct_a_before = await db.get_account_info(portfolio_id="portfolio_a")
        acct_b_before = await db.get_account_info(portfolio_id="portfolio_b")

        try:
            await db.update_account(0, 0, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        acct_a_after = await db.get_account_info(portfolio_id="portfolio_a")
        acct_b_after = await db.get_account_info(portfolio_id="portfolio_b")

        assert acct_a_after == acct_a_before, "portfolio_a account changed by injection"
        assert acct_b_after == acct_b_before, "portfolio_b account changed by injection"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_record_trade_no_cross_portfolio_leak(self, db, payload):
        """record_trade with injection payload does not add trades to other portfolios."""
        trades_a_before = await db.get_recent_trades(portfolio_id="portfolio_a")
        trades_b_before = await db.get_recent_trades(portfolio_id="portfolio_b")

        try:
            await db.record_trade("AAPL", "BUY", 10, 180.0, portfolio_id=payload)
        except (ValidationError, ValueError):
            pass

        trades_a_after = await db.get_recent_trades(portfolio_id="portfolio_a")
        trades_b_after = await db.get_recent_trades(portfolio_id="portfolio_b")

        assert len(trades_a_after) == len(trades_a_before), "portfolio_a trades changed"
        assert len(trades_b_after) == len(trades_b_before), "portfolio_b trades changed"


# ──────────────────────────────────────────────
# Data Integrity After All Injections
# ──────────────────────────────────────────────


class TestDataIntegrityAfterInjections:
    """Run all injection payloads and verify seed data remains intact."""

    @pytest.mark.asyncio
    async def test_full_injection_barrage_preserves_data(self, db):
        """Fire all payloads at all methods, then verify seed data integrity."""
        import aiosqlite

        for payload in INJECTION_PAYLOADS:
            # SELECT methods
            for method_name in [
                "get_positions",
                "get_account_info",
                "get_equity_history",
                "get_recent_trades",
            ]:
                try:
                    method = getattr(db, method_name)
                    await method(portfolio_id=payload)
                except (ValidationError, ValueError):
                    pass

            # has_recent_*
            try:
                await db.has_recent_buy_trade("AAPL", 600, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass
            try:
                await db.has_recent_sell_trade("AAPL", 600, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass

            # WRITE methods
            try:
                await db.record_trade("AAPL", "BUY", 10, 180.0, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass
            try:
                await db.update_position("AAPL", 999, 1.0, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass
            try:
                await db.update_account(0, 0, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass
            try:
                await db.save_equity_snapshot(0, 0, 0, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass
            try:
                await db.record_signal("AAPL", "test", "BUY", 0.5, portfolio_id=payload)
            except (ValidationError, ValueError):
                pass

        # Verify all tables still exist
        async with aiosqlite.connect(db.db_path) as conn:
            for table in ["positions", "trades", "account", "equity_history", "signals"]:
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                )
                assert await cursor.fetchone() is not None, f"Table '{table}' destroyed"

        # Verify portfolio_a data intact
        pos_a = await db.get_positions(portfolio_id="portfolio_a")
        assert len(pos_a) == 1
        assert pos_a[0]["symbol"] == "AAPL"
        assert pos_a[0]["quantity"] == 100

        trades_a = await db.get_recent_trades(portfolio_id="portfolio_a")
        assert len(trades_a) == 1
        assert trades_a[0]["symbol"] == "AAPL"

        acct_a = await db.get_account_info(portfolio_id="portfolio_a")
        assert acct_a["cash"] == 50000
        assert acct_a["equity"] == 55000

        # Verify portfolio_b data intact
        pos_b = await db.get_positions(portfolio_id="portfolio_b")
        assert len(pos_b) == 1
        assert pos_b[0]["symbol"] == "MSFT"
        assert pos_b[0]["quantity"] == 50

        trades_b = await db.get_recent_trades(portfolio_id="portfolio_b")
        assert len(trades_b) == 1
        assert trades_b[0]["symbol"] == "MSFT"

        acct_b = await db.get_account_info(portfolio_id="portfolio_b")
        assert acct_b["cash"] == 80000
        assert acct_b["equity"] == 82000

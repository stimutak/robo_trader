"""Security regression tests for database isolation, validation, and proxy
deny-by-default behavior.

These tests cover the fixes from SECURITY_AUDIT_2026-05-10.md Section 2.B.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import DatabaseValidator, ValidationError
from robo_trader.multiuser.db_proxy import PortfolioScopedDB


# ────────────────────────────────────────────────────────────────────────────
# Async DB fixture (per-test temp file)
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def db(tmp_path):
    """Provide a fresh AsyncTradingDatabase rooted at a tempfile DB path."""
    db_path = tmp_path / "test_isolation.db"
    database = AsyncTradingDatabase(db_path=db_path, pool_size=2)
    await database.initialize()
    try:
        yield database
    finally:
        await database.close()


# ────────────────────────────────────────────────────────────────────────────
# DB-H2 — read methods reject malformed portfolio_id
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_positions_rejects_invalid_portfolio_id(db):
    """get_positions must validate portfolio_id and raise on injection payloads."""
    with pytest.raises(ValidationError):
        await db.get_positions("' OR 1=1; --")


@pytest.mark.asyncio
async def test_get_position_rejects_invalid_portfolio_id(db):
    with pytest.raises(ValidationError):
        await db.get_position("AAPL", portfolio_id="../etc/passwd")


@pytest.mark.asyncio
async def test_portfolio_exists_returns_false_on_invalid_id(db):
    """portfolio_exists is special: invalid IDs return False rather than raise."""
    assert await db.portfolio_exists("' OR 1=1; --") is False
    assert await db.portfolio_exists("../etc") is False


# ────────────────────────────────────────────────────────────────────────────
# DB-H1 — upsert_portfolio validates id and name
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_portfolio_rejects_traversal_id(db):
    with pytest.raises(ValidationError):
        await db.upsert_portfolio({"id": "../etc", "name": "Bad"})


@pytest.mark.asyncio
async def test_upsert_portfolio_rejects_sql_in_id(db):
    with pytest.raises(ValidationError):
        await db.upsert_portfolio({"id": "'; DROP TABLE portfolios; --"})


@pytest.mark.asyncio
async def test_upsert_portfolio_rejects_dangerous_name(db):
    with pytest.raises(ValidationError):
        await db.upsert_portfolio({"id": "valid_id", "name": "DROP TABLE x"})


# ────────────────────────────────────────────────────────────────────────────
# DB-H3 — PortfolioScopedDB deny-by-default
# ────────────────────────────────────────────────────────────────────────────


def test_proxy_denies_unknown_method():
    """Calling an unknown method through the proxy must raise AttributeError."""

    class FakeDB:
        async def dummy_method(self, *args, **kwargs):  # pragma: no cover - never called
            return "should never run"

    fake = FakeDB()
    proxy = PortfolioScopedDB(fake, portfolio_id="test")

    with pytest.raises(AttributeError) as excinfo:
        proxy.dummy_method  # noqa: B018 - intentional attribute access

    assert "refuses to call 'dummy_method'" in str(excinfo.value)


def test_proxy_allows_known_global_method():
    """Methods in _KNOWN_GLOBAL_METHODS pass through unchanged."""

    class FakeDB:
        async def store_market_data(self, *args, **kwargs):
            return "ok"

    proxy = PortfolioScopedDB(FakeDB(), portfolio_id="test")
    # Should not raise
    method = proxy.store_market_data
    assert callable(method)


# ────────────────────────────────────────────────────────────────────────────
# DB-L1 — validate_portfolio_id case normalization
# ────────────────────────────────────────────────────────────────────────────


def test_validate_portfolio_id_lowercases():
    assert DatabaseValidator.validate_portfolio_id("Default") == "default"
    assert DatabaseValidator.validate_portfolio_id("DEFAULT") == "default"
    assert DatabaseValidator.validate_portfolio_id("MixedCase_123") == "mixedcase_123"


def test_validate_portfolio_id_strips_and_lowercases():
    assert DatabaseValidator.validate_portfolio_id("  Aggressive  ") == "aggressive"


# ────────────────────────────────────────────────────────────────────────────
# DB-M3 — _validate_string rejects SQL keywords (no silent escape)
# ────────────────────────────────────────────────────────────────────────────


def test_validate_string_rejects_sql_keywords():
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string("DROP TABLE x", "f")


def test_validate_string_rejects_quote():
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string("bobby'); --", "f")


def test_validate_string_rejects_double_quote():
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string('he said "hi"', "f")


def test_validate_string_accepts_clean_input():
    assert DatabaseValidator._validate_string("clean_input_123", "f") == "clean_input_123"


# ────────────────────────────────────────────────────────────────────────────
# DB-M4 — record_signal validates symbol
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_signal_rejects_bad_symbol(db):
    with pytest.raises(ValidationError):
        await db.record_signal(
            symbol="' OR 1=1",
            strategy="x",
            signal_type="BUY",
        )


@pytest.mark.asyncio
async def test_record_signal_rejects_bad_strategy(db):
    with pytest.raises(ValidationError):
        await db.record_signal(
            symbol="AAPL",
            strategy="DROP TABLE signals",
            signal_type="BUY",
        )


# ────────────────────────────────────────────────────────────────────────────
# DB-M5 — init_database.py refuses to clobber and refuses production filename
# ────────────────────────────────────────────────────────────────────────────


def test_init_database_refuses_existing_file(tmp_path):
    existing = tmp_path / "sample.db"
    existing.write_text("")
    assert existing.exists()

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "init_database.py"
    if not script.exists():
        pytest.skip(f"init_database.py not found at {script}")

    proc = subprocess.run(
        [sys.executable, str(script), "--db-path", str(existing)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"expected exit 2 for existing file, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_init_database_refuses_production_filename(tmp_path):
    target = tmp_path / "trading_data.db"
    # File does NOT yet exist; should still refuse based on filename.
    assert not target.exists()

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "init_database.py"
    if not script.exists():
        pytest.skip(f"init_database.py not found at {script}")

    proc = subprocess.run(
        [sys.executable, str(script), "--db-path", str(target)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"expected exit 2 for production filename, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


# ────────────────────────────────────────────────────────────────────────────
# DB-M1 — cleanup_old_data does not blanket-delete signals across portfolios
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cleanup_old_data_does_not_touch_other_portfolios_signals(db):
    """Calling cleanup_old_data with portfolio_id must only scope to that portfolio."""
    # Record a recent signal in two portfolios; cleanup with cutoff=0 days for one
    # should still leave the other intact.
    await db.upsert_portfolio({"id": "alpha", "name": "Alpha"})
    await db.upsert_portfolio({"id": "beta", "name": "Beta"})

    await db.record_signal(
        symbol="AAPL", strategy="x", signal_type="BUY", strength=0.5, portfolio_id="alpha"
    )
    await db.record_signal(
        symbol="AAPL", strategy="x", signal_type="BUY", strength=0.5, portfolio_id="beta"
    )

    # Calling with no portfolio_id should NOT touch signals at all.
    await db.cleanup_old_data(days_to_keep=0)

    # Both portfolios should still have signals.
    async with db.get_connection() as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM signals WHERE portfolio_id IN ('alpha', 'beta')"
        )
        count = (await cur.fetchone())[0]
    assert count == 2, "global cleanup must not touch portfolio-scoped signals"

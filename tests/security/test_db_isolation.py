"""Security regression tests for database isolation, validation, and proxy
deny-by-default behavior.

These tests cover the fixes from SECURITY_AUDIT_2026-05-10.md Section 2.B
AND the Round-2 follow-ups from SECURITY_AUDIT_ROUND2_2026-05-10.md Section 2.B
(DB-R2-M1, DB-R2-M2, DB-R2-L1, DB-R2-L3).
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_validator import (
    DatabaseValidator,
    ValidationError,
    validate_portfolio_id,
)
from robo_trader.multiuser.db_proxy import PortfolioScopedDB
from robo_trader.multiuser.portfolio_config import load_portfolio_configs


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
    # DB-R2-L1: name validation now rejects literal SQL terminators / quotes
    # (not bare keywords). Use a classic injection payload that includes them.
    with pytest.raises(ValidationError):
        await db.upsert_portfolio({"id": "valid_id", "name": "x'); DROP TABLE x; --"})


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
# DB-M3 / DB-R2-L1 — _validate_string rejects SQL terminators / quotes only
# (the over-broad keyword denylist was removed per DB-R2-L1; literal terminator
# checks remain as defense-in-depth alongside parameterized queries).
# ────────────────────────────────────────────────────────────────────────────


def test_validate_string_rejects_terminator_with_keyword():
    """Classic injection (terminator + keyword) still fails because it has ';' and '--'."""
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string("DROP TABLE x; --", "f")


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
    # DB-R2-L1: strategy validation rejects literal terminators / quotes,
    # not bare keywords. Use a classic injection payload with terminators.
    with pytest.raises(ValidationError):
        await db.record_signal(
            symbol="AAPL",
            strategy="'; DROP TABLE signals; --",
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


# ────────────────────────────────────────────────────────────────────────────
# DB-R2-M1 — load_portfolio_configs lowercases BEFORE dedupe check
# ────────────────────────────────────────────────────────────────────────────


def test_load_portfolio_configs_rejects_case_collision_dupes():
    """'Default' and 'default' must be detected as duplicates after normalization."""
    payload = json.dumps([
        {"id": "Default", "name": "First", "starting_cash": 50000, "symbols": "AAPL"},
        {"id": "default", "name": "Second", "starting_cash": 60000, "symbols": "MSFT"},
    ])
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_portfolio_configs()


def test_load_portfolio_configs_rejects_whitespace_collision_dupes():
    """'  default  ' and 'default' must collide after strip+lower."""
    payload = json.dumps([
        {"id": "default", "name": "First", "starting_cash": 50000, "symbols": "AAPL"},
        {"id": "  DEFAULT  ", "name": "Second", "starting_cash": 60000, "symbols": "MSFT"},
    ])
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_portfolio_configs()


def test_load_portfolio_configs_normalizes_id_on_resulting_config():
    """The PortfolioConfig.id field must be the lowercased/stripped form so
    downstream lookups (DB queries via validate_portfolio_id) match."""
    payload = json.dumps([
        {"id": "  Aggressive  ", "name": "A", "starting_cash": 50000, "symbols": "NVDA"},
        {"id": "CONSERVATIVE", "name": "B", "starting_cash": 50000, "symbols": "JNJ"},
    ])
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        configs = load_portfolio_configs()
    ids = sorted(c.id for c in configs)
    assert ids == ["aggressive", "conservative"]


def test_load_portfolio_configs_rejects_non_string_id():
    payload = json.dumps([{"id": 12345, "name": "Bad"}])
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError):
            load_portfolio_configs()


def test_load_portfolio_configs_rejects_empty_id_after_strip():
    payload = json.dumps([{"id": "   ", "name": "Bad"}])
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError):
            load_portfolio_configs()


# ────────────────────────────────────────────────────────────────────────────
# DB-R2-M2 — sync_db_reader validates portfolio_id on every public method
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sync_reader(tmp_path):
    """Provide a SyncDatabaseReader pointed at an empty temp DB.

    The DB does not need to be initialized for these validation tests —
    validate_portfolio_id runs at the top of each method, before any query.
    """
    from sync_db_reader import SyncDatabaseReader

    db_path = tmp_path / "sync_isolation.db"
    # Create empty file so sqlite open doesn't fail on the validation-passes path.
    db_path.touch()
    return SyncDatabaseReader(db_path=str(db_path))


@pytest.mark.parametrize(
    "method_name,extra_args",
    [
        ("get_positions", ()),
        ("get_recent_trades", ()),
        ("get_account_info", ()),
        ("get_signals", ()),
        ("get_equity_history", ()),
    ],
)
def test_sync_reader_rejects_sql_injection_portfolio_id(sync_reader, method_name, extra_args):
    """Every public sync_db_reader method that accepts portfolio_id must
    raise ValidationError on injection-style payloads."""
    method = getattr(sync_reader, method_name)
    with pytest.raises(ValidationError):
        if method_name == "get_recent_trades":
            method(*extra_args, portfolio_id="' OR 1=1; --")
        elif method_name == "get_signals":
            method(*extra_args, portfolio_id="../etc/passwd")
        else:
            method(portfolio_id="' OR 1=1; --")


def test_sync_reader_rejects_traversal_in_get_equity_history(sync_reader):
    with pytest.raises(ValidationError):
        sync_reader.get_equity_history(portfolio_id="../../etc/passwd")


def test_sync_reader_rejects_quote_in_get_account_info(sync_reader):
    with pytest.raises(ValidationError):
        sync_reader.get_account_info(portfolio_id='"; DROP TABLE account; --')


def test_sync_reader_normalizes_valid_mixed_case(sync_reader):
    """Validation lowercases the id; mixed-case input should NOT raise."""
    # We don't care about the query result here — just that validation passes.
    # An empty DB will yield [] from the read; that's fine.
    try:
        result = sync_reader.get_positions(portfolio_id="Default")
    except ValidationError:
        pytest.fail("Mixed-case 'Default' must pass validation (it's just lowercased)")
    assert isinstance(result, list)


def test_validate_portfolio_id_module_level_alias_works():
    """The module-level validate_portfolio_id should mirror the static method."""
    assert validate_portfolio_id("Default") == "default"
    with pytest.raises(ValidationError):
        validate_portfolio_id("' OR 1=1")


# ────────────────────────────────────────────────────────────────────────────
# DB-R2-L1 — _validate_string drops keyword denylist; only literal terminators rejected
# ────────────────────────────────────────────────────────────────────────────


def test_validate_string_accepts_legitimate_words_with_sql_keywords():
    """Legitimate metadata containing words like 'select', 'drop', 'update'
    in business context (e.g. JSON-encoded strategy notes) MUST pass.

    The previous over-broad keyword denylist rejected these; defense now
    relies on parameterized queries, not keyword scanning."""
    # All of these used to be rejected by the denylist:
    assert (
        DatabaseValidator._validate_string("user wants to select aggressive mode", "f")
        == "user wants to select aggressive mode"
    )
    assert (
        DatabaseValidator._validate_string("price drop alert: NVDA", "f")
        == "price drop alert: NVDA"
    )
    assert (
        DatabaseValidator._validate_string("strategy update applied", "f")
        == "strategy update applied"
    )
    assert (
        DatabaseValidator._validate_string("delete confirmation pending", "f")
        == "delete confirmation pending"
    )
    assert (
        DatabaseValidator._validate_string("insert order at market", "f")
        == "insert order at market"
    )


def test_validate_string_still_rejects_literal_terminators():
    """Literal SQL terminators / comment markers / quotes are still rejected."""
    for bad in [";", "--", "/*", "*/", "'", '"']:
        with pytest.raises(ValidationError):
            DatabaseValidator._validate_string(f"prefix {bad} suffix", "f")


def test_validate_string_rejects_classic_injection_payload():
    """Classic injection patterns still fail because they contain quotes/terminators."""
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string("'; DROP TABLE x; --", "f")
    with pytest.raises(ValidationError):
        DatabaseValidator._validate_string("admin'--", "f")


def test_validate_string_accepts_json_metadata_with_keyword_substrings():
    """JSON-style metadata blobs that happen to contain keyword substrings
    (without quote/terminator chars) must pass."""
    # No quotes, no semicolons, no comment markers — even if it contains
    # words like "select" or "update" as substrings.
    blob = "regime=range_bound mode=update_pending action=select_top_k k=5"
    assert DatabaseValidator._validate_string(blob, "f") == blob


# ────────────────────────────────────────────────────────────────────────────
# DB-R2-L3 — init_database.py refuses symlinks
# ────────────────────────────────────────────────────────────────────────────


def test_init_database_refuses_symlink(tmp_path):
    """A symlink (even pointing to a fresh path) must be rejected outright."""
    target = tmp_path / "real_target.db"
    link = tmp_path / "via_symlink.db"
    # Create symlink that does NOT yet point at an existing file — that's fine,
    # we just want to verify symlink detection trips.
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support symlink creation")

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "init_database.py"
    if not script.exists():
        pytest.skip(f"init_database.py not found at {script}")

    proc = subprocess.run(
        [sys.executable, str(script), "--db-path", str(link)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"expected exit 2 for symlink path, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "symlink" in (proc.stderr + proc.stdout).lower()


def test_init_database_refuses_resolved_production_filename(tmp_path):
    """Even a path like ./subdir/../trading_data.db (resolves to trading_data.db
    under the cwd) must be refused without --force."""
    # tmp_path / "subdir" / ".." / "trading_data.db" resolves to tmp_path/trading_data.db
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    tricky = subdir / ".." / "trading_data.db"
    # File doesn't exist yet — relies on filename + resolve check.

    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "init_database.py"
    if not script.exists():
        pytest.skip(f"init_database.py not found at {script}")

    proc = subprocess.run(
        [sys.executable, str(script), "--db-path", str(tricky)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode == 2, (
        f"expected exit 2 for resolved production filename, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )

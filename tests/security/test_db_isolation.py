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
    payload = json.dumps(
        [
            {"id": "Default", "name": "First", "starting_cash": 50000, "symbols": "AAPL"},
            {"id": "default", "name": "Second", "starting_cash": 60000, "symbols": "MSFT"},
        ]
    )
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_portfolio_configs()


def test_load_portfolio_configs_rejects_whitespace_collision_dupes():
    """'  default  ' and 'default' must collide after strip+lower."""
    payload = json.dumps(
        [
            {"id": "default", "name": "First", "starting_cash": 50000, "symbols": "AAPL"},
            {"id": "  DEFAULT  ", "name": "Second", "starting_cash": 60000, "symbols": "MSFT"},
        ]
    )
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError, match=r"[Dd]uplicate"):
            load_portfolio_configs()


def test_load_portfolio_configs_normalizes_id_on_resulting_config():
    """The PortfolioConfig.id field must be the lowercased/stripped form so
    downstream lookups (DB queries via validate_portfolio_id) match."""
    payload = json.dumps(
        [
            {"id": "  Aggressive  ", "name": "A", "starting_cash": 50000, "symbols": "NVDA"},
            {"id": "CONSERVATIVE", "name": "B", "starting_cash": 50000, "symbols": "JNJ"},
        ]
    )
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


# ────────────────────────────────────────────────────────────────────────────
# Followup audit (SECURITY_AUDIT_2026-05-10_FOLLOWUP.md section 2.D)
# ────────────────────────────────────────────────────────────────────────────


def test_portfolio_config_clamps_max_position_pct_d_9():
    """D-9: per-portfolio max_position_pct must be clamped to <= 0.25."""
    payload = json.dumps(
        [
            {
                "id": "greedy",
                "name": "Greedy",
                "starting_cash": 50000,
                "symbols": "NVDA",
                "max_position_pct": 0.95,  # would let one position consume 95%
                "max_open_positions": 9999,  # absurd
                "trailing_stop_pct": 5.0,  # 500%
                "stop_loss_pct": 0.99,  # 99%
            }
        ]
    )
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        configs = load_portfolio_configs()
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.max_position_pct == 0.25, "max_position_pct must clamp to 0.25"
    assert cfg.max_open_positions == 50, "max_open_positions must clamp to 50"
    assert cfg.trailing_stop_pct == 0.5, "trailing_stop_pct must clamp to 0.5"
    assert cfg.stop_loss_pct == 0.5, "stop_loss_pct must clamp to 0.5"


def test_portfolio_config_passes_through_safe_values_d_9():
    """D-9: values below the hard ceilings must pass through unchanged."""
    payload = json.dumps(
        [
            {
                "id": "safe",
                "name": "Safe",
                "starting_cash": 50000,
                "symbols": "AAPL",
                "max_position_pct": 0.10,
                "max_open_positions": 8,
                "trailing_stop_pct": 0.05,
                "stop_loss_pct": 0.02,
            }
        ]
    )
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        configs = load_portfolio_configs()
    cfg = configs[0]
    assert cfg.max_position_pct == 0.10
    assert cfg.max_open_positions == 8
    assert cfg.trailing_stop_pct == 0.05
    assert cfg.stop_loss_pct == 0.02


def test_portfolio_config_rejects_negative_risk_overrides_d_9():
    """Negative risk overrides must raise ValueError, not silently clamp."""
    payload = json.dumps(
        [
            {
                "id": "neg",
                "name": "Neg",
                "starting_cash": 50000,
                "symbols": "AAPL",
                "max_position_pct": -0.1,
            }
        ]
    )
    with patch.dict(os.environ, {"PORTFOLIOS": payload}, clear=False):
        with pytest.raises(ValueError, match="non-negative"):
            load_portfolio_configs()


@pytest.mark.asyncio
async def test_db_proxy_cleanup_old_data_requires_portfolio_id_d_8(db):
    """D-8: cleanup_old_data is no longer in _KNOWN_GLOBAL_METHODS. A scoped
    proxy must auto-inject the holder's portfolio_id, never call without it.

    This ensures that a scoped DB holder cannot accidentally blanket-clean
    signals across all portfolios.
    """
    from robo_trader.multiuser.db_proxy import (
        _KNOWN_GLOBAL_METHODS,
        _PORTFOLIO_SCOPED_METHODS,
        PortfolioScopedDB,
    )

    # Structural assertion: the proxy must classify cleanup_old_data as
    # scoped, not global.
    assert "cleanup_old_data" not in _KNOWN_GLOBAL_METHODS, (
        "D-8: cleanup_old_data must NOT be classified as a global method; "
        "doing so would let a scoped holder blanket-clean across portfolios."
    )
    assert "cleanup_old_data" in _PORTFOLIO_SCOPED_METHODS, (
        "D-8: cleanup_old_data must be in the portfolio-scoped set so the "
        "proxy auto-injects portfolio_id."
    )

    # Behavioral assertion: calls through the scoped proxy must scope to the
    # holder's portfolio. Set up two portfolios with recent signals and verify
    # that proxy cleanup for portfolio "alpha" only touches alpha's signals.
    await db.upsert_portfolio({"id": "alpha", "name": "Alpha"})
    await db.upsert_portfolio({"id": "beta", "name": "Beta"})
    await db.record_signal(
        symbol="AAPL",
        strategy="x",
        signal_type="BUY",
        strength=0.5,
        portfolio_id="alpha",
    )
    await db.record_signal(
        symbol="AAPL",
        strategy="x",
        signal_type="BUY",
        strength=0.5,
        portfolio_id="beta",
    )

    scoped = PortfolioScopedDB(db, portfolio_id="alpha")

    # Spy on the underlying cleanup_old_data to verify portfolio_id is
    # auto-injected by the proxy. Behavioral assertion via call inspection
    # avoids relying on CURRENT_TIMESTAMP / local-tz cutoff math (signals
    # use SQLite CURRENT_TIMESTAMP which is UTC, while cleanup uses
    # datetime.now() locally; with days_to_keep=0 the cutoff comparison
    # can be flaky across timezones).
    seen_kwargs = {}
    original = db.cleanup_old_data

    async def spy_cleanup(*args, **kwargs):
        seen_kwargs.update(kwargs)
        return await original(*args, **kwargs)

    db.cleanup_old_data = spy_cleanup  # type: ignore[assignment]
    try:
        await scoped.cleanup_old_data(days_to_keep=0)
    finally:
        db.cleanup_old_data = original  # type: ignore[assignment]

    assert seen_kwargs.get("portfolio_id") == "alpha", (
        "D-8: scoped proxy must auto-inject portfolio_id; " f"actual kwargs: {seen_kwargs}"
    )

    # beta's signal must remain untouched regardless of cutoff timing.
    async with db.get_connection() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM signals WHERE portfolio_id = 'beta'")
        beta_count = (await cur.fetchone())[0]
    assert beta_count == 1, "beta's signal must NOT be touched"


@pytest.mark.asyncio
async def test_db_proxy_cleanup_global_via_raw_db_d_8(db):
    """D-8: callers that want global cleanup must go through _db directly.
    The underlying cleanup_old_data still cleans market_data unconditionally
    when called without a portfolio_id."""
    from robo_trader.multiuser.db_proxy import PortfolioScopedDB

    scoped = PortfolioScopedDB(db, portfolio_id="alpha")
    # Direct call on the underlying DB without portfolio_id must work.
    await scoped._db.cleanup_old_data(days_to_keep=0)


def test_migration_table_name_allowlisted_d_17():
    """D-17: _migrate_table_add_portfolio_id must reject table_names that
    are not in the hardcoded allowlist, before any f-string DDL executes."""
    import asyncio as _asyncio

    from robo_trader.multiuser.migration import (
        ALLOWED_MIGRATION_TABLES,
        MultiuserMigration,
    )

    # Structural assertion: the allowlist exists and is non-empty.
    assert "positions" in ALLOWED_MIGRATION_TABLES
    assert "trades" in ALLOWED_MIGRATION_TABLES
    assert "drop_users" not in ALLOWED_MIGRATION_TABLES

    # Behavioral assertion: an unexpected table_name raises before any DDL.
    mig = MultiuserMigration(db_path=Path("/tmp/_d17_test_does_not_exist.db"))

    async def _run():
        # We never reach a real connection because the assertion fires first.
        # Pass a sentinel conn=None; the validation is the first line of the
        # function and never touches conn before failing.
        with pytest.raises(ValueError, match="unexpected table_name"):
            await mig._migrate_table_add_portfolio_id(
                conn=None,  # type: ignore[arg-type]
                table_name="users; DROP TABLE foo",
                create_sql="",
                insert_sql="",
                index_sql=[],
            )

    _asyncio.get_event_loop().run_until_complete(_run()) if False else _asyncio.run(_run())


def test_migration_table_name_allowlist_accepts_known_d_17():
    """The allowlist must include every table currently migrated by v1."""
    from robo_trader.multiuser.migration import ALLOWED_MIGRATION_TABLES

    # These are the table_name values passed in _apply_migration_v1.
    for name in ["positions", "trades", "account", "equity_history", "signals"]:
        assert (
            name in ALLOWED_MIGRATION_TABLES
        ), f"D-17: migration v1 migrates {name!r} but the allowlist excludes it"


@pytest.mark.asyncio
async def test_batch_store_market_data_rejects_missing_keys_d_16(db):
    """D-16: rows missing a bind-parameter key must raise ValueError before
    executemany sees them."""
    bad_row = {
        "symbol": "AAPL",
        "timestamp": "2026-01-01T00:00:00",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        # missing "volume"
    }
    with pytest.raises(ValueError, match="missing keys"):
        await db.batch_store_market_data([bad_row])


@pytest.mark.asyncio
async def test_batch_store_market_data_rejects_non_dict_row_d_16(db):
    with pytest.raises(ValueError, match="must be a dict"):
        await db.batch_store_market_data([("AAPL", "ts", 1, 2, 3, 4, 5)])


@pytest.mark.asyncio
async def test_batch_store_market_data_accepts_valid_d_16(db):
    """D-16: a fully-populated row must pass validation and store."""
    good_row = {
        "symbol": "AAPL",
        "timestamp": "2026-01-01 00:00:00",
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1000,
    }
    await db.batch_store_market_data([good_row])


def test_validate_symbol_rejects_quote_chars_via_regex_d_12():
    """D-12: even after removing the dead keyword denylist, quotes and SQL
    metacharacters must still be rejected — by the regex.

    Note: bare uppercase words like "DROP" are valid 4-letter symbols
    (the regex admits any 1-5 uppercase letters); the actual SQL injection
    guard is parameterized queries everywhere in the data layer. The regex
    rejects anything with quotes, semicolons, comment markers, or
    lowercase/punctuation that can't appear in a real ticker."""
    for bad in ["AAPL'", 'AAPL"', "AA;PL", "A--PL", "A/*PL", "DROP TABLE"]:
        with pytest.raises(ValidationError):
            DatabaseValidator.validate_symbol(bad)


def test_validate_symbol_accepts_legitimate_d_12():
    """D-12: legitimate symbols (including dot-suffix) still pass after the
    dead-code removal."""
    assert DatabaseValidator.validate_symbol("AAPL") == "AAPL"
    assert DatabaseValidator.validate_symbol("brk.b") == "BRK.B"
    assert DatabaseValidator.validate_symbol("MSFT") == "MSFT"

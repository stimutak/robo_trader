"""Tests for :class:`EquityHistoryFreshnessCheck` (spec §7.3).

Covers the decision matrix:

- missing DB file → BLOCK
- empty table → WARN (first-run)
- fresh row → PASS
- Friday-row, Monday-now → PASS (weekend bridge)
- 3-day-old row → BLOCK
- multi-portfolio: any-fresh-passes
- sqlite read error → BLOCK
- parsing edge cases on ``count_trading_days``
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pytest

from robo_trader.market_hours import count_trading_days
from robo_trader.preflight import CheckStatus, PreflightContext
from robo_trader.preflight.equity_history_freshness_check import (
    EquityHistoryFreshnessCheck,
)


@pytest.fixture
def equity_db(tmp_path: Path) -> Callable[..., Path]:
    """Build a minimal ``trading_data.db`` mirroring the production schema.

    Returns a function ``add(portfolio_id, timestamp, equity=...)`` that
    inserts a row. The DB file lives at ``tmp_path / "trading_data.db"`` so
    it's discovered by ``EquityHistoryFreshnessCheck`` running with
    ``project_root=tmp_path``.

    Mirrors the production schema (database_async.py:343) — same column
    names and types, just without the ancillary tables we don't query.
    """
    db_path = tmp_path / "trading_data.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE equity_history (
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
        """)
    conn.commit()

    def add(portfolio_id: str, timestamp: str, equity: float = 100_000.0) -> None:
        # Use the timestamp string as the date too — tests don't care about
        # date column, only timestamp.
        date_str = timestamp[:10]
        conn.execute(
            "INSERT INTO equity_history "
            "(portfolio_id, date, equity, timestamp) VALUES (?, ?, ?, ?)",
            (portfolio_id, date_str, equity, timestamp),
        )
        conn.commit()

    yield add
    conn.close()


def _freeze_market_time(monkeypatch: pytest.MonkeyPatch, when: datetime) -> None:
    """Patch ``get_market_time`` everywhere the check imports it from.

    The check imports ``get_market_time`` into its own module namespace, so
    we patch the binding the check actually uses — patching the source
    module would have no effect since the name is already resolved.
    """
    monkeypatch.setattr(
        "robo_trader.preflight.equity_history_freshness_check.get_market_time",
        lambda: when,
    )


class TestDBFileMissing:
    def test_missing_db_blocks(self, preflight_context: PreflightContext) -> None:
        # tmp_path has no trading_data.db by default.
        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "trading_data.db not found" in result.message
        assert "START_TRADER.sh" in result.remediation
        assert result.details["db_path"].endswith("trading_data.db")


class TestEmptyTable:
    def test_empty_table_warns_with_first_run_message(
        self,
        equity_db: Callable[..., Path],  # fixture creates schema only
        preflight_context: PreflightContext,
    ) -> None:
        # Don't add any rows — fixture creates an empty table.
        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.WARN
        assert "first-run" in result.message
        assert result.details["row_count"] == 0


class TestFreshRow:
    def test_single_row_from_today_passes(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Wednesday 2026-05-20 mid-morning.
        now = datetime(2026, 5, 20, 10, 30, 0)
        equity_db("default", now.strftime("%Y-%m-%d %H:%M:%S"))
        _freeze_market_time(monkeypatch, now)

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.details["trading_days_elapsed"] == 0


class TestWeekendBridge:
    def test_friday_row_monday_now_passes(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Row written Friday 2026-05-15 16:30 (EOD), startup Monday 2026-05-18 09:00.
        # Trading-day delta is 1 (Monday), wall-clock is ~64h.
        friday_ts = "2026-05-15 16:30:00"
        monday_now = datetime(2026, 5, 18, 9, 0, 0)
        equity_db("default", friday_ts)
        _freeze_market_time(monkeypatch, monday_now)

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.details["trading_days_elapsed"] == 1


class TestStaleRow:
    def test_three_trading_days_old_blocks(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Row written Wed 2026-05-13, startup the following Wed 2026-05-20.
        # That's 5 trading days in between.
        equity_db("default", "2026-05-13 16:30:00")
        _freeze_market_time(monkeypatch, datetime(2026, 5, 20, 9, 0, 0))

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert result.details["trading_days_elapsed"] == 5
        assert "trading days old" in result.message
        assert "reconcile_positions.py" in result.remediation


class TestMultiPortfolio:
    def test_any_fresh_portfolio_passes(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Aggressive portfolio is stale (10 trading days ago).
        # Conservative is fresh (today). PASS overall — MAX wins.
        now = datetime(2026, 5, 20, 10, 0, 0)
        equity_db("aggressive", "2026-05-04 16:30:00")
        equity_db("conservative", now.strftime("%Y-%m-%d %H:%M:%S"))
        _freeze_market_time(monkeypatch, now)

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.details["trading_days_elapsed"] == 0


class TestSqliteError:
    def test_sqlite_error_blocks(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force the check's sqlite3.connect to raise. We patch sqlite3.connect
        # inside the check module since that's the binding it resolves.
        def boom(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(
            "robo_trader.preflight.equity_history_freshness_check.sqlite3.connect",
            boom,
        )

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "sqlite read error" in result.message
        assert "database is locked" in result.details["error"]


class TestMalformedTimestamp:
    def test_unparseable_timestamp_blocks(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
    ) -> None:
        # Inject a junk timestamp the parser can't match.
        equity_db("default", "garbage-not-a-timestamp")

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "could not parse" in result.message


class TestCheckMetadata:
    def test_protocol_attributes_present(self) -> None:
        check = EquityHistoryFreshnessCheck()
        assert check.name == "equity_history_freshness"
        assert check.description == "Equity history freshness"
        assert check.timeout_seconds == 3.0


# ---------------------------------------------------------------------------
# count_trading_days unit tests (lives in robo_trader.market_hours but
# introduced in this commit, so its tests live alongside the consumer).
# ---------------------------------------------------------------------------


class TestCountTradingDays:
    def test_same_day_returns_zero(self) -> None:
        # Mid-day Wednesday → mid-day Wednesday.
        d = datetime(2026, 5, 20, 12, 0, 0)
        assert count_trading_days(d, d) == 0

    def test_reversed_interval_returns_zero(self) -> None:
        # end < start should not produce negative counts.
        later = datetime(2026, 5, 20, 12, 0, 0)
        earlier = datetime(2026, 5, 18, 12, 0, 0)
        assert count_trading_days(later, earlier) == 0

    def test_weekend_bridge_friday_to_monday(self) -> None:
        # Fri 2026-05-15 EOD → Mon 2026-05-18 morning: only Monday counts.
        friday = datetime(2026, 5, 15, 16, 30, 0)
        monday = datetime(2026, 5, 18, 9, 0, 0)
        assert count_trading_days(friday, monday) == 1

    def test_full_week_bridge(self) -> None:
        # Fri 2026-05-15 → Fri 2026-05-22: Mon, Tue, Wed, Thu, Fri = 5.
        assert (
            count_trading_days(
                datetime(2026, 5, 15, 16, 30, 0),
                datetime(2026, 5, 22, 16, 30, 0),
            )
            == 5
        )

    def test_holiday_bridge_excludes_thanksgiving(self) -> None:
        # Wed before Thanksgiving 2026 → Fri after.
        # 2026 Thanksgiving = Thursday Nov 26. Wed Nov 25 → Fri Nov 27.
        # Interval (start, end] = {Nov 26 (holiday, skip), Nov 27 (Fri)} → 1.
        wed_before = datetime(2026, 11, 25, 16, 0, 0)
        fri_after = datetime(2026, 11, 27, 13, 0, 0)
        assert count_trading_days(wed_before, fri_after) == 1

    def test_multi_week_gap(self) -> None:
        # 2 calendar weeks: Mon 2026-05-04 → Mon 2026-05-18.
        # Trading days in (May 4, May 18]: Tue 5, Wed 6, Thu 7, Fri 8,
        # Mon 11, Tue 12, Wed 13, Thu 14, Fri 15, Mon 18 = 10.
        start = datetime(2026, 5, 4, 12, 0, 0)
        end = datetime(2026, 5, 18, 9, 0, 0)
        assert count_trading_days(start, end) == 10

    def test_one_calendar_day_weekday_to_weekday(self) -> None:
        # Tue 2026-05-19 → Wed 2026-05-20: just Wednesday = 1.
        assert (
            count_trading_days(
                datetime(2026, 5, 19, 16, 0, 0),
                datetime(2026, 5, 20, 16, 0, 0),
            )
            == 1
        )

    def test_accepts_naive_or_aware_datetimes(self) -> None:
        # The helper works on .date() so timezone shouldn't matter for
        # day-counting. Naive vs aware should produce the same answer.
        from datetime import timezone

        naive_start = datetime(2026, 5, 15, 16, 30, 0)
        aware_start = naive_start.replace(tzinfo=timezone.utc)
        end = datetime(2026, 5, 18, 9, 0, 0)
        assert count_trading_days(naive_start, end) == count_trading_days(aware_start, end)


# ---------------------------------------------------------------------------
# Smoke: ensure the timedelta-based logic and "row from yesterday" produce
# a sensible PASS too, so we know we haven't off-by-one'd the boundary.
# ---------------------------------------------------------------------------


class TestBoundary:
    def test_one_trading_day_old_passes(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Row Tue, now Wed → 1 trading day elapsed → still PASS (<=1).
        equity_db("default", "2026-05-19 16:30:00")
        _freeze_market_time(monkeypatch, datetime(2026, 5, 20, 9, 0, 0))

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.details["trading_days_elapsed"] == 1

    def test_two_trading_days_old_blocks(
        self,
        equity_db: Callable[..., Path],
        preflight_context: PreflightContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Row Mon, now Wed → Tue + Wed = 2 trading days → BLOCK (>1).
        equity_db("default", "2026-05-18 16:30:00")
        _freeze_market_time(monkeypatch, datetime(2026, 5, 20, 9, 0, 0))

        result = EquityHistoryFreshnessCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert result.details["trading_days_elapsed"] == 2


# Silence the unused-import warning for timedelta if it ever gets pruned.
_ = timedelta

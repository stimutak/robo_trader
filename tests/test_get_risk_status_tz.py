"""Regression tests for the get_risk_status daily-P&L timezone bug.

Per docs/dashboard_risk_tab_backend_followup.md section 4: get_risk_status
filtered "today" trades with a naive .date() comparison,
datetime.fromisoformat(ts).date() == datetime.now().date().

Trade timestamps are stored as naive-UTC strings (SQLite CURRENT_TIMESTAMP),
so a trade after 8 PM ET -- e.g. 21:30 ET, stored as the next UTC calendar
day -- was attributed to the wrong day and dropped from the current ET day's
daily_pnl, skewing daily_loss_pct and the kill-switch threshold.

These tests hit the real /api/risk/status route with a patched trade reader
and a frozen ET clock, so they exercise the production filter (not a copy),
are deterministic regardless of when/where they run, and fail if the fix is
reverted.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

ET = ZoneInfo("America/New_York")
# Frozen "now" = 2026-06-24 21:30 ET (extended hours). In UTC this is
# 2026-06-25 01:30, i.e. the NEXT calendar day -- the exact condition that
# broke the old naive comparison.
FROZEN_ET_NOW = datetime(2026, 6, 24, 21, 30, tzinfo=ET)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return FROZEN_ET_NOW.astimezone(tz) if tz else FROZEN_ET_NOW.replace(tzinfo=None)


@pytest.fixture
def client():
    # config.py requires IBKR_* at import; dummy values, the route is tested
    # with a mocked reader and never connects.
    env = {
        "DASH_AUTH_ENABLED": "false",
        "ADVANCED_RISK_ENABLED": "true",
        "IBKR_HOST": "127.0.0.1",
        "IBKR_PORT": "7497",
        "IBKR_CLIENT_ID": "1",
    }
    with patch.dict(os.environ, env, clear=False):
        from app import app

        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


def _reader(trades):
    reader = MagicMock()
    reader.get_recent_trades.return_value = trades
    reader.get_positions.return_value = []
    return patch("sync_db_reader.SyncDatabaseReader", MagicMock(return_value=reader))


def _daily_pnl(client, trades):
    with _reader(trades), patch("app.datetime", _FrozenDateTime):
        resp = client.get("/api/risk/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("error") is None
    return data["risk_metrics"]["daily_pnl"]


def test_after_8pm_et_trade_counts_today(client):
    # 2026-06-25 01:30 UTC == 2026-06-24 21:30 ET: same ET day as frozen now.
    trade = {"timestamp": "2026-06-25 01:30:00", "pnl": -150.0, "symbol": "AAA"}
    assert _daily_pnl(client, [trade]) == -150.0


def test_prior_et_day_trade_excluded_today(client):
    # 2026-06-23 16:00 UTC == 2026-06-23 12:00 ET: a different ET day.
    trade = {"timestamp": "2026-06-23 16:00:00", "pnl": -150.0, "symbol": "AAA"}
    assert _daily_pnl(client, [trade]) == 0.0

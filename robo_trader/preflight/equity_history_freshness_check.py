"""G2 — EquityHistoryFreshnessCheck.

Verifies that ``trading_data.db`` has a recent ``equity_history`` row.
Equity snapshots are written at end-of-day for every active portfolio.
A row that is more than 1 *trading day* old means the system did not
complete its last expected EOD cycle — usually a sign of an unclean
shutdown that left positions out of sync with IBKR.

Why trading-day delta, not wall-clock delta
-------------------------------------------
A Monday morning startup will legitimately see a Friday row that is
60+ hours old. Counting wall-clock hours would false-positive every
Monday and every holiday Tuesday. :func:`count_trading_days` already
knows the NYSE calendar, so we lean on it.

Decision matrix (spec §7.3)
---------------------------
======================================  ======  =================================
condition                               result  rationale
======================================  ======  =================================
``trading_data.db`` missing             BLOCK   system unconfigured
empty ``equity_history`` table          WARN    first-run; not a livelock
                                                (per Q11.1 design decision)
MAX(timestamp) within 1 trading day     PASS    normal startup
MAX(timestamp) > 1 trading day old      BLOCK   stale; positions may not match
sqlite read error                       BLOCK   fail-closed
======================================  ======  =================================

Multi-portfolio note
--------------------
``equity_history`` is partitioned by ``portfolio_id``. We take the
**max timestamp across all portfolios** — if any portfolio is fresh,
the whole startup passes. This avoids blocking when a disabled
portfolio hasn't traded in weeks.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Optional

from robo_trader.market_hours import count_trading_days
from robo_trader.utils.market_time import get_market_time

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus

# SQLite stores DATETIME values written via ``CURRENT_TIMESTAMP`` as naive
# UTC strings in this format (``YYYY-MM-DD HH:MM:SS[.ffffff]``). We parse
# without assuming a timezone offset, then compare against the date portion
# of ``get_market_time()`` — :func:`count_trading_days` operates on
# ``.date()`` so the cross-timezone fuzziness doesn't affect the count.
_SQLITE_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_sqlite_timestamp(raw: str) -> Optional[datetime]:
    """Parse a sqlite ``CURRENT_TIMESTAMP``-style string. Returns None on failure."""
    for fmt in _SQLITE_TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


class EquityHistoryFreshnessCheck:
    """Confirms the latest equity_history row is no more than 1 trading day old."""

    name = "equity_history_freshness"
    description = "Equity history freshness"
    timeout_seconds = 3.0

    def run(self, context: PreflightContext) -> CheckResult:
        db_path = context.project_root / "trading_data.db"

        if not db_path.exists():
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"trading_data.db not found at {db_path}",
                remediation=(
                    "The trading database is missing. This usually means the system "
                    "has never been started or the data directory was wiped. Run "
                    "./START_TRADER.sh to initialize, then re-run preflight."
                ),
                details={"db_path": str(db_path)},
            )

        try:
            max_ts_raw = self._query_max_timestamp(db_path)
        except sqlite3.Error as exc:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"sqlite read error: {exc}",
                remediation=(
                    "Could not read equity_history from trading_data.db. The DB may "
                    "be locked, corrupted, or schema-mismatched. Try "
                    "`sqlite3 trading_data.db 'PRAGMA integrity_check;'`. If this "
                    "is a known-good transient (e.g. another process is writing), "
                    "wait and re-run; otherwise `--force` after investigating."
                ),
                details={"db_path": str(db_path), "error": str(exc)},
            )

        if max_ts_raw is None:
            # Empty table — first-run case per Q11.1.
            return CheckResult(
                name=self.name,
                status=CheckStatus.WARN,
                message="equity_history is empty (first-run portfolio)",
                remediation=(
                    "No equity snapshots have been written yet. This is normal for "
                    "a brand-new install or a newly-created portfolio. The first "
                    "successful EOD cycle will populate this table. Not blocking; "
                    "verify positions manually if you weren't expecting an empty "
                    "history."
                ),
                details={"db_path": str(db_path), "row_count": 0},
            )

        max_ts = _parse_sqlite_timestamp(max_ts_raw)
        if max_ts is None:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"could not parse equity_history MAX(timestamp)={max_ts_raw!r}",
                remediation=(
                    "The most recent equity_history row has a timestamp that "
                    "doesn't match the expected SQLite CURRENT_TIMESTAMP format. "
                    "The DB schema may be out of date. Run any pending migrations, "
                    "or `--force` if you've confirmed positions match IBKR."
                ),
                details={"db_path": str(db_path), "raw_timestamp": max_ts_raw},
            )

        now = get_market_time()
        trading_days_elapsed = count_trading_days(start=max_ts, end=now)

        details = {
            "db_path": str(db_path),
            "max_timestamp": max_ts_raw,
            "trading_days_elapsed": trading_days_elapsed,
        }

        if trading_days_elapsed <= 1:
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message=(
                    f"latest equity row at {max_ts_raw} "
                    f"({trading_days_elapsed} trading day(s) ago)"
                ),
                details=details,
            )

        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=(
                f"latest equity row at {max_ts_raw} is " f"{trading_days_elapsed} trading days old"
            ),
            remediation=(
                f"Last equity row is {trading_days_elapsed} trading days old. "
                "This usually means a prior session died without writing a "
                "snapshot. Verify positions match IBKR "
                "(`scripts/reconcile_positions.py`) before resuming, or `--force` "
                "if you've already confirmed."
            ),
            details=details,
        )

    @staticmethod
    def _query_max_timestamp(db_path) -> Optional[str]:
        """Return the freshest ``timestamp`` across all portfolios, or None if empty.

        Opens the DB read-only via the ``file:...?mode=ro`` URI so the check
        cannot accidentally mutate state — preflight is observation-only
        (spec §5.7).
        """
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM equity_history")
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

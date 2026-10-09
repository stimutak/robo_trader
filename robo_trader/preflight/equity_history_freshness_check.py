"""G2 — EquityHistoryFreshnessCheck.

Verifies that the configured trading ledger has a recent ``equity_history`` row.
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
configured trading ledger missing      BLOCK   system unconfigured
empty ``equity_history`` table          WARN    first-run; not a livelock
                                                (per Q11.1 design decision)
MAX(timestamp) within 1 trading day     PASS    normal startup
MAX(timestamp) > 1 trading day old      BLOCK   stale; positions may not match
sqlite read error                       BLOCK   fail-closed
======================================  ======  =================================

Multi-portfolio note
--------------------
Every configured active portfolio must have its own fresh snapshot. Disabled
portfolios cannot mask an active portfolio's stale or missing history. Recovery
uses a separately reviewed authenticated bootstrap checkpoint, never a copied
balance or a timestamp rewrite.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from robo_trader.market_hours import count_trading_days
from robo_trader.multiuser.portfolio_config import load_portfolio_configs
from robo_trader.safety.sqlite_identity import (
    SQLiteIdentityError,
    SQLitePathBinding,
    sqlite_connection_file_identity,
)
from robo_trader.utils.market_time import MARKET_TZ, get_market_time

from .protocol import PreflightContext
from .result import CheckResult, CheckStatus

# SQLite stores DATETIME values written via ``CURRENT_TIMESTAMP`` as naive
# UTC strings in this format (``YYYY-MM-DD HH:MM:SS[.ffffff]``). We parse
# as UTC, convert to market time, and only then count market-calendar dates.
_SQLITE_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)


def _parse_sqlite_timestamp(raw: str) -> Optional[datetime]:
    """Parse a sqlite ``CURRENT_TIMESTAMP``-style string. Returns None on failure."""
    if not isinstance(raw, str):
        return None
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
        configured_path = Path(context.env.get("RT_DB_PATH", "trading_data.db"))
        db_path = (
            configured_path
            if configured_path.is_absolute()
            else context.project_root / configured_path
        )

        if not db_path.exists():
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"configured trading database not found at {db_path}",
                remediation=(
                    f"Verify RT_DB_PATH and restore the expected ledger at {db_path} "
                    "from a verified backup if it is missing. Do not create or replace "
                    "the ledger merely to pass preflight; re-run the check after the "
                    "configured ledger is available."
                ),
                details={"db_path": str(db_path)},
            )

        try:
            active_ids = tuple(
                config.id for config in load_portfolio_configs(context.env) if config.active
            )
            if not active_ids:
                raise ValueError("no active portfolios are configured")
            timestamps = self._query_portfolio_timestamps(db_path, active_ids, context)
        except (sqlite3.Error, ValueError, OSError, SQLiteIdentityError) as exc:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"equity freshness inspection blocked for {db_path}: {exc}",
                remediation=(
                    f"Could not read equity_history from {db_path}. The ledger may be "
                    "locked, corrupted, or schema-mismatched. Inspect that exact file "
                    "with read-only SQLite tooling. If this is a known-good transient "
                    "(for example, another process is writing), wait and re-run. "
                    "Keep startup blocked until the configured ledger can be read; "
                    "diagnostic evidence does not authorize startup."
                ),
                details={"db_path": str(db_path), "error": str(exc)},
            )

        results = {
            portfolio_id: self._assess_timestamp(
                db_path, timestamps[portfolio_id][0], portfolio_id, timestamps[portfolio_id][1]
            )
            for portfolio_id in active_ids
        }
        if len(results) == 1:
            return next(iter(results.values()))
        blocked = [pid for pid, result in results.items() if result.status is CheckStatus.BLOCK]
        warnings = [pid for pid, result in results.items() if result.status is CheckStatus.WARN]
        status = (
            CheckStatus.BLOCK if blocked else CheckStatus.WARN if warnings else CheckStatus.PASS
        )
        return CheckResult(
            name=self.name,
            status=status,
            message="Active portfolio equity freshness: "
            + "; ".join(f"{pid}: {result.message}" for pid, result in results.items()),
            remediation="\n".join(
                result.remediation for result in results.values() if result.remediation
            ),
            details={
                "db_path": str(db_path),
                "portfolios": {
                    pid: {"status": result.status.value, **result.details}
                    for pid, result in results.items()
                },
            },
        )

    def _assess_timestamp(
        self, db_path: Path, max_ts_raw: Optional[str], portfolio_id: str, source: str
    ) -> CheckResult:
        if max_ts_raw is None:
            # Empty table — first-run case per Q11.1.
            return CheckResult(
                name=self.name,
                status=CheckStatus.WARN,
                message=f"equity_history is empty in {db_path} (first-run portfolio)",
                remediation=(
                    f"No equity snapshots have been written to {db_path}. This is "
                    "normal for a brand-new install or a newly-created portfolio. "
                    "The first successful EOD cycle will populate this table. Not "
                    "blocking; verify positions manually if you weren't expecting "
                    "an empty history."
                ),
                details={"db_path": str(db_path), "row_count": 0},
            )

        max_ts = _parse_sqlite_timestamp(max_ts_raw)
        if max_ts is None:
            timestamp_repr = repr(max_ts_raw)
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=(
                    f"could not parse equity_history MAX(timestamp)="
                    f"{timestamp_repr} in {db_path}"
                ),
                remediation=(
                    f"The most recent equity_history row in {db_path} has a timestamp "
                    "that doesn't match the expected SQLite CURRENT_TIMESTAMP format. "
                    "Inspect the configured ledger and any pending migrations without "
                    "rewriting history. Keep startup blocked; any broker-ledger "
                    "reconciliation output is evidence only and does not authorize "
                    "startup."
                ),
                details={"db_path": str(db_path), "raw_timestamp": max_ts_raw},
            )

        now = get_market_time()
        if now.tzinfo is None:
            now = now.replace(tzinfo=MARKET_TZ)
        max_ts = max_ts.replace(tzinfo=timezone.utc).astimezone(MARKET_TZ)
        if max_ts > now:
            return CheckResult(
                name=self.name,
                status=CheckStatus.BLOCK,
                message=f"equity timestamp for {portfolio_id} is in the future",
                remediation="Inspect timestamps without rewriting history; keep startup blocked.",
                details={
                    "db_path": str(db_path),
                    "portfolio_id": portfolio_id,
                    "raw_timestamp": max_ts_raw,
                },
            )
        trading_days_elapsed = count_trading_days(start=max_ts, end=now)

        details = {
            "db_path": str(db_path),
            "portfolio_id": portfolio_id,
            "source": source,
            "max_timestamp": max_ts_raw,
            "trading_days_elapsed": trading_days_elapsed,
        }

        if trading_days_elapsed <= 1:
            return CheckResult(
                name=self.name,
                status=CheckStatus.PASS,
                message=(
                    f"latest {source} valuation in {db_path} at {max_ts_raw} "
                    f"({trading_days_elapsed} trading day(s) ago)"
                ),
                details=details,
            )

        return CheckResult(
            name=self.name,
            status=CheckStatus.BLOCK,
            message=(
                f"latest equity row in {db_path} at {max_ts_raw} is "
                f"{trading_days_elapsed} trading days old"
            ),
            remediation=(
                f"The last equity row in {db_path} is {trading_days_elapsed} trading "
                "days old. This usually means a prior session died without writing "
                "a snapshot. Do not copy old balances or rewrite timestamps. "
                "Preview a current authenticated valuation with "
                "scripts/bootstrap_exact_paper_state.py preview --append-equity-checkpoint "
                "only if the ledger has not yet been bootstrapped. Collect evidence with the read-only diagnostic "
                "`python3 scripts/reconcile_broker_ledger.py --portfolio-id "
                "<portfolio_id>`. Review the resulting broker-versus-ledger report; "
                "it does not modify state, clear safety controls, bypass preflight, "
                "or authorize startup."
            ),
            details=details,
        )

    @staticmethod
    def _query_portfolio_timestamps(
        db_path: Path, active_ids: tuple[str, ...], context: PreflightContext
    ) -> dict[str, tuple[Optional[str], str]]:
        """Read each portfolio in one query-only transaction without creation."""
        binding = SQLitePathBinding.open_for_initialization(db_path, create=False)
        connection = None
        try:
            connection = sqlite3.connect(
                db_path.absolute().as_uri() + "?mode=ro", uri=True, timeout=2.0
            )
            bound = binding.bind_sqlite_connection(sqlite_connection_file_identity(connection))
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            timestamps = {}
            total = connection.execute("SELECT COUNT(*) FROM equity_history").fetchone()[0]
            for portfolio_id in active_ids:
                value = connection.execute(
                    "SELECT MAX(timestamp) FROM equity_history WHERE portfolio_id=?",
                    (portfolio_id,),
                ).fetchone()[0]
                source = "equity_history"
                parsed = _parse_sqlite_timestamp(value) if isinstance(value, str) else None
                now = get_market_time()
                if now.tzinfo is None:
                    now = now.replace(tzinfo=MARKET_TZ)
                if parsed is not None:
                    observed = parsed.replace(tzinfo=timezone.utc).astimezone(MARKET_TZ)
                    if observed <= now and count_trading_days(start=observed, end=now) > 1:
                        if connection.execute(
                            "SELECT 1 FROM sqlite_master WHERE type='table' "
                            "AND name='bootstrap_equity_checkpoints'"
                        ).fetchone():
                            from robo_trader.config import load_runtime_contract_from_env
                            from robo_trader.equity_checkpoint import read_bootstrap_checkpoint

                            runtime = load_runtime_contract_from_env(
                                context.env, project_root=context.project_root
                            )
                            checkpoint = read_bootstrap_checkpoint(
                                connection, portfolio_id, runtime, db_path
                            )
                            if checkpoint is not None:
                                value = checkpoint
                                source = "authenticated_bootstrap_checkpoint"
                if value is None and total:
                    # A missing active portfolio cannot borrow another portfolio's history.
                    raise ValueError(
                        f"equity history is missing for active portfolio {portfolio_id}"
                    )
                if value is None:
                    for table, query in (
                        ("account", "SELECT 1 FROM account WHERE portfolio_id=? LIMIT 1"),
                        ("positions", "SELECT 1 FROM positions WHERE portfolio_id=? LIMIT 1"),
                        ("trades", "SELECT 1 FROM trades WHERE portfolio_id=? LIMIT 1"),
                    ):
                        if (
                            connection.execute(
                                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                (table,),
                            ).fetchone()
                            and connection.execute(
                                query,
                                (portfolio_id,),
                            ).fetchone()
                        ):
                            raise ValueError(
                                f"equity history is missing for existing portfolio {portfolio_id}"
                            )
                timestamps[portfolio_id] = (value, source)
            bound.assert_connection_identity(sqlite_connection_file_identity(connection))
            binding.assert_path_identity()
            return timestamps
        finally:
            if connection is not None:
                connection.close()
            binding.close()

#!/usr/bin/env python3
"""Preview or apply one sealed exact paper-simulator accounting epoch.

This command never connects to IBKR and never places orders.  The candidate
must already reference reviewed read-only reconciliation evidence proving zero
IBKR paper exposure and zero open orders.  ``preview`` is strictly read-only;
``apply`` requires a stopped runtime, an exact typed confirmation, and a new
verified SQLite online backup before it performs insert-only bootstrap writes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robo_trader.database_async import AsyncTradingDatabase  # noqa: E402
from robo_trader.financial_state_bootstrap import (  # noqa: E402
    ExactStateBootstrapCandidate,
    ExactStateBootstrapError,
    inspect_legacy_state,
)
from robo_trader.runtime_lifecycle_lock import RuntimeLifecycleLock  # noqa: E402

APPLY_CONFIRMATION = "APPLY_SEALED_EXACT_STATE_BOOTSTRAP"


def _load_candidate(path: Path) -> ExactStateBootstrapCandidate:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ExactStateBootstrapError("candidate must be an absolute regular file")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExactStateBootstrapError("candidate document cannot be read") from exc
    return ExactStateBootstrapCandidate.from_mapping(raw)


def _assert_stopped() -> None:
    lsof = Path("/usr/sbin/lsof")
    pgrep = Path("/usr/bin/pgrep")
    if not lsof.is_file() or not pgrep.is_file():
        raise RuntimeError("cannot prove the trader and Gateway are stopped")
    for pattern in (r"(^|/)runner_async\.py( |$)", r"IB Gateway|ibgateway|tws\.jar"):
        completed = subprocess.run(
            [str(pgrep), "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("trader and IBKR Gateway must remain stopped")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove process state")
    for port in (4001, 4002):
        completed = subprocess.run(
            [str(lsof), "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            raise RuntimeError("IBKR API listener must remain stopped")
        if completed.returncode not in {0, 1}:
            raise RuntimeError("cannot prove IBKR listener state")


def _online_backup(source: Path, target: Path) -> dict[str, object]:
    if not target.is_absolute() or target.exists() or target.is_symlink():
        raise RuntimeError("backup target must be a new absolute path")
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise RuntimeError("backup target parent must be an existing regular directory")
    source_connection = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)
    target_connection: sqlite3.Connection | None = None
    try:
        target_connection = sqlite3.connect(str(target))
        source_connection.backup(target_connection)
        target_connection.commit()
        integrity = target_connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise RuntimeError("bootstrap backup failed SQLite integrity verification")
        rows = {
            table: target_connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("account", "positions", "trades", "equity_history")
        }
        return {"backup_path": str(target), "row_counts": rows}
    except BaseException:
        if target_connection is not None:
            target_connection.close()
            target_connection = None
        # Leave a failed target in place for forensics; never overwrite or
        # silently retry it on the next operator invocation.
        raise
    finally:
        source_connection.close()
        if target_connection is not None:
            target_connection.close()


def preview(candidate: ExactStateBootstrapCandidate, database_path: Path) -> dict[str, object]:
    if Path(candidate.database_path) != database_path:
        raise ExactStateBootstrapError("candidate database path does not match --db-path")
    legacy = inspect_legacy_state(database_path)
    if legacy["snapshot_hash"] != candidate.legacy_snapshot_hash:
        raise ExactStateBootstrapError("legacy ledger differs from the reviewed candidate")
    actual = {
        (row["portfolio_id"], row["symbol"]): int(row["quantity"])
        for row in legacy["position_rows"]
    }
    expected = {
        (candidate.portfolio_id, position.symbol): position.quantity
        for position in candidate.positions
    }
    if actual != expected:
        raise ExactStateBootstrapError("candidate does not cover every nonzero legacy position")
    return {
        "authorizes_startup": False,
        "bootstrap_id": candidate.bootstrap_id,
        "candidate_fingerprint": candidate.fingerprint(),
        "legacy_snapshot_hash": candidate.legacy_snapshot_hash,
        "position_count": len(candidate.positions),
        "schema_version": 1,
        "status": "READY_FOR_OFFLINE_APPLY",
    }


async def _apply(
    candidate: ExactStateBootstrapCandidate,
    database_path: Path,
    operator_reason: str,
) -> dict[str, object]:
    database = AsyncTradingDatabase(database_path)
    try:
        await database.initialize()
        receipt = await database.apply_exact_state_bootstrap(
            candidate,
            operator_reason=operator_reason,
        )
    finally:
        await database.close()
    return {
        "authorizes_startup": False,
        "bootstrap_id": receipt.bootstrap_id,
        "candidate_fingerprint": receipt.candidate_fingerprint,
        "committed_at": receipt.committed_at.isoformat(),
        "database_device": receipt.database_device,
        "database_inode": receipt.database_inode,
        "operator_action_id": receipt.operator_action_id,
        "schema_version": 1,
        "status": "BOOTSTRAPPED_GATE_A_STILL_CLOSED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preview", "apply"):
        child = subparsers.add_parser(command)
        child.add_argument("--candidate", type=Path, required=True)
        child.add_argument("--db-path", type=Path, required=True)
        child.add_argument("--json", action="store_true", required=True)
        if command == "apply":
            child.add_argument("--backup-path", type=Path, required=True)
            child.add_argument("--reason", required=True)
            child.add_argument("--confirm", required=True, metavar=APPLY_CONFIRMATION)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        database_path = args.db_path
        if not database_path.is_absolute():
            raise ExactStateBootstrapError("--db-path must be absolute")
        candidate = _load_candidate(args.candidate)
        report = preview(candidate, database_path)
        if args.command == "apply":
            if args.confirm != APPLY_CONFIRMATION:
                raise ExactStateBootstrapError(f"confirmation must be exactly {APPLY_CONFIRMATION}")
            if len(args.reason.strip()) < 10:
                raise ExactStateBootstrapError("--reason must be a specific sentence")
            lock = RuntimeLifecycleLock()
            if not lock.acquire():
                raise RuntimeError("runtime lifecycle lock is already held")
            try:
                _assert_stopped()
                backup = _online_backup(database_path, args.backup_path)
                report = asyncio.run(_apply(candidate, database_path, args.reason.strip()))
                report["backup"] = backup
            finally:
                lock.release()
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "authorizes_startup": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                    "schema_version": 1,
                    "status": "BLOCKED",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

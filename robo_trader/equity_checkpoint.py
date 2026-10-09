"""Immutable current valuation from an authenticated stopped-system bootstrap.

Legacy daily history is untouched. The checkpoint carries exact economics,
observation time and sealed candidate lineage, and is usable only while that
reviewed legacy state remains unchanged. It never authorizes trading readiness.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from decimal import Decimal, localcontext
from pathlib import Path

import aiosqlite

from robo_trader.financial_state_bootstrap import (
    ExactStateBootstrapCandidate,
    ExactStateBootstrapError,
    _canonical_legacy_rows,
)

_TABLE_SQL = """CREATE TABLE bootstrap_equity_checkpoints (
    bootstrap_id TEXT PRIMARY KEY NOT NULL,
    candidate_fingerprint TEXT NOT NULL,
    valuation_payload_json TEXT NOT NULL,
    FOREIGN KEY(bootstrap_id) REFERENCES paper_state_bootstraps(bootstrap_id)
)"""
_TRIGGER_SQL = {
    f"bootstrap_equity_checkpoints_no_{operation.lower()}": (
        f"CREATE TRIGGER bootstrap_equity_checkpoints_no_{operation.lower()} "
        f"BEFORE {operation} ON bootstrap_equity_checkpoints BEGIN "
        "SELECT RAISE(ABORT, 'bootstrap equity checkpoints are append-only'); END"
    )
    for operation in ("UPDATE", "DELETE")
}


def _normalized(sql: str) -> str:
    return " ".join(sql.split()).casefold()


def checkpoint_values(candidate: ExactStateBootstrapCandidate) -> dict[str, str]:
    with localcontext() as context:
        context.prec = 64
        positions_value = sum(
            (Decimal(position.quantity) * position.mark_price for position in candidate.positions),
            Decimal(0),
        )
        unrealized = sum(
            (
                Decimal(position.quantity) * (position.mark_price - position.cost_basis)
                for position in candidate.positions
            ),
            Decimal(0),
        )
        equity = candidate.account.cash + positions_value
    return {
        "source": "AUTHENTICATED_LOCAL_PAPER_BOOTSTRAP",
        "portfolio_id": candidate.portfolio_id,
        "observed_at": candidate.effective_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
        "cash_text": format(candidate.account.cash, "f"),
        "equity_text": format(equity, "f"),
        "positions_value_text": format(positions_value, "f"),
        "realized_pnl_text": format(candidate.account.realized_pnl, "f"),
        "unrealized_pnl_text": format(unrealized, "f"),
    }


def _payload(candidate: ExactStateBootstrapCandidate) -> str:
    return json.dumps(checkpoint_values(candidate), sort_keys=True, separators=(",", ":"))


async def append_bootstrap_checkpoint(
    connection: aiosqlite.Connection,
    candidate: ExactStateBootstrapCandidate,
) -> None:
    """Caller has validated fresh evidence and owns the bootstrap transaction."""
    existing = await connection.execute(
        "SELECT sql FROM main.sqlite_master WHERE name='bootstrap_equity_checkpoints'"
    )
    row = await existing.fetchone()
    if row is None:
        await connection.execute(_TABLE_SQL)
        for sql in _TRIGGER_SQL.values():
            await connection.execute(sql)
    elif _normalized(row[0]) != _normalized(_TABLE_SQL):
        raise ExactStateBootstrapError("equity checkpoint provenance schema is malformed")
    triggers = await connection.execute(
        "SELECT name,sql FROM main.sqlite_master WHERE type='trigger' "
        "AND tbl_name='bootstrap_equity_checkpoints'"
    )
    actual = {name: _normalized(sql) for name, sql in await triggers.fetchall()}
    if actual != {name: _normalized(sql) for name, sql in _TRIGGER_SQL.items()}:
        raise ExactStateBootstrapError("equity checkpoint provenance triggers are malformed")
    values = (candidate.bootstrap_id, candidate.fingerprint(), _payload(candidate))
    await connection.execute("INSERT INTO main.bootstrap_equity_checkpoints VALUES (?,?,?)", values)
    written = await connection.execute(
        "SELECT * FROM main.bootstrap_equity_checkpoints WHERE bootstrap_id=?",
        (candidate.bootstrap_id,),
    )
    if tuple(await written.fetchone() or ()) != values:
        raise ExactStateBootstrapError(
            "equity checkpoint insertion did not persist exact provenance"
        )


def read_bootstrap_checkpoint(
    connection: sqlite3.Connection,
    portfolio_id: str,
    runtime_contract: object,
    db_path: Path,
) -> str | None:
    """Revalidate lineage and unchanged ledger in the caller's query-only snapshot."""
    table = connection.execute(
        "SELECT sql FROM main.sqlite_master WHERE name='bootstrap_equity_checkpoints'"
    ).fetchone()
    if table is None:
        return None
    if _normalized(table[0]) != _normalized(_TABLE_SQL):
        raise ValueError("equity checkpoint schema is malformed")
    triggers = dict(
        connection.execute(
            "SELECT name,sql FROM main.sqlite_master WHERE type='trigger' "
            "AND tbl_name='bootstrap_equity_checkpoints'"
        )
    )
    if {name: _normalized(sql) for name, sql in triggers.items()} != {
        name: _normalized(sql) for name, sql in _TRIGGER_SQL.items()
    }:
        raise ValueError("equity checkpoint triggers are malformed")
    rows = connection.execute(
        "SELECT c.bootstrap_id,c.candidate_fingerprint,c.valuation_payload_json,"
        "b.candidate_payload_json,b.candidate_fingerprint,b.database_device,b.database_inode "
        "FROM main.bootstrap_equity_checkpoints c JOIN main.paper_state_bootstraps b "
        "ON b.bootstrap_id=c.bootstrap_id WHERE b.portfolio_id=?",
        (portfolio_id,),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("ambiguous equity checkpoint lineage")
    row = rows[0]
    candidate = ExactStateBootstrapCandidate.from_mapping(json.loads(row[3]))
    metadata = db_path.stat()
    if (
        candidate.bootstrap_id != row[0]
        or candidate.portfolio_id != portfolio_id
        or candidate.fingerprint() != row[1]
        or candidate.fingerprint() != row[4]
        or candidate.database_path != str(db_path)
        or candidate.database_identity != runtime_contract.database_identity
        or candidate.account_scope != runtime_contract.safety_account_scope
        or candidate.execution_domain_scope != runtime_contract.safety_execution_domain_scope
        or (row[5], row[6]) != (metadata.st_dev, metadata.st_ino)
        or row[2] != _payload(candidate)
    ):
        raise ValueError("equity checkpoint does not match the sealed runtime and candidate")
    queries = (
        "SELECT portfolio_id,cash,equity,daily_pnl,realized_pnl,unrealized_pnl,timestamp FROM main.account ORDER BY portfolio_id",
        "SELECT portfolio_id,symbol,quantity,avg_cost,market_price,timestamp FROM main.positions WHERE quantity<>0 ORDER BY portfolio_id,symbol",
        "SELECT id,portfolio_id,symbol,side,quantity,price,notional,slippage,commission,pnl,timestamp FROM main.trades ORDER BY id",
        "SELECT id,portfolio_id,date,equity,cash,positions_value,realized_pnl,unrealized_pnl,timestamp FROM main.equity_history ORDER BY id",
    )
    legacy = _canonical_legacy_rows(*(connection.execute(query).fetchall() for query in queries))
    if hashlib.sha256(legacy.encode()).hexdigest() != candidate.legacy_snapshot_hash:
        raise ValueError(
            "ledger changed since authenticated equity checkpoint; collect new evidence"
        )
    return checkpoint_values(candidate)["observed_at"]

"""Sealed PR4B bridge from reviewed legacy state into an exact FIFO epoch.

The bridge records prospective opening balances.  It deliberately does not
invent historical fills, executions, or commissions.  A zero opening
commission means that pre-epoch fee history is outside this epoch, not that the
historical broker commission was zero.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import aiosqlite

from .fifo_fixture_migration import (
    _TABLE_SQL,
    _TRIGGER_SQL,
    FIFO_ACCOUNTING_COMPONENT,
    FIFO_ACCOUNTING_MIGRATIONS,
    _normalize_sql,
)


class FifoBootstrapError(RuntimeError):
    """A reviewed legacy candidate cannot be appended as a FIFO epoch."""


def _derived_id(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


@dataclass(frozen=True, slots=True)
class LegacyOpeningBalance:
    opening_balance_id: str
    lot_id: str
    con_id: int
    symbol: str
    direction: str
    opened_quantity_text: str
    cost_basis_text: str
    mark_price_text: str
    mark_observed_at: str
    mark_evidence_fingerprint: str

    def public_dict(self) -> dict[str, object]:
        return {
            "con_id": self.con_id,
            "cost_basis_text": self.cost_basis_text,
            "direction": self.direction,
            "lot_id": self.lot_id,
            "mark_evidence_fingerprint": self.mark_evidence_fingerprint,
            "mark_observed_at": self.mark_observed_at,
            "mark_price_text": self.mark_price_text,
            "opened_quantity_text": self.opened_quantity_text,
            "opening_balance_id": self.opening_balance_id,
            "opening_commission_minor": 0,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class LegacyFifoBootstrapPlan:
    epoch_id: str
    bootstrap_id: str
    candidate_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    effective_at: str
    reconciliation_snapshot_id: str
    reconciliation_report_hash: str
    broker_snapshot_hash: str
    legacy_snapshot_hash: str
    cash_text: str
    realized_pnl_text: str
    daily_pnl_text: str
    daily_pnl_baseline_text: str
    daily_pnl_date: str
    opening_balances: tuple[LegacyOpeningBalance, ...]

    def public_dict(self) -> dict[str, object]:
        return {
            "account_baseline": {
                "cash_text": self.cash_text,
                "daily_pnl_baseline_text": self.daily_pnl_baseline_text,
                "daily_pnl_date": self.daily_pnl_date,
                "daily_pnl_text": self.daily_pnl_text,
                "realized_pnl_text": self.realized_pnl_text,
            },
            "bootstrap_id": self.bootstrap_id,
            "candidate_fingerprint": self.candidate_fingerprint,
            "epoch_id": self.epoch_id,
            "authorizes_startup": False,
            "opening_balances": [value.public_dict() for value in self.opening_balances],
            "opening_commission_semantics": "UNKNOWN_PRE_EPOCH_HISTORY_NOT_RECONSTRUCTED",
            "origin_kind": "LEGACY_AGGREGATE_OPENING_BALANCE",
            "pre_epoch_history_reconstructed": False,
            "synthetic_fill_count": 0,
        }


def build_legacy_fifo_bootstrap_plan(
    *,
    bootstrap_id: str,
    candidate_fingerprint: str,
    execution_domain_scope: str,
    account_scope: str,
    portfolio_id: str,
    effective_at: str,
    reconciliation_snapshot_id: str,
    reconciliation_report_hash: str,
    broker_snapshot_hash: str,
    legacy_snapshot_hash: str,
    cash_text: str,
    realized_pnl_text: str,
    daily_pnl_text: str,
    daily_pnl_baseline_text: str,
    daily_pnl_date: str,
    positions: Sequence[dict[str, object]],
) -> LegacyFifoBootstrapPlan:
    """Derive deterministic epoch/opening identities from one sealed candidate."""

    epoch_id = _derived_id("fepoch", "legacy-bootstrap", candidate_fingerprint)
    openings: list[LegacyOpeningBalance] = []
    seen_contracts: set[int] = set()
    seen_symbols: set[str] = set()
    for position in positions:
        con_id = position.get("con_id")
        symbol = position.get("symbol")
        quantity = position.get("quantity")
        if type(con_id) is not int or con_id <= 0:
            raise FifoBootstrapError("every legacy opening balance requires a positive con_id")
        if type(symbol) is not str or not symbol or type(quantity) is not int or quantity == 0:
            raise FifoBootstrapError("legacy opening balance identity is malformed")
        if con_id in seen_contracts or symbol in seen_symbols:
            raise FifoBootstrapError("legacy opening balance identities are not unique")
        seen_contracts.add(con_id)
        seen_symbols.add(symbol)
        opening_balance_id = _derived_id("fobal", epoch_id, con_id, symbol, candidate_fingerprint)
        openings.append(
            LegacyOpeningBalance(
                opening_balance_id=opening_balance_id,
                lot_id=_derived_id("flot", epoch_id, opening_balance_id),
                con_id=con_id,
                symbol=symbol,
                direction="LONG" if quantity > 0 else "SHORT",
                opened_quantity_text=str(abs(quantity)),
                cost_basis_text=str(position["cost_basis_text"]),
                mark_price_text=str(position["mark_price_text"]),
                mark_observed_at=str(position["mark_observed_at"]),
                mark_evidence_fingerprint=str(position["mark_evidence_fingerprint"]),
            )
        )
    if [opening.symbol for opening in openings] != sorted(seen_symbols):
        raise FifoBootstrapError("legacy opening balances must be sorted by symbol")
    return LegacyFifoBootstrapPlan(
        epoch_id=epoch_id,
        bootstrap_id=bootstrap_id,
        candidate_fingerprint=candidate_fingerprint,
        execution_domain_scope=execution_domain_scope,
        account_scope=account_scope,
        portfolio_id=portfolio_id,
        effective_at=effective_at,
        reconciliation_snapshot_id=reconciliation_snapshot_id,
        reconciliation_report_hash=reconciliation_report_hash,
        broker_snapshot_hash=broker_snapshot_hash,
        legacy_snapshot_hash=legacy_snapshot_hash,
        cash_text=cash_text,
        realized_pnl_text=realized_pnl_text,
        daily_pnl_text=daily_pnl_text,
        daily_pnl_baseline_text=daily_pnl_baseline_text,
        daily_pnl_date=daily_pnl_date,
        opening_balances=tuple(openings),
    )


async def _assert_fifo_schema(connection: aiosqlite.Connection) -> None:
    foreign_keys = await connection.execute("PRAGMA foreign_keys")
    if await foreign_keys.fetchone() != (1,):
        raise FifoBootstrapError("FIFO accounting requires foreign-key enforcement")
    temporary = await connection.execute(
        "SELECT type,name FROM temp.sqlite_master WHERE name LIKE 'fifo_%' ORDER BY type,name"
    )
    if await temporary.fetchone() is not None:
        raise FifoBootstrapError("temporary FIFO objects cannot shadow durable state")
    tables = await connection.execute(
        "SELECT name,sql FROM main.sqlite_master "
        "WHERE type='table' AND name LIKE 'fifo_%' ORDER BY name"
    )
    actual_tables = {str(name): _normalize_sql(sql) for name, sql in await tables.fetchall()}
    if set(actual_tables) != set(_TABLE_SQL):
        raise FifoBootstrapError("FIFO accounting table set is incomplete or unknown")
    for name, statement in _TABLE_SQL.items():
        if actual_tables[name] != _normalize_sql(statement):
            raise FifoBootstrapError(f"FIFO accounting table {name} is malformed")
    triggers = await connection.execute(
        "SELECT name,sql FROM main.sqlite_master "
        "WHERE type='trigger' AND name LIKE 'fifo_%' ORDER BY name"
    )
    actual_triggers = {str(name): _normalize_sql(sql) for name, sql in await triggers.fetchall()}
    if set(actual_triggers) != set(_TRIGGER_SQL):
        raise FifoBootstrapError("FIFO accounting trigger set is incomplete or unknown")
    for name, statement in _TRIGGER_SQL.items():
        if actual_triggers[name] != _normalize_sql(statement):
            raise FifoBootstrapError(f"FIFO accounting trigger {name} is malformed")
    versions = await connection.execute(
        "SELECT version,description FROM fifo_schema_migrations "
        "WHERE component=? ORDER BY version",
        (FIFO_ACCOUNTING_COMPONENT,),
    )
    if await versions.fetchall() != list(FIFO_ACCOUNTING_MIGRATIONS):
        raise FifoBootstrapError("FIFO accounting migration evidence is incomplete")
    foreign_key_check = await connection.execute("PRAGMA main.foreign_key_check")
    if await foreign_key_check.fetchone() is not None:
        raise FifoBootstrapError("FIFO accounting schema has foreign-key violations")


async def prepare_fifo_accounting_schema_in_transaction(
    connection: aiosqlite.Connection,
    *,
    applied_at: str,
) -> None:
    """Create the FIFO schema only inside the caller's already-held transaction."""

    if not connection.in_transaction:
        raise FifoBootstrapError("FIFO schema preparation requires an active transaction")
    objects = await connection.execute(
        "SELECT type,name FROM main.sqlite_master " "WHERE name LIKE 'fifo_%' ORDER BY type,name"
    )
    existing = await objects.fetchall()
    if existing:
        await _assert_fifo_schema(connection)
        return
    for statement in _TABLE_SQL.values():
        await connection.execute(statement)
    await connection.executemany(
        """
        INSERT INTO fifo_schema_migrations(component,version,description,applied_at)
        VALUES (?,?,?,?)
        """,
        tuple(
            (FIFO_ACCOUNTING_COMPONENT, version, description, applied_at)
            for version, description in FIFO_ACCOUNTING_MIGRATIONS
        ),
    )
    for statement in _TRIGGER_SQL.values():
        await connection.execute(statement)
    await _assert_fifo_schema(connection)


async def append_legacy_fifo_bootstrap_in_transaction(
    connection: aiosqlite.Connection,
    *,
    plan: LegacyFifoBootstrapPlan,
    operator_action_id: str,
    recorded_at: str,
) -> None:
    """Append a legacy epoch and its explicit non-fill opening records."""

    if not connection.in_transaction:
        raise FifoBootstrapError("legacy FIFO bootstrap requires an active transaction")
    if type(plan) is not LegacyFifoBootstrapPlan:
        raise FifoBootstrapError("legacy FIFO bootstrap plan has the wrong type")
    if type(operator_action_id) is not str or not operator_action_id:
        raise FifoBootstrapError("legacy FIFO bootstrap requires an administrator action")
    await _assert_fifo_schema(connection)
    existing = await connection.execute(
        """
        SELECT epoch_id FROM fifo_accounting_epochs
        WHERE epoch_id=? OR source_fingerprint=? OR (
            execution_domain_scope=? AND account_scope=? AND portfolio_id=?
        )
        """,
        (
            plan.epoch_id,
            plan.candidate_fingerprint,
            plan.execution_domain_scope,
            plan.account_scope,
            plan.portfolio_id,
        ),
    )
    if await existing.fetchone() is not None:
        raise FifoBootstrapError("FIFO accounting scope already has a sealed epoch")
    await connection.execute(
        """
        INSERT INTO fifo_accounting_epochs(
            epoch_id,schema_version,execution_domain_scope,account_scope,portfolio_id,
            origin_kind,source_fingerprint,effective_at,created_at
        ) VALUES (?,?,?,?,?,'LEGACY_AGGREGATE_OPENING_BALANCE',?,?,?)
        """,
        (
            plan.epoch_id,
            1,
            plan.execution_domain_scope,
            plan.account_scope,
            plan.portfolio_id,
            plan.candidate_fingerprint,
            plan.effective_at,
            recorded_at,
        ),
    )
    await connection.execute(
        """
        INSERT INTO fifo_legacy_bootstrap_lineage(
            epoch_id,bootstrap_id,candidate_fingerprint,reconciliation_snapshot_id,
            reconciliation_report_hash,broker_snapshot_hash,legacy_snapshot_hash,
            operator_action_id,recorded_at
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            plan.epoch_id,
            plan.bootstrap_id,
            plan.candidate_fingerprint,
            plan.reconciliation_snapshot_id,
            plan.reconciliation_report_hash,
            plan.broker_snapshot_hash,
            plan.legacy_snapshot_hash,
            operator_action_id,
            recorded_at,
        ),
    )
    await connection.execute(
        """
        INSERT INTO fifo_epoch_account_baselines(
            epoch_id,cash_text,realized_pnl_text,daily_pnl_text,
            daily_pnl_baseline_text,daily_pnl_date,recorded_at
        ) VALUES (?,?,?,?,?,?,?)
        """,
        (
            plan.epoch_id,
            plan.cash_text,
            plan.realized_pnl_text,
            plan.daily_pnl_text,
            plan.daily_pnl_baseline_text,
            plan.daily_pnl_date,
            recorded_at,
        ),
    )
    for opening in plan.opening_balances:
        await connection.execute(
            """
            INSERT INTO fifo_opening_balances(
                opening_balance_id,epoch_id,con_id,symbol,direction,
                opened_quantity_text,cost_basis_text,mark_price_text,
                mark_observed_at,mark_evidence_fingerprint,recorded_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                opening.opening_balance_id,
                plan.epoch_id,
                opening.con_id,
                opening.symbol,
                opening.direction,
                opening.opened_quantity_text,
                opening.cost_basis_text,
                opening.mark_price_text,
                opening.mark_observed_at,
                opening.mark_evidence_fingerprint,
                recorded_at,
            ),
        )
        await connection.execute(
            """
            INSERT INTO fifo_lot_openings(
                lot_id,epoch_id,opening_fill_id,opening_balance_id,lot_ordinal,
                con_id,symbol,direction,opened_quantity_text,open_price_text,
                opening_commission_minor,opened_sequence,opened_at
            ) VALUES (?,?,NULL,?,0,?,?,?,?,?,0,0,?)
            """,
            (
                opening.lot_id,
                plan.epoch_id,
                opening.opening_balance_id,
                opening.con_id,
                opening.symbol,
                opening.direction,
                opening.opened_quantity_text,
                opening.cost_basis_text,
                plan.effective_at,
            ),
        )
    counts = await connection.execute(
        """
        SELECT
          (SELECT COUNT(*) FROM fifo_fills WHERE epoch_id=?),
          (SELECT COUNT(*) FROM fifo_commissions WHERE epoch_id=?),
          (SELECT COUNT(*) FROM fifo_opening_balances WHERE epoch_id=?),
          (SELECT COUNT(*) FROM fifo_lot_openings WHERE epoch_id=? AND opening_fill_id IS NOT NULL)
        """,
        (plan.epoch_id, plan.epoch_id, plan.epoch_id, plan.epoch_id),
    )
    if await counts.fetchone() != (0, 0, len(plan.opening_balances), 0):
        raise FifoBootstrapError("legacy FIFO bootstrap fabricated or omitted event records")
    lineage = await connection.execute(
        """
        SELECT e.origin_kind,e.source_fingerprint,l.bootstrap_id,
               l.candidate_fingerprint,l.reconciliation_snapshot_id,
               l.reconciliation_report_hash,l.broker_snapshot_hash,
               l.legacy_snapshot_hash,l.operator_action_id,
               p.candidate_fingerprint,p.operator_action_id,a.evidence_hash
        FROM fifo_accounting_epochs e
        JOIN fifo_legacy_bootstrap_lineage l ON l.epoch_id=e.epoch_id
        JOIN paper_state_bootstraps p ON p.bootstrap_id=l.bootstrap_id
        JOIN administrator_actions a ON a.action_id=l.operator_action_id
        WHERE e.epoch_id=?
        """,
        (plan.epoch_id,),
    )
    if await lineage.fetchone() != (
        "LEGACY_AGGREGATE_OPENING_BALANCE",
        plan.candidate_fingerprint,
        plan.bootstrap_id,
        plan.candidate_fingerprint,
        plan.reconciliation_snapshot_id,
        plan.reconciliation_report_hash,
        plan.broker_snapshot_hash,
        plan.legacy_snapshot_hash,
        operator_action_id,
        plan.candidate_fingerprint,
        operator_action_id,
        plan.candidate_fingerprint,
    ):
        raise FifoBootstrapError("FIFO epoch is not bound to the exact bootstrap lineage")
    baseline = await connection.execute(
        """
        SELECT cash_text,realized_pnl_text,daily_pnl_text,
               daily_pnl_baseline_text,daily_pnl_date
        FROM fifo_epoch_account_baselines WHERE epoch_id=?
        """,
        (plan.epoch_id,),
    )
    if await baseline.fetchone() != (
        plan.cash_text,
        plan.realized_pnl_text,
        plan.daily_pnl_text,
        plan.daily_pnl_baseline_text,
        plan.daily_pnl_date,
    ):
        raise FifoBootstrapError("FIFO pre-epoch account baseline changed during append")
    openings = await connection.execute(
        """
        SELECT b.opening_balance_id,l.lot_id,b.con_id,b.symbol,b.direction,
               b.opened_quantity_text,b.cost_basis_text,b.mark_price_text,
               b.mark_observed_at,b.mark_evidence_fingerprint,
               l.opening_commission_minor,l.opened_sequence
        FROM fifo_opening_balances b
        JOIN fifo_lot_openings l
          ON l.epoch_id=b.epoch_id AND l.opening_balance_id=b.opening_balance_id
        WHERE b.epoch_id=? ORDER BY b.symbol
        """,
        (plan.epoch_id,),
    )
    expected_openings = [
        (
            opening.opening_balance_id,
            opening.lot_id,
            opening.con_id,
            opening.symbol,
            opening.direction,
            opening.opened_quantity_text,
            opening.cost_basis_text,
            opening.mark_price_text,
            opening.mark_observed_at,
            opening.mark_evidence_fingerprint,
            0,
            0,
        )
        for opening in plan.opening_balances
    ]
    if await openings.fetchall() != expected_openings:
        raise FifoBootstrapError("FIFO opening balances changed during append")

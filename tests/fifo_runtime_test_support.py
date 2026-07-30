"""Synthetic, non-authoritative FIFO epochs for isolated runtime tests only."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from robo_trader.accounting.fifo_bootstrap import prepare_fifo_accounting_schema_in_transaction
from robo_trader.accounting.fifo_fixture_migration import _legacy_opening_manifest_hash
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.safety.models import utc_to_text


def _identifier(prefix: str, *parts: object) -> str:
    material = "\x1f".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


async def install_synthetic_fifo_epoch(
    database: AsyncTradingDatabase,
    *,
    execution_domain_scope: str,
    account_scope: str,
    portfolio_id: str,
    con_id: int,
    symbol: str,
) -> str:
    """Install one explicit opening balance in a pytest-isolated database.

    This helper is intentionally under ``tests/``.  It never reads a user
    database and cannot be imported by the runtime package.
    """

    effective_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    effective_text = utc_to_text(effective_at)
    source_fingerprint = hashlib.sha256(
        f"synthetic-test-epoch:{execution_domain_scope}:{account_scope}:{portfolio_id}".encode()
    ).hexdigest()
    epoch_id = _identifier("fepoch", source_fingerprint)
    async with database.get_connection() as connection:
        await connection.execute("BEGIN IMMEDIATE")
        try:
            await prepare_fifo_accounting_schema_in_transaction(
                connection,
                applied_at=effective_text,
            )
            account = await (
                await connection.execute(
                    """
                    SELECT cash_text,realized_pnl_text,daily_pnl_text,
                           daily_pnl_baseline_text,daily_pnl_date
                    FROM paper_account_settlement_state WHERE portfolio_id=?
                    """,
                    (portfolio_id,),
                )
            ).fetchone()
            position = await (
                await connection.execute(
                    """
                    SELECT quantity,cost_basis_text,mark_price_text
                    FROM positions JOIN paper_position_settlement_state USING(portfolio_id,symbol)
                    WHERE portfolio_id=? AND symbol=?
                    """,
                    (portfolio_id, symbol),
                )
            ).fetchone()
            if account is None:
                raise AssertionError("synthetic FIFO epoch requires seeded exact account state")
            empty_epoch = position is None or position[0] == 0
            await connection.execute(
                """
                INSERT INTO fifo_accounting_epochs(
                    epoch_id,schema_version,execution_domain_scope,account_scope,
                    portfolio_id,origin_kind,source_fingerprint,effective_at,created_at
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    epoch_id,
                    1,
                    execution_domain_scope,
                    account_scope,
                    portfolio_id,
                    "EMPTY_LEDGER" if empty_epoch else "LEGACY_AGGREGATE_OPENING_BALANCE",
                    source_fingerprint,
                    effective_text,
                    effective_text,
                ),
            )
            await connection.execute(
                """
                INSERT INTO fifo_epoch_account_baselines(
                    epoch_id,cash_text,realized_pnl_text,daily_pnl_text,
                    daily_pnl_baseline_text,daily_pnl_date,recorded_at
                ) VALUES (?,?,?,?,?,?,?)
                """,
                (epoch_id, *account, effective_text),
            )
            if empty_epoch:
                await connection.commit()
                return epoch_id
            quantity = int(position[0])
            opening_balance_id = _identifier("fobal", epoch_id, con_id, symbol)
            lot_id = _identifier("flot", epoch_id, opening_balance_id)
            direction = "LONG" if quantity > 0 else "SHORT"
            manifest_row = (
                opening_balance_id,
                lot_id,
                con_id,
                symbol,
                direction,
                str(abs(quantity)),
                str(position[1]),
                str(position[2]),
                effective_text,
                "7" * 64,
                0,
                0,
                effective_text,
            )
            await connection.execute(
                """
                INSERT INTO fifo_legacy_bootstrap_lineage(
                    epoch_id,bootstrap_id,candidate_fingerprint,
                    opening_manifest_count,opening_manifest_hash,
                    reconciliation_snapshot_id,reconciliation_report_hash,
                    broker_snapshot_hash,legacy_snapshot_hash,operator_action_id,recorded_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    epoch_id,
                    _identifier("pboot", epoch_id),
                    source_fingerprint,
                    1,
                    _legacy_opening_manifest_hash([manifest_row]),
                    "synthetic-reconciliation",
                    "4" * 64,
                    "5" * 64,
                    "6" * 64,
                    _identifier("padmin", epoch_id),
                    effective_text,
                ),
            )
            await connection.execute(
                """
                INSERT INTO fifo_opening_balances(
                    opening_balance_id,epoch_id,con_id,symbol,direction,
                    opened_quantity_text,cost_basis_text,mark_price_text,
                    mark_observed_at,mark_evidence_fingerprint,recorded_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    opening_balance_id,
                    epoch_id,
                    con_id,
                    symbol,
                    direction,
                    str(abs(quantity)),
                    str(position[1]),
                    str(position[2]),
                    effective_text,
                    "7" * 64,
                    effective_text,
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
                    lot_id,
                    epoch_id,
                    opening_balance_id,
                    con_id,
                    symbol,
                    direction,
                    str(abs(quantity)),
                    str(position[1]),
                    effective_text,
                ),
            )
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise
    return epoch_id

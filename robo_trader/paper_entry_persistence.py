"""Entry transaction engine; deliberately not a runtime submission/writer API.

The caller must already own a write transaction and, before using this in the
runtime, authenticate a consumed entry execution dispatch. This module stages
storage only: it never commits, issues a receipt, or releases journal capacity.
Its result is plain data and is invalid as execution or release authority.
"""

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from .accounting.fifo import FillSide
from .accounting.fifo_runtime import (
    RuntimePaperFillEvidence,
    append_runtime_fill_on_aiosqlite_worker,
)
from .database_migrations import assert_paper_settlement_hot_schema
from .paper_entry_settlement import (
    PaperEntryTerminalRecord,
    validate_stored_paper_entry_terminal_record,
)
from .paper_terminal_settlement import _exact_decimal_multiply
from .safety.models import (
    ValidationError,
    _exact_decimal_add,
    _exact_decimal_subtract,
    canonical_json,
    parse_fixed_decimal,
    parse_utc_text,
    sha256_text,
    utc_to_text,
)
from .safety.sqlite_identity import SQLitePathBinding


@dataclass(frozen=True)
class StagedEntrySettlement:
    """Uncommitted storage identities, never a producer-owned receipt."""

    settlement_id: str
    trade_id: int | None
    fingerprint: str


async def _one(connection, sql, values=()):
    return await (await connection.execute(sql, values)).fetchone()


async def stage_entry_settlement(
    connection,
    record,
    *,
    database,
    runtime_contract,
    journal,
    committed_at,
):
    """Stage one entry under a caller transaction; rollback this stage on error.

    Exact retries authenticate existing terminal/trade/FIFO storage and return
    its original identities. The authority-checking public adapter is separate.
    """
    from .database_async import AsyncTradingDatabase

    if type(database) is not AsyncTradingDatabase or not connection.in_transaction:
        raise ValidationError("entry storage requires an owned database transaction")
    if type(record) is not PaperEntryTerminalRecord:
        raise ValidationError("entry storage requires an exact terminal record")
    validate_stored_paper_entry_terminal_record(
        record.payload_json,
        fingerprint=record.fingerprint,
        journal=journal,
    )
    data = json.loads(record.payload_json)
    claim, reservation = data["claim"], data["reservation"]
    capacity = json.loads(reservation["payload_json"])
    outcome, pre, post = data["outcome"], data["pre_account"], data["post_values"]
    portfolio, symbol = claim["portfolio_id"], capacity["symbol"]
    path, identity = database._expected_safety_database(runtime_contract=runtime_contract)
    if (
        runtime_contract.execution_mode != "paper"
        or runtime_contract.state_namespace != "paper"
        or runtime_contract.ibkr_readonly is not True
        or runtime_contract.safety_account_scope != claim["account_scope"]
        or runtime_contract.safety_execution_domain_scope != claim["execution_domain_scope"]
    ):
        raise ValidationError("entry storage scope differs from validated paper runtime")
    if (
        type(committed_at) is not datetime
        or committed_at.tzinfo is None
        or committed_at.utcoffset() != timezone.utc.utcoffset(committed_at)
        or committed_at > datetime.now(timezone.utc)
        or parse_utc_text(outcome["observed_at"]) > committed_at
    ):
        raise ValidationError("entry storage time is invalid")
    stamp = utc_to_text(committed_at)
    descriptor = await database._sqlite_descriptor_identity(connection)
    if (descriptor.device, descriptor.inode) != database._expected_database_file_identity:
        raise ValidationError("entry storage database identity changed")
    await assert_paper_settlement_hot_schema(connection)
    # A savepoint makes a caught failure safe even if the caller subsequently
    # commits its outer transaction. No statement below escapes this boundary.
    binding = SQLitePathBinding.open_readonly(path)
    try:
        binding = binding.bind_sqlite_connection(descriptor)
        await connection.execute("SAVEPOINT entry_terminal_stage")
    except BaseException:
        binding.close()
        raise
    try:
        collisions = await _one(
            connection,
            """SELECT 1 FROM main.paper_reduction_settlements
            WHERE reservation_id=? OR claim_id=? OR order_ref=? OR request_fingerprint=? LIMIT 1""",
            (reservation["event_id"], claim["claim_id"], outcome["order_ref"], record.fingerprint),
        )
        if collisions:
            from .paper_entry_storage_replay import read_entry_settlement

            persisted = await read_entry_settlement(
                connection,
                record,
                database=database,
                runtime_contract=runtime_contract,
                journal=journal,
            )
            binding.assert_connection_identity(
                await database._sqlite_descriptor_identity(connection)
            )
            await connection.execute("RELEASE SAVEPOINT entry_terminal_stage")
            return StagedEntrySettlement(
                persisted.settlement_id,
                persisted.trade_id,
                persisted.fingerprint,
            )
        if await _one(connection, "SELECT 1 FROM main.portfolios WHERE id=?", (portfolio,)) is None:
            raise ValidationError("entry portfolio is unavailable")
        rows = await (
            await connection.execute(
                "SELECT quantity,typeof(quantity) FROM main.positions WHERE symbol=?", (symbol,)
            )
        ).fetchall()
        if any(type(q) is not int or kind != "integer" or q != 0 for q, kind in rows):
            raise ValidationError("entry requires zero gross held symbol quantity")
        account = await _one(
            connection,
            """SELECT cash_text,realized_pnl_text,daily_pnl_text,
            daily_pnl_baseline_text,daily_pnl_date FROM main.paper_account_settlement_state
            WHERE portfolio_id=?""",
            (portfolio,),
        )
        if account != tuple(
            pre[k]
            for k in ("cash", "realized_pnl", "daily_pnl", "daily_pnl_baseline", "daily_pnl_date")
        ):
            raise ValidationError("entry account pre-state changed")
        legacy = await _one(
            connection,
            "SELECT cash,realized_pnl,daily_pnl,equity FROM main.account WHERE portfolio_id=?",
            (portfolio,),
        )
        if legacy is None or legacy[:3] != tuple(
            float(Decimal(pre[k])) for k in ("cash", "realized_pnl", "daily_pnl")
        ):
            raise ValidationError("entry account compatibility projection changed")
        position = await _one(
            connection,
            """SELECT cost_basis_text,mark_price_text,source_settlement_id
            FROM main.paper_position_settlement_state WHERE portfolio_id=? AND symbol=?""",
            (portfolio, symbol),
        )
        expected_position = tuple(
            pre[k]
            for k in ("position_cost_basis", "position_mark_price", "position_source_settlement_id")
        )
        if (position if position is not None else (None, None, None)) != expected_position:
            raise ValidationError("entry position metadata changed")
        legacy_position = await _one(
            connection,
            "SELECT avg_cost,market_price FROM main.positions WHERE portfolio_id=? AND symbol=?",
            (portfolio, symbol),
        )
        if (legacy_position is None) != (position is None) or (
            position is not None
            and legacy_position != (float(Decimal(position[0])), float(Decimal(position[1])))
        ):
            raise ValidationError("entry position compatibility projection changed")
        database._paper_settlement_fault("ENTRY_AFTER_PRESTATE")
        quantity = parse_fixed_decimal(outcome["filled_quantity"])
        price = (
            None
            if outcome["exact_fill_price"] is None
            else parse_fixed_decimal(outcome["exact_fill_price"])
        )
        trade_id, fifo = None, None
        fill = outcome["fill_evidence"]
        if quantity:
            fifo = await append_runtime_fill_on_aiosqlite_worker(
                connection,
                RuntimePaperFillEvidence(
                    claim["execution_domain_scope"],
                    claim["account_scope"],
                    portfolio,
                    claim["con_id"],
                    symbol,
                    FillSide.BUY,
                    quantity,
                    price,
                    fill["execution_id"],
                    record.fingerprint,
                    fill["commission_minor"],
                    fill["commission_currency"],
                    fill["commission_source"],
                    parse_utc_text(fill["occurred_at"]),
                ),
            )
            if (
                fifo.replayed
                or fifo.signed_quantity != quantity
                or fifo.average_cost != price
                or fifo.fill_realized_pnl != 0
                or fifo.total_realized_pnl != Decimal(pre["realized_pnl"])
            ):
                raise ValidationError("entry FIFO differs from proposed accounting")
            database._paper_settlement_fault("ENTRY_AFTER_FIFO")
            cursor = await connection.execute(
                """INSERT INTO main.trades
                (portfolio_id,symbol,side,quantity,price,notional,slippage,commission,pnl,timestamp)
                VALUES (?,?,'BUY',?,?,?,0,0,0,?)""",
                (
                    portfolio,
                    symbol,
                    int(quantity),
                    float(price),
                    float(_exact_decimal_multiply(quantity, price, "entry notional")),
                    stamp,
                ),
            )
            trade_id = cursor.lastrowid
            database._paper_settlement_fault("ENTRY_AFTER_TRADE")
        settlement_id = "pset-" + uuid.uuid4().hex
        receipt_payload = canonical_json(
            dict(
                settlement_id=settlement_id,
                request_fingerprint=record.fingerprint,
                trade_id=trade_id,
                database_path=str(path),
                database_identity=identity,
                database_device=descriptor.device,
                database_inode=descriptor.inode,
                committed_at=stamp,
                schema_version=1,
                settlement_kind="ENTRY",
            )
        )
        fingerprint = sha256_text(receipt_payload)
        await connection.execute(
            """INSERT INTO main.paper_reduction_settlements
            (settlement_id,execution_domain_scope,account_scope,portfolio_id,con_id,symbol,
            reservation_id,claim_id,order_ref,protective_quote_payload,request_fingerprint,
            request_payload_json,terminal_status,trade_id,database_path,database_identity,
            database_device,database_inode,committed_at,receipt_fingerprint,schema_version,settlement_kind)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'ENTRY')""",
            (
                settlement_id,
                claim["execution_domain_scope"],
                claim["account_scope"],
                portfolio,
                claim["con_id"],
                symbol,
                reservation["event_id"],
                claim["claim_id"],
                outcome["order_ref"],
                canonical_json(data["quote"]),
                record.fingerprint,
                record.payload_json,
                outcome["status"],
                trade_id,
                str(path),
                identity,
                descriptor.device,
                descriptor.inode,
                stamp,
                fingerprint,
            ),
        )
        database._paper_settlement_fault("ENTRY_AFTER_TERMINAL")
        if quantity:
            await connection.execute(
                """INSERT INTO main.positions
                (portfolio_id,symbol,quantity,avg_cost,market_price,timestamp) VALUES (?,?,?,?,?,?)
                ON CONFLICT(portfolio_id,symbol) DO UPDATE SET quantity=excluded.quantity,
                avg_cost=excluded.avg_cost,market_price=excluded.market_price,timestamp=excluded.timestamp""",
                (
                    portfolio,
                    symbol,
                    int(quantity),
                    float(price),
                    float(Decimal(post["position_mark_price"])),
                    stamp,
                ),
            )
            await connection.execute(
                """INSERT INTO main.paper_position_settlement_state
                (portfolio_id,symbol,cost_basis_text,mark_price_text,source_settlement_id,updated_at,
                 origin_bootstrap_id)
                VALUES (?,?,?,?,?,?,(SELECT origin_bootstrap_id
                    FROM main.paper_account_settlement_state WHERE portfolio_id=?)) ON CONFLICT(portfolio_id,symbol) DO UPDATE SET
                cost_basis_text=excluded.cost_basis_text,mark_price_text=excluded.mark_price_text,
                source_settlement_id=excluded.source_settlement_id,updated_at=excluded.updated_at""",
                (
                    portfolio,
                    symbol,
                    post["position_cost_basis"],
                    post["position_mark_price"],
                    settlement_id,
                    stamp,
                    portfolio,
                ),
            )
            database._paper_settlement_fault("ENTRY_AFTER_POSITION")
            unrealized = _exact_decimal_subtract(
                _exact_decimal_add(
                    Decimal(post["daily_pnl"]), Decimal(post["daily_pnl_baseline"]), "unrealized"
                ),
                Decimal(post["realized_pnl"]),
                "unrealized",
            )
            # Equity is only a legacy display projection. Preserve its prior
            # value and apply this fill's exact marked P&L delta; do not promote
            # its floating-point value into authoritative risk evidence.
            equity = _exact_decimal_add(
                Decimal(str(legacy[3])),
                _exact_decimal_subtract(
                    Decimal(post["daily_pnl"]), Decimal(pre["daily_pnl"]), "entry P&L delta"
                ),
                "entry compatibility equity",
            )
            await connection.execute(
                """UPDATE main.account SET cash=?,realized_pnl=?,daily_pnl=?,
                unrealized_pnl=?,equity=?,timestamp=? WHERE portfolio_id=?""",
                (
                    float(Decimal(post["cash"])),
                    float(Decimal(post["realized_pnl"])),
                    float(Decimal(post["daily_pnl"])),
                    float(unrealized),
                    float(equity),
                    stamp,
                    portfolio,
                ),
            )
            await connection.execute(
                """INSERT INTO main.paper_fifo_settlement_links
                (settlement_id,request_fingerprint,epoch_id,fill_id,event_sequence,execution_id,
                 commission_minor,commission_currency,commission_source,fifo_state_fingerprint,committed_at)
                 VALUES (?,?,?,?,?,?,0,'USD',?,?,?)""",
                (
                    settlement_id,
                    record.fingerprint,
                    fifo.epoch_id,
                    fifo.fill_id,
                    fifo.event_sequence,
                    fill["execution_id"],
                    fill["commission_source"],
                    fifo.state_fingerprint,
                    stamp,
                ),
            )
            database._paper_settlement_fault("ENTRY_AFTER_FIFO_LINK")
        await connection.execute(
            """UPDATE main.paper_account_settlement_state SET
            cash_text=?,realized_pnl_text=?,daily_pnl_text=?,source_settlement_id=?,updated_at=?
            WHERE portfolio_id=?""",
            (
                post["cash"],
                post["realized_pnl"],
                post["daily_pnl"],
                settlement_id,
                stamp,
                portfolio,
            ),
        )
        database._paper_settlement_fault("ENTRY_AFTER_ACCOUNT")
        if await database._sqlite_descriptor_identity(connection) != descriptor:
            raise ValidationError("entry database descriptor changed during staging")
        binding.assert_connection_identity(await database._sqlite_descriptor_identity(connection))
        await connection.execute("RELEASE SAVEPOINT entry_terminal_stage")
        return StagedEntrySettlement(settlement_id, trade_id, fingerprint)
    except BaseException:
        await connection.execute("ROLLBACK TO SAVEPOINT entry_terminal_stage")
        await connection.execute("RELEASE SAVEPOINT entry_terminal_stage")
        raise
    finally:
        binding.close()

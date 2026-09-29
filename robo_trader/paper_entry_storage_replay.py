"""Read-only entry storage verification; no execution, repair, or release permit."""

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import re

from .accounting.fifo import FillSide
from .accounting.fifo_runtime import RuntimePaperFillEvidence, verify_runtime_fill_in_transaction
from .database_migrations import assert_paper_settlement_hot_schema
from .paper_entry_settlement import (
    PaperEntryTerminalRecord,
    validate_stored_paper_entry_terminal_record,
)
from .paper_terminal_settlement import _exact_decimal_multiply
from .safety.models import ValidationError, canonical_json, parse_utc_text, sha256_text
from .safety.sqlite_identity import SQLitePathBinding


@dataclass(frozen=True)
class EntrySettlementRow:
    """Verified storage data, deliberately not a producer-owned commit receipt."""

    settlement_id: str
    trade_id: int | None
    fingerprint: str
    committed_at: datetime


async def read_entry_settlement(connection, record, *, database, runtime_contract, journal):
    """Authenticate a previous entry inside one caller-owned read snapshot.

    Current positions/cash may have advanced since this historical record; full
    ledger continuity belongs to mixed-history replay. This function verifies
    the row's own immutable terminal/trade/FIFO correspondence and file identity.
    It never creates missing state or grants permission to execute again.
    """
    from .database_async import AsyncTradingDatabase

    if type(database) is not AsyncTradingDatabase or not connection.in_transaction:
        raise ValidationError("entry recovery requires a database read transaction")
    if type(record) is not PaperEntryTerminalRecord:
        raise ValidationError("entry recovery requires an exact terminal record")
    validate_stored_paper_entry_terminal_record(
        record.payload_json, fingerprint=record.fingerprint, journal=journal
    )
    data = json.loads(record.payload_json)
    claim = data["claim"]
    path, identity = database._expected_safety_database(runtime_contract=runtime_contract)
    if (
        runtime_contract.execution_mode != "paper"
        or runtime_contract.state_namespace != "paper"
        or runtime_contract.ibkr_readonly is not True
        or runtime_contract.safety_account_scope != claim["account_scope"]
        or runtime_contract.safety_execution_domain_scope != claim["execution_domain_scope"]
    ):
        raise ValidationError("entry recovery runtime scope differs")
    binding = SQLitePathBinding.open_readonly(path)
    try:
        descriptor = await database._sqlite_descriptor_identity(connection)
        binding = binding.bind_sqlite_connection(descriptor)
        if (descriptor.device, descriptor.inode) != database._expected_database_file_identity:
            raise ValidationError("entry recovery database identity changed")
        await assert_paper_settlement_hot_schema(connection)
        result = await connection._execute(
            _read_entry_storage_on_connection,
            connection._conn,
            record,
            path,
            identity,
            descriptor.device,
            descriptor.inode,
        )
        binding.assert_connection_identity(await database._sqlite_descriptor_identity(connection))
        return result
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError("entry storage verification failed") from error
    finally:
        binding.close()


def _read_entry_storage_on_connection(connection, record, path, identity, device, inode):
    """Shared read-only SQL/FIFO core for owned async and bootstrap snapshots.

    Callers validate the record against actual journal history, runtime scope,
    schema and path binding. This returns data only, never an owned receipt.
    Aiosqlite invokes it on its worker; synchronous reconciliation uses its own
    held read transaction. Neither transaction is committed or rolled back here.
    """
    if not connection.in_transaction or type(record) is not PaperEntryTerminalRecord:
        raise ValidationError("entry storage core requires a record and read transaction")
    data = json.loads(record.payload_json)
    claim, reservation = data["claim"], data["reservation"]
    capacity = json.loads(reservation["payload_json"])
    outcome = data["outcome"]
    row_factory = connection.row_factory
    try:
        # FIFO's SQL contract uses tuples. Restore the caller's row factory on
        # both success and failure; this changes no persistent database state.
        connection.row_factory = None
        cursor = connection.execute(
            """SELECT * FROM main.paper_reduction_settlements
            WHERE reservation_id=? OR claim_id=? OR order_ref=? OR request_fingerprint=?""",
            (reservation["event_id"], claim["claim_id"], outcome["order_ref"], record.fingerprint),
        )
        rows = cursor.fetchall()
        if not rows:
            raise ValidationError("entry terminal storage is absent")
        if len(rows) != 1:
            raise ValidationError("entry terminal identities resolve to conflicting rows")
        row = dict(zip((column[0] for column in cursor.description), rows[0]))
        expected = dict(
            settlement_kind="ENTRY",
            schema_version=1,
            execution_domain_scope=claim["execution_domain_scope"],
            account_scope=claim["account_scope"],
            portfolio_id=claim["portfolio_id"],
            con_id=claim["con_id"],
            symbol=capacity["symbol"],
            reservation_id=reservation["event_id"],
            claim_id=claim["claim_id"],
            order_ref=outcome["order_ref"],
            protective_quote_payload=canonical_json(data["quote"]),
            request_fingerprint=record.fingerprint,
            request_payload_json=record.payload_json,
            terminal_status=outcome["status"],
            database_path=str(path),
            database_identity=identity,
            database_device=device,
            database_inode=inode,
        )
        if any(type(row[k]) is not type(v) or row[k] != v for k, v in expected.items()):
            raise ValidationError(
                "entry terminal row differs from journal-bound record or database"
            )
        if (
            type(row["settlement_id"]) is not str
            or re.fullmatch(r"pset-[0-9a-f]{32}", row["settlement_id"]) is None
        ):
            raise ValidationError("entry terminal storage ID is invalid")
        committed_at = parse_utc_text(row["committed_at"])
        if committed_at < parse_utc_text(outcome["observed_at"]) or committed_at > datetime.now(
            timezone.utc
        ):
            raise ValidationError("entry terminal storage time is invalid")
        receipt = {
            k: row[k]
            for k in (
                "settlement_id",
                "request_fingerprint",
                "trade_id",
                "database_path",
                "database_identity",
                "database_device",
                "database_inode",
                "committed_at",
                "schema_version",
                "settlement_kind",
            )
        }
        if sha256_text(canonical_json(receipt)) != row["receipt_fingerprint"]:
            raise ValidationError("entry terminal storage fingerprint differs")
        link = (
            connection.execute(
                """SELECT request_fingerprint,epoch_id,fill_id,event_sequence,
            execution_id,commission_minor,commission_currency,commission_source,fifo_state_fingerprint,
            committed_at FROM main.paper_fifo_settlement_links WHERE settlement_id=?""",
                (row["settlement_id"],),
            )
        ).fetchone()
        quantity = Decimal(outcome["filled_quantity"])
        if quantity:
            if type(row["trade_id"]) is not int or row["trade_id"] <= 0:
                raise ValidationError("entry filled terminal lacks a trade identity")
            price = Decimal(outcome["exact_fill_price"])
            trade = (
                connection.execute(
                    """SELECT portfolio_id,symbol,side,quantity,price,
                notional,slippage,commission,pnl,timestamp FROM main.trades WHERE id=?""",
                    (row["trade_id"],),
                )
            ).fetchone()
            expected_trade = (
                claim["portfolio_id"],
                capacity["symbol"],
                "BUY",
                int(quantity),
                float(price),
                float(_exact_decimal_multiply(quantity, price, "entry replay notional")),
                0,
                0,
                0,
                row["committed_at"],
            )
            if trade != expected_trade:
                raise ValidationError("entry terminal trade differs from exact fill")
            fill = outcome["fill_evidence"]
            evidence = RuntimePaperFillEvidence(
                claim["execution_domain_scope"],
                claim["account_scope"],
                claim["portfolio_id"],
                claim["con_id"],
                capacity["symbol"],
                FillSide.BUY,
                quantity,
                price,
                fill["execution_id"],
                record.fingerprint,
                fill["commission_minor"],
                fill["commission_currency"],
                fill["commission_source"],
                parse_utc_text(fill["occurred_at"]),
            )
            # Run the strictly existing-fill verifier on the owning worker;
            # unlike append, absence fails before it can create a FIFO event.
            fifo = verify_runtime_fill_in_transaction(connection, evidence)
            expected_link = (
                record.fingerprint,
                fifo.epoch_id,
                fifo.fill_id,
                fifo.event_sequence,
                fill["execution_id"],
                fill["commission_minor"],
                fill["commission_currency"],
                fill["commission_source"],
                fifo.state_fingerprint,
                row["committed_at"],
            )
            if link != expected_link or not fifo.replayed:
                raise ValidationError("entry terminal FIFO linkage differs")
            if (
                fifo.signed_quantity != Decimal(data["post_values"]["position_quantity"])
                or fifo.average_cost != Decimal(data["post_values"]["position_cost_basis"])
                or fifo.fill_realized_pnl != 0
                or fifo.total_realized_pnl != Decimal(data["post_values"]["realized_pnl"])
            ):
                raise ValidationError("entry terminal FIFO accounting differs")
        else:
            orphan_fill = (
                connection.execute(
                    "SELECT 1 FROM main.fifo_fills WHERE idempotency_key=? LIMIT 1",
                    (record.fingerprint,),
                )
            ).fetchone()
            if row["trade_id"] is not None or link is not None or orphan_fill is not None:
                raise ValidationError("zero-fill entry claims executed storage")
        return EntrySettlementRow(
            row["settlement_id"], row["trade_id"], row["receipt_fingerprint"], committed_at
        )
    finally:
        connection.row_factory = row_factory

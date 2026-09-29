"""Read-only reconstruction of paper cash and quantities for entry evidence.

This collector grants no order authority or launch readiness. Fresh broker-bound
valuation, reconciliation eligibility, and reservations are separate inputs; the
runtime must hold its account order lock while collecting and consuming them.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from robo_trader.accounting.fifo import FifoLedger
from robo_trader.config import RuntimeContract
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.database_migrations import assert_paper_settlement_hot_schema
from robo_trader.database_validator import DatabaseValidator
from robo_trader.paper_entry_settlement import PaperEntryTerminalRecord
from robo_trader.paper_entry_storage_replay import read_entry_settlement
from robo_trader.reconciliation.runtime_integration import _read_validated_bootstrap_candidates
from robo_trader.safety.journal import SafetyJournal
from robo_trader.safety.models import parse_fixed_decimal, parse_utc_text
from robo_trader.safety.sqlite_identity import SQLiteIdentityError, SQLitePathBinding


class PaperRiskLedgerSnapshotError(RuntimeError):
    """The complete account ledger cannot be reconstructed without ambiguity."""


@dataclass(frozen=True)
class PaperRiskPosition:
    portfolio_id: str
    symbol: str
    con_id: int
    quantity: Decimal


@dataclass(frozen=True)
class PaperRiskFill:
    portfolio_id: str
    symbol: str
    execution_id: str
    occurred_at: datetime


@dataclass(frozen=True)
class PaperRiskLedgerSnapshot:
    account_scope: str
    execution_domain_scope: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    observed_at: datetime
    portfolio_cash: tuple[tuple[str, Decimal], ...]
    bootstrap_effective_at: tuple[tuple[str, datetime], ...]
    positions: tuple[PaperRiskPosition, ...]
    fills: tuple[PaperRiskFill, ...]

    def fingerprint(self) -> str:
        payload = json.dumps(_snapshot_state(self), separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def _snapshot_state(snapshot):
    """Validate exact nested types before producing an unambiguous seal state."""

    def text(value):
        if type(value) is not str or not value or value != value.strip():
            raise PaperRiskLedgerSnapshotError("snapshot text is malformed")
        return value

    def decimal(value):
        if type(value) is not Decimal or not value.is_finite():
            raise PaperRiskLedgerSnapshotError("snapshot money/quantity is not an exact Decimal")
        return str(value)

    def timestamp(value):
        if (
            type(value) is not datetime
            or value.tzinfo is None
            or value.utcoffset() != timezone.utc.utcoffset(value)
        ):
            raise PaperRiskLedgerSnapshotError("snapshot timestamp must be exact UTC datetime")
        return value.isoformat(timespec="microseconds")

    if type(snapshot) is not PaperRiskLedgerSnapshot:
        raise PaperRiskLedgerSnapshotError("snapshot type is malformed")
    if any(
        type(value) is not tuple
        for value in (
            snapshot.portfolio_cash,
            snapshot.bootstrap_effective_at,
            snapshot.positions,
            snapshot.fills,
        )
    ):
        raise PaperRiskLedgerSnapshotError("snapshot collections must be immutable tuples")
    cash = []
    for row in snapshot.portfolio_cash:
        if type(row) is not tuple or len(row) != 2:
            raise PaperRiskLedgerSnapshotError("snapshot portfolio cash is malformed")
        cash.append((text(row[0]), decimal(row[1])))
    portfolios = tuple(p for p, _ in cash)
    if not portfolios or portfolios != tuple(sorted(set(portfolios))):
        raise PaperRiskLedgerSnapshotError("snapshot portfolios must be unique and sorted")
    observed = timestamp(snapshot.observed_at)
    history = []
    for row in snapshot.bootstrap_effective_at:
        if type(row) is not tuple or len(row) != 2:
            raise PaperRiskLedgerSnapshotError("snapshot bootstrap history is malformed")
        history.append((text(row[0]), timestamp(row[1])))
        if row[1] > snapshot.observed_at:
            raise PaperRiskLedgerSnapshotError("snapshot bootstrap history starts in the future")
    if tuple(p for p, _ in history) != portfolios:
        raise PaperRiskLedgerSnapshotError("snapshot bootstrap history coverage is incomplete")
    positions = []
    for position in snapshot.positions:
        if type(position) is not PaperRiskPosition:
            raise PaperRiskLedgerSnapshotError("snapshot position type is malformed")
        q = decimal(position.quantity)
        if (
            type(position.con_id) is not int
            or position.con_id <= 0
            or not position.quantity
            or position.quantity != position.quantity.to_integral_value()
            or position.portfolio_id not in portfolios
        ):
            raise PaperRiskLedgerSnapshotError("snapshot position values are malformed")
        positions.append((text(position.portfolio_id), text(position.symbol), position.con_id, q))
    keys = tuple((p, s) for p, s, _, _ in positions)
    if keys != tuple(sorted(set(keys))):
        raise PaperRiskLedgerSnapshotError("snapshot positions must be unique and sorted")
    fills = []
    for fill in snapshot.fills:
        if type(fill) is not PaperRiskFill or fill.portfolio_id not in portfolios:
            raise PaperRiskLedgerSnapshotError("snapshot fill type/scope is malformed")
        occurred = timestamp(fill.occurred_at)
        if not re.fullmatch(r"lpfill-[0-9a-f]{32}", text(fill.execution_id)):
            raise PaperRiskLedgerSnapshotError("snapshot execution identity is malformed")
        fills.append((text(fill.portfolio_id), text(fill.symbol), fill.execution_id, occurred))
    for value in (snapshot.database_device, snapshot.database_inode):
        if type(value) is not int or value < 0:
            raise PaperRiskLedgerSnapshotError("snapshot database inode is malformed")
    return (
        "paper-risk-ledger-v2",
        text(snapshot.account_scope),
        text(snapshot.execution_domain_scope),
        text(snapshot.database_path),
        text(snapshot.database_identity),
        snapshot.database_device,
        snapshot.database_inode,
        observed,
        cash,
        history,
        positions,
        fills,
    )


def _snapshot_registry():
    registry = {}
    lock = threading.Lock()

    def issue(**values):
        result = PaperRiskLedgerSnapshot(**values)
        key = id(result)

        def discard(reference):
            with lock:
                if registry.get(key, (None,))[0] is reference:
                    registry.pop(key, None)

        with lock:
            registry[key] = (weakref.ref(result, discard), result.fingerprint())
        return result

    def verify(result):
        if type(result) is not PaperRiskLedgerSnapshot:
            raise PaperRiskLedgerSnapshotError("exact owned ledger snapshot required")
        with lock:
            record = registry.get(id(result))
        if record is None or record[0]() is not result or record[1] != result.fingerprint():
            raise PaperRiskLedgerSnapshotError("ledger snapshot is unowned or changed")

    return issue, verify


_issue_snapshot, assert_owned_paper_risk_ledger_snapshot = _snapshot_registry()


def _verify_fifo(connection, candidates, expected_fills, expected_links):
    ledger = FifoLedger(connection, allow_other_objects=True)
    expected_epochs = {
        candidate.fifo_bootstrap_plan().epoch_id for candidate in candidates.values()
    }
    actual_epochs = {
        row[0] for row in connection.execute("SELECT epoch_id FROM fifo_accounting_epochs")
    }
    if actual_epochs != expected_epochs:
        raise PaperRiskLedgerSnapshotError(
            "FIFO epochs do not cover exactly the account portfolios"
        )
    for epoch in sorted(expected_epochs):
        ledger.verify_epoch_integrity(epoch)
    rows = connection.execute("""
        SELECT e.portfolio_id,f.execution_id,f.symbol,f.con_id,f.side,
               f.quantity_text,f.price_text,c.amount_minor,f.idempotency_key
        FROM fifo_fills f JOIN fifo_accounting_epochs e ON e.epoch_id=f.epoch_id
        JOIN fifo_commissions c ON c.epoch_id=f.epoch_id AND c.fill_id=f.fill_id
        ORDER BY e.portfolio_id,f.execution_id
    """).fetchall()
    actual = tuple(
        (
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            parse_fixed_decimal(r[5], "FIFO quantity"),
            parse_fixed_decimal(r[6], "FIFO price"),
            r[7],
            r[8],
        )
        for r in rows
    )
    if actual != tuple(sorted(expected_fills)):
        raise PaperRiskLedgerSnapshotError("FIFO fills and terminal outbox disagree")
    link_count = connection.execute("SELECT COUNT(*) FROM paper_fifo_settlement_links").fetchone()[
        0
    ]
    links = connection.execute("""
        SELECT l.settlement_id,l.request_fingerprint,l.execution_id,l.commission_minor,
               l.commission_currency,l.commission_source
        FROM paper_fifo_settlement_links l
        JOIN fifo_fills f ON f.epoch_id=l.epoch_id AND f.fill_id=l.fill_id
        JOIN fifo_position_snapshots s ON s.epoch_id=f.epoch_id AND s.source_fill_id=f.fill_id
        WHERE l.request_fingerprint=f.idempotency_key AND l.execution_id=f.execution_id
          AND l.event_sequence=f.event_sequence AND l.fifo_state_fingerprint=s.state_fingerprint
        ORDER BY l.settlement_id
    """).fetchall()
    if link_count != len(expected_links) or tuple(links) != tuple(sorted(expected_links)):
        raise PaperRiskLedgerSnapshotError("FIFO terminal links are missing or mismatched")


async def _replay_entry(
    connection,
    row,
    database,
    runtime,
    candidates,
    cash,
    quantities,
    latest_cash_source,
    latest_position_source,
    fills,
    expected_fifo_links,
    expected_fifo_fills,
):
    """Validate an entry on the collector's transaction, then advance history.

    No independent connection or owned execution capability is used here: the
    collector needs immutable facts from exactly its existing read snapshot.
    """
    if not runtime.safety_journal_path:
        raise PaperRiskLedgerSnapshotError("entry history requires the safety journal")
    record = PaperEntryTerminalRecord(row[2])
    if record.fingerprint != row[1]:
        raise PaperRiskLedgerSnapshotError("entry record fingerprint differs")
    stored = await read_entry_settlement(
        connection,
        record,
        database=database,
        runtime_contract=runtime,
        journal=SafetyJournal(runtime.safety_journal_path),
    )
    if stored.settlement_id != row[0]:
        raise PaperRiskLedgerSnapshotError("entry terminal identity differs")
    data = json.loads(record.payload_json)
    claim, pre, post, outcome = (
        data["claim"],
        data["pre_account"],
        data["post_values"],
        data["outcome"],
    )
    capacity = json.loads(data["reservation"]["payload_json"])
    portfolio, symbol, con_id = claim["portfolio_id"], capacity["symbol"], claim["con_id"]
    key = (portfolio, symbol)
    if portfolio not in cash or parse_fixed_decimal(pre["cash"]) != cash[portfolio]:
        raise PaperRiskLedgerSnapshotError("entry cash history is discontinuous")
    # Gross flatness is checked across all bootstrapped portfolios, not merely
    # the entry's portfolio or the signed account sum. An absent key is legal
    # only here, after the journal-bound flat-entry record and storage validate.
    if any(q != 0 for (_, s), (_, q) in quantities.items() if s == symbol):
        raise PaperRiskLedgerSnapshotError("entry position history is not gross-flat")
    if key in quantities and quantities[key][0] != con_id:
        raise PaperRiskLedgerSnapshotError("entry contract history changed")
    if pre["position_source_settlement_id"] != latest_position_source.get(key):
        raise PaperRiskLedgerSnapshotError("entry position source history is discontinuous")
    at = parse_utc_text(outcome["observed_at"])
    if at < candidates[portfolio].effective_at or at > datetime.now(timezone.utc):
        raise PaperRiskLedgerSnapshotError("entry fill timestamp is outside the accounting epoch")
    quantity = parse_fixed_decimal(outcome["filled_quantity"])
    cash[portfolio] = parse_fixed_decimal(post["cash"])
    latest_cash_source[portfolio] = stored.settlement_id
    if quantity:
        quantities[key] = (con_id, parse_fixed_decimal(post["position_quantity"]))
        latest_position_source[key] = stored.settlement_id
        fill = outcome["fill_evidence"]
        fills.append(PaperRiskFill(portfolio, symbol, fill["execution_id"], at))
        expected_fifo_links.append(
            (
                stored.settlement_id,
                record.fingerprint,
                fill["execution_id"],
                fill["commission_minor"],
                fill["commission_currency"],
                fill["commission_source"],
            )
        )
        expected_fifo_fills.append(
            (
                portfolio,
                fill["execution_id"],
                symbol,
                con_id,
                "BUY",
                quantity,
                parse_fixed_decimal(outcome["exact_fill_price"]),
                fill["commission_minor"],
                record.fingerprint,
            )
        )


async def collect_paper_risk_ledger_snapshot(
    database: AsyncTradingDatabase, runtime: RuntimeContract
) -> PaperRiskLedgerSnapshot:
    """Rebuild complete paper history and projections in one read snapshot."""
    if type(database) is not AsyncTradingDatabase:
        raise PaperRiskLedgerSnapshotError("exact shared database is required")
    async with database.get_connection() as connection:
        await connection.execute("BEGIN")
        try:
            return await _collect_paper_risk_ledger_snapshot_in_transaction(
                database, runtime, connection
            )
        finally:
            await connection.rollback()


async def _collect_paper_risk_ledger_snapshot_in_transaction(database, runtime, connection):
    """Validate history in an existing transaction without committing or rolling back.

    Reduction persistence reuses this before any mutation when ENTRY history
    exists. A second connection would validate a different snapshot and deadlock
    the single-connection pool. The caller owns transaction lifecycle.
    """
    if type(database) is not AsyncTradingDatabase or type(runtime) is not RuntimeContract:
        raise PaperRiskLedgerSnapshotError("exact shared database and runtime are required")
    if (
        runtime.execution_mode != "paper"
        or runtime.execution_source != "paper_simulator"
        or runtime.state_namespace != "paper"
        or runtime.ibkr_readonly is not True
        or runtime.safety_execution_domain_scope != "paper-simulator-v1"
    ):
        raise PaperRiskLedgerSnapshotError("explicit paper read-only runtime is required")
    expected_path, identity = database._expected_safety_database(runtime_contract=runtime)
    if not connection.in_transaction:
        raise PaperRiskLedgerSnapshotError("ledger reconstruction requires a transaction")
    binding = None
    try:
        binding = SQLitePathBinding.open_readonly(database.db_path)
        descriptor = await database._sqlite_descriptor_identity(connection)
        binding = binding.bind_sqlite_connection(descriptor)
        if (descriptor.device, descriptor.inode) != database._expected_database_file_identity:
            raise PaperRiskLedgerSnapshotError("ledger database identity was replaced")
        observed_at = datetime.now(timezone.utc)
        await assert_paper_settlement_hot_schema(connection)
        candidates = await _read_validated_bootstrap_candidates(connection, runtime, binding)
        cash = {portfolio: candidate.account.cash for portfolio, candidate in candidates.items()}
        quantities = {
            (portfolio, p.symbol): (p.con_id, Decimal(p.quantity))
            for portfolio, candidate in candidates.items()
            for p in candidate.positions
        }
        latest_cash_source = {portfolio: None for portfolio in candidates}
        latest_position_source = {key: None for key in quantities}
        fingerprints = await (
            await connection.execute(
                "SELECT portfolio_id,candidate_fingerprint FROM main.paper_state_bootstraps"
            )
        ).fetchall()
        if dict(fingerprints) != {p: c.fingerprint() for p, c in candidates.items()}:
            raise PaperRiskLedgerSnapshotError("bootstrap candidate fingerprint mismatch")
        fills = []
        expected_fifo_fills = []
        expected_fifo_links = []
        rows = await (await connection.execute("""
            SELECT settlement_id,request_fingerprint,request_payload_json,
                   protective_quote_payload,trade_id,database_path,database_identity,
                   database_device,database_inode,committed_at,receipt_fingerprint,schema_version,
                   portfolio_id,symbol,execution_domain_scope,account_scope,settlement_kind
            FROM main.paper_reduction_settlements ORDER BY rowid
        """)).fetchall()
        for row in rows:
            if row[16] == "ENTRY":
                await _replay_entry(
                    connection,
                    row,
                    database,
                    runtime,
                    candidates,
                    cash,
                    quantities,
                    latest_cash_source,
                    latest_position_source,
                    fills,
                    expected_fifo_links,
                    expected_fifo_fills,
                )
                continue
            if row[16] != "REDUCTION":
                raise PaperRiskLedgerSnapshotError("unknown terminal settlement kind")
            receipt = database._paper_settlement_receipt_from_row(row[:12])
            request = receipt.request
            if (
                row[12:16]
                != (
                    request.portfolio_id,
                    request.symbol,
                    runtime.safety_execution_domain_scope,
                    runtime.safety_account_scope,
                )
                or request.account_scope != runtime.safety_account_scope
                or request.execution_domain_scope != runtime.safety_execution_domain_scope
                or receipt.database_path != str(expected_path)
                or receipt.database_identity != identity
                or (receipt.database_device, receipt.database_inode)
                != (descriptor.device, descriptor.inode)
            ):
                raise PaperRiskLedgerSnapshotError(
                    "terminal receipt scope or database identity mismatch"
                )
            key = (request.portfolio_id, request.symbol)
            if (
                request.portfolio_id not in cash
                or key not in quantities
                or quantities[key] != (request.con_id, request.expected_pre_position_quantity)
            ):
                raise PaperRiskLedgerSnapshotError("terminal position history is discontinuous")
            if request.expected_pre_cash != cash[request.portfolio_id]:
                raise PaperRiskLedgerSnapshotError("terminal cash history is discontinuous")
            if request.outcome_at < candidates[
                request.portfolio_id
            ].effective_at or request.outcome_at > datetime.now(timezone.utc):
                raise PaperRiskLedgerSnapshotError(
                    "terminal fill timestamp is outside the accounting epoch"
                )
            cash[request.portfolio_id] = request.expected_post_cash
            quantities[key] = (request.con_id, request.expected_post_position_quantity)
            latest_cash_source[request.portfolio_id] = receipt.settlement_id
            if request.filled_quantity:
                latest_position_source[key] = receipt.settlement_id
                fills.append(
                    PaperRiskFill(
                        request.portfolio_id,
                        request.symbol,
                        request.fill_execution_id,
                        request.outcome_at,
                    )
                )
                expected_fifo_links.append(
                    (
                        receipt.settlement_id,
                        request.fingerprint(),
                        request.fill_execution_id,
                        request.fill_commission_minor,
                        request.fill_commission_currency,
                        request.fill_commission_source,
                    )
                )
                expected_fifo_fills.append(
                    (
                        request.portfolio_id,
                        request.fill_execution_id,
                        request.symbol,
                        request.con_id,
                        "SELL" if request.side.value == "SELL" else "BUY",
                        request.filled_quantity,
                        request.fill_price,
                        request.fill_commission_minor,
                        request.fingerprint(),
                    )
                )
        # Projection state may change marks/P&L, but cash and quantity
        # changes require committed trading or an explicitly supported
        # accounting event. Keeping an old source ID cannot bless a rewrite.
        compatibility_accounts = await (
            await connection.execute("SELECT portfolio_id FROM main.account ORDER BY portfolio_id")
        ).fetchall()
        if tuple(row[0] for row in compatibility_accounts) != tuple(sorted(cash)):
            raise PaperRiskLedgerSnapshotError("account projection coverage is incomplete")
        account_rows = await (await connection.execute("""
            SELECT portfolio_id,cash_text,source_settlement_id FROM main.paper_account_settlement_state
        """)).fetchall()
        if len(account_rows) != len(cash) or any(
            p not in cash
            or parse_fixed_decimal(value, "cash") != cash[p]
            or source != latest_cash_source[p]
            for p, value, source in account_rows
        ):
            raise PaperRiskLedgerSnapshotError(
                "mutable cash projection disagrees with immutable history"
            )
        position_rows = await (await connection.execute("""
            SELECT p.portfolio_id,p.symbol,p.quantity,s.source_settlement_id
            FROM main.positions p LEFT JOIN main.paper_position_settlement_state s
              ON s.portfolio_id=p.portfolio_id AND s.symbol=p.symbol
        """)).fetchall()
        seen = set()
        for portfolio, symbol, quantity, source in position_rows:
            key = (portfolio, symbol)
            if (
                portfolio not in cash
                or DatabaseValidator.validate_symbol(symbol) != symbol
                or type(quantity) is not int
                or key in seen
                or Decimal(quantity) != quantities.get(key, (None, Decimal(0)))[1]
                or source != latest_position_source.get(key)
            ):
                raise PaperRiskLedgerSnapshotError(
                    "mutable position projection disagrees with immutable history"
                )
            seen.add(key)
        if any(quantity != 0 and key not in seen for key, (_, quantity) in quantities.items()):
            raise PaperRiskLedgerSnapshotError("nonzero position projection is missing")
        await connection._execute(
            _verify_fifo,
            connection._conn,
            candidates,
            expected_fifo_fills,
            expected_fifo_links,
        )
        binding.assert_connection_identity(await database._sqlite_descriptor_identity(connection))
        return _issue_snapshot(
            account_scope=runtime.safety_account_scope,
            execution_domain_scope=runtime.safety_execution_domain_scope,
            database_path=str(expected_path),
            database_identity=identity,
            database_device=descriptor.device,
            database_inode=descriptor.inode,
            observed_at=observed_at,
            portfolio_cash=tuple(sorted(cash.items())),
            bootstrap_effective_at=tuple(
                (p, c.effective_at) for p, c in sorted(candidates.items())
            ),
            positions=tuple(
                PaperRiskPosition(p, s, c, q) for (p, s), (c, q) in sorted(quantities.items()) if q
            ),
            fills=tuple(fills),
        )
    except PaperRiskLedgerSnapshotError:
        raise
    except SQLiteIdentityError as exc:
        raise PaperRiskLedgerSnapshotError("paper ledger database identity changed") from exc
    except Exception as exc:
        raise PaperRiskLedgerSnapshotError(f"paper ledger snapshot rejected: {exc}") from exc
    finally:
        if binding is not None:
            binding.close()


def assert_complete_paper_daily_history(
    snapshot: PaperRiskLedgerSnapshot, *, portfolio_id: str, as_of: datetime
) -> None:
    """Post-bootstrap receipts prove complete days only after bootstrap's NY date.

    Cash/positions do not prove pre-bootstrap gross executions. Until an
    authenticated historical execution import exists, the bootstrap day is
    unknown even when the local terminal outbox is empty.
    """
    assert_owned_paper_risk_ledger_snapshot(snapshot)
    if (
        type(as_of) is not datetime
        or as_of.tzinfo is None
        or as_of.utcoffset() != timezone.utc.utcoffset(as_of)
    ):
        raise PaperRiskLedgerSnapshotError("daily history requires exact UTC time")
    origins = dict(snapshot.bootstrap_effective_at)
    if portfolio_id not in origins:
        raise PaperRiskLedgerSnapshotError("daily history has no portfolio coverage")
    zone = ZoneInfo("America/New_York")
    if (
        as_of < snapshot.observed_at
        or as_of.astimezone(zone).date() <= origins[portfolio_id].astimezone(zone).date()
    ):
        raise PaperRiskLedgerSnapshotError(
            "daily history is incomplete for the bootstrap trading date"
        )

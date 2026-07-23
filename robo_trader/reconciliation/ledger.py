"""Immutable SQLite reader for portfolio-scoped reconciliation evidence."""

from __future__ import annotations

import re
import sqlite3
import stat
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

from .errors import LedgerSafetyError
from .models import (
    AggregatedLedgerPosition,
    LedgerPosition,
    LedgerSnapshot,
    LedgerTrade,
)

_PORTFOLIO_ID = re.compile(r"^[a-z0-9_-]{1,64}$")
_SYMBOL = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")
_ACCOUNT_FRAGMENT = re.compile(r"(?:DU|U)\d{4,}", re.IGNORECASE)
_REQUIRED_COLUMNS = {
    "positions": {
        "portfolio_id": "TEXT",
        "symbol": "TEXT",
        "quantity": "INTEGER",
        "avg_cost": "REAL",
        "timestamp": "DATETIME",
    },
    "trades": {
        "id": "INTEGER",
        "portfolio_id": "TEXT",
        "symbol": "TEXT",
        "side": "TEXT",
        "quantity": "INTEGER",
        "price": "REAL",
        "timestamp": "DATETIME",
    },
    "account": {"portfolio_id": "TEXT"},
}


def validate_portfolio_ids(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(value).strip().lower() for value in values)
    if not normalized or any(
        not _PORTFOLIO_ID.fullmatch(value) or _ACCOUNT_FRAGMENT.search(value)
        for value in normalized
    ):
        raise LedgerSafetyError("one or more explicit portfolio IDs are invalid")
    if len(normalized) != len(set(normalized)):
        raise LedgerSafetyError("portfolio IDs must be unique")
    return tuple(sorted(normalized))


def resolve_database_path(project_root: Path, database_path: str) -> Path:
    path = Path(database_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    if path.is_symlink():
        raise LedgerSafetyError("ledger path must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LedgerSafetyError("ledger database is missing") from exc
    if not resolved.is_file():
        raise LedgerSafetyError("ledger database is not a regular file")
    sidecars = (
        ("WAL", Path(f"{resolved}-wal")),
        ("shared-memory sidecar", Path(f"{resolved}-shm")),
        ("rollback journal", Path(f"{resolved}-journal")),
    )
    for label, sidecar in sidecars:
        try:
            metadata = sidecar.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise LedgerSafetyError(f"ledger {label} cannot be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_size:
            raise LedgerSafetyError(
                f"ledger has an ambiguous {label}; immutable reconciliation would be incomplete"
            )
    return resolved


def _authorizer(
    action: int,
    argument1: Optional[str],
    argument2: Optional[str],
    database: Optional[str],
    source: Optional[str],
) -> int:
    del argument2, database, source
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_TRANSACTION,
    }
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and str(argument1).casefold() == "table_info":
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _decimal(value: object, field: str, *, nonnegative: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise LedgerSafetyError(f"ledger {field} is not a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise LedgerSafetyError(f"ledger {field} is not a finite decimal") from exc
    if not result.is_finite() or (nonnegative and result < 0):
        raise LedgerSafetyError(f"ledger {field} is not a valid finite decimal")
    return result


def _safe_ledger_text(value: object, field: str, *, max_length: int = 128) -> str:
    text = str(value)
    if (
        not text
        or len(text) > max_length
        or any(ord(character) < 32 for character in text)
        or _ACCOUNT_FRAGMENT.search(text)
    ):
        raise LedgerSafetyError(f"ledger {field} contains invalid or sensitive text")
    return text


def _parse_ledger_timestamp(value: object, field: str) -> datetime:
    """Parse SQLite/ISO timestamps and normalize them to aware UTC values."""
    text = _safe_ledger_text(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LedgerSafetyError(f"ledger {field} is malformed") from exc
    # SQLite CURRENT_TIMESTAMP is UTC but omits an offset.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise LedgerSafetyError(f"ledger {field} is malformed") from exc


class ImmutableLedgerReader:
    """Read a stable ledger without invoking any application write path."""

    def __init__(self, project_root: Path, database_path: str):
        self.path = resolve_database_path(project_root, database_path)

    def _connect(self) -> sqlite3.Connection:
        encoded = quote(str(self.path), safe="/")
        uri = f"file:{encoded}?mode=ro&immutable=1"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=1.0, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            if connection.execute("PRAGMA query_only").fetchone()[0] != 1:
                raise LedgerSafetyError("SQLite query-only mode could not be proven")
            connection.set_authorizer(_authorizer)
            return connection
        except sqlite3.Error as exc:
            raise LedgerSafetyError("immutable ledger connection failed") from exc

    @staticmethod
    def _validate_schema(connection: sqlite3.Connection) -> None:
        for table, required in _REQUIRED_COLUMNS.items():
            try:
                rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            except sqlite3.Error as exc:
                raise LedgerSafetyError("ledger schema cannot be inspected") from exc
            if not rows:
                raise LedgerSafetyError("ledger schema is missing required tables")
            actual = {str(row["name"]): str(row["type"]).upper() for row in rows}
            for column, expected_type in required.items():
                if actual.get(column) != expected_type:
                    raise LedgerSafetyError("ledger schema is not reconciliation-compatible")

    def read(
        self, portfolio_ids: Iterable[str], *, recent_trade_limit: int = 500
    ) -> LedgerSnapshot:
        selected = validate_portfolio_ids(portfolio_ids)
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            self._validate_schema(connection)
            known_rows = connection.execute("""
                SELECT portfolio_id FROM account
                UNION SELECT portfolio_id FROM positions
                UNION SELECT portfolio_id FROM trades
                ORDER BY portfolio_id
                """).fetchall()
            known = tuple(str(row[0]) for row in known_rows)
            if any(
                not _PORTFOLIO_ID.fullmatch(value)
                or value != value.lower()
                or _ACCOUNT_FRAGMENT.search(value)
                for value in known
            ):
                raise LedgerSafetyError("ledger contains ambiguous portfolio identity")
            missing = sorted(set(selected) - set(known))
            if missing:
                raise LedgerSafetyError("one or more selected portfolios do not exist")

            position_identity_rows = connection.execute("""
                SELECT portfolio_id, symbol
                FROM positions
                ORDER BY portfolio_id, symbol
                """).fetchall()
            position_identities: set[tuple[str, str]] = set()
            for row in position_identity_rows:
                portfolio_id = str(row["portfolio_id"])
                raw_symbol = str(row["symbol"]).strip()
                symbol = raw_symbol.upper()
                identity = (portfolio_id, symbol)
                if (
                    portfolio_id not in known
                    or portfolio_id != portfolio_id.lower()
                    or raw_symbol != symbol
                    or not _SYMBOL.fullmatch(symbol)
                ):
                    raise LedgerSafetyError("ledger position identity is malformed")
                if identity in position_identities:
                    raise LedgerSafetyError(
                        "ledger contains duplicate portfolio and symbol positions"
                    )
                position_identities.add(identity)

            raw_positions = connection.execute("""
                SELECT portfolio_id, symbol, quantity, avg_cost, timestamp,
                       typeof(quantity) AS quantity_type
                FROM positions
                WHERE quantity != 0
                ORDER BY portfolio_id, symbol
                """).fetchall()
            positions: list[LedgerPosition] = []
            for row in raw_positions:
                portfolio_id = str(row["portfolio_id"])
                raw_symbol = str(row["symbol"]).strip()
                symbol = raw_symbol.upper()
                if (
                    portfolio_id not in known
                    or portfolio_id != portfolio_id.lower()
                    or raw_symbol != symbol
                    or not _SYMBOL.fullmatch(symbol)
                ):
                    raise LedgerSafetyError("ledger position identity is malformed")
                if row["quantity_type"] != "integer":
                    raise LedgerSafetyError(
                        "ledger quantity is fractional despite the INTEGER position schema"
                    )
                quantity = _decimal(row["quantity"], "position quantity")
                average_cost = _decimal(row["avg_cost"], "position average cost", nonnegative=True)
                positions.append(
                    LedgerPosition(
                        portfolio_id=portfolio_id,
                        symbol=symbol,
                        quantity=quantity,
                        average_cost=average_cost,
                        timestamp=_parse_ledger_timestamp(row["timestamp"], "position timestamp"),
                    )
                )

            active = tuple(sorted({position.portfolio_id for position in positions}))
            blockers: list[str] = []
            caveats = [
                "LOCAL_LEDGER_HAS_NO_CONID_EXCHANGE_OR_CURRENCY",
                "LOCAL_TRADES_HAVE_NO_BROKER_ORDER_OR_EXECUTION_IDS",
                "BROKER_AND_LOCAL_PAPER_EXECUTOR_ARE_SEPARATE_EXECUTION_DOMAINS",
            ]
            unselected_active = sorted(set(active) - set(selected))
            if unselected_active:
                blockers.append("UNSELECTED_ACTIVE_PORTFOLIOS")
            if len(active) > 1:
                blockers.append("AMBIGUOUS_MULTI_PORTFOLIO_BROKER_ALLOCATION")

            selected_positions = [
                position for position in positions if position.portfolio_id in selected
            ]
            aggregates = self._aggregate(selected_positions, blockers, caveats)

            placeholders = ",".join("?" for _ in selected)
            # The only interpolated characters are one parameter marker per
            # already-validated portfolio ID; every value remains SQL-bound.
            trade_query = (
                """
                SELECT id, portfolio_id, symbol, side, quantity, price, timestamp,
                       typeof(quantity) AS quantity_type
                FROM trades
                WHERE portfolio_id IN ("""
                + placeholders  # nosec B608
                + """)
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """
            )
            trade_rows = connection.execute(
                trade_query,
                (*selected, recent_trade_limit),
            ).fetchall()
            trades = []
            for row in trade_rows:
                if row["quantity_type"] != "integer":
                    raise LedgerSafetyError(
                        "ledger trade quantity is fractional despite the INTEGER schema"
                    )
                raw_symbol = str(row["symbol"]).strip()
                symbol = raw_symbol.upper()
                side = str(row["side"]).strip().upper()
                if raw_symbol != symbol or not _SYMBOL.fullmatch(symbol):
                    raise LedgerSafetyError("ledger trade symbol is malformed")
                if side not in {"BUY", "SELL", "BUY_TO_COVER", "SELL_SHORT"}:
                    raise LedgerSafetyError("ledger trade side is malformed")
                quantity = _decimal(row["quantity"], "trade quantity")
                if quantity <= 0:
                    raise LedgerSafetyError("ledger trade quantity must be positive")
                trades.append(
                    LedgerTrade(
                        local_trade_id=int(row["id"]),
                        portfolio_id=str(row["portfolio_id"]),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=_decimal(row["price"], "trade price", nonnegative=True),
                        timestamp=_parse_ledger_timestamp(row["timestamp"], "trade timestamp"),
                    )
                )
            connection.execute("ROLLBACK")
            return LedgerSnapshot(
                selected_portfolio_ids=selected,
                known_portfolio_ids=known,
                active_portfolio_ids=active,
                positions=tuple(selected_positions),
                aggregated_positions=aggregates,
                recent_trades=tuple(trades),
                blockers=tuple(sorted(set(blockers))),
                caveats=tuple(sorted(set(caveats))),
            )
        except sqlite3.Error as exc:
            raise LedgerSafetyError("ledger read failed closed") from exc
        finally:
            connection.close()

    @staticmethod
    def _aggregate(
        positions: Iterable[LedgerPosition], blockers: list[str], caveats: list[str]
    ) -> tuple[AggregatedLedgerPosition, ...]:
        grouped: dict[str, list[LedgerPosition]] = defaultdict(list)
        for position in positions:
            grouped[position.symbol].append(position)

        result = []
        for symbol in sorted(grouped):
            allocations = tuple(sorted(grouped[symbol], key=lambda item: item.portfolio_id))
            signs = {1 if item.quantity > 0 else -1 for item in allocations}
            offsetting = len(signs) > 1
            total = sum((item.quantity for item in allocations), Decimal("0"))
            average_cost: Optional[Decimal]
            if offsetting:
                blockers.append(f"OFFSETTING_PORTFOLIO_POSITIONS:{symbol}")
                caveats.append(f"AGGREGATE_COST_UNDEFINED_FOR_OFFSETTING_POSITIONS:{symbol}")
                average_cost = None
                if total == 0:
                    blockers.append(f"NET_ZERO_MASKS_PORTFOLIO_EXPOSURE:{symbol}")
            else:
                denominator = sum((abs(item.quantity) for item in allocations), Decimal("0"))
                numerator = sum(
                    (abs(item.quantity) * item.average_cost for item in allocations),
                    Decimal("0"),
                )
                average_cost = numerator / denominator
            result.append(
                AggregatedLedgerPosition(
                    symbol=symbol,
                    quantity=total,
                    average_cost=average_cost,
                    allocations=allocations,
                    has_offsetting_allocations=offsetting,
                )
            )
        return tuple(result)

"""Exact, append-only bootstrap records for a legacy paper-simulator ledger.

The legacy SQLite projections use ``REAL`` and cannot become financial
authority merely by passing through ``Decimal(str(value))``.  A bootstrap is
therefore an explicit, fingerprinted accounting epoch.  Its values are
operator-reviewed inputs bound to a read-only broker snapshot proving that the
separate IBKR paper account has no exposure or open orders.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping

from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .safety.models import decimal_to_fixed, utc_to_text

BOOTSTRAP_SCHEMA_VERSION = 1
BOOTSTRAP_ID_PREFIX = "pboot-"
MAX_MARK_AGE = timedelta(minutes=5)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_BOOTSTRAP_ID = re.compile(r"^pboot-[0-9a-f]{32}$")
_ACCOUNT_SCOPE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")


class ExactStateBootstrapError(ValueError):
    """A candidate cannot safely establish a sealed accounting epoch."""


def _decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    if type(value) not in {Decimal, str}:
        raise ExactStateBootstrapError(f"{label} is not an exact decimal")
    try:
        exact = value if type(value) is Decimal else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ExactStateBootstrapError(f"{label} is not an exact decimal") from exc
    if not exact.is_finite() or (positive and exact <= 0):
        raise ExactStateBootstrapError(f"{label} is outside its valid range")
    # Reject noncanonical textual inputs instead of silently normalizing them.
    if isinstance(value, str) and decimal_to_fixed(exact) != value:
        raise ExactStateBootstrapError(f"{label} is not canonical")
    return exact


def _utc(value: object, label: str) -> datetime:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ExactStateBootstrapError(f"{label} is invalid") from exc
    else:
        raise ExactStateBootstrapError(f"{label} is invalid")
    if result.tzinfo is None or result.utcoffset() is None:
        raise ExactStateBootstrapError(f"{label} must be timezone-aware")
    return result.astimezone(timezone.utc)


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise ExactStateBootstrapError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ExactStateBootstrapError(f"{label} is malformed")
    return value


@dataclass(frozen=True, slots=True)
class ExactBootstrapPosition:
    symbol: str
    quantity: int
    cost_basis: Decimal
    mark_price: Decimal
    mark_observed_at: datetime
    mark_evidence_fingerprint: str

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        if not _SYMBOL.fullmatch(symbol):
            raise ExactStateBootstrapError("bootstrap position symbol is malformed")
        if isinstance(self.quantity, bool) or type(self.quantity) is not int or self.quantity == 0:
            raise ExactStateBootstrapError("bootstrap position quantity must be a nonzero integer")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self, "cost_basis", _decimal(self.cost_basis, "cost_basis", positive=True)
        )
        object.__setattr__(
            self, "mark_price", _decimal(self.mark_price, "mark_price", positive=True)
        )
        object.__setattr__(
            self,
            "mark_observed_at",
            _utc(self.mark_observed_at, "mark_observed_at"),
        )
        object.__setattr__(
            self,
            "mark_evidence_fingerprint",
            _hash(self.mark_evidence_fingerprint, "mark_evidence_fingerprint"),
        )

    def public_dict(self) -> dict[str, object]:
        return {
            "cost_basis_text": decimal_to_fixed(self.cost_basis),
            "mark_evidence_fingerprint": self.mark_evidence_fingerprint,
            "mark_observed_at": utc_to_text(self.mark_observed_at),
            "mark_price_text": decimal_to_fixed(self.mark_price),
            "quantity": self.quantity,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class ExactBootstrapAccount:
    cash: Decimal
    realized_pnl: Decimal
    daily_pnl: Decimal
    daily_pnl_baseline: Decimal
    daily_pnl_date: date

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash", _decimal(self.cash, "cash"))
        object.__setattr__(self, "realized_pnl", _decimal(self.realized_pnl, "realized_pnl"))
        object.__setattr__(self, "daily_pnl", _decimal(self.daily_pnl, "daily_pnl"))
        object.__setattr__(
            self,
            "daily_pnl_baseline",
            _decimal(self.daily_pnl_baseline, "daily_pnl_baseline"),
        )
        if type(self.daily_pnl_date) is not date:
            raise ExactStateBootstrapError("daily_pnl_date must be an exact date")

    def public_dict(self) -> dict[str, str]:
        return {
            "cash_text": decimal_to_fixed(self.cash),
            "daily_pnl_baseline_text": decimal_to_fixed(self.daily_pnl_baseline),
            "daily_pnl_date": self.daily_pnl_date.isoformat(),
            "daily_pnl_text": decimal_to_fixed(self.daily_pnl),
            "realized_pnl_text": decimal_to_fixed(self.realized_pnl),
        }


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapCandidate:
    bootstrap_id: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    database_path: str
    database_identity: str
    reconciliation_snapshot_id: str
    reconciliation_report_hash: str
    broker_snapshot_hash: str
    legacy_snapshot_hash: str
    broker_position_count: int
    broker_open_order_count: int
    effective_at: datetime
    account: ExactBootstrapAccount
    positions: tuple[ExactBootstrapPosition, ...]
    schema_version: int = BOOTSTRAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BOOTSTRAP_SCHEMA_VERSION:
            raise ExactStateBootstrapError("unsupported bootstrap schema version")
        if not _BOOTSTRAP_ID.fullmatch(self.bootstrap_id):
            raise ExactStateBootstrapError("bootstrap_id is malformed")
        if self.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE:
            raise ExactStateBootstrapError("bootstrap is not bound to paper-simulator-v1")
        if not _ACCOUNT_SCOPE.fullmatch(self.account_scope):
            raise ExactStateBootstrapError("account_scope is malformed")
        _safe_id(self.portfolio_id, "portfolio_id")
        path = Path(self.database_path)
        if not path.is_absolute() or str(path) != self.database_path:
            raise ExactStateBootstrapError("database_path must be absolute and lexical")
        _safe_id(self.database_identity, "database_identity")
        _safe_id(self.reconciliation_snapshot_id, "reconciliation_snapshot_id")
        for value, label in (
            (self.reconciliation_report_hash, "reconciliation_report_hash"),
            (self.broker_snapshot_hash, "broker_snapshot_hash"),
            (self.legacy_snapshot_hash, "legacy_snapshot_hash"),
        ):
            _hash(value, label)
        if (
            type(self.broker_position_count) is not int
            or type(self.broker_open_order_count) is not int
            or self.broker_position_count != 0
            or self.broker_open_order_count != 0
        ):
            raise ExactStateBootstrapError(
                "IBKR paper account must have zero exposure and zero open orders"
            )
        effective_at = _utc(self.effective_at, "effective_at")
        object.__setattr__(self, "effective_at", effective_at)
        positions = tuple(self.positions)
        if not positions:
            raise ExactStateBootstrapError("bootstrap must describe every nonzero legacy position")
        if any(type(position) is not ExactBootstrapPosition for position in positions):
            raise ExactStateBootstrapError("bootstrap positions are malformed")
        symbols = [position.symbol for position in positions]
        if len(symbols) != len(set(symbols)) or symbols != sorted(symbols):
            raise ExactStateBootstrapError("bootstrap positions must be unique and sorted")
        for position in positions:
            age = effective_at - position.mark_observed_at
            if age < timedelta(0) or age > MAX_MARK_AGE:
                raise ExactStateBootstrapError("bootstrap protective mark is future or stale")
        object.__setattr__(self, "positions", positions)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account": self.account.public_dict(),
            "account_scope": self.account_scope,
            "bootstrap_id": self.bootstrap_id,
            "broker_open_order_count": self.broker_open_order_count,
            "broker_position_count": self.broker_position_count,
            "broker_snapshot_hash": self.broker_snapshot_hash,
            "database_identity": self.database_identity,
            "database_path": self.database_path,
            "effective_at": utc_to_text(self.effective_at),
            "execution_domain_scope": self.execution_domain_scope,
            "legacy_snapshot_hash": self.legacy_snapshot_hash,
            "portfolio_id": self.portfolio_id,
            "positions": [position.public_dict() for position in self.positions],
            "reconciliation_report_hash": self.reconciliation_report_hash,
            "reconciliation_snapshot_id": self.reconciliation_snapshot_id,
            "schema_version": self.schema_version,
        }

    def canonical_payload(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_payload().encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExactStateBootstrapCandidate":
        if not isinstance(raw, Mapping):
            raise ExactStateBootstrapError("bootstrap document must be an object")
        expected = {
            "account",
            "account_scope",
            "bootstrap_id",
            "broker_open_order_count",
            "broker_position_count",
            "broker_snapshot_hash",
            "database_identity",
            "database_path",
            "effective_at",
            "execution_domain_scope",
            "legacy_snapshot_hash",
            "portfolio_id",
            "positions",
            "reconciliation_report_hash",
            "reconciliation_snapshot_id",
            "schema_version",
        }
        if set(raw) != expected:
            raise ExactStateBootstrapError("bootstrap document fields are incomplete or unknown")
        account_raw = raw["account"]
        if not isinstance(account_raw, Mapping):
            raise ExactStateBootstrapError("bootstrap account is malformed")
        try:
            daily_pnl_date = date.fromisoformat(str(account_raw.get("daily_pnl_date")))
        except ValueError as exc:
            raise ExactStateBootstrapError("daily_pnl_date is invalid") from exc
        account = ExactBootstrapAccount(
            cash=account_raw.get("cash_text"),
            realized_pnl=account_raw.get("realized_pnl_text"),
            daily_pnl=account_raw.get("daily_pnl_text"),
            daily_pnl_baseline=account_raw.get("daily_pnl_baseline_text"),
            daily_pnl_date=daily_pnl_date,
        )
        positions_raw = raw["positions"]
        if not isinstance(positions_raw, list):
            raise ExactStateBootstrapError("bootstrap positions must be a list")
        positions = tuple(
            ExactBootstrapPosition(
                symbol=item["symbol"],
                quantity=item["quantity"],
                cost_basis=item["cost_basis_text"],
                mark_price=item["mark_price_text"],
                mark_observed_at=item["mark_observed_at"],
                mark_evidence_fingerprint=item["mark_evidence_fingerprint"],
            )
            for item in positions_raw
            if isinstance(item, Mapping)
        )
        if len(positions) != len(positions_raw):
            raise ExactStateBootstrapError("bootstrap position item is malformed")
        return cls(
            bootstrap_id=raw["bootstrap_id"],
            execution_domain_scope=raw["execution_domain_scope"],
            account_scope=raw["account_scope"],
            portfolio_id=raw["portfolio_id"],
            database_path=raw["database_path"],
            database_identity=raw["database_identity"],
            reconciliation_snapshot_id=raw["reconciliation_snapshot_id"],
            reconciliation_report_hash=raw["reconciliation_report_hash"],
            broker_snapshot_hash=raw["broker_snapshot_hash"],
            legacy_snapshot_hash=raw["legacy_snapshot_hash"],
            broker_position_count=raw["broker_position_count"],
            broker_open_order_count=raw["broker_open_order_count"],
            effective_at=raw["effective_at"],
            account=account,
            positions=positions,
            schema_version=raw["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class ExactStateBootstrapReceipt:
    bootstrap_id: str
    candidate_fingerprint: str
    operator_action_id: str
    database_device: int
    database_inode: int
    committed_at: datetime


def _canonical_legacy_rows(
    account_rows: Iterable[sqlite3.Row],
    position_rows: Iterable[sqlite3.Row],
    trade_summary: sqlite3.Row,
) -> str:
    payload = {
        "account": [list(row) for row in account_rows],
        "positions": [list(row) for row in position_rows],
        "trade_summary": list(trade_summary),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def inspect_legacy_state(database_path: Path) -> dict[str, object]:
    """Read and fingerprint the legacy projections without creating SQLite files."""

    path = Path(database_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ExactStateBootstrapError("legacy database must be an existing regular path")
    uri = path.as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        required = {"account", "positions", "trades", "portfolios"}
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not required.issubset(tables):
            raise ExactStateBootstrapError("legacy database schema is incomplete")
        account_rows = connection.execute(
            "SELECT portfolio_id,cash,equity,daily_pnl,realized_pnl,unrealized_pnl,timestamp "
            "FROM account ORDER BY portfolio_id"
        ).fetchall()
        position_rows = connection.execute(
            "SELECT portfolio_id,symbol,quantity,avg_cost,market_price,timestamp "
            "FROM positions WHERE quantity <> 0 ORDER BY portfolio_id,symbol"
        ).fetchall()
        trade_summary = connection.execute(
            "SELECT COUNT(*),COALESCE(MIN(id),0),COALESCE(MAX(id),0),"
            "COALESCE(SUM(quantity),0) FROM trades"
        ).fetchone()
        if trade_summary is None:
            raise ExactStateBootstrapError("legacy trade summary is unavailable")
        payload = _canonical_legacy_rows(account_rows, position_rows, trade_summary)
        return {
            "account_rows": [dict(row) for row in account_rows],
            "position_rows": [dict(row) for row in position_rows],
            "snapshot_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "trade_count": trade_summary[0],
        }
    finally:
        connection.close()

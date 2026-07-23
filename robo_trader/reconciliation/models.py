"""Immutable evidence models used by the reconciliation engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence

from .errors import BrokerEvidenceError

_SYMBOL = re.compile(r"^[A-Z]{1,5}(?:\.[A-Z]{1,2})?$")
_TEXT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_PORTFOLIO_ID = re.compile(r"^[a-z0-9_-]{1,64}$")
_ACCOUNT_SHAPED = re.compile(r"^(?:DU|U)\d+$", re.IGNORECASE)
_ACCOUNT_FRAGMENT = re.compile(r"(?:DU|U)\d{4,}", re.IGNORECASE)


def _safe_text(value: object, field_name: str, *, max_length: int = 128) -> str:
    text = str(value).strip()
    if (
        not text
        or len(text) > max_length
        or any(ord(character) < 32 for character in text)
        or _ACCOUNT_FRAGMENT.search(text)
    ):
        raise BrokerEvidenceError(f"{field_name} is invalid or contains sensitive identity")
    return text


def _safe_portfolio_id(value: object, field_name: str) -> str:
    text = str(value).strip()
    if text != text.lower() or not _PORTFOLIO_ID.fullmatch(text) or _ACCOUNT_FRAGMENT.search(text):
        raise BrokerEvidenceError(f"{field_name} is invalid or contains sensitive identity")
    return text


def _assert_public_safe(value: object, field_name: str) -> None:
    """Reject account-shaped text before it can reach a public report."""
    if isinstance(value, str):
        if _ACCOUNT_FRAGMENT.search(value):
            raise BrokerEvidenceError(f"{field_name} contains sensitive identity")
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_public_safe(str(key), field_name)
            _assert_public_safe(nested, field_name)
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _assert_public_safe(nested, field_name)


def finite_decimal(value: Any, field_name: str) -> Decimal:
    """Parse a finite Decimal without accepting booleans or binary-float drift."""
    if isinstance(value, bool):
        raise BrokerEvidenceError(f"{field_name} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BrokerEvidenceError(f"{field_name} must be a finite decimal") from exc
    if not result.is_finite():
        raise BrokerEvidenceError(f"{field_name} must be a finite decimal")
    return result


def canonical_decimal(value: Decimal) -> str:
    """Serialize a Decimal deterministically without exponent notation."""
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def aware_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise BrokerEvidenceError(f"{field_name} must be timezone-aware")
    return value


@dataclass(frozen=True)
class ContractIdentity:
    con_id: int
    symbol: str
    local_symbol: str
    security_type: str
    currency: str
    exchange: str
    primary_exchange: str
    trading_class: str

    def __post_init__(self) -> None:
        if isinstance(self.con_id, bool) or not isinstance(self.con_id, int) or self.con_id <= 0:
            raise BrokerEvidenceError("broker contract has an invalid conId")
        symbol = self.symbol.strip().upper()
        if not _SYMBOL.fullmatch(symbol) or self.local_symbol.strip().upper() != symbol:
            raise BrokerEvidenceError("broker contract symbol identity is ambiguous")
        if self.security_type != "STK" or self.currency != "USD" or self.exchange != "SMART":
            raise BrokerEvidenceError(
                "broker contract is not an explicitly qualified SMART/USD stock"
            )
        primary_exchange = _safe_text(self.primary_exchange, "broker contract primary exchange")
        trading_class = _safe_text(self.trading_class, "broker contract trading class")
        if not _TEXT_ID.fullmatch(primary_exchange) or not _TEXT_ID.fullmatch(trading_class):
            raise BrokerEvidenceError("broker contract identity is incomplete")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "local_symbol", symbol)
        object.__setattr__(self, "primary_exchange", primary_exchange)
        object.__setattr__(self, "trading_class", trading_class)

    def public_dict(self) -> dict[str, object]:
        return {
            "con_id": self.con_id,
            "symbol": self.symbol,
            "local_symbol": self.local_symbol,
            "security_type": self.security_type,
            "currency": self.currency,
            "exchange": self.exchange,
            "primary_exchange": self.primary_exchange,
            "trading_class": self.trading_class,
        }


@dataclass(frozen=True)
class BrokerPosition:
    contract: ContractIdentity
    quantity: Decimal
    average_cost: Decimal

    def __post_init__(self) -> None:
        quantity = finite_decimal(self.quantity, "broker position quantity")
        average_cost = finite_decimal(self.average_cost, "broker position average cost")
        if quantity == 0:
            raise BrokerEvidenceError("broker snapshot contains a zero position")
        if average_cost < 0:
            raise BrokerEvidenceError("broker position average cost cannot be negative")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "average_cost", average_cost)


@dataclass(frozen=True)
class BrokerOpenOrder:
    order_id: str
    client_id: int
    contract: ContractIdentity
    side: str
    quantity: Decimal
    filled: Decimal
    remaining: Decimal
    order_type: str
    status: str
    limit_price: Optional[Decimal] = None
    auxiliary_price: Optional[Decimal] = None
    permanent_id: Optional[str] = None
    time_in_force: Optional[str] = None
    average_fill_price: Optional[Decimal] = None
    last_status_at: Optional[datetime] = None
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.order_id.isdigit() or len(self.order_id) > 20:
            raise BrokerEvidenceError("broker open order is missing an order ID")
        side = self.side.strip().upper()
        if side not in {"BUY", "SELL", "BOT", "SLD"}:
            raise BrokerEvidenceError("broker open order has an invalid side")
        quantity = finite_decimal(self.quantity, "broker order quantity")
        filled = finite_decimal(self.filled, "broker order filled quantity")
        remaining = finite_decimal(self.remaining, "broker order remaining quantity")
        if quantity <= 0 or filled < 0 or remaining < 0 or filled + remaining != quantity:
            raise BrokerEvidenceError("broker open order quantities are inconsistent")
        for name in ("limit_price", "auxiliary_price", "average_fill_price"):
            value = getattr(self, name)
            if value is not None:
                parsed = finite_decimal(value, f"broker order {name}")
                if parsed < 0:
                    raise BrokerEvidenceError(f"broker order {name} cannot be negative")
                object.__setattr__(self, name, parsed)
        if self.permanent_id is not None and (
            not self.permanent_id.isdigit() or len(self.permanent_id) > 20
        ):
            raise BrokerEvidenceError("broker open order has an invalid permanent ID")
        if (
            isinstance(self.client_id, bool)
            or not isinstance(self.client_id, int)
            or self.client_id < 0
        ):
            raise BrokerEvidenceError("broker open order has an invalid client ID")
        if self.last_status_at is not None:
            object.__setattr__(
                self,
                "last_status_at",
                aware_datetime(self.last_status_at, "broker order status timestamp"),
            )
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_type", _safe_text(self.order_type, "broker order type"))
        object.__setattr__(self, "status", _safe_text(self.status, "broker order status"))
        if self.time_in_force is not None:
            object.__setattr__(
                self,
                "time_in_force",
                _safe_text(self.time_in_force, "broker order time in force"),
            )
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "filled", filled)
        object.__setattr__(self, "remaining", remaining)
        object.__setattr__(self, "unavailable", MappingProxyType(dict(self.unavailable)))

    @property
    def identity(self) -> tuple[int, str]:
        """IBKR order identity is unique only within its client ID."""
        return (self.client_id, self.order_id)


@dataclass(frozen=True)
class BrokerExecution:
    execution_id: str
    order_id: Optional[str]
    contract: ContractIdentity
    side: str
    quantity: Decimal
    price: Decimal
    executed_at: datetime
    permanent_id: Optional[str] = None
    client_id: Optional[int] = None
    execution_exchange: Optional[str] = None
    average_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    commission_currency: Optional[str] = None
    realized_pnl: Optional[Decimal] = None
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        execution_id = _safe_text(self.execution_id, "broker execution identity")
        if not _TEXT_ID.fullmatch(execution_id) or _ACCOUNT_SHAPED.fullmatch(execution_id):
            raise BrokerEvidenceError("broker execution identity is incomplete")
        object.__setattr__(self, "execution_id", execution_id)
        if self.order_id is not None and (not self.order_id.isdigit() or len(self.order_id) > 20):
            raise BrokerEvidenceError("broker execution order identity is invalid")
        quantity = finite_decimal(self.quantity, "broker execution quantity")
        price = finite_decimal(self.price, "broker execution price")
        if quantity <= 0 or price < 0:
            raise BrokerEvidenceError("broker execution quantity or price is invalid")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        side = self.side.strip().upper()
        if side not in {"BUY", "SELL", "BOT", "SLD"}:
            raise BrokerEvidenceError("broker execution side is invalid")
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self, "executed_at", aware_datetime(self.executed_at, "broker execution timestamp")
        )
        for name in ("average_price", "commission", "realized_pnl"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_decimal(value, f"broker execution {name}"))
        if self.permanent_id is not None and (
            not self.permanent_id.isdigit() or len(self.permanent_id) > 20
        ):
            raise BrokerEvidenceError("broker execution permanent identity is invalid")
        if self.client_id is not None and (isinstance(self.client_id, bool) or self.client_id < 0):
            raise BrokerEvidenceError("broker execution client identity is invalid")
        if self.execution_exchange is not None:
            object.__setattr__(
                self,
                "execution_exchange",
                _safe_text(self.execution_exchange, "broker execution exchange"),
            )
        if self.commission_currency is not None:
            object.__setattr__(
                self,
                "commission_currency",
                _safe_text(self.commission_currency, "broker commission currency"),
            )
        object.__setattr__(self, "unavailable", MappingProxyType(dict(self.unavailable)))


@dataclass(frozen=True)
class BrokerExecutionScope:
    """Exact bounded execution request represented by the broker snapshot."""

    kind: str
    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if self.kind != "bounded_execution_filter":
            raise BrokerEvidenceError("broker execution scope is unsupported")
        object.__setattr__(
            self,
            "start_at",
            aware_datetime(self.start_at, "broker execution scope start"),
        )
        object.__setattr__(
            self,
            "end_at",
            aware_datetime(self.end_at, "broker execution scope end"),
        )
        if self.end_at <= self.start_at:
            raise BrokerEvidenceError("broker execution scope is not a positive bounded window")
        if self.start_at.microsecond != 0:
            raise BrokerEvidenceError(
                "broker execution scope start is not the exact whole-second wire filter"
            )
        duration_seconds = (self.end_at - self.start_at).total_seconds()
        # The upper bound is the 60-second snapshot collection limit plus the
        # sub-second amount discarded by IBKR's whole-second wire format.
        if duration_seconds < 86400 or duration_seconds >= 86461:
            raise BrokerEvidenceError("broker execution scope is not the exact bounded window")

    def public_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "start_at": self.start_at.isoformat(),
            "end_at": self.end_at.isoformat(),
        }


@dataclass(frozen=True)
class BrokerSnapshot:
    schema_version: int
    account_alias: str
    broker_time_before: datetime
    broker_time_after: datetime
    retrieved_at: datetime
    execution_scope: BrokerExecutionScope
    positions: tuple[BrokerPosition, ...] = ()
    open_orders: tuple[BrokerOpenOrder, ...] = ()
    recent_executions: tuple[BrokerExecution, ...] = ()
    balances: Mapping[str, Decimal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BrokerEvidenceError("unsupported broker snapshot schema")
        if not self.account_alias.startswith("***") or len(self.account_alias) > 7:
            raise BrokerEvidenceError("broker snapshot account alias is not masked")
        for name in ("broker_time_before", "broker_time_after", "retrieved_at"):
            object.__setattr__(self, name, aware_datetime(getattr(self, name), name))
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "open_orders", tuple(self.open_orders))
        object.__setattr__(self, "recent_executions", tuple(self.recent_executions))
        if len({order.identity for order in self.open_orders}) != len(self.open_orders):
            raise BrokerEvidenceError("broker snapshot has duplicate open-order identity")
        if len({execution.execution_id for execution in self.recent_executions}) != len(
            self.recent_executions
        ):
            raise BrokerEvidenceError("broker snapshot has duplicate execution identity")
        if not isinstance(self.execution_scope, BrokerExecutionScope):
            raise BrokerEvidenceError("broker execution scope evidence is missing")
        if any(
            execution.executed_at < self.execution_scope.start_at
            or execution.executed_at > self.execution_scope.end_at
            for execution in self.recent_executions
        ):
            raise BrokerEvidenceError(
                "broker snapshot execution falls outside the exact requested scope"
            )
        normalized_balances = {
            _safe_text(key, "broker balance identity"): finite_decimal(
                value, "broker balance value"
            )
            for key, value in self.balances.items()
        }
        balance_tags = {key.split(":", 1)[0] for key in normalized_balances}
        if not {"NetLiquidation", "TotalCashValue"}.issubset(balance_tags):
            raise BrokerEvidenceError("broker snapshot is missing required account balances")
        object.__setattr__(self, "balances", MappingProxyType(normalized_balances))


@dataclass(frozen=True)
class LedgerPosition:
    portfolio_id: str
    symbol: str
    quantity: Decimal
    average_cost: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "portfolio_id",
            _safe_portfolio_id(self.portfolio_id, "ledger portfolio identity"),
        )
        symbol = self.symbol.strip().upper()
        if self.symbol != symbol or not _SYMBOL.fullmatch(symbol):
            raise BrokerEvidenceError("ledger position symbol identity is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self,
            "timestamp",
            aware_datetime(self.timestamp, "ledger position timestamp"),
        )


@dataclass(frozen=True)
class AggregatedLedgerPosition:
    symbol: str
    quantity: Decimal
    average_cost: Optional[Decimal]
    allocations: tuple[LedgerPosition, ...]
    has_offsetting_allocations: bool = False


@dataclass(frozen=True)
class LedgerTrade:
    local_trade_id: int
    portfolio_id: str
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "portfolio_id",
            _safe_portfolio_id(self.portfolio_id, "ledger portfolio identity"),
        )
        symbol = self.symbol.strip().upper()
        if self.symbol != symbol or not _SYMBOL.fullmatch(symbol):
            raise BrokerEvidenceError("ledger trade symbol identity is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(
            self,
            "timestamp",
            aware_datetime(self.timestamp, "ledger trade timestamp"),
        )


@dataclass(frozen=True)
class LedgerSnapshot:
    selected_portfolio_ids: tuple[str, ...]
    known_portfolio_ids: tuple[str, ...]
    active_portfolio_ids: tuple[str, ...]
    positions: tuple[LedgerPosition, ...]
    aggregated_positions: tuple[AggregatedLedgerPosition, ...]
    recent_trades: tuple[LedgerTrade, ...]
    blockers: tuple[str, ...] = ()
    caveats: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "selected_portfolio_ids",
            "known_portfolio_ids",
            "active_portfolio_ids",
        ):
            values = tuple(
                _safe_portfolio_id(value, f"ledger {field_name}")
                for value in getattr(self, field_name)
            )
            object.__setattr__(self, field_name, values)
        _assert_public_safe(self.blockers, "ledger blockers")
        _assert_public_safe(self.caveats, "ledger caveats")


@dataclass(frozen=True)
class PositionComparison:
    symbol: str
    status: str
    reasons: tuple[str, ...]
    broker_contract: Optional[ContractIdentity]
    broker_quantity: Optional[Decimal]
    ledger_quantity: Optional[Decimal]
    broker_average_cost: Optional[Decimal]
    ledger_average_cost: Optional[Decimal]
    allocations: tuple[LedgerPosition, ...]

    @property
    def is_quantity_cost_difference(self) -> bool:
        return self.status not in {"quantity_cost_match"}


@dataclass(frozen=True)
class NonComparableEvidence:
    evidence_type: str
    broker_identifier: str
    symbol: str
    status: str
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        evidence_type = _safe_text(self.evidence_type, "evidence type")
        broker_identifier = _safe_text(self.broker_identifier, "broker evidence identity")
        symbol = self.symbol.strip().upper()
        status = _safe_text(self.status, "evidence status")
        reason = _safe_text(self.reason, "evidence reason")
        if not _TEXT_ID.fullmatch(evidence_type) or not _TEXT_ID.fullmatch(broker_identifier):
            raise BrokerEvidenceError("non-comparable evidence identity is invalid")
        if self.symbol != symbol or not _SYMBOL.fullmatch(symbol):
            raise BrokerEvidenceError("non-comparable evidence symbol is invalid")
        _assert_public_safe(self.details, "non-comparable evidence details")

        def freeze(value: object) -> object:
            if isinstance(value, Mapping):
                return MappingProxyType({str(key): freeze(nested) for key, nested in value.items()})
            if isinstance(value, (list, tuple)):
                return tuple(freeze(nested) for nested in value)
            return value

        frozen_details = freeze(self.details)
        if not isinstance(frozen_details, Mapping):
            raise BrokerEvidenceError("non-comparable evidence details are invalid")
        object.__setattr__(self, "evidence_type", evidence_type)
        object.__setattr__(self, "broker_identifier", broker_identifier)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "details", frozen_details)


@dataclass(frozen=True)
class ReconciliationReport:
    generated_at: datetime
    runtime_fingerprint: str
    database_identity: str
    account_alias: str
    selected_portfolio_ids: tuple[str, ...]
    status: str
    blockers: tuple[str, ...]
    caveats: tuple[str, ...]
    position_comparisons: tuple[PositionComparison, ...]
    open_order_comparisons: tuple[NonComparableEvidence, ...]
    execution_comparisons: tuple[NonComparableEvidence, ...]
    broker_snapshot: BrokerSnapshot
    ledger_snapshot: LedgerSnapshot
    mutated_state: bool = False
    authorizes_startup: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generated_at",
            aware_datetime(self.generated_at, "reconciliation report timestamp"),
        )
        runtime_fingerprint = _safe_text(
            self.runtime_fingerprint, "runtime fingerprint", max_length=256
        )
        database_identity = _safe_text(self.database_identity, "database identity", max_length=256)
        selected_portfolio_ids = tuple(
            _safe_portfolio_id(value, "selected portfolio identity")
            for value in self.selected_portfolio_ids
        )
        if self.status not in {
            "BLOCKED",
            "INCOMPLETE",
            "MISMATCH",
            "QUANTITY_COST_COMPARABLE_ONLY",
        }:
            raise BrokerEvidenceError("reconciliation report status is invalid")
        if (
            not self.account_alias.startswith("***")
            or len(self.account_alias) > 7
            or _ACCOUNT_FRAGMENT.search(self.account_alias)
        ):
            raise BrokerEvidenceError("reconciliation account alias is not masked")
        _assert_public_safe(self.blockers, "reconciliation blockers")
        _assert_public_safe(self.caveats, "reconciliation caveats")
        object.__setattr__(self, "runtime_fingerprint", runtime_fingerprint)
        object.__setattr__(self, "database_identity", database_identity)
        object.__setattr__(self, "selected_portfolio_ids", selected_portfolio_ids)

    def public_dict(self) -> dict[str, object]:
        def position_dict(item: PositionComparison) -> dict[str, object]:
            return {
                "symbol": item.symbol,
                "status": item.status,
                "reasons": list(item.reasons),
                "broker_contract": (
                    item.broker_contract.public_dict() if item.broker_contract else None
                ),
                "broker_quantity": (
                    canonical_decimal(item.broker_quantity)
                    if item.broker_quantity is not None
                    else None
                ),
                "ledger_quantity": (
                    canonical_decimal(item.ledger_quantity)
                    if item.ledger_quantity is not None
                    else None
                ),
                "broker_average_cost": (
                    canonical_decimal(item.broker_average_cost)
                    if item.broker_average_cost is not None
                    else None
                ),
                "ledger_average_cost": (
                    canonical_decimal(item.ledger_average_cost)
                    if item.ledger_average_cost is not None
                    else None
                ),
                "allocations": [
                    {
                        "portfolio_id": allocation.portfolio_id,
                        "quantity": canonical_decimal(allocation.quantity),
                        "average_cost": canonical_decimal(allocation.average_cost),
                        "timestamp": allocation.timestamp.isoformat(),
                    }
                    for allocation in item.allocations
                ],
            }

        def evidence_dict(item: NonComparableEvidence) -> dict[str, object]:
            def public_value(value: object) -> object:
                if isinstance(value, Mapping):
                    return {
                        str(key): public_value(nested)
                        for key, nested in sorted(value.items(), key=lambda pair: str(pair[0]))
                    }
                if isinstance(value, tuple):
                    return [public_value(nested) for nested in value]
                return value

            return {
                "evidence_type": item.evidence_type,
                "broker_identifier": item.broker_identifier,
                "symbol": item.symbol,
                "status": item.status,
                "reason": item.reason,
                "details": public_value(item.details),
            }

        return {
            "schema_version": 1,
            "generated_at": self.generated_at.isoformat(),
            "runtime_fingerprint": self.runtime_fingerprint,
            "database_identity": self.database_identity,
            "account_alias": self.account_alias,
            "selected_portfolio_ids": list(self.selected_portfolio_ids),
            "status": self.status,
            "mutated_state": self.mutated_state,
            "authorizes_startup": self.authorizes_startup,
            "blockers": list(self.blockers),
            "caveats": list(self.caveats),
            "broker_freshness": {
                "broker_time_before": self.broker_snapshot.broker_time_before.isoformat(),
                "broker_time_after": self.broker_snapshot.broker_time_after.isoformat(),
                "retrieved_at": self.broker_snapshot.retrieved_at.isoformat(),
                "execution_scope": self.broker_snapshot.execution_scope.public_dict(),
            },
            "broker_balances": {
                key: canonical_decimal(value)
                for key, value in sorted(self.broker_snapshot.balances.items())
            },
            "ledger_scope": {
                "known_portfolio_ids": list(self.ledger_snapshot.known_portfolio_ids),
                "active_portfolio_ids": list(self.ledger_snapshot.active_portfolio_ids),
            },
            "positions": [position_dict(item) for item in self.position_comparisons],
            "open_orders": [evidence_dict(item) for item in self.open_order_comparisons],
            "recent_executions": [evidence_dict(item) for item in self.execution_comparisons],
            "local_recent_trades": [
                {
                    "local_trade_id": trade.local_trade_id,
                    "portfolio_id": trade.portfolio_id,
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": canonical_decimal(trade.quantity),
                    "price": canonical_decimal(trade.price),
                    "timestamp": trade.timestamp.isoformat(),
                    "broker_execution_id": None,
                    "broker_order_id": None,
                }
                for trade in self.ledger_snapshot.recent_trades
            ],
        }


def as_tuple(values: Sequence[Any]) -> tuple[Any, ...]:
    """Small typed helper for adapters constructing immutable snapshots."""
    return tuple(values)

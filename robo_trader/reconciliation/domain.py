"""Immutable, versioned records for dormant broker reconciliation.

These records normalize read-only evidence only.  They carry no broker client,
database handle, order capability, or runtime activation authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from enum import Enum
from typing import Iterable

from .errors import ReconciliationError

DOMAIN_SCHEMA_VERSION = 1
PAPER_SIMULATOR_SCOPE = "paper-simulator-v1"
IBKR_READ_ONLY_SCOPE = "ibkr-read-only-v1"
MAX_COLLECTION_WINDOW_SECONDS = 60
MAX_RETRIEVAL_DELAY_SECONDS = 5

_ACCOUNT_SCOPE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_ACCOUNT_FRAGMENT = re.compile(r"(?:DU|U)\d{4,}", re.IGNORECASE)
_MASKED_ACCOUNT_ALIAS = re.compile(r"^\*{3}[A-Za-z0-9]{1,4}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_COLLECTION_EVIDENCE_ID = re.compile(r"^broker-collection-v1-[0-9a-f]{64}$")


class ReconciliationDomainError(ReconciliationError):
    """A normalized reconciliation record cannot be trusted."""


class ExecutionDomainScope(str, Enum):
    """Execution authorities that must never be silently conflated."""

    PAPER_SIMULATOR = PAPER_SIMULATOR_SCOPE
    IBKR_READ_ONLY = IBKR_READ_ONLY_SCOPE


class BrokerOrderCollection(str, Enum):
    """The broker collection in which an order was observed."""

    OPEN = "open"
    COMPLETED = "completed"


class BrokerOrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class BrokerCollectionKind(str, Enum):
    """Broker collections whose empty results require explicit evidence."""

    POSITIONS = "positions"
    OPEN_ORDERS = "open_orders"
    COMPLETED_ORDERS = "completed_orders"
    EXECUTIONS = "executions"
    COMMISSIONS = "commissions"


def _strict_text(value: object, field_name: str, *, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or value != value.strip() or not pattern.fullmatch(value):
        raise ReconciliationDomainError(f"{field_name} is malformed")
    if _ACCOUNT_FRAGMENT.search(value):
        raise ReconciliationDomainError(f"{field_name} contains raw account identity")
    return value


def _account_scope(value: object) -> str:
    normalized = _strict_text(value, "account_scope", pattern=_ACCOUNT_SCOPE)
    if len(set(normalized.removeprefix("acct_v1_"))) == 1:
        raise ReconciliationDomainError("account_scope uses a placeholder digest")
    return normalized


def _timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ReconciliationDomainError(f"{field_name} must be timezone-aware")
    try:
        normalized = value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise ReconciliationDomainError(f"{field_name} is invalid") from exc
    return normalized


def _decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if isinstance(value, bool) or isinstance(value, float):
        raise ReconciliationDomainError(
            f"{field_name} must be an exact decimal, not a binary float"
        )
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ReconciliationDomainError(f"{field_name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ReconciliationDomainError(f"{field_name} must be a finite decimal")
    if positive and parsed <= 0:
        raise ReconciliationDomainError(f"{field_name} must be positive")
    if nonnegative and parsed < 0:
        raise ReconciliationDomainError(f"{field_name} must be nonnegative")
    return parsed


def _schema_version(value: object, label: str) -> int:
    if type(value) is not int or value != DOMAIN_SCHEMA_VERSION:
        raise ReconciliationDomainError(f"{label} schema version is unsupported")
    return value


def _arithmetic_precision(*values: Decimal) -> int:
    """Return enough precision to align and add all finite decimal operands exactly."""

    highest_place = max(value.adjusted() for value in values)
    exponents = tuple(value.as_tuple().exponent for value in values)
    if any(type(exponent) is not int for exponent in exponents):
        raise ReconciliationDomainError("decimal arithmetic requires finite operands")
    lowest_place = min(exponent for exponent in exponents if isinstance(exponent, int))
    return max(1, highest_place - lowest_place + 2)


def _exact_context(precision: int) -> Context:
    """Build a context that cannot inherit ambient decimal process state."""

    return Context(
        prec=precision,
        rounding=ROUND_HALF_EVEN,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def canonical_decimal(value: Decimal) -> str:
    """Serialize an exact decimal without exponent or negative-zero drift."""

    if type(value) is not Decimal or not value.is_finite():
        raise ReconciliationDomainError("canonical decimal must be finite and exact")
    if value.is_zero():
        return "0"
    sign, digits, exponent = value.as_tuple()
    if type(exponent) is not int:
        raise ReconciliationDomainError("canonical decimal must have a finite exponent")
    coefficient = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        rendered = coefficient + ("0" * exponent)
    else:
        split_at = len(coefficient) + exponent
        if split_at <= 0:
            rendered = "0." + ("0" * -split_at) + coefficient
        else:
            rendered = coefficient[:split_at] + "." + coefficient[split_at:]
        rendered = rendered.rstrip("0").rstrip(".")
    return ("-" if sign else "") + rendered


def canonical_timestamp(value: datetime) -> str:
    """Serialize one normalized UTC instant deterministically."""

    normalized = _timestamp(value, "timestamp")
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(payload: object) -> str:
    """Return the sole JSON encoding admitted to evidence fingerprints."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint(prefix: str, payload: object) -> str:
    """Build a namespaced SHA-256 identity from canonical public evidence."""

    normalized_prefix = _strict_text(prefix, "fingerprint prefix", pattern=_IDENTIFIER)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{normalized_prefix}-{digest}"


def _positive_identifier(value: object, field_name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ReconciliationDomainError(f"{field_name} must be a {qualifier} integer")
    return value


@dataclass(frozen=True, slots=True)
class NormalizedBrokerAccount:
    """Masked, exact account evidence from an IBKR read-only session."""

    account_scope: str
    account_alias: str
    account_type: str
    base_currency: str
    total_cash: Decimal
    buying_power: Decimal
    observed_at: datetime
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker account")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker account source is not read-only IBKR")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        object.__setattr__(
            self,
            "account_alias",
            _strict_text(
                self.account_alias,
                "account_alias",
                pattern=_MASKED_ACCOUNT_ALIAS,
            ),
        )
        if self.account_type != "paper":
            raise ReconciliationDomainError("broker account type is not paper")
        object.__setattr__(
            self,
            "base_currency",
            _strict_text(self.base_currency, "base_currency", pattern=_CURRENCY),
        )
        object.__setattr__(self, "total_cash", _decimal(self.total_cash, "total_cash"))
        object.__setattr__(
            self,
            "buying_power",
            _decimal(self.buying_power, "buying_power", nonnegative=True),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "account observed_at"),
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_alias": self.account_alias,
            "account_scope": self.account_scope,
            "account_type": self.account_type,
            "base_currency": self.base_currency,
            "buying_power": canonical_decimal(self.buying_power),
            "observed_at": canonical_timestamp(self.observed_at),
            "schema_version": self.schema_version,
            "source_scope": self.source_scope.value,
            "total_cash": canonical_decimal(self.total_cash),
        }


@dataclass(frozen=True, slots=True)
class NormalizedBrokerPosition:
    """One signed, qualified broker position."""

    account_scope: str
    con_id: int
    symbol: str
    currency: str
    signed_quantity: Decimal
    average_cost: Decimal
    observed_at: datetime
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker position")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker position source is not read-only IBKR")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        object.__setattr__(self, "con_id", _positive_identifier(self.con_id, "con_id"))
        object.__setattr__(
            self,
            "symbol",
            _strict_text(self.symbol, "symbol", pattern=_SYMBOL),
        )
        object.__setattr__(
            self,
            "currency",
            _strict_text(self.currency, "currency", pattern=_CURRENCY),
        )
        quantity = _decimal(self.signed_quantity, "signed_quantity")
        if quantity == 0:
            raise ReconciliationDomainError("broker position quantity cannot be zero")
        object.__setattr__(self, "signed_quantity", quantity)
        object.__setattr__(
            self,
            "average_cost",
            _decimal(self.average_cost, "average_cost", nonnegative=True),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "position observed_at"),
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "average_cost": canonical_decimal(self.average_cost),
            "con_id": self.con_id,
            "currency": self.currency,
            "observed_at": canonical_timestamp(self.observed_at),
            "schema_version": self.schema_version,
            "signed_quantity": canonical_decimal(self.signed_quantity),
            "source_scope": self.source_scope.value,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class NormalizedBrokerOrder:
    """One open or completed broker order with exact lifecycle identity."""

    account_scope: str
    collection: BrokerOrderCollection
    broker_order_id: int
    client_id: int
    con_id: int
    symbol: str
    side: BrokerOrderSide
    total_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    status: str
    observed_at: datetime
    permanent_id: int | None = None
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker order")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker order source is not read-only IBKR")
        if type(self.collection) is not BrokerOrderCollection:
            raise ReconciliationDomainError("broker order collection is invalid")
        if type(self.side) is not BrokerOrderSide:
            raise ReconciliationDomainError("broker order side is invalid")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        object.__setattr__(
            self,
            "broker_order_id",
            _positive_identifier(self.broker_order_id, "broker_order_id"),
        )
        object.__setattr__(
            self,
            "client_id",
            _positive_identifier(self.client_id, "client_id", allow_zero=True),
        )
        object.__setattr__(self, "con_id", _positive_identifier(self.con_id, "con_id"))
        object.__setattr__(
            self,
            "symbol",
            _strict_text(self.symbol, "symbol", pattern=_SYMBOL),
        )
        total = _decimal(self.total_quantity, "total_quantity", positive=True)
        filled = _decimal(self.filled_quantity, "filled_quantity", nonnegative=True)
        remaining = _decimal(self.remaining_quantity, "remaining_quantity", nonnegative=True)
        with localcontext(_exact_context(_arithmetic_precision(total, filled, remaining))):
            quantities_match = filled + remaining == total
        if not quantities_match:
            raise ReconciliationDomainError("broker order quantities are inconsistent")
        object.__setattr__(self, "total_quantity", total)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "remaining_quantity", remaining)
        object.__setattr__(
            self,
            "status",
            _strict_text(self.status, "order status", pattern=_IDENTIFIER),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "order observed_at"),
        )
        if self.permanent_id is not None:
            object.__setattr__(
                self,
                "permanent_id",
                _positive_identifier(self.permanent_id, "permanent_id"),
            )

    @property
    def identity(self) -> tuple[int, int]:
        return (self.client_id, self.broker_order_id)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "broker_order_id": self.broker_order_id,
            "client_id": self.client_id,
            "collection": self.collection.value,
            "con_id": self.con_id,
            "filled_quantity": canonical_decimal(self.filled_quantity),
            "observed_at": canonical_timestamp(self.observed_at),
            "permanent_id": self.permanent_id,
            "remaining_quantity": canonical_decimal(self.remaining_quantity),
            "schema_version": self.schema_version,
            "side": self.side.value,
            "source_scope": self.source_scope.value,
            "status": self.status,
            "symbol": self.symbol,
            "total_quantity": canonical_decimal(self.total_quantity),
        }


@dataclass(frozen=True, slots=True)
class NormalizedBrokerExecution:
    """One broker execution with explicit commission availability."""

    account_scope: str
    execution_id: str
    con_id: int
    symbol: str
    side: BrokerOrderSide
    quantity: Decimal
    price: Decimal
    executed_at: datetime
    broker_order_id: int | None = None
    permanent_id: int | None = None
    commission: Decimal | None = None
    commission_currency: str | None = None
    commission_unavailable_reason: str | None = None
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker execution")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker execution source is not read-only IBKR")
        if type(self.side) is not BrokerOrderSide:
            raise ReconciliationDomainError("broker execution side is invalid")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        object.__setattr__(
            self,
            "execution_id",
            _strict_text(self.execution_id, "execution_id", pattern=_IDENTIFIER),
        )
        object.__setattr__(self, "con_id", _positive_identifier(self.con_id, "con_id"))
        object.__setattr__(
            self,
            "symbol",
            _strict_text(self.symbol, "symbol", pattern=_SYMBOL),
        )
        object.__setattr__(self, "quantity", _decimal(self.quantity, "quantity", positive=True))
        object.__setattr__(self, "price", _decimal(self.price, "price", positive=True))
        object.__setattr__(
            self,
            "executed_at",
            _timestamp(self.executed_at, "execution executed_at"),
        )
        for field_name in ("broker_order_id", "permanent_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _positive_identifier(value, field_name),
                )
        if self.commission is None:
            if self.commission_currency is not None or self.commission_unavailable_reason is None:
                raise ReconciliationDomainError(
                    "missing commission requires exactly one unavailable reason"
                )
            object.__setattr__(
                self,
                "commission_unavailable_reason",
                _strict_text(
                    self.commission_unavailable_reason,
                    "commission unavailable reason",
                    pattern=_IDENTIFIER,
                ),
            )
        else:
            if self.commission_currency is None or self.commission_unavailable_reason is not None:
                raise ReconciliationDomainError(
                    "available commission requires currency and no unavailable reason"
                )
            object.__setattr__(
                self,
                "commission",
                _decimal(self.commission, "commission"),
            )
            object.__setattr__(
                self,
                "commission_currency",
                _strict_text(
                    self.commission_currency,
                    "commission_currency",
                    pattern=_CURRENCY,
                ),
            )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "broker_order_id": self.broker_order_id,
            "commission": (None if self.commission is None else canonical_decimal(self.commission)),
            "commission_currency": self.commission_currency,
            "commission_unavailable_reason": self.commission_unavailable_reason,
            "con_id": self.con_id,
            "executed_at": canonical_timestamp(self.executed_at),
            "execution_id": self.execution_id,
            "permanent_id": self.permanent_id,
            "price": canonical_decimal(self.price),
            "quantity": canonical_decimal(self.quantity),
            "schema_version": self.schema_version,
            "side": self.side.value,
            "source_scope": self.source_scope.value,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class BrokerCollectionEvidence:
    """Bounded proof that one read-only broker collection was actually queried."""

    account_scope: str
    collection: BrokerCollectionKind
    evidence_id: str
    result_count: int
    observed_at: datetime
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker collection evidence")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker collection evidence is not read-only IBKR")
        if type(self.collection) is not BrokerCollectionKind:
            raise ReconciliationDomainError("broker collection evidence kind is invalid")
        object.__setattr__(self, "account_scope", _account_scope(self.account_scope))
        object.__setattr__(
            self,
            "evidence_id",
            _strict_text(
                self.evidence_id,
                "broker collection evidence_id",
                pattern=_COLLECTION_EVIDENCE_ID,
            ),
        )
        object.__setattr__(
            self,
            "result_count",
            _positive_identifier(self.result_count, "result_count", allow_zero=True),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "collection evidence observed_at"),
        )

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account_scope": self.account_scope,
            "collection": self.collection.value,
            "evidence_id": self.evidence_id,
            "observed_at": canonical_timestamp(self.observed_at),
            "result_count": self.result_count,
            "schema_version": self.schema_version,
            "source_scope": self.source_scope.value,
        }


@dataclass(frozen=True, slots=True)
class BrokerEvidenceCompleteness:
    """Explicit producer claims for each required broker collection."""

    account: bool
    positions: bool
    open_orders: bool
    completed_orders: bool
    executions: bool
    commissions: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, field_name)) is not bool for field_name in self.__slots__):
            raise ReconciliationDomainError("broker completeness flags must be exact booleans")

    @property
    def complete(self) -> bool:
        return all(getattr(self, field_name) for field_name in self.__slots__)

    def canonical_dict(self) -> dict[str, bool]:
        return {field_name: getattr(self, field_name) for field_name in self.__slots__}


def _require_unique(values: Iterable[object], label: str) -> None:
    frozen = tuple(values)
    if len(frozen) != len(set(frozen)):
        raise ReconciliationDomainError(f"broker snapshot contains duplicate {label}")


@dataclass(frozen=True, slots=True)
class NormalizedBrokerSnapshot:
    """One immutable, bounded IBKR read-only evidence generation."""

    account: NormalizedBrokerAccount
    observed_from: datetime
    observed_through: datetime
    retrieved_at: datetime
    completeness: BrokerEvidenceCompleteness
    collection_evidence: tuple[BrokerCollectionEvidence, ...]
    positions: tuple[NormalizedBrokerPosition, ...] = ()
    orders: tuple[NormalizedBrokerOrder, ...] = ()
    executions: tuple[NormalizedBrokerExecution, ...] = ()
    schema_version: int = DOMAIN_SCHEMA_VERSION
    source_scope: ExecutionDomainScope = ExecutionDomainScope.IBKR_READ_ONLY

    def __post_init__(self) -> None:
        _schema_version(self.schema_version, "broker snapshot")
        if self.source_scope is not ExecutionDomainScope.IBKR_READ_ONLY:
            raise ReconciliationDomainError("broker snapshot source is not read-only IBKR")
        if type(self.account) is not NormalizedBrokerAccount:
            raise ReconciliationDomainError("broker snapshot account is not normalized")
        if type(self.completeness) is not BrokerEvidenceCompleteness:
            raise ReconciliationDomainError("broker snapshot completeness is malformed")
        for field_name in ("observed_from", "observed_through", "retrieved_at"):
            object.__setattr__(self, field_name, _timestamp(getattr(self, field_name), field_name))
        if self.observed_through < self.observed_from:
            raise ReconciliationDomainError("broker snapshot observation window is reversed")
        if (
            self.observed_through - self.observed_from
        ).total_seconds() > MAX_COLLECTION_WINDOW_SECONDS:
            raise ReconciliationDomainError("broker snapshot observation window is unbounded")
        if self.retrieved_at < self.observed_through:
            raise ReconciliationDomainError(
                "broker snapshot retrieval predates completed observation"
            )
        if (
            self.retrieved_at - self.observed_through
        ).total_seconds() > MAX_RETRIEVAL_DELAY_SECONDS:
            raise ReconciliationDomainError("broker snapshot retrieval delay is unbounded")
        for field_name, expected_type in (
            ("collection_evidence", BrokerCollectionEvidence),
            ("positions", NormalizedBrokerPosition),
            ("orders", NormalizedBrokerOrder),
            ("executions", NormalizedBrokerExecution),
        ):
            values = tuple(getattr(self, field_name))
            if any(type(value) is not expected_type for value in values):
                raise ReconciliationDomainError(f"broker snapshot {field_name} are not normalized")
            object.__setattr__(self, field_name, values)
        all_records = (
            (self.account,)
            + self.collection_evidence
            + self.positions
            + self.orders
            + self.executions
        )
        if any(record.account_scope != self.account.account_scope for record in all_records):
            raise ReconciliationDomainError("broker snapshot spans multiple account scopes")
        if any(
            record.observed_at < self.observed_from or record.observed_at > self.observed_through
            for record in (
                (self.account,) + self.collection_evidence + self.positions + self.orders
            )
        ):
            raise ReconciliationDomainError("broker evidence falls outside its observation window")
        if any(execution.executed_at > self.observed_through for execution in self.executions):
            raise ReconciliationDomainError("broker execution is later than its snapshot")
        _require_unique((position.con_id for position in self.positions), "position conId")
        _require_unique((position.symbol for position in self.positions), "position symbol")
        _require_unique(
            ((order.collection.value, order.identity) for order in self.orders),
            "order identity within one collection",
        )
        permanent_ids = tuple(
            (order.collection.value, order.permanent_id)
            for order in self.orders
            if order.permanent_id is not None
        )
        _require_unique(permanent_ids, "permanent order identity within one collection")
        _require_unique(
            (execution.execution_id for execution in self.executions),
            "execution identity",
        )
        _require_unique(
            (evidence.collection for evidence in self.collection_evidence),
            "collection evidence kind",
        )
        _require_unique(
            (evidence.evidence_id for evidence in self.collection_evidence),
            "collection evidence identity",
        )
        evidence_by_kind = {evidence.collection: evidence for evidence in self.collection_evidence}
        expected_counts = {
            BrokerCollectionKind.POSITIONS: len(self.positions),
            BrokerCollectionKind.OPEN_ORDERS: len(self.open_orders),
            BrokerCollectionKind.COMPLETED_ORDERS: len(self.completed_orders),
            BrokerCollectionKind.EXECUTIONS: len(self.executions),
            BrokerCollectionKind.COMMISSIONS: len(self.executions),
        }
        completeness_by_kind = {
            BrokerCollectionKind.POSITIONS: self.completeness.positions,
            BrokerCollectionKind.OPEN_ORDERS: self.completeness.open_orders,
            BrokerCollectionKind.COMPLETED_ORDERS: self.completeness.completed_orders,
            BrokerCollectionKind.EXECUTIONS: self.completeness.executions,
            BrokerCollectionKind.COMMISSIONS: self.completeness.commissions,
        }
        for kind, is_complete in completeness_by_kind.items():
            evidence = evidence_by_kind.get(kind)
            if is_complete and evidence is None:
                raise ReconciliationDomainError(
                    f"complete {kind.value} collection lacks explicit evidence"
                )
            if not is_complete and evidence is not None:
                raise ReconciliationDomainError(
                    f"incomplete {kind.value} collection contradicts explicit evidence"
                )
            if evidence is not None and evidence.result_count != expected_counts[kind]:
                raise ReconciliationDomainError(
                    f"{kind.value} collection evidence count is inconsistent"
                )
        if self.completeness.commissions and any(
            execution.commission is None for execution in self.executions
        ):
            raise ReconciliationDomainError(
                "complete commissions collection contains unavailable commission"
            )

    @property
    def open_orders(self) -> tuple[NormalizedBrokerOrder, ...]:
        return tuple(
            order for order in self.orders if order.collection is BrokerOrderCollection.OPEN
        )

    @property
    def completed_orders(self) -> tuple[NormalizedBrokerOrder, ...]:
        return tuple(
            order for order in self.orders if order.collection is BrokerOrderCollection.COMPLETED
        )

    def is_fresh(self, *, now: datetime, max_age_seconds: float) -> bool:
        checked_at = _timestamp(now, "freshness clock")
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(float(max_age_seconds))
            or max_age_seconds <= 0
        ):
            raise ReconciliationDomainError("freshness bound must be finite and positive")
        age = (checked_at - self.retrieved_at).total_seconds()
        return 0 <= age <= float(max_age_seconds)

    def canonical_dict(self) -> dict[str, object]:
        return {
            "account": self.account.canonical_dict(),
            "completeness": self.completeness.canonical_dict(),
            "collection_evidence": [
                evidence.canonical_dict()
                for evidence in sorted(
                    self.collection_evidence,
                    key=lambda value: value.collection.value,
                )
            ],
            "executions": [
                execution.canonical_dict()
                for execution in sorted(self.executions, key=lambda value: value.execution_id)
            ],
            "observed_from": canonical_timestamp(self.observed_from),
            "observed_through": canonical_timestamp(self.observed_through),
            "orders": [
                order.canonical_dict()
                for order in sorted(
                    self.orders,
                    key=lambda value: (
                        value.collection.value,
                        value.client_id,
                        value.broker_order_id,
                    ),
                )
            ],
            "positions": [
                position.canonical_dict()
                for position in sorted(self.positions, key=lambda value: value.con_id)
            ],
            "retrieved_at": canonical_timestamp(self.retrieved_at),
            "schema_version": self.schema_version,
            "source_scope": self.source_scope.value,
        }

    def canonical_payload(self) -> str:
        return canonical_json(self.canonical_dict())

    @property
    def snapshot_id(self) -> str:
        return fingerprint("broker-reconciliation-v1", self.canonical_dict())

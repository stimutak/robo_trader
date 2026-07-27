"""Immutable, exact-value models for the dormant order-safety core.

This module deliberately contains no configuration, broker, database, or runtime
imports.  Values crossing the safety boundary are strict: quantities are
``Decimal`` instances (never floats), timestamps are UTC, and broker account
numbers are rejected in every caller-controlled text field.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, Inexact, InvalidOperation, Rounded, localcontext
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

MODEL_VERSION = 1
MAX_DECIMAL_DIGITS = 28
MAX_DECIMAL_SCALE = 12
_SAFETY_DECIMAL_PRECISION = MAX_DECIMAL_DIGITS + MAX_DECIMAL_SCALE + 1
SAFETY_MAX_EVIDENCE_AGE_SECONDS = 30
_BROKER_ACCOUNT_RE = re.compile(r"(?i)(?:DU|DF|U|F)\d{4,}")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_SCOPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_OPAQUE_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_DATABASE_IDENTITY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}:[0-9a-f]{12}$")


class ValidationError(ValueError):
    """A value cannot safely cross the safety-core boundary."""


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    BUY_TO_COVER = "BUY_TO_COVER"
    SELL_SHORT = "SELL_SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"


class TerminalOrderStatus(str, Enum):
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    NO_SUBMISSION_CONFIRMED = "NO_SUBMISSION_CONFIRMED"


class EvidenceStatus(str, Enum):
    AUTHORITATIVE = "AUTHORITATIVE"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class TransportState(str, Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    AMBIGUOUS = "AMBIGUOUS"


class ReconciliationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class DecisionOutcome(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class RiskEffect(str, Enum):
    REDUCING = "REDUCING"
    INCREASING = "INCREASING"
    UNKNOWN = "UNKNOWN"


class JournalEventType(str, Enum):
    SAFETY_DECISION = "SAFETY_DECISION"
    RESERVATION_ACQUIRED = "RESERVATION_ACQUIRED"
    SUBMISSION_STARTED = "SUBMISSION_STARTED"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    TERMINAL_RECONCILED = "TERMINAL_RECONCILED"


def _strict_version(value: object) -> int:
    if type(value) is not int or value != MODEL_VERSION:
        raise ValidationError(f"schema_version must be exactly {MODEL_VERSION}")
    return value


def _strict_text(
    value: object,
    field: str,
    *,
    max_length: int = 256,
    allow_empty: bool = False,
    pattern: Optional[re.Pattern[str]] = None,
) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{field} must be a string")
    if value != value.strip():
        raise ValidationError(f"{field} must not contain surrounding whitespace")
    if not value and not allow_empty:
        raise ValidationError(f"{field} must not be empty")
    if len(value) > max_length:
        raise ValidationError(f"{field} exceeds {max_length} characters")
    if "\x00" in value:
        raise ValidationError(f"{field} contains a NUL byte")
    if _BROKER_ACCOUNT_RE.search(value):
        raise ValidationError(f"{field} must not contain a raw broker account number")
    if field.endswith("_scope") and value.isdecimal():
        raise ValidationError(f"{field} must be opaque, not a numeric account identifier")
    if pattern is not None and not pattern.fullmatch(value):
        raise ValidationError(f"{field} has an invalid format")
    return value


def _strict_database_identity(value: object) -> str:
    """Validate the runtime contract's opaque path-hash identity.

    The final component is hexadecimal, so a legitimate digest can contain a
    substring such as ``f12345`` that resembles an IBKR account.  Validate the
    complete producer-owned format instead of applying the free-text secret
    scanner to an already-opaque digest.
    """

    if not isinstance(value, str):
        raise ValidationError("database_identity must be a string")
    if not _DATABASE_IDENTITY_RE.fullmatch(value):
        raise ValidationError("database_identity must be an opaque path-hash identity")
    return value


def _strict_decimal(
    value: object,
    field: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if type(value) is not Decimal:
        raise ValidationError(f"{field} must be an exact Decimal")
    if not value.is_finite():
        raise ValidationError(f"{field} must be finite")
    sign, digits, exponent = value.as_tuple()
    if exponent > 0:
        raise ValidationError(f"{field} must not use positive exponent notation")
    if len(digits) > MAX_DECIMAL_DIGITS:
        raise ValidationError(f"{field} exceeds {MAX_DECIMAL_DIGITS} digits")
    if exponent < -MAX_DECIMAL_SCALE:
        raise ValidationError(f"{field} exceeds scale {MAX_DECIMAL_SCALE}")
    if sign and value.is_zero():
        raise ValidationError(f"{field} must not be signed zero")
    if positive and value <= 0:
        raise ValidationError(f"{field} must be positive")
    if nonnegative and value < 0:
        raise ValidationError(f"{field} must be nonnegative")
    return value


def _exact_decimal_add(left: Decimal, right: Decimal, field: str) -> Decimal:
    """Add validated values exactly, independent of the ambient Decimal context."""

    _strict_decimal(left, f"{field} left operand")
    _strict_decimal(right, f"{field} right operand")
    try:
        with localcontext() as context:
            context.prec = _SAFETY_DECIMAL_PRECISION
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            result = left + right
    except (Inexact, Rounded, InvalidOperation) as exc:
        raise ValidationError(f"{field} arithmetic was not exact") from exc
    return _strict_decimal(result, field)


def _exact_decimal_subtract(left: Decimal, right: Decimal, field: str) -> Decimal:
    """Subtract validated values exactly, independent of ambient precision."""

    negated = right if right.is_zero() else right.copy_negate()
    return _exact_decimal_add(left, negated, field)


def _strict_account_scope(value: object) -> str:
    if not isinstance(value, str) or not _OPAQUE_ACCOUNT_SCOPE_RE.fullmatch(value):
        raise ValidationError("account_scope must be an opaque acct_v1_<64 lowercase hex> value")
    return value


def _strict_internal_id(value: object, field: str, prefix: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        rf"{re.escape(prefix)}-[0-9a-f]{{32}}", value
    ):
        raise ValidationError(f"{field} must be an opaque {prefix} identifier")
    return value


def _strict_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _strict_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValidationError(f"{field} must be a positive integer")
    return value


def _strict_utc(value: object, field: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValidationError(f"{field} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field} must be timezone-aware UTC")
    if value.utcoffset().total_seconds() != 0:
        raise ValidationError(f"{field} must be UTC, not a local timestamp")
    return value.astimezone(timezone.utc)


def _strict_enum(value: object, enum_type: type[Enum], field: str) -> Enum:
    if type(value) is not enum_type:
        raise ValidationError(f"{field} must be a {enum_type.__name__}")
    return value


def decimal_to_fixed(value: Decimal) -> str:
    """Serialize a validated Decimal without exponent notation."""

    _strict_decimal(value, "decimal")
    rendered = format(value, "f")
    if "e" in rendered.lower():
        raise ValidationError("Decimal serialization must not use exponent notation")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if rendered in {"", "-0"}:
        rendered = "0"
    return rendered


def utc_to_text(value: datetime) -> str:
    value = _strict_utc(value, "timestamp")
    rendered = value.isoformat(timespec="microseconds")
    return rendered.replace("+00:00", "Z")


def parse_fixed_decimal(value: object, field: str = "decimal") -> Decimal:
    """Parse journal JSON decimal text while rejecting ambiguous encodings."""

    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty fixed-point string")
    if "e" in value.lower() or value.startswith("+") or value != value.strip():
        raise ValidationError(f"{field} must not use exponent or ambiguous notation")
    if not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?", value):
        raise ValidationError(f"{field} must be canonical fixed-point text")
    parsed = Decimal(value)
    _strict_decimal(parsed, field)
    if decimal_to_fixed(parsed) != value:
        raise ValidationError(f"{field} is not canonical")
    return parsed


def parse_utc_text(value: object, field: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValidationError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValidationError(f"{field} is not a valid UTC timestamp") from exc
    if utc_to_text(parsed) != value:
        raise ValidationError(f"{field} is not canonical")
    return parsed


def canonical_json(value: Mapping[str, Any]) -> str:
    """Return deterministic JSON, rejecting unsupported or inexact values."""

    def normalize(item: Any) -> Any:
        if item is None or type(item) in {str, int, bool}:
            return item
        if type(item) is Decimal:
            return decimal_to_fixed(item)
        if isinstance(item, datetime):
            return utc_to_text(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, Mapping):
            return {str(key): normalize(subvalue) for key, subvalue in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(subvalue) for subvalue in item]
        raise ValidationError(f"unsupported canonical JSON type: {type(item).__name__}")

    normalized = normalize(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OrderIntent:
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    side: OrderSide
    quantity: Decimal
    account_current_quantity: Decimal
    target_quantity: Decimal
    portfolio_current_quantity: Decimal
    portfolio_target_quantity: Decimal
    created_at: datetime
    reduce_only: bool = False
    reason: str = ""
    strategy: str = ""
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_enum(self.side, OrderSide, "side")
        _strict_decimal(self.quantity, "quantity", positive=True)
        _strict_decimal(self.account_current_quantity, "account_current_quantity")
        _strict_decimal(self.target_quantity, "target_quantity")
        _strict_decimal(self.portfolio_current_quantity, "portfolio_current_quantity")
        _strict_decimal(self.portfolio_target_quantity, "portfolio_target_quantity")
        object.__setattr__(self, "created_at", _strict_utc(self.created_at, "created_at"))
        if type(self.reduce_only) is not bool:
            raise ValidationError("reduce_only must be a bool")
        _strict_text(self.reason, "reason", max_length=256, allow_empty=True)
        _strict_text(self.strategy, "strategy", max_length=128, allow_empty=True)

    def authorization_payload(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "account_scope": self.account_scope,
                "account_current_quantity": self.account_current_quantity,
                "con_id": self.con_id,
                "created_at": self.created_at,
                "execution_domain_scope": self.execution_domain_scope,
                "portfolio_id": self.portfolio_id,
                "portfolio_current_quantity": self.portfolio_current_quantity,
                "portfolio_target_quantity": self.portfolio_target_quantity,
                "quantity": self.quantity,
                "reason": self.reason,
                "reduce_only": self.reduce_only,
                "schema_version": self.schema_version,
                "side": self.side,
                "strategy": self.strategy,
                "symbol": self.symbol,
                "target_quantity": self.target_quantity,
            }
        )

    def canonical_payload(self) -> str:
        return canonical_json(self.authorization_payload())

    def fingerprint(self) -> str:
        return sha256_text(self.canonical_payload())


@dataclass(frozen=True)
class ExposureEvidence:
    execution_domain_scope: str
    account_scope: str
    con_id: int
    symbol: str
    position_quantity: Decimal
    observed_at: datetime
    status: EvidenceStatus
    source: str
    snapshot_id: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_decimal(self.position_quantity, "position_quantity")
        object.__setattr__(self, "observed_at", _strict_utc(self.observed_at, "observed_at"))
        _strict_enum(self.status, EvidenceStatus, "status")
        _strict_text(self.source, "source", max_length=128)
        _strict_text(self.snapshot_id, "snapshot_id", max_length=128, pattern=_SCOPE_RE)

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "con_id": self.con_id,
                "execution_domain_scope": self.execution_domain_scope,
                "observed_at": self.observed_at,
                "position_quantity": self.position_quantity,
                "schema_version": self.schema_version,
                "snapshot_id": self.snapshot_id,
                "source": self.source,
                "status": self.status,
                "symbol": self.symbol,
            }
        )


@dataclass(frozen=True)
class PortfolioAllocationEvidence:
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    position_quantity: Decimal
    aggregate_allocated_quantity: Decimal
    has_offsetting_allocations: bool
    observed_at: datetime
    status: EvidenceStatus
    source: str
    snapshot_id: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_decimal(self.position_quantity, "position_quantity")
        _strict_decimal(self.aggregate_allocated_quantity, "aggregate_allocated_quantity")
        if type(self.has_offsetting_allocations) is not bool:
            raise ValidationError("has_offsetting_allocations must be a bool")
        object.__setattr__(self, "observed_at", _strict_utc(self.observed_at, "observed_at"))
        _strict_enum(self.status, EvidenceStatus, "status")
        _strict_text(self.source, "source", max_length=128)
        _strict_text(self.snapshot_id, "snapshot_id", max_length=128, pattern=_SCOPE_RE)

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "aggregate_allocated_quantity": self.aggregate_allocated_quantity,
                "con_id": self.con_id,
                "execution_domain_scope": self.execution_domain_scope,
                "has_offsetting_allocations": self.has_offsetting_allocations,
                "observed_at": self.observed_at,
                "portfolio_id": self.portfolio_id,
                "position_quantity": self.position_quantity,
                "schema_version": self.schema_version,
                "snapshot_id": self.snapshot_id,
                "source": self.source,
                "status": self.status,
                "symbol": self.symbol,
            }
        )


@dataclass(frozen=True)
class GateContext:
    execution_domain_scope: str
    account_scope: str
    con_id: int
    evaluated_at: datetime
    max_evidence_age_seconds: int
    transport_state: TransportState
    reconciliation_status: ReconciliationStatus
    open_orders_complete: bool
    open_orders_all_clients: bool
    open_orders_snapshot_stable: bool
    open_orders_observed_at: datetime
    open_orders_snapshot_id: str
    active_order_count: int
    soft_entry_allowed: bool = True
    hard_block_reasons: Tuple[str, ...] = ()
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        object.__setattr__(self, "evaluated_at", _strict_utc(self.evaluated_at, "evaluated_at"))
        if (
            type(self.max_evidence_age_seconds) is not int
            or self.max_evidence_age_seconds < 0
            or self.max_evidence_age_seconds > SAFETY_MAX_EVIDENCE_AGE_SECONDS
        ):
            raise ValidationError(
                "max_evidence_age_seconds exceeds the safety-owned 30-second ceiling"
            )
        _strict_enum(self.transport_state, TransportState, "transport_state")
        _strict_enum(self.reconciliation_status, ReconciliationStatus, "reconciliation_status")
        if type(self.open_orders_complete) is not bool:
            raise ValidationError("open_orders_complete must be a bool")
        if type(self.open_orders_all_clients) is not bool:
            raise ValidationError("open_orders_all_clients must be a bool")
        if type(self.open_orders_snapshot_stable) is not bool:
            raise ValidationError("open_orders_snapshot_stable must be a bool")
        object.__setattr__(
            self,
            "open_orders_observed_at",
            _strict_utc(self.open_orders_observed_at, "open_orders_observed_at"),
        )
        _strict_text(
            self.open_orders_snapshot_id,
            "open_orders_snapshot_id",
            max_length=128,
            pattern=_SCOPE_RE,
        )
        if type(self.active_order_count) is not int or self.active_order_count < 0:
            raise ValidationError("active_order_count must be a nonnegative integer")
        if type(self.soft_entry_allowed) is not bool:
            raise ValidationError("soft_entry_allowed must be a bool")
        if not isinstance(self.hard_block_reasons, tuple):
            raise ValidationError("hard_block_reasons must be a tuple")
        for reason in self.hard_block_reasons:
            _strict_text(reason, "hard_block_reason", max_length=128)

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "active_order_count": self.active_order_count,
                "con_id": self.con_id,
                "evaluated_at": self.evaluated_at,
                "execution_domain_scope": self.execution_domain_scope,
                "hard_block_reasons": self.hard_block_reasons,
                "max_evidence_age_seconds": self.max_evidence_age_seconds,
                "open_orders_all_clients": self.open_orders_all_clients,
                "open_orders_complete": self.open_orders_complete,
                "open_orders_observed_at": self.open_orders_observed_at,
                "open_orders_snapshot_id": self.open_orders_snapshot_id,
                "open_orders_snapshot_stable": self.open_orders_snapshot_stable,
                "reconciliation_status": self.reconciliation_status,
                "schema_version": self.schema_version,
                "soft_entry_allowed": self.soft_entry_allowed,
                "transport_state": self.transport_state,
            }
        )


@dataclass(frozen=True)
class SafetyDecision:
    outcome: DecisionOutcome
    risk_effect: RiskEffect
    reason_codes: Tuple[str, ...]
    current_quantity: Optional[Decimal]
    computed_target_quantity: Optional[Decimal]
    intent_fingerprint: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_enum(self.outcome, DecisionOutcome, "outcome")
        _strict_enum(self.risk_effect, RiskEffect, "risk_effect")
        if not isinstance(self.reason_codes, tuple) or not self.reason_codes:
            raise ValidationError("reason_codes must be a non-empty tuple")
        for reason in self.reason_codes:
            _strict_text(reason, "reason_code", max_length=128)
        if self.current_quantity is not None:
            _strict_decimal(self.current_quantity, "current_quantity")
        if self.computed_target_quantity is not None:
            _strict_decimal(self.computed_target_quantity, "computed_target_quantity")
        if not re.fullmatch(r"[0-9a-f]{64}", self.intent_fingerprint):
            raise ValidationError("intent_fingerprint must be a lowercase SHA-256 digest")
        if self.outcome is DecisionOutcome.ALLOW:
            if self.risk_effect is not RiskEffect.REDUCING:
                raise ValidationError("ALLOW decision must be strictly reducing")
            if self.current_quantity is None or self.computed_target_quantity is None:
                raise ValidationError("ALLOW decision requires exact current and target")
            if (
                self.current_quantity == 0
                or self.computed_target_quantity.copy_abs() >= self.current_quantity.copy_abs()
            ):
                raise ValidationError("ALLOW decision must strictly reduce absolute exposure")
            if (
                self.computed_target_quantity != 0
                and self.computed_target_quantity.is_signed() != self.current_quantity.is_signed()
            ):
                raise ValidationError("ALLOW decision must not cross through zero")

    @property
    def allowed(self) -> bool:
        return self.outcome is DecisionOutcome.ALLOW

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "computed_target_quantity": self.computed_target_quantity,
                "current_quantity": self.current_quantity,
                "intent_fingerprint": self.intent_fingerprint,
                "outcome": self.outcome,
                "reason_codes": self.reason_codes,
                "risk_effect": self.risk_effect,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True)
class ReconciliationEvidence:
    execution_domain_scope: str
    reservation_id: str
    claim_id: Optional[str]
    claim_sequence: int
    submission_descriptor_fingerprint: Optional[str]
    order_ref: Optional[str]
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    observed_at: datetime
    position_observed_at: datetime
    max_evidence_age_seconds: int
    account_position_quantity: Decimal
    portfolio_position_quantity: Decimal
    aggregate_allocated_quantity: Decimal
    has_offsetting_allocations: bool
    status: ReconciliationStatus
    transport_state: TransportState
    open_orders_complete: bool
    open_orders_all_clients: bool
    open_orders_snapshot_stable: bool
    active_order_count: int
    terminal_order_status: TerminalOrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    source: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        if self.claim_id is not None:
            _strict_internal_id(self.claim_id, "claim_id", "claim")
        if type(self.claim_sequence) is not int or self.claim_sequence <= 0:
            raise ValidationError("claim_sequence must be a positive integer")
        if self.submission_descriptor_fingerprint is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.submission_descriptor_fingerprint
        ):
            raise ValidationError("submission_descriptor_fingerprint must be a SHA-256 digest")
        if self.order_ref is not None:
            _strict_text(self.order_ref, "order_ref", max_length=128)
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        object.__setattr__(self, "observed_at", _strict_utc(self.observed_at, "observed_at"))
        object.__setattr__(
            self,
            "position_observed_at",
            _strict_utc(self.position_observed_at, "position_observed_at"),
        )
        if (
            type(self.max_evidence_age_seconds) is not int
            or self.max_evidence_age_seconds < 0
            or self.max_evidence_age_seconds > SAFETY_MAX_EVIDENCE_AGE_SECONDS
        ):
            raise ValidationError(
                "max_evidence_age_seconds exceeds the safety-owned 30-second ceiling"
            )
        _strict_decimal(self.account_position_quantity, "account_position_quantity")
        _strict_decimal(self.portfolio_position_quantity, "portfolio_position_quantity")
        _strict_decimal(self.aggregate_allocated_quantity, "aggregate_allocated_quantity")
        if type(self.has_offsetting_allocations) is not bool:
            raise ValidationError("has_offsetting_allocations must be a bool")
        _strict_enum(self.status, ReconciliationStatus, "status")
        _strict_enum(self.transport_state, TransportState, "transport_state")
        if type(self.open_orders_complete) is not bool:
            raise ValidationError("open_orders_complete must be a bool")
        if type(self.open_orders_all_clients) is not bool:
            raise ValidationError("open_orders_all_clients must be a bool")
        if type(self.open_orders_snapshot_stable) is not bool:
            raise ValidationError("open_orders_snapshot_stable must be a bool")
        if type(self.active_order_count) is not int or self.active_order_count < 0:
            raise ValidationError("active_order_count must be nonnegative")
        _strict_enum(self.terminal_order_status, TerminalOrderStatus, "terminal_order_status")
        _strict_decimal(self.filled_quantity, "filled_quantity", nonnegative=True)
        _strict_decimal(self.remaining_quantity, "remaining_quantity", nonnegative=True)
        _strict_text(self.source, "source", max_length=128)

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "account_position_quantity": self.account_position_quantity,
                "active_order_count": self.active_order_count,
                "aggregate_allocated_quantity": self.aggregate_allocated_quantity,
                "claim_id": self.claim_id,
                "claim_sequence": self.claim_sequence,
                "con_id": self.con_id,
                "execution_domain_scope": self.execution_domain_scope,
                "filled_quantity": self.filled_quantity,
                "has_offsetting_allocations": self.has_offsetting_allocations,
                "max_evidence_age_seconds": self.max_evidence_age_seconds,
                "observed_at": self.observed_at,
                "open_orders_all_clients": self.open_orders_all_clients,
                "open_orders_complete": self.open_orders_complete,
                "open_orders_snapshot_stable": self.open_orders_snapshot_stable,
                "order_ref": self.order_ref,
                "portfolio_id": self.portfolio_id,
                "portfolio_position_quantity": self.portfolio_position_quantity,
                "position_observed_at": self.position_observed_at,
                "remaining_quantity": self.remaining_quantity,
                "reservation_id": self.reservation_id,
                "schema_version": self.schema_version,
                "source": self.source,
                "status": self.status,
                "submission_descriptor_fingerprint": (self.submission_descriptor_fingerprint),
                "symbol": self.symbol,
                "terminal_order_status": self.terminal_order_status,
                "transport_state": self.transport_state,
            }
        )


@dataclass(frozen=True)
class LocalPaperTerminalEvidence:
    """Exact durable local-ledger evidence; never broker reconciliation truth."""

    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    reservation_id: str
    claim_id: str
    claim_sequence: int
    submission_descriptor_fingerprint: str
    protective_quote_fingerprint: str
    order_ref: str
    settlement_id: str
    settlement_request_fingerprint: str
    settlement_receipt_fingerprint: str
    database_path: str
    database_identity: str
    database_device: int
    database_inode: int
    committed_at: datetime
    terminal_status: TerminalOrderStatus
    filled_quantity: Decimal
    remaining_quantity: Decimal
    pre_position_quantity: Decimal
    final_position_quantity: Decimal
    pre_aggregate_quantity: Decimal
    final_aggregate_quantity: Decimal
    source: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_positive_int(self.con_id, "con_id")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        _strict_internal_id(self.claim_id, "claim_id", "claim")
        _strict_positive_int(self.claim_sequence, "claim_sequence")
        _strict_hash(
            self.submission_descriptor_fingerprint,
            "submission_descriptor_fingerprint",
        )
        _strict_hash(self.protective_quote_fingerprint, "protective_quote_fingerprint")
        _strict_text(self.order_ref, "order_ref", max_length=128)
        if not isinstance(self.settlement_id, str) or not re.fullmatch(
            r"pset-[0-9a-f]{32}", self.settlement_id
        ):
            raise ValidationError("settlement_id must be an opaque paper-settlement identifier")
        _strict_hash(self.settlement_request_fingerprint, "settlement_request_fingerprint")
        _strict_hash(self.settlement_receipt_fingerprint, "settlement_receipt_fingerprint")
        path = Path(self.database_path)
        if (
            not path.is_absolute()
            or str(path) != self.database_path
            or path.parent.resolve(strict=False) / path.name != path
        ):
            raise ValidationError("database_path must be absolute and preserve its lexical leaf")
        _strict_database_identity(self.database_identity)
        for field_name in ("database_device", "database_inode"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValidationError(f"{field_name} must be a nonnegative integer")
        object.__setattr__(
            self,
            "committed_at",
            _strict_utc(self.committed_at, "committed_at"),
        )
        _strict_enum(self.terminal_status, TerminalOrderStatus, "terminal_status")
        for field_name in (
            "filled_quantity",
            "remaining_quantity",
            "pre_position_quantity",
            "final_position_quantity",
            "pre_aggregate_quantity",
            "final_aggregate_quantity",
        ):
            value = _strict_decimal(getattr(self, field_name), field_name)
            if value != value.to_integral_value():
                raise ValidationError(f"{field_name} must be an integral share quantity")
        if self.filled_quantity < 0 or self.remaining_quantity < 0:
            raise ValidationError("terminal fill quantities must be nonnegative")
        if self.terminal_status is TerminalOrderStatus.FILLED:
            if self.filled_quantity <= 0 or self.remaining_quantity != 0:
                raise ValidationError("FILLED local evidence must describe a complete fill")
        elif (
            self.terminal_status
            in {
                TerminalOrderStatus.CANCELLED,
                TerminalOrderStatus.REJECTED,
                TerminalOrderStatus.EXPIRED,
                TerminalOrderStatus.NO_SUBMISSION_CONFIRMED,
            }
            and self.filled_quantity != 0
        ):
            raise ValidationError("unfilled local terminal status cannot contain a fill")
        if self.source != "LOCAL_PAPER_SETTLEMENT_LEDGER":
            raise ValidationError("local paper terminal evidence has an untrusted source")

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "claim_id": self.claim_id,
                "claim_sequence": self.claim_sequence,
                "committed_at": self.committed_at,
                "con_id": self.con_id,
                "database_device": self.database_device,
                "database_identity": self.database_identity,
                "database_inode": self.database_inode,
                "database_path": self.database_path,
                "execution_domain_scope": self.execution_domain_scope,
                "filled_quantity": self.filled_quantity,
                "final_aggregate_quantity": self.final_aggregate_quantity,
                "final_position_quantity": self.final_position_quantity,
                "order_ref": self.order_ref,
                "portfolio_id": self.portfolio_id,
                "pre_aggregate_quantity": self.pre_aggregate_quantity,
                "pre_position_quantity": self.pre_position_quantity,
                "protective_quote_fingerprint": self.protective_quote_fingerprint,
                "remaining_quantity": self.remaining_quantity,
                "reservation_id": self.reservation_id,
                "schema_version": self.schema_version,
                "settlement_id": self.settlement_id,
                "settlement_receipt_fingerprint": self.settlement_receipt_fingerprint,
                "settlement_request_fingerprint": self.settlement_request_fingerprint,
                "source": self.source,
                "submission_descriptor_fingerprint": (self.submission_descriptor_fingerprint),
                "symbol": self.symbol,
                "terminal_status": self.terminal_status,
            }
        )


@dataclass(frozen=True)
class SubmissionDescriptor:
    execution_domain_scope: str
    account_scope: str
    con_id: int
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    time_in_force: TimeInForce
    outside_regular_hours: bool
    order_ref: str
    attempt_number: int = 1
    slice_count: int = 1
    bracket: bool = False
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_text(self.execution_domain_scope, "execution_domain_scope", pattern=_SCOPE_RE)
        _strict_account_scope(self.account_scope)
        if type(self.con_id) is not int or self.con_id <= 0:
            raise ValidationError("con_id must be a positive integer")
        _strict_enum(self.side, OrderSide, "side")
        if self.side not in {OrderSide.SELL, OrderSide.BUY_TO_COVER}:
            raise ValidationError("submission side must be semantic SELL or BUY_TO_COVER")
        _strict_decimal(self.quantity, "quantity", positive=True)
        _strict_enum(self.order_type, OrderType, "order_type")
        if self.limit_price is not None:
            _strict_decimal(self.limit_price, "limit_price", positive=True)
        if self.stop_price is not None:
            _strict_decimal(self.stop_price, "stop_price", positive=True)
        if self.order_type is OrderType.MARKET and (
            self.limit_price is not None or self.stop_price is not None
        ):
            raise ValidationError("MARKET orders cannot carry limit or stop prices")
        if self.order_type is OrderType.LIMIT and (
            self.limit_price is None or self.stop_price is not None
        ):
            raise ValidationError("LIMIT orders require only limit_price")
        if self.order_type is OrderType.STOP and (
            self.stop_price is None or self.limit_price is not None
        ):
            raise ValidationError("STOP orders require only stop_price")
        if self.order_type is OrderType.STOP_LIMIT and (
            self.stop_price is None or self.limit_price is None
        ):
            raise ValidationError("STOP_LIMIT orders require both prices")
        _strict_enum(self.time_in_force, TimeInForce, "time_in_force")
        if type(self.outside_regular_hours) is not bool:
            raise ValidationError("outside_regular_hours must be a bool")
        _strict_text(self.order_ref, "order_ref", max_length=128)
        if self.attempt_number != 1 or type(self.attempt_number) is not int:
            raise ValidationError("only one submission attempt is permitted")
        if self.slice_count != 1 or type(self.slice_count) is not int:
            raise ValidationError("order slicing is not permitted")
        if self.bracket is not False:
            raise ValidationError("bracket orders are not permitted")

    def canonical_payload(self) -> str:
        return canonical_json(
            {
                "account_scope": self.account_scope,
                "attempt_number": self.attempt_number,
                "bracket": self.bracket,
                "con_id": self.con_id,
                "execution_domain_scope": self.execution_domain_scope,
                "limit_price": self.limit_price,
                "order_ref": self.order_ref,
                "order_type": self.order_type,
                "outside_regular_hours": self.outside_regular_hours,
                "quantity": self.quantity,
                "schema_version": self.schema_version,
                "side": self.side,
                "slice_count": self.slice_count,
                "stop_price": self.stop_price,
                "time_in_force": self.time_in_force,
            }
        )

    def fingerprint(self) -> str:
        return sha256_text(self.canonical_payload())


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    idempotency_key: str
    intent_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    sequence: int
    acquired_at: datetime
    newly_acquired: bool
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        _strict_text(
            self.idempotency_key,
            "idempotency_key",
            max_length=128,
        )
        _strict_hash(self.intent_fingerprint, "intent_fingerprint")
        _strict_text(
            self.execution_domain_scope,
            "execution_domain_scope",
            pattern=_SCOPE_RE,
        )
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_positive_int(self.con_id, "con_id")
        _strict_positive_int(self.sequence, "sequence")
        object.__setattr__(
            self,
            "acquired_at",
            _strict_utc(self.acquired_at, "acquired_at"),
        )
        if type(self.newly_acquired) is not bool:
            raise ValidationError("newly_acquired must be a bool")


@dataclass(frozen=True)
class SubmissionClaim:
    claim_id: str
    reservation_id: str
    reservation_sequence: int
    idempotency_key: str
    submission_descriptor_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    order_ref: str
    sequence: int
    claimed_at: datetime
    granted: bool
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_internal_id(self.claim_id, "claim_id", "claim")
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        _strict_positive_int(self.reservation_sequence, "reservation_sequence")
        _strict_text(
            self.idempotency_key,
            "idempotency_key",
            max_length=128,
        )
        _strict_hash(
            self.submission_descriptor_fingerprint,
            "submission_descriptor_fingerprint",
        )
        _strict_text(
            self.execution_domain_scope,
            "execution_domain_scope",
            pattern=_SCOPE_RE,
        )
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_positive_int(self.con_id, "con_id")
        _strict_text(self.order_ref, "order_ref", max_length=128)
        _strict_positive_int(self.sequence, "sequence")
        if self.sequence <= self.reservation_sequence:
            raise ValidationError("claim sequence must follow reservation sequence")
        object.__setattr__(
            self,
            "claimed_at",
            _strict_utc(self.claimed_at, "claimed_at"),
        )
        if type(self.granted) is not bool:
            raise ValidationError("granted must be a bool")


_PERMIT_TOKEN = object()


class SubmissionPermit:
    """Ephemeral one-shot authority returned only by a newly committed claim.

    The permit intentionally cannot be copied, pickled, or reconstructed from
    replay. Only the issuing live journal can consume it and yield the exact
    immutable descriptor once.
    """

    __slots__ = ("_claim_id", "_consumed", "_lock")

    def __init__(
        self,
        claim_id: str,
        *,
        _token: object,
    ) -> None:
        if _token is not _PERMIT_TOKEN:
            raise TypeError("SubmissionPermit can only be issued by SafetyJournal")
        self._claim_id = claim_id
        self._consumed = False
        self._lock = threading.Lock()

    @classmethod
    def _issue(cls, claim_id: str) -> "SubmissionPermit":
        return cls(claim_id, _token=_PERMIT_TOKEN)

    @property
    def claim_id(self) -> str:
        return self._claim_id

    @property
    def consumed(self) -> bool:
        with self._lock:
            return self._consumed

    def _mark_consumed(self) -> None:
        with self._lock:
            if self._consumed:
                raise RuntimeError("submission permit has already been consumed")
            self._consumed = True

    def consume(self) -> SubmissionDescriptor:
        raise RuntimeError("permit authority must be consumed by its issuing SafetyJournal")

    def __copy__(self):
        raise TypeError("SubmissionPermit cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("SubmissionPermit cannot be copied")

    def __reduce__(self):
        raise TypeError("SubmissionPermit cannot be serialized")


@dataclass(frozen=True)
class JournalEvent:
    sequence: int
    event_id: str
    event_type: JournalEventType
    occurred_at: datetime
    idempotency_key: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    intent_fingerprint: str
    claim_id: Optional[str]
    payload_json: str
    previous_chain_hash: str
    payload_hash: str
    chain_hash: str
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_positive_int(self.sequence, "sequence")
        _strict_internal_id(self.event_id, "event_id", "evt")
        _strict_enum(self.event_type, JournalEventType, "event_type")
        object.__setattr__(
            self,
            "occurred_at",
            _strict_utc(self.occurred_at, "occurred_at"),
        )
        _strict_text(
            self.idempotency_key,
            "idempotency_key",
            max_length=128,
        )
        _strict_text(
            self.execution_domain_scope,
            "execution_domain_scope",
            pattern=_SCOPE_RE,
        )
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_positive_int(self.con_id, "con_id")
        _strict_hash(self.intent_fingerprint, "intent_fingerprint")
        if self.claim_id is not None:
            _strict_internal_id(self.claim_id, "claim_id", "claim")
        if not isinstance(self.payload_json, str):
            raise ValidationError("payload_json must be a string")
        try:
            payload = json.loads(self.payload_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValidationError("payload_json must contain valid JSON") from exc
        if not isinstance(payload, dict) or canonical_json(payload) != self.payload_json:
            raise ValidationError("payload_json must be a canonical JSON object")
        _strict_hash(self.previous_chain_hash, "previous_chain_hash")
        _strict_hash(self.payload_hash, "payload_hash")
        _strict_hash(self.chain_hash, "chain_hash")


@dataclass(frozen=True)
class ReplayReservation:
    reservation_id: str
    idempotency_key: str
    intent_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    portfolio_id: str
    con_id: int
    symbol: str
    side: OrderSide
    quantity: Decimal
    target_quantity: Decimal
    portfolio_target_quantity: Decimal
    acquired_at: datetime
    acquired_sequence: int
    claim_id: Optional[str]
    reservation_sequence: int
    submission_descriptor_fingerprint: Optional[str]
    order_ref: Optional[str]
    claim_sequence: Optional[int]
    claim_time: Optional[datetime]
    outcome_unknown: bool
    quarantined: bool
    released: bool
    terminal_sequence: Optional[int]
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        _strict_internal_id(self.reservation_id, "reservation_id", "res")
        _strict_text(
            self.idempotency_key,
            "idempotency_key",
            max_length=128,
        )
        _strict_hash(self.intent_fingerprint, "intent_fingerprint")
        _strict_text(
            self.execution_domain_scope,
            "execution_domain_scope",
            pattern=_SCOPE_RE,
        )
        _strict_account_scope(self.account_scope)
        _strict_text(self.portfolio_id, "portfolio_id", pattern=_SCOPE_RE)
        _strict_positive_int(self.con_id, "con_id")
        _strict_text(self.symbol, "symbol", max_length=32, pattern=_SYMBOL_RE)
        _strict_enum(self.side, OrderSide, "side")
        _strict_decimal(self.quantity, "quantity", positive=True)
        _strict_decimal(self.target_quantity, "target_quantity")
        _strict_decimal(
            self.portfolio_target_quantity,
            "portfolio_target_quantity",
        )
        object.__setattr__(
            self,
            "acquired_at",
            _strict_utc(self.acquired_at, "acquired_at"),
        )
        _strict_positive_int(self.acquired_sequence, "acquired_sequence")
        _strict_positive_int(self.reservation_sequence, "reservation_sequence")
        if self.reservation_sequence != self.acquired_sequence:
            raise ValidationError("reservation_sequence must equal acquired_sequence")
        claim_fields = (
            self.claim_id,
            self.submission_descriptor_fingerprint,
            self.order_ref,
            self.claim_sequence,
            self.claim_time,
        )
        if any(value is not None for value in claim_fields):
            if any(value is None for value in claim_fields):
                raise ValidationError("claim fields must be present together")
            _strict_internal_id(self.claim_id, "claim_id", "claim")
            _strict_hash(
                self.submission_descriptor_fingerprint,
                "submission_descriptor_fingerprint",
            )
            _strict_text(self.order_ref, "order_ref", max_length=128)
            _strict_positive_int(self.claim_sequence, "claim_sequence")
            if self.claim_sequence <= self.reservation_sequence:
                raise ValidationError("claim sequence must follow reservation sequence")
            object.__setattr__(
                self,
                "claim_time",
                _strict_utc(self.claim_time, "claim_time"),
            )
        for field, value in (
            ("outcome_unknown", self.outcome_unknown),
            ("quarantined", self.quarantined),
            ("released", self.released),
        ):
            if type(value) is not bool:
                raise ValidationError(f"{field} must be a bool")
        if self.terminal_sequence is not None:
            _strict_positive_int(self.terminal_sequence, "terminal_sequence")
            if self.claim_sequence is None or self.terminal_sequence <= self.claim_sequence:
                raise ValidationError("terminal_sequence must follow claim_sequence")


@dataclass(frozen=True)
class ReplayState:
    last_sequence: int
    last_chain_hash: str
    events: Tuple[JournalEvent, ...]
    reservations: Tuple[ReplayReservation, ...]
    active_reservations: Tuple[ReplayReservation, ...]
    quarantined_reservations: Tuple[ReplayReservation, ...]
    schema_version: int = MODEL_VERSION

    def __post_init__(self) -> None:
        _strict_version(self.schema_version)
        if type(self.last_sequence) is not int or self.last_sequence < 0:
            raise ValidationError("last_sequence must be a nonnegative integer")
        _strict_hash(self.last_chain_hash, "last_chain_hash")
        for field, values, value_type in (
            ("events", self.events, JournalEvent),
            ("reservations", self.reservations, ReplayReservation),
            ("active_reservations", self.active_reservations, ReplayReservation),
            (
                "quarantined_reservations",
                self.quarantined_reservations,
                ReplayReservation,
            ),
        ):
            if not isinstance(values, tuple):
                raise ValidationError(f"{field} must be a tuple")
            if any(not isinstance(value, value_type) for value in values):
                raise ValidationError(f"{field} contains an invalid model")
        if self.last_sequence != len(self.events):
            raise ValidationError("last_sequence must match event count")
        if self.events:
            if self.events[-1].sequence != self.last_sequence:
                raise ValidationError("last event sequence does not match replay state")
            if self.events[-1].chain_hash != self.last_chain_hash:
                raise ValidationError("last chain hash does not match replay state")
        elif self.last_chain_hash != "0" * 64:
            raise ValidationError("empty replay state must use the zero chain hash")
        reservation_ids = {item.reservation_id for item in self.reservations}
        if len(reservation_ids) != len(self.reservations):
            raise ValidationError("reservations contain duplicate identifiers")
        if any(
            item.reservation_id not in reservation_ids
            for item in self.active_reservations + self.quarantined_reservations
        ):
            raise ValidationError("replay subsets contain an unknown reservation")

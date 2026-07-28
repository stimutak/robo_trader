"""Dormant, pure Gate-A entry-risk contract.

This module defines one immutable ``Signal -> EntryIntent -> RiskDecision``
boundary.  It deliberately imports no runner, executor, database, or broker
implementation and grants no order-submission authority.  A later integration
PR must supply authoritative evidence and consume decisions at the final order
boundary.
"""

from __future__ import annotations

import re
from dataclasses import InitVar, dataclass
from datetime import datetime, timedelta, timezone
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    ROUND_DOWN,
    Context,
    Decimal,
    DecimalException,
    Inexact,
    Rounded,
    localcontext,
)
from enum import Enum
from typing import Optional

ENTRY_RISK_CONTRACT_VERSION = 1
GATE_A_MAX_POSITION_FRACTION = Decimal("0.02")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_SECTOR = re.compile(r"^[A-Za-z][A-Za-z0-9 &./_-]{0,63}$")
_BROKER_IDENTITY = re.compile(r"^[A-Z0-9][A-Z0-9._:/-]{0,63}$")
_ENTRY_INTENT_MARKER = object()
_QUOTE_EVIDENCE_MARKER = object()


class EntryRiskContractError(ValueError):
    """A contract record or configuration is malformed."""


class EntrySide(str, Enum):
    BUY = "BUY"
    SELL_SHORT = "SELL_SHORT"


class SignalSource(str, Enum):
    BASE_STRATEGY = "base_strategy"
    AI_DISCOVERY = "ai_discovery"
    PAIRS = "pairs"
    SMART_EXECUTION = "smart_execution"


class RiskReason(str, Enum):
    APPROVED = "approved"
    CONTRACT_NOT_READY = "contract_not_ready"
    STRATEGY_NOT_READY = "strategy_not_ready"
    SIDE_NOT_READY = "side_not_ready"
    INTENT_NOT_ACTIVE = "intent_not_active"
    INTENT_BROKER_LINEAGE_MISMATCH = "intent_broker_lineage_mismatch"
    MISSING_EVIDENCE_SCOPE = "missing_evidence_scope"
    EVIDENCE_SCOPE_MISMATCH = "evidence_scope_mismatch"
    MISSING_QUOTE = "missing_quote"
    MISSING_SECTOR = "missing_sector"
    MISSING_CORRELATION = "missing_correlation"
    MISSING_LIQUIDITY = "missing_liquidity"
    MISSING_CASH = "missing_cash"
    MISSING_BUYING_POWER = "missing_buying_power"
    MISSING_DAILY_NOTIONAL = "missing_daily_notional"
    MISSING_PORTFOLIO_EQUITY = "missing_portfolio_equity"
    MISSING_SYMBOL_EXPOSURE = "missing_symbol_exposure"
    MISSING_SECTOR_EXPOSURE = "missing_sector_exposure"
    MISSING_PORTFOLIO_EXPOSURE = "missing_portfolio_exposure"
    STALE_ACCOUNT_EVIDENCE = "stale_account_evidence"
    QUOTE_NOT_REFRESHED = "quote_not_refreshed"
    QUOTE_BROKER_LINEAGE_MISMATCH = "quote_broker_lineage_mismatch"
    CORRELATION_LIMIT = "correlation_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    NO_CAPACITY = "no_capacity"


class LimitingCapacity(str, Enum):
    REQUEST = "request"
    SYMBOL = "symbol"
    SECTOR = "sector"
    PORTFOLIO = "portfolio"
    LIQUIDITY = "liquidity"
    CASH = "cash"
    BUYING_POWER = "buying_power"
    DAILY_NOTIONAL = "daily_notional"


def _identifier(value: object, field_name: str) -> str:
    if type(value) is not str or value != value.strip() or not _IDENTIFIER.fullmatch(value):
        raise EntryRiskContractError(f"{field_name} is malformed")
    return value


def _symbol(value: object) -> str:
    if type(value) is not str or value != value.upper() or not _SYMBOL.fullmatch(value):
        raise EntryRiskContractError("symbol is malformed")
    return value


def _sector(value: object) -> str:
    if type(value) is not str or value != value.strip() or not _SECTOR.fullmatch(value):
        raise EntryRiskContractError("sector is malformed")
    return value


def _broker_identity(value: object, field_name: str) -> str:
    if type(value) is not str or not _BROKER_IDENTITY.fullmatch(value):
        raise EntryRiskContractError(f"{field_name} is malformed")
    return value


def _transport_generation(value: object, field_name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise EntryRiskContractError(f"{field_name} is malformed")
    return value


def _timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EntryRiskContractError(f"{field_name} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise EntryRiskContractError(f"{field_name} is invalid") from exc


def _decimal(
    value: object,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise EntryRiskContractError(f"{field_name} must be an exact finite Decimal")
    if positive and value <= 0:
        raise EntryRiskContractError(f"{field_name} must be positive")
    if nonnegative and value < 0:
        raise EntryRiskContractError(f"{field_name} must be nonnegative")
    return value


def _fraction(value: object, field_name: str, *, positive: bool = False) -> Decimal:
    parsed = _decimal(value, field_name, positive=positive, nonnegative=not positive)
    if parsed > 1:
        raise EntryRiskContractError(f"{field_name} must be a unit fraction")
    return parsed


def _schema_version(value: object) -> int:
    if type(value) is not int or value != ENTRY_RISK_CONTRACT_VERSION:
        raise EntryRiskContractError("entry-risk schema version is unsupported")
    return value


def _flag(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise EntryRiskContractError(f"{field_name} must be an explicit boolean")
    return value


@dataclass(frozen=True, slots=True)
class EntryBrokerContractIdentity:
    """One exact, fully qualified SMART/USD stock contract."""

    con_id: int
    symbol: str
    local_symbol: str
    security_type: str
    currency: str
    exchange: str
    primary_exchange: str
    trading_class: str

    def __post_init__(self) -> None:
        if type(self.con_id) is not int or self.con_id <= 0:
            raise EntryRiskContractError("broker contract con_id must be a positive integer")
        symbol = _symbol(self.symbol)
        local_symbol = _symbol(self.local_symbol)
        if local_symbol != symbol:
            raise EntryRiskContractError("broker contract local_symbol must match symbol")
        if self.security_type != "STK":
            raise EntryRiskContractError("broker contract security_type must be STK")
        if self.currency != "USD":
            raise EntryRiskContractError("broker contract currency must be USD")
        if self.exchange != "SMART":
            raise EntryRiskContractError("broker contract exchange must be SMART")
        _broker_identity(self.primary_exchange, "broker contract primary_exchange")
        _broker_identity(self.trading_class, "broker contract trading_class")


@dataclass(frozen=True, slots=True)
class EntrySignal:
    signal_id: str
    portfolio_id: str
    symbol: str
    side: EntrySide
    source: SignalSource
    confidence_fraction: Decimal
    requested_position_fraction: Decimal
    source_data_version: str
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    observed_at: datetime
    expires_at: datetime
    schema_version: int = ENTRY_RISK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        object.__setattr__(self, "signal_id", _identifier(self.signal_id, "signal_id"))
        object.__setattr__(self, "portfolio_id", _identifier(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if type(self.side) is not EntrySide:
            raise EntryRiskContractError("signal side is invalid")
        if type(self.source) is not SignalSource:
            raise EntryRiskContractError("signal source is invalid")
        object.__setattr__(
            self,
            "confidence_fraction",
            _fraction(self.confidence_fraction, "confidence_fraction"),
        )
        object.__setattr__(
            self,
            "requested_position_fraction",
            _fraction(
                self.requested_position_fraction,
                "requested_position_fraction",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "source_data_version",
            _identifier(self.source_data_version, "source_data_version"),
        )
        if type(self.broker_contract) is not EntryBrokerContractIdentity:
            raise EntryRiskContractError("signal broker contract is malformed")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("signal symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        observed = _timestamp(self.observed_at, "signal observed_at")
        expires = _timestamp(self.expires_at, "signal expires_at")
        if expires <= observed:
            raise EntryRiskContractError("signal expiry must follow its observation")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)

    def __copy__(self) -> "EntrySignal":
        raise TypeError("entry signal cannot be copied")

    def __deepcopy__(self, memo: object) -> "EntrySignal":
        del memo
        raise TypeError("entry signal cannot be copied")

    def __reduce__(self) -> str:
        raise TypeError("entry signal cannot be pickled")


@dataclass(frozen=True, slots=True)
class EntryIntent:
    intent_id: str
    signal_id: str
    portfolio_id: str
    symbol: str
    side: EntrySide
    source: SignalSource
    confidence_fraction: Decimal
    requested_position_fraction: Decimal
    source_data_version: str
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    quote_refresh_request_id: str
    created_at: datetime
    expires_at: datetime
    schema_version: int = ENTRY_RISK_CONTRACT_VERSION
    _builder_marker: InitVar[object] = None

    def __post_init__(self, _builder_marker: object) -> None:
        if _builder_marker is not _ENTRY_INTENT_MARKER:
            raise EntryRiskContractError("entry intent is not signal-derived")
        _schema_version(self.schema_version)
        for field_name in (
            "intent_id",
            "signal_id",
            "portfolio_id",
            "source_data_version",
            "quote_refresh_request_id",
        ):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if type(self.side) is not EntrySide or type(self.source) is not SignalSource:
            raise EntryRiskContractError("entry intent side or source is invalid")
        if type(self.broker_contract) is not EntryBrokerContractIdentity:
            raise EntryRiskContractError("entry intent broker contract is malformed")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("entry intent symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        object.__setattr__(
            self,
            "confidence_fraction",
            _fraction(self.confidence_fraction, "confidence_fraction"),
        )
        object.__setattr__(
            self,
            "requested_position_fraction",
            _fraction(
                self.requested_position_fraction,
                "requested_position_fraction",
                positive=True,
            ),
        )
        created = _timestamp(self.created_at, "intent created_at")
        expires = _timestamp(self.expires_at, "intent expires_at")
        if expires <= created:
            raise EntryRiskContractError("entry intent expiry must follow creation")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)

    def __copy__(self) -> "EntryIntent":
        raise TypeError("entry intent cannot be copied")

    def __deepcopy__(self, memo: object) -> "EntryIntent":
        del memo
        raise TypeError("entry intent cannot be copied")

    def __reduce__(self) -> str:
        raise TypeError("entry intent cannot be pickled")


def build_entry_intent(
    signal: EntrySignal,
    *,
    intent_id: str,
    quote_refresh_request_id: str,
    created_at: datetime,
) -> EntryIntent:
    """Transform one exact signal into an immutable quote-bound intent."""

    if type(signal) is not EntrySignal:
        raise EntryRiskContractError("entry intent requires an exact EntrySignal")
    created = _timestamp(created_at, "intent created_at")
    if created < signal.observed_at or created >= signal.expires_at:
        raise EntryRiskContractError("signal is not active at intent creation")
    return EntryIntent(
        intent_id=intent_id,
        signal_id=signal.signal_id,
        portfolio_id=signal.portfolio_id,
        symbol=signal.symbol,
        side=signal.side,
        source=signal.source,
        confidence_fraction=signal.confidence_fraction,
        requested_position_fraction=signal.requested_position_fraction,
        source_data_version=signal.source_data_version,
        broker_contract=signal.broker_contract,
        transport_generation=signal.transport_generation,
        quote_refresh_request_id=quote_refresh_request_id,
        created_at=created,
        expires_at=signal.expires_at,
        _builder_marker=_ENTRY_INTENT_MARKER,
    )


@dataclass(frozen=True, slots=True)
class RefreshedQuoteEvidence:
    quote_id: str
    refresh_request_id: str
    symbol: str
    broker_contract: EntryBrokerContractIdentity
    price_usd: Decimal
    observed_at: datetime
    transport_generation: str
    source: str
    _builder_marker: InitVar[object] = None

    def __post_init__(self, _builder_marker: object) -> None:
        if _builder_marker is not _QUOTE_EVIDENCE_MARKER:
            raise EntryRiskContractError("refreshed quote is not factory-produced")
        for field_name in ("quote_id", "refresh_request_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if type(self.broker_contract) is not EntryBrokerContractIdentity:
            raise EntryRiskContractError("quote broker contract is malformed")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("quote symbol does not match broker contract")
        object.__setattr__(self, "price_usd", _decimal(self.price_usd, "price_usd", positive=True))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "quote observed_at"))
        if type(self.source) is not str or self.source != "live-broker":
            raise EntryRiskContractError("refreshed quote source must be live-broker")

    def __copy__(self) -> "RefreshedQuoteEvidence":
        raise TypeError("refreshed quote evidence cannot be copied")

    def __deepcopy__(self, memo: object) -> "RefreshedQuoteEvidence":
        del memo
        raise TypeError("refreshed quote evidence cannot be copied")

    def __reduce__(self) -> str:
        raise TypeError("refreshed quote evidence cannot be pickled")


def build_refreshed_quote_evidence(
    *,
    quote_id: str,
    refresh_request_id: str,
    broker_contract: EntryBrokerContractIdentity,
    price_usd: Decimal,
    observed_at: datetime,
    transport_generation: str,
    source: str,
) -> RefreshedQuoteEvidence:
    """Build an immutable quote bound to one exact contract and transport."""

    if type(broker_contract) is not EntryBrokerContractIdentity:
        raise EntryRiskContractError("quote requires an exact broker contract")
    return RefreshedQuoteEvidence(
        quote_id=quote_id,
        refresh_request_id=refresh_request_id,
        symbol=broker_contract.symbol,
        broker_contract=broker_contract,
        price_usd=price_usd,
        observed_at=observed_at,
        transport_generation=transport_generation,
        source=source,
        _builder_marker=_QUOTE_EVIDENCE_MARKER,
    )


@dataclass(frozen=True, slots=True)
class CorrelationEvidence:
    complete: bool
    existing_position_count: int
    max_absolute_correlation: Decimal
    observed_at: datetime

    def __post_init__(self) -> None:
        _flag(self.complete, "correlation complete")
        if type(self.existing_position_count) is not int or self.existing_position_count < 0:
            raise EntryRiskContractError("existing_position_count must be nonnegative")
        value = _fraction(self.max_absolute_correlation, "max_absolute_correlation")
        if self.existing_position_count == 0 and value != 0:
            raise EntryRiskContractError("an empty portfolio must have zero correlation")
        object.__setattr__(self, "max_absolute_correlation", value)
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "correlation observed_at"),
        )


@dataclass(frozen=True, slots=True)
class LiquidityEvidence:
    complete: bool
    average_daily_dollar_volume_usd: Decimal
    observed_at: datetime

    def __post_init__(self) -> None:
        _flag(self.complete, "liquidity complete")
        object.__setattr__(
            self,
            "average_daily_dollar_volume_usd",
            _decimal(
                self.average_daily_dollar_volume_usd,
                "average_daily_dollar_volume_usd",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "observed_at",
            _timestamp(self.observed_at, "liquidity observed_at"),
        )


@dataclass(frozen=True, slots=True)
class EntryRiskEvidence:
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    observed_at: Optional[datetime] = None
    quote: Optional[RefreshedQuoteEvidence] = None
    sector: Optional[str] = None
    correlation: Optional[CorrelationEvidence] = None
    liquidity: Optional[LiquidityEvidence] = None
    portfolio_equity_usd: Optional[Decimal] = None
    cash_available_usd: Optional[Decimal] = None
    buying_power_usd: Optional[Decimal] = None
    current_symbol_gross_notional_usd: Optional[Decimal] = None
    current_sector_gross_notional_usd: Optional[Decimal] = None
    portfolio_gross_notional_usd: Optional[Decimal] = None
    daily_executed_notional_usd: Optional[Decimal] = None

    def __post_init__(self) -> None:
        if self.portfolio_id is not None:
            object.__setattr__(
                self,
                "portfolio_id",
                _identifier(self.portfolio_id, "portfolio_id"),
            )
        if self.symbol is not None:
            object.__setattr__(self, "symbol", _symbol(self.symbol))
        if self.observed_at is not None:
            object.__setattr__(
                self,
                "observed_at",
                _timestamp(self.observed_at, "risk evidence observed_at"),
            )
        if self.quote is not None and type(self.quote) is not RefreshedQuoteEvidence:
            raise EntryRiskContractError("quote evidence is malformed")
        if self.sector is not None:
            object.__setattr__(self, "sector", _sector(self.sector))
        if self.correlation is not None and type(self.correlation) is not CorrelationEvidence:
            raise EntryRiskContractError("correlation evidence is malformed")
        if self.liquidity is not None and type(self.liquidity) is not LiquidityEvidence:
            raise EntryRiskContractError("liquidity evidence is malformed")
        for field_name in (
            "portfolio_equity_usd",
            "cash_available_usd",
            "buying_power_usd",
            "current_symbol_gross_notional_usd",
            "current_sector_gross_notional_usd",
            "portfolio_gross_notional_usd",
            "daily_executed_notional_usd",
        ):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _decimal(
                        value,
                        field_name,
                        positive=field_name == "portfolio_equity_usd",
                        nonnegative=field_name != "portfolio_equity_usd",
                    ),
                )


@dataclass(frozen=True, slots=True)
class EntryRiskLimits:
    max_position_fraction: Decimal
    max_sector_fraction: Decimal
    max_portfolio_gross_fraction: Decimal
    max_absolute_correlation: Decimal
    minimum_average_daily_dollar_volume_usd: Decimal
    max_order_fraction_of_daily_dollar_volume: Decimal
    max_daily_notional_usd: Decimal
    max_quote_age: timedelta
    max_account_evidence_age: timedelta

    def __post_init__(self) -> None:
        position = _fraction(
            self.max_position_fraction,
            "max_position_fraction",
            positive=True,
        )
        if position > GATE_A_MAX_POSITION_FRACTION:
            raise EntryRiskContractError("Gate-A position cap cannot exceed 2%")
        object.__setattr__(self, "max_position_fraction", position)
        object.__setattr__(
            self,
            "max_sector_fraction",
            _fraction(self.max_sector_fraction, "max_sector_fraction", positive=True),
        )
        object.__setattr__(
            self,
            "max_portfolio_gross_fraction",
            _fraction(
                self.max_portfolio_gross_fraction,
                "max_portfolio_gross_fraction",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_absolute_correlation",
            _fraction(self.max_absolute_correlation, "max_absolute_correlation"),
        )
        object.__setattr__(
            self,
            "minimum_average_daily_dollar_volume_usd",
            _decimal(
                self.minimum_average_daily_dollar_volume_usd,
                "minimum_average_daily_dollar_volume_usd",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_order_fraction_of_daily_dollar_volume",
            _fraction(
                self.max_order_fraction_of_daily_dollar_volume,
                "max_order_fraction_of_daily_dollar_volume",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "max_daily_notional_usd",
            _decimal(self.max_daily_notional_usd, "max_daily_notional_usd", positive=True),
        )
        for field_name in ("max_quote_age", "max_account_evidence_age"):
            value = getattr(self, field_name)
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise EntryRiskContractError(f"{field_name} must be a positive timedelta")


@dataclass(frozen=True, slots=True)
class EntryFeatureFlags:
    risk_contract_enabled: bool
    refreshed_quote_revalidation_enabled: bool
    base_strategy_entries_enabled: bool
    short_entries_enabled: bool
    ai_discovery_entries_enabled: bool
    pairs_entries_enabled: bool
    smart_execution_entries_enabled: bool

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            _flag(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    intent_id: str
    signal_id: str
    portfolio_id: str
    symbol: str
    side: EntrySide
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    evaluated_at: datetime
    risk_approved: bool
    reasons: tuple[RiskReason, ...]
    approved_quantity: int
    approved_notional_usd: Decimal
    quote_id: Optional[str]
    limiting_capacity: Optional[LimitingCapacity]
    authorizes_order_submission: bool = False
    runtime_integration_ready: bool = False
    schema_version: int = ENTRY_RISK_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _schema_version(self.schema_version)
        for field_name in ("intent_id", "signal_id", "portfolio_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if type(self.side) is not EntrySide:
            raise EntryRiskContractError("risk decision side is invalid")
        if type(self.broker_contract) is not EntryBrokerContractIdentity:
            raise EntryRiskContractError("risk decision broker contract is malformed")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("risk decision symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        object.__setattr__(self, "evaluated_at", _timestamp(self.evaluated_at, "evaluated_at"))
        _flag(self.risk_approved, "risk_approved")
        if (
            type(self.reasons) is not tuple
            or not self.reasons
            or any(type(reason) is not RiskReason for reason in self.reasons)
        ):
            raise EntryRiskContractError("risk decision reasons are invalid")
        if type(self.approved_quantity) is not int or self.approved_quantity < 0:
            raise EntryRiskContractError("approved_quantity must be nonnegative")
        object.__setattr__(
            self,
            "approved_notional_usd",
            _decimal(self.approved_notional_usd, "approved_notional_usd", nonnegative=True),
        )
        if self.quote_id is not None:
            object.__setattr__(self, "quote_id", _identifier(self.quote_id, "quote_id"))
        if (
            self.limiting_capacity is not None
            and type(self.limiting_capacity) is not LimitingCapacity
        ):
            raise EntryRiskContractError("limiting_capacity is invalid")
        if (
            self.authorizes_order_submission is not False
            or self.runtime_integration_ready is not False
        ):
            raise EntryRiskContractError("the dormant contract cannot authorize runtime submission")
        if self.risk_approved:
            if (
                self.reasons != (RiskReason.APPROVED,)
                or self.approved_quantity <= 0
                or self.approved_notional_usd <= 0
                or self.quote_id is None
                or self.limiting_capacity is None
            ):
                raise EntryRiskContractError("approved risk decision is internally inconsistent")
        elif self.approved_quantity != 0 or self.approved_notional_usd != 0:
            raise EntryRiskContractError("rejected risk decision cannot approve exposure")


def _rejected(
    intent: EntryIntent,
    evaluated_at: datetime,
    reasons: list[RiskReason],
    quote: Optional[RefreshedQuoteEvidence],
) -> RiskDecision:
    ordered = tuple(dict.fromkeys(reasons))
    return RiskDecision(
        intent_id=intent.intent_id,
        signal_id=intent.signal_id,
        portfolio_id=intent.portfolio_id,
        symbol=intent.symbol,
        side=intent.side,
        broker_contract=intent.broker_contract,
        transport_generation=intent.transport_generation,
        evaluated_at=evaluated_at,
        risk_approved=False,
        reasons=ordered,
        approved_quantity=0,
        approved_notional_usd=Decimal(0),
        quote_id=None if quote is None else quote.quote_id,
        limiting_capacity=None,
    )


def _source_enabled(source: SignalSource, flags: EntryFeatureFlags) -> bool:
    return {
        SignalSource.BASE_STRATEGY: flags.base_strategy_entries_enabled,
        SignalSource.AI_DISCOVERY: flags.ai_discovery_entries_enabled,
        SignalSource.PAIRS: flags.pairs_entries_enabled,
        SignalSource.SMART_EXECUTION: flags.smart_execution_entries_enabled,
    }[source]


def _evidence_is_current(
    observed_at: datetime,
    *,
    intent: EntryIntent,
    evaluated_at: datetime,
    max_age: timedelta,
) -> bool:
    return (
        intent.created_at <= observed_at <= evaluated_at and evaluated_at - observed_at <= max_age
    )


def _isolated_context(required_precision: int) -> Context:
    if required_precision <= 0 or required_precision > MAX_PREC:
        raise EntryRiskContractError("exact risk arithmetic precision is unsupported")
    return Context(
        prec=max(32, required_precision),
        rounding=ROUND_DOWN,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        capitals=1,
        clamp=0,
    )


def _exact_multiply(left: Decimal, right: Decimal, field_name: str) -> Decimal:
    precision = len(left.as_tuple().digits) + len(right.as_tuple().digits) + 2
    try:
        with localcontext(_isolated_context(precision)) as context:
            result = left * right
            if context.flags[Inexact] or context.flags[Rounded]:
                raise EntryRiskContractError(f"{field_name} multiplication was not exact")
            return result
    except (DecimalException, OverflowError, ValueError) as exc:
        raise EntryRiskContractError(f"{field_name} multiplication failed") from exc


def _exact_subtract(left: Decimal, right: Decimal, field_name: str) -> Decimal:
    left_exponent = int(left.as_tuple().exponent)
    right_exponent = int(right.as_tuple().exponent)
    precision = max(left.adjusted(), right.adjusted()) - min(left_exponent, right_exponent) + 3
    try:
        with localcontext(_isolated_context(precision)) as context:
            result = left - right
            if context.flags[Inexact] or context.flags[Rounded]:
                raise EntryRiskContractError(f"{field_name} subtraction was not exact")
            return result
    except (DecimalException, OverflowError, ValueError) as exc:
        raise EntryRiskContractError(f"{field_name} subtraction failed") from exc


def _exact_ratio_floor(capacity_usd: Decimal, price_usd: Decimal) -> int:
    capacity_numerator, capacity_denominator = capacity_usd.as_integer_ratio()
    price_numerator, price_denominator = price_usd.as_integer_ratio()
    return (capacity_numerator * price_denominator) // (capacity_denominator * price_numerator)


def _floor_quantity(capacity_usd: Decimal, price_usd: Decimal) -> int:
    exact_floor = _exact_ratio_floor(capacity_usd, price_usd)
    precision = (
        len(str(exact_floor))
        + len(capacity_usd.as_tuple().digits)
        + len(price_usd.as_tuple().digits)
        + 2
    )
    try:
        with localcontext(_isolated_context(precision)):
            quotient = capacity_usd / price_usd
            rounded = int(quotient.to_integral_value(rounding=ROUND_DOWN))
    except (DecimalException, OverflowError, ValueError) as exc:
        raise EntryRiskContractError("quantity flooring failed") from exc
    if rounded != exact_floor:
        raise EntryRiskContractError("Decimal ROUND_DOWN quantity flooring was not exact")
    return rounded


def _minimum_capacity(
    capacities: tuple[tuple[LimitingCapacity, Decimal], ...],
) -> tuple[LimitingCapacity, Decimal]:
    precision = max(len(value.as_tuple().digits) for _, value in capacities) + 2
    with localcontext(_isolated_context(precision)):
        return min(capacities, key=lambda item: item[1])


def _assert_approved_within_all_capacities(
    *,
    quantity: int,
    approved_notional_usd: Decimal,
    price_usd: Decimal,
    capacities: tuple[tuple[LimitingCapacity, Decimal], ...],
) -> None:
    independently_calculated_notional = _exact_multiply(
        price_usd,
        Decimal(quantity),
        "approved_notional_usd postcondition",
    )
    if approved_notional_usd != independently_calculated_notional:
        raise EntryRiskContractError("approved notional postcondition failed")
    for capacity_name, capacity_usd in capacities:
        if approved_notional_usd > capacity_usd or quantity > _exact_ratio_floor(
            capacity_usd, price_usd
        ):
            raise EntryRiskContractError(
                f"approved quantity exceeded the exact {capacity_name.value} capacity"
            )


def evaluate_entry_intent(
    intent: EntryIntent,
    evidence: EntryRiskEvidence,
    limits: EntryRiskLimits,
    flags: EntryFeatureFlags,
    *,
    expected_broker_contract: EntryBrokerContractIdentity,
    expected_transport_generation: str,
    evaluated_at: datetime,
) -> RiskDecision:
    """Evaluate one intent using complete, refreshed evidence and exact units."""

    if type(intent) is not EntryIntent:
        raise EntryRiskContractError("risk evaluation requires an exact EntryIntent")
    if type(evidence) is not EntryRiskEvidence:
        raise EntryRiskContractError("risk evaluation requires exact EntryRiskEvidence")
    if type(limits) is not EntryRiskLimits or type(flags) is not EntryFeatureFlags:
        raise EntryRiskContractError("risk evaluation configuration is malformed")
    if type(expected_broker_contract) is not EntryBrokerContractIdentity:
        raise EntryRiskContractError("expected broker contract is missing or malformed")
    active_generation = _transport_generation(
        expected_transport_generation,
        "expected_transport_generation",
    )
    now = _timestamp(evaluated_at, "evaluated_at")
    reasons: list[RiskReason] = []

    if not flags.risk_contract_enabled or not flags.refreshed_quote_revalidation_enabled:
        reasons.append(RiskReason.CONTRACT_NOT_READY)
    if not _source_enabled(intent.source, flags):
        reasons.append(RiskReason.STRATEGY_NOT_READY)
    if intent.side is EntrySide.SELL_SHORT and not flags.short_entries_enabled:
        reasons.append(RiskReason.SIDE_NOT_READY)
    if now < intent.created_at or now >= intent.expires_at:
        reasons.append(RiskReason.INTENT_NOT_ACTIVE)
    if (
        intent.broker_contract != expected_broker_contract
        or intent.transport_generation != active_generation
    ):
        reasons.append(RiskReason.INTENT_BROKER_LINEAGE_MISMATCH)

    if evidence.portfolio_id is None or evidence.symbol is None:
        reasons.append(RiskReason.MISSING_EVIDENCE_SCOPE)
    elif evidence.portfolio_id != intent.portfolio_id or evidence.symbol != intent.symbol:
        reasons.append(RiskReason.EVIDENCE_SCOPE_MISMATCH)

    required_money = (
        ("portfolio_equity_usd", RiskReason.MISSING_PORTFOLIO_EQUITY),
        ("cash_available_usd", RiskReason.MISSING_CASH),
        ("buying_power_usd", RiskReason.MISSING_BUYING_POWER),
        ("current_symbol_gross_notional_usd", RiskReason.MISSING_SYMBOL_EXPOSURE),
        ("current_sector_gross_notional_usd", RiskReason.MISSING_SECTOR_EXPOSURE),
        ("portfolio_gross_notional_usd", RiskReason.MISSING_PORTFOLIO_EXPOSURE),
        ("daily_executed_notional_usd", RiskReason.MISSING_DAILY_NOTIONAL),
    )
    for field_name, reason in required_money:
        if getattr(evidence, field_name) is None:
            reasons.append(reason)
    if evidence.observed_at is None:
        reasons.append(RiskReason.STALE_ACCOUNT_EVIDENCE)
    elif not _evidence_is_current(
        evidence.observed_at,
        intent=intent,
        evaluated_at=now,
        max_age=limits.max_account_evidence_age,
    ):
        reasons.append(RiskReason.STALE_ACCOUNT_EVIDENCE)
    if evidence.sector is None:
        reasons.append(RiskReason.MISSING_SECTOR)
    if evidence.correlation is None or not evidence.correlation.complete:
        reasons.append(RiskReason.MISSING_CORRELATION)
    elif not _evidence_is_current(
        evidence.correlation.observed_at,
        intent=intent,
        evaluated_at=now,
        max_age=limits.max_account_evidence_age,
    ):
        reasons.append(RiskReason.MISSING_CORRELATION)
    if evidence.liquidity is None or not evidence.liquidity.complete:
        reasons.append(RiskReason.MISSING_LIQUIDITY)
    elif not _evidence_is_current(
        evidence.liquidity.observed_at,
        intent=intent,
        evaluated_at=now,
        max_age=limits.max_account_evidence_age,
    ):
        reasons.append(RiskReason.MISSING_LIQUIDITY)

    quote = evidence.quote
    if quote is None:
        reasons.append(RiskReason.MISSING_QUOTE)
    elif (
        quote.symbol != intent.symbol
        or quote.refresh_request_id != intent.quote_refresh_request_id
        or not _evidence_is_current(
            quote.observed_at,
            intent=intent,
            evaluated_at=now,
            max_age=limits.max_quote_age,
        )
    ):
        reasons.append(RiskReason.QUOTE_NOT_REFRESHED)
    if quote is not None and (
        quote.broker_contract != expected_broker_contract
        or quote.transport_generation != active_generation
        or quote.broker_contract != intent.broker_contract
        or quote.transport_generation != intent.transport_generation
    ):
        reasons.append(RiskReason.QUOTE_BROKER_LINEAGE_MISMATCH)

    if reasons:
        return _rejected(intent, now, reasons, quote)

    assert quote is not None
    assert evidence.correlation is not None
    assert evidence.liquidity is not None
    assert evidence.portfolio_equity_usd is not None
    assert evidence.cash_available_usd is not None
    assert evidence.buying_power_usd is not None
    assert evidence.current_symbol_gross_notional_usd is not None
    assert evidence.current_sector_gross_notional_usd is not None
    assert evidence.portfolio_gross_notional_usd is not None
    assert evidence.daily_executed_notional_usd is not None

    if evidence.correlation.max_absolute_correlation > limits.max_absolute_correlation:
        return _rejected(intent, now, [RiskReason.CORRELATION_LIMIT], quote)
    if (
        evidence.liquidity.average_daily_dollar_volume_usd
        < limits.minimum_average_daily_dollar_volume_usd
    ):
        return _rejected(intent, now, [RiskReason.LIQUIDITY_LIMIT], quote)

    equity = evidence.portfolio_equity_usd
    capacities = (
        (
            LimitingCapacity.REQUEST,
            _exact_multiply(
                equity,
                intent.requested_position_fraction,
                "requested position capacity",
            ),
        ),
        (
            LimitingCapacity.SYMBOL,
            _exact_subtract(
                _exact_multiply(
                    equity,
                    limits.max_position_fraction,
                    "symbol position capacity",
                ),
                evidence.current_symbol_gross_notional_usd,
                "remaining symbol position capacity",
            ),
        ),
        (
            LimitingCapacity.SECTOR,
            _exact_subtract(
                _exact_multiply(
                    equity,
                    limits.max_sector_fraction,
                    "sector position capacity",
                ),
                evidence.current_sector_gross_notional_usd,
                "remaining sector position capacity",
            ),
        ),
        (
            LimitingCapacity.PORTFOLIO,
            _exact_subtract(
                _exact_multiply(
                    equity,
                    limits.max_portfolio_gross_fraction,
                    "portfolio gross capacity",
                ),
                evidence.portfolio_gross_notional_usd,
                "remaining portfolio gross capacity",
            ),
        ),
        (
            LimitingCapacity.LIQUIDITY,
            _exact_multiply(
                evidence.liquidity.average_daily_dollar_volume_usd,
                limits.max_order_fraction_of_daily_dollar_volume,
                "liquidity capacity",
            ),
        ),
        (LimitingCapacity.CASH, evidence.cash_available_usd),
        (LimitingCapacity.BUYING_POWER, evidence.buying_power_usd),
        (
            LimitingCapacity.DAILY_NOTIONAL,
            _exact_subtract(
                limits.max_daily_notional_usd,
                evidence.daily_executed_notional_usd,
                "remaining daily notional capacity",
            ),
        ),
    )
    limiting_capacity, capacity_usd = _minimum_capacity(capacities)
    if capacity_usd <= 0:
        return _rejected(intent, now, [RiskReason.NO_CAPACITY], quote)
    quantity = _floor_quantity(capacity_usd, quote.price_usd)
    if quantity <= 0:
        return _rejected(intent, now, [RiskReason.NO_CAPACITY], quote)
    approved_notional = _exact_multiply(
        quote.price_usd,
        Decimal(quantity),
        "approved_notional_usd",
    )
    _assert_approved_within_all_capacities(
        quantity=quantity,
        approved_notional_usd=approved_notional,
        price_usd=quote.price_usd,
        capacities=capacities,
    )

    return RiskDecision(
        intent_id=intent.intent_id,
        signal_id=intent.signal_id,
        portfolio_id=intent.portfolio_id,
        symbol=intent.symbol,
        side=intent.side,
        broker_contract=intent.broker_contract,
        transport_generation=intent.transport_generation,
        evaluated_at=now,
        risk_approved=True,
        reasons=(RiskReason.APPROVED,),
        approved_quantity=quantity,
        approved_notional_usd=approved_notional,
        quote_id=quote.quote_id,
        limiting_capacity=limiting_capacity,
    )


__all__ = [
    "ENTRY_RISK_CONTRACT_VERSION",
    "GATE_A_MAX_POSITION_FRACTION",
    "CorrelationEvidence",
    "EntryBrokerContractIdentity",
    "EntryFeatureFlags",
    "EntryIntent",
    "EntryRiskContractError",
    "EntryRiskEvidence",
    "EntryRiskLimits",
    "EntrySide",
    "EntrySignal",
    "LimitingCapacity",
    "LiquidityEvidence",
    "RefreshedQuoteEvidence",
    "RiskDecision",
    "RiskReason",
    "SignalSource",
    "build_entry_intent",
    "build_refreshed_quote_evidence",
    "evaluate_entry_intent",
]

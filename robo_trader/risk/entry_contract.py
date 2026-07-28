"""Dormant, pure Gate-A entry-risk contract.

This module defines one immutable ``Signal -> EntryIntent -> RiskDecision``
boundary.  It deliberately imports no runner, executor, database, or broker
implementation and grants no order-submission authority.  A later integration
PR must supply authoritative evidence and consume decisions at the final order
boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import weakref
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
from typing import Optional, cast

ENTRY_RISK_CONTRACT_VERSION = 1
GATE_A_MAX_POSITION_FRACTION = Decimal("0.02")

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_SECTOR = re.compile(r"^[A-Za-z][A-Za-z0-9 &./_-]{0,63}$")
_BROKER_IDENTITY = re.compile(r"^[A-Z0-9][A-Z0-9._:/-]{0,63}$")


class _DecisionReplayTombstones:
    """Bounded live-window semantic replay records."""

    __slots__ = ("_entries", "_limit", "_seen_bits")

    _SEEN_BIT_COUNT = 1 << 20
    _SEEN_HASH_COUNT = 8

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("decision replay tombstone limit must be positive")
        self._entries: dict[tuple[object, ...], datetime] = {}
        self._limit = limit
        self._seen_bits = 0

    @classmethod
    def _seen_indices(cls, replay_key: tuple[object, ...]) -> tuple[int, ...]:
        encoded = json.dumps(
            replay_key,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest = hashlib.sha256(encoded).digest()
        return tuple(
            int.from_bytes(digest[offset : offset + 4], "big") % cls._SEEN_BIT_COUNT
            for offset in range(0, cls._SEEN_HASH_COUNT * 4, 4)
        )

    def record(
        self,
        replay_key: tuple[object, ...],
        *,
        expires_at: datetime,
        consumed_at: datetime,
        evaluated_at: Optional[datetime] = None,
    ) -> None:
        if evaluated_at is not None and consumed_at < evaluated_at:
            raise EntryRiskContractError("RiskDecision cannot be consumed before its evaluation")
        if consumed_at >= expires_at:
            raise EntryRiskContractError("RiskDecision expired before consumption")

        seen_indices = self._seen_indices(replay_key)
        if replay_key in self._entries or all(
            self._seen_bits & (1 << index) for index in seen_indices
        ):
            raise EntryRiskContractError("RiskDecision semantic replay detected")

        # A rejected call must not weaken replay protection for any other
        # decision.  Derive the next registry state without mutating the live
        # tombstones, then commit it only after every validation succeeds.
        survivors = {key: expiry for key, expiry in self._entries.items() if expiry > consumed_at}
        if len(survivors) >= self._limit:
            raise EntryRiskContractError("RiskDecision live replay registry capacity exhausted")
        survivors[replay_key] = expires_at
        self._entries = survivors
        for index in seen_indices:
            self._seen_bits |= 1 << index


def _build_capability_authority():
    seal = object()
    registry: dict[
        int,
        tuple[weakref.ReferenceType[object], type[object], tuple[object, ...]],
    ] = {}
    consumed_decisions = _DecisionReplayTombstones(4096)
    lock = threading.Lock()

    def is_sealed(candidate: object) -> bool:
        return candidate is seal

    def mint(capability_type, values: dict[str, object]):
        capability = capability_type(**values, _seal=seal)
        state = _capability_state(capability)
        object_id = id(capability)

        def discard(reference: weakref.ReferenceType[object]) -> None:
            with lock:
                registered = registry.get(object_id)
                if registered is not None and registered[0] is reference:
                    registry.pop(object_id, None)

        reference = weakref.ref(capability, discard)
        with lock:
            registry[object_id] = (reference, capability_type, state)
        return capability

    def consume(
        capability: object,
        expected_type,
        *,
        consumed_at: Optional[datetime] = None,
    ):
        if type(capability) is not expected_type:
            raise EntryRiskContractError(f"exact {expected_type.__name__} capability is required")
        with lock:
            registered = registry.pop(id(capability), None)
        try:
            current_state = _capability_state(capability)
        except Exception:
            current_state = None
        if (
            registered is None
            or registered[0]() is not capability
            or registered[1] is not expected_type
            or registered[2] != current_state
        ):
            raise EntryRiskContractError(
                f"{expected_type.__name__} is altered, copied, forged, or already consumed"
            )
        try:
            capability.__post_init__(seal)
            revalidated_state = _capability_state(capability)
        except Exception as exc:
            raise EntryRiskContractError(
                f"{expected_type.__name__} failed authorization-boundary revalidation"
            ) from exc
        if revalidated_state != registered[2]:
            raise EntryRiskContractError(
                f"{expected_type.__name__} changed during authorization-boundary revalidation"
            )
        if expected_type is RiskDecision:
            if type(consumed_at) is not datetime:
                raise EntryRiskContractError(
                    "RiskDecision consumption requires an exact consumed_at"
                )
            replay_key = _risk_decision_replay_key(capability)
            with lock:
                consumed_decisions.record(
                    replay_key,
                    expires_at=capability.expires_at,
                    consumed_at=consumed_at,
                    evaluated_at=capability.evaluated_at,
                )
        return capability

    def transfer_decision(decision: RiskDecision) -> ConsumedRiskDecision:
        return ConsumedRiskDecision(
            (
                decision.intent_id,
                decision.signal_id,
                decision.portfolio_id,
                decision.symbol,
                decision.side,
                _contract_state(decision.broker_contract),
                decision.transport_generation,
                decision.evaluated_at,
                decision.risk_approved,
                decision.reasons,
                decision.approved_quantity,
                decision.approved_notional_usd,
                decision.quote_id,
                decision.limiting_capacity,
                decision.schema_version,
                decision.expires_at,
            ),
            _seal=seal,
        )

    return is_sealed, mint, consume, transfer_decision


(
    _is_capability_seal,
    _mint_capability,
    _consume_capability,
    _transfer_risk_decision,
) = _build_capability_authority()


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


class RefreshedQuoteSource(str, Enum):
    LIVE_BROKER = "live-broker"


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
    MISSING_ML_CORROBORATION = "missing_ml_corroboration"
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
    QUOTE_SOURCE_MISMATCH = "quote_source_mismatch"
    CORRELATION_LIMIT = "correlation_limit"
    LIQUIDITY_LIMIT = "liquidity_limit"
    ML_CORROBORATION_REQUIRED = "ml_corroboration_required"
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
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise EntryRiskContractError(f"{field_name} must be an exact timezone-aware datetime")
    try:
        normalized = value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise EntryRiskContractError(f"{field_name} is invalid") from exc
    if type(normalized) is not datetime:
        raise EntryRiskContractError(f"{field_name} did not normalize to an exact datetime")
    return normalized


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


def _validate_broker_contract(
    value: object,
    field_name: str = "broker contract",
) -> EntryBrokerContractIdentity:
    if type(value) is not EntryBrokerContractIdentity:
        raise EntryRiskContractError(f"{field_name} is malformed")
    try:
        if type(value.con_id) is not int or value.con_id <= 0:
            raise EntryRiskContractError(f"{field_name} con_id must be a positive integer")
        symbol = _symbol(value.symbol)
        local_symbol = _symbol(value.local_symbol)
        if local_symbol != symbol:
            raise EntryRiskContractError(f"{field_name} local_symbol must match symbol")
        if type(value.security_type) is not str or value.security_type != "STK":
            raise EntryRiskContractError(f"{field_name} security_type must be STK")
        if type(value.currency) is not str or value.currency != "USD":
            raise EntryRiskContractError(f"{field_name} currency must be USD")
        if type(value.exchange) is not str or value.exchange != "SMART":
            raise EntryRiskContractError(f"{field_name} exchange must be SMART")
        _broker_identity(value.primary_exchange, f"{field_name} primary_exchange")
        _broker_identity(value.trading_class, f"{field_name} trading_class")
    except AttributeError as exc:
        raise EntryRiskContractError(f"{field_name} is malformed") from exc
    return value


class _SealedCapability:
    """Copy/pickle guard shared by short-lived one-shot contract records."""

    __slots__ = ("__weakref__",)

    def __copy__(self):
        raise TypeError(f"{type(self).__name__} cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError(f"{type(self).__name__} cannot be copied")

    def __reduce__(self):
        raise TypeError(f"{type(self).__name__} cannot be pickled")

    def __reduce_ex__(self, protocol: object):
        del protocol
        raise TypeError(f"{type(self).__name__} cannot be pickled")


def _require_sealed(value: object, record_name: str) -> None:
    if not _is_capability_seal(value):
        raise EntryRiskContractError(f"{record_name} is factory-only")


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
        _validate_broker_contract(self)


@dataclass(frozen=True, slots=True)
class EntrySignal(_SealedCapability):
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
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "entry signal")
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
        _validate_broker_contract(self.broker_contract, "signal broker contract")
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


def build_entry_signal(
    *,
    signal_id: str,
    portfolio_id: str,
    symbol: str,
    side: EntrySide,
    source: SignalSource,
    confidence_fraction: Decimal,
    requested_position_fraction: Decimal,
    source_data_version: str,
    broker_contract: EntryBrokerContractIdentity,
    transport_generation: str,
    observed_at: datetime,
    expires_at: datetime,
) -> EntrySignal:
    """Mint one immutable, single-use entry signal."""

    return _mint_capability(
        EntrySignal,
        {
            "signal_id": signal_id,
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "side": side,
            "source": source,
            "confidence_fraction": confidence_fraction,
            "requested_position_fraction": requested_position_fraction,
            "source_data_version": source_data_version,
            "broker_contract": broker_contract,
            "transport_generation": transport_generation,
            "observed_at": observed_at,
            "expires_at": expires_at,
        },
    )


@dataclass(frozen=True, slots=True)
class EntryIntent(_SealedCapability):
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
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "entry intent")
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
        _validate_broker_contract(self.broker_contract, "entry intent broker contract")
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
    signal = _consume_capability(signal, EntrySignal)
    created = _timestamp(created_at, "intent created_at")
    if created < signal.observed_at or created >= signal.expires_at:
        raise EntryRiskContractError("signal is not active at intent creation")
    return _mint_capability(
        EntryIntent,
        {
            "intent_id": intent_id,
            "signal_id": signal.signal_id,
            "portfolio_id": signal.portfolio_id,
            "symbol": signal.symbol,
            "side": signal.side,
            "source": signal.source,
            "confidence_fraction": signal.confidence_fraction,
            "requested_position_fraction": signal.requested_position_fraction,
            "source_data_version": signal.source_data_version,
            "broker_contract": signal.broker_contract,
            "transport_generation": signal.transport_generation,
            "quote_refresh_request_id": quote_refresh_request_id,
            "created_at": created,
            "expires_at": signal.expires_at,
        },
    )


@dataclass(frozen=True, slots=True)
class LiveBrokerQuoteSourceCapability(_SealedCapability):
    """One producer-owned broker quote receipt transferable exactly once."""

    producer_id: str
    quote_id: str
    refresh_request_id: str
    symbol: str
    broker_contract: EntryBrokerContractIdentity
    price_usd: Decimal
    observed_at: datetime
    transport_generation: str
    source: RefreshedQuoteSource
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "live broker quote source")
        for field_name in ("producer_id", "quote_id", "refresh_request_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        _validate_broker_contract(self.broker_contract, "quote broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("quote symbol does not match broker contract")
        object.__setattr__(self, "price_usd", _decimal(self.price_usd, "price_usd", positive=True))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "quote observed_at"))
        if self.source is not RefreshedQuoteSource.LIVE_BROKER:
            raise EntryRiskContractError("refreshed quote source must be live-broker")


@dataclass(frozen=True, slots=True)
class RefreshedQuoteEvidence(_SealedCapability):
    producer_id: str
    quote_id: str
    refresh_request_id: str
    symbol: str
    broker_contract: EntryBrokerContractIdentity
    price_usd: Decimal
    observed_at: datetime
    transport_generation: str
    source: RefreshedQuoteSource
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "refreshed quote evidence")
        for field_name in ("producer_id", "quote_id", "refresh_request_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        _validate_broker_contract(self.broker_contract, "quote broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("quote symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        object.__setattr__(self, "price_usd", _decimal(self.price_usd, "price_usd", positive=True))
        object.__setattr__(self, "observed_at", _timestamp(self.observed_at, "quote observed_at"))
        if self.source is not RefreshedQuoteSource.LIVE_BROKER:
            raise EntryRiskContractError("refreshed quote source must be live-broker")


def produce_live_broker_quote_source(
    *,
    producer_id: str,
    quote_id: str,
    refresh_request_id: str,
    broker_contract: EntryBrokerContractIdentity,
    price_usd: Decimal,
    observed_at: datetime,
    transport_generation: str,
) -> LiveBrokerQuoteSourceCapability:
    """Mint the typed source capability a broker quote producer transfers."""

    _validate_broker_contract(broker_contract, "quote source broker contract")
    return _mint_capability(
        LiveBrokerQuoteSourceCapability,
        {
            "producer_id": producer_id,
            "quote_id": quote_id,
            "refresh_request_id": refresh_request_id,
            "symbol": broker_contract.symbol,
            "broker_contract": broker_contract,
            "price_usd": price_usd,
            "observed_at": observed_at,
            "transport_generation": transport_generation,
            "source": RefreshedQuoteSource.LIVE_BROKER,
        },
    )


def build_refreshed_quote_evidence(
    *,
    source_capability: LiveBrokerQuoteSourceCapability,
) -> RefreshedQuoteEvidence:
    """Build an immutable quote bound to one exact contract and transport."""

    source = _consume_capability(source_capability, LiveBrokerQuoteSourceCapability)
    return _mint_capability(
        RefreshedQuoteEvidence,
        {
            "producer_id": source.producer_id,
            "quote_id": source.quote_id,
            "refresh_request_id": source.refresh_request_id,
            "symbol": source.symbol,
            "broker_contract": source.broker_contract,
            "price_usd": source.price_usd,
            "observed_at": source.observed_at,
            "transport_generation": source.transport_generation,
            "source": source.source,
        },
    )


@dataclass(frozen=True, slots=True)
class CorrelationEvidence(_SealedCapability):
    portfolio_id: str
    symbol: str
    source_data_version: str
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    complete: bool
    existing_position_count: int
    max_absolute_correlation: Decimal
    observed_at: datetime
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "correlation evidence")
        object.__setattr__(self, "portfolio_id", _identifier(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(
            self,
            "source_data_version",
            _identifier(self.source_data_version, "source_data_version"),
        )
        _validate_broker_contract(self.broker_contract, "correlation broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("correlation symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
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


def build_correlation_evidence(
    *,
    portfolio_id: str,
    symbol: str,
    source_data_version: str,
    broker_contract: EntryBrokerContractIdentity,
    transport_generation: str,
    complete: bool,
    existing_position_count: int,
    max_absolute_correlation: Decimal,
    observed_at: datetime,
) -> CorrelationEvidence:
    return _mint_capability(
        CorrelationEvidence,
        {
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "source_data_version": source_data_version,
            "broker_contract": broker_contract,
            "transport_generation": transport_generation,
            "complete": complete,
            "existing_position_count": existing_position_count,
            "max_absolute_correlation": max_absolute_correlation,
            "observed_at": observed_at,
        },
    )


@dataclass(frozen=True, slots=True)
class LiquidityEvidence(_SealedCapability):
    portfolio_id: str
    symbol: str
    source_data_version: str
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    complete: bool
    average_daily_dollar_volume_usd: Decimal
    observed_at: datetime
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "liquidity evidence")
        object.__setattr__(self, "portfolio_id", _identifier(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        object.__setattr__(
            self,
            "source_data_version",
            _identifier(self.source_data_version, "source_data_version"),
        )
        _validate_broker_contract(self.broker_contract, "liquidity broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("liquidity symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
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


def build_liquidity_evidence(
    *,
    portfolio_id: str,
    symbol: str,
    source_data_version: str,
    broker_contract: EntryBrokerContractIdentity,
    transport_generation: str,
    complete: bool,
    average_daily_dollar_volume_usd: Decimal,
    observed_at: datetime,
) -> LiquidityEvidence:
    return _mint_capability(
        LiquidityEvidence,
        {
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "source_data_version": source_data_version,
            "broker_contract": broker_contract,
            "transport_generation": transport_generation,
            "complete": complete,
            "average_daily_dollar_volume_usd": average_daily_dollar_volume_usd,
            "observed_at": observed_at,
        },
    )


@dataclass(frozen=True, slots=True)
class MLCorroborationEvidence(_SealedCapability):
    signal_id: str
    portfolio_id: str
    symbol: str
    source_data_version: str
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    model_id: str
    corroborated: bool
    confidence_fraction: Decimal
    observed_at: datetime
    expires_at: datetime
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "ML corroboration evidence")
        for field_name in ("signal_id", "portfolio_id", "source_data_version", "model_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        _validate_broker_contract(self.broker_contract, "ML broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("ML symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        _flag(self.corroborated, "ML corroborated")
        object.__setattr__(
            self,
            "confidence_fraction",
            _fraction(self.confidence_fraction, "ML confidence_fraction"),
        )
        observed = _timestamp(self.observed_at, "ML observed_at")
        expires = _timestamp(self.expires_at, "ML expires_at")
        if expires <= observed:
            raise EntryRiskContractError("ML corroboration expiry must follow observation")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)


def build_ml_corroboration_evidence(
    *,
    signal_id: str,
    portfolio_id: str,
    symbol: str,
    source_data_version: str,
    broker_contract: EntryBrokerContractIdentity,
    transport_generation: str,
    model_id: str,
    corroborated: bool,
    confidence_fraction: Decimal,
    observed_at: datetime,
    expires_at: datetime,
) -> MLCorroborationEvidence:
    return _mint_capability(
        MLCorroborationEvidence,
        {
            "signal_id": signal_id,
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "source_data_version": source_data_version,
            "broker_contract": broker_contract,
            "transport_generation": transport_generation,
            "model_id": model_id,
            "corroborated": corroborated,
            "confidence_fraction": confidence_fraction,
            "observed_at": observed_at,
            "expires_at": expires_at,
        },
    )


@dataclass(frozen=True, slots=True)
class EntryRiskEvidence(_SealedCapability):
    portfolio_id: Optional[str] = None
    symbol: Optional[str] = None
    observed_at: Optional[datetime] = None
    quote: Optional[RefreshedQuoteEvidence] = None
    sector: Optional[str] = None
    correlation: Optional[CorrelationEvidence] = None
    liquidity: Optional[LiquidityEvidence] = None
    ml_corroboration: Optional[MLCorroborationEvidence] = None
    portfolio_equity_usd: Optional[Decimal] = None
    cash_available_usd: Optional[Decimal] = None
    buying_power_usd: Optional[Decimal] = None
    current_symbol_gross_notional_usd: Optional[Decimal] = None
    current_sector_gross_notional_usd: Optional[Decimal] = None
    portfolio_gross_notional_usd: Optional[Decimal] = None
    daily_executed_notional_usd: Optional[Decimal] = None
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "entry risk evidence")
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
        if self.quote is not None:
            if type(self.quote) is not RefreshedQuoteEvidence:
                raise EntryRiskContractError("quote evidence is malformed")
            self.quote.__post_init__(_seal)
        if self.sector is not None:
            object.__setattr__(self, "sector", _sector(self.sector))
        if self.correlation is not None:
            if type(self.correlation) is not CorrelationEvidence:
                raise EntryRiskContractError("correlation evidence is malformed")
            self.correlation.__post_init__(_seal)
        if self.liquidity is not None:
            if type(self.liquidity) is not LiquidityEvidence:
                raise EntryRiskContractError("liquidity evidence is malformed")
            self.liquidity.__post_init__(_seal)
        if self.ml_corroboration is not None:
            if type(self.ml_corroboration) is not MLCorroborationEvidence:
                raise EntryRiskContractError("ML corroboration evidence is malformed")
            self.ml_corroboration.__post_init__(_seal)
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


def build_entry_risk_evidence(
    *,
    portfolio_id: Optional[str] = None,
    symbol: Optional[str] = None,
    observed_at: Optional[datetime] = None,
    quote: Optional[RefreshedQuoteEvidence] = None,
    sector: Optional[str] = None,
    correlation: Optional[CorrelationEvidence] = None,
    liquidity: Optional[LiquidityEvidence] = None,
    ml_corroboration: Optional[MLCorroborationEvidence] = None,
    portfolio_equity_usd: Optional[Decimal] = None,
    cash_available_usd: Optional[Decimal] = None,
    buying_power_usd: Optional[Decimal] = None,
    current_symbol_gross_notional_usd: Optional[Decimal] = None,
    current_sector_gross_notional_usd: Optional[Decimal] = None,
    portfolio_gross_notional_usd: Optional[Decimal] = None,
    daily_executed_notional_usd: Optional[Decimal] = None,
) -> EntryRiskEvidence:
    """Transfer exact component evidence into one single-use risk snapshot."""

    if quote is not None:
        quote = _consume_capability(quote, RefreshedQuoteEvidence)
    if correlation is not None:
        correlation = _consume_capability(correlation, CorrelationEvidence)
    if liquidity is not None:
        liquidity = _consume_capability(liquidity, LiquidityEvidence)
    if ml_corroboration is not None:
        ml_corroboration = _consume_capability(
            ml_corroboration,
            MLCorroborationEvidence,
        )
    return _mint_capability(
        EntryRiskEvidence,
        {
            "portfolio_id": portfolio_id,
            "symbol": symbol,
            "observed_at": observed_at,
            "quote": quote,
            "sector": sector,
            "correlation": correlation,
            "liquidity": liquidity,
            "ml_corroboration": ml_corroboration,
            "portfolio_equity_usd": portfolio_equity_usd,
            "cash_available_usd": cash_available_usd,
            "buying_power_usd": buying_power_usd,
            "current_symbol_gross_notional_usd": current_symbol_gross_notional_usd,
            "current_sector_gross_notional_usd": current_sector_gross_notional_usd,
            "portfolio_gross_notional_usd": portfolio_gross_notional_usd,
            "daily_executed_notional_usd": daily_executed_notional_usd,
        },
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
        if position > Decimal("0.02"):
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
            if type(value) is not timedelta or value <= timedelta(0):
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
class RiskDecision(_SealedCapability):
    intent_id: str
    signal_id: str
    portfolio_id: str
    symbol: str
    side: EntrySide
    broker_contract: EntryBrokerContractIdentity
    transport_generation: str
    evaluated_at: datetime
    expires_at: datetime
    risk_approved: bool
    reasons: tuple[RiskReason, ...]
    approved_quantity: int
    approved_notional_usd: Decimal
    quote_id: Optional[str]
    limiting_capacity: Optional[LimitingCapacity]
    authorizes_order_submission: bool = False
    runtime_integration_ready: bool = False
    schema_version: int = ENTRY_RISK_CONTRACT_VERSION
    _seal: InitVar[object] = None

    def __post_init__(self, _seal: object) -> None:
        _require_sealed(_seal, "risk decision")
        _schema_version(self.schema_version)
        for field_name in ("intent_id", "signal_id", "portfolio_id"):
            object.__setattr__(self, field_name, _identifier(getattr(self, field_name), field_name))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if type(self.side) is not EntrySide:
            raise EntryRiskContractError("risk decision side is invalid")
        _validate_broker_contract(self.broker_contract, "risk decision broker contract")
        if self.broker_contract.symbol != self.symbol:
            raise EntryRiskContractError("risk decision symbol does not match broker contract")
        object.__setattr__(
            self,
            "transport_generation",
            _transport_generation(self.transport_generation, "transport_generation"),
        )
        _flag(self.risk_approved, "risk_approved")
        evaluated = _timestamp(self.evaluated_at, "evaluated_at")
        expires = _timestamp(self.expires_at, "risk decision expires_at")
        if self.risk_approved and expires <= evaluated:
            raise EntryRiskContractError("approved risk decision expiry must follow evaluation")
        object.__setattr__(self, "evaluated_at", evaluated)
        object.__setattr__(self, "expires_at", expires)
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


class ConsumedRiskDecision(tuple):
    """Immutable terminal snapshot returned after one semantic decision consume."""

    __slots__ = ()
    _VALUE_COUNT = 16

    def __new__(
        cls,
        values: tuple[object, ...],
        *,
        _seal: object = None,
    ) -> ConsumedRiskDecision:
        _require_sealed(_seal, "consumed risk decision")
        if type(values) is not tuple or len(values) != cls._VALUE_COUNT:
            raise EntryRiskContractError("consumed risk decision snapshot is malformed")
        return cast(ConsumedRiskDecision, tuple.__new__(cls, values))

    @property
    def intent_id(self) -> str:
        return cast(str, self[0])

    @property
    def signal_id(self) -> str:
        return cast(str, self[1])

    @property
    def portfolio_id(self) -> str:
        return cast(str, self[2])

    @property
    def symbol(self) -> str:
        return cast(str, self[3])

    @property
    def side(self) -> EntrySide:
        return cast(EntrySide, self[4])

    @property
    def broker_contract(self) -> tuple[object, ...]:
        return cast(tuple[object, ...], self[5])

    @property
    def transport_generation(self) -> str:
        return cast(str, self[6])

    @property
    def evaluated_at(self) -> datetime:
        return cast(datetime, self[7])

    @property
    def risk_approved(self) -> bool:
        return cast(bool, self[8])

    @property
    def reasons(self) -> tuple[RiskReason, ...]:
        return cast(tuple[RiskReason, ...], self[9])

    @property
    def approved_quantity(self) -> int:
        return cast(int, self[10])

    @property
    def approved_notional_usd(self) -> Decimal:
        return cast(Decimal, self[11])

    @property
    def quote_id(self) -> Optional[str]:
        return cast(Optional[str], self[12])

    @property
    def limiting_capacity(self) -> Optional[LimitingCapacity]:
        return cast(Optional[LimitingCapacity], self[13])

    @property
    def schema_version(self) -> int:
        return cast(int, self[14])

    @property
    def expires_at(self) -> datetime:
        return cast(datetime, self[15])

    @property
    def authorizes_order_submission(self) -> bool:
        return False

    @property
    def runtime_integration_ready(self) -> bool:
        return False

    def __copy__(self):
        raise TypeError("ConsumedRiskDecision cannot be copied")

    def __deepcopy__(self, memo: object):
        del memo
        raise TypeError("ConsumedRiskDecision cannot be copied")

    def __reduce__(self):
        raise TypeError("ConsumedRiskDecision cannot be pickled")

    def __reduce_ex__(self, protocol: object):
        del protocol
        raise TypeError("ConsumedRiskDecision cannot be pickled")


def _rejected(
    intent: EntryIntent,
    evaluated_at: datetime,
    reasons: list[RiskReason],
    quote: Optional[RefreshedQuoteEvidence],
) -> RiskDecision:
    ordered = tuple(dict.fromkeys(reasons))
    return _mint_capability(
        RiskDecision,
        {
            "intent_id": intent.intent_id,
            "signal_id": intent.signal_id,
            "portfolio_id": intent.portfolio_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "broker_contract": intent.broker_contract,
            "transport_generation": intent.transport_generation,
            "evaluated_at": evaluated_at,
            "expires_at": intent.expires_at,
            "risk_approved": False,
            "reasons": ordered,
            "approved_quantity": 0,
            "approved_notional_usd": Decimal(0),
            "quote_id": None if quote is None else quote.quote_id,
            "limiting_capacity": None,
        },
    )


def _contract_state(contract: EntryBrokerContractIdentity) -> tuple[object, ...]:
    return (
        contract.con_id,
        contract.symbol,
        contract.local_symbol,
        contract.security_type,
        contract.currency,
        contract.exchange,
        contract.primary_exchange,
        contract.trading_class,
    )


def _risk_decision_replay_key(decision: object) -> tuple[object, ...]:
    if type(decision) is not RiskDecision:
        raise EntryRiskContractError("exact RiskDecision is required for replay protection")
    return (
        "risk-decision-v1",
        decision.schema_version,
        decision.portfolio_id,
        decision.intent_id,
    )


def _optional_decimal_state(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def _optional_time_state(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat(timespec="microseconds")


def _capability_state(capability: object) -> tuple[object, ...]:
    """Return an immutable exact snapshot used by the sealed registry."""

    if type(capability) is EntrySignal:
        signal = cast(EntrySignal, capability)
        return (
            "signal",
            signal.signal_id,
            signal.portfolio_id,
            signal.symbol,
            signal.side.value,
            signal.source.value,
            str(signal.confidence_fraction),
            str(signal.requested_position_fraction),
            signal.source_data_version,
            _contract_state(signal.broker_contract),
            signal.transport_generation,
            signal.observed_at.isoformat(timespec="microseconds"),
            signal.expires_at.isoformat(timespec="microseconds"),
            signal.schema_version,
        )
    if type(capability) is EntryIntent:
        intent = cast(EntryIntent, capability)
        return (
            "intent",
            intent.intent_id,
            intent.signal_id,
            intent.portfolio_id,
            intent.symbol,
            intent.side.value,
            intent.source.value,
            str(intent.confidence_fraction),
            str(intent.requested_position_fraction),
            intent.source_data_version,
            _contract_state(intent.broker_contract),
            intent.transport_generation,
            intent.quote_refresh_request_id,
            intent.created_at.isoformat(timespec="microseconds"),
            intent.expires_at.isoformat(timespec="microseconds"),
            intent.schema_version,
        )
    if type(capability) in {
        LiveBrokerQuoteSourceCapability,
        RefreshedQuoteEvidence,
    }:
        quote = cast(
            LiveBrokerQuoteSourceCapability | RefreshedQuoteEvidence,
            capability,
        )
        return (
            type(quote).__name__,
            quote.producer_id,
            quote.quote_id,
            quote.refresh_request_id,
            quote.symbol,
            _contract_state(quote.broker_contract),
            str(quote.price_usd),
            quote.observed_at.isoformat(timespec="microseconds"),
            quote.transport_generation,
            quote.source.value,
        )
    if type(capability) is CorrelationEvidence:
        correlation = cast(CorrelationEvidence, capability)
        return (
            "correlation",
            correlation.portfolio_id,
            correlation.symbol,
            correlation.source_data_version,
            _contract_state(correlation.broker_contract),
            correlation.transport_generation,
            correlation.complete,
            correlation.existing_position_count,
            str(correlation.max_absolute_correlation),
            correlation.observed_at.isoformat(timespec="microseconds"),
        )
    if type(capability) is LiquidityEvidence:
        liquidity = cast(LiquidityEvidence, capability)
        return (
            "liquidity",
            liquidity.portfolio_id,
            liquidity.symbol,
            liquidity.source_data_version,
            _contract_state(liquidity.broker_contract),
            liquidity.transport_generation,
            liquidity.complete,
            str(liquidity.average_daily_dollar_volume_usd),
            liquidity.observed_at.isoformat(timespec="microseconds"),
        )
    if type(capability) is MLCorroborationEvidence:
        ml = cast(MLCorroborationEvidence, capability)
        return (
            "ml",
            ml.signal_id,
            ml.portfolio_id,
            ml.symbol,
            ml.source_data_version,
            _contract_state(ml.broker_contract),
            ml.transport_generation,
            ml.model_id,
            ml.corroborated,
            str(ml.confidence_fraction),
            ml.observed_at.isoformat(timespec="microseconds"),
            ml.expires_at.isoformat(timespec="microseconds"),
        )
    if type(capability) is EntryRiskEvidence:
        evidence = cast(EntryRiskEvidence, capability)
        return (
            "risk-evidence",
            evidence.portfolio_id,
            evidence.symbol,
            _optional_time_state(evidence.observed_at),
            None if evidence.quote is None else _capability_state(evidence.quote),
            evidence.sector,
            None if evidence.correlation is None else _capability_state(evidence.correlation),
            None if evidence.liquidity is None else _capability_state(evidence.liquidity),
            (
                None
                if evidence.ml_corroboration is None
                else _capability_state(evidence.ml_corroboration)
            ),
            _optional_decimal_state(evidence.portfolio_equity_usd),
            _optional_decimal_state(evidence.cash_available_usd),
            _optional_decimal_state(evidence.buying_power_usd),
            _optional_decimal_state(evidence.current_symbol_gross_notional_usd),
            _optional_decimal_state(evidence.current_sector_gross_notional_usd),
            _optional_decimal_state(evidence.portfolio_gross_notional_usd),
            _optional_decimal_state(evidence.daily_executed_notional_usd),
        )
    if type(capability) is RiskDecision:
        decision = cast(RiskDecision, capability)
        return (
            "decision",
            decision.intent_id,
            decision.signal_id,
            decision.portfolio_id,
            decision.symbol,
            decision.side.value,
            _contract_state(decision.broker_contract),
            decision.transport_generation,
            decision.evaluated_at.isoformat(timespec="microseconds"),
            decision.expires_at.isoformat(timespec="microseconds"),
            decision.risk_approved,
            tuple(reason.value for reason in decision.reasons),
            decision.approved_quantity,
            str(decision.approved_notional_usd),
            decision.quote_id,
            None if decision.limiting_capacity is None else decision.limiting_capacity.value,
            decision.authorizes_order_submission,
            decision.runtime_integration_ready,
            decision.schema_version,
        )
    raise EntryRiskContractError("unknown sealed capability type")


def assert_and_consume_risk_decision(
    decision: object,
    *,
    consumed_at: datetime,
) -> ConsumedRiskDecision:
    """Consume an exact decision once at a future dormant integration seam."""

    consumption_time = _timestamp(consumed_at, "risk decision consumed_at")
    consumed = _consume_capability(
        decision,
        RiskDecision,
        consumed_at=consumption_time,
    )
    return _transfer_risk_decision(consumed)


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
    return intent.created_at <= observed_at <= evaluated_at and evaluated_at - observed_at < max_age


def _approved_decision_expiry(
    *,
    intent: EntryIntent,
    evidence: EntryRiskEvidence,
    quote: RefreshedQuoteEvidence,
    correlation: CorrelationEvidence,
    liquidity: LiquidityEvidence,
    limits: EntryRiskLimits,
) -> datetime:
    """Return the earliest deadline of every evidence item used for approval."""

    if evidence.observed_at is None:
        raise EntryRiskContractError("approved evidence lacks an account observation time")
    deadlines = [
        intent.expires_at,
        quote.observed_at + limits.max_quote_age,
        evidence.observed_at + limits.max_account_evidence_age,
        correlation.observed_at + limits.max_account_evidence_age,
        liquidity.observed_at + limits.max_account_evidence_age,
    ]
    if evidence.ml_corroboration is not None:
        deadlines.extend(
            (
                evidence.ml_corroboration.expires_at,
                evidence.ml_corroboration.observed_at + limits.max_account_evidence_age,
            )
        )
    return min(deadlines)


def _scoped_market_evidence_matches(
    evidence: object,
    *,
    intent: EntryIntent,
    expected_broker_contract: EntryBrokerContractIdentity,
    expected_transport_generation: str,
) -> bool:
    return (
        getattr(evidence, "portfolio_id", None) == intent.portfolio_id
        and getattr(evidence, "symbol", None) == intent.symbol
        and getattr(evidence, "source_data_version", None) == intent.source_data_version
        and getattr(evidence, "broker_contract", None) == intent.broker_contract
        and getattr(evidence, "broker_contract", None) == expected_broker_contract
        and getattr(evidence, "transport_generation", None) == intent.transport_generation
        and getattr(evidence, "transport_generation", None) == expected_transport_generation
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
    expected_quote_producer_id: str,
    evaluated_at: datetime,
) -> RiskDecision:
    """Evaluate one intent using complete, refreshed evidence and exact units."""

    if type(intent) is not EntryIntent:
        raise EntryRiskContractError("risk evaluation requires an exact EntryIntent")
    if type(evidence) is not EntryRiskEvidence:
        raise EntryRiskContractError("risk evaluation requires exact EntryRiskEvidence")
    if type(limits) is not EntryRiskLimits or type(flags) is not EntryFeatureFlags:
        raise EntryRiskContractError("risk evaluation configuration is malformed")
    try:
        limits.__post_init__()
        flags.__post_init__()
    except EntryRiskContractError:
        raise
    except Exception as exc:
        raise EntryRiskContractError("risk evaluation configuration is malformed") from exc
    _validate_broker_contract(expected_broker_contract, "expected broker contract")
    active_generation = _transport_generation(
        expected_transport_generation,
        "expected_transport_generation",
    )
    active_quote_producer = _identifier(
        expected_quote_producer_id,
        "expected_quote_producer_id",
    )
    intent = _consume_capability(intent, EntryIntent)
    evidence = _consume_capability(evidence, EntryRiskEvidence)
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
    elif not _scoped_market_evidence_matches(
        evidence.correlation,
        intent=intent,
        expected_broker_contract=expected_broker_contract,
        expected_transport_generation=active_generation,
    ):
        reasons.append(RiskReason.EVIDENCE_SCOPE_MISMATCH)
    elif not _evidence_is_current(
        evidence.correlation.observed_at,
        intent=intent,
        evaluated_at=now,
        max_age=limits.max_account_evidence_age,
    ):
        reasons.append(RiskReason.MISSING_CORRELATION)
    if evidence.liquidity is None or not evidence.liquidity.complete:
        reasons.append(RiskReason.MISSING_LIQUIDITY)
    elif not _scoped_market_evidence_matches(
        evidence.liquidity,
        intent=intent,
        expected_broker_contract=expected_broker_contract,
        expected_transport_generation=active_generation,
    ):
        reasons.append(RiskReason.EVIDENCE_SCOPE_MISMATCH)
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
    if quote is not None and quote.producer_id != active_quote_producer:
        reasons.append(RiskReason.QUOTE_SOURCE_MISMATCH)

    ml = evidence.ml_corroboration
    if ml is not None and not _scoped_market_evidence_matches(
        ml,
        intent=intent,
        expected_broker_contract=expected_broker_contract,
        expected_transport_generation=active_generation,
    ):
        reasons.append(RiskReason.EVIDENCE_SCOPE_MISMATCH)
    if intent.source is SignalSource.AI_DISCOVERY:
        if ml is None:
            reasons.append(RiskReason.MISSING_ML_CORROBORATION)
        elif (
            ml.signal_id != intent.signal_id
            or not ml.corroborated
            or ml.confidence_fraction <= 0
            or intent.confidence_fraction <= 0
            or now >= ml.expires_at
            or not _evidence_is_current(
                ml.observed_at,
                intent=intent,
                evaluated_at=now,
                max_age=limits.max_account_evidence_age,
            )
        ):
            reasons.append(RiskReason.ML_CORROBORATION_REQUIRED)

    if reasons:
        return _rejected(intent, now, reasons, quote)

    if (
        quote is None
        or evidence.correlation is None
        or evidence.liquidity is None
        or evidence.portfolio_equity_usd is None
        or evidence.cash_available_usd is None
        or evidence.buying_power_usd is None
        or evidence.current_symbol_gross_notional_usd is None
        or evidence.current_sector_gross_notional_usd is None
        or evidence.portfolio_gross_notional_usd is None
        or evidence.daily_executed_notional_usd is None
    ):
        raise EntryRiskContractError("required risk evidence vanished after validation")

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
    decision_expiry = _approved_decision_expiry(
        intent=intent,
        evidence=evidence,
        quote=quote,
        correlation=evidence.correlation,
        liquidity=evidence.liquidity,
        limits=limits,
    )
    if decision_expiry <= now:
        raise EntryRiskContractError("approved evidence has no remaining freshness window")

    return _mint_capability(
        RiskDecision,
        {
            "intent_id": intent.intent_id,
            "signal_id": intent.signal_id,
            "portfolio_id": intent.portfolio_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "broker_contract": intent.broker_contract,
            "transport_generation": intent.transport_generation,
            "evaluated_at": now,
            "expires_at": decision_expiry,
            "risk_approved": True,
            "reasons": (RiskReason.APPROVED,),
            "approved_quantity": quantity,
            "approved_notional_usd": approved_notional,
            "quote_id": quote.quote_id,
            "limiting_capacity": limiting_capacity,
        },
    )


__all__ = [
    "ENTRY_RISK_CONTRACT_VERSION",
    "GATE_A_MAX_POSITION_FRACTION",
    "CorrelationEvidence",
    "ConsumedRiskDecision",
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
    "LiveBrokerQuoteSourceCapability",
    "MLCorroborationEvidence",
    "RefreshedQuoteEvidence",
    "RefreshedQuoteSource",
    "RiskDecision",
    "RiskReason",
    "SignalSource",
    "assert_and_consume_risk_decision",
    "build_correlation_evidence",
    "build_entry_risk_evidence",
    "build_entry_signal",
    "build_entry_intent",
    "build_liquidity_evidence",
    "build_ml_corroboration_evidence",
    "build_refreshed_quote_evidence",
    "evaluate_entry_intent",
    "produce_live_broker_quote_source",
]

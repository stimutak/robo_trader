"""Versioned, fail-closed market-data contracts for the active runtime."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Iterable, Mapping, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from .market_hours import get_market_session
from .protective_quote_evidence import MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH

CANONICAL_BAR_SCHEMA_VERSION = 1
MAX_MARKET_DATA_AGE_SECONDS = 24 * 60 * 60
_EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")


class MarketDataContractError(ValueError):
    """Market data cannot be admitted to a trading consumer."""


class MarketDataIdentityError(MarketDataContractError):
    """Market data cannot be bound to one exact broker contract."""


class MarketSession(str, Enum):
    PRE_MARKET = "pre-market"
    REGULAR = "regular"
    AFTER_HOURS = "after-hours"


class MarketSessionPolicy(str, Enum):
    REGULAR_ONLY = "regular-only"
    EXTENDED = "extended"


class AdjustmentState(str, Enum):
    UNKNOWN = "unknown"
    RAW = "raw"
    ADJUSTED = "adjusted"


class MarketDataSource(str, Enum):
    IBKR_HISTORICAL_TRADES = "ibkr-historical-trades"
    IBKR_LIVE_LAST_TRADE = "ibkr-live-last-trade"


class BarTimestampSemantics(str, Enum):
    BAR_START = "bar-start"


class BarQualityFlag(str, Enum):
    ZERO_VOLUME = "zero-volume"


def market_data_max_age_seconds(interval_seconds: int) -> float:
    """Return one finite, bounded freshness limit for runtime and UI readers."""

    if type(interval_seconds) is not int or interval_seconds <= 0:
        raise MarketDataContractError("market-data interval must be a positive integer")
    default_max_age = max(180, (interval_seconds * 2) + 30)
    relative_ceiling = max(default_max_age, (interval_seconds * 3) + 60)
    hard_ceiling = min(MAX_MARKET_DATA_AGE_SECONDS, relative_ceiling)
    try:
        configured = float(os.getenv("MARKET_DATA_MAX_AGE_SECONDS", str(default_max_age)))
    except (TypeError, ValueError) as exc:
        raise MarketDataContractError("MARKET_DATA_MAX_AGE_SECONDS must be numeric") from exc
    if not math.isfinite(configured) or configured <= 0 or configured > hard_ceiling:
        raise MarketDataContractError(
            "MARKET_DATA_MAX_AGE_SECONDS must be finite, positive, and no greater "
            f"than {hard_ceiling:g} for this timeframe"
        )
    return configured


@dataclass(frozen=True, slots=True)
class BrokerProtectiveQuote:
    """One exact live last-trade quote from a current IBKR generation."""

    schema_version: int
    symbol: str
    con_id: int
    exchange: str
    primary_exchange: str
    currency: str
    security_type: str
    price: Decimal
    source_timestamp: datetime
    retrieval_timestamp: datetime
    session: MarketSession
    source: MarketDataSource
    source_event_id: str
    transport_generation: str
    market_data_type: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise MarketDataContractError("protective quote schema version is unsupported")
        if not isinstance(self.symbol, str) or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9._-]{0,31}", self.symbol
        ):
            raise MarketDataIdentityError("protective quote symbol is malformed")
        if type(self.con_id) is not int or self.con_id <= 0:
            raise MarketDataIdentityError("protective quote con_id is invalid")
        if (
            self.exchange != "SMART"
            or not self.primary_exchange
            or self.currency != "USD"
            or self.security_type != "STK"
        ):
            raise MarketDataIdentityError("protective quote contract identity is invalid")
        if type(self.price) is not Decimal or not self.price.is_finite() or self.price <= 0:
            raise MarketDataContractError("protective quote price is invalid")
        for field_name in ("source_timestamp", "retrieval_timestamp"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise MarketDataContractError(f"protective quote {field_name} is invalid")
            object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        if type(self.session) is not MarketSession:
            raise MarketDataContractError("protective quote session is invalid")
        if self.source_timestamp > self.retrieval_timestamp:
            raise MarketDataContractError("protective quote event follows retrieval time")
        if self.source is not MarketDataSource.IBKR_LIVE_LAST_TRADE:
            raise MarketDataContractError("protective quote source is not live IBKR last trade")
        if type(self.market_data_type) is not int or self.market_data_type != 1:
            raise MarketDataContractError("protective quote market data is not live")
        if (
            not isinstance(self.source_event_id, str)
            or not self.source_event_id
            or self.source_event_id != self.source_event_id.strip()
            or len(self.source_event_id) > MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH
        ):
            raise MarketDataIdentityError("protective quote source_event_id is malformed")
        if (
            not isinstance(self.transport_generation, str)
            or not self.transport_generation
            or self.transport_generation != self.transport_generation.strip()
            or len(self.transport_generation) > 256
        ):
            raise MarketDataIdentityError("protective quote transport_generation is malformed")


def bar_interval_seconds(bar_size: str) -> int:
    """Return the requested interval in seconds or reject ambiguity."""

    normalized = str(bar_size).strip().lower()
    values = normalized.split()
    if not values:
        raise MarketDataContractError("bar size is missing")
    try:
        amount = int(values[0])
    except (TypeError, ValueError) as exc:
        raise MarketDataContractError("bar size amount is invalid") from exc
    if amount <= 0:
        raise MarketDataContractError("bar size amount must be positive")
    unit = values[1] if len(values) > 1 else "min"
    if unit.startswith("sec"):
        seconds = amount
    elif unit.startswith("min"):
        seconds = amount * 60
    elif unit.startswith("hour"):
        seconds = amount * 3600
    elif unit.startswith("day"):
        seconds = amount * 86400
    else:
        raise MarketDataContractError("bar size unit is unsupported")
    if (
        seconds >= 86400
        or "day" in normalized
        or "week" in normalized
        or "month" in normalized
        or re.fullmatch(r"\d+\s*(d|w|wk|wks|mo|mos)", normalized) is not None
    ):
        raise MarketDataContractError(
            "active trading requires timezone-aware intraday datetime bars; "
            "daily-or-coarser bars are unsupported"
        )
    return seconds


def _decimal(value: Any, field_name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise MarketDataContractError(f"broker bar {field_name} is invalid")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise MarketDataContractError(f"broker bar {field_name} is invalid") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise MarketDataContractError(f"broker bar {field_name} is invalid")
    return result


def _aware_utc(value: Any, field_name: str) -> datetime:
    if isinstance(value, (int, float, bool)):
        raise MarketDataContractError(f"numeric {field_name} is ambiguous")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketDataContractError(f"invalid {field_name}: {value!r}") from exc
    if pd.isna(parsed):
        raise MarketDataContractError(f"{field_name} is missing")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MarketDataContractError(f"timezone-naive {field_name} rejected")
    return parsed.tz_convert("UTC").to_pydatetime()


@dataclass(frozen=True, slots=True)
class HistoricalBarContract:
    """Identity, source, session, and retrieval metadata for one bar batch."""

    schema_version: int
    symbol: str
    con_id: int
    exchange: str
    primary_exchange: str
    timezone_name: str
    timeframe: str
    session_policy: MarketSessionPolicy
    source: MarketDataSource
    retrieval_time: datetime
    broker_time: datetime
    adjustment_state: AdjustmentState
    transport_generation: str
    timestamp_semantics: BarTimestampSemantics
    use_rth: bool
    what_to_show: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CANONICAL_BAR_SCHEMA_VERSION
        ):
            raise MarketDataContractError("canonical bar schema version is unsupported")
        if not isinstance(self.symbol, str) or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9._-]{0,31}", self.symbol
        ):
            raise MarketDataIdentityError("canonical bar symbol is malformed")
        if type(self.con_id) is not int or self.con_id <= 0:
            raise MarketDataIdentityError("canonical bar con_id is invalid")
        if self.exchange != "SMART" or not re.fullmatch(
            r"[A-Z0-9][A-Z0-9._:/-]{0,63}", self.primary_exchange
        ):
            raise MarketDataIdentityError("canonical bar exchange identity is incomplete")
        if self.timezone_name != "UTC":
            raise MarketDataContractError("canonical bar timezone must be UTC")
        bar_interval_seconds(self.timeframe)
        if self.use_rth is not (self.session_policy is MarketSessionPolicy.REGULAR_ONLY):
            raise MarketDataContractError("bar session policy and use_rth disagree")
        for field_name in ("retrieval_time", "broker_time"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise MarketDataContractError(f"{field_name} must be timezone-aware")
            object.__setattr__(self, field_name, value.astimezone(timezone.utc))
        if abs((self.broker_time - self.retrieval_time).total_seconds()) > 120:
            raise MarketDataContractError(
                "bar broker and retrieval timestamps exceed clock-skew tolerance"
            )
        if type(self.session_policy) is not MarketSessionPolicy:
            raise MarketDataContractError("bar session policy is invalid")
        if self.source is not MarketDataSource.IBKR_HISTORICAL_TRADES:
            raise MarketDataContractError("historical bar source is invalid")
        if type(self.adjustment_state) is not AdjustmentState:
            raise MarketDataContractError("bar adjustment state is invalid")
        if self.timestamp_semantics is not BarTimestampSemantics.BAR_START:
            raise MarketDataContractError("bar timestamp semantics are invalid")
        if self.what_to_show != "TRADES":
            raise MarketDataContractError("canonical historical bars must be TRADES")
        if (
            not isinstance(self.transport_generation, str)
            or not self.transport_generation
            or self.transport_generation != self.transport_generation.strip()
            or len(self.transport_generation) > 128
        ):
            raise MarketDataIdentityError("bar transport generation is malformed")


@dataclass(frozen=True, slots=True)
class CanonicalBar:
    """One immutable validated OHLCV event."""

    contract: HistoricalBarContract
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    session: MarketSession
    quality_flags: tuple[BarQualityFlag, ...] = ()

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise MarketDataContractError("canonical bar timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        for field_name in ("open", "high", "low", "close"):
            value = getattr(self, field_name)
            if type(value) is not Decimal or not value.is_finite() or value <= 0:
                raise MarketDataContractError(f"canonical bar {field_name} is invalid")
        if self.high < max(self.open, self.low, self.close) or self.low > min(
            self.open, self.high, self.close
        ):
            raise MarketDataContractError("broker OHLC ordering is invalid")
        if type(self.volume) is not int or self.volume < 0:
            raise MarketDataContractError("broker volume cannot be negative")
        if type(self.session) is not MarketSession:
            raise MarketDataContractError("canonical bar session is invalid")
        if (
            self.contract.session_policy is MarketSessionPolicy.REGULAR_ONLY
            and self.session is not MarketSession.REGULAR
        ):
            raise MarketDataContractError("canonical bar violates regular-only policy")
        expected_flags = (BarQualityFlag.ZERO_VOLUME,) if self.volume == 0 else ()
        if self.quality_flags != expected_flags:
            raise MarketDataContractError("canonical bar quality flags are inconsistent")


@dataclass(frozen=True, slots=True)
class CanonicalBarBatch:
    """An atomic contract-lineage batch admitted to runtime consumers."""

    contract: HistoricalBarContract
    bars: tuple[CanonicalBar, ...]

    def __post_init__(self) -> None:
        if not self.bars:
            raise MarketDataContractError("canonical bar batch is empty")
        if any(bar.contract is not self.contract for bar in self.bars):
            raise MarketDataIdentityError("canonical bars do not share exact batch identity")

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "open": [float(bar.open) for bar in self.bars],
                "high": [float(bar.high) for bar in self.bars],
                "low": [float(bar.low) for bar in self.bars],
                "close": [float(bar.close) for bar in self.bars],
                "volume": [bar.volume for bar in self.bars],
            },
            index=pd.DatetimeIndex([bar.timestamp for bar in self.bars], name="timestamp"),
        )
        frame.attrs["market_data_contract"] = self.contract
        frame.attrs["canonical_bar_batch"] = self
        frame.attrs["market_data_quality_flags"] = tuple(
            tuple(flag.value for flag in bar.quality_flags) for bar in self.bars
        )
        return frame

    def storage_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "schema_version": self.contract.schema_version,
                "symbol": self.contract.symbol,
                "con_id": self.contract.con_id,
                "exchange": self.contract.exchange,
                "primary_exchange": self.contract.primary_exchange,
                "timeframe": self.contract.timeframe,
                "interval_seconds": bar_interval_seconds(self.contract.timeframe),
                "timezone_name": self.contract.timezone_name,
                "session_policy": self.contract.session_policy.value,
                "timestamp": bar.timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": bar.volume,
                "session": bar.session.value,
                "source": self.contract.source.value,
                "retrieval_timestamp": self.contract.retrieval_time.isoformat(),
                "broker_timestamp": self.contract.broker_time.isoformat(),
                "adjustment_state": self.contract.adjustment_state.value,
                "quality_flags": ",".join(flag.value for flag in bar.quality_flags),
                "transport_generation": self.contract.transport_generation,
                "timestamp_semantics": self.contract.timestamp_semantics.value,
                "use_rth": self.contract.use_rth,
                "what_to_show": self.contract.what_to_show,
            }
            for bar in self.bars
        ]


CANONICAL_STORAGE_KEYS = frozenset(
    {
        "schema_version",
        "symbol",
        "con_id",
        "exchange",
        "primary_exchange",
        "timeframe",
        "interval_seconds",
        "timezone_name",
        "session_policy",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "session",
        "source",
        "retrieval_timestamp",
        "broker_timestamp",
        "adjustment_state",
        "quality_flags",
        "transport_generation",
        "timestamp_semantics",
        "use_rth",
        "what_to_show",
    }
)


def validate_canonical_storage_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct and validate one complete canonical persistence record."""

    if not isinstance(row, Mapping) or set(row) != CANONICAL_STORAGE_KEYS:
        raise MarketDataContractError("canonical storage row schema is invalid")
    try:
        session_policy = MarketSessionPolicy(row["session_policy"])
        source = MarketDataSource(row["source"])
        adjustment_state = AdjustmentState(row["adjustment_state"])
        timestamp_semantics = BarTimestampSemantics(row["timestamp_semantics"])
        session = MarketSession(row["session"])
    except (TypeError, ValueError) as exc:
        raise MarketDataContractError("canonical storage enum value is invalid") from exc
    use_rth = row["use_rth"]
    if type(use_rth) is not bool:
        raise MarketDataContractError("canonical storage use_rth must be boolean")
    interval_seconds = bar_interval_seconds(row["timeframe"])
    if type(row["interval_seconds"]) is not int or row["interval_seconds"] != interval_seconds:
        raise MarketDataContractError("canonical storage interval is inconsistent")
    contract = HistoricalBarContract(
        schema_version=row["schema_version"],
        symbol=row["symbol"],
        con_id=row["con_id"],
        exchange=row["exchange"],
        primary_exchange=row["primary_exchange"],
        timezone_name=row["timezone_name"],
        timeframe=row["timeframe"],
        session_policy=session_policy,
        source=source,
        retrieval_time=_aware_utc(row["retrieval_timestamp"], "retrieval timestamp"),
        broker_time=_aware_utc(row["broker_timestamp"], "broker timestamp"),
        adjustment_state=adjustment_state,
        transport_generation=row["transport_generation"],
        timestamp_semantics=timestamp_semantics,
        use_rth=use_rth,
        what_to_show=row["what_to_show"],
    )
    timestamp = _aware_utc(row["timestamp"], "canonical event timestamp")
    if timestamp > contract.retrieval_time:
        raise MarketDataContractError("canonical event follows retrieval timestamp")
    if get_market_session(timestamp) != session.value:
        raise MarketDataContractError("canonical storage session contradicts event time")
    volume = row["volume"]
    if type(volume) is not int:
        raise MarketDataContractError("canonical storage volume must be an integer")
    flags_text = row["quality_flags"]
    if not isinstance(flags_text, str):
        raise MarketDataContractError("canonical storage quality flags are malformed")
    try:
        quality_flags = tuple(BarQualityFlag(flag) for flag in flags_text.split(",") if flag)
    except ValueError as exc:
        raise MarketDataContractError("canonical storage quality flag is invalid") from exc
    bar = CanonicalBar(
        contract=contract,
        timestamp=timestamp,
        open=_decimal(row["open"], "open", positive=True),
        high=_decimal(row["high"], "high", positive=True),
        low=_decimal(row["low"], "low", positive=True),
        close=_decimal(row["close"], "close", positive=True),
        volume=volume,
        session=session,
        quality_flags=quality_flags,
    )
    return {
        **row,
        "timestamp": bar.timestamp.isoformat(),
        "open": float(bar.open),
        "high": float(bar.high),
        "low": float(bar.low),
        "close": float(bar.close),
        "session": bar.session.value,
        "retrieval_timestamp": contract.retrieval_time.isoformat(),
        "broker_timestamp": contract.broker_time.isoformat(),
        "quality_flags": ",".join(flag.value for flag in bar.quality_flags),
        "use_rth": contract.use_rth,
    }


def canonicalize_historical_bars(
    *,
    symbol: str,
    records: Iterable[Mapping[str, Any]],
    lineage: Any,
    bar_size: str,
    use_rth: bool,
    what_to_show: str,
    now: Optional[datetime] = None,
) -> CanonicalBarBatch:
    """Validate wire records and bind them atomically to broker lineage."""

    interval_seconds = bar_interval_seconds(bar_size)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError("validation clock must be timezone-aware")
    current = current.astimezone(timezone.utc)
    requested = str(symbol).strip().upper()
    if getattr(lineage, "symbol", None) != requested:
        raise MarketDataIdentityError("historical lineage symbol mismatch")
    con_id = getattr(lineage, "con_id", None)
    exchange = getattr(lineage, "exchange", None)
    primary_exchange = getattr(lineage, "primary_exchange", None)
    retrieval_time = getattr(lineage, "retrieval_timestamp", None)
    broker_time = getattr(lineage, "broker_timestamp", None)
    transport_generation = getattr(lineage, "transport_generation", None)
    if type(con_id) is not int:
        raise MarketDataIdentityError("historical lineage con_id is invalid")
    if type(exchange) is not str or type(primary_exchange) is not str:
        raise MarketDataIdentityError("historical lineage exchange is invalid")
    if not isinstance(retrieval_time, datetime) or not isinstance(broker_time, datetime):
        raise MarketDataIdentityError("historical lineage clocks are invalid")
    if type(transport_generation) is not str:
        raise MarketDataIdentityError("historical lineage generation is invalid")
    contract = HistoricalBarContract(
        schema_version=CANONICAL_BAR_SCHEMA_VERSION,
        symbol=requested,
        con_id=con_id,
        exchange=exchange,
        primary_exchange=primary_exchange,
        timezone_name="UTC",
        timeframe=bar_size,
        session_policy=(
            MarketSessionPolicy.REGULAR_ONLY if use_rth else MarketSessionPolicy.EXTENDED
        ),
        source=MarketDataSource.IBKR_HISTORICAL_TRADES,
        retrieval_time=retrieval_time,
        broker_time=broker_time,
        adjustment_state=AdjustmentState.UNKNOWN,
        transport_generation=transport_generation,
        timestamp_semantics=BarTimestampSemantics.BAR_START,
        use_rth=use_rth,
        what_to_show=what_to_show,
    )
    bars: list[CanonicalBar] = []
    previous: Optional[CanonicalBar] = None
    seen: set[datetime] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise MarketDataContractError("broker bar record is malformed")
        timestamp = _aware_utc(raw.get("date"), "broker timestamp")
        if timestamp in seen:
            raise MarketDataContractError("duplicate broker timestamps rejected")
        if previous is not None and timestamp <= previous.timestamp:
            raise MarketDataContractError("reversed or out-of-order broker timestamps rejected")
        if timestamp > current:
            raise MarketDataContractError("broker bars contain a future timestamp")
        session_text = get_market_session(timestamp)
        if session_text == "closed":
            raise MarketDataContractError("broker bar is outside an admitted market session")
        session = MarketSession(session_text)
        if use_rth and session is not MarketSession.REGULAR:
            raise MarketDataContractError("regular-hours response contains an extended-session bar")
        volume_decimal = _decimal(raw.get("volume"), "volume")
        if volume_decimal != volume_decimal.to_integral_value():
            raise MarketDataContractError("broker bar volume is invalid")
        try:
            volume = int(volume_decimal)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketDataContractError("broker bar volume is invalid") from exc
        bar = CanonicalBar(
            contract=contract,
            timestamp=timestamp,
            open=_decimal(raw.get("open"), "open", positive=True),
            high=_decimal(raw.get("high"), "high", positive=True),
            low=_decimal(raw.get("low"), "low", positive=True),
            close=_decimal(raw.get("close"), "close", positive=True),
            volume=volume,
            session=session,
            quality_flags=((BarQualityFlag.ZERO_VOLUME,) if volume == 0 else ()),
        )
        if previous is not None:
            same_session = (
                previous.session is bar.session
                and previous.timestamp.astimezone(_EXCHANGE_TIMEZONE).date()
                == bar.timestamp.astimezone(_EXCHANGE_TIMEZONE).date()
            )
            if same_session and (bar.timestamp - previous.timestamp).total_seconds() > (
                interval_seconds * 1.5
            ):
                raise MarketDataContractError("unexpected in-session broker bar gap rejected")
        seen.add(timestamp)
        bars.append(bar)
        previous = bar
    if not bars:
        raise MarketDataContractError("canonical bar batch is empty")
    max_age_seconds = market_data_max_age_seconds(interval_seconds)
    age = (current - bars[-1].timestamp).total_seconds()
    if age > max_age_seconds:
        raise MarketDataContractError(f"stale broker bars rejected (age={age:.1f}s)")
    return CanonicalBarBatch(contract=contract, bars=tuple(bars))

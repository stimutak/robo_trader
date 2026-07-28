"""Immutable producer-owned quote evidence for protective reductions.

The stop-loss monitor is the production producer.  Consumers must call
``assert_producer_owned_protective_quote`` before trusting an object: a copied
or caller-constructed dataclass is deliberately not sufficient evidence.

Legacy price callbacks can still be represented, but their explicit
``LEGACY_CALLBACK`` provenance and missing broker lineage keeps them distinct
from a contract-bound live broker quote.  The paper-settlement readiness gate
remains closed until the gateway consumes the stronger form.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH = 128
_TEXT_RE = re.compile(rf"^[^\x00-\x1f\x7f]{{1,{MAX_PROTECTIVE_SOURCE_EVENT_ID_LENGTH}}}$")
_PRODUCER_MARKER = object()
_REGISTRY_KEY = secrets.token_bytes(32)
_REGISTRY_LOCK = threading.Lock()


class ProtectiveQuoteValidationError(ValueError):
    """Protective quote evidence was malformed or lacked producer ownership."""


class ProtectiveQuoteSource(str, Enum):
    """Origin explicitly attached to one accepted protective quote."""

    LIVE_BROKER = "live-broker"
    LEGACY_CALLBACK = "legacy-callback"


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ProtectiveQuoteValidationError(f"{field_name} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.utcoffset() != timedelta(0):  # pragma: no cover - astimezone invariant
        raise ProtectiveQuoteValidationError(f"{field_name} must normalize to UTC")
    return normalized


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or value != value.strip() or not _TEXT_RE.fullmatch(value):
        raise ProtectiveQuoteValidationError(f"{field_name} is malformed")
    return value


@dataclass(frozen=True, repr=False)
class ProtectiveQuoteEvidence:
    """One exact quote accepted and sequenced by a stop-loss monitor."""

    # ``dataclasses.dataclass(..., weakref_slot=True)`` was added in Python
    # 3.11, while this project supports Python 3.10.  Declaring the slots
    # explicitly keeps the evidence immutable, without a ``__dict__``, and
    # weak-referenceable on every supported interpreter.  The generated repr
    # stays disabled so the internal producer marker is never rendered.
    __slots__ = (
        "portfolio_id",
        "symbol",
        "price",
        "source_timestamp",
        "receipt_monotonic",
        "receipt_order",
        "source",
        "con_id",
        "transport_generation",
        "source_event_id",
        "quote_id",
        "_producer_marker",
        "__weakref__",
    )

    portfolio_id: str
    symbol: str
    price: Decimal
    source_timestamp: datetime
    receipt_monotonic: float
    receipt_order: int
    source: ProtectiveQuoteSource
    con_id: Optional[int]
    transport_generation: Optional[str]
    source_event_id: Optional[str]
    quote_id: str
    _producer_marker: object

    def __post_init__(self) -> None:
        _text(self.portfolio_id, "portfolio_id")
        if type(self.symbol) is not str or not _SYMBOL_RE.fullmatch(self.symbol):
            raise ProtectiveQuoteValidationError("symbol is malformed")
        if type(self.price) is not Decimal or not self.price.is_finite() or self.price <= 0:
            raise ProtectiveQuoteValidationError("price must be an exact positive Decimal")
        object.__setattr__(
            self,
            "source_timestamp",
            _utc(self.source_timestamp, "source_timestamp"),
        )
        if (
            type(self.receipt_monotonic) is not float
            or not math.isfinite(self.receipt_monotonic)
            or self.receipt_monotonic < 0.0
        ):
            raise ProtectiveQuoteValidationError(
                "receipt_monotonic must be finite and non-negative"
            )
        if type(self.receipt_order) is not int or self.receipt_order <= 0:
            raise ProtectiveQuoteValidationError("receipt_order must be a positive integer")
        if type(self.source) is not ProtectiveQuoteSource:
            raise ProtectiveQuoteValidationError("source must be ProtectiveQuoteSource")
        if self.con_id is not None and (type(self.con_id) is not int or self.con_id <= 0):
            raise ProtectiveQuoteValidationError("con_id must be a positive integer or None")
        if self.transport_generation is not None:
            _text(self.transport_generation, "transport_generation")
        if self.source_event_id is not None:
            _text(self.source_event_id, "source_event_id")
        if self.source is ProtectiveQuoteSource.LIVE_BROKER:
            if self.con_id is None or self.transport_generation is None:
                raise ProtectiveQuoteValidationError(
                    "live broker quote requires contract and transport lineage"
                )
        elif self.con_id is not None or self.transport_generation is not None:
            raise ProtectiveQuoteValidationError(
                "legacy callback must not claim broker contract or transport lineage"
            )
        if not re.fullmatch(r"quote:v1:[0-9a-f]{64}", self.quote_id):
            raise ProtectiveQuoteValidationError("quote_id is malformed")
        if self._producer_marker is not _PRODUCER_MARKER:
            raise ProtectiveQuoteValidationError("quote lacks producer ownership")

    @property
    def fingerprint(self) -> str:
        """Return the strict 64-hex hash expected by settlement records."""

        return self.quote_id.removeprefix("quote:v1:")

    def canonical_payload(self) -> str:
        """Return the durable canonical quote lineage bound by ``fingerprint``."""

        payload = _evidence_payload(self)
        if not hmac.compare_digest(
            hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            self.fingerprint,
        ):
            raise ProtectiveQuoteValidationError("quote payload does not match fingerprint")
        return payload


_RegistryEntry = tuple[
    weakref.ReferenceType[ProtectiveQuoteEvidence],
    weakref.ReferenceType[object],
    str,
]
_REGISTRY: dict[int, _RegistryEntry] = {}


def _canonical_payload(
    *,
    portfolio_id: str,
    symbol: str,
    price: Decimal,
    source_timestamp: datetime,
    receipt_monotonic: float,
    receipt_order: int,
    source: ProtectiveQuoteSource,
    con_id: Optional[int],
    transport_generation: Optional[str],
    source_event_id: Optional[str],
) -> str:
    return json.dumps(
        {
            "con_id": con_id,
            "portfolio_id": portfolio_id,
            "price": str(price),
            "receipt_monotonic": receipt_monotonic.hex(),
            "receipt_order": receipt_order,
            "source": source.value,
            "source_event_id": source_event_id,
            "source_timestamp": source_timestamp.isoformat(timespec="microseconds"),
            "symbol": symbol,
            "transport_generation": transport_generation,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _evidence_payload(evidence: ProtectiveQuoteEvidence) -> str:
    return _canonical_payload(
        portfolio_id=evidence.portfolio_id,
        symbol=evidence.symbol,
        price=evidence.price,
        source_timestamp=evidence.source_timestamp,
        receipt_monotonic=evidence.receipt_monotonic,
        receipt_order=evidence.receipt_order,
        source=evidence.source,
        con_id=evidence.con_id,
        transport_generation=evidence.transport_generation,
        source_event_id=evidence.source_event_id,
    )


def _digest(evidence: ProtectiveQuoteEvidence) -> str:
    return hmac.new(
        _REGISTRY_KEY,
        f"{_evidence_payload(evidence)}|{evidence.quote_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _discard_evidence(
    object_id: int,
    reference: weakref.ReferenceType[ProtectiveQuoteEvidence],
) -> None:
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _REGISTRY.pop(object_id, None)


def _produce_protective_quote(
    producer: object,
    *,
    portfolio_id: str,
    symbol: str,
    price: Decimal,
    source_timestamp: datetime,
    receipt_monotonic: float,
    receipt_order: int,
    source: ProtectiveQuoteSource,
    con_id: Optional[int] = None,
    transport_generation: Optional[str] = None,
    source_event_id: Optional[str] = None,
) -> ProtectiveQuoteEvidence:
    """Produce evidence only for an exact live ``StopLossMonitor`` owner."""

    # The lazy import avoids a module cycle while preventing arbitrary objects
    # from becoming evidence producers.
    from .stop_loss_monitor import StopLossMonitor

    if type(producer) is not StopLossMonitor:
        raise ProtectiveQuoteValidationError("producer must be exactly StopLossMonitor")
    normalized_time = _utc(source_timestamp, "source_timestamp")
    # Validate every field before canonicalization so malformed enum/string
    # inputs cannot escape as incidental AttributeError/TypeError exceptions.
    ProtectiveQuoteEvidence(
        portfolio_id=portfolio_id,
        symbol=symbol,
        price=price,
        source_timestamp=normalized_time,
        receipt_monotonic=receipt_monotonic,
        receipt_order=receipt_order,
        source=source,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
        quote_id="quote:v1:" + ("0" * 64),
        _producer_marker=_PRODUCER_MARKER,
    )
    payload = _canonical_payload(
        portfolio_id=portfolio_id,
        symbol=symbol,
        price=price,
        source_timestamp=normalized_time,
        receipt_monotonic=receipt_monotonic,
        receipt_order=receipt_order,
        source=source,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
    )
    quote_id = f"quote:v1:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
    evidence = ProtectiveQuoteEvidence(
        portfolio_id=portfolio_id,
        symbol=symbol,
        price=price,
        source_timestamp=normalized_time,
        receipt_monotonic=receipt_monotonic,
        receipt_order=receipt_order,
        source=source,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
        quote_id=quote_id,
        _producer_marker=_PRODUCER_MARKER,
    )
    object_id = id(evidence)

    def discard(reference: weakref.ReferenceType[ProtectiveQuoteEvidence]) -> None:
        _discard_evidence(object_id, reference)

    evidence_reference = weakref.ref(evidence, discard)
    producer_reference = weakref.ref(producer)
    # Digest construction allocates temporary objects and can trigger cyclic
    # garbage collection.  A prior quote's weakref cleanup must never run while
    # this non-reentrant registry lock is held, or it will deadlock trying to
    # remove its own entry (observed on Python 3.10).
    evidence_digest = _digest(evidence)
    with _REGISTRY_LOCK:
        _REGISTRY[object_id] = (evidence_reference, producer_reference, evidence_digest)
    return evidence


def assert_producer_owned_protective_quote(
    evidence: ProtectiveQuoteEvidence,
    *,
    producer: object | None = None,
) -> ProtectiveQuoteEvidence:
    """Return exact registered evidence or reject copies, mutation, and forgery."""

    if type(evidence) is not ProtectiveQuoteEvidence:
        raise ProtectiveQuoteValidationError("exact ProtectiveQuoteEvidence is required")
    evidence_digest = _digest(evidence)
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(id(evidence))
        if registered is None or registered[0]() is not evidence:
            raise ProtectiveQuoteValidationError("quote is not producer-owned")
        registered_producer = registered[1]()
        if registered_producer is None:
            raise ProtectiveQuoteValidationError("quote producer is no longer available")
        if producer is not None and registered_producer is not producer:
            raise ProtectiveQuoteValidationError("quote belongs to a different producer")
        if evidence._producer_marker is not _PRODUCER_MARKER or not hmac.compare_digest(
            registered[2], evidence_digest
        ):
            raise ProtectiveQuoteValidationError("quote changed after production")
    return evidence


def assert_current_authoritative_protective_quote(
    evidence: ProtectiveQuoteEvidence,
    *,
    producer: object,
    expected_portfolio_id: str,
    expected_symbol: str,
    expected_con_id: int,
    expected_transport_generation: str,
    expected_source_event_id: str | None = None,
) -> ProtectiveQuoteEvidence:
    """Revalidate exact immutable live quote evidence before consumption.

    This is the single strict gateway-facing seam.  It trusts neither caller
    clocks nor caller freshness limits: both clocks and ``max_price_age_seconds``
    come from the exact monitor that produced the registered object. A newer
    quote does not retroactively invalidate a still-fresh latched stop crossing;
    producer ownership, immutable payload, contract lineage, and freshness are
    the authority rather than mutable compatibility caches.
    """

    from .stop_loss_monitor import StopLossMonitor

    if type(producer) is not StopLossMonitor:
        raise ProtectiveQuoteValidationError("producer must be exactly StopLossMonitor")
    quote = assert_producer_owned_protective_quote(evidence, producer=producer)
    _text(expected_portfolio_id, "expected_portfolio_id")
    if type(expected_symbol) is not str or not _SYMBOL_RE.fullmatch(expected_symbol):
        raise ProtectiveQuoteValidationError("expected_symbol is malformed")
    if type(expected_con_id) is not int or expected_con_id <= 0:
        raise ProtectiveQuoteValidationError("expected_con_id must be a positive integer")
    _text(expected_transport_generation, "expected_transport_generation")
    if expected_source_event_id is not None:
        _text(expected_source_event_id, "expected_source_event_id")

    if quote.source is not ProtectiveQuoteSource.LIVE_BROKER:
        raise ProtectiveQuoteValidationError("legacy quote is not gateway-authoritative")
    if (
        producer.portfolio_id != expected_portfolio_id
        or quote.portfolio_id != expected_portfolio_id
    ):
        raise ProtectiveQuoteValidationError("quote portfolio does not match")
    if quote.symbol != expected_symbol:
        raise ProtectiveQuoteValidationError("quote symbol does not match")
    if quote.con_id != expected_con_id:
        raise ProtectiveQuoteValidationError("quote contract does not match")
    if quote.transport_generation != expected_transport_generation:
        raise ProtectiveQuoteValidationError("quote transport generation does not match")
    if expected_source_event_id is not None and quote.source_event_id != expected_source_event_id:
        raise ProtectiveQuoteValidationError("quote source event does not match")

    max_age = producer.max_price_age_seconds
    if isinstance(max_age, bool) or not isinstance(max_age, (int, float)):
        raise ProtectiveQuoteValidationError("producer max quote age is malformed")
    max_age_float = float(max_age)
    if not math.isfinite(max_age_float) or max_age_float <= 0.0:
        raise ProtectiveQuoteValidationError("producer max quote age is malformed")
    now = producer._utcnow()
    monotonic_now = producer._monotonic()
    normalized_now = _utc(now, "producer wall clock")
    if type(monotonic_now) is not float or not math.isfinite(monotonic_now) or monotonic_now < 0.0:
        raise ProtectiveQuoteValidationError("producer monotonic clock is malformed")

    wall_age = (normalized_now - quote.source_timestamp).total_seconds()
    monotonic_age = monotonic_now - quote.receipt_monotonic
    if wall_age < 0.0 or monotonic_age < 0.0:
        raise ProtectiveQuoteValidationError("quote clock moved backwards or is in the future")
    if wall_age > max_age_float or monotonic_age > max_age_float:
        raise ProtectiveQuoteValidationError("quote is stale")

    # Recompute the public payload hash as a final durable-audit invariant.
    # Registry verification above separately proves exact object ownership and
    # detects post-production mutation through the keyed in-process digest.
    quote.canonical_payload()
    return quote

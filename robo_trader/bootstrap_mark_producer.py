"""Producer-coupled protective marks for exact-state bootstrap.

This stage accepts no caller-supplied price, artifact, JSON, output path, or
signing capability.  It derives one immutable unsigned accounting mark from an
exact live quote still owned by the ``StopLossMonitor`` that accepted it.  The
quote and runtime database are revalidated immediately before the single
narrow receiver capability is invoked.

The result is deliberately a bootstrap accounting record, not live quote
authority.  Its lineage identifies the originating live broker quote, but the
result itself cannot be passed back through the protective-quote gateway.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import stat
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Generic, Protocol, TypeVar

from .config import PAPER_ONLY_EXECUTION_SOURCE, RuntimeContract
from .market_data_contract import BrokerProtectiveQuote
from .protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    ProtectiveQuoteValidationError,
    assert_current_authoritative_protective_quote,
)
from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
from .stop_loss_monitor import StopLossMonitor

BOOTSTRAP_MARK_SCHEMA_VERSION = 1
BOOTSTRAP_MARK_SOURCE = "pr3-validated-market-data-v1"

_ACCOUNT_SCOPE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_DATABASE_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_PORTFOLIO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_QUOTE_ID = re.compile(r"^quote:v1:[0-9a-f]{64}$")
_RUNTIME_FINGERPRINT = re.compile(r"^[0-9a-f]{16,64}$")
_SOURCE_EVENT_ID = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_TRANSPORT_GENERATION = re.compile(r"^[^\x00-\x1f\x7f]{1,128}$")
_UNSIGNED_MARK_PRODUCER_MARKER = object()
_UNSIGNED_MARK_REGISTRY_KEY = secrets.token_bytes(32)


class BootstrapMarkBlocked(ValueError):
    """No unsigned protective mark may be delivered."""


def _strict_text(value: object, field_name: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or value != value.strip() or not pattern.fullmatch(value):
        raise BootstrapMarkBlocked(f"{field_name} is malformed")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise BootstrapMarkBlocked(f"{field_name} must be a positive integer")
    return value


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise BootstrapMarkBlocked(f"{field_name} must be timezone-aware")
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError) as exc:
        raise BootstrapMarkBlocked(f"{field_name} is invalid") from exc


def _canonical_decimal(value: object) -> str:
    if type(value) is not Decimal or not value.is_finite() or value <= 0:
        raise BootstrapMarkBlocked("mark price must be an exact positive Decimal")
    sign, digits, exponent = value.as_tuple()
    if type(exponent) is not int:  # pragma: no cover - finite Decimal invariant
        raise BootstrapMarkBlocked("mark price has an invalid exponent")
    coefficient = "".join(str(digit) for digit in digits)
    if exponent >= 0:
        rendered = coefficient + ("0" * exponent)
    else:
        split_at = len(coefficient) + exponent
        rendered = (
            "0." + ("0" * -split_at) + coefficient
            if split_at <= 0
            else coefficient[:split_at] + "." + coefficient[split_at:]
        )
        rendered = rendered.rstrip("0").rstrip(".")
    return ("-" if sign else "") + rendered


def _canonical_timestamp(value: datetime) -> str:
    return _utc(value, "observed_at").isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class UnsignedBootstrapProtectiveMark:
    """Canonical non-authorizing accounting mark with live-quote lineage."""

    portfolio_id: str
    symbol: str
    price: Decimal
    observed_at: datetime
    source_event_id: str
    con_id: int
    transport_generation: str
    protective_quote_id: str
    runtime_fingerprint: str
    execution_domain_scope: str
    account_scope: str
    database_identity: str
    database_device: int
    database_inode: int
    _producer_marker: object = field(repr=False, compare=False)
    source: str = BOOTSTRAP_MARK_SOURCE
    protective_quote_source: ProtectiveQuoteSource = ProtectiveQuoteSource.LIVE_BROKER
    schema_version: int = BOOTSTRAP_MARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise BootstrapMarkBlocked("bootstrap mark schema version is unsupported")
        if self._producer_marker is not _UNSIGNED_MARK_PRODUCER_MARKER:
            raise BootstrapMarkBlocked("unsigned mark lacks producer ownership")
        _strict_text(self.portfolio_id, "portfolio_id", _PORTFOLIO_ID)
        _strict_text(self.symbol, "symbol", _SYMBOL)
        _canonical_decimal(self.price)
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        _strict_text(self.source_event_id, "source_event_id", _SOURCE_EVENT_ID)
        _positive_int(self.con_id, "con_id")
        _strict_text(
            self.transport_generation,
            "transport_generation",
            _TRANSPORT_GENERATION,
        )
        _strict_text(self.protective_quote_id, "protective_quote_id", _QUOTE_ID)
        _strict_text(
            self.runtime_fingerprint,
            "runtime_fingerprint",
            _RUNTIME_FINGERPRINT,
        )
        if self.execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE:
            raise BootstrapMarkBlocked("bootstrap mark is outside the paper execution domain")
        account_scope = _strict_text(self.account_scope, "account_scope", _ACCOUNT_SCOPE)
        if len(set(account_scope.removeprefix("acct_v1_"))) == 1:
            raise BootstrapMarkBlocked("account_scope uses a placeholder digest")
        _strict_text(self.database_identity, "database_identity", _DATABASE_IDENTITY)
        _positive_int(self.database_device, "database_device")
        _positive_int(self.database_inode, "database_inode")
        if self.source != BOOTSTRAP_MARK_SOURCE:
            raise BootstrapMarkBlocked("bootstrap accounting mark source is invalid")
        if self.protective_quote_source is not ProtectiveQuoteSource.LIVE_BROKER:
            raise BootstrapMarkBlocked("bootstrap mark lacks live protective quote lineage")

    @property
    def mutated_state(self) -> bool:
        return False

    @property
    def authorizes_startup(self) -> bool:
        return False

    def canonical_dict(self) -> dict[str, object]:
        """Return the complete payload offered to the trusted core receiver."""

        return {
            "account_scope": self.account_scope,
            "authorizes_startup": False,
            "con_id": self.con_id,
            "database_device": self.database_device,
            "database_identity": self.database_identity,
            "database_inode": self.database_inode,
            "execution_domain_scope": self.execution_domain_scope,
            "mutated_state": False,
            "observed_at": _canonical_timestamp(self.observed_at),
            "portfolio_id": self.portfolio_id,
            "price_text": _canonical_decimal(self.price),
            "protective_quote_id": self.protective_quote_id,
            "protective_quote_source": self.protective_quote_source.value,
            "runtime_fingerprint": self.runtime_fingerprint,
            "schema_version": self.schema_version,
            "source": self.source,
            "source_event_id": self.source_event_id,
            "symbol": self.symbol,
            "transport_generation": self.transport_generation,
        }

    def canonical_payload(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class _UnsignedMarkRegistryEntry:
    result: UnsignedBootstrapProtectiveMark
    receiver: object
    digest: str
    registration_token: object


_UNSIGNED_MARK_REGISTRY: dict[int, _UnsignedMarkRegistryEntry] = {}
_CONSUMED_UNSIGNED_MARK_REGISTRATIONS: set[object] = set()
_UNSIGNED_MARK_REGISTRY_LOCK = threading.Lock()


def _unsigned_mark_digest(result: UnsignedBootstrapProtectiveMark) -> str:
    return hmac.new(
        _UNSIGNED_MARK_REGISTRY_KEY,
        result.canonical_payload().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _register_unsigned_mark(
    result: UnsignedBootstrapProtectiveMark,
    receiver: object,
) -> object:
    registration_token = object()
    entry = _UnsignedMarkRegistryEntry(
        result=result,
        receiver=receiver,
        digest=_unsigned_mark_digest(result),
        registration_token=registration_token,
    )
    with _UNSIGNED_MARK_REGISTRY_LOCK:
        if id(result) in _UNSIGNED_MARK_REGISTRY:  # pragma: no cover - new object invariant
            raise BootstrapMarkBlocked("unsigned mark registration identity collided")
        _UNSIGNED_MARK_REGISTRY[id(result)] = entry
    return registration_token


def _abandon_unsigned_mark_registration(
    result: UnsignedBootstrapProtectiveMark,
    registration_token: object,
) -> None:
    with _UNSIGNED_MARK_REGISTRY_LOCK:
        entry = _UNSIGNED_MARK_REGISTRY.get(id(result))
        if entry is not None and entry.registration_token is registration_token:
            _UNSIGNED_MARK_REGISTRY.pop(id(result), None)
        _CONSUMED_UNSIGNED_MARK_REGISTRATIONS.discard(registration_token)


def _assert_unsigned_mark_registration_consumed(
    result: UnsignedBootstrapProtectiveMark,
    registration_token: object,
) -> None:
    with _UNSIGNED_MARK_REGISTRY_LOCK:
        if registration_token in _CONSUMED_UNSIGNED_MARK_REGISTRATIONS:
            _CONSUMED_UNSIGNED_MARK_REGISTRATIONS.remove(registration_token)
            return
        entry = _UNSIGNED_MARK_REGISTRY.get(id(result))
        if entry is not None and entry.registration_token is registration_token:
            _UNSIGNED_MARK_REGISTRY.pop(id(result), None)
    raise BootstrapMarkBlocked(
        "bootstrap protective mark receiver did not authenticate its one-shot result"
    )


def assert_producer_owned_unsigned_bootstrap_protective_mark(
    result: UnsignedBootstrapProtectiveMark,
    *,
    receiver: object,
) -> UnsignedBootstrapProtectiveMark:
    """Consume one exact result registered to the asserting receiver.

    Trusted signing receivers must call this before reading or persisting the
    canonical payload.  Success is one-shot.  A copy, reconstruction, replay,
    changed result, or different receiver has no producer authority.
    """

    if type(result) is not UnsignedBootstrapProtectiveMark:
        raise BootstrapMarkBlocked("exact UnsignedBootstrapProtectiveMark is required")
    try:
        current_digest = _unsigned_mark_digest(result)
    except Exception as exc:
        raise BootstrapMarkBlocked("unsigned mark changed after production") from exc
    with _UNSIGNED_MARK_REGISTRY_LOCK:
        entry = _UNSIGNED_MARK_REGISTRY.get(id(result))
        if entry is None or entry.result is not result:
            raise BootstrapMarkBlocked("unsigned mark is not producer-owned or was replayed")
        if entry.receiver is not receiver:
            raise BootstrapMarkBlocked("unsigned mark belongs to a different receiver")
        if (
            result._producer_marker is not _UNSIGNED_MARK_PRODUCER_MARKER
            or not hmac.compare_digest(entry.digest, current_digest)
        ):
            _UNSIGNED_MARK_REGISTRY.pop(id(result), None)
            raise BootstrapMarkBlocked("unsigned mark changed after production")
        _UNSIGNED_MARK_REGISTRY.pop(id(result), None)
        _CONSUMED_UNSIGNED_MARK_REGISTRATIONS.add(entry.registration_token)
    return result


ReceiverResult = TypeVar("ReceiverResult", covariant=True)


class BootstrapProtectiveMarkReceiver(Protocol, Generic[ReceiverResult]):
    """The sole post-validation handoff available to a trusted core signer."""

    def receive_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> ReceiverResult:
        """Consume one validated unsigned protective accounting mark."""


class ProtectiveQuoteCollector(Protocol):
    """Read-only live-quote capability required by bootstrap mark collection."""

    @property
    def is_connected(self) -> bool:
        """Return exact current broker connectivity state."""

    async def get_protective_quotes(
        self,
        symbols: list[str] | tuple[str, ...],
        *,
        active_symbols: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[BrokerProtectiveQuote, ...]:
        """Collect exact typed live quotes from one current transport generation."""


@dataclass(frozen=True, slots=True)
class _DatabaseBinding:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True, slots=True)
class _MarkOnlyProducerBinding:
    runtime_fingerprint: str
    account_scope: str
    database_identity: str
    database_binding: _DatabaseBinding
    portfolio_id: str


_MARK_ONLY_PRODUCERS: weakref.WeakKeyDictionary[object, _MarkOnlyProducerBinding] = (
    weakref.WeakKeyDictionary()
)
_MARK_ONLY_PRODUCERS_LOCK = threading.Lock()


def _validate_runtime(runtime: object) -> tuple[RuntimeContract, _DatabaseBinding]:
    if type(runtime) is not RuntimeContract:
        raise BootstrapMarkBlocked("mark producer requires an exact RuntimeContract")
    if (
        runtime.execution_mode != "paper"
        or runtime.execution_source != PAPER_ONLY_EXECUTION_SOURCE
        or runtime.state_namespace != "paper"
        or runtime.account_type != "paper"
        or runtime.ibkr_port != 4002
        or runtime.ibkr_readonly is not True
        or runtime.safety_execution_domain_scope != PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE
    ):
        raise BootstrapMarkBlocked("runtime is not sealed paper/read-only topology")
    host = runtime.ibkr_host.casefold()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if host not in {"localhost", "localhost."} and not (
        address is not None and address.is_loopback
    ):
        raise BootstrapMarkBlocked("runtime broker host is not loopback")
    if not isinstance(runtime.safety_account_scope, str):
        raise BootstrapMarkBlocked("runtime account scope is unavailable")
    account_scope = _strict_text(
        runtime.safety_account_scope,
        "runtime account_scope",
        _ACCOUNT_SCOPE,
    )
    if len(set(account_scope.removeprefix("acct_v1_"))) == 1:
        raise BootstrapMarkBlocked("runtime account_scope uses a placeholder digest")
    path = runtime.database_path
    if type(path) is not str or not os.path.isabs(path) or os.path.normpath(path) != path:
        raise BootstrapMarkBlocked("runtime database path must be absolute and lexical")
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise BootstrapMarkBlocked("runtime database cannot be inspected") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise BootstrapMarkBlocked(
            "runtime database must be a single-link non-symlink regular file"
        )
    return runtime, _DatabaseBinding(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _assert_database_binding(runtime: RuntimeContract, expected: _DatabaseBinding) -> None:
    _, observed = _validate_runtime(runtime)
    if observed != expected:
        raise BootstrapMarkBlocked("runtime database changed during mark production")


async def _deny_mark_only_reduction(*_args: object, **_kwargs: object) -> None:
    raise BootstrapMarkBlocked("mark-only producer has no reduction capability")


def create_runtime_bound_mark_only_producer(
    runtime_contract: RuntimeContract,
    *,
    portfolio_id: str,
) -> StopLossMonitor:
    """Create an exact ``StopLossMonitor`` that can only accept quote evidence.

    ``StopLossMonitor`` requires a reduction callback at construction.  This
    factory supplies a permanent deny callback and registers the exact monitor
    against the runtime/database binding.  The collector rejects a changed
    callback, any stop/execution state, or runtime drift before broker access.
    """

    normalized_portfolio = _strict_text(portfolio_id, "portfolio_id", _PORTFOLIO_ID)
    runtime, database_binding = _validate_runtime(runtime_contract)
    if not isinstance(runtime.safety_account_scope, str):  # pragma: no cover - validated above
        raise BootstrapMarkBlocked("runtime account scope is unavailable")
    producer = StopLossMonitor(
        execute_reduction=_deny_mark_only_reduction,
        risk_manager=None,
        portfolio_id=normalized_portfolio,
    )
    binding = _MarkOnlyProducerBinding(
        runtime_fingerprint=runtime.fingerprint,
        account_scope=runtime.safety_account_scope,
        database_identity=runtime.database_identity,
        database_binding=database_binding,
        portfolio_id=normalized_portfolio,
    )
    with _MARK_ONLY_PRODUCERS_LOCK:
        _MARK_ONLY_PRODUCERS[producer] = binding
    return producer


def _assert_producer_context(
    producer: object,
    *,
    runtime: RuntimeContract,
    database_binding: _DatabaseBinding,
    portfolio_id: str,
) -> None:
    if type(producer) is not StopLossMonitor:
        raise BootstrapMarkBlocked("mark collector requires an exact StopLossMonitor producer")
    if producer.portfolio_id != portfolio_id:
        raise BootstrapMarkBlocked("protective quote producer portfolio does not match")
    with _MARK_ONLY_PRODUCERS_LOCK:
        mark_only_binding = _MARK_ONLY_PRODUCERS.get(producer)
    if mark_only_binding is None:
        return
    if not isinstance(runtime.safety_account_scope, str):  # pragma: no cover - validated above
        raise BootstrapMarkBlocked("runtime account scope is unavailable")
    expected = _MarkOnlyProducerBinding(
        runtime_fingerprint=runtime.fingerprint,
        account_scope=runtime.safety_account_scope,
        database_identity=runtime.database_identity,
        database_binding=database_binding,
        portfolio_id=portfolio_id,
    )
    if mark_only_binding != expected:
        raise BootstrapMarkBlocked("mark-only producer runtime binding changed")
    if producer._execute_reduction is not _deny_mark_only_reduction:
        raise BootstrapMarkBlocked("mark-only producer reduction capability changed")
    if (
        producer.active_stops
        or producer.monitoring_active
        or producer._pending_stop_triggers
        or producer._latched_stop_crossings
        or producer._queued_stop_orders
        or producer._inflight_stop_orders
    ):
        raise BootstrapMarkBlocked("mark-only producer contains trading execution state")


@dataclass(frozen=True, slots=True)
class _CollectorCapabilityIdentity:
    collector_type: type[object]
    method_function: object


def _quote_collector_capability(
    source: object,
    *,
    expected: _CollectorCapabilityIdentity | None = None,
):
    try:
        connected = getattr(source, "is_connected")
        capability = getattr(source, "get_protective_quotes")
    except Exception as exc:
        raise BootstrapMarkBlocked("protective quote collector capability is unavailable") from exc
    if type(connected) is not bool or connected is not True:
        raise BootstrapMarkBlocked("protective quote collector is not connected")
    method_owner = getattr(capability, "__self__", None)
    method_function = getattr(capability, "__func__", None)
    if method_owner is not source or method_function is None or not callable(capability):
        raise BootstrapMarkBlocked("protective quote collector capability is not a bound method")
    identity = _CollectorCapabilityIdentity(type(source), method_function)
    if expected is not None and identity != expected:
        raise BootstrapMarkBlocked("protective quote collector capability changed")
    return capability, identity


def _validated_broker_quote(
    value: object,
    *,
    symbol: str,
    con_id: int,
    transport_generation: str,
) -> tuple[BrokerProtectiveQuote, tuple[object, ...]]:
    if type(value) is not BrokerProtectiveQuote:
        raise BootstrapMarkBlocked("collector must return exact BrokerProtectiveQuote evidence")
    try:
        checked = BrokerProtectiveQuote(
            schema_version=value.schema_version,
            symbol=value.symbol,
            con_id=value.con_id,
            exchange=value.exchange,
            primary_exchange=value.primary_exchange,
            currency=value.currency,
            security_type=value.security_type,
            price=value.price,
            source_timestamp=value.source_timestamp,
            retrieval_timestamp=value.retrieval_timestamp,
            session=value.session,
            source=value.source,
            source_event_id=value.source_event_id,
            transport_generation=value.transport_generation,
            market_data_type=value.market_data_type,
        )
    except Exception as exc:
        raise BootstrapMarkBlocked(
            "collector returned malformed protective quote evidence"
        ) from exc
    if checked != value:
        raise BootstrapMarkBlocked("collector protective quote changed during validation")
    if checked.symbol != symbol:
        raise BootstrapMarkBlocked("collector protective quote symbol does not match")
    if checked.con_id != con_id:
        raise BootstrapMarkBlocked("collector protective quote contract does not match")
    if checked.transport_generation != transport_generation:
        raise BootstrapMarkBlocked("collector protective quote transport generation does not match")
    signature = (
        checked.schema_version,
        checked.symbol,
        checked.con_id,
        checked.exchange,
        checked.primary_exchange,
        checked.currency,
        checked.security_type,
        checked.price,
        checked.source_timestamp,
        checked.retrieval_timestamp,
        checked.session,
        checked.source,
        checked.source_event_id,
        checked.transport_generation,
        checked.market_data_type,
    )
    return checked, signature


def _assert_broker_quote_unchanged(
    value: BrokerProtectiveQuote,
    expected_signature: tuple[object, ...],
    *,
    symbol: str,
    con_id: int,
    transport_generation: str,
) -> None:
    _, current = _validated_broker_quote(
        value,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
    )
    if current != expected_signature:
        raise BootstrapMarkBlocked("collector protective quote changed after collection")


def _revalidate_quote(
    quote: ProtectiveQuoteEvidence,
    *,
    producer: object,
    portfolio_id: str,
    symbol: str,
    con_id: int,
    transport_generation: str,
    source_event_id: str,
) -> ProtectiveQuoteEvidence:
    try:
        return assert_current_authoritative_protective_quote(
            quote,
            producer=producer,
            expected_portfolio_id=portfolio_id,
            expected_symbol=symbol,
            expected_con_id=con_id,
            expected_transport_generation=transport_generation,
            expected_source_event_id=source_event_id,
        )
    except ProtectiveQuoteValidationError as exc:
        raise BootstrapMarkBlocked(f"protective quote is not authoritative: {exc}") from exc


def produce_bootstrap_protective_mark(
    quote: ProtectiveQuoteEvidence,
    producer: object,
    runtime_contract: RuntimeContract,
    receiver: BootstrapProtectiveMarkReceiver[ReceiverResult],
    *,
    expected_portfolio_id: str,
    expected_symbol: str,
    expected_con_id: int,
    expected_transport_generation: str,
    expected_source_event_id: str,
) -> ReceiverResult:
    """Deliver one runtime-bound unsigned mark after final quote revalidation."""

    portfolio_id = _strict_text(
        expected_portfolio_id,
        "expected_portfolio_id",
        _PORTFOLIO_ID,
    )
    symbol = _strict_text(expected_symbol, "expected_symbol", _SYMBOL)
    con_id = _positive_int(expected_con_id, "expected_con_id")
    transport_generation = _strict_text(
        expected_transport_generation,
        "expected_transport_generation",
        _TRANSPORT_GENERATION,
    )
    source_event_id = _strict_text(
        expected_source_event_id,
        "expected_source_event_id",
        _SOURCE_EVENT_ID,
    )
    runtime, database_binding = _validate_runtime(runtime_contract)
    checked_quote = _revalidate_quote(
        quote,
        producer=producer,
        portfolio_id=portfolio_id,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
    )
    account_scope = runtime.safety_account_scope
    if not isinstance(account_scope, str):  # pragma: no cover - validated above
        raise BootstrapMarkBlocked("runtime account scope is unavailable")
    result = UnsignedBootstrapProtectiveMark(
        portfolio_id=portfolio_id,
        symbol=symbol,
        price=checked_quote.price,
        observed_at=checked_quote.source_timestamp,
        source_event_id=source_event_id,
        con_id=con_id,
        transport_generation=transport_generation,
        protective_quote_id=checked_quote.quote_id,
        runtime_fingerprint=runtime.fingerprint,
        execution_domain_scope=PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE,
        account_scope=account_scope,
        database_identity=runtime.database_identity,
        database_device=database_binding.device,
        database_inode=database_binding.inode,
        _producer_marker=_UNSIGNED_MARK_PRODUCER_MARKER,
    )

    # Both mutable authorities can change after result construction.  Check the
    # exact database file and producer clock/ownership lineage immediately
    # before resolving or invoking the only external capability.
    _assert_database_binding(runtime, database_binding)
    _revalidate_quote(
        quote,
        producer=producer,
        portfolio_id=portfolio_id,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=source_event_id,
    )
    capability = getattr(receiver, "receive_unsigned_bootstrap_protective_mark", None)
    if not callable(capability):
        raise BootstrapMarkBlocked("bootstrap protective mark receiver capability is unavailable")
    registration_token = _register_unsigned_mark(result, receiver)
    try:
        received = capability(result)
    except BaseException:
        _abandon_unsigned_mark_registration(result, registration_token)
        raise
    _assert_unsigned_mark_registration_consumed(result, registration_token)
    return received


async def collect_and_produce_bootstrap_protective_mark(
    quote_source: ProtectiveQuoteCollector,
    producer: StopLossMonitor,
    runtime_contract: RuntimeContract,
    receiver: BootstrapProtectiveMarkReceiver[ReceiverResult],
    *,
    expected_portfolio_id: str,
    expected_symbol: str,
    expected_con_id: int,
    expected_transport_generation: str,
    expected_source_event_id: str | None = None,
) -> ReceiverResult:
    """Collect one live quote and deliver its producer-owned accounting mark.

    The source is called only through ``get_protective_quotes`` and receives no
    artifact, signer, output path, monitor, or receiver capability.  The exact
    typed quote is published through the portfolio-owned monitor, which is the
    only supported way to create ``ProtectiveQuoteEvidence``.
    """

    portfolio_id = _strict_text(
        expected_portfolio_id,
        "expected_portfolio_id",
        _PORTFOLIO_ID,
    )
    symbol = _strict_text(expected_symbol, "expected_symbol", _SYMBOL)
    con_id = _positive_int(expected_con_id, "expected_con_id")
    transport_generation = _strict_text(
        expected_transport_generation,
        "expected_transport_generation",
        _TRANSPORT_GENERATION,
    )
    source_event_id = (
        None
        if expected_source_event_id is None
        else _strict_text(
            expected_source_event_id,
            "expected_source_event_id",
            _SOURCE_EVENT_ID,
        )
    )
    runtime, database_binding = _validate_runtime(runtime_contract)
    _assert_producer_context(
        producer,
        runtime=runtime,
        database_binding=database_binding,
        portfolio_id=portfolio_id,
    )
    typed_producer = producer
    capability, capability_identity = _quote_collector_capability(quote_source)
    try:
        collected = await capability((symbol,), active_symbols=(symbol,))
    except BootstrapMarkBlocked:
        raise
    except Exception as exc:
        raise BootstrapMarkBlocked("protective quote collection failed closed") from exc
    _quote_collector_capability(quote_source, expected=capability_identity)
    if type(collected) is not tuple or len(collected) != 1:
        raise BootstrapMarkBlocked("protective quote collection is not exact and complete")
    source_quote = collected[0]
    checked_quote, quote_signature = _validated_broker_quote(
        source_quote,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
    )
    if source_event_id is not None and checked_quote.source_event_id != source_event_id:
        raise BootstrapMarkBlocked("collector protective quote source event does not match")
    _assert_database_binding(runtime, database_binding)
    _assert_producer_context(
        producer,
        runtime=runtime,
        database_binding=database_binding,
        portfolio_id=portfolio_id,
    )
    accepted = await typed_producer.update_price(
        checked_quote.symbol,
        checked_quote.price,
        source_timestamp=checked_quote.source_timestamp,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=checked_quote.con_id,
        transport_generation=checked_quote.transport_generation,
        source_event_id=checked_quote.source_event_id,
    )
    if accepted is not True:
        raise BootstrapMarkBlocked("portfolio producer rejected the collected protective quote")

    _quote_collector_capability(quote_source, expected=capability_identity)
    _assert_broker_quote_unchanged(
        source_quote,
        quote_signature,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
    )
    _assert_database_binding(runtime, database_binding)
    _assert_producer_context(
        producer,
        runtime=runtime,
        database_binding=database_binding,
        portfolio_id=portfolio_id,
    )
    evidence = typed_producer.get_protective_quote_evidence(symbol)
    if evidence is None:
        raise BootstrapMarkBlocked("portfolio producer did not retain protective quote evidence")
    checked_evidence = _revalidate_quote(
        evidence,
        producer=typed_producer,
        portfolio_id=portfolio_id,
        symbol=symbol,
        con_id=con_id,
        transport_generation=transport_generation,
        source_event_id=checked_quote.source_event_id,
    )
    return produce_bootstrap_protective_mark(
        checked_evidence,
        typed_producer,
        runtime,
        receiver,
        expected_portfolio_id=portfolio_id,
        expected_symbol=symbol,
        expected_con_id=con_id,
        expected_transport_generation=transport_generation,
        expected_source_event_id=checked_quote.source_event_id,
    )

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

import ipaddress
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Generic, Protocol, TypeVar

from .config import PAPER_ONLY_EXECUTION_SOURCE, RuntimeContract
from .protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    ProtectiveQuoteValidationError,
    assert_current_authoritative_protective_quote,
)
from .runtime_contract_constants import PAPER_SAFETY_EXECUTION_DOMAIN_SCOPE

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
    source: str = BOOTSTRAP_MARK_SOURCE
    protective_quote_source: ProtectiveQuoteSource = ProtectiveQuoteSource.LIVE_BROKER
    schema_version: int = BOOTSTRAP_MARK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise BootstrapMarkBlocked("bootstrap mark schema version is unsupported")
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


ReceiverResult = TypeVar("ReceiverResult", covariant=True)


class BootstrapProtectiveMarkReceiver(Protocol, Generic[ReceiverResult]):
    """The sole post-validation handoff available to a trusted core signer."""

    def receive_unsigned_bootstrap_protective_mark(
        self,
        result: UnsignedBootstrapProtectiveMark,
    ) -> ReceiverResult:
        """Consume one validated unsigned protective accounting mark."""


@dataclass(frozen=True, slots=True)
class _DatabaseBinding:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


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
    return capability(result)

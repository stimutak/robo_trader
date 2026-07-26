"""Producer-owned immutable broker evidence for paper safety authorization.

The subprocess broker client is the only production producer.  Public model
constructors validate shape and exact numeric values, while the snapshot
registry additionally proves that a specific snapshot object crossed the
producer factory rather than being assembled by an order caller.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional, Tuple

from robo_trader.reconciliation.identity import (
    RuntimeSafetyContext,
    assert_validated_runtime_safety_context,
    validate_ibc_safety_config,
)

from .broker_account_identity import is_supported_paper_account_identifier
from .safety.models import ValidationError, canonical_json

_ACCOUNT_SCOPE_RE = re.compile(r"^acct_v1_[0-9a-f]{64}$")
_SYMBOL_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}$")
_ORDER_STATUSES = frozenset(
    {
        "ApiPending",
        "PendingSubmit",
        "PendingCancel",
        "PreSubmitted",
        "Submitted",
        "ApiCancelled",
        "Cancelled",
        "Filled",
        "Inactive",
    }
)
_ORDER_TYPES = frozenset(
    {
        "MKT",
        "LMT",
        "STP",
        "STP LMT",
        "TRAIL",
        "TRAIL LIMIT",
        "MOC",
        "LOC",
        "MIT",
        "LIT",
        "REL",
        "PEG MID",
        "MIDPRICE",
        "MTL",
    }
)
_TIME_IN_FORCE = frozenset({"DAY", "GTC", "IOC", "GTD", "OPG", "FOK", "DTC"})
_PRODUCER_MARKER = object()
_IBC_PROOF_KEY = secrets.token_bytes(32)
_CAPABILITY_MARKER = object()
_CONTRACT_PRODUCER_MARKER = object()
_CONTRACT_CAPABILITY_MARKER = object()


def _utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValidationError(f"{field_name} must be timezone-aware")
    normalized = value.astimezone(timezone.utc)
    if normalized.utcoffset() != timedelta(0):  # pragma: no cover - astimezone invariant
        raise ValidationError(f"{field_name} must normalize to UTC")
    return normalized


def _text(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 128
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValidationError(f"{field_name} is malformed")
    return value


def _symbol(value: object, field_name: str = "symbol") -> str:
    if not isinstance(value, str) or not _SYMBOL_RE.fullmatch(value):
        raise ValidationError(f"{field_name} is malformed")
    return value


def _decimal(value: object, field_name: str, *, positive: bool = False) -> Decimal:
    if type(value) is not Decimal or not value.is_finite():
        raise ValidationError(f"{field_name} must be an exact finite Decimal")
    if positive and value <= 0:
        raise ValidationError(f"{field_name} must be positive")
    return value


@dataclass(frozen=True, slots=True)
class BrokerSafetyContract:
    """One exact, broker-qualified SMART/USD stock identity."""

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
            raise ValidationError("con_id must be a positive integer")
        _symbol(self.symbol)
        _symbol(self.local_symbol, "local_symbol")
        if self.local_symbol != self.symbol:
            raise ValidationError("local_symbol must exactly match symbol")
        if self.security_type != "STK":
            raise ValidationError("security_type must be STK")
        if self.currency != "USD":
            raise ValidationError("currency must be USD")
        if self.exchange != "SMART":
            raise ValidationError("exchange must be SMART")
        _text(self.primary_exchange, "primary_exchange")
        _text(self.trading_class, "trading_class")


@dataclass(frozen=True, slots=True)
class BrokerSafetyPosition:
    """One exact non-zero account position."""

    contract: BrokerSafetyContract
    quantity: Decimal

    def __post_init__(self) -> None:
        if type(self.contract) is not BrokerSafetyContract:
            raise ValidationError("position contract must be BrokerSafetyContract")
        _decimal(self.quantity, "position quantity")
        if self.quantity == 0:
            raise ValidationError("position quantity must be non-zero")


@dataclass(frozen=True, slots=True)
class BrokerSafetyOpenOrder:
    """Exact active order evidence returned by IBKR for any client."""

    broker_order_id: int
    permanent_id: Optional[int]
    client_id: int
    contract: BrokerSafetyContract
    side: str
    status: str
    order_type: str
    time_in_force: str
    total_quantity: Decimal
    filled_quantity: Decimal
    remaining_quantity: Decimal
    limit_price: Optional[Decimal]
    stop_price: Optional[Decimal]
    average_fill_price: Optional[Decimal]
    last_status_at: Optional[datetime]

    def __post_init__(self) -> None:
        if type(self.broker_order_id) is not int or self.broker_order_id <= 0:
            raise ValidationError("broker_order_id must be a positive integer")
        if self.permanent_id is not None and (
            type(self.permanent_id) is not int or self.permanent_id <= 0
        ):
            raise ValidationError("permanent_id must be a positive integer or None")
        if type(self.client_id) is not int or self.client_id < 0:
            raise ValidationError("client_id must be a nonnegative integer")
        if type(self.contract) is not BrokerSafetyContract:
            raise ValidationError("order contract must be BrokerSafetyContract")
        if self.side not in {"BUY", "SELL"}:
            raise ValidationError("broker order side is unsupported")
        for field_name in ("status", "order_type", "time_in_force"):
            _text(getattr(self, field_name), field_name)
        if self.status not in _ORDER_STATUSES:
            raise ValidationError("broker order status is unsupported")
        if self.order_type not in _ORDER_TYPES:
            raise ValidationError("broker order type is unsupported")
        if self.time_in_force not in _TIME_IN_FORCE:
            raise ValidationError("broker order time in force is unsupported")
        total = _decimal(self.total_quantity, "total_quantity", positive=True)
        filled = _decimal(self.filled_quantity, "filled_quantity")
        remaining = _decimal(self.remaining_quantity, "remaining_quantity")
        if filled < 0 or remaining < 0 or filled + remaining != total:
            raise ValidationError("broker order quantities are inconsistent")
        for field_name in ("limit_price", "stop_price", "average_fill_price"):
            value = getattr(self, field_name)
            if value is not None:
                _decimal(value, field_name, positive=True)
        if self.last_status_at is not None:
            object.__setattr__(
                self,
                "last_status_at",
                _utc(self.last_status_at, "last_status_at"),
            )


@dataclass(frozen=True)
class _BrokerSnapshotCapability:
    account_scope: str
    requested_symbol: str
    runtime_fingerprint: str
    broker_host: str
    broker_port: int
    broker_client_id: int
    read_only: bool
    transport_generation: str
    _expected_account: str = field(repr=False, compare=False)
    _ibc_config_hash: str = field(repr=False, compare=False)
    _producer_marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class BrokerSafetySnapshot:
    """One immutable, current-generation paper/read-only account snapshot."""

    account_scope: str
    requested_symbol: str
    observed_at: datetime
    broker_time_before: datetime
    broker_time_after: datetime
    snapshot_id: str
    source: str
    transport_generation: str
    runtime_fingerprint: str
    broker_host: str
    broker_port: int
    broker_client_id: int
    read_only: bool
    ibc_proof_id: str
    ibc_proof_source: str
    requested_contract: BrokerSafetyContract
    positions: Tuple[BrokerSafetyPosition, ...]
    open_orders: Tuple[BrokerSafetyOpenOrder, ...]
    _producer_marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _PRODUCER_MARKER:
            raise ValidationError("BrokerSafetySnapshot requires the broker producer boundary")
        if not isinstance(self.account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(
            self.account_scope
        ):
            raise ValidationError("account_scope must be an opaque account scope")
        if len(set(self.account_scope.removeprefix("acct_v1_"))) == 1:
            raise ValidationError("account_scope must not use a placeholder digest")
        _symbol(self.requested_symbol, "requested_symbol")
        object.__setattr__(self, "observed_at", _utc(self.observed_at, "observed_at"))
        object.__setattr__(
            self,
            "broker_time_before",
            _utc(self.broker_time_before, "broker_time_before"),
        )
        object.__setattr__(
            self,
            "broker_time_after",
            _utc(self.broker_time_after, "broker_time_after"),
        )
        if self.broker_time_after < self.broker_time_before:
            raise ValidationError("broker snapshot times are reversed")
        _text(self.snapshot_id, "snapshot_id")
        _text(self.source, "source")
        _text(self.transport_generation, "transport_generation")
        _text(self.runtime_fingerprint, "runtime_fingerprint")
        _text(self.broker_host, "broker_host")
        normalized_host = self.broker_host.casefold()
        try:
            address = ipaddress.ip_address(normalized_host)
        except ValueError:
            address = None
        if normalized_host not in {"localhost", "localhost."} and not (
            address is not None and address.is_loopback
        ):
            raise ValidationError("broker snapshot host must be loopback")
        if self.broker_port != 4002:
            raise ValidationError("broker snapshot must use the paper port")
        if type(self.broker_client_id) is not int or self.broker_client_id <= 0:
            raise ValidationError("broker_client_id must be a positive integer")
        if self.read_only is not True:
            raise ValidationError("broker snapshot must be read-only")
        if not re.fullmatch(r"ibc-proof-v1-[0-9a-f]{64}", self.ibc_proof_id):
            raise ValidationError("ibc_proof_id must be an opaque proof identifier")
        if self.ibc_proof_source != "validated-ibc-readonly-paper-v1":
            raise ValidationError("IBC proof provenance is unsupported")
        if type(self.requested_contract) is not BrokerSafetyContract:
            raise ValidationError("requested_contract must be BrokerSafetyContract")
        if self.requested_contract.symbol != self.requested_symbol:
            raise ValidationError("requested contract does not match requested symbol")
        if not isinstance(self.positions, tuple) or any(
            type(item) is not BrokerSafetyPosition for item in self.positions
        ):
            raise ValidationError("positions must be an immutable BrokerSafetyPosition tuple")
        if not isinstance(self.open_orders, tuple) or any(
            type(item) is not BrokerSafetyOpenOrder for item in self.open_orders
        ):
            raise ValidationError("open_orders must be an immutable BrokerSafetyOpenOrder tuple")
        matching = [item for item in self.positions if item.contract == self.requested_contract]
        if len(matching) != 1:
            raise ValidationError("requested held position must appear exactly once")

    @property
    def positions_complete(self) -> bool:
        return True

    @property
    def open_orders_complete(self) -> bool:
        return True

    @property
    def open_orders_all_clients(self) -> bool:
        return True

    @property
    def open_orders_stable(self) -> bool:
        return True

    @property
    def unknown_order_count(self) -> int:
        return 0

    @property
    def active_con_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({item.contract.con_id for item in self.open_orders}))


@dataclass(frozen=True)
class _BrokerContractSnapshotCapability:
    account_scope: str
    requested_symbol: str
    runtime_fingerprint: str
    broker_host: str
    broker_port: int
    broker_client_id: int
    read_only: bool
    transport_generation: str
    _expected_account: str = field(repr=False, compare=False)
    _ibc_config_hash: str = field(repr=False, compare=False)
    _producer_marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class BrokerContractSafetySnapshot:
    """Qualified stock and current read-only transport proof without account state."""

    account_scope: str
    requested_symbol: str
    broker_time_before: datetime
    broker_time_after: datetime
    retrieved_at: datetime
    snapshot_id: str
    source: str
    transport_generation: str
    runtime_fingerprint: str
    broker_host: str
    broker_port: int
    broker_client_id: int
    read_only: bool
    ibc_proof_id: str
    ibc_proof_source: str
    qualified_contract: BrokerSafetyContract
    _producer_marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._producer_marker is not _CONTRACT_PRODUCER_MARKER:
            raise ValidationError(
                "BrokerContractSafetySnapshot requires the broker producer boundary"
            )
        if not isinstance(self.account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(
            self.account_scope
        ):
            raise ValidationError("account_scope must be an opaque account scope")
        if len(set(self.account_scope.removeprefix("acct_v1_"))) == 1:
            raise ValidationError("account_scope must not use a placeholder digest")
        _symbol(self.requested_symbol, "requested_symbol")
        object.__setattr__(
            self,
            "broker_time_before",
            _utc(self.broker_time_before, "broker_time_before"),
        )
        object.__setattr__(
            self,
            "broker_time_after",
            _utc(self.broker_time_after, "broker_time_after"),
        )
        object.__setattr__(self, "retrieved_at", _utc(self.retrieved_at, "retrieved_at"))
        if self.broker_time_after < self.broker_time_before:
            raise ValidationError("broker contract snapshot times are reversed")
        _text(self.snapshot_id, "snapshot_id")
        _text(self.source, "source")
        _text(self.transport_generation, "transport_generation")
        _text(self.runtime_fingerprint, "runtime_fingerprint")
        _text(self.broker_host, "broker_host")
        normalized_host = self.broker_host.casefold()
        try:
            address = ipaddress.ip_address(normalized_host)
        except ValueError:
            address = None
        if normalized_host not in {"localhost", "localhost."} and not (
            address is not None and address.is_loopback
        ):
            raise ValidationError("broker contract snapshot host must be loopback")
        if self.broker_port != 4002:
            raise ValidationError("broker contract snapshot must use the paper port")
        if type(self.broker_client_id) is not int or self.broker_client_id <= 0:
            raise ValidationError("broker_client_id must be a positive integer")
        if self.read_only is not True:
            raise ValidationError("broker contract snapshot must be read-only")
        if not re.fullmatch(r"ibc-proof-v1-[0-9a-f]{64}", self.ibc_proof_id):
            raise ValidationError("ibc_proof_id must be an opaque proof identifier")
        if self.ibc_proof_source != "validated-ibc-readonly-paper-v1":
            raise ValidationError("IBC proof provenance is unsupported")
        if type(self.qualified_contract) is not BrokerSafetyContract:
            raise ValidationError("qualified_contract must be BrokerSafetyContract")
        if self.qualified_contract.symbol != self.requested_symbol:
            raise ValidationError("qualified contract does not match requested symbol")


_REGISTRY_LOCK = threading.RLock()
_REGISTRY: dict[int, tuple[weakref.ReferenceType[BrokerSafetySnapshot], str]] = {}
_CAPABILITY_REGISTRY_LOCK = threading.RLock()
_CAPABILITY_REGISTRY: dict[int, tuple[weakref.ReferenceType[_BrokerSnapshotCapability], str]] = {}
_CONTRACT_CAPABILITY_REGISTRY_LOCK = threading.RLock()
_CONTRACT_CAPABILITY_REGISTRY: dict[
    int,
    tuple[weakref.ReferenceType[_BrokerContractSnapshotCapability], str],
] = {}
_CONTRACT_SNAPSHOT_REGISTRY_LOCK = threading.RLock()
_CONTRACT_SNAPSHOT_REGISTRY: dict[
    int,
    tuple[weakref.ReferenceType[BrokerContractSafetySnapshot], str],
] = {}


def _capability_payload(capability: _BrokerSnapshotCapability) -> str:
    return canonical_json(
        {
            "account_scope": capability.account_scope,
            "broker_client_id": capability.broker_client_id,
            "broker_host": capability.broker_host,
            "broker_port": capability.broker_port,
            "expected_account": capability._expected_account,
            "ibc_config_hash": capability._ibc_config_hash,
            "read_only": capability.read_only,
            "requested_symbol": capability.requested_symbol,
            "runtime_fingerprint": capability.runtime_fingerprint,
            "transport_generation": capability.transport_generation,
        }
    )


def _capability_digest(capability: _BrokerSnapshotCapability) -> str:
    return hmac.new(
        _IBC_PROOF_KEY,
        _capability_payload(capability).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _discard_capability(
    object_id: int,
    reference: weakref.ReferenceType[_BrokerSnapshotCapability],
) -> None:
    with _CAPABILITY_REGISTRY_LOCK:
        registered = _CAPABILITY_REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _CAPABILITY_REGISTRY.pop(object_id, None)


def _issue_broker_snapshot_capability(
    runtime_context: RuntimeSafetyContext,
    *,
    connection_identity: tuple[str, int, int, bool],
    transport_generation: str,
    requested_symbol: str,
) -> _BrokerSnapshotCapability:
    """Issue one one-shot snapshot capability from exact validated runtime state."""

    context = assert_validated_runtime_safety_context(runtime_context)
    connection = context.diagnostic_connection
    expected_identity = (
        connection.host,
        connection.port,
        connection.client_id,
        connection.readonly,
    )
    if connection_identity != expected_identity:
        raise ValidationError("broker connection differs from validated runtime identity")
    current_ibc_hash = validate_ibc_safety_config(
        context.project_root / "config" / "ibc" / "config.ini"
    )
    if not hmac.compare_digest(current_ibc_hash, context.ibc_config_hash):
        raise ValidationError("validated IBC configuration changed before capability issuance")
    expected_account = context.expected_account_for_provider
    account_scope = getattr(context.runtime_contract, "safety_account_scope", None)
    runtime_fingerprint = context.runtime_contract.fingerprint
    if not is_supported_paper_account_identifier(
        expected_account,
        environment=context.runtime_contract.environment,
    ):
        raise ValidationError("validated runtime lacks a paper account")
    if not isinstance(account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(account_scope):
        raise ValidationError("validated runtime lacks an opaque account scope")
    _symbol(requested_symbol, "requested_symbol")
    capability = _BrokerSnapshotCapability(
        account_scope=account_scope,
        requested_symbol=requested_symbol,
        runtime_fingerprint=runtime_fingerprint,
        broker_host=connection.host,
        broker_port=connection.port,
        broker_client_id=connection.client_id,
        read_only=connection.readonly,
        transport_generation=transport_generation,
        _expected_account=expected_account,
        _ibc_config_hash=current_ibc_hash,
        _producer_marker=_CAPABILITY_MARKER,
    )
    object_id = id(capability)

    def discard(reference: weakref.ReferenceType[_BrokerSnapshotCapability]) -> None:
        _discard_capability(object_id, reference)

    reference = weakref.ref(capability, discard)
    digest = _capability_digest(capability)
    with _CAPABILITY_REGISTRY_LOCK:
        _CAPABILITY_REGISTRY[object_id] = (reference, digest)
    return capability


def _consume_broker_snapshot_capability(
    capability: _BrokerSnapshotCapability,
) -> _BrokerSnapshotCapability:
    if type(capability) is not _BrokerSnapshotCapability:
        raise ValidationError("exact broker snapshot capability is required")
    with _CAPABILITY_REGISTRY_LOCK:
        registered = _CAPABILITY_REGISTRY.pop(id(capability), None)
        if registered is None or registered[0]() is not capability:
            raise ValidationError("broker snapshot capability is absent or already consumed")
        if capability._producer_marker is not _CAPABILITY_MARKER or not hmac.compare_digest(
            registered[1], _capability_digest(capability)
        ):
            raise ValidationError("broker snapshot capability changed after issuance")
    return capability


def _broker_contract_capability_payload(
    capability: _BrokerContractSnapshotCapability,
) -> str:
    return canonical_json(
        {
            "account_scope": capability.account_scope,
            "broker_client_id": capability.broker_client_id,
            "broker_host": capability.broker_host,
            "broker_port": capability.broker_port,
            "expected_account": capability._expected_account,
            "ibc_config_hash": capability._ibc_config_hash,
            "read_only": capability.read_only,
            "requested_symbol": capability.requested_symbol,
            "runtime_fingerprint": capability.runtime_fingerprint,
            "transport_generation": capability.transport_generation,
        }
    )


def _broker_contract_capability_digest(
    capability: _BrokerContractSnapshotCapability,
) -> str:
    return hmac.new(
        _IBC_PROOF_KEY,
        _broker_contract_capability_payload(capability).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _discard_broker_contract_capability(
    object_id: int,
    reference: weakref.ReferenceType[_BrokerContractSnapshotCapability],
) -> None:
    with _CONTRACT_CAPABILITY_REGISTRY_LOCK:
        registered = _CONTRACT_CAPABILITY_REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _CONTRACT_CAPABILITY_REGISTRY.pop(object_id, None)


def _issue_broker_contract_snapshot_capability(
    runtime_context: RuntimeSafetyContext,
    *,
    connection_identity: tuple[str, int, int, bool],
    transport_generation: str,
    requested_symbol: str,
) -> _BrokerContractSnapshotCapability:
    """Issue one contract-only capability from exact current runtime state."""

    context = assert_validated_runtime_safety_context(runtime_context)
    connection = context.diagnostic_connection
    expected_identity = (
        connection.host,
        connection.port,
        connection.client_id,
        connection.readonly,
    )
    if connection_identity != expected_identity:
        raise ValidationError("broker connection differs from validated runtime identity")
    current_ibc_hash = validate_ibc_safety_config(
        context.project_root / "config" / "ibc" / "config.ini"
    )
    if not hmac.compare_digest(current_ibc_hash, context.ibc_config_hash):
        raise ValidationError("validated IBC configuration changed before capability issuance")
    expected_account = context.expected_account_for_provider
    account_scope = getattr(context.runtime_contract, "safety_account_scope", None)
    runtime_fingerprint = context.runtime_contract.fingerprint
    if not is_supported_paper_account_identifier(
        expected_account,
        environment=context.runtime_contract.environment,
    ):
        raise ValidationError("validated runtime lacks a paper account")
    if not isinstance(account_scope, str) or not _ACCOUNT_SCOPE_RE.fullmatch(account_scope):
        raise ValidationError("validated runtime lacks an opaque account scope")
    _symbol(requested_symbol, "requested_symbol")
    capability = _BrokerContractSnapshotCapability(
        account_scope=account_scope,
        requested_symbol=requested_symbol,
        runtime_fingerprint=runtime_fingerprint,
        broker_host=connection.host,
        broker_port=connection.port,
        broker_client_id=connection.client_id,
        read_only=connection.readonly,
        transport_generation=transport_generation,
        _expected_account=expected_account,
        _ibc_config_hash=current_ibc_hash,
        _producer_marker=_CONTRACT_CAPABILITY_MARKER,
    )
    object_id = id(capability)

    def discard(reference: weakref.ReferenceType[_BrokerContractSnapshotCapability]) -> None:
        _discard_broker_contract_capability(object_id, reference)

    reference = weakref.ref(capability, discard)
    digest = _broker_contract_capability_digest(capability)
    with _CONTRACT_CAPABILITY_REGISTRY_LOCK:
        _CONTRACT_CAPABILITY_REGISTRY[object_id] = (reference, digest)
    return capability


def _consume_broker_contract_snapshot_capability(
    capability: _BrokerContractSnapshotCapability,
) -> _BrokerContractSnapshotCapability:
    if type(capability) is not _BrokerContractSnapshotCapability:
        raise ValidationError("exact broker contract snapshot capability is required")
    with _CONTRACT_CAPABILITY_REGISTRY_LOCK:
        registered = _CONTRACT_CAPABILITY_REGISTRY.pop(id(capability), None)
        if registered is None or registered[0]() is not capability:
            raise ValidationError(
                "broker contract snapshot capability is absent or already consumed"
            )
        if (
            capability._producer_marker is not _CONTRACT_CAPABILITY_MARKER
            or not hmac.compare_digest(
                registered[1],
                _broker_contract_capability_digest(capability),
            )
        ):
            raise ValidationError("broker contract snapshot capability changed after issuance")
    return capability


def _contract_payload(contract: BrokerSafetyContract) -> dict[str, object]:
    return {
        "con_id": contract.con_id,
        "currency": contract.currency,
        "exchange": contract.exchange,
        "local_symbol": contract.local_symbol,
        "primary_exchange": contract.primary_exchange,
        "security_type": contract.security_type,
        "symbol": contract.symbol,
        "trading_class": contract.trading_class,
    }


def _snapshot_payload(snapshot: BrokerSafetySnapshot) -> str:
    return canonical_json(
        {
            "account_scope": snapshot.account_scope,
            "broker_time_after": snapshot.broker_time_after,
            "broker_time_before": snapshot.broker_time_before,
            "broker_client_id": snapshot.broker_client_id,
            "broker_host": snapshot.broker_host,
            "broker_port": snapshot.broker_port,
            "ibc_proof_id": snapshot.ibc_proof_id,
            "ibc_proof_source": snapshot.ibc_proof_source,
            "observed_at": snapshot.observed_at,
            "open_orders": tuple(
                {
                    "average_fill_price": item.average_fill_price,
                    "broker_order_id": item.broker_order_id,
                    "client_id": item.client_id,
                    "contract": _contract_payload(item.contract),
                    "filled_quantity": item.filled_quantity,
                    "last_status_at": item.last_status_at,
                    "limit_price": item.limit_price,
                    "order_type": item.order_type,
                    "permanent_id": item.permanent_id,
                    "remaining_quantity": item.remaining_quantity,
                    "side": item.side,
                    "status": item.status,
                    "stop_price": item.stop_price,
                    "time_in_force": item.time_in_force,
                    "total_quantity": item.total_quantity,
                }
                for item in snapshot.open_orders
            ),
            "positions": tuple(
                {
                    "contract": _contract_payload(item.contract),
                    "quantity": item.quantity,
                }
                for item in snapshot.positions
            ),
            "requested_contract": _contract_payload(snapshot.requested_contract),
            "requested_symbol": snapshot.requested_symbol,
            "read_only": snapshot.read_only,
            "runtime_fingerprint": snapshot.runtime_fingerprint,
            "snapshot_id": snapshot.snapshot_id,
            "source": snapshot.source,
            "transport_generation": snapshot.transport_generation,
        }
    )


def _discard_snapshot(
    object_id: int,
    reference: weakref.ReferenceType[BrokerSafetySnapshot],
) -> None:
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _REGISTRY.pop(object_id, None)


def _produce_broker_safety_snapshot(
    *,
    capability: _BrokerSnapshotCapability,
    observed_at: datetime,
    broker_time_before: datetime,
    broker_time_after: datetime,
    snapshot_id: str,
    source: str,
    requested_contract: BrokerSafetyContract,
    positions: Tuple[BrokerSafetyPosition, ...],
    open_orders: Tuple[BrokerSafetyOpenOrder, ...],
) -> BrokerSafetySnapshot:
    """Internal factory used by the validated subprocess broker client only."""

    capability = _consume_broker_snapshot_capability(capability)
    proof_payload = _capability_payload(capability)
    ibc_proof_id = (
        "ibc-proof-v1-"
        + hmac.new(
            _IBC_PROOF_KEY,
            proof_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    snapshot = BrokerSafetySnapshot(
        account_scope=capability.account_scope,
        requested_symbol=capability.requested_symbol,
        observed_at=observed_at,
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
        snapshot_id=snapshot_id,
        source=source,
        transport_generation=capability.transport_generation,
        runtime_fingerprint=capability.runtime_fingerprint,
        broker_host=capability.broker_host,
        broker_port=capability.broker_port,
        broker_client_id=capability.broker_client_id,
        read_only=capability.read_only,
        ibc_proof_id=ibc_proof_id,
        ibc_proof_source="validated-ibc-readonly-paper-v1",
        requested_contract=requested_contract,
        positions=positions,
        open_orders=open_orders,
        _producer_marker=_PRODUCER_MARKER,
    )
    object_id = id(snapshot)

    def discard(current: weakref.ReferenceType[BrokerSafetySnapshot]) -> None:
        _discard_snapshot(object_id, current)

    reference = weakref.ref(snapshot, discard)
    digest = hashlib.sha256(_snapshot_payload(snapshot).encode("utf-8")).hexdigest()
    with _REGISTRY_LOCK:
        _REGISTRY[object_id] = (reference, digest)
    return snapshot


def assert_producer_owned_broker_safety_snapshot(snapshot: BrokerSafetySnapshot) -> None:
    """Reject copied, deserialized, forged, or mutated snapshot objects."""

    if type(snapshot) is not BrokerSafetySnapshot:
        raise ValidationError("snapshot must be an exact BrokerSafetySnapshot")
    with _REGISTRY_LOCK:
        registered = _REGISTRY.get(id(snapshot))
        if registered is None or registered[0]() is not snapshot:
            raise ValidationError("BrokerSafetySnapshot is not producer-owned")
        digest = hashlib.sha256(_snapshot_payload(snapshot).encode("utf-8")).hexdigest()
        if not hmac.compare_digest(registered[1], digest):
            raise ValidationError("BrokerSafetySnapshot changed after production")


def _broker_contract_snapshot_payload(snapshot: BrokerContractSafetySnapshot) -> str:
    return canonical_json(
        {
            "account_scope": snapshot.account_scope,
            "broker_client_id": snapshot.broker_client_id,
            "broker_host": snapshot.broker_host,
            "broker_port": snapshot.broker_port,
            "broker_time_after": snapshot.broker_time_after,
            "broker_time_before": snapshot.broker_time_before,
            "ibc_proof_id": snapshot.ibc_proof_id,
            "ibc_proof_source": snapshot.ibc_proof_source,
            "qualified_contract": _contract_payload(snapshot.qualified_contract),
            "read_only": snapshot.read_only,
            "requested_symbol": snapshot.requested_symbol,
            "retrieved_at": snapshot.retrieved_at,
            "runtime_fingerprint": snapshot.runtime_fingerprint,
            "snapshot_id": snapshot.snapshot_id,
            "source": snapshot.source,
            "transport_generation": snapshot.transport_generation,
        }
    )


def _broker_contract_snapshot_digest(snapshot: BrokerContractSafetySnapshot) -> str:
    return hmac.new(
        _IBC_PROOF_KEY,
        _broker_contract_snapshot_payload(snapshot).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _discard_broker_contract_snapshot(
    object_id: int,
    reference: weakref.ReferenceType[BrokerContractSafetySnapshot],
) -> None:
    with _CONTRACT_SNAPSHOT_REGISTRY_LOCK:
        registered = _CONTRACT_SNAPSHOT_REGISTRY.get(object_id)
        if registered is not None and registered[0] is reference:
            _CONTRACT_SNAPSHOT_REGISTRY.pop(object_id, None)


def _produce_broker_contract_safety_snapshot(
    *,
    capability: _BrokerContractSnapshotCapability,
    broker_time_before: datetime,
    broker_time_after: datetime,
    retrieved_at: datetime,
    snapshot_id: str,
    source: str,
    qualified_contract: BrokerSafetyContract,
) -> BrokerContractSafetySnapshot:
    """Produce contract-only evidence through a one-shot runtime capability."""

    capability = _consume_broker_contract_snapshot_capability(capability)
    proof_payload = _broker_contract_capability_payload(capability)
    ibc_proof_id = (
        "ibc-proof-v1-"
        + hmac.new(
            _IBC_PROOF_KEY,
            proof_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
    )
    snapshot = BrokerContractSafetySnapshot(
        account_scope=capability.account_scope,
        requested_symbol=capability.requested_symbol,
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
        retrieved_at=retrieved_at,
        snapshot_id=snapshot_id,
        source=source,
        transport_generation=capability.transport_generation,
        runtime_fingerprint=capability.runtime_fingerprint,
        broker_host=capability.broker_host,
        broker_port=capability.broker_port,
        broker_client_id=capability.broker_client_id,
        read_only=capability.read_only,
        ibc_proof_id=ibc_proof_id,
        ibc_proof_source="validated-ibc-readonly-paper-v1",
        qualified_contract=qualified_contract,
        _producer_marker=_CONTRACT_PRODUCER_MARKER,
    )
    object_id = id(snapshot)

    def discard(reference: weakref.ReferenceType[BrokerContractSafetySnapshot]) -> None:
        _discard_broker_contract_snapshot(object_id, reference)

    reference = weakref.ref(snapshot, discard)
    digest = _broker_contract_snapshot_digest(snapshot)
    with _CONTRACT_SNAPSHOT_REGISTRY_LOCK:
        _CONTRACT_SNAPSHOT_REGISTRY[object_id] = (reference, digest)
    return snapshot


def assert_producer_owned_broker_contract_safety_snapshot(
    snapshot: BrokerContractSafetySnapshot,
) -> None:
    """Reject copied, replayed, forged, or mutated contract-only evidence."""

    if type(snapshot) is not BrokerContractSafetySnapshot:
        raise ValidationError("snapshot must be an exact BrokerContractSafetySnapshot")
    with _CONTRACT_SNAPSHOT_REGISTRY_LOCK:
        registered = _CONTRACT_SNAPSHOT_REGISTRY.get(id(snapshot))
        if registered is None or registered[0]() is not snapshot:
            raise ValidationError("BrokerContractSafetySnapshot is not producer-owned")
        digest = _broker_contract_snapshot_digest(snapshot)
        if not hmac.compare_digest(registered[1], digest):
            raise ValidationError("BrokerContractSafetySnapshot changed after production")

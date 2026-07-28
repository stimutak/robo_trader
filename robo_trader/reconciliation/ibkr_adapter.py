"""Fail-closed adapter for the dedicated IBKR diagnostic transport."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import re
import secrets
import threading
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Awaitable, Mapping, Protocol, SupportsIndex

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient

from .domain import (
    BrokerCollectionEvidence,
    BrokerCollectionKind,
    BrokerEvidenceCompleteness,
    BrokerOrderCollection,
    BrokerOrderSide,
    NormalizedBrokerAccount,
    NormalizedBrokerExecution,
    NormalizedBrokerOrder,
    NormalizedBrokerPosition,
    NormalizedBrokerSnapshot,
    ReconciliationDomainError,
)
from .errors import BrokerEvidenceError
from .identity import RuntimeSafetyContext, mask_account_identifier
from .models import (
    BrokerExecution,
    BrokerExecutionScope,
    BrokerOpenOrder,
    BrokerPosition,
    BrokerSnapshot,
    ContractIdentity,
    canonical_decimal,
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "snapshot_schema_version",
        "account",
        "account_type",
        "account_structure",
        "base_currency",
        "total_cash",
        "buying_power",
        "account_observed_at",
        "broker_time_before",
        "broker_time_after",
        "retrieved_at",
        "positions",
        "balances",
        "open_orders",
        "completed_orders",
        "executions",
        "execution_scope",
        "completeness",
        "collection_evidence",
    }
)
_CONTRACT_KEYS = frozenset(
    {
        "con_id",
        "symbol",
        "local_symbol",
        "security_type",
        "currency",
        "exchange",
        "primary_exchange",
        "trading_class",
    }
)
_POSITION_KEYS = frozenset({"account", "contract", "quantity", "avg_cost"})
_BALANCE_KEYS = frozenset({"tag", "currency", "value"})
_OPEN_ORDER_KEYS = frozenset(
    {
        "account",
        "broker_order_id",
        "permanent_id",
        "client_id",
        "contract",
        "side",
        "status",
        "order_type",
        "time_in_force",
        "total_quantity",
        "filled_quantity",
        "remaining_quantity",
        "limit_price",
        "stop_price",
        "avg_fill_price",
        "last_status_at",
        "unavailable",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "account",
        "execution_id",
        "broker_order_id",
        "permanent_id",
        "client_id",
        "contract",
        "side",
        "quantity",
        "price",
        "average_price",
        "executed_at",
        "execution_exchange",
        "commission",
        "commission_currency",
        "realized_pnl",
        "unavailable",
    }
)
_EXECUTION_SCOPE_KEYS = frozenset({"kind", "start_at", "end_at"})
_ORDER_OPTIONAL_KEYS = frozenset(
    {"permanent_id", "limit_price", "stop_price", "avg_fill_price", "last_status_at"}
)
_EXECUTION_OPTIONAL_KEYS = frozenset({"broker_order_id", "permanent_id", "realized_pnl"})
_COMPLETENESS_KEYS = frozenset(
    {
        "account",
        "positions",
        "open_orders",
        "completed_orders",
        "executions",
        "commissions",
    }
)
_COLLECTION_EVIDENCE_KEYS = frozenset(
    {"collection", "evidence_id", "observed_at", "result_count", "scope"}
)
_COMPLETED_ORDER_SCOPE_KEYS = frozenset(
    {
        "kind",
        "api_method",
        "api_only",
        "all_clients",
        "request_count",
        "stability_check",
        "retention_scope",
        "full_history",
        "request_started_at",
        "request_completed_at",
        "verification_started_at",
        "verification_completed_at",
        "broker_time_before",
        "broker_time_after",
    }
)
_BROKER_RESULT_PURPOSE = "bootstrap-broker-signing-v1"
_BROKER_RESULT_MARKER = object()
_BROKER_RESULT_REGISTRY_KEY = secrets.token_bytes(32)
_BROKER_RESULT_REGISTRY_LOCK = threading.Lock()
_BROKER_CAPABILITY_REGISTRY_KEY = secrets.token_bytes(32)
_BROKER_CAPABILITY_REGISTRY_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class CompletedOrderCollectionScope:
    """Exact, bounded semantics of IBKR's completed-order collection request."""

    kind: str
    api_method: str
    api_only: bool
    all_clients: bool
    request_count: int
    stability_check: str
    retention_scope: str
    full_history: bool
    request_started_at: datetime
    request_completed_at: datetime
    verification_started_at: datetime
    verification_completed_at: datetime
    broker_time_before: datetime
    broker_time_after: datetime

    def canonical_dict(self) -> dict[str, object]:
        return {
            "all_clients": self.all_clients,
            "api_method": self.api_method,
            "api_only": self.api_only,
            "broker_time_after": self.broker_time_after.isoformat(),
            "broker_time_before": self.broker_time_before.isoformat(),
            "full_history": self.full_history,
            "kind": self.kind,
            "request_count": self.request_count,
            "request_completed_at": self.request_completed_at.isoformat(),
            "request_started_at": self.request_started_at.isoformat(),
            "retention_scope": self.retention_scope,
            "stability_check": self.stability_check,
            "verification_completed_at": self.verification_completed_at.isoformat(),
            "verification_started_at": self.verification_started_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class _BrokerSnapshotResultCapability:
    purpose: str
    producer_id: int
    nonce: str
    _marker: object = field(repr=False, compare=False)


_BROKER_CAPABILITY_REGISTRY: dict[
    int,
    tuple[_BrokerSnapshotResultCapability, str],
] = {}


def _capability_digest(capability: _BrokerSnapshotResultCapability) -> str:
    return hmac.new(
        _BROKER_CAPABILITY_REGISTRY_KEY,
        (f"{capability.purpose}|{capability.producer_id}|{capability.nonce}").encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _consume_result_capability(capability: _BrokerSnapshotResultCapability) -> None:
    if type(capability) is not _BrokerSnapshotResultCapability:
        raise BrokerEvidenceError("broker producer result capability is invalid")
    digest = _capability_digest(capability)
    with _BROKER_CAPABILITY_REGISTRY_LOCK:
        registered = _BROKER_CAPABILITY_REGISTRY.pop(id(capability), None)
    if (
        registered is None
        or registered[0] is not capability
        or capability._marker is not _BROKER_RESULT_MARKER
        or capability.purpose != _BROKER_RESULT_PURPOSE
        or capability.producer_id <= 0
        or not hmac.compare_digest(registered[1], digest)
    ):
        raise BrokerEvidenceError("broker producer capability is absent or already consumed")


@dataclass(frozen=True, repr=False)
class BrokerSnapshotProducerResult:
    """One-shot producer-owned handoff exclusively for the broker signer."""

    __slots__ = (
        "snapshot",
        "completed_order_scope",
        "purpose",
        "_producer_marker",
        "_construction_capability",
        "__weakref__",
    )

    snapshot: NormalizedBrokerSnapshot
    completed_order_scope: CompletedOrderCollectionScope
    purpose: str
    _producer_marker: object
    _construction_capability: _BrokerSnapshotResultCapability

    def __post_init__(self) -> None:
        if type(self.snapshot) is not NormalizedBrokerSnapshot:
            raise BrokerEvidenceError("broker producer result is not normalized")
        if type(self.completed_order_scope) is not CompletedOrderCollectionScope:
            raise BrokerEvidenceError("broker producer completed-order scope is invalid")
        if self.purpose != _BROKER_RESULT_PURPOSE:
            raise BrokerEvidenceError("broker producer result purpose is invalid")
        if self._producer_marker is not _BROKER_RESULT_MARKER:
            raise BrokerEvidenceError("broker producer result requires producer ownership")
        _consume_result_capability(self._construction_capability)

    @property
    def snapshot_id(self) -> str:
        return self.snapshot.snapshot_id

    @property
    def canonical_payload(self) -> str:
        return _broker_result_payload(self)

    def __copy__(self) -> BrokerSnapshotProducerResult:
        raise TypeError("broker producer result cannot be copied")

    def __deepcopy__(self, memo: object) -> BrokerSnapshotProducerResult:
        raise TypeError("broker producer result cannot be copied")

    def __reduce__(self) -> str | tuple[Any, ...]:
        raise TypeError("broker producer result cannot be pickled")

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        raise TypeError("broker producer result cannot be pickled")


_BrokerResultRegistryEntry = tuple[
    weakref.ReferenceType[BrokerSnapshotProducerResult],
    str,
    str,
]
_BROKER_RESULT_REGISTRY: dict[int, _BrokerResultRegistryEntry] = {}


def _broker_result_payload(result: BrokerSnapshotProducerResult) -> str:
    return json.dumps(
        {
            "completed_order_collection_scope": result.completed_order_scope.canonical_dict(),
            "purpose": result.purpose,
            "snapshot": json.loads(result.snapshot.canonical_payload()),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _broker_result_digest(result: BrokerSnapshotProducerResult) -> str:
    return hmac.new(
        _BROKER_RESULT_REGISTRY_KEY,
        f"{result.snapshot_id}|{_broker_result_payload(result)}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _produce_broker_snapshot_result(
    producer: object,
    *,
    snapshot: NormalizedBrokerSnapshot,
    completed_order_scope: CompletedOrderCollectionScope,
) -> BrokerSnapshotProducerResult:
    if type(producer) is not IBKRDiagnosticSnapshotProvider:
        raise BrokerEvidenceError("broker producer result requires the diagnostic provider")
    capability = _BrokerSnapshotResultCapability(
        purpose=_BROKER_RESULT_PURPOSE,
        producer_id=id(producer),
        nonce=secrets.token_hex(32),
        _marker=_BROKER_RESULT_MARKER,
    )
    capability_digest = _capability_digest(capability)
    with _BROKER_CAPABILITY_REGISTRY_LOCK:
        _BROKER_CAPABILITY_REGISTRY[id(capability)] = (capability, capability_digest)
    result = BrokerSnapshotProducerResult(
        snapshot=snapshot,
        completed_order_scope=completed_order_scope,
        purpose=capability.purpose,
        _producer_marker=_BROKER_RESULT_MARKER,
        _construction_capability=capability,
    )
    object_id = id(result)

    def discard(reference: weakref.ReferenceType[BrokerSnapshotProducerResult]) -> None:
        with _BROKER_RESULT_REGISTRY_LOCK:
            registered = _BROKER_RESULT_REGISTRY.get(object_id)
            if registered is not None and registered[0] is reference:
                _BROKER_RESULT_REGISTRY.pop(object_id, None)

    reference = weakref.ref(result, discard)
    digest = _broker_result_digest(result)
    with _BROKER_RESULT_REGISTRY_LOCK:
        _BROKER_RESULT_REGISTRY[object_id] = (reference, digest, capability.purpose)
    return result


def assert_producer_owned_broker_snapshot_result(
    result: BrokerSnapshotProducerResult,
) -> BrokerSnapshotProducerResult:
    """Claim the exact signer-purpose handoff once; reject forgery and replay."""

    if type(result) is not BrokerSnapshotProducerResult:
        raise BrokerEvidenceError("exact broker producer result is required")
    digest = _broker_result_digest(result)
    with _BROKER_RESULT_REGISTRY_LOCK:
        registered = _BROKER_RESULT_REGISTRY.pop(id(result), None)
        if registered is None or registered[0]() is not result:
            raise BrokerEvidenceError("broker producer result is absent or already consumed")
        if (
            registered[2] != _BROKER_RESULT_PURPOSE
            or result.purpose != _BROKER_RESULT_PURPOSE
            or result._producer_marker is not _BROKER_RESULT_MARKER
            or not hmac.compare_digest(registered[1], digest)
        ):
            raise BrokerEvidenceError("broker producer result failed ownership validation")
    return result


class DiagnosticTransport(Protocol):
    """Transport subset available to this adapter."""

    async def start(self) -> None:
        """Start an isolated diagnostic worker."""

    async def connect(
        self,
        host: str,
        port: int,
        client_id: int,
        readonly: bool,
        timeout: float = 30.0,
    ) -> bool:
        """Connect using the already validated diagnostic identity."""

    async def get_broker_snapshot(
        self, expected_account: str, *, max_age_seconds: float
    ) -> dict[str, Any]:
        """Return the transport-v1 diagnostic snapshot."""

    async def stop(self) -> None:
        """Stop and reap the isolated worker."""


def _record(value: object, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise BrokerEvidenceError(f"diagnostic broker {label} schema is invalid")
    return value


def _records(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise BrokerEvidenceError(f"diagnostic broker {label} must be a list")
    return value


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    return parsed


def _optional_timestamp(value: object, label: str) -> datetime | None:
    return None if value is None else _timestamp(value, label)


def _completed_order_scope(
    value: object,
    *,
    broker_time_before: datetime,
    broker_time_after: datetime,
) -> CompletedOrderCollectionScope:
    record = _record(value, _COMPLETED_ORDER_SCOPE_KEYS, "completed-order scope")
    if (
        record["kind"] != "ibkr_current_retained_completed_orders"
        or record["api_method"] != "reqCompletedOrders"
        or record["api_only"] is not False
        or record["all_clients"] is not True
        or type(record["request_count"]) is not int
        or record["request_count"] != 2
        or record["stability_check"] != "identical_second_read"
        or record["retention_scope"] != "current_tws_or_gateway_retained_set"
        or record["full_history"] is not False
    ):
        raise BrokerEvidenceError("diagnostic broker completed-order scope is unsupported")
    scope = CompletedOrderCollectionScope(
        kind=record["kind"],
        api_method=record["api_method"],
        api_only=record["api_only"],
        all_clients=record["all_clients"],
        request_count=record["request_count"],
        stability_check=record["stability_check"],
        retention_scope=record["retention_scope"],
        full_history=record["full_history"],
        request_started_at=_timestamp(record["request_started_at"], "request start"),
        request_completed_at=_timestamp(record["request_completed_at"], "request completion"),
        verification_started_at=_timestamp(record["verification_started_at"], "verification start"),
        verification_completed_at=_timestamp(
            record["verification_completed_at"], "verification completion"
        ),
        broker_time_before=_timestamp(record["broker_time_before"], "scope broker time before"),
        broker_time_after=_timestamp(record["broker_time_after"], "scope broker time after"),
    )
    if (
        scope.broker_time_before != broker_time_before
        or scope.broker_time_after != broker_time_after
    ):
        raise BrokerEvidenceError("diagnostic broker completed-order bounds are inconsistent")
    if not (
        scope.request_started_at
        <= scope.request_completed_at
        <= scope.verification_started_at
        <= scope.verification_completed_at
    ):
        raise BrokerEvidenceError("diagnostic broker completed-order request bounds are reversed")
    if (
        (scope.verification_completed_at - scope.request_started_at).total_seconds() > 60.0
        or scope.request_started_at < broker_time_before - timedelta(seconds=120)
        or scope.verification_completed_at > broker_time_after + timedelta(seconds=120)
    ):
        raise BrokerEvidenceError("diagnostic broker completed-order request bounds are unbounded")
    return scope


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    try:
        parsed = Decimal(value)
    except Exception as exc:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid") from exc
    if not parsed.is_finite() or canonical_decimal(parsed) != value:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    return parsed


def _optional_decimal(value: object, label: str) -> Decimal | None:
    return None if value is None else _decimal(value, label)


def _optional_identifier(value: object, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    return str(value)


def _integer_identifier(
    value: object,
    label: str,
    *,
    allow_zero: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    return value


def _unavailable(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(reason, str) or not reason
        for key, reason in value.items()
    ):
        raise BrokerEvidenceError("diagnostic broker unavailable evidence is invalid")
    return MappingProxyType(dict(value))


def _require_unavailable_consistency(
    record: Mapping[str, Any],
    optional_fields: frozenset[str],
) -> None:
    unavailable = _unavailable(record["unavailable"])
    if not set(unavailable).issubset(optional_fields):
        raise BrokerEvidenceError("diagnostic broker unavailable-field schema is invalid")
    for field_name in optional_fields:
        if record[field_name] is None and field_name not in unavailable:
            raise BrokerEvidenceError("diagnostic broker silently omitted optional evidence")
        if record[field_name] is not None and field_name in unavailable:
            raise BrokerEvidenceError("diagnostic broker optional evidence is contradictory")


def _contract(value: object) -> ContractIdentity:
    record = _record(value, _CONTRACT_KEYS, "contract")
    return ContractIdentity(
        con_id=record["con_id"],
        symbol=record["symbol"],
        local_symbol=record["local_symbol"],
        security_type=record["security_type"],
        currency=record["currency"],
        exchange=record["exchange"],
        primary_exchange=record["primary_exchange"],
        trading_class=record["trading_class"],
    )


def _require_account(record: Mapping[str, Any], expected_account: str) -> None:
    if record["account"] != expected_account:
        raise BrokerEvidenceError("diagnostic broker account identity is inconsistent")


def _complete_collection_evidence(
    record: Mapping[str, Any],
    *,
    broker_time_before: datetime,
    broker_time_after: datetime,
    expected_counts: Mapping[BrokerCollectionKind, int],
) -> dict[BrokerCollectionKind, Mapping[str, Any]]:
    completeness = _record(record["completeness"], _COMPLETENESS_KEYS, "completeness")
    if any(completeness[field_name] is not True for field_name in _COMPLETENESS_KEYS):
        raise BrokerEvidenceError("diagnostic broker evidence is incomplete")

    evidence_by_kind: dict[BrokerCollectionKind, Mapping[str, Any]] = {}
    seen_ids: set[str] = set()
    for raw_evidence in _records(record["collection_evidence"], "collection evidence"):
        evidence = _record(
            raw_evidence,
            _COLLECTION_EVIDENCE_KEYS,
            "collection evidence",
        )
        try:
            kind = BrokerCollectionKind(evidence["collection"])
        except (TypeError, ValueError) as exc:
            raise BrokerEvidenceError(
                "diagnostic broker collection evidence kind is invalid"
            ) from exc
        if kind in evidence_by_kind:
            raise BrokerEvidenceError("diagnostic broker collection evidence is duplicated")
        evidence_id = evidence["evidence_id"]
        if not isinstance(evidence_id, str) or evidence_id in seen_ids:
            raise BrokerEvidenceError("diagnostic broker collection evidence ID is invalid")
        observed_at = _timestamp(evidence["observed_at"], "collection evidence timestamp")
        if not broker_time_before <= observed_at <= broker_time_after:
            raise BrokerEvidenceError(
                "diagnostic broker collection evidence is outside collection bounds"
            )
        result_count = evidence["result_count"]
        if (
            type(result_count) is not int
            or result_count < 0
            or result_count != expected_counts.get(kind)
        ):
            raise BrokerEvidenceError("diagnostic broker collection evidence count is inconsistent")
        if kind is BrokerCollectionKind.COMPLETED_ORDERS:
            _completed_order_scope(
                evidence["scope"],
                broker_time_before=broker_time_before,
                broker_time_after=broker_time_after,
            )
        elif evidence["scope"] is not None:
            raise BrokerEvidenceError("diagnostic broker collection scope is unexpected")
        evidence_by_kind[kind] = evidence
        seen_ids.add(evidence_id)
    if set(evidence_by_kind) != set(BrokerCollectionKind):
        raise BrokerEvidenceError("diagnostic broker collection evidence is incomplete")
    return evidence_by_kind


def snapshot_from_transport(
    payload: object,
    *,
    expected_account: str,
) -> BrokerSnapshot:
    """Convert one strictly validated transport-v1 payload into evidence models."""
    if not isinstance(expected_account, str) or not expected_account.strip():
        raise BrokerEvidenceError("diagnostic broker expected account is unavailable")
    record = _record(payload, _TOP_LEVEL_KEYS, "snapshot")
    if record["snapshot_schema_version"] != 3:
        raise BrokerEvidenceError("diagnostic broker snapshot schema is unsupported")
    if record["account"] != expected_account:
        raise BrokerEvidenceError("diagnostic broker account identity does not match runtime")

    positions = []
    for raw_position in _records(record["positions"], "positions"):
        position = _record(raw_position, _POSITION_KEYS, "position")
        _require_account(position, expected_account)
        positions.append(
            BrokerPosition(
                contract=_contract(position["contract"]),
                quantity=_decimal(position["quantity"], "position quantity"),
                average_cost=_decimal(position["avg_cost"], "position average cost"),
            )
        )

    balances: dict[str, Decimal] = {}
    for raw_balance in _records(record["balances"], "balances"):
        balance = _record(raw_balance, _BALANCE_KEYS, "balance")
        tag = balance["tag"]
        currency = balance["currency"]
        if not isinstance(tag, str) or not isinstance(currency, str):
            raise BrokerEvidenceError("diagnostic broker balance identity is invalid")
        identity = f"{tag}:{currency}"
        if identity in balances:
            raise BrokerEvidenceError("diagnostic broker balance identity is duplicated")
        balances[identity] = _decimal(balance["value"], "balance value")

    open_orders = []
    for raw_order in _records(record["open_orders"], "open orders"):
        order = _record(raw_order, _OPEN_ORDER_KEYS, "open order")
        _require_account(order, expected_account)
        order_id = _optional_identifier(order["broker_order_id"], "order ID")
        if order_id is None:
            raise BrokerEvidenceError("diagnostic broker order ID is missing")
        open_orders.append(
            BrokerOpenOrder(
                order_id=order_id,
                contract=_contract(order["contract"]),
                side=order["side"],
                quantity=_decimal(order["total_quantity"], "order quantity"),
                filled=_decimal(order["filled_quantity"], "order filled quantity"),
                remaining=_decimal(order["remaining_quantity"], "order remaining quantity"),
                order_type=order["order_type"],
                status=order["status"],
                limit_price=_optional_decimal(order["limit_price"], "order limit price"),
                auxiliary_price=_optional_decimal(order["stop_price"], "order stop price"),
                permanent_id=_optional_identifier(order["permanent_id"], "permanent ID"),
                client_id=order["client_id"],
                time_in_force=order["time_in_force"],
                average_fill_price=_optional_decimal(
                    order["avg_fill_price"], "order average fill price"
                ),
                last_status_at=_optional_timestamp(
                    order["last_status_at"], "order status timestamp"
                ),
                unavailable=_unavailable(order["unavailable"]),
            )
        )

    executions = []
    for raw_execution in _records(record["executions"], "executions"):
        execution = _record(raw_execution, _EXECUTION_KEYS, "execution")
        _require_account(execution, expected_account)
        execution_id = execution["execution_id"]
        if not isinstance(execution_id, str):
            raise BrokerEvidenceError("diagnostic broker execution ID is invalid")
        executions.append(
            BrokerExecution(
                execution_id=execution_id,
                order_id=_optional_identifier(execution["broker_order_id"], "execution order ID"),
                contract=_contract(execution["contract"]),
                side=execution["side"],
                quantity=_decimal(execution["quantity"], "execution quantity"),
                price=_decimal(execution["price"], "execution price"),
                executed_at=_timestamp(execution["executed_at"], "execution timestamp"),
                permanent_id=_optional_identifier(
                    execution["permanent_id"], "execution permanent ID"
                ),
                client_id=execution["client_id"],
                execution_exchange=execution["execution_exchange"],
                average_price=_optional_decimal(
                    execution["average_price"], "execution average price"
                ),
                commission=_optional_decimal(execution["commission"], "execution commission"),
                commission_currency=execution["commission_currency"],
                realized_pnl=_optional_decimal(execution["realized_pnl"], "execution realized PnL"),
                unavailable=_unavailable(execution["unavailable"]),
            )
        )

    execution_scope = _record(record["execution_scope"], _EXECUTION_SCOPE_KEYS, "execution scope")
    if execution_scope["kind"] != "bounded_execution_filter":
        raise BrokerEvidenceError("diagnostic broker execution scope is unsupported")
    scope_start = _timestamp(execution_scope["start_at"], "execution scope start")
    scope_end = _timestamp(execution_scope["end_at"], "execution scope end")
    broker_time_before = _timestamp(record["broker_time_before"], "broker time before")
    broker_time_after = _timestamp(record["broker_time_after"], "broker time after")
    retrieved_at = _timestamp(record["retrieved_at"], "retrieval timestamp")
    expected_scope_start = broker_time_before.replace(microsecond=0) - timedelta(hours=24)
    if scope_start != expected_scope_start:
        raise BrokerEvidenceError(
            "diagnostic broker execution scope does not match the wire filter"
        )
    if not broker_time_before <= scope_end <= broker_time_after:
        raise BrokerEvidenceError("diagnostic broker execution scope is inconsistent")
    if any(execution.commission is None for execution in executions):
        raise BrokerEvidenceError("diagnostic broker commission evidence is incomplete")
    _complete_collection_evidence(
        record,
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
        expected_counts={
            BrokerCollectionKind.POSITIONS: len(positions),
            BrokerCollectionKind.OPEN_ORDERS: len(open_orders),
            BrokerCollectionKind.COMPLETED_ORDERS: len(
                _records(record["completed_orders"], "completed orders")
            ),
            BrokerCollectionKind.EXECUTIONS: len(executions),
            BrokerCollectionKind.COMMISSIONS: len(executions),
        },
    )

    snapshot = BrokerSnapshot(
        schema_version=1,
        account_alias=mask_account_identifier(expected_account),
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
        retrieved_at=retrieved_at,
        execution_scope=BrokerExecutionScope(
            kind=execution_scope["kind"],
            start_at=scope_start,
            end_at=scope_end,
        ),
        positions=tuple(positions),
        open_orders=tuple(open_orders),
        recent_executions=tuple(executions),
        balances=MappingProxyType(balances),
    )
    return snapshot


def _normalized_snapshot_from_transport(
    payload: object,
    *,
    expected_account: str,
    account_scope: str,
    max_age_seconds: float,
    now: datetime | None = None,
) -> NormalizedBrokerSnapshot:
    """Map only complete, fresh transport-v2 evidence into the PR5 domain."""

    if not isinstance(expected_account, str) or not expected_account.strip():
        raise BrokerEvidenceError("diagnostic broker expected account is unavailable")
    record = _record(payload, _TOP_LEVEL_KEYS, "snapshot")
    if record["snapshot_schema_version"] != 3:
        raise BrokerEvidenceError("diagnostic broker snapshot schema is unsupported")
    if record["account"] != expected_account:
        raise BrokerEvidenceError("diagnostic broker account identity does not match runtime")
    if record["account_type"] != "paper":
        raise BrokerEvidenceError("diagnostic broker account type is not paper")
    account_structure = record["account_structure"]
    if (
        not isinstance(account_structure, str)
        or account_structure != account_structure.strip().upper()
        or not re.fullmatch(r"[A-Z][A-Z0-9 _-]{0,63}", account_structure)
    ):
        raise BrokerEvidenceError("diagnostic broker account structure is unavailable")

    broker_time_before = _timestamp(record["broker_time_before"], "broker time before")
    broker_time_after = _timestamp(record["broker_time_after"], "broker time after")
    retrieved_at = _timestamp(record["retrieved_at"], "retrieval timestamp")
    account_observed_at = _timestamp(
        record["account_observed_at"],
        "account observation timestamp",
    )
    if broker_time_after < broker_time_before:
        raise BrokerEvidenceError("diagnostic broker collection window is reversed")
    if not broker_time_before <= account_observed_at <= broker_time_after:
        raise BrokerEvidenceError("diagnostic broker account evidence is outside bounds")
    execution_scope = _record(
        record["execution_scope"],
        _EXECUTION_SCOPE_KEYS,
        "execution scope",
    )
    if execution_scope["kind"] != "bounded_execution_filter":
        raise BrokerEvidenceError("diagnostic broker execution scope is unsupported")
    execution_start = _timestamp(execution_scope["start_at"], "execution scope start")
    execution_end = _timestamp(execution_scope["end_at"], "execution scope end")
    if execution_start != broker_time_before.replace(microsecond=0) - timedelta(hours=24):
        raise BrokerEvidenceError(
            "diagnostic broker execution scope does not match the wire filter"
        )
    if not broker_time_before <= execution_end <= broker_time_after:
        raise BrokerEvidenceError("diagnostic broker execution scope is inconsistent")

    raw_positions = _records(record["positions"], "positions")
    raw_open_orders = _records(record["open_orders"], "open orders")
    raw_completed_orders = _records(record["completed_orders"], "completed orders")
    raw_executions = _records(record["executions"], "executions")
    evidence_by_kind = _complete_collection_evidence(
        record,
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
        expected_counts={
            BrokerCollectionKind.POSITIONS: len(raw_positions),
            BrokerCollectionKind.OPEN_ORDERS: len(raw_open_orders),
            BrokerCollectionKind.COMPLETED_ORDERS: len(raw_completed_orders),
            BrokerCollectionKind.EXECUTIONS: len(raw_executions),
            BrokerCollectionKind.COMMISSIONS: len(raw_executions),
        },
    )

    balances: dict[tuple[str, str], Decimal] = {}
    for raw_balance in _records(record["balances"], "balances"):
        balance = _record(raw_balance, _BALANCE_KEYS, "balance")
        tag = balance["tag"]
        currency = balance["currency"]
        if not isinstance(tag, str) or not isinstance(currency, str):
            raise BrokerEvidenceError("diagnostic broker balance identity is invalid")
        identity = (tag, currency)
        if identity in balances:
            raise BrokerEvidenceError("diagnostic broker balance identity is duplicated")
        balances[identity] = _decimal(balance["value"], "balance value")
    base_currency = record["base_currency"]
    if not isinstance(base_currency, str):
        raise BrokerEvidenceError("diagnostic broker base currency is invalid")
    total_cash = _decimal(record["total_cash"], "total cash")
    buying_power = _decimal(record["buying_power"], "buying power")
    if buying_power < 0:
        raise BrokerEvidenceError("diagnostic broker buying power is negative")
    if balances.get(("TotalCashValue", base_currency)) != total_cash:
        raise BrokerEvidenceError("diagnostic broker total cash evidence is inconsistent")
    if balances.get(("BuyingPower", base_currency)) != buying_power:
        raise BrokerEvidenceError("diagnostic broker buying power evidence is inconsistent")
    if ("NetLiquidation", base_currency) not in balances:
        raise BrokerEvidenceError("diagnostic broker base currency evidence is incomplete")

    position_observed_at = _timestamp(
        evidence_by_kind[BrokerCollectionKind.POSITIONS]["observed_at"],
        "position collection timestamp",
    )
    positions = []
    for raw_position in raw_positions:
        position = _record(raw_position, _POSITION_KEYS, "position")
        _require_account(position, expected_account)
        contract = _contract(position["contract"])
        positions.append(
            NormalizedBrokerPosition(
                account_scope=account_scope,
                con_id=contract.con_id,
                symbol=contract.symbol,
                currency=contract.currency,
                signed_quantity=_decimal(position["quantity"], "position quantity"),
                average_cost=_decimal(position["avg_cost"], "position average cost"),
                observed_at=position_observed_at,
            )
        )

    orders = []
    order_identities: dict[BrokerOrderCollection, set[tuple[int, int]]] = {
        BrokerOrderCollection.OPEN: set(),
        BrokerOrderCollection.COMPLETED: set(),
    }
    for collection, kind, raw_orders in (
        (
            BrokerOrderCollection.OPEN,
            BrokerCollectionKind.OPEN_ORDERS,
            raw_open_orders,
        ),
        (
            BrokerOrderCollection.COMPLETED,
            BrokerCollectionKind.COMPLETED_ORDERS,
            raw_completed_orders,
        ),
    ):
        observed_at = _timestamp(
            evidence_by_kind[kind]["observed_at"],
            f"{kind.value} collection timestamp",
        )
        for raw_order in raw_orders:
            order = _record(raw_order, _OPEN_ORDER_KEYS, f"{collection.value} order")
            _require_account(order, expected_account)
            _require_unavailable_consistency(order, _ORDER_OPTIONAL_KEYS)
            contract = _contract(order["contract"])
            broker_order_id = _integer_identifier(order["broker_order_id"], "order ID")
            client_id = _integer_identifier(
                order["client_id"],
                "order client ID",
                allow_zero=True,
            )
            order_identity = (client_id, broker_order_id)
            if order_identity in order_identities[collection]:
                raise BrokerEvidenceError("diagnostic broker order identity is duplicated")
            order_identities[collection].add(order_identity)
            try:
                side = BrokerOrderSide(order["side"])
            except (TypeError, ValueError) as exc:
                raise BrokerEvidenceError("diagnostic broker order side is invalid") from exc
            if collection is BrokerOrderCollection.COMPLETED and order["status"] not in {
                "ApiCancelled",
                "Cancelled",
                "Filled",
                "Inactive",
            }:
                raise BrokerEvidenceError("diagnostic broker completed order is not terminal")
            orders.append(
                NormalizedBrokerOrder(
                    account_scope=account_scope,
                    collection=collection,
                    broker_order_id=broker_order_id,
                    client_id=client_id,
                    con_id=contract.con_id,
                    symbol=contract.symbol,
                    side=side,
                    total_quantity=_decimal(order["total_quantity"], "order quantity"),
                    filled_quantity=_decimal(
                        order["filled_quantity"],
                        "order filled quantity",
                    ),
                    remaining_quantity=_decimal(
                        order["remaining_quantity"],
                        "order remaining quantity",
                    ),
                    status=order["status"],
                    observed_at=observed_at,
                    permanent_id=(
                        None
                        if order["permanent_id"] is None
                        else _integer_identifier(order["permanent_id"], "permanent ID")
                    ),
                )
            )
    if order_identities[BrokerOrderCollection.OPEN].intersection(
        order_identities[BrokerOrderCollection.COMPLETED]
    ):
        raise BrokerEvidenceError("diagnostic broker order spans open and completed collections")

    executions = []
    for raw_execution in raw_executions:
        execution = _record(raw_execution, _EXECUTION_KEYS, "execution")
        _require_account(execution, expected_account)
        _require_unavailable_consistency(execution, _EXECUTION_OPTIONAL_KEYS)
        contract = _contract(execution["contract"])
        commission = _decimal(execution["commission"], "execution commission")
        commission_currency = execution["commission_currency"]
        if not isinstance(commission_currency, str):
            raise BrokerEvidenceError("diagnostic broker commission currency is invalid")
        try:
            side = BrokerOrderSide(execution["side"])
        except (TypeError, ValueError) as exc:
            raise BrokerEvidenceError("diagnostic broker execution side is invalid") from exc
        executed_at = _timestamp(execution["executed_at"], "execution timestamp")
        if not execution_start <= executed_at <= execution_end:
            raise BrokerEvidenceError("diagnostic broker execution is outside requested scope")
        executions.append(
            NormalizedBrokerExecution(
                account_scope=account_scope,
                execution_id=execution["execution_id"],
                con_id=contract.con_id,
                symbol=contract.symbol,
                side=side,
                quantity=_decimal(execution["quantity"], "execution quantity"),
                price=_decimal(execution["price"], "execution price"),
                executed_at=executed_at,
                broker_order_id=(
                    None
                    if execution["broker_order_id"] is None
                    else _integer_identifier(
                        execution["broker_order_id"],
                        "execution order ID",
                    )
                ),
                permanent_id=(
                    None
                    if execution["permanent_id"] is None
                    else _integer_identifier(
                        execution["permanent_id"],
                        "execution permanent ID",
                    )
                ),
                commission=commission,
                commission_currency=commission_currency,
            )
        )

    completeness_record = _record(
        record["completeness"],
        _COMPLETENESS_KEYS,
        "completeness",
    )
    completeness = BrokerEvidenceCompleteness(
        **{field_name: completeness_record[field_name] for field_name in _COMPLETENESS_KEYS}
    )
    collection_evidence = tuple(
        BrokerCollectionEvidence(
            account_scope=account_scope,
            collection=kind,
            evidence_id=evidence["evidence_id"],
            result_count=evidence["result_count"],
            observed_at=_timestamp(
                evidence["observed_at"],
                "collection evidence timestamp",
            ),
        )
        for kind, evidence in sorted(
            evidence_by_kind.items(),
            key=lambda item: item[0].value,
        )
    )
    try:
        snapshot = NormalizedBrokerSnapshot(
            account=NormalizedBrokerAccount(
                account_scope=account_scope,
                account_alias=mask_account_identifier(expected_account),
                account_type=record["account_type"],
                base_currency=base_currency,
                total_cash=total_cash,
                buying_power=buying_power,
                observed_at=account_observed_at,
            ),
            observed_from=broker_time_before,
            observed_through=broker_time_after,
            retrieved_at=retrieved_at,
            completeness=completeness,
            collection_evidence=collection_evidence,
            positions=tuple(positions),
            orders=tuple(orders),
            executions=tuple(executions),
        )
        checked_at = datetime.now(timezone.utc) if now is None else now
        if not snapshot.is_fresh(now=checked_at, max_age_seconds=max_age_seconds):
            raise BrokerEvidenceError("diagnostic broker normalized snapshot is stale")
        return snapshot
    except ReconciliationDomainError as exc:
        raise BrokerEvidenceError("diagnostic broker normalized snapshot is invalid") from exc


def normalized_snapshot_from_transport(
    payload: object,
    *,
    expected_account: str,
    account_scope: str,
    max_age_seconds: float,
    now: datetime | None = None,
) -> NormalizedBrokerSnapshot:
    """Expose normalized mapping with one fail-closed adapter error surface."""

    try:
        return _normalized_snapshot_from_transport(
            payload,
            expected_account=expected_account,
            account_scope=account_scope,
            max_age_seconds=max_age_seconds,
            now=now,
        )
    except ReconciliationDomainError as exc:
        raise BrokerEvidenceError("diagnostic broker normalized snapshot is invalid") from exc


def _completed_order_scope_from_transport(payload: object) -> CompletedOrderCollectionScope:
    record = _record(payload, _TOP_LEVEL_KEYS, "snapshot")
    broker_time_before = _timestamp(record["broker_time_before"], "broker time before")
    broker_time_after = _timestamp(record["broker_time_after"], "broker time after")
    matches = [
        evidence
        for evidence in _records(record["collection_evidence"], "collection evidence")
        if isinstance(evidence, dict) and evidence.get("collection") == "completed_orders"
    ]
    if len(matches) != 1:
        raise BrokerEvidenceError("diagnostic broker completed-order scope is unavailable")
    evidence = _record(matches[0], _COLLECTION_EVIDENCE_KEYS, "collection evidence")
    return _completed_order_scope(
        evidence["scope"],
        broker_time_before=broker_time_before,
        broker_time_after=broker_time_after,
    )


class IBKRDiagnosticSnapshotProvider:
    """Expose only snapshot and cleanup capabilities to reconciliation."""

    __slots__ = ("_transport", "_expected_account", "_account_scope")

    def __init__(
        self,
        transport: DiagnosticTransport,
        *,
        expected_account: str,
        account_scope: str | None = None,
    ) -> None:
        self._transport = transport
        self._expected_account = expected_account
        self._account_scope = account_scope

    async def get_broker_snapshot(
        self, expected_account: str, *, max_age_seconds: float
    ) -> BrokerSnapshot:
        if expected_account != self._expected_account:
            raise BrokerEvidenceError("diagnostic broker account identity does not match runtime")
        payload = await self._transport.get_broker_snapshot(
            expected_account,
            max_age_seconds=max_age_seconds,
        )
        return snapshot_from_transport(payload, expected_account=expected_account)

    async def produce_normalized_snapshot(
        self,
        *,
        max_age_seconds: float = 30.0,
    ) -> BrokerSnapshotProducerResult:
        """Produce unsigned canonical PR5 evidence for a separate core signer."""

        if not isinstance(self._account_scope, str) or not self._account_scope:
            raise BrokerEvidenceError("diagnostic broker account scope is unavailable")
        payload = await self._transport.get_broker_snapshot(
            self._expected_account,
            max_age_seconds=max_age_seconds,
        )
        snapshot = normalized_snapshot_from_transport(
            payload,
            expected_account=self._expected_account,
            account_scope=self._account_scope,
            max_age_seconds=max_age_seconds,
        )
        completed_order_scope = _completed_order_scope_from_transport(payload)
        return _produce_broker_snapshot_result(
            self,
            snapshot=snapshot,
            completed_order_scope=completed_order_scope,
        )

    async def close(self) -> None:
        await _stop_transport_required(self._transport)


async def _stop_transport_required(
    transport: DiagnosticTransport,
    *,
    attempts: int = 2,
) -> None:
    """Require provider cleanup and retry one transient transport failure."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            await transport.stop()
            return
        except Exception as exc:
            last_error = exc
    raise BrokerEvidenceError("diagnostic broker transport cleanup failed") from last_error


async def await_cleanup_required(cleanup: Awaitable[None]) -> bool:
    """Finish required cleanup despite caller cancellation.

    Return whether cancellation arrived while cleanup was shielded so the
    caller can re-raise it only after the transport is known to be reaped.
    """

    cleanup_task = asyncio.ensure_future(cleanup)
    cancellation_received = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cancellation_received = True
            continue
    try:
        cleanup_task.result()
    except asyncio.CancelledError as exc:
        raise BrokerEvidenceError("diagnostic broker cleanup was cancelled internally") from exc
    return cancellation_received


async def build_diagnostic_provider(
    runtime: RuntimeSafetyContext,
    *,
    transport_factory=SubprocessIBKRClient,
) -> IBKRDiagnosticSnapshotProvider:
    """Start a dedicated paper/read-only transport or fail closed and reap it."""
    connection = runtime.diagnostic_connection
    expected_account = runtime.expected_account_for_provider
    if not isinstance(expected_account, str) or not expected_account.strip():
        raise BrokerEvidenceError("diagnostic broker expected account is unavailable")
    transport = transport_factory()
    try:
        await transport.start()
        connected = await transport.connect(
            host=connection.host,
            port=connection.port,
            client_id=connection.client_id,
            readonly=connection.readonly,
            timeout=30.0,
        )
        if connected is not True:
            raise BrokerEvidenceError("diagnostic broker connection was not established")
    except BaseException as exc:
        try:
            cleanup_cancelled = await await_cleanup_required(_stop_transport_required(transport))
        except Exception as cleanup_exc:
            raise BrokerEvidenceError(
                "diagnostic broker provider initialization cleanup failed"
            ) from cleanup_exc
        if isinstance(exc, asyncio.CancelledError):
            raise
        if not isinstance(exc, Exception):
            raise
        if cleanup_cancelled:
            raise asyncio.CancelledError
        raise BrokerEvidenceError("diagnostic broker provider initialization failed") from exc
    return IBKRDiagnosticSnapshotProvider(
        transport,
        expected_account=expected_account,
        account_scope=getattr(runtime.runtime_contract, "safety_account_scope", None),
    )


async def diagnostic_provider_factory(
    runtime: RuntimeSafetyContext,
) -> IBKRDiagnosticSnapshotProvider:
    """Production reconciliation provider factory."""
    return await build_diagnostic_provider(runtime)

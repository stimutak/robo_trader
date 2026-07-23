"""Fail-closed adapter for the dedicated IBKR diagnostic transport."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient

from .errors import BrokerEvidenceError
from .identity import RuntimeSafetyContext, mask_account_identifier
from .models import (
    BrokerExecution,
    BrokerExecutionScope,
    BrokerOpenOrder,
    BrokerPosition,
    BrokerSnapshot,
    ContractIdentity,
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "snapshot_schema_version",
        "account",
        "broker_time_before",
        "broker_time_after",
        "retrieved_at",
        "positions",
        "balances",
        "open_orders",
        "executions",
        "execution_scope",
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


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid")
    try:
        parsed = Decimal(value)
    except Exception as exc:
        raise BrokerEvidenceError(f"diagnostic broker {label} is invalid") from exc
    if not parsed.is_finite() or str(parsed) != value:
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


def _unavailable(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(reason, str) or not reason
        for key, reason in value.items()
    ):
        raise BrokerEvidenceError("diagnostic broker unavailable evidence is invalid")
    return MappingProxyType(dict(value))


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


def snapshot_from_transport(
    payload: object,
    *,
    expected_account: str,
) -> BrokerSnapshot:
    """Convert one strictly validated transport-v1 payload into evidence models."""
    if not isinstance(expected_account, str) or not expected_account.strip():
        raise BrokerEvidenceError("diagnostic broker expected account is unavailable")
    record = _record(payload, _TOP_LEVEL_KEYS, "snapshot")
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

    snapshot = BrokerSnapshot(
        schema_version=record["snapshot_schema_version"],
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


class IBKRDiagnosticSnapshotProvider:
    """Expose only snapshot and cleanup capabilities to reconciliation."""

    __slots__ = ("_transport", "_expected_account")

    def __init__(
        self,
        transport: DiagnosticTransport,
        *,
        expected_account: str,
    ) -> None:
        self._transport = transport
        self._expected_account = expected_account

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
    except Exception as exc:
        try:
            await _stop_transport_required(transport)
        except Exception as cleanup_exc:
            raise BrokerEvidenceError(
                "diagnostic broker provider initialization cleanup failed"
            ) from cleanup_exc
        raise BrokerEvidenceError("diagnostic broker provider initialization failed") from exc
    return IBKRDiagnosticSnapshotProvider(
        transport,
        expected_account=expected_account,
    )


async def diagnostic_provider_factory(
    runtime: RuntimeSafetyContext,
) -> IBKRDiagnosticSnapshotProvider:
    """Production reconciliation provider factory."""
    return await build_diagnostic_provider(runtime)

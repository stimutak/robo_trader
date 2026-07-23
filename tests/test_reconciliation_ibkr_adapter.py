from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.clients import subprocess_ibkr_client as client_module
from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient
from robo_trader.reconciliation.broker import assert_read_only_provider_surface
from robo_trader.reconciliation.errors import BrokerEvidenceError
from robo_trader.reconciliation.ibkr_adapter import (
    IBKRDiagnosticSnapshotProvider,
    build_diagnostic_provider,
    snapshot_from_transport,
)

ACCOUNT = "DU_TEST_4567"
NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _contract():
    return {
        "con_id": 265598,
        "symbol": "AAPL",
        "local_symbol": "AAPL",
        "security_type": "STK",
        "currency": "USD",
        "exchange": "SMART",
        "primary_exchange": "NASDAQ",
        "trading_class": "NMS",
    }


def _payload():
    return {
        "snapshot_schema_version": 1,
        "account": ACCOUNT,
        "broker_time_before": (NOW - timedelta(seconds=2)).isoformat(),
        "broker_time_after": NOW.isoformat(),
        "retrieved_at": NOW.isoformat(),
        "positions": [
            {
                "account": ACCOUNT,
                "contract": _contract(),
                "quantity": "10.5",
                "avg_cost": "123.45",
            }
        ],
        "balances": [
            {"tag": "NetLiquidation", "currency": "USD", "value": "100000"},
            {"tag": "TotalCashValue", "currency": "USD", "value": "25000.5"},
        ],
        "open_orders": [
            {
                "account": ACCOUNT,
                "broker_order_id": 101,
                "permanent_id": 501,
                "client_id": 7,
                "contract": _contract(),
                "side": "BUY",
                "status": "Submitted",
                "order_type": "LMT",
                "time_in_force": "DAY",
                "total_quantity": "2",
                "filled_quantity": "0",
                "remaining_quantity": "2",
                "limit_price": "120.25",
                "stop_price": None,
                "avg_fill_price": None,
                "last_status_at": NOW.isoformat(),
                "unavailable": {
                    "stop_price": "not supplied for this order type",
                    "avg_fill_price": "order has no fills",
                },
            }
        ],
        "executions": [
            {
                "account": ACCOUNT,
                "execution_id": "0001.01",
                "broker_order_id": 101,
                "permanent_id": 501,
                "client_id": 7,
                "contract": _contract(),
                "side": "BUY",
                "quantity": "1.25",
                "price": "120.25",
                "average_price": "120.25",
                "executed_at": (NOW - timedelta(minutes=1)).isoformat(),
                "execution_exchange": "NASDAQ",
                "commission": None,
                "commission_currency": None,
                "realized_pnl": None,
                "unavailable": {
                    "commission": "not returned by the bounded execution request",
                    "commission_currency": "not returned by the bounded execution request",
                    "realized_pnl": "not returned by the bounded execution request",
                },
            }
        ],
        "execution_scope": {
            "kind": "bounded_execution_filter",
            "start_at": (NOW - timedelta(hours=24, seconds=2)).isoformat(),
            "end_at": NOW.isoformat(),
        },
    }


def test_transport_payload_maps_to_immutable_reconciliation_models():
    snapshot = snapshot_from_transport(_payload(), expected_account=ACCOUNT)

    assert snapshot.account_alias == "***4567"
    assert snapshot.execution_scope.public_dict() == {
        "kind": "bounded_execution_filter",
        "start_at": (NOW - timedelta(hours=24, seconds=2)).isoformat(),
        "end_at": NOW.isoformat(),
    }
    assert snapshot.positions[0].quantity == Decimal("10.5")
    assert snapshot.positions[0].average_cost == Decimal("123.45")
    assert snapshot.balances == {
        "NetLiquidation:USD": Decimal("100000"),
        "TotalCashValue:USD": Decimal("25000.5"),
    }
    assert isinstance(snapshot.balances, MappingProxyType)
    assert snapshot.open_orders[0].order_id == "101"
    assert snapshot.open_orders[0].identity == (7, "101")
    assert snapshot.open_orders[0].permanent_id == "501"
    assert snapshot.open_orders[0].contract.public_dict() == _contract()
    assert snapshot.open_orders[0].auxiliary_price is None
    assert snapshot.open_orders[0].unavailable == {
        "stop_price": "not supplied for this order type",
        "avg_fill_price": "order has no fills",
    }
    assert isinstance(snapshot.open_orders[0].unavailable, MappingProxyType)
    assert snapshot.recent_executions[0].execution_id == "0001.01"
    assert snapshot.recent_executions[0].commission is None
    assert snapshot.recent_executions[0].unavailable == {
        "commission": "not returned by the bounded execution request",
        "commission_currency": "not returned by the bounded execution request",
        "realized_pnl": "not returned by the bounded execution request",
    }

    with pytest.raises(TypeError):
        snapshot.balances["BuyingPower:USD"] = Decimal("1")
    with pytest.raises(TypeError):
        snapshot.open_orders[0].unavailable["stop_price"] = "changed"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(account="DU_OTHER_0000"),
        lambda payload: payload.update(extra="drift"),
        lambda payload: payload["positions"][0].update(extra="drift"),
        lambda payload: payload["balances"].append(payload["balances"][0].copy()),
        lambda payload: payload["execution_scope"].update(kind="unbounded"),
        lambda payload: payload["execution_scope"].update(
            start_at=(NOW - timedelta(days=2)).isoformat()
        ),
        lambda payload: payload["execution_scope"].update(
            start_at=(NOW - timedelta(hours=24, seconds=1)).isoformat()
        ),
        lambda payload: payload["execution_scope"].update(
            end_at=(NOW + timedelta(microseconds=1)).isoformat()
        ),
        lambda payload: payload["executions"][0].update(
            executed_at=(NOW + timedelta(microseconds=1)).isoformat()
        ),
        lambda payload: payload["open_orders"][0].update(total_quantity=2),
    ],
)
def test_adapter_rejects_identity_schema_scope_and_type_drift(mutate):
    payload = _payload()
    mutate(payload)

    with pytest.raises(BrokerEvidenceError):
        snapshot_from_transport(payload, expected_account=ACCOUNT)


@pytest.mark.asyncio
async def test_provider_exposes_only_snapshot_protocol_and_converts_payload():
    transport = SimpleNamespace(
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(),
    )
    provider = IBKRDiagnosticSnapshotProvider(transport, expected_account=ACCOUNT)

    assert_read_only_provider_surface(provider)
    snapshot = await provider.get_broker_snapshot(ACCOUNT, max_age_seconds=30.0)
    assert snapshot.account_alias == "***4567"
    transport.get_broker_snapshot.assert_awaited_once_with(ACCOUNT, max_age_seconds=30.0)

    await provider.close()
    transport.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_provider_close_retries_transient_transport_cleanup_failure():
    transport = SimpleNamespace(
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(side_effect=[RuntimeError("transient cleanup failure"), None]),
    )
    provider = IBKRDiagnosticSnapshotProvider(transport, expected_account=ACCOUNT)

    await provider.close()

    assert transport.stop.await_count == 2


@pytest.mark.asyncio
async def test_provider_rejects_changed_expected_account_without_transport_call():
    transport = SimpleNamespace(
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(),
    )
    provider = IBKRDiagnosticSnapshotProvider(transport, expected_account=ACCOUNT)

    with pytest.raises(BrokerEvidenceError, match="does not match runtime"):
        await provider.get_broker_snapshot("DU_OTHER_0000", max_age_seconds=30.0)

    transport.get_broker_snapshot.assert_not_awaited()


def _runtime():
    return SimpleNamespace(
        diagnostic_connection=SimpleNamespace(
            host="127.0.0.1",
            port=4002,
            client_id=997,
            readonly=True,
        ),
        expected_account_for_provider=ACCOUNT,
    )


@pytest.mark.asyncio
async def test_factory_uses_only_validated_diagnostic_connection_identity():
    transport = SimpleNamespace(
        start=AsyncMock(),
        connect=AsyncMock(return_value=True),
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(),
    )

    provider = await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    transport.start.assert_awaited_once()
    transport.connect.assert_awaited_once_with(
        host="127.0.0.1",
        port=4002,
        client_id=997,
        readonly=True,
        timeout=30.0,
    )
    assert_read_only_provider_surface(provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("connect_result", [False, None])
async def test_factory_fails_closed_and_reaps_unconnected_worker(connect_result):
    transport = SimpleNamespace(
        start=AsyncMock(),
        connect=AsyncMock(return_value=connect_result),
        stop=AsyncMock(),
    )

    with pytest.raises(BrokerEvidenceError, match="initialization failed"):
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    transport.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_reaps_worker_when_connect_raises():
    transport = SimpleNamespace(
        start=AsyncMock(),
        connect=AsyncMock(side_effect=RuntimeError(f"secret {ACCOUNT}")),
        stop=AsyncMock(),
    )

    with pytest.raises(BrokerEvidenceError) as exc_info:
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    assert ACCOUNT not in str(exc_info.value)
    transport.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_connect_failure_retries_transient_debug_log_unlink(monkeypatch, tmp_path):
    transport = SubprocessIBKRClient()
    transport.start = AsyncMock()
    transport.connect = AsyncMock(side_effect=RuntimeError("simulated connect failure"))
    debug_path = tmp_path / "provider_init_unlink_retry.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = debug_path.open("a")
    transport._debug_log_file = debug_file
    transport._debug_log_path = str(debug_path)
    real_unlink = client_module.os.unlink
    unlink_attempts = 0

    def unlink_fails_once(path):
        nonlocal unlink_attempts
        unlink_attempts += 1
        if unlink_attempts == 1:
            raise OSError("transient unlink failure")
        real_unlink(path)

    monkeypatch.setattr(client_module.os, "unlink", unlink_fails_once)

    with pytest.raises(BrokerEvidenceError, match="provider initialization failed"):
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    assert unlink_attempts == 2
    assert debug_file.closed
    assert not debug_path.exists()
    assert transport._debug_log_file is None
    assert transport._debug_log_path is None


class _ProviderCloseFailsOnce:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.attempts = 0

    @property
    def closed(self):
        return self.wrapped.closed

    def close(self):
        self.attempts += 1
        if self.attempts == 1:
            raise OSError("transient close failure")
        self.wrapped.close()


@pytest.mark.asyncio
async def test_factory_connect_failure_retries_transient_debug_log_close(tmp_path):
    transport = SubprocessIBKRClient()
    transport.start = AsyncMock()
    transport.connect = AsyncMock(side_effect=RuntimeError("simulated connect failure"))
    debug_path = tmp_path / "provider_init_close_retry.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = _ProviderCloseFailsOnce(debug_path.open("a"))
    transport._debug_log_file = debug_file
    transport._debug_log_path = str(debug_path)

    with pytest.raises(BrokerEvidenceError, match="provider initialization failed"):
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    assert debug_file.attempts == 2
    assert debug_file.closed
    assert not debug_path.exists()
    assert transport._debug_log_file is None
    assert transport._debug_log_path is None


@pytest.mark.asyncio
async def test_factory_surfaces_persistent_initialization_cleanup_failure():
    transport = SimpleNamespace(
        start=AsyncMock(),
        connect=AsyncMock(side_effect=RuntimeError("simulated connect failure")),
        stop=AsyncMock(side_effect=RuntimeError("persistent cleanup failure")),
    )

    with pytest.raises(BrokerEvidenceError, match="initialization cleanup failed"):
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    assert transport.stop.await_count == 2


@pytest.mark.asyncio
async def test_factory_rejects_missing_account_before_constructing_transport():
    runtime = _runtime()
    runtime.expected_account_for_provider = ""
    constructed = False

    def transport_factory():
        nonlocal constructed
        constructed = True
        raise AssertionError("must not construct transport")

    with pytest.raises(BrokerEvidenceError, match="account is unavailable"):
        await build_diagnostic_provider(runtime, transport_factory=transport_factory)

    assert constructed is False

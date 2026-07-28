import asyncio
import copy
import json
import pickle
from dataclasses import replace
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
    BrokerSnapshotProducerResult,
    IBKRDiagnosticSnapshotProvider,
    assert_producer_owned_broker_snapshot_result,
    build_diagnostic_provider,
    normalized_snapshot_from_transport,
    snapshot_from_transport,
)

ACCOUNT = "DU_TEST_4567"
ACCOUNT_SCOPE = "acct_v1_0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
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
        "snapshot_schema_version": 3,
        "account": ACCOUNT,
        "account_type": "paper",
        "account_structure": "INDIVIDUAL",
        "base_currency": "USD",
        "total_cash": "25000.5",
        "buying_power": "999999",
        "account_observed_at": NOW.isoformat(),
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
            {"tag": "BuyingPower", "currency": "USD", "value": "999999"},
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
        "completed_orders": [],
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
                "commission": "1.23",
                "commission_currency": "USD",
                "realized_pnl": "4.56",
                "unavailable": {},
            }
        ],
        "completeness": {
            "account": True,
            "positions": True,
            "open_orders": True,
            "completed_orders": True,
            "executions": True,
            "commissions": True,
        },
        "collection_evidence": [
            {
                "collection": collection,
                "evidence_id": f"broker-collection-v1-{index:064x}",
                "observed_at": NOW.isoformat(),
                "result_count": count,
                "scope": (
                    {
                        "kind": "ibkr_current_retained_completed_orders",
                        "api_method": "reqCompletedOrders",
                        "api_only": False,
                        "all_clients": True,
                        "request_count": 2,
                        "stability_check": "identical_second_read",
                        "retention_scope": "current_tws_or_gateway_retained_set",
                        "full_history": False,
                        "request_started_at": (
                            NOW - timedelta(seconds=1, microseconds=800000)
                        ).isoformat(),
                        "request_completed_at": (
                            NOW - timedelta(seconds=1, microseconds=700000)
                        ).isoformat(),
                        "verification_started_at": (
                            NOW - timedelta(microseconds=200000)
                        ).isoformat(),
                        "verification_completed_at": (
                            NOW - timedelta(microseconds=100000)
                        ).isoformat(),
                        "broker_time_before": (NOW - timedelta(seconds=2)).isoformat(),
                        "broker_time_after": NOW.isoformat(),
                    }
                    if collection == "completed_orders"
                    else None
                ),
            }
            for index, (collection, count) in enumerate(
                (
                    ("positions", 1),
                    ("open_orders", 1),
                    ("completed_orders", 0),
                    ("executions", 1),
                    ("commissions", 1),
                ),
                start=1,
            )
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
        "BuyingPower:USD": Decimal("999999"),
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
    assert snapshot.recent_executions[0].commission == Decimal("1.23")
    assert snapshot.recent_executions[0].unavailable == {}

    with pytest.raises(TypeError):
        snapshot.balances["BuyingPower:USD"] = Decimal("1")
    with pytest.raises(TypeError):
        snapshot.open_orders[0].unavailable["stop_price"] = "changed"


def test_adapter_accepts_worker_canonical_small_fixed_point_and_rejects_exponent():
    payload = _payload()
    payload["positions"][0]["quantity"] = "0.00000001"

    snapshot = snapshot_from_transport(payload, expected_account=ACCOUNT)
    assert snapshot.positions[0].quantity == Decimal("0.00000001")

    payload["positions"][0]["quantity"] = "1E-8"
    with pytest.raises(BrokerEvidenceError):
        snapshot_from_transport(payload, expected_account=ACCOUNT)


def test_complete_transport_maps_to_normalized_snapshot():
    snapshot = normalized_snapshot_from_transport(
        _payload(),
        expected_account=ACCOUNT,
        account_scope=ACCOUNT_SCOPE,
        max_age_seconds=30.0,
        now=NOW + timedelta(seconds=1),
    )
    assert snapshot.account.account_type == "paper"
    assert snapshot.account.base_currency == "USD"
    assert snapshot.account.total_cash == Decimal("25000.5")
    assert snapshot.account.buying_power == Decimal("999999")
    assert snapshot.completeness.complete is True
    assert snapshot.completed_orders == ()
    assert snapshot.executions[0].commission == Decimal("1.23")
    assert ACCOUNT not in snapshot.canonical_payload()


def test_zero_broker_collections_require_positive_zero_count_evidence():
    payload = _payload()
    payload["positions"] = []
    payload["open_orders"] = []
    for evidence in payload["collection_evidence"]:
        if evidence["collection"] in {"positions", "open_orders"}:
            evidence["result_count"] = 0

    snapshot = normalized_snapshot_from_transport(
        payload,
        expected_account=ACCOUNT,
        account_scope=ACCOUNT_SCOPE,
        max_age_seconds=30.0,
        now=NOW + timedelta(seconds=1),
    )
    assert snapshot.positions == ()
    assert snapshot.open_orders == ()

    payload["collection_evidence"] = [
        evidence
        for evidence in payload["collection_evidence"]
        if evidence["collection"] != "positions"
    ]
    with pytest.raises(BrokerEvidenceError, match="incomplete"):
        normalized_snapshot_from_transport(
            payload,
            expected_account=ACCOUNT,
            account_scope=ACCOUNT_SCOPE,
            max_age_seconds=30.0,
            now=NOW + timedelta(seconds=1),
        )


def test_completed_order_transport_maps_to_completed_domain_collection():
    payload = _payload()
    completed = payload["open_orders"][0].copy()
    completed.update(
        broker_order_id=202,
        permanent_id=602,
        status="Filled",
        filled_quantity="2",
        remaining_quantity="0",
        avg_fill_price="120.25",
        unavailable={"stop_price": "not supplied for this order type"},
    )
    payload["completed_orders"] = [completed]
    for evidence in payload["collection_evidence"]:
        if evidence["collection"] == "completed_orders":
            evidence["result_count"] = 1

    snapshot = normalized_snapshot_from_transport(
        payload,
        expected_account=ACCOUNT,
        account_scope=ACCOUNT_SCOPE,
        max_age_seconds=30.0,
        now=NOW + timedelta(seconds=1),
    )

    assert len(snapshot.completed_orders) == 1
    assert snapshot.completed_orders[0].broker_order_id == 202
    assert snapshot.completed_orders[0].status == "Filled"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload["completeness"].update(commissions=False),
        lambda payload: payload["executions"][0].update(commission=None),
        lambda payload: payload["collection_evidence"][0].update(result_count=0),
        lambda payload: payload.update(account="DU_OTHER_0000"),
    ],
)
def test_normalized_mapper_rejects_partial_or_wrong_account_evidence(mutate):
    payload = _payload()
    mutate(payload)

    with pytest.raises(BrokerEvidenceError):
        normalized_snapshot_from_transport(
            payload,
            expected_account=ACCOUNT,
            account_scope=ACCOUNT_SCOPE,
            max_age_seconds=30.0,
            now=NOW + timedelta(seconds=1),
        )


def test_normalized_mapper_rejects_stale_transport_evidence():
    with pytest.raises(BrokerEvidenceError, match="stale"):
        normalized_snapshot_from_transport(
            _payload(),
            expected_account=ACCOUNT,
            account_scope=ACCOUNT_SCOPE,
            max_age_seconds=30.0,
            now=NOW + timedelta(minutes=5),
        )


@pytest.mark.asyncio
async def test_provider_exposes_typed_unsigned_normalized_producer_result():
    transport = SimpleNamespace(
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(),
    )
    provider = IBKRDiagnosticSnapshotProvider(
        transport,
        expected_account=ACCOUNT,
        account_scope=ACCOUNT_SCOPE,
    )

    result = await provider.produce_normalized_snapshot(max_age_seconds=30.0)

    assert type(result) is BrokerSnapshotProducerResult
    assert result.snapshot.account.account_scope == ACCOUNT_SCOPE
    assert result.purpose == "bootstrap-broker-signing-v1"
    canonical = json.loads(result.canonical_payload)
    assert canonical["purpose"] == "bootstrap-broker-signing-v1"
    assert canonical["completed_order_collection_scope"]["all_clients"] is True
    assert canonical["completed_order_collection_scope"]["request_count"] == 2
    assert canonical["completed_order_collection_scope"]["full_history"] is False
    assert ACCOUNT not in result.canonical_payload
    assert assert_producer_owned_broker_snapshot_result(result) is result
    with pytest.raises(BrokerEvidenceError, match="already consumed"):
        assert_producer_owned_broker_snapshot_result(result)
    transport.get_broker_snapshot.assert_awaited_once_with(
        ACCOUNT,
        max_age_seconds=30.0,
    )


@pytest.mark.asyncio
async def test_producer_result_rejects_construction_copy_replace_pickle_and_replay():
    transport = SimpleNamespace(
        get_broker_snapshot=AsyncMock(return_value=_payload()),
        stop=AsyncMock(),
    )
    provider = IBKRDiagnosticSnapshotProvider(
        transport,
        expected_account=ACCOUNT,
        account_scope=ACCOUNT_SCOPE,
    )
    result = await provider.produce_normalized_snapshot(max_age_seconds=30.0)

    with pytest.raises(TypeError):
        BrokerSnapshotProducerResult(snapshot=result.snapshot)
    with pytest.raises(TypeError):
        copy.copy(result)
    with pytest.raises(TypeError):
        copy.deepcopy(result)
    with pytest.raises(BrokerEvidenceError, match="already consumed"):
        replace(result)
    with pytest.raises(TypeError):
        pickle.dumps(result)

    assert assert_producer_owned_broker_snapshot_result(result) is result
    with pytest.raises(BrokerEvidenceError, match="already consumed"):
        assert_producer_owned_broker_snapshot_result(result)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda scope: scope.update(api_only=True),
        lambda scope: scope.update(all_clients=False),
        lambda scope: scope.update(full_history=True),
        lambda scope: scope.update(retention_scope="full_history"),
        lambda scope: scope.update(api_method="reqOpenOrders"),
        lambda scope: scope.update(request_count=1),
        lambda scope: scope.update(stability_check="none"),
        lambda scope: scope.update(verification_started_at=scope["request_started_at"]),
        lambda scope: scope.pop("broker_time_before"),
    ],
)
def test_completed_order_scope_rejects_overclaim_and_bound_tampering(mutate):
    payload = _payload()
    scope = next(
        evidence["scope"]
        for evidence in payload["collection_evidence"]
        if evidence["collection"] == "completed_orders"
    )
    mutate(scope)

    with pytest.raises(BrokerEvidenceError):
        normalized_snapshot_from_transport(
            payload,
            expected_account=ACCOUNT,
            account_scope=ACCOUNT_SCOPE,
            max_age_seconds=30.0,
            now=NOW + timedelta(seconds=1),
        )


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
        runtime_contract=SimpleNamespace(safety_account_scope=ACCOUNT_SCOPE),
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
@pytest.mark.parametrize("cancel_phase", ["start", "connect"])
async def test_factory_cancellation_reaps_worker_and_propagates(cancel_phase):
    transport = SimpleNamespace(
        start=AsyncMock(),
        connect=AsyncMock(return_value=True),
        stop=AsyncMock(),
    )
    getattr(transport, cancel_phase).side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)

    transport.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_factory_cleanup_is_shielded_from_repeated_cancellation():
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def stop():
        cleanup_started.set()
        await release_cleanup.wait()
        cleanup_finished.set()

    transport = SimpleNamespace(
        start=AsyncMock(side_effect=asyncio.CancelledError),
        connect=AsyncMock(return_value=True),
        stop=AsyncMock(side_effect=stop),
    )
    task = asyncio.create_task(
        build_diagnostic_provider(_runtime(), transport_factory=lambda: transport)
    )
    await cleanup_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert cleanup_finished.is_set()
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

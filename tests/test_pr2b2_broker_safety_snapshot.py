import asyncio
import copy
import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from robo_trader.broker_safety_evidence import (
    BrokerContractSafetySnapshot,
    BrokerSafetyContract,
    BrokerSafetyPosition,
    BrokerSafetySnapshot,
    _BrokerContractSnapshotCapability,
    _BrokerSnapshotCapability,
    _issue_broker_contract_snapshot_capability,
    _issue_broker_snapshot_capability,
    _produce_broker_contract_safety_snapshot,
    _produce_broker_safety_snapshot,
    assert_producer_owned_broker_contract_safety_snapshot,
    assert_producer_owned_broker_safety_snapshot,
)
from robo_trader.clients import ibkr_subprocess_worker as worker
from robo_trader.clients.subprocess_ibkr_client import (
    BrokerSnapshotAccountMismatchError,
    IBKRTransportPoisonedError,
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.config import RuntimeContract, _derive_safety_account_scope
from robo_trader.reconciliation.identity import validate_runtime_safety
from robo_trader.safety.models import ValidationError

ACCOUNT = "DU1234567"
SAFETY_SCOPE_KEY = "0123456789abcdef" * 4
ACCOUNT_SCOPE = _derive_safety_account_scope(SAFETY_SCOPE_KEY, ACCOUNT)
NOW = datetime.now(timezone.utc)


def _contract(symbol="AAPL", con_id=265598):
    return SimpleNamespace(
        conId=con_id,
        symbol=symbol,
        localSymbol=symbol,
        secType="STK",
        currency="USD",
        exchange="SMART",
        primaryExchange="NASDAQ",
        tradingClass="NMS",
    )


class _FakeIB:
    def __init__(self):
        self.calls = []
        self.connected = True
        self.account_reads = 0
        self.contract = _contract()

    def isConnected(self):
        self.calls.append("isConnected")
        return self.connected

    def managedAccounts(self):
        self.calls.append("managedAccounts")
        self.account_reads += 1
        return [ACCOUNT]

    async def reqCurrentTimeAsync(self):
        self.calls.append("reqCurrentTimeAsync")
        return NOW + timedelta(milliseconds=len(self.calls))

    async def reqPositionsAsync(self):
        self.calls.append("reqPositionsAsync")
        return [
            SimpleNamespace(
                account=ACCOUNT,
                contract=self.contract,
                position="10.5000",
                avgCost="123.4500",
            )
        ]

    async def reqAllOpenOrdersAsync(self):
        self.calls.append("reqAllOpenOrdersAsync")
        order = SimpleNamespace(
            account=ACCOUNT,
            orderId=101,
            permId=501,
            clientId=0,
            action="SELL",
            orderType="STP",
            tif="GTC",
            totalQuantity="2",
            lmtPrice="0",
            auxPrice="115.25",
        )
        status = SimpleNamespace(
            status="Submitted",
            filled="0",
            remaining="2",
            avgFillPrice="0",
        )
        return [
            SimpleNamespace(
                order=order,
                orderStatus=status,
                contract=self.contract,
                log=[SimpleNamespace(time=NOW)],
            )
        ]

    async def qualifyContractsAsync(self, contract):
        self.calls.append("qualifyContractsAsync")
        if getattr(contract, "conId", 0) > 0:
            return [contract]
        symbol = str(getattr(contract, "symbol", "")).strip().upper()
        return [_contract(symbol, 265598 if symbol == "AAPL" else 272093)]


@pytest.fixture
def fake_ib(monkeypatch):
    fake = _FakeIB()
    monkeypatch.setattr(worker, "ib", fake)
    monkeypatch.setattr(
        worker,
        "worker_connection_identity",
        ("127.0.0.1", 4002, 7, True),
    )
    return fake


async def _worker_payload(fake_ib, symbol="AAPL"):
    result = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": symbol}
    )
    assert result["status"] == "success"
    return result["data"]


async def _worker_contract_payload(fake_ib, symbol="MSFT"):
    result = await worker.handle_get_broker_contract_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": symbol}
    )
    assert result["status"] == "success"
    return result["data"]


def _connected_client(monkeypatch, payload):
    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        generation_id="snapshot-generation",
        process=SimpleNamespace(poll=lambda: None),
    )
    client.process = generation.process
    client._generation = generation
    with client._connection_state_lock:
        client._connected = True
        client._connection_identity = ("127.0.0.1", 4002, 7, True)
        client._connection_generation_id = generation.generation_id
    execute = AsyncMock(return_value=payload)
    monkeypatch.setattr(client, "_execute_command_unlocked", execute)
    return client, generation, execute


def _runtime_context(
    tmp_path,
    monkeypatch,
    *,
    account=ACCOUNT,
    account_scope=None,
    host="127.0.0.1",
):
    ibc_path = tmp_path / "config" / "ibc" / "config.ini"
    ibc_path.parent.mkdir(parents=True, exist_ok=True)
    ibc_path.write_text("ReadOnlyApi=yes\nTradingMode=paper\n")
    if account_scope is None:
        account_scope = _derive_safety_account_scope(SAFETY_SCOPE_KEY, account)
    contract = RuntimeContract(
        environment="dev",
        execution_mode="paper",
        execution_source="paper_simulator",
        ibkr_host=host,
        ibkr_port=4002,
        ibkr_readonly=True,
        database_path=str(tmp_path / "paper.db"),
        account_alias="***" + account[-4:],
        account_type="paper",
        model_artifact_set="tests",
        build_id="tests",
        state_namespace="paper",
        safety_account_scope=account_scope,
        safety_execution_domain_scope="paper-safety-v1",
        safety_journal_path=str(tmp_path / "safety.db"),
    )
    monkeypatch.setattr(
        "robo_trader.config.load_runtime_contract_from_env",
        lambda _environment: contract,
    )
    context = validate_runtime_safety(
        tmp_path,
        {
            "IBKR_ACCOUNT": account,
            "IBKR_CLIENT_ID": "1",
            "IBKR_RECONCILIATION_CLIENT_ID": "7",
            "SAFETY_ACCOUNT_SCOPE_KEY": SAFETY_SCOPE_KEY,
        },
    )
    return context, ibc_path


@pytest.mark.asyncio
async def test_worker_safety_snapshot_is_stable_all_client_and_read_only(fake_ib):
    payload = await _worker_payload(fake_ib)

    assert payload["safety_snapshot_schema_version"] == 1
    assert payload["requested_symbol"] == "AAPL"
    assert payload["positions"][0]["quantity"] == "10.5"
    assert payload["open_orders"][0]["side"] == "SELL"
    assert payload["positions_complete"] is True
    assert payload["open_orders_complete"] is True
    assert payload["open_orders_all_clients"] is True
    assert payload["open_orders_stable"] is True
    assert payload["unknown_order_count"] == 0
    assert fake_ib.calls.count("reqPositionsAsync") == 2
    assert fake_ib.calls.count("reqAllOpenOrdersAsync") == 2
    assert not any(
        token in call.lower()
        for call in fake_ib.calls
        for token in ("placeorder", "cancelorder", "exercise", "accountsummary")
    )


@pytest.mark.asyncio
async def test_worker_safety_snapshot_fails_before_reads_for_wrong_or_multiple_account(
    fake_ib, monkeypatch
):
    wrong = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": "DU_WRONG", "requested_symbol": "AAPL"}
    )
    assert wrong["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert not any(call.startswith("req") for call in fake_ib.calls)

    fake_ib.calls.clear()
    live = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": "U1234567", "requested_symbol": "AAPL"}
    )
    assert live["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert not any(call.startswith("req") for call in fake_ib.calls)

    fake_ib.calls.clear()
    monkeypatch.setattr(fake_ib, "managedAccounts", lambda: [ACCOUNT, "DU_SECOND"])
    multiple = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "AAPL"}
    )
    assert multiple["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert not any(call.startswith("req") for call in fake_ib.calls)


@pytest.mark.asyncio
async def test_worker_safety_snapshot_rejects_remote_connection_before_broker_reads(
    fake_ib, monkeypatch
):
    monkeypatch.setattr(
        worker,
        "worker_connection_identity",
        ("192.0.2.10", 4002, 7, True),
    )

    result = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "AAPL"}
    )

    assert result["error_type"] == "ConnectionError"
    assert fake_ib.calls == ["isConnected"]


@pytest.mark.asyncio
async def test_worker_safety_snapshot_fails_closed_for_missing_or_ambiguous_symbol(
    fake_ib, monkeypatch
):
    missing = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "MSFT"}
    )
    assert missing["status"] == worker.PROTOCOL_ERROR_STATUS

    async def duplicate_positions():
        return [
            SimpleNamespace(
                account=ACCOUNT,
                contract=_contract("AAPL", 265598),
                position="10",
                avgCost="100",
            ),
            SimpleNamespace(
                account=ACCOUNT,
                contract=_contract("AAPL", 999999),
                position="1",
                avgCost="101",
            ),
        ]

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", duplicate_positions)
    duplicate = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "AAPL"}
    )
    assert duplicate["status"] == worker.PROTOCOL_ERROR_STATUS


@pytest.mark.asyncio
async def test_worker_safety_snapshot_fails_closed_on_state_change_or_unknown_order(
    fake_ib, monkeypatch
):
    position_reads = 0

    async def changing_positions():
        nonlocal position_reads
        position_reads += 1
        return [
            SimpleNamespace(
                account=ACCOUNT,
                contract=fake_ib.contract,
                position="10" if position_reads == 1 else "9",
                avgCost="123.45",
            )
        ]

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", changing_positions)
    changed = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "AAPL"}
    )
    assert changed["status"] == worker.PROTOCOL_ERROR_STATUS

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", _FakeIB().reqPositionsAsync)
    original_orders = fake_ib.reqAllOpenOrdersAsync

    async def unknown_order():
        trades = await original_orders()
        trades[0].order.action = "UNKNOWN"
        return trades

    monkeypatch.setattr(fake_ib, "reqAllOpenOrdersAsync", unknown_order)
    unknown = await worker.handle_get_broker_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "AAPL"}
    )
    assert unknown["status"] == worker.PROTOCOL_ERROR_STATUS


@pytest.mark.asyncio
async def test_client_produces_current_generation_immutable_snapshot_without_history_cache(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    snapshot = await client.get_broker_safety_snapshot(
        runtime_context,
        "aapl",
    )

    assert type(snapshot) is BrokerSafetySnapshot
    assert snapshot.transport_generation == generation.generation_id
    assert snapshot.account_scope == ACCOUNT_SCOPE
    assert snapshot.broker_host == "127.0.0.1"
    assert snapshot.broker_port == 4002
    assert snapshot.broker_client_id == 7
    assert snapshot.read_only is True
    assert snapshot.ibc_proof_source == "validated-ibc-readonly-paper-v1"
    assert snapshot.ibc_proof_id.startswith("ibc-proof-v1-")
    assert runtime_context.ibc_config_hash not in repr(snapshot)
    assert ACCOUNT not in repr(snapshot)
    assert hashlib.sha256(ACCOUNT.encode()).hexdigest() not in repr(snapshot)
    assert snapshot.requested_contract.con_id == 265598
    assert snapshot.positions[0].quantity.as_tuple().exponent == -1
    assert snapshot.open_orders_all_clients is True
    assert client._historical_lineage_by_symbol == {}
    assert_producer_owned_broker_safety_snapshot(snapshot)
    with pytest.raises(FrozenInstanceError):
        snapshot.requested_symbol = "MSFT"
    with pytest.raises(ValidationError, match="not producer-owned"):
        assert_producer_owned_broker_safety_snapshot(replace(snapshot))
    execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("positions_complete", False),
        ("open_orders_complete", False),
        ("open_orders_all_clients", False),
        ("open_orders_stable", False),
        ("unknown_order_count", 1),
    ],
)
async def test_client_rejects_incomplete_or_unknown_order_evidence(
    fake_ib, monkeypatch, tmp_path, field, value
):
    payload = copy.deepcopy(await _worker_payload(fake_ib))
    payload[field] = value
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    with pytest.raises(IBKRTransportPoisonedError):
        await client.get_broker_safety_snapshot(
            runtime_context,
            "AAPL",
        )
    assert generation.poisoned_reason is not None

    duplicate_payload = copy.deepcopy(await _worker_payload(fake_ib))
    duplicate_payload["positions"].append(copy.deepcopy(duplicate_payload["positions"][0]))
    client, generation, _ = _connected_client(monkeypatch, duplicate_payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    with pytest.raises(IBKRTransportPoisonedError, match="duplicated"):
        await client.get_broker_safety_snapshot(runtime_context, "AAPL")
    assert generation.poisoned_reason is not None

    unknown_order_payload = copy.deepcopy(await _worker_payload(fake_ib))
    unknown_order_payload["open_orders"][0]["order_type"] = "MYSTERY"
    client, generation, _ = _connected_client(monkeypatch, unknown_order_payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    with pytest.raises(IBKRTransportPoisonedError, match="unsupported"):
        await client.get_broker_safety_snapshot(runtime_context, "AAPL")
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_client_fails_closed_on_account_or_generation_change(fake_ib, monkeypatch, tmp_path):
    payload = copy.deepcopy(await _worker_payload(fake_ib))
    payload["account"] = "DU_WRONG"
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    with pytest.raises(BrokerSnapshotAccountMismatchError) as exc_info:
        await client.get_broker_safety_snapshot(
            runtime_context,
            "AAPL",
        )
    assert ACCOUNT not in str(exc_info.value)
    assert "DU_WRONG" not in str(exc_info.value)
    assert generation.poisoned_reason is not None

    payload = await _worker_payload(fake_ib)
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    async def swap_generation(_command, timeout):
        client._generation = _WorkerGeneration(
            generation_id="replacement-generation",
            process=SimpleNamespace(poll=lambda: None),
        )
        return payload

    monkeypatch.setattr(client, "_execute_command_unlocked", swap_generation)
    with pytest.raises(IBKRTransportPoisonedError, match="generation changed"):
        await client.get_broker_safety_snapshot(
            runtime_context,
            "AAPL",
        )
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_client_rejects_stale_or_duplicate_position_evidence(fake_ib, monkeypatch, tmp_path):
    stale_payload = copy.deepcopy(await _worker_payload(fake_ib))
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=31)
    stale_payload["retrieved_at"] = stale_time.isoformat()
    stale_payload["broker_time_before"] = stale_time.isoformat()
    stale_payload["broker_time_after"] = stale_time.isoformat()
    client, generation, _ = _connected_client(monkeypatch, stale_payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    with pytest.raises(IBKRTransportPoisonedError, match="freshness"):
        await client.get_broker_safety_snapshot(
            runtime_context,
            "AAPL",
        )
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_client_rejects_same_account_scope_relabel(fake_ib, monkeypatch, tmp_path):
    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    relabeled_contract = replace(
        runtime_context.runtime_contract,
        safety_account_scope="acct_v1_" + ("fedcba9876543210" * 4),
    )
    relabeled_context = replace(runtime_context, runtime_contract=relabeled_contract)

    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_safety_snapshot(relabeled_context, "AAPL")

    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_rejects_live_account_and_remote_host(fake_ib, monkeypatch, tmp_path):
    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    live_context, _ = _runtime_context(
        tmp_path / "live",
        monkeypatch,
        account="U1234567",
    )
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context") as exc_info:
        await client.get_broker_safety_snapshot(live_context, "AAPL")
    assert "U1234567" not in str(exc_info.value)
    execute.assert_not_awaited()

    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path / "remote", monkeypatch)
    object.__setattr__(runtime_context.diagnostic_connection, "host", "192.0.2.10")
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_safety_snapshot(runtime_context, "AAPL")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()

    payload = await _worker_payload(fake_ib)
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, ibc_path = _runtime_context(tmp_path / "during", monkeypatch)

    async def change_ibc_during_snapshot(_command, timeout):
        ibc_path.write_text("ReadOnlyApi=no\nTradingMode=paper\n")
        return payload

    monkeypatch.setattr(
        client,
        "_execute_command_unlocked",
        change_ibc_during_snapshot,
    )
    with pytest.raises(IBKRTransportPoisonedError, match="changed during snapshot"):
        await client.get_broker_safety_snapshot(runtime_context, "AAPL")
    assert generation.poisoned_reason is not None

    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_safety_snapshot(
            SimpleNamespace(
                diagnostic_connection=SimpleNamespace(
                    host="127.0.0.1",
                    port=4002,
                    client_id=7,
                    readonly=True,
                )
            ),
            "AAPL",
        )
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_rejects_changed_ibc_and_readonly_request_only(fake_ib, monkeypatch, tmp_path):
    payload = await _worker_payload(fake_ib)
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, ibc_path = _runtime_context(tmp_path / "changed", monkeypatch)
    ibc_path.write_text("ReadOnlyApi=no\nTradingMode=paper\n")

    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_safety_snapshot(runtime_context, "AAPL")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()


def _factory_evidence():
    contract = BrokerSafetyContract(
        con_id=265598,
        symbol="AAPL",
        local_symbol="AAPL",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
    )
    return contract, (BrokerSafetyPosition(contract=contract, quantity=Decimal("10")),)


def test_snapshot_factory_requires_exact_one_shot_registered_capability(tmp_path, monkeypatch):
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    capability = _issue_broker_snapshot_capability(
        runtime_context,
        connection_identity=("127.0.0.1", 4002, 7, True),
        transport_generation="generation-1",
        requested_symbol="AAPL",
    )
    contract, positions = _factory_evidence()
    factory_args = {
        "observed_at": NOW,
        "broker_time_before": NOW,
        "broker_time_after": NOW,
        "snapshot_id": "broker-safety-v1-test",
        "source": "ibkr-subprocess-safety-v1",
        "requested_contract": contract,
        "positions": positions,
        "open_orders": (),
    }

    snapshot = _produce_broker_safety_snapshot(capability=capability, **factory_args)
    assert_producer_owned_broker_safety_snapshot(snapshot)
    with pytest.raises(ValidationError, match="already consumed"):
        _produce_broker_safety_snapshot(capability=capability, **factory_args)

    forged = _BrokerSnapshotCapability(
        account_scope=ACCOUNT_SCOPE,
        requested_symbol="AAPL",
        runtime_fingerprint=runtime_context.runtime_contract.fingerprint,
        broker_host="127.0.0.1",
        broker_port=4002,
        broker_client_id=7,
        read_only=True,
        transport_generation="generation-1",
        _expected_account=ACCOUNT,
        _ibc_config_hash=runtime_context.ibc_config_hash,
        _producer_marker=object(),
    )
    with pytest.raises(ValidationError, match="absent"):
        _produce_broker_safety_snapshot(capability=forged, **factory_args)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _produce_broker_safety_snapshot(
            capability=forged,
            account_scope=ACCOUNT_SCOPE,
            **factory_args,
        )


@pytest.mark.asyncio
async def test_lifecycle_callback_holds_lock_until_stop_can_run(fake_ib, monkeypatch, tmp_path):
    payload = await _worker_payload(fake_ib)
    client, _, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    stop_called = asyncio.Event()

    async def callback(snapshot):
        assert_producer_owned_broker_safety_snapshot(snapshot)
        entered.set()
        await release.wait()
        return snapshot.snapshot_id

    async def fake_stop_unlocked():
        stop_called.set()

    monkeypatch.setattr(client, "_stop_unlocked", fake_stop_unlocked)
    snapshot_task = asyncio.create_task(
        client.run_with_locked_broker_safety_snapshot(runtime_context, "AAPL", callback)
    )
    await entered.wait()
    stop_task = asyncio.create_task(client.stop())
    await asyncio.sleep(0)
    assert not stop_called.is_set()

    release.set()
    assert (await snapshot_task).startswith("broker-safety-v1-")
    await stop_task
    assert stop_called.is_set()


@pytest.mark.asyncio
async def test_lifecycle_callback_holds_lock_until_reconnect_can_run(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, _, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def callback(snapshot):
        assert_producer_owned_broker_safety_snapshot(snapshot)
        entered.set()
        await release.wait()
        return snapshot.snapshot_id

    monkeypatch.setattr(client, "_check_zombie_connections", AsyncMock(return_value=(0, "")))
    monkeypatch.setattr(client, "_accept_ping_response", lambda _data, _generation: True)
    snapshot_task = asyncio.create_task(
        client.run_with_locked_broker_safety_snapshot(runtime_context, "AAPL", callback)
    )
    await entered.wait()
    reconnect_task = asyncio.create_task(
        client.connect(host="127.0.0.1", port=4002, client_id=7, readonly=True)
    )
    await asyncio.sleep(0)
    assert not reconnect_task.done()
    assert execute.await_count == 1

    release.set()
    assert (await snapshot_task).startswith("broker-safety-v1-")
    assert await reconnect_task is True
    assert execute.await_count == 2


@pytest.mark.asyncio
async def test_lifecycle_callback_generation_drift_poisoned_before_return(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    async def replace_generation(snapshot):
        assert_producer_owned_broker_safety_snapshot(snapshot)
        client._generation = _WorkerGeneration(
            generation_id="replacement-generation",
            process=SimpleNamespace(poll=lambda: None),
        )
        return snapshot

    with pytest.raises(IBKRTransportPoisonedError, match="generation changed"):
        await client.run_with_locked_broker_safety_snapshot(
            runtime_context,
            "AAPL",
            replace_generation,
        )
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_worker_contract_snapshot_qualifies_unheld_symbol_without_account_state_reads(
    fake_ib,
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")

    assert payload["requested_symbol"] == "MSFT"
    assert payload["qualified_contract"]["symbol"] == "MSFT"
    assert payload["qualified_contract"]["con_id"] == 272093
    assert set(payload) == {
        "contract_safety_snapshot_schema_version",
        "account",
        "requested_symbol",
        "broker_time_before",
        "broker_time_after",
        "retrieved_at",
        "qualified_contract",
    }
    assert fake_ib.calls.count("qualifyContractsAsync") == 2
    assert fake_ib.calls.count("reqCurrentTimeAsync") == 2
    assert fake_ib.calls.count("managedAccounts") == 2
    assert not any(
        token in call
        for call in fake_ib.calls
        for token in (
            "reqPositions",
            "reqAllOpenOrders",
            "reqAccountSummary",
            "reqExecutions",
            "placeOrder",
            "cancelOrder",
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "ambiguous", "changed"])
async def test_worker_contract_snapshot_rejects_unstable_qualification(
    fake_ib, monkeypatch, failure
):
    calls = 0

    async def unsafe_qualification(contract):
        nonlocal calls
        calls += 1
        fake_ib.calls.append("qualifyContractsAsync")
        if failure == "missing":
            return []
        if failure == "ambiguous":
            return [_contract("MSFT", 272093), _contract("MSFT", 272094)]
        return [_contract("MSFT", 272093 if calls == 1 else 272094)]

    monkeypatch.setattr(fake_ib, "qualifyContractsAsync", unsafe_qualification)
    result = await worker.handle_get_broker_contract_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "MSFT"}
    )

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert not any(
        token in call
        for call in fake_ib.calls
        for token in ("reqPositions", "reqAllOpenOrders", "placeOrder", "cancelOrder")
    )


@pytest.mark.asyncio
async def test_worker_contract_snapshot_rejects_account_live_remote_and_non_readonly_before_proof(
    fake_ib, monkeypatch
):
    wrong = await worker.handle_get_broker_contract_safety_snapshot(
        {"expected_account": "DU7654321", "requested_symbol": "MSFT"}
    )
    assert wrong["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert "qualifyContractsAsync" not in fake_ib.calls

    fake_ib.calls.clear()
    live = await worker.handle_get_broker_contract_safety_snapshot(
        {"expected_account": "U1234567", "requested_symbol": "MSFT"}
    )
    assert live["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert fake_ib.calls == ["isConnected"]

    fake_ib.calls.clear()
    monkeypatch.setattr(
        worker,
        "worker_connection_identity",
        ("192.0.2.10", 4002, 7, True),
    )
    remote = await worker.handle_get_broker_contract_safety_snapshot(
        {"expected_account": ACCOUNT, "requested_symbol": "MSFT"}
    )
    assert remote["error_type"] == "ConnectionError"
    assert fake_ib.calls == []

    monkeypatch.setattr(
        worker,
        "worker_connection_identity",
        ("127.0.0.1", 4002, 7, False),
    )
    request_only_readonly = await worker.handle_get_broker_contract_safety_snapshot(
        {
            "expected_account": ACCOUNT,
            "requested_symbol": "MSFT",
            "readonly": True,
        }
    )
    assert request_only_readonly["error_type"] == "ConnectionError"
    assert fake_ib.calls == []


@pytest.mark.asyncio
async def test_client_contract_snapshot_succeeds_for_unheld_symbol_and_has_no_account_state(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    snapshot = await client.get_broker_contract_safety_snapshot(runtime_context, "msft")

    assert type(snapshot) is BrokerContractSafetySnapshot
    assert snapshot.qualified_contract.symbol == "MSFT"
    assert snapshot.qualified_contract.con_id == 272093
    assert snapshot.transport_generation == generation.generation_id
    assert snapshot.account_scope == ACCOUNT_SCOPE
    assert snapshot.broker_host == "127.0.0.1"
    assert snapshot.broker_port == 4002
    assert snapshot.read_only is True
    assert runtime_context.ibc_config_hash not in repr(snapshot)
    assert ACCOUNT not in repr(snapshot)
    assert {
        "positions",
        "balances",
        "executions",
        "open_orders",
    }.isdisjoint({item.name for item in fields(snapshot)})
    assert_producer_owned_broker_contract_safety_snapshot(snapshot)
    with pytest.raises(ValidationError, match="not producer-owned"):
        assert_producer_owned_broker_contract_safety_snapshot(replace(snapshot))
    execute.assert_awaited_once()


def _contract_factory_args():
    contract = BrokerSafetyContract(
        con_id=272093,
        symbol="MSFT",
        local_symbol="MSFT",
        security_type="STK",
        currency="USD",
        exchange="SMART",
        primary_exchange="NASDAQ",
        trading_class="NMS",
    )
    return {
        "broker_time_before": NOW,
        "broker_time_after": NOW,
        "retrieved_at": NOW,
        "snapshot_id": "broker-contract-safety-v1-test",
        "source": "ibkr-subprocess-contract-safety-v1",
        "qualified_contract": contract,
    }


def test_contract_snapshot_factory_rejects_forged_copied_and_replayed_capability(
    tmp_path, monkeypatch
):
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    def issue():
        return _issue_broker_contract_snapshot_capability(
            runtime_context,
            connection_identity=("127.0.0.1", 4002, 7, True),
            transport_generation="generation-1",
            requested_symbol="MSFT",
        )

    capability = issue()
    copied = replace(capability)
    with pytest.raises(ValidationError, match="absent"):
        _produce_broker_contract_safety_snapshot(
            capability=copied,
            **_contract_factory_args(),
        )

    forged = _BrokerContractSnapshotCapability(
        account_scope=ACCOUNT_SCOPE,
        requested_symbol="MSFT",
        runtime_fingerprint=runtime_context.runtime_contract.fingerprint,
        broker_host="127.0.0.1",
        broker_port=4002,
        broker_client_id=7,
        read_only=True,
        transport_generation="generation-1",
        _expected_account=ACCOUNT,
        _ibc_config_hash=runtime_context.ibc_config_hash,
        _producer_marker=object(),
    )
    with pytest.raises(ValidationError, match="absent"):
        _produce_broker_contract_safety_snapshot(
            capability=forged,
            **_contract_factory_args(),
        )

    mutated = issue()
    object.__setattr__(mutated, "transport_generation", "forged-generation")
    with pytest.raises(ValidationError, match="changed after issuance"):
        _produce_broker_contract_safety_snapshot(
            capability=mutated,
            **_contract_factory_args(),
        )

    snapshot = _produce_broker_contract_safety_snapshot(
        capability=capability,
        **_contract_factory_args(),
    )
    assert_producer_owned_broker_contract_safety_snapshot(snapshot)
    with pytest.raises(ValidationError, match="already consumed"):
        _produce_broker_contract_safety_snapshot(
            capability=capability,
            **_contract_factory_args(),
        )
    with pytest.raises(ValidationError, match="not producer-owned"):
        assert_producer_owned_broker_contract_safety_snapshot(copy.copy(snapshot))
    with pytest.raises(ValidationError, match="producer boundary"):
        replace(snapshot, _producer_marker=object())


@pytest.mark.asyncio
async def test_client_contract_snapshot_rejects_account_generation_context_and_ibc_drift(
    fake_ib, monkeypatch, tmp_path
):
    payload = copy.deepcopy(await _worker_contract_payload(fake_ib, "MSFT"))
    payload["account"] = "DU7654321"
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path / "account", monkeypatch)
    with pytest.raises(BrokerSnapshotAccountMismatchError):
        await client.get_broker_contract_safety_snapshot(runtime_context, "MSFT")
    assert generation.poisoned_reason is not None

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path / "generation", monkeypatch)

    async def swap_generation(_command, timeout):
        client._generation = _WorkerGeneration(
            generation_id="replacement-generation",
            process=SimpleNamespace(poll=lambda: None),
        )
        return payload

    monkeypatch.setattr(client, "_execute_command_unlocked", swap_generation)
    with pytest.raises(IBKRTransportPoisonedError, match="generation changed"):
        await client.get_broker_contract_safety_snapshot(runtime_context, "MSFT")
    assert generation.poisoned_reason is not None

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path / "context", monkeypatch)
    object.__setattr__(runtime_context.runtime_contract, "build_id", "changed")
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_contract_safety_snapshot(runtime_context, "MSFT")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    runtime_context, ibc_path = _runtime_context(tmp_path / "ibc-before", monkeypatch)
    ibc_path.write_text("ReadOnlyApi=no\nTradingMode=paper\n")
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_contract_safety_snapshot(runtime_context, "MSFT")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, ibc_path = _runtime_context(tmp_path / "ibc-during", monkeypatch)

    async def change_ibc(_command, timeout):
        ibc_path.write_text("ReadOnlyApi=no\nTradingMode=paper\n")
        return payload

    monkeypatch.setattr(client, "_execute_command_unlocked", change_ibc)
    with pytest.raises(IBKRTransportPoisonedError, match="changed during contract snapshot"):
        await client.get_broker_contract_safety_snapshot(runtime_context, "MSFT")
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_client_contract_snapshot_rejects_live_remote_and_request_only_readonly_context(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    live_context, _ = _runtime_context(
        tmp_path / "live",
        monkeypatch,
        account="U1234567",
    )
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_contract_safety_snapshot(live_context, "MSFT")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    remote_context, _ = _runtime_context(tmp_path / "remote", monkeypatch)
    object.__setattr__(remote_context.diagnostic_connection, "host", "192.0.2.10")
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_contract_safety_snapshot(remote_context, "MSFT")
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()

    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, execute = _connected_client(monkeypatch, payload)
    with pytest.raises(IBKRTransportPoisonedError, match="runtime context"):
        await client.get_broker_contract_safety_snapshot(
            SimpleNamespace(
                diagnostic_connection=SimpleNamespace(
                    host="127.0.0.1",
                    port=4002,
                    client_id=7,
                    readonly=True,
                )
            ),
            "MSFT",
        )
    assert generation.poisoned_reason is not None
    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_contract_snapshot_lifecycle_callback_rejects_generation_drift_before_return(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)

    async def final_dispatch(snapshot):
        assert_producer_owned_broker_contract_safety_snapshot(snapshot)
        client._generation = _WorkerGeneration(
            generation_id="replacement-generation",
            process=SimpleNamespace(poll=lambda: None),
        )
        return snapshot.qualified_contract.con_id

    with pytest.raises(IBKRTransportPoisonedError, match="generation changed"):
        await client.run_with_locked_broker_contract_safety_snapshot(
            runtime_context,
            "MSFT",
            final_dispatch,
        )
    assert generation.poisoned_reason is not None


@pytest.mark.asyncio
async def test_contract_snapshot_lifecycle_callback_holds_lock_against_stop(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, _, _ = _connected_client(monkeypatch, payload)
    runtime_context, _ = _runtime_context(tmp_path, monkeypatch)
    entered = asyncio.Event()
    release = asyncio.Event()
    stop_called = asyncio.Event()

    async def final_dispatch(snapshot):
        assert_producer_owned_broker_contract_safety_snapshot(snapshot)
        entered.set()
        await release.wait()
        return snapshot.qualified_contract.con_id

    async def fake_stop_unlocked():
        stop_called.set()

    monkeypatch.setattr(client, "_stop_unlocked", fake_stop_unlocked)
    dispatch_task = asyncio.create_task(
        client.run_with_locked_broker_contract_safety_snapshot(
            runtime_context,
            "MSFT",
            final_dispatch,
        )
    )
    await entered.wait()
    stop_task = asyncio.create_task(client.stop())
    await asyncio.sleep(0)
    assert not stop_called.is_set()

    release.set()
    assert await dispatch_task == 272093
    await stop_task
    assert stop_called.is_set()


@pytest.mark.asyncio
async def test_contract_snapshot_lifecycle_callback_poisoned_on_final_ibc_drift(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_contract_payload(fake_ib, "MSFT")
    client, generation, _ = _connected_client(monkeypatch, payload)
    runtime_context, ibc_path = _runtime_context(tmp_path, monkeypatch)

    async def drift_ibc_after_proof(snapshot):
        assert_producer_owned_broker_contract_safety_snapshot(snapshot)
        ibc_path.write_text("ReadOnlyApi=no\nTradingMode=paper\n")
        return snapshot.qualified_contract.con_id

    with pytest.raises(IBKRTransportPoisonedError, match="during finalization"):
        await client.run_with_locked_broker_contract_safety_snapshot(
            runtime_context,
            "MSFT",
            drift_ibc_after_proof,
        )
    assert generation.poisoned_reason is not None

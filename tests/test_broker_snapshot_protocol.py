import asyncio
import copy
import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from robo_trader.clients import ibkr_subprocess_worker as worker
from robo_trader.clients import subprocess_ibkr_client as client_module
from robo_trader.clients.subprocess_ibkr_client import (
    BrokerSnapshotAccountMismatchError,
    IBKRError,
    IBKRTimeoutError,
    IBKRTransportPoisonedError,
    SubprocessCrashError,
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.utils import ibkr_safe

ACCOUNT = "DU1234567"
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


def _completed_trade(*, filled="2"):
    order = SimpleNamespace(
        account=ACCOUNT,
        orderId=202,
        permId=602,
        clientId=0,
        action="SELL",
        orderType="LMT",
        tif="DAY",
        totalQuantity="2",
        filledQuantity=filled,
        lmtPrice="125.00",
        auxPrice="1.7976931348623157e+308",
    )
    status = SimpleNamespace(
        status="Filled",
        avgFillPrice="125.00",
    )
    return SimpleNamespace(
        order=order,
        orderStatus=status,
        contract=_contract(),
        log=[],
    )


def _account_summary_values(net_liquidation="100000.00", total_cash="25000.500"):
    return [
        SimpleNamespace(
            account=ACCOUNT,
            tag="AccountType",
            currency="",
            value="INDIVIDUAL",
        ),
        SimpleNamespace(
            account=ACCOUNT,
            tag="NetLiquidation",
            currency="USD",
            value=net_liquidation,
        ),
        SimpleNamespace(
            account=ACCOUNT,
            tag="TotalCashValue",
            currency="USD",
            value=total_cash,
        ),
        SimpleNamespace(
            account=ACCOUNT,
            tag="BuyingPower",
            currency="USD",
            value="999999",
        ),
    ]


class _FakeAccountSummaryWrapper:
    def __init__(self):
        self.acctSummary = {}
        self.pending = {}

    def startReq(self, request_id):
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        return future

    def _endReq(self, request_id):
        future = self.pending.pop(request_id, None)
        if future is not None and not future.done():
            future.set_result(None)

    def accountSummary(self, request_id, account, tag, value, currency):
        key = (account, tag, currency)
        self.acctSummary[key] = SimpleNamespace(
            request_id=request_id,
            account=account,
            tag=tag,
            currency=currency,
            value=value,
        )


class _FakeAccountSummaryClient:
    def __init__(self, owner):
        self.owner = owner
        self.next_request_id = 1
        self.cancelled_request_ids = []

    def getReqId(self):
        request_id = self.next_request_id
        self.next_request_id += 1
        return request_id

    def reqAccountSummary(self, request_id, group, tags):
        self.owner.calls.append("reqAccountSummary")
        self.owner.account_summary_requests.append((request_id, group, tags))
        if self.owner.account_summary_hangs:
            return
        batch_index = min(
            len(self.owner.account_summary_requests) - 1,
            len(self.owner.account_summary_batches) - 1,
        )
        for value in self.owner.account_summary_batches[batch_index]:
            if self.owner.account_summary_bypasses_scoped_callback:
                _FakeAccountSummaryWrapper.accountSummary(
                    self.owner.wrapper,
                    request_id,
                    value.account,
                    value.tag,
                    value.value,
                    value.currency,
                )
            else:
                self.owner.wrapper.accountSummary(
                    request_id,
                    value.account,
                    value.tag,
                    value.value,
                    value.currency,
                )
        for foreign_request_id, value in self.owner.foreign_account_summary_callbacks:
            self.owner.wrapper.accountSummary(
                foreign_request_id,
                value.account,
                value.tag,
                value.value,
                value.currency,
            )
        self.owner.wrapper._endReq(request_id)

    def cancelAccountSummary(self, request_id):
        self.owner.calls.append("cancelAccountSummary")
        self.cancelled_request_ids.append(request_id)
        if self.owner.account_summary_cancel_raises:
            raise RuntimeError("raw cancellation failure with sensitive context")


class _FakeIB:
    def __init__(self):
        self.calls = []
        self.connected = True
        self.account_reads = 0
        self.contract = _contract()
        self.execution_filters = []
        self.account_summary_batches = [_account_summary_values()]
        self.account_summary_requests = []
        self.account_summary_hangs = False
        self.account_summary_bypasses_scoped_callback = False
        self.account_summary_cancel_raises = False
        self.foreign_account_summary_callbacks = []
        self.wrapper = _FakeAccountSummaryWrapper()
        self.client = _FakeAccountSummaryClient(self)

    def isConnected(self):
        self.calls.append("isConnected")
        return self.connected

    def managedAccounts(self):
        self.calls.append("managedAccounts")
        self.account_reads += 1
        return [ACCOUNT]

    async def reqCurrentTimeAsync(self):
        self.calls.append("reqCurrentTimeAsync")
        # Broker clock evidence is captured when the worker call runs.  Using
        # the module-import timestamp makes this otherwise valid fake expire
        # when the full suite takes longer than the safety skew window.
        return datetime.now(timezone.utc)

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
            action="BUY",
            orderType="LMT",
            tif="DAY",
            totalQuantity="2",
            lmtPrice="120.25",
            auxPrice="1.7976931348623157e+308",
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

    async def reqCompletedOrdersAsync(self, api_only):
        self.calls.append("reqCompletedOrdersAsync")
        assert api_only is False
        return []

    async def reqExecutionsAsync(self, execution_filter):
        self.calls.append("reqExecutionsAsync")
        self.execution_filters.append(execution_filter)
        assert execution_filter.acctCode == ACCOUNT
        assert execution_filter.time.endswith(" UTC")
        execution = SimpleNamespace(
            execId="0001.01",
            orderId=101,
            permId=501,
            clientId=0,
            acctNumber=ACCOUNT,
            side="BOT",
            shares="1.25",
            price="120.25",
            avgPrice="120.25",
            time=NOW,
            exchange="NASDAQ",
        )
        return [
            SimpleNamespace(
                execution=execution,
                contract=self.contract,
                commissionReport=SimpleNamespace(
                    execId="0001.01",
                    commission="1.23",
                    currency="USD",
                    realizedPNL="4.56",
                ),
                time=NOW,
            )
        ]

    def accountValues(self, account):
        self.calls.append("accountValues")
        assert account == ACCOUNT
        return [
            SimpleNamespace(
                account=ACCOUNT,
                tag="NetLiquidation",
                currency="USD",
                value="1.00",
            ),
            SimpleNamespace(
                account=ACCOUNT,
                tag="TotalCashValue",
                currency="USD",
                value="2.00",
            ),
            SimpleNamespace(
                account=ACCOUNT,
                tag="BuyingPower",
                currency="USD",
                value="999999",
            ),
        ]

    async def qualifyContractsAsync(self, contract):
        self.calls.append("qualifyContractsAsync")
        return [contract]


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


@pytest.mark.asyncio
async def test_worker_snapshot_is_fresh_atomic_read_only_and_precise(fake_ib):
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "success"
    data = result["data"]
    assert data["snapshot_schema_version"] == 2
    assert data["account"] == ACCOUNT
    assert data["positions"][0]["quantity"] == "10.5"
    assert data["positions"][0]["avg_cost"] == "123.45"
    assert {item["tag"] for item in data["balances"]} == {
        "BuyingPower",
        "NetLiquidation",
        "TotalCashValue",
    }
    assert data["open_orders"][0]["stop_price"] is None
    assert "stop_price" in data["open_orders"][0]["unavailable"]
    assert data["executions"][0]["commission"] == "1.23"
    assert data["executions"][0]["commission_currency"] == "USD"
    assert data["executions"][0]["unavailable"] == {}
    assert data["account_type"] == "paper"
    assert data["account_structure"] == "INDIVIDUAL"
    assert data["base_currency"] == "USD"
    assert data["total_cash"] == "25000.5"
    assert data["buying_power"] == "999999"
    assert data["completed_orders"] == []
    assert all(data["completeness"].values())
    assert {item["collection"]: item["result_count"] for item in data["collection_evidence"]} == {
        "commissions": 1,
        "completed_orders": 0,
        "executions": 1,
        "open_orders": 1,
        "positions": 1,
    }
    assert data["execution_scope"]["kind"] == "bounded_execution_filter"
    broker_before = datetime.fromisoformat(data["broker_time_before"])
    broker_after = datetime.fromisoformat(data["broker_time_after"])
    scope_start = datetime.fromisoformat(data["execution_scope"]["start_at"])
    scope_end = datetime.fromisoformat(data["execution_scope"]["end_at"])
    assert scope_start == broker_before.replace(microsecond=0) - timedelta(hours=24)
    assert fake_ib.execution_filters[0].time == scope_start.strftime("%Y%m%d %H:%M:%S UTC")
    assert broker_before <= scope_end <= broker_after
    assert scope_end != datetime.fromisoformat(data["retrieved_at"])

    allowed = {
        "isConnected",
        "managedAccounts",
        "reqCurrentTimeAsync",
        "reqPositionsAsync",
        "reqAllOpenOrdersAsync",
        "reqCompletedOrdersAsync",
        "reqExecutionsAsync",
        "reqAccountSummary",
        "cancelAccountSummary",
        "qualifyContractsAsync",
    }
    assert set(fake_ib.calls) <= allowed
    assert "reqPositionsAsync" in fake_ib.calls
    assert "reqAllOpenOrdersAsync" in fake_ib.calls
    assert "reqExecutionsAsync" in fake_ib.calls
    assert "reqAccountSummary" in fake_ib.calls
    assert "cancelAccountSummary" in fake_ib.calls
    assert "accountValues" not in fake_ib.calls
    execution_request_index = fake_ib.calls.index("reqExecutionsAsync")
    assert fake_ib.calls[execution_request_index - 1] == "reqCurrentTimeAsync"
    assert not any(
        token in call.lower()
        for call in fake_ib.calls
        for token in ("placeorder", "cancelorder", "exercise", "replaceorder")
    )


@pytest.mark.asyncio
async def test_worker_snapshot_rechecks_account_and_suppresses_sensitive_errors(
    fake_ib, monkeypatch
):
    def changing_accounts():
        fake_ib.calls.append("managedAccounts")
        fake_ib.account_reads += 1
        return [ACCOUNT] if fake_ib.account_reads == 1 else ["DU_WRONG_ACCOUNT"]

    monkeypatch.setattr(fake_ib, "managedAccounts", changing_accounts)
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result == {
        "status": worker.PROTOCOL_ERROR_STATUS,
        "error": "Broker snapshot collection failed",
        "error_type": worker.PROTOCOL_ERROR_TYPE,
    }
    assert ACCOUNT not in result["error"]
    assert "DU_WRONG_ACCOUNT" not in result["error"]


@pytest.mark.asyncio
async def test_worker_positively_evidences_zero_positions_and_orders(fake_ib, monkeypatch):
    monkeypatch.setattr(fake_ib, "reqPositionsAsync", AsyncMock(return_value=[]))
    monkeypatch.setattr(fake_ib, "reqAllOpenOrdersAsync", AsyncMock(return_value=[]))

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "success"
    assert result["data"]["positions"] == []
    assert result["data"]["open_orders"] == []
    assert result["data"]["completed_orders"] == []
    counts = {
        evidence["collection"]: evidence["result_count"]
        for evidence in result["data"]["collection_evidence"]
    }
    assert counts["positions"] == 0
    assert counts["open_orders"] == 0
    assert counts["completed_orders"] == 0


@pytest.mark.asyncio
async def test_worker_collects_and_stability_checks_completed_orders(fake_ib, monkeypatch):
    monkeypatch.setattr(
        fake_ib,
        "reqCompletedOrdersAsync",
        AsyncMock(return_value=[_completed_trade()]),
    )

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "success"
    completed = result["data"]["completed_orders"][0]
    assert completed["broker_order_id"] == 202
    assert completed["status"] == "Filled"
    assert completed["filled_quantity"] == "2"
    assert completed["remaining_quantity"] == "0"
    assert fake_ib.reqCompletedOrdersAsync.await_count == 2


@pytest.mark.asyncio
async def test_worker_rejects_completed_order_mutation(fake_ib, monkeypatch):
    monkeypatch.setattr(
        fake_ib,
        "reqCompletedOrdersAsync",
        AsyncMock(side_effect=[[_completed_trade(filled="2")], [_completed_trade(filled="1")]]),
    )

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE


@pytest.mark.asyncio
async def test_worker_repeated_snapshot_forces_fresh_account_summary(fake_ib):
    fake_ib.account_summary_batches = [
        _account_summary_values("100000.00", "25000.00"),
        _account_summary_values("101000.00", "26000.00"),
    ]

    first = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})
    second = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert first["status"] == "success"
    assert second["status"] == "success"
    first_balances = {
        (row["tag"], row["currency"]): row["value"] for row in first["data"]["balances"]
    }
    second_balances = {
        (row["tag"], row["currency"]): row["value"] for row in second["data"]["balances"]
    }
    assert first_balances[("NetLiquidation", "USD")] == "100000"
    assert second_balances[("NetLiquidation", "USD")] == "101000"
    assert first_balances[("TotalCashValue", "USD")] == "25000"
    assert second_balances[("TotalCashValue", "USD")] == "26000"
    assert len(fake_ib.account_summary_requests) == 2
    assert fake_ib.client.cancelled_request_ids == [1, 2]
    assert "accountSummary" not in fake_ib.wrapper.__dict__


@pytest.mark.asyncio
async def test_worker_account_summary_ignores_foreign_request_cache_interleaving(fake_ib):
    fake_ib.account_summary_batches = [
        _account_summary_values("101000.00", "26000.00"),
    ]
    fake_ib.foreign_account_summary_callbacks = [
        (999, value) for value in _account_summary_values("1.00", "2.00")
    ]

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "success"
    balances = {(row["tag"], row["currency"]): row["value"] for row in result["data"]["balances"]}
    assert balances[("NetLiquidation", "USD")] == "101000"
    assert balances[("TotalCashValue", "USD")] == "26000"
    assert fake_ib.wrapper.acctSummary[(ACCOUNT, "NetLiquidation", "USD")].value == "1.00"
    assert fake_ib.wrapper.acctSummary[(ACCOUNT, "TotalCashValue", "USD")].value == "2.00"
    assert "accountSummary" not in fake_ib.wrapper.__dict__


@pytest.mark.asyncio
async def test_worker_same_request_wrong_account_summary_callback_blocks(fake_ib):
    fake_ib.account_summary_batches[0].append(
        SimpleNamespace(
            account="DU7654321",
            tag="BuyingPower",
            currency="USD",
            value="1",
        )
    )

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE
    assert "DU7654321" not in json.dumps(result)


@pytest.mark.asyncio
async def test_worker_conflicting_duplicate_account_summary_callback_blocks(fake_ib):
    fake_ib.account_summary_batches[0].append(
        SimpleNamespace(
            account=ACCOUNT,
            tag="BuyingPower",
            currency="USD",
            value="1",
        )
    )

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE


@pytest.mark.asyncio
async def test_worker_account_summary_fails_closed_without_owned_callback_provenance(fake_ib):
    fake_ib.account_summary_bypasses_scoped_callback = True

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result == {
        "status": worker.PROTOCOL_ERROR_STATUS,
        "error": "Broker snapshot collection failed",
        "error_type": worker.PROTOCOL_ERROR_TYPE,
    }
    assert fake_ib.wrapper.acctSummary
    assert fake_ib.client.cancelled_request_ids == [1]
    assert "accountSummary" not in fake_ib.wrapper.__dict__


@pytest.mark.asyncio
async def test_worker_wrong_expected_account_performs_zero_data_reads(fake_ib):
    result = await worker.handle_get_broker_snapshot({"expected_account": "DU_WRONG_ACCOUNT"})

    assert result["status"] == "error"
    assert result["error_type"] == "BrokerSnapshotAccountMismatchError"
    assert not any(
        call.startswith("req")
        or call
        in {
            "accountValues",
            "cancelAccountSummary",
            "qualifyContractsAsync",
        }
        for call in fake_ib.calls
    )


@pytest.mark.asyncio
async def test_worker_snapshot_requires_exact_persisted_diagnostic_identity(fake_ib, monkeypatch):
    monkeypatch.setattr(
        worker,
        "worker_connection_identity",
        ("127.0.0.1", 4001, 7, True),
    )

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "error"
    assert not any(call.startswith("req") for call in fake_ib.calls)


@pytest.mark.asyncio
async def test_worker_rejects_mid_collection_position_mutation(fake_ib, monkeypatch):
    original = fake_ib.reqPositionsAsync
    reads = 0

    async def changing_positions():
        nonlocal reads
        reads += 1
        positions = await original()
        if reads == 2:
            positions[0].position = "11"
        return positions

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", changing_positions)
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE
    assert reads == 2


def _mutate_avg_fill_price(trade, read_number):
    trade.orderStatus.filled = "1"
    trade.orderStatus.remaining = "1"
    trade.orderStatus.avgFillPrice = "120" if read_number == 1 else "121"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        lambda trade, read: (setattr(trade.order, "lmtPrice", "121.25") if read == 2 else None),
        lambda trade, read: (setattr(trade.order, "action", "SELL") if read == 2 else None),
        lambda trade, read: (setattr(trade.order, "orderType", "MKT") if read == 2 else None),
        lambda trade, read: setattr(trade.order, "tif", "GTC") if read == 2 else None,
        lambda trade, read: (setattr(trade.order, "permId", 999) if read == 2 else None),
        lambda trade, read: (setattr(trade.order, "auxPrice", "119") if read == 2 else None),
        _mutate_avg_fill_price,
        lambda trade, read: (
            setattr(trade.log[-1], "time", NOW + timedelta(seconds=1)) if read == 2 else None
        ),
        lambda trade, read: (setattr(trade.contract, "conId", 999999) if read == 2 else None),
        lambda trade, read: (
            setattr(trade.orderStatus, "status", "PreSubmitted") if read == 2 else None
        ),
    ],
    ids=[
        "limit-price",
        "side",
        "order-type",
        "time-in-force",
        "permanent-id",
        "stop-price",
        "average-fill-price",
        "status-timestamp",
        "contract-identity",
        "status",
    ],
)
async def test_worker_rejects_any_emitted_order_state_mutation(fake_ib, monkeypatch, mutate):
    original = fake_ib.reqAllOpenOrdersAsync
    reads = 0

    async def changing_orders():
        nonlocal reads
        reads += 1
        trades = await original()
        mutate(trades[0], reads)
        return trades

    monkeypatch.setattr(fake_ib, "reqAllOpenOrdersAsync", changing_orders)

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE
    assert reads == 2


@pytest.mark.asyncio
async def test_worker_rejects_cross_account_execution(fake_ib, monkeypatch):
    original = fake_ib.reqExecutionsAsync

    async def wrong_execution(execution_filter):
        fills = await original(execution_filter)
        fills[0].execution.acctNumber = "DU_WRONG_ACCOUNT"
        return fills

    monkeypatch.setattr(fake_ib, "reqExecutionsAsync", wrong_execution)
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})
    assert result["status"] == worker.PROTOCOL_ERROR_STATUS
    assert result["error_type"] == worker.PROTOCOL_ERROR_TYPE
    assert "DU_WRONG_ACCOUNT" not in result["error"]


@pytest.mark.asyncio
async def test_worker_rejects_execution_after_pre_request_cutoff(fake_ib, monkeypatch):
    original = fake_ib.reqExecutionsAsync

    async def execution_after_cutoff(execution_filter):
        fills = await original(execution_filter)
        fills[0].execution.time = NOW + timedelta(days=1)
        fills[0].time = NOW + timedelta(days=1)
        return fills

    monkeypatch.setattr(fake_ib, "reqExecutionsAsync", execution_after_cutoff)

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result == {
        "status": worker.PROTOCOL_ERROR_STATUS,
        "error": "Broker snapshot collection failed",
        "error_type": worker.PROTOCOL_ERROR_TYPE,
    }


@pytest.mark.asyncio
async def test_worker_emits_available_commission_evidence_canonically(fake_ib, monkeypatch):
    original = fake_ib.reqExecutionsAsync

    async def with_commission(execution_filter):
        fills = await original(execution_filter)
        fills[0].commissionReport = SimpleNamespace(
            execId="0001.01",
            commission="1.2300",
            currency="USD",
            realizedPNL="-2.500",
        )
        return fills

    monkeypatch.setattr(fake_ib, "reqExecutionsAsync", with_commission)
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    execution = result["data"]["executions"][0]
    assert execution["commission"] == "1.23"
    assert execution["commission_currency"] == "USD"
    assert execution["realized_pnl"] == "-2.5"
    assert execution["unavailable"] == {}


@pytest.mark.asyncio
async def test_worker_partial_commission_callback_times_out_and_blocks(fake_ib, monkeypatch):
    original = fake_ib.reqExecutionsAsync

    async def without_matching_commission(execution_filter):
        fills = await original(execution_filter)
        fills[0].commissionReport.execId = ""
        return fills

    monkeypatch.setattr(fake_ib, "reqExecutionsAsync", without_matching_commission)
    monkeypatch.setattr(worker, "BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS", 0.001)

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result == {
        "status": "error",
        "error": "Broker snapshot collection timed out",
        "error_type": "TimeoutError",
        "detail": "Broker snapshot stage timed out: commissions",
    }


@pytest.mark.asyncio
async def test_worker_sorts_open_orders_by_client_then_order_id(fake_ib, monkeypatch):
    original = fake_ib.reqAllOpenOrdersAsync

    async def multiple_orders():
        first = (await original())[0]
        first.order.clientId = 2
        first.order.orderId = 1
        second = copy.deepcopy(first)
        second.order.clientId = 1
        second.order.orderId = 100
        second.order.permId = 600
        return [first, second]

    monkeypatch.setattr(fake_ib, "reqAllOpenOrdersAsync", multiple_orders)
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert [
        (item["client_id"], item["broker_order_id"]) for item in result["data"]["open_orders"]
    ] == [(1, 100), (2, 1)]


@pytest.mark.asyncio
async def test_worker_command_routes_atomic_snapshot(fake_ib):
    response = await worker.handle_command(
        {
            "command": "get_broker_snapshot",
            "params": {"expected_account": ACCOUNT},
        }
    )
    assert response["status"] == "success"
    assert response["data"]["account"] == ACCOUNT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "params",
    [
        {"port": 4001, "readonly": True, "client_id": 7},
        {"port": 4002, "readonly": 1, "client_id": 7},
        {"port": 4002, "readonly": True, "client_id": True},
    ],
)
async def test_worker_diagnostic_connect_rejects_before_ib_client_creation(monkeypatch, params):
    factory = Mock()
    monkeypatch.setattr(worker, "ib", None)
    monkeypatch.setattr(worker, "IB", factory)

    result = await worker.handle_connect(params)

    assert result["status"] == "error"
    factory.assert_not_called()


@pytest.mark.asyncio
async def test_worker_accepts_existing_zero_client_id(monkeypatch):
    observed = {}

    class ConnectIB:
        def __init__(self):
            self.client = SimpleNamespace(serverVersion=lambda: 180)

        async def connectAsync(self, **kwargs):
            observed.update(kwargs)

        def isConnected(self):
            return True

        def managedAccounts(self):
            return [ACCOUNT]

    monkeypatch.setattr(worker, "ib", None)
    monkeypatch.setattr(worker, "worker_connection_identity", None)
    monkeypatch.setattr(worker, "gateway_api_down", False)
    monkeypatch.setattr(worker, "IB", ConnectIB)
    monkeypatch.setattr(worker.asyncio, "sleep", AsyncMock())

    response = await worker.handle_connect(
        {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 0,
            "readonly": True,
            "timeout": 1.0,
        }
    )

    assert response["status"] == "success"
    assert observed["clientId"] == 0
    assert response["data"]["client_id"] == 0


@pytest.mark.asyncio
async def test_worker_connect_stderr_masks_managed_account(monkeypatch, capsys):
    class ConnectIB:
        def __init__(self):
            self.client = SimpleNamespace(serverVersion=lambda: 180)

        async def connectAsync(self, **kwargs):
            return None

        def isConnected(self):
            return True

        def managedAccounts(self):
            return [ACCOUNT]

    monkeypatch.setattr(worker, "ib", None)
    monkeypatch.setattr(worker, "worker_connection_identity", None)
    monkeypatch.setattr(worker, "gateway_api_down", False)
    monkeypatch.setattr(worker, "IB", ConnectIB)
    monkeypatch.setattr(worker.asyncio, "sleep", AsyncMock())

    response = await worker.handle_connect(
        {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 7,
            "readonly": True,
            "timeout": 1.0,
        }
    )

    assert response["status"] == "success"
    assert ACCOUNT not in capsys.readouterr().err


async def _worker_payload(fake_ib):
    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})
    assert result["status"] == "success"
    return result["data"]


def _connected_client(monkeypatch, payload):
    client = SubprocessIBKRClient()
    generation = _WorkerGeneration(
        generation_id="snapshot-generation",
        process=SimpleNamespace(poll=lambda: 0),
    )
    client._generation = generation
    with client._connection_state_lock:
        client._connected = True
        client._connection_identity = ("127.0.0.1", 4002, 7, True)
        client._connection_generation_id = generation.generation_id
    execute = AsyncMock(return_value=payload)
    monkeypatch.setattr(client, "_execute_command_unlocked", execute)
    return client, execute


def _transport_response_client(response, *, swap_generation=False):
    """Route one synthetic worker result through the real parent classifier."""
    client = SubprocessIBKRClient()
    process = SimpleNamespace(poll=lambda: None)
    generation = _WorkerGeneration(
        generation_id="snapshot-transport-generation",
        process=process,
    )

    class ImmediateResponseStdin:
        def write(self, _value):
            if swap_generation:
                client._generation = SimpleNamespace(
                    generation_id="replacement-generation",
                )
            with generation.state_lock:
                pending = next(iter(generation.pending.values()))
            pending.future.set_result(dict(response))

        def flush(self):
            return None

    process.stdin = ImmediateResponseStdin()
    client.process = process
    client._generation = generation
    return client, generation


@pytest.mark.asyncio
async def test_worker_snapshot_timeout_preserves_parent_timeout_poison(fake_ib, monkeypatch):
    async def timeout_positions():
        raise TimeoutError("snapshot timeout with sensitive diagnostic context")

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", timeout_positions)
    worker_response = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert worker_response == {
        "status": "error",
        "error": "Broker snapshot collection timed out",
        "error_type": "TimeoutError",
        "detail": "Broker snapshot stage timed out: positions_initial",
    }
    assert ACCOUNT not in worker_response["error"]
    assert "sensitive diagnostic context" not in json.dumps(worker_response)

    client, generation = _transport_response_client(worker_response)
    with pytest.raises(IBKRTimeoutError, match="Broker snapshot collection timed out"):
        await client._execute_command_unlocked({"command": "get_broker_snapshot"})

    assert generation.poisoned_reason is not None
    assert "worker-reported broker timeout" in generation.poisoned_reason
    assert "get_broker_snapshot" in generation.poisoned_reason


@pytest.mark.asyncio
async def test_worker_account_summary_timeout_cancels_subscription_and_is_sanitized(
    fake_ib, monkeypatch
):
    fake_ib.account_summary_hangs = True
    monkeypatch.setattr(worker, "BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS", 0.001)

    worker_response = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert worker_response == {
        "status": "error",
        "error": "Broker snapshot collection timed out",
        "error_type": "TimeoutError",
        "detail": "Broker snapshot stage timed out: account_summary",
    }
    assert fake_ib.client.cancelled_request_ids == [1]
    assert fake_ib.wrapper.pending == {}
    assert "accountSummary" not in fake_ib.wrapper.__dict__


@pytest.mark.asyncio
async def test_worker_account_summary_cancellation_restores_callback_and_subscription(fake_ib):
    fake_ib.account_summary_hangs = True

    task = asyncio.create_task(worker._request_fresh_broker_account_summary(ACCOUNT))
    while not fake_ib.account_summary_requests:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake_ib.client.cancelled_request_ids == [1]
    assert fake_ib.wrapper.pending == {}
    assert "accountSummary" not in fake_ib.wrapper.__dict__


@pytest.mark.asyncio
async def test_worker_account_summary_cancel_failure_still_cleans_request_and_callback(fake_ib):
    fake_ib.account_summary_cancel_raises = True

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result == {
        "status": worker.PROTOCOL_ERROR_STATUS,
        "error": "Broker snapshot collection failed",
        "error_type": worker.PROTOCOL_ERROR_TYPE,
    }
    assert fake_ib.client.cancelled_request_ids == [1]
    assert fake_ib.wrapper.pending == {}
    assert "accountSummary" not in fake_ib.wrapper.__dict__
    assert "sensitive context" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", sorted(worker.BROKER_SNAPSHOT_REQUEST_STAGES))
async def test_worker_snapshot_stage_deadlines_are_bounded_and_sanitized(stage, monkeypatch):
    sensitive_detail = "DU_SECRET_ACCOUNT raw broker exception"

    async def never_finishes():
        await asyncio.Future()
        raise AssertionError(sensitive_detail)

    monkeypatch.setattr(worker, "BROKER_SNAPSHOT_STAGE_TIMEOUT_SECONDS", 0.001)

    with pytest.raises(worker.BrokerSnapshotStageTimeout) as exc_info:
        await worker._await_broker_snapshot_stage(stage, never_finishes())

    assert exc_info.value.stage == stage
    assert str(exc_info.value) == f"Broker snapshot stage timed out: {stage}"
    assert sensitive_detail not in str(exc_info.value)


@pytest.mark.asyncio
async def test_worker_snapshot_routes_every_broker_await_through_stage_deadline(
    fake_ib, monkeypatch
):
    observed = []

    async def record_stage(stage, request):
        observed.append(stage)
        return await request

    monkeypatch.setattr(worker, "_await_broker_snapshot_stage", record_stage)

    result = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert result["status"] == "success"
    assert observed == [
        "broker_time_before",
        "positions_initial",
        "positions_initial_identity",
        "position_identity",
        "open_orders_initial",
        "open_order_identity",
        "completed_orders_initial",
        "broker_time_execution_cutoff",
        "executions",
        "commissions",
        "execution_identity",
        "account_summary",
        "positions_final",
        "positions_final_identity",
        "open_orders_final",
        "open_orders_final_identity",
        "completed_orders_final",
        "completed_orders_final_identity",
        "broker_time_after",
    ]


@pytest.mark.asyncio
async def test_worker_snapshot_protocol_error_poisons_exact_parent_generation(fake_ib, monkeypatch):
    original = fake_ib.reqPositionsAsync
    reads = 0

    async def changing_positions():
        nonlocal reads
        reads += 1
        positions = await original()
        if reads == 2:
            positions[0].position = "11"
        return positions

    monkeypatch.setattr(fake_ib, "reqPositionsAsync", changing_positions)
    worker_response = await worker.handle_get_broker_snapshot({"expected_account": ACCOUNT})

    assert worker_response["status"] == worker.PROTOCOL_ERROR_STATUS
    assert worker_response["error_type"] == worker.PROTOCOL_ERROR_TYPE

    client, original_generation = _transport_response_client(
        worker_response,
        swap_generation=True,
    )
    with pytest.raises(IBKRTransportPoisonedError, match="Unknown worker response status"):
        await client._execute_command_unlocked({"command": "get_broker_snapshot"})

    assert original_generation.poisoned_reason == "unknown response status"
    assert client._generation.generation_id == "replacement-generation"


@pytest.mark.asyncio
async def test_worker_snapshot_account_mismatch_is_safely_classified_and_poisoned(
    fake_ib,
):
    worker_response = await worker.handle_get_broker_snapshot(
        {"expected_account": "DU_EXPECTED_SECRET"}
    )

    assert worker_response == {
        "status": "error",
        "error": "Broker snapshot account mismatch",
        "error_type": "BrokerSnapshotAccountMismatchError",
    }
    assert ACCOUNT not in worker_response["error"]
    assert "DU_EXPECTED_SECRET" not in worker_response["error"]

    client, generation = _transport_response_client(worker_response)
    with pytest.raises(
        BrokerSnapshotAccountMismatchError,
        match="Broker snapshot account mismatch",
    ) as exc_info:
        await client._execute_command_unlocked({"command": "get_broker_snapshot"})

    assert ACCOUNT not in str(exc_info.value)
    assert "DU_EXPECTED_SECRET" not in str(exc_info.value)
    assert generation.poisoned_reason == ("worker-reported broker snapshot account mismatch")


@pytest.mark.asyncio
async def test_diagnostic_connect_failure_is_redacted_at_worker_and_parent(
    monkeypatch,
    capsys,
    caplog,
):
    raw_identity = "DU1234567"

    class LeakyConnectIB:
        async def connectAsync(self, **_kwargs):
            raise RuntimeError(f"broker rejected account {raw_identity}")

    monkeypatch.setattr(worker, "IB", LeakyConnectIB)
    monkeypatch.setattr(worker, "ib", None)
    monkeypatch.setattr(worker, "gateway_api_down", False)
    monkeypatch.setattr(worker, "gateway_failure_detail", "")

    worker_response = await worker.handle_connect(
        {
            "host": "127.0.0.1",
            "port": 4002,
            "client_id": 7,
            "readonly": True,
            "timeout": 1.0,
        }
    )

    serialized = json.dumps(worker_response)
    assert raw_identity not in serialized
    assert raw_identity not in capsys.readouterr().err
    assert "traceback" not in worker_response
    assert worker_response["error"] == "Diagnostic broker connection failed"

    # Defense in depth: even an older or compromised worker response containing
    # free-form sensitive text must be sanitized again by the parent.
    worker_response["error"] = f"broker rejected account {raw_identity}"
    worker_response["traceback"] = f"trace included {raw_identity}"
    client, _generation = _transport_response_client(worker_response)
    with pytest.raises(IBKRError) as exc_info:
        await client._execute_command_unlocked({"command": "connect"})

    assert raw_identity not in str(exc_info.value)
    assert raw_identity not in caplog.text


@pytest.mark.asyncio
async def test_diagnostic_disconnect_failure_is_redacted_at_worker_and_parent(
    monkeypatch,
    capsys,
    caplog,
):
    raw_identity = "DU_DISCONNECT_SECRET"

    class ConnectedIB:
        pass

    monkeypatch.setattr(worker, "ib", ConnectedIB())
    monkeypatch.setattr(
        worker,
        "safe_disconnect",
        Mock(side_effect=RuntimeError(f"disconnect account {raw_identity}")),
    )

    worker_response = await worker.handle_disconnect()

    assert worker_response == {
        "status": "error",
        "error": "Diagnostic broker disconnect failed",
        "error_type": "BrokerDisconnectionError",
    }
    assert raw_identity not in json.dumps(worker_response)
    assert raw_identity not in capsys.readouterr().err

    # Treat disconnect responses as untrusted even if an older or compromised
    # worker serializes a raw broker exception.
    worker_response["error"] = f"disconnect account {raw_identity}"
    worker_response["error_type"] = f"Leaky{raw_identity}"
    worker_response["detail"] = f"detail {raw_identity}"
    client, _generation = _transport_response_client(worker_response)
    with pytest.raises(IBKRError) as exc_info:
        await client._execute_command_unlocked({"command": "disconnect"})

    assert raw_identity not in str(exc_info.value)
    assert raw_identity not in caplog.text


@pytest.mark.asyncio
async def test_diagnostic_disconnect_real_helper_suppresses_exception_details(
    monkeypatch,
    capsys,
    caplog,
):
    raw_identity = "DU123456789"

    class ConnectedIB:
        def isConnected(self):
            return True

        def disconnect(self):
            return None

    def leaky_disconnect(_ib):
        raise RuntimeError(f"disconnect failed for account {raw_identity}")

    connected_ib = ConnectedIB()
    monkeypatch.setattr(worker, "ib", connected_ib)
    monkeypatch.setattr(ibkr_safe, "_call_original_disconnect", leaky_disconnect)

    worker_response = await worker.handle_disconnect()

    assert worker_response == {"status": "success", "data": {"disconnected": True}}
    assert raw_identity not in capsys.readouterr().err
    assert raw_identity not in caplog.text
    assert "ib.disconnect() raised an exception" in caplog.text


def test_worker_exit_cleanup_stderr_redacts_disconnect_exception(monkeypatch, capsys):
    raw_identity = "DU_EXIT_SECRET"
    monkeypatch.setattr(worker, "ib", object())
    monkeypatch.setattr(
        worker,
        "safe_disconnect",
        Mock(side_effect=RuntimeError(f"cleanup account {raw_identity}")),
    )

    worker._cleanup_on_exit()

    stderr = capsys.readouterr().err
    assert raw_identity not in stderr
    assert "Disconnect error" in stderr


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"port": 4001, "client_id": 7, "readonly": True},
        {"port": 4002, "client_id": True, "readonly": True},
        {"port": 4002, "client_id": 7, "readonly": 1},
    ],
)
async def test_client_diagnostic_connect_rejects_before_transport(monkeypatch, kwargs):
    client = SubprocessIBKRClient()
    zombie_check = AsyncMock()
    monkeypatch.setattr(client, "_check_zombie_connections", zombie_check)

    with pytest.raises(ValueError, match="Diagnostic client"):
        await client.connect(**kwargs)

    zombie_check.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_connect_logs_never_expose_managed_account(monkeypatch, caplog):
    client = SubprocessIBKRClient()
    process = SimpleNamespace(poll=lambda: None)
    generation = SimpleNamespace(
        generation_id="log-generation",
        process=process,
        state_lock=threading.Lock(),
        poisoned_reason=None,
    )
    client.process = process
    client._generation = generation
    monkeypatch.setattr(client, "_check_zombie_connections", AsyncMock(return_value=(0, "none")))
    monkeypatch.setattr(
        client,
        "_execute_command_unlocked",
        AsyncMock(
            return_value={
                "connected": True,
                "accounts": [ACCOUNT],
                "server_version": 180,
            }
        ),
    )

    assert await client.connect(port=4002, client_id=7, readonly=True)
    assert ACCOUNT not in caplog.text


@pytest.mark.asyncio
async def test_client_accepts_strict_snapshot(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    client, execute = _connected_client(monkeypatch, payload)

    snapshot = await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert snapshot == payload
    execute.assert_awaited_once_with(
        {
            "command": "get_broker_snapshot",
            "params": {"expected_account": ACCOUNT},
        },
        timeout=30.0,
    )


@pytest.mark.asyncio
async def test_client_snapshot_removes_diagnostic_worker_log(fake_ib, monkeypatch, tmp_path):
    payload = await _worker_payload(fake_ib)
    client, _ = _connected_client(monkeypatch, payload)
    generation = client._generation
    debug_path = tmp_path / "worker_debug_snapshot.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = debug_path.open("a")
    generation.debug_log_file = debug_file
    generation.debug_log_path = str(debug_path)
    client._debug_log_file = debug_file
    client._debug_log_path = str(debug_path)

    snapshot = await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert snapshot == payload
    assert debug_file.closed
    assert not debug_path.exists()
    assert generation.debug_log_file is None
    assert generation.debug_log_path is None
    assert client._debug_log_file is None
    assert client._debug_log_path is None


@pytest.mark.asyncio
async def test_client_failed_snapshot_also_removes_diagnostic_worker_log(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, execute = _connected_client(monkeypatch, payload)
    generation = client._generation
    debug_path = tmp_path / "worker_debug_failed_snapshot.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = debug_path.open("a")
    generation.debug_log_file = debug_file
    generation.debug_log_path = str(debug_path)
    client._debug_log_file = debug_file
    client._debug_log_path = str(debug_path)
    execute.side_effect = IBKRTransportPoisonedError("simulated response failure")

    with pytest.raises(IBKRTransportPoisonedError, match="simulated response failure"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert debug_file.closed
    assert not debug_path.exists()
    assert generation.debug_log_file is None
    assert generation.debug_log_path is None
    assert client._debug_log_file is None
    assert client._debug_log_path is None


class _CloseFailsOnce:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.close_attempts = 0

    @property
    def closed(self):
        return self.wrapped.closed

    def close(self):
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise OSError("transient close failure")
        self.wrapped.close()


@pytest.mark.asyncio
async def test_client_debug_log_close_failure_retains_handle_and_retries(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, _ = _connected_client(monkeypatch, payload)
    generation = client._generation
    debug_path = tmp_path / "worker_debug_close_retry.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = _CloseFailsOnce(debug_path.open("a"))
    generation.debug_log_file = debug_file
    generation.debug_log_path = str(debug_path)
    client._debug_log_file = debug_file
    client._debug_log_path = str(debug_path)

    with pytest.raises(SubprocessCrashError, match="debug log cleanup"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert debug_file.close_attempts == 1
    assert not debug_file.closed
    assert not debug_path.exists()
    assert generation.debug_log_file is debug_file
    assert client._debug_log_file is debug_file
    assert generation.debug_log_path is None
    assert client._debug_log_path is None

    client._cleanup_worker_debug_log(generation, required=True)

    assert debug_file.close_attempts == 2
    assert debug_file.closed
    assert generation.debug_log_file is None
    assert client._debug_log_file is None


@pytest.mark.asyncio
async def test_client_debug_log_unlink_failure_retains_path_and_retries(
    fake_ib, monkeypatch, tmp_path
):
    payload = await _worker_payload(fake_ib)
    client, _ = _connected_client(monkeypatch, payload)
    generation = client._generation
    debug_path = tmp_path / "worker_debug_unlink_retry.log"
    debug_path.write_text("diagnostic evidence")
    debug_file = debug_path.open("a")
    generation.debug_log_file = debug_file
    generation.debug_log_path = str(debug_path)
    client._debug_log_file = debug_file
    client._debug_log_path = str(debug_path)
    real_unlink = client_module.os.unlink
    unlink_attempts = 0

    def unlink_fails_once(path):
        nonlocal unlink_attempts
        unlink_attempts += 1
        if unlink_attempts == 1:
            raise OSError("transient unlink failure")
        real_unlink(path)

    monkeypatch.setattr(client_module.os, "unlink", unlink_fails_once)

    with pytest.raises(SubprocessCrashError, match="debug log cleanup"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert unlink_attempts == 1
    assert debug_file.closed
    assert debug_path.exists()
    assert generation.debug_log_file is None
    assert client._debug_log_file is None
    assert generation.debug_log_path == str(debug_path)
    assert client._debug_log_path == str(debug_path)

    client._cleanup_worker_debug_log(generation, required=True)

    assert unlink_attempts == 2
    assert not debug_path.exists()
    assert generation.debug_log_path is None
    assert client._debug_log_path is None


@pytest.mark.asyncio
async def test_client_wrong_account_exception_is_masked(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    client, _ = _connected_client(monkeypatch, payload)
    monkeypatch.setattr(client, "_poison_generation", Mock())

    with pytest.raises(BrokerSnapshotAccountMismatchError) as exc_info:
        await client.get_broker_snapshot("DU_EXPECTED_SECRET", max_age_seconds=60)

    message = str(exc_info.value)
    assert "DU_EXPECTED_SECRET" not in message
    assert ACCOUNT not in message
    assert "***CRET" in message
    assert "***4567" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("max_age", [float("nan"), float("inf"), 301, 0, True])
async def test_client_rejects_unbounded_freshness_before_transport(fake_ib, monkeypatch, max_age):
    payload = await _worker_payload(fake_ib)
    client, execute = _connected_client(monkeypatch, payload)

    with pytest.raises(ValueError, match="finite and at most"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=max_age)

    execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_client_generation_swap_during_snapshot_fails_closed(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    client, execute = _connected_client(monkeypatch, payload)
    original_generation = client._generation
    replacement = SimpleNamespace(generation_id="replacement")

    async def swap_generation(*args, **kwargs):
        client._generation = replacement
        return payload

    execute.side_effect = swap_generation
    poison = Mock()
    monkeypatch.setattr(client, "_poison_generation", poison)

    with pytest.raises(IBKRTransportPoisonedError, match="generation changed"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    poison.assert_called_once_with(
        original_generation, "worker generation changed during broker snapshot"
    )


@pytest.mark.asyncio
async def test_client_poison_before_snapshot_validation_never_validates(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    client, execute = _connected_client(monkeypatch, payload)
    generation = client._generation

    async def poison_after_response(*args, **kwargs):
        client._poison_generation(generation, "post-response protocol poison")
        return payload

    execute.side_effect = poison_after_response
    validate = Mock(wraps=client._validate_broker_snapshot)
    monkeypatch.setattr(client, "_validate_broker_snapshot", validate)

    with pytest.raises(IBKRTransportPoisonedError, match="generation is poisoned"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    validate.assert_not_called()


@pytest.mark.asyncio
async def test_client_poison_after_snapshot_validation_never_returns_data(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    client, _ = _connected_client(monkeypatch, payload)
    generation = client._generation
    original_validate = client._validate_broker_snapshot

    def validate_then_poison(*args, **kwargs):
        validated = original_validate(*args, **kwargs)
        client._poison_generation(generation, "post-validation protocol poison")
        return validated

    monkeypatch.setattr(client, "_validate_broker_snapshot", validate_then_poison)

    with pytest.raises(IBKRTransportPoisonedError, match="generation is poisoned"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    assert generation.poisoned_reason == "post-validation protocol poison"
    assert client.is_connected is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(snapshot_schema_version=1),
        lambda data: data["positions"][0].update(quantity="10.50"),
        lambda data: data["positions"][0]["contract"].update(con_id=True),
        lambda data: data["positions"][0]["contract"].update(local_symbol="AAPL ALIAS"),
        lambda data: data["positions"].append(copy.deepcopy(data["positions"][0])),
        lambda data: data.update(retrieved_at=(NOW - timedelta(minutes=5)).isoformat()),
        lambda data: data.update(
            broker_time_after=(
                datetime.fromisoformat(data["broker_time_before"]) + timedelta(minutes=2)
            ).isoformat()
        ),
        lambda data: data.update(
            broker_time_before=(NOW - timedelta(minutes=10)).isoformat(),
            broker_time_after=(NOW - timedelta(minutes=9)).isoformat(),
        ),
        lambda data: data["open_orders"][0]["unavailable"].pop("stop_price"),
        lambda data: data["executions"][0].update(commission=None),
        lambda data: data["executions"][0].update(executed_at="2026-01-01T00:00:00"),
        lambda data: data["executions"][0].update(
            executed_at=(
                datetime.fromisoformat(data["execution_scope"]["end_at"]) + timedelta(seconds=1)
            ).isoformat()
        ),
        lambda data: data["execution_scope"].update(
            start_at=(
                datetime.fromisoformat(data["execution_scope"]["start_at"]) + timedelta(seconds=1)
            ).isoformat()
        ),
        lambda data: data["execution_scope"].update(
            end_at=(
                datetime.fromisoformat(data["broker_time_after"]) + timedelta(microseconds=1)
            ).isoformat()
        ),
        lambda data: data["open_orders"][0].update(filled_quantity="1"),
        lambda data: data["balances"].append(copy.deepcopy(data["balances"][0])),
        lambda data: data.pop("execution_scope"),
        lambda data: data.update(account_type="live"),
        lambda data: data.update(buying_power="1"),
        lambda data: data["completeness"].update(completed_orders=False),
        lambda data: data["collection_evidence"].pop(),
        lambda data: data["collection_evidence"][0].update(result_count=0),
        lambda data: data["executions"][0].update(commission_currency=None),
    ],
)
async def test_client_poison_fails_closed_on_malformed_snapshot(fake_ib, monkeypatch, mutation):
    payload = await _worker_payload(fake_ib)
    mutation(payload)
    client, _ = _connected_client(monkeypatch, payload)
    poison = Mock()
    monkeypatch.setattr(client, "_poison_generation", poison)

    with pytest.raises(IBKRTransportPoisonedError):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)

    poison.assert_called_once()


@pytest.mark.asyncio
async def test_client_rejects_incomplete_commission_evidence(fake_ib, monkeypatch):
    payload = await _worker_payload(fake_ib)
    execution = payload["executions"][0]
    execution["commission"] = None
    execution["unavailable"] = {}
    client, _ = _connected_client(monkeypatch, payload)
    monkeypatch.setattr(client, "_poison_generation", Mock())

    with pytest.raises(IBKRTransportPoisonedError, match="canonical decimal"):
        await client.get_broker_snapshot(ACCOUNT, max_age_seconds=60)


def test_broker_worker_environment_never_inherits_signing_authority(monkeypatch):
    for name in client_module._WORKER_FORBIDDEN_SIGNING_ENV:
        monkeypatch.setenv(name, f"secret-for-{name}")

    child_env = client_module._build_worker_environment("generation-1")

    assert not client_module._WORKER_FORBIDDEN_SIGNING_ENV.intersection(child_env)
    assert not any("PRIVATE_KEY" in name or "SIGNING_KEY" in name for name in child_env)

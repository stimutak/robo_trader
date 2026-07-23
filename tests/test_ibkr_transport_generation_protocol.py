import asyncio
import json
import queue
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from robo_trader.clients import ibkr_subprocess_worker as worker
from robo_trader.clients import subprocess_ibkr_client as client_module
from robo_trader.clients.subprocess_ibkr_client import (
    IBKRConnectionConflictError,
    IBKRTimeoutError,
    IBKRTransportPoisonedError,
    SubprocessCrashError,
    SubprocessIBKRClient,
    _WorkerGeneration,
)
from robo_trader.connection_health import ConnectionHealth, HealthStatus


class _LineStream:
    def __init__(self):
        self.lines = queue.Queue()
        self.closed = False

    def feed(self, value):
        if not self.closed:
            self.lines.put(value)

    def readline(self):
        value = self.lines.get()
        return "" if value is None else value

    def close(self):
        if not self.closed:
            self.closed = True
            self.lines.put(None)


class _FakeStdin:
    def __init__(self, process, handler):
        self.process = process
        self.handler = handler
        self.writes = []

    def write(self, value):
        self.writes.append(value)
        self.handler(self.process, json.loads(value))

    def flush(self):
        return None


class _FakeProcess:
    def __init__(self, handler):
        self.stdout = _LineStream()
        self.stderr = _LineStream()
        self.stdin = _FakeStdin(self, handler)
        self.returncode = None
        self.pid = 9911
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1
        self.returncode = -15
        self.stdout.close()
        self.stderr.close()

    def kill(self):
        self.killed += 1
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise AssertionError("fake worker was not terminated")
        return self.returncode


def _response(request, *, data=None, **overrides):
    response = {
        "protocol_version": 1,
        "generation_id": request["generation_id"],
        "request_id": request["request_id"],
        "command": request["command"],
        "status": "success",
        "data": data or {"ok": True},
    }
    response.update(overrides)
    return response


def _attach_client(handler, generation_id="generation-one"):
    client = SubprocessIBKRClient()
    process = _FakeProcess(handler)
    generation = _WorkerGeneration(generation_id, process)
    client.process = process
    client._generation = generation
    thread = threading.Thread(target=client._read_loop, args=(generation,), daemon=True)
    generation.stdout_thread = thread
    thread.start()
    return client, process, generation


def _feed(process, response):
    process.stdout.feed(json.dumps(response) + "\n")


async def _wait_for_poison(generation):
    for _ in range(100):
        if generation.poisoned_reason:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("worker generation was not poisoned")


@pytest.mark.asyncio
async def test_timeout_poison_rejects_next_command_and_late_response():
    held = {}

    def handler(process, request):
        held["process"] = process
        held["request"] = request

    client, process, generation = _attach_client(handler)
    with pytest.raises(IBKRTimeoutError):
        await client._execute_command({"command": "A"}, timeout=0.01)

    _feed(process, _response(held["request"]))
    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "B"})
    assert len(process.stdin.writes) == 1
    assert generation.poisoned_reason.startswith("command timeout")


@pytest.mark.asyncio
async def test_cancellation_after_flush_poison_rejects_late_response():
    held = {}

    def handler(process, request):
        held.update(process=process, request=request)

    client, process, generation = _attach_client(handler)
    task = asyncio.create_task(client._execute_command({"command": "A"}))
    while not process.stdin.writes:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _feed(process, _response(held["request"]))
    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "B"})
    assert generation.poisoned_reason.startswith("command cancelled")


@pytest.mark.asyncio
async def test_queued_command_fails_without_write_after_first_command_poisons():
    def handler(process, request):
        return None

    client, process, generation = _attach_client(handler)
    first = asyncio.create_task(client._execute_command({"command": "A"}, timeout=0.01))
    while not process.stdin.writes:
        await asyncio.sleep(0)
    second = asyncio.create_task(client._execute_command({"command": "B"}))

    with pytest.raises(IBKRTimeoutError):
        await first
    with pytest.raises(IBKRTransportPoisonedError):
        await second
    assert len(process.stdin.writes) == 1
    assert generation.poisoned_reason.startswith("command timeout")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda response: response.update(generation_id="old-generation"), "stale"),
        (lambda response: response.update(request_id="unknown"), "unknown"),
        (lambda response: response.update(command="other"), "command mismatch"),
        (lambda response: response.update(protocol_version=999), "protocol"),
        (lambda response: response.pop("request_id"), "malformed"),
        (lambda response: response.update(status="mystery"), "unknown response status"),
        (lambda response: response.update(data=[]), "malformed success"),
    ],
)
async def test_bad_response_identity_poisons_generation(mutate, reason):
    def handler(process, request):
        response = _response(request)
        mutate(response)
        _feed(process, response)

    client, _, generation = _attach_client(handler)
    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "ping"})
    assert reason in generation.poisoned_reason


@pytest.mark.asyncio
async def test_malformed_and_duplicate_responses_poison_generation():
    def malformed_handler(process, request):
        process.stdout.feed("{broken\n")

    client, _, malformed_generation = _attach_client(malformed_handler)
    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "ping"})
    assert "malformed JSON" in malformed_generation.poisoned_reason

    def duplicate_handler(process, request):
        response = _response(request)
        _feed(process, response)
        _feed(process, response)

    client, _, duplicate_generation = _attach_client(duplicate_handler, "generation-two")
    await client._execute_command({"command": "ping"})
    await _wait_for_poison(duplicate_generation)
    assert duplicate_generation.poisoned_reason == "duplicate response"


@pytest.mark.asyncio
async def test_new_generation_cannot_consume_old_generation_response():
    old_held = {}

    def old_handler(process, request):
        old_held.update(process=process, request=request)

    client, old_process, old_generation = _attach_client(old_handler, "old")
    old_task = asyncio.create_task(client._execute_command({"command": "old"}))
    while not old_process.stdin.writes:
        await asyncio.sleep(0)
    old_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await old_task
    old_generation.stdout_thread.join(timeout=1)
    assert not old_generation.stdout_thread.is_alive()

    def new_handler(process, request):
        _feed(process, _response(request, data={"generation": "new"}))

    new_process = _FakeProcess(new_handler)
    new_generation = _WorkerGeneration("new", new_process)
    client.process = new_process
    client._generation = new_generation
    new_thread = threading.Thread(target=client._read_loop, args=(new_generation,), daemon=True)
    new_generation.stdout_thread = new_thread
    new_thread.start()

    _feed(old_process, _response(old_held["request"], data={"generation": "old"}))
    assert await client._execute_command({"command": "new"}) == {"generation": "new"}
    assert new_generation.poisoned_reason is None


@pytest.mark.asyncio
async def test_concurrent_start_creates_only_one_worker(monkeypatch):
    created = []

    # The production allowlist intentionally rejects GitHub Actions'
    # /opt/hostedtoolcache interpreter. This test replaces Popen and never
    # executes it, so allow only this process's resolved interpreter prefix in
    # the test rather than weakening the runtime policy.
    interpreter_prefix = client_module.Path(client_module.sys.executable).resolve().parent
    monkeypatch.setattr(
        client_module,
        "_INTERPRETER_PREFIX_ALLOWLIST",
        client_module._INTERPRETER_PREFIX_ALLOWLIST + (interpreter_prefix,),
    )

    def popen(*args, **kwargs):
        process = _FakeProcess(lambda process, request: None)
        created.append(process)
        return process

    monkeypatch.setattr(client_module.subprocess, "Popen", popen)
    client = SubprocessIBKRClient()
    await asyncio.gather(client.start(), client.start())
    assert len(created) == 1
    await client.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_inflight_command_and_prevents_later_write():
    held = {}

    def handler(process, request):
        held.update(process=process, request=request)

    client, process, _ = _attach_client(handler)
    command = asyncio.create_task(client._execute_command({"command": "ping"}))
    while not process.stdin.writes:
        await asyncio.sleep(0)
    stop = asyncio.create_task(client.stop())
    await asyncio.sleep(0)
    assert not stop.done()

    _feed(process, _response(held["request"]))
    await command
    await stop
    assert process.poll() is not None
    assert len(process.stdin.writes) == 1
    with pytest.raises(SubprocessCrashError):
        await client._execute_command({"command": "after-stop"})
    assert len(process.stdin.writes) == 1


@pytest.mark.asyncio
async def test_completed_request_tombstones_are_bounded():
    def handler(process, request):
        _feed(process, _response(request))

    client, _, generation = _attach_client(handler)
    for _ in range(1030):
        await client._execute_command({"command": "ping"})
    assert len(generation.completed) == 1024
    assert len(generation.completed_order) == 1024


@pytest.mark.asyncio
async def test_poison_reaper_escalates_to_kill_for_stubborn_worker():
    class StubbornProcess(_FakeProcess):
        def terminate(self):
            self.terminated += 1

        def wait(self, timeout=None):
            if self.returncode is None:
                raise client_module.subprocess.TimeoutExpired("worker", timeout)
            return self.returncode

    client = SubprocessIBKRClient()
    process = StubbornProcess(
        lambda process, request: _feed(process, _response(request, request_id="unknown"))
    )
    generation = _WorkerGeneration("stubborn", process)
    client.process = process
    client._generation = generation
    thread = threading.Thread(target=client._read_loop, args=(generation,), daemon=True)
    generation.stdout_thread = thread
    thread.start()

    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "ping"})
    for _ in range(100):
        if process.killed:
            break
        await asyncio.sleep(0.02)
    assert process.terminated == 1
    assert process.killed == 1


@pytest.mark.asyncio
async def test_ensure_healthy_never_auto_retries_poisoned_generation():
    def handler(process, request):
        process.stdout.feed("{broken\n")

    client, _, generation = _attach_client(handler)
    with pytest.raises(IBKRTransportPoisonedError):
        await client._execute_command({"command": "ping"})
    with pytest.raises(IBKRTransportPoisonedError, match="Refusing automatic retry"):
        await client.ensure_healthy()
    assert generation.poisoned_reason


def _valid_historical_data(symbol="AAPL", contract_symbol="AAPL", con_id=265598):
    now = datetime.now(timezone.utc).isoformat()
    return {
        "bars": [],
        "requested_symbol": symbol,
        "qualified_contract": {
            "symbol": contract_symbol,
            "local_symbol": contract_symbol,
            "con_id": con_id,
            "security_type": "STK",
            "exchange": "SMART",
            "primary_exchange": "NASDAQ",
            "currency": "USD",
            "trading_class": "NMS",
        },
        "broker_timestamp": now,
        "retrieval_timestamp": now,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        _valid_historical_data(symbol="MSFT"),
        _valid_historical_data(contract_symbol="MSFT"),
        _valid_historical_data(con_id=0),
        _valid_historical_data(con_id=True),
    ],
)
async def test_historical_response_symbol_and_contract_identity_fail_closed(data):
    def handler(process, request):
        _feed(process, _response(request, data=data))

    client, _, generation = _attach_client(handler)
    with pytest.raises(IBKRTransportPoisonedError):
        await client.get_historical_bars("AAPL")
    assert generation.poisoned_reason


@pytest.mark.asyncio
async def test_worker_historical_response_uses_qualified_identity_and_aware_times(
    monkeypatch,
):
    contract = SimpleNamespace(
        symbol="AAPL",
        localSymbol="AAPL",
        conId=265598,
        secType="STK",
        exchange="SMART",
        primaryExchange="NASDAQ",
        currency="USD",
        tradingClass="NMS",
    )
    bar = SimpleNamespace(
        date=datetime(2026, 7, 23, 14, 30, tzinfo=timezone.utc),
        open=100,
        high=101,
        low=99,
        close=100.5,
        volume=10,
        average=100.2,
        barCount=3,
    )

    class FakeIB:
        def isConnected(self):
            return True

        def qualifyContracts(self, requested):
            return [contract]

        def reqCurrentTime(self):
            return datetime(2026, 7, 23, 14, 31, tzinfo=timezone.utc)

        async def reqHistoricalDataAsync(self, requested, **kwargs):
            assert requested is contract
            assert kwargs["formatDate"] == 2
            return [bar]

    monkeypatch.setattr(worker, "ib", FakeIB())
    result = await worker.handle_get_historical_bars({"symbol": "AAPL"})
    assert result["status"] == "success"
    assert result["data"]["qualified_contract"]["con_id"] == 265598
    assert datetime.fromisoformat(result["data"]["broker_timestamp"]).utcoffset() is not None
    assert datetime.fromisoformat(result["data"]["bars"][0]["date"]).utcoffset() is not None


@pytest.mark.asyncio
async def test_worker_rejects_none_qualification_and_naive_broker_time(monkeypatch):
    class NoneQualificationIB:
        def isConnected(self):
            return True

        async def qualifyContractsAsync(self, requested):
            return [None]

    monkeypatch.setattr(worker, "ib", NoneQualificationIB())
    result = await worker.handle_get_historical_bars({"symbol": "AAPL"})
    assert result["status"] == "error"
    assert "exactly one qualified contract" in result["error"]

    class BoolConIdIB:
        def isConnected(self):
            return True

        async def qualifyContractsAsync(self, requested):
            return [SimpleNamespace(conId=True)]

    monkeypatch.setattr(worker, "ib", BoolConIdIB())
    with pytest.raises(ValueError, match="valid conId"):
        await worker._qualify_one_contract(object())
    daily = await worker.handle_get_historical_bars({"symbol": "AAPL", "bar_size": "1 day"})
    assert daily["status"] == "error"
    assert "only intraday datetime" in daily["error"]

    assert (
        datetime.fromisoformat(_valid_historical_data()["broker_timestamp"]).utcoffset() is not None
    )
    with pytest.raises(ValueError, match="timezone-naive"):
        worker._aware_iso(datetime(2026, 7, 23, 14, 30))


@pytest.mark.asyncio
async def test_repeated_connect_is_idempotent_and_conflicting_identity_is_rejected():
    def handler(process, request):
        if request["command"] == "ping":
            data = {
                "pong": True,
                "connected": True,
                "gateway_api_down": False,
                "detail": "",
            }
        else:
            assert request["command"] == "connect"
            data = {
                "connected": True,
                "accounts": ["DU123"],
                "client_id": request["params"]["client_id"],
                "server_version": 180,
            }
        _feed(
            process,
            _response(
                request,
                data=data,
            ),
        )

    client, process, _ = _attach_client(handler)

    async def no_zombies(port):
        return 0, "none"

    client._check_zombie_connections = no_zombies
    results = await asyncio.gather(
        client.connect(port=4002, client_id=7, readonly=True),
        client.connect(port=4002, client_id=7, readonly=True),
    )
    assert results == [True, True]
    assert len(process.stdin.writes) == 2

    with pytest.raises(IBKRConnectionConflictError, match="stop\\(\\).*start"):
        await client.connect(port=4002, client_id=8, readonly=True)
    assert len(process.stdin.writes) == 3


@pytest.mark.asyncio
async def test_disconnected_ping_drives_health_failure_and_clean_reconnect():
    state = {"connected": False}
    commands = []

    def handler(process, request):
        commands.append(request["command"])
        if request["command"] == "ping":
            data = {
                "pong": True,
                "connected": state["connected"],
                "gateway_api_down": False,
                "detail": "",
            }
        elif request["command"] == "connect":
            state["connected"] = True
            data = {
                "connected": True,
                "accounts": ["DU123"],
                "client_id": request["params"]["client_id"],
                "server_version": 180,
            }
        else:
            raise AssertionError(f"unexpected command: {request['command']}")
        _feed(process, _response(request, data=data))

    client, _, _ = _attach_client(handler)
    client._connected = True
    client._connection_identity = ("127.0.0.1", 4002, 7, True)
    client._connection_start_time = datetime.now(timezone.utc)
    health = ConnectionHealth(client, max_consecutive_failures=1)

    assert await health.perform_check() is HealthStatus.UNHEALTHY
    assert client.is_connected is False
    assert client._connection_identity is None
    assert client._connection_start_time is None

    async def no_zombies(port):
        return 0, "none"

    client._check_zombie_connections = no_zombies
    assert await client.connect(port=4002, client_id=7, readonly=True) is True
    assert await health.perform_check() is HealthStatus.HEALTHY
    assert commands == ["ping", "connect", "ping"]


@pytest.mark.asyncio
async def test_connect_probes_and_replaces_stale_cached_session():
    state = {"connected": False}
    commands = []

    def handler(process, request):
        commands.append(request["command"])
        if request["command"] == "ping":
            data = {
                "pong": True,
                "connected": state["connected"],
                "gateway_api_down": False,
                "detail": "",
            }
        elif request["command"] == "connect":
            state["connected"] = True
            data = {
                "connected": True,
                "accounts": ["DU123"],
                "client_id": request["params"]["client_id"],
                "server_version": 180,
            }
        else:
            raise AssertionError(f"unexpected command: {request['command']}")
        _feed(process, _response(request, data=data))

    client, _, _ = _attach_client(handler)
    client._connected = True
    client._connection_identity = ("127.0.0.1", 4002, 7, True)

    async def no_zombies(port):
        return 0, "none"

    client._check_zombie_connections = no_zombies
    assert await client.connect(port=4002, client_id=7, readonly=True) is True
    assert commands == ["ping", "connect"]


@pytest.mark.asyncio
async def test_worker_refuses_active_connect_and_cleans_stale_instance(monkeypatch):
    class Existing:
        def __init__(self, connected):
            self.connected = connected

        def isConnected(self):
            return self.connected

    monkeypatch.setattr(worker, "ib", Existing(True))
    active = await worker.handle_connect({})
    assert active["status"] == "error"
    assert "already has an active" in active["error"]

    stale = Existing(False)
    disconnected = []
    monkeypatch.setattr(worker, "ib", stale)
    monkeypatch.setattr(worker, "safe_disconnect", lambda value: disconnected.append(value))
    monkeypatch.setattr(
        worker,
        "IB",
        lambda: (_ for _ in ()).throw(RuntimeError("new worker sentinel")),
    )
    replacement = await worker.handle_connect({})
    assert replacement["status"] == "error"
    assert disconnected == [stale]


@pytest.mark.asyncio
async def test_timeout_is_restarted_before_robust_retry(monkeypatch):
    from robo_trader.utils import robust_connection

    instances = []

    class FakeClient:
        def __init__(self):
            self.process = None
            self.starts = 0
            self.connects = 0
            self.connected = False
            instances.append(self)

        async def start(self):
            self.starts += 1
            self.process = SimpleNamespace(pid=1000 + self.starts)

        async def stop(self):
            self.connected = False
            self.process = None

        async def connect(self, **kwargs):
            self.connects += 1
            if self.connects == 1:
                raise IBKRTimeoutError("generation timed out")
            self.connected = True
            return True

        async def get_accounts(self):
            return ["DU123"]

        @property
        def is_connected(self):
            return self.connected

    class NoopFileLock:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(client_module, "SubprocessIBKRClient", FakeClient)
    monkeypatch.setattr(robust_connection, "_ConnectFileLock", NoopFileLock)
    monkeypatch.setattr(
        robust_connection,
        "check_tws_zombie_connections",
        lambda port: (0, "none"),
    )
    monkeypatch.setattr(
        robust_connection.RobustConnectionManager,
        "_calculate_backoff_delay",
        lambda self, attempt: 0,
    )

    result = await robust_connection.connect_ibkr_robust_subprocess(
        port=4002, client_id=7, max_retries=2
    )
    assert result is instances[0]
    assert instances[0].starts == 2
    assert instances[0].connects == 2


@pytest.mark.asyncio
async def test_historical_transport_rejects_alias_identity_and_daily_bars():
    bad_data = _valid_historical_data()
    bad_data["qualified_contract"]["local_symbol"] = "AAPL ALIAS"

    def handler(process, request):
        _feed(process, _response(request, data=bad_data))

    client, process, generation = _attach_client(handler)
    with pytest.raises(IBKRTransportPoisonedError, match="alias"):
        await client.get_historical_bars("AAPL")
    assert generation.poisoned_reason

    fresh, fresh_process, _ = _attach_client(handler, "fresh")
    with pytest.raises(ValueError, match="only intraday datetime"):
        await fresh.get_historical_bars("AAPL", bar_size="1 day")
    assert not fresh_process.stdin.writes

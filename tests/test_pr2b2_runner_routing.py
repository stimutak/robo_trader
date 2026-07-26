"""Focused routing tests for the runner's central paper-order choke point."""

from __future__ import annotations

import asyncio
import inspect
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.execution import ExecutionResult, Order, PaperExecutor
from robo_trader.paper_reduction_gateway import (
    PaperReductionGateway,
    PaperReductionGatewayError,
)
from robo_trader.runner_async import AsyncRunner


class _EntrySerializationProbe(AbstractAsyncContextManager):
    def __init__(self) -> None:
        self.active = False
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        assert self.active is False
        self.active = True
        self.enter_count += 1
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        assert self.active is True
        self.active = False
        self.exit_count += 1


def _exact_gateway(
    *,
    started: bool = True,
    reduction_result: ExecutionResult | None = None,
) -> tuple[PaperReductionGateway, _EntrySerializationProbe]:
    """Build only the exact gateway surface consumed by the runner seam."""

    gateway = object.__new__(PaperReductionGateway)
    gateway._started = started
    gateway.submit_reduction = AsyncMock(
        return_value=reduction_result or ExecutionResult(True, "gateway reduction filled", 100.0)
    )
    probe = _EntrySerializationProbe()
    gateway.serialize_entry = MagicMock(return_value=probe)
    return gateway, probe


def _runner(
    gateway: object,
    *,
    live_feed: bool = True,
    emergency_block: bool = False,
    kill_switch_block: bool = False,
    kill_switch_reason: str = "Operator kill switch",
    circuit_allows: bool = True,
) -> AsyncRunner:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.portfolio_id = "default"
    runner.paper_reduction_gateway = gateway
    runner._order_admission_lock = asyncio.Lock()
    runner._symbol_cycle_abort_event = asyncio.Event()
    runner._cycle_worker_tasks = set()
    runner._order_admitted_tasks = set()
    runner._kill_switch_log_last = {}
    runner._kill_switch_log_throttle_seconds = 60.0

    now = datetime.now(timezone.utc)
    monotonic_now = 1000.0
    runner._protective_feed_status = (
        {
            "AAPL": {
                "available": True,
                "live_grade": True,
                "source": "live_protective",
            }
        }
        if live_feed
        else {}
    )
    runner.stop_loss_monitor = SimpleNamespace(
        last_prices={"AAPL": 100.0},
        price_event_times={"AAPL": now},
        price_receipt_monotonic={"AAPL": monotonic_now},
        max_price_age_seconds=10.0,
        _utcnow=lambda: now,
        _monotonic=lambda: monotonic_now,
    )

    runner.risk = SimpleNamespace(
        emergency_shutdown_triggered=emergency_block,
    )
    runner.advanced_risk = SimpleNamespace(
        kill_switch=SimpleNamespace(
            triggered=kill_switch_block,
            trigger_reason=kill_switch_reason,
        )
    )
    runner.circuit_breaker = SimpleNamespace(
        can_proceed=AsyncMock(return_value=circuit_allows),
        record_success=AsyncMock(),
        record_failure=AsyncMock(),
    )
    runner.rate_limiter = SimpleNamespace(acquire=AsyncMock())
    runner.executor = SimpleNamespace(
        place_order=MagicMock(return_value=ExecutionResult(True, "entry filled", 100.0))
    )
    runner.monitor = MagicMock()
    runner.monitor.end_timer.return_value = 0.0
    return runner


def _order(side: str) -> Order:
    return Order(
        symbol="AAPL",
        quantity=2,
        side=side,
        price=100.0,
        order_ref=f"routing-{side.lower()}",
    )


def test_persistent_reconnect_reuses_exact_registered_paper_executor() -> None:
    runner = object.__new__(AsyncRunner)
    runner.slippage_bps = 1.5
    runner.use_smart_execution = False
    runner.executor = PaperExecutor(
        slippage_bps=runner.slippage_bps,
        use_smart_execution=runner.use_smart_execution,
    )
    original = runner.executor

    runner._initialize_or_reuse_paper_executor()

    assert runner.executor is original


def test_runner_accepts_exact_absolute_binding_for_relative_database_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    runner = object.__new__(AsyncRunner)
    runner._raw_db = AsyncTradingDatabase(Path("paper-ledger.db"))
    runner.cfg = SimpleNamespace(
        runtime_contract=SimpleNamespace(database_path=str(tmp_path / "paper-ledger.db"))
    )

    runner._assert_runtime_ledger_database()

    runner.cfg = SimpleNamespace(
        runtime_contract=SimpleNamespace(database_path=str(tmp_path / "other-ledger.db"))
    )
    with pytest.raises(RuntimeError, match="validated runtime ledger"):
        runner._assert_runtime_ledger_database()


def test_runner_database_fallback_uses_validated_contract_path() -> None:
    setup_source = inspect.getsource(AsyncRunner.setup)

    assert "AsyncTradingDatabase(Path(self.cfg.runtime_contract.database_path))" in setup_source
    assert "else AsyncTradingDatabase()" not in setup_source


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL_SHORT"])
@pytest.mark.parametrize(
    ("blocker", "expected_message"),
    [
        ("kill_switch", "Kill switch active"),
        ("daily_loss", "Daily loss limit exceeded"),
        ("circuit", "Circuit breaker open"),
    ],
)
async def test_entries_remain_blocked_before_gateway_or_executor(
    side: str,
    blocker: str,
    expected_message: str,
) -> None:
    gateway, _ = _exact_gateway()
    runner = _runner(
        gateway,
        kill_switch_block=blocker in {"kill_switch", "daily_loss"},
        kill_switch_reason=(
            "Daily loss limit exceeded" if blocker == "daily_loss" else "Operator kill switch"
        ),
        circuit_allows=blocker != "circuit",
    )

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert expected_message in result.message
    gateway.submit_reduction.assert_not_awaited()
    gateway.serialize_entry.assert_not_called()
    runner.executor.place_order.assert_not_called()
    runner.rate_limiter.acquire.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
async def test_reductions_bypass_entry_soft_blocks_through_gateway_once(
    side: str,
) -> None:
    expected = ExecutionResult(True, "gateway reduction filled", 99.5)
    gateway, _ = _exact_gateway(reduction_result=expected)
    runner = _runner(
        gateway,
        emergency_block=True,
        kill_switch_block=True,
        kill_switch_reason="Daily loss limit exceeded",
        circuit_allows=False,
    )

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result is expected
    gateway.submit_reduction.assert_awaited_once_with(
        order=_order(side),
        portfolio_id="default",
    )
    gateway.serialize_entry.assert_not_called()
    runner.circuit_breaker.can_proceed.assert_not_awaited()
    runner.rate_limiter.acquire.assert_not_awaited()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
@pytest.mark.parametrize("gateway_kind", ["missing", "wrong_type", "stopped"])
async def test_reductions_require_exact_started_account_gateway(
    side: str,
    gateway_kind: str,
) -> None:
    if gateway_kind == "missing":
        gateway: object = None
    elif gateway_kind == "wrong_type":
        gateway = SimpleNamespace(
            started=True,
            submit_reduction=AsyncMock(),
        )
    else:
        gateway, _ = _exact_gateway(started=False)
    runner = _runner(gateway)

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert "safety gateway unavailable" in result.message.lower()
    submit = getattr(gateway, "submit_reduction", None)
    if submit is not None:
        submit.assert_not_awaited()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
async def test_reductions_require_live_protective_feed(side: str) -> None:
    gateway, _ = _exact_gateway()
    runner = _runner(gateway, live_feed=False)

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert "live protective feed unavailable" in result.message.lower()
    gateway.submit_reduction.assert_not_awaited()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_reduction_gateway_failure_fails_closed() -> None:
    gateway, _ = _exact_gateway()
    gateway.submit_reduction.side_effect = PaperReductionGatewayError(
        "final evidence no longer authorizes reduction"
    )
    runner = _runner(gateway)

    with pytest.raises(
        PaperReductionGatewayError,
        match="final evidence no longer authorizes reduction",
    ):
        await runner._place_order_with_circuit_breaker(_order("SELL"))

    gateway.submit_reduction.assert_awaited_once()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["SELL", "BUY_TO_COVER"])
async def test_transport_abort_blocks_reduction_before_gateway(side: str) -> None:
    gateway, _ = _exact_gateway()
    runner = _runner(gateway)
    runner._symbol_cycle_abort_event.set()

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert "broker transport unavailable" in result.message.lower()
    gateway.submit_reduction.assert_not_awaited()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL_SHORT"])
async def test_entry_dispatch_occurs_inside_gateway_serialization(
    side: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr("robo_trader.runner_async.is_extended_hours", lambda: False)
    gateway, probe = _exact_gateway()
    runner = _runner(gateway)

    def place_order(order: Order) -> ExecutionResult:
        assert order == _order(side)
        assert probe.active is True
        return ExecutionResult(True, "entry filled", 100.0)

    runner.executor.place_order.side_effect = place_order

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is True
    assert probe.active is False
    assert probe.enter_count == 1
    assert probe.exit_count == 1
    gateway.serialize_entry.assert_called_once_with()
    gateway.submit_reduction.assert_not_awaited()
    runner.executor.place_order.assert_called_once_with(_order(side))


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL_SHORT"])
@pytest.mark.parametrize("gateway_kind", ["missing", "wrong_type", "stopped"])
async def test_entries_require_exact_started_account_gateway(
    side: str,
    gateway_kind: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr("robo_trader.runner_async.is_extended_hours", lambda: False)
    if gateway_kind == "missing":
        gateway: object = None
    elif gateway_kind == "wrong_type":
        gateway = SimpleNamespace(
            started=True,
            serialize_entry=MagicMock(),
        )
    else:
        gateway, _ = _exact_gateway(started=False)
    runner = _runner(gateway)

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert "safety gateway unavailable" in result.message.lower()
    serialize_entry = getattr(gateway, "serialize_entry", None)
    if serialize_entry is not None:
        serialize_entry.assert_not_called()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["HOLD", "CANCEL", ""])
async def test_unsupported_sides_reject_before_any_dispatch(side: str) -> None:
    gateway, _ = _exact_gateway()
    runner = _runner(gateway)

    result = await runner._place_order_with_circuit_breaker(_order(side))

    assert result.ok is False
    assert result.message == "Unsupported order side"
    gateway.submit_reduction.assert_not_awaited()
    gateway.serialize_entry.assert_not_called()
    runner.circuit_breaker.can_proceed.assert_not_awaited()
    runner.executor.place_order.assert_not_called()

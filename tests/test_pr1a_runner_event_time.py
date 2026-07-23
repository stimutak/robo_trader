"""Fail-closed event-time and symbol-cycle containment for PR 1A."""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from robo_trader.clients.subprocess_ibkr_client import IBKRTimeoutError
from robo_trader.execution import Order
from robo_trader.monitoring.performance import PerformanceMonitor
from robo_trader.runner_async import (
    AsyncRunner,
    MarketDataContractError,
    SymbolCycleAbortError,
    SymbolResult,
)


def _bars(dates: list[object], *, symbol: str | None = None) -> pd.DataFrame:
    size = len(dates)
    data: dict[str, object] = {
        "date": dates,
        "open": [100.0] * size,
        "high": [102.0] * size,
        "low": [99.0] * size,
        "close": [101.0] * size,
        "volume": [1000] * size,
    }
    if symbol is not None:
        data["symbol"] = [symbol] * size
    return pd.DataFrame(data)


def test_normalizes_aware_dates_to_utc_datetime_index() -> None:
    now = pd.Timestamp("2026-07-23T15:02:00Z")
    frame = _bars(
        [
            "2026-07-23T11:00:00-04:00",
            "2026-07-23T11:01:00-04:00",
        ]
    )

    result = AsyncRunner._normalize_broker_bars("AAPL", frame, "1 min", now=now)

    assert isinstance(result.index, pd.DatetimeIndex)
    assert str(result.index.tz) == "UTC"
    assert result.index.name == "timestamp"
    assert "date" not in result.columns
    assert result.index.tolist() == [
        pd.Timestamp("2026-07-23T15:00:00Z"),
        pd.Timestamp("2026-07-23T15:01:00Z"),
    ]


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-07-23 11:00:00", "2026-07-23 11:01:00"],  # naive
        ["2026-07-23T15:00:00Z", "2026-07-23 11:01:00"],  # mixed aware/naive
        ["2026-11-01 01:30:00"],  # ambiguous DST and naive
        [pd.NaT],
        ["2026-07-23T15:00:00Z", "2026-07-23T15:00:00Z"],  # duplicate
        ["2026-07-23T15:01:00Z", "2026-07-23T15:00:00Z"],  # reversed
        ["2026-07-23"],  # daily date has no explicit timezone
    ],
)
def test_rejects_ambiguous_or_non_monotonic_dates(dates: list[object]) -> None:
    with pytest.raises(MarketDataContractError):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            _bars(dates),
            "1 min",
            now=pd.Timestamp("2026-07-23T15:02:00Z"),
        )


def test_rejects_range_index_without_date_and_stale_bars() -> None:
    no_dates = _bars(["2026-07-23T15:00:00Z"]).drop(columns=["date"])
    with pytest.raises(MarketDataContractError, match="RangeIndex"):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            no_dates,
            "1 min",
            now=pd.Timestamp("2026-07-23T15:01:00Z"),
        )

    with pytest.raises(MarketDataContractError, match="stale"):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            _bars(["2026-07-23T14:00:00Z"]),
            "1 min",
            now=pd.Timestamp("2026-07-23T15:00:00Z"),
        )


def test_rejects_non_finite_ohlcv() -> None:
    frame = _bars(["2026-07-23T15:00:00Z"])
    frame.loc[0, "close"] = float("inf")
    with pytest.raises(MarketDataContractError, match="invalid"):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            frame,
            "1 min",
            now=pd.Timestamp("2026-07-23T15:01:00Z"),
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda frame: frame.assign(date=["2026-07-23T15:01:00.000001Z"]),
            "future",
        ),
        (lambda frame: frame.assign(symbol=["MSFT"]), "identity"),
        (lambda frame: frame.assign(high=[98.0]), "ordering"),
        (lambda frame: frame.assign(volume=[float("inf")]), "invalid"),
    ],
)
def test_rejects_future_identity_and_value_integrity_failures(mutate, match: str) -> None:
    frame = mutate(_bars(["2026-07-23T15:01:00Z"], symbol="AAPL"))
    with pytest.raises(MarketDataContractError, match=match):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            frame,
            "1 min",
            now=pd.Timestamp("2026-07-23T15:01:00Z"),
        )


def test_rejects_date_column_index_disagreement() -> None:
    frame = _bars(["2026-07-23T15:00:00Z"])
    frame.index = pd.DatetimeIndex(["2026-07-23T14:59:00Z"])
    with pytest.raises(MarketDataContractError, match="disagree"):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            frame,
            "1 min",
            now=pd.Timestamp("2026-07-23T15:01:00Z"),
        )


@pytest.mark.parametrize(
    "bar_size",
    ["1 day", "2 days", "1 D", "1D", "1 week", "1 month", "24 hours"],
)
def test_active_runtime_rejects_daily_or_coarser_bars(bar_size: str) -> None:
    with pytest.raises(MarketDataContractError, match="intraday"):
        AsyncRunner._normalize_broker_bars(
            "AAPL",
            _bars(["2026-07-23T00:00:00Z"]),
            bar_size,
            now=pd.Timestamp("2026-07-23T00:01:00Z"),
        )


def _runner_for_fetch(fetch_side_effect: BaseException) -> AsyncRunner:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = SimpleNamespace(is_connected=True)
    runner.health = MagicMock()
    runner.duration = "2 D"
    runner.bar_size = "1 min"
    runner.production_monitor = None
    runner.monitor = PerformanceMonitor()
    runner.db = MagicMock()
    runner.db.batch_store_market_data = AsyncMock()
    runner._fetch_historical_bars = AsyncMock(side_effect=fetch_side_effect)
    return runner


@pytest.mark.parametrize(
    "qualified",
    [[], [None], [SimpleNamespace(conId=0)], [SimpleNamespace(conId=1)] * 2],
)
@pytest.mark.parametrize("async_api", [False, True])
@pytest.mark.asyncio
async def test_legacy_contract_qualification_must_be_unique_and_valid(
    qualified: list[object],
    async_api: bool,
) -> None:
    class LegacyIB:
        def isConnected(self) -> bool:
            return True

        def reqHistoricalData(self, *_args, **_kwargs):
            raise AssertionError("historical request must not receive ambiguous contract")

    ib = LegacyIB()
    if async_api:
        ib.qualifyContractsAsync = AsyncMock(return_value=qualified)
    else:
        ib.qualifyContracts = MagicMock(return_value=qualified)
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.ib = ib

    with pytest.raises(MarketDataContractError):
        await runner._fetch_historical_bars("AAPL")


def _qualified_contract(**overrides):
    values = {
        "conId": 123,
        "symbol": "AAPL",
        "localSymbol": "AAPL",
        "secType": "STK",
        "currency": "USD",
        "exchange": "SMART",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_legacy_contract_requires_full_stock_identity() -> None:
    valid = _qualified_contract()
    assert AsyncRunner._validate_qualified_contract("AAPL", [valid]) is valid

    for invalid in (
        _qualified_contract(conId=True),
        _qualified_contract(symbol="MSFT"),
        _qualified_contract(localSymbol="MSFT"),
        _qualified_contract(secType="OPT"),
        _qualified_contract(currency="EUR"),
        _qualified_contract(exchange="NYSE"),
    ):
        with pytest.raises(MarketDataContractError):
            AsyncRunner._validate_qualified_contract("AAPL", [invalid])


@pytest.mark.asyncio
async def test_bad_event_time_has_no_downstream_side_effects() -> None:
    runner = _runner_for_fetch(MarketDataContractError("naive timestamp"))
    runner.market_data_cache = OrderedDict()
    runner.max_cache_size = 10
    runner.latest_prices = {}
    runner.latest_price_times = {}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.update_price = AsyncMock()
    runner.stop_loss_monitor.execute_stop_loss = AsyncMock()
    runner.use_advanced_risk = True
    runner.advanced_risk = MagicMock()
    runner.ml_enhanced_strategy = MagicMock()
    runner.ml_enhanced_strategy.analyze = AsyncMock()

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        result = await runner.process_symbol("AAPL")

    assert result.executed is False
    assert result.message == "No data available"
    runner.db.batch_store_market_data.assert_not_awaited()
    assert runner.market_data_cache == {}
    assert runner.latest_prices == {}
    assert runner.latest_price_times == {}
    runner.advanced_risk.update_market_prices.assert_not_called()
    runner.stop_loss_monitor.update_price.assert_not_awaited()
    runner.stop_loss_monitor.execute_stop_loss.assert_not_awaited()
    runner.ml_enhanced_strategy.analyze.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bar_age_seconds", [5, 11, 30, 59])
@pytest.mark.parametrize("monitor_enabled", [True, False])
async def test_historical_bar_is_observational_and_always_blocks_trading(
    bar_age_seconds: int,
    monitor_enabled: bool,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    timestamp = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=bar_age_seconds)
    frame = pd.DataFrame(
        {"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1]},
        index=pd.DatetimeIndex([timestamp], name="timestamp"),
    )
    runner.fetch_and_store_data = AsyncMock(return_value=frame)
    runner.market_data_cache = OrderedDict()
    runner.max_cache_size = 10
    runner.latest_prices = {}
    runner.latest_price_times = {}
    runner.stop_loss_monitor = MagicMock() if monitor_enabled else None
    if runner.stop_loss_monitor:
        runner.stop_loss_monitor.update_price = AsyncMock(return_value=False)
    runner.use_advanced_risk = True
    runner.advanced_risk = MagicMock()
    runner.use_ml_enhanced = True
    runner.ml_enhanced_strategy = MagicMock()
    runner.ml_enhanced_strategy.analyze = AsyncMock()
    runner.executor = MagicMock()

    result = await runner.process_symbol("AAPL")

    assert result.executed is False
    assert result.message.startswith("Protective feed unavailable")
    assert runner._protective_feed_status["AAPL"]["available"] is False
    assert runner._protective_feed_status["AAPL"]["source"] == "historical_bar"
    assert runner._protective_feed_status["AAPL"]["live_grade"] is False
    assert runner.market_data_cache["AAPL"] is frame
    assert runner.latest_prices == {"AAPL": 101.0}
    assert runner.latest_price_sources == {"AAPL": "historical_bar"}
    runner.advanced_risk.update_market_prices.assert_called_once_with({"AAPL": 101.0})
    if runner.stop_loss_monitor:
        runner.stop_loss_monitor.update_price.assert_not_awaited()
    runner.ml_enhanced_strategy.analyze.assert_not_awaited()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_cross_bar_anomaly_has_zero_downstream_side_effects() -> None:
    runner = _runner_for_fetch(MarketDataContractError("unused"))
    now = pd.Timestamp.now(tz="UTC")
    runner._fetch_historical_bars = AsyncMock(
        return_value=pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 10000.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [100, 100],
            },
            index=pd.DatetimeIndex(
                [now - pd.Timedelta(minutes=1), now],
                name="timestamp",
            ),
        )
    )
    runner._trusted_bar_closes = {}
    runner.advanced_risk = MagicMock()

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        result = await runner.fetch_and_store_data("AAPL")

    assert result is None
    runner.db.batch_store_market_data.assert_not_awaited()
    runner.advanced_risk.update_market_prices.assert_not_called()
    assert runner._trusted_bar_closes == {}


@pytest.mark.parametrize("split_factor", [2.0, 3.0, 4.0])
def test_common_split_ratio_vs_prior_trusted_price_is_quarantined(
    split_factor: float,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._trusted_bar_closes = {"AAPL": 100.0}
    split_price = 100.0 / split_factor
    frame = pd.DataFrame(
        {
            "open": [split_price],
            "high": [split_price],
            "low": [split_price],
            "close": [split_price],
        },
        index=pd.DatetimeIndex([pd.Timestamp.now(tz="UTC")]),
    )
    with pytest.raises(MarketDataContractError, match="latest bar OHLC anomaly"):
        runner._validate_cross_bar_anomaly("AAPL", frame)


def test_intrabar_spread_has_only_the_mathematically_reachable_upper_bound() -> None:
    source = inspect.getsource(AsyncRunner._validate_cross_bar_anomaly)

    assert "intrabar_spread >= limit" in source
    assert "intrabar_spread <= reciprocal" not in source


@pytest.mark.asyncio
async def test_process_symbol_rejects_naive_latest_timestamp_before_state_mutation() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.fetch_and_store_data = AsyncMock(
        return_value=pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "volume": [100],
            },
            index=pd.DatetimeIndex(["2026-07-23 15:00:00"]),
        )
    )
    runner.market_data_cache = OrderedDict()
    runner.latest_prices = {}
    runner.latest_price_times = {}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.update_price = AsyncMock()

    with pytest.raises(MarketDataContractError, match="timezone-naive latest timestamp"):
        await runner.process_symbol("AAPL")

    assert runner.market_data_cache == {}
    assert runner.latest_prices == {}
    assert runner.latest_price_times == {}
    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_process_symbol_normalizes_aware_latest_timestamp_to_utc() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    frame = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.0],
            "volume": [100],
        },
        index=pd.DatetimeIndex(["2026-07-23T11:00:00-04:00"]),
    )
    runner.fetch_and_store_data = AsyncMock(return_value=frame)
    runner.market_data_cache = OrderedDict()
    runner.max_cache_size = 10
    runner.latest_prices = {}
    runner.latest_price_times = {}
    runner.stop_loss_monitor = MagicMock()
    runner.stop_loss_monitor.update_price = AsyncMock()
    runner.use_advanced_risk = False

    result = await runner.process_symbol("AAPL")

    assert result.executed is False
    assert runner.latest_price_times["AAPL"] == datetime(2026, 7, 23, 15, 0, tzinfo=timezone.utc)
    assert runner._protective_feed_status["AAPL"]["source_timestamp"] == (
        "2026-07-23T15:00:00+00:00"
    )
    runner.stop_loss_monitor.update_price.assert_not_awaited()


@pytest.mark.asyncio
async def test_transport_timeout_reports_health_and_aborts() -> None:
    error = IBKRTimeoutError("generation poisoned after timeout")
    runner = _runner_for_fetch(error)

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        with pytest.raises(SymbolCycleAbortError):
            await runner.fetch_and_store_data("AAPL")

    runner.health.record_failure.assert_called_once_with(error, "fetch_and_store_data")
    runner.db.batch_store_market_data.assert_not_awaited()


def _runner_for_parallel(process_symbol) -> AsyncRunner:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner._cycle_executed_buys_lock = asyncio.Lock()
    runner._cycle_executed_buys = set()
    runner._cycle_executed_shorts_lock = asyncio.Lock()
    runner._cycle_executed_shorts = set()
    runner.max_concurrent_symbols = 1
    runner.process_symbol = process_symbol
    runner.monitor = MagicMock()
    runner.update_position_market_prices = AsyncMock()
    return runner


def _configure_order_runtime(runner: AsyncRunner, executor_result=None) -> None:
    runner._order_admission_lock = asyncio.Lock()
    runner._symbol_cycle_abort_event = asyncio.Event()
    runner._cycle_worker_tasks = set()
    runner._order_admitted_tasks = set()
    runner._kill_switch_log_last = {}
    runner._kill_switch_log_throttle_seconds = 60
    runner._protective_feed_status = {
        symbol: {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
        }
        for symbol in ("AAPL", "MSFT", "TSLA", "NVDA")
    }
    runner.risk = MagicMock(emergency_shutdown_triggered=False)
    runner.advanced_risk = None
    runner.circuit_breaker = MagicMock()
    runner.circuit_breaker.can_proceed = AsyncMock(return_value=True)
    runner.circuit_breaker.record_success = AsyncMock()
    runner.circuit_breaker.record_failure = AsyncMock()
    runner.rate_limiter = MagicMock()
    runner.rate_limiter.acquire = AsyncMock()
    runner.executor = MagicMock()
    runner.executor.place_order.return_value = executor_result or SimpleNamespace(
        ok=True,
        fill_price=100.0,
        msg="filled",
        message="filled",
    )
    runner.monitor = PerformanceMonitor()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"])
@pytest.mark.parametrize(
    "status",
    [
        None,
        {"available": False, "live_grade": False, "source": "historical_bar"},
        {"available": True, "live_grade": False, "source": "live_protective"},
        {"available": True, "live_grade": True, "source": "historical_bar"},
    ],
)
async def test_central_order_admission_requires_exact_live_protection(
    side: str,
    status: dict | None,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _configure_order_runtime(runner)
    runner._protective_feed_status = {} if status is None else {"AAPL": status}

    result = await runner._place_order_with_circuit_breaker(
        Order(symbol="AAPL", quantity=1, side=side, price=100.0)
    )

    assert result.ok is False
    assert "live protective feed unavailable" in result.message.lower()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"])
async def test_central_order_admission_accepts_exact_live_protection(side: str) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _configure_order_runtime(runner)

    result = await runner._place_order_with_circuit_breaker(
        Order(symbol="AAPL", quantity=1, side=side, price=100.0)
    )

    assert result.ok is True
    runner.executor.place_order.assert_called_once()


def test_pairs_historical_cache_cannot_mutate_strategy_state() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.pairs_strategy = MagicMock()
    runner._protective_feed_status = {
        "AAPL": {
            "available": False,
            "live_grade": False,
            "source": "historical_bar",
        },
        "MSFT": {
            "available": False,
            "live_grade": False,
            "source": "historical_bar",
        },
    }

    admitted = runner._pairs_execution_admitted(("AAPL", "MSFT"))

    assert admitted is False
    runner.pairs_strategy.update_position.assert_not_called()


def test_pairs_live_protection_still_cannot_bypass_atomic_lifecycle_quarantine() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.pairs_strategy = MagicMock()
    runner._protective_feed_status = {
        symbol: {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
        }
        for symbol in ("AAPL", "MSFT")
    }
    pair = ("AAPL", "MSFT")

    admitted = runner._pairs_execution_admitted(pair)

    assert admitted is False
    runner.pairs_strategy.update_position.assert_not_called()


@pytest.mark.asyncio
async def test_transport_abort_cancels_remaining_symbols() -> None:
    reached_second = False

    async def process(symbol: str):
        nonlocal reached_second
        if symbol == "AAPL":
            raise SymbolCycleAbortError("poisoned generation")
        reached_second = True
        raise AssertionError("remaining symbol must not run")

    runner = _runner_for_parallel(process)
    results = await runner.run_parallel(["AAPL", "MSFT"])

    assert results == []
    assert reached_second is False
    runner.update_position_market_prices.assert_awaited_once_with({})


@pytest.mark.asyncio
async def test_empty_symbol_cycle_is_a_safe_no_op() -> None:
    runner = _runner_for_parallel(AsyncMock())
    assert await runner.run_parallel([]) == []
    runner.process_symbol.assert_not_awaited()
    runner.update_position_market_prices.assert_awaited_once_with({})


@pytest.mark.asyncio
async def test_cancelling_run_parallel_cancels_and_awaits_children() -> None:
    started = asyncio.Event()
    child_cancelled = asyncio.Event()
    execution_reached = False

    async def process(_symbol: str):
        nonlocal execution_reached
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise
        execution_reached = True

    runner = _runner_for_parallel(process)
    parent = asyncio.create_task(runner.run_parallel(["AAPL"]))
    await asyncio.wait_for(started.wait(), timeout=1)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent

    assert child_cancelled.is_set()
    assert execution_reached is False
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_transport_abort_drains_running_sibling_and_blocks_its_order() -> None:
    sibling_started = asyncio.Event()
    sibling_settled = asyncio.Event()

    async def process(symbol: str):
        if symbol == "AAPL":
            await sibling_started.wait()
            raise SymbolCycleAbortError("poisoned generation")
        sibling_started.set()
        while not runner._symbol_cycle_abort_event.is_set():
            await asyncio.sleep(0)
        result = await runner._place_order_with_circuit_breaker(
            Order(symbol="MSFT", quantity=1, side="BUY", price=100.0)
        )
        assert result.ok is False
        sibling_settled.set()
        return SymbolResult("MSFT", 0, 100.0, 0, False, result.message)

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    _configure_order_runtime(runner)
    results = await runner.run_parallel(["AAPL", "MSFT"])

    assert len(results) == 1
    assert sibling_settled.is_set()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_abort_before_order_admission_has_no_fill() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _configure_order_runtime(runner)
    await runner._order_admission_lock.acquire()
    abort_task = asyncio.create_task(runner._latch_symbol_cycle_abort())
    await asyncio.sleep(0)
    order_task = asyncio.create_task(
        runner._place_order_with_circuit_breaker(
            Order(symbol="AAPL", quantity=1, side="BUY", price=100.0)
        )
    )
    runner._order_admission_lock.release()

    await abort_task
    result = await order_task
    assert result.ok is False
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_fill_then_sibling_abort_drains_accounting() -> None:
    fill_happened = asyncio.Event()
    accounting_started = asyncio.Event()
    release_accounting = asyncio.Event()
    accounting = AsyncMock()

    async def process(symbol: str):
        if symbol == "MSFT":
            await fill_happened.wait()
            raise SymbolCycleAbortError("poisoned generation")
        result = await runner._place_order_with_circuit_breaker(
            Order(symbol="AAPL", quantity=1, side="BUY", price=100.0)
        )
        assert result.ok
        assert runner._symbol_cycle_abort_event.is_set()
        accounting_started.set()
        await release_accounting.wait()
        await accounting("AAPL", result.fill_price)
        return SymbolResult("AAPL", 1, 100.0, 1, True, "accounted")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    _configure_order_runtime(runner)

    def fill(_order):
        fill_happened.set()
        return SimpleNamespace(ok=True, fill_price=100.0, msg="filled", message="filled")

    runner.executor.place_order.side_effect = fill

    async def hold_after_fill_until_abort():
        await runner._symbol_cycle_abort_event.wait()

    runner.circuit_breaker.record_success = AsyncMock(side_effect=hold_after_fill_until_abort)
    cycle = asyncio.create_task(runner.run_parallel(["AAPL", "MSFT"]))
    await accounting_started.wait()
    assert not cycle.done()
    release_accounting.set()
    results = await cycle

    assert [result.symbol for result in results] == ["AAPL"]
    runner.executor.place_order.assert_called_once()
    accounting.assert_awaited_once_with("AAPL", 100.0)


@pytest.mark.asyncio
async def test_parent_cancel_after_fill_waits_for_accounting() -> None:
    fill_happened = asyncio.Event()
    release_accounting = asyncio.Event()
    accounting_done = asyncio.Event()

    async def process(_symbol: str):
        result = await runner._place_order_with_circuit_breaker(
            Order(symbol="AAPL", quantity=1, side="BUY", price=100.0)
        )
        fill_happened.set()
        await release_accounting.wait()
        accounting_done.set()
        return SymbolResult("AAPL", 1, 100.0, 1, result.ok, "accounted")

    runner = _runner_for_parallel(process)
    _configure_order_runtime(runner)
    parent = asyncio.create_task(runner.run_parallel(["AAPL"]))
    await fill_happened.wait()
    parent.cancel()
    await asyncio.sleep(0)
    assert not parent.done()
    parent.cancel()
    await asyncio.sleep(0)
    assert not parent.done()
    release_accounting.set()
    with pytest.raises(asyncio.CancelledError):
        await parent
    assert accounting_done.is_set()


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_admission_does_not_leak_cycle_lock() -> None:
    runner = _runner_for_parallel(AsyncMock())
    _configure_order_runtime(runner)
    await runner._order_admission_lock.acquire()
    cycle = asyncio.create_task(runner.run_parallel(["AAPL"]))
    await asyncio.sleep(0)
    cycle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cycle
    assert runner._run_parallel_lock.locked() is False
    runner._order_admission_lock.release()


@pytest.mark.asyncio
async def test_next_cycle_resets_abort_only_after_prior_drain() -> None:
    calls = 0

    async def process(symbol: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SymbolCycleAbortError("first cycle poisoned")
        return SymbolResult(symbol, 0, 100.0, 0, False, "healthy next cycle")

    runner = _runner_for_parallel(process)
    assert await runner.run_parallel(["AAPL"]) == []
    second = await runner.run_parallel(["AAPL"])
    assert len(second) == 1
    assert second[0].message == "healthy next cycle"


@pytest.mark.asyncio
async def test_extended_hours_blocks_entry_sides_but_not_exit() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _configure_order_runtime(runner)
    with patch("robo_trader.runner_async.is_extended_hours", return_value=True):
        for side in ("BUY", "SELL_SHORT"):
            result = await runner._place_order_with_circuit_breaker(
                Order(symbol="AAPL", quantity=1, side=side, price=100.0)
            )
            assert result.ok is False
        exit_result = await runner._place_order_with_circuit_breaker(
            Order(symbol="AAPL", quantity=1, side="SELL", price=100.0)
        )
    assert exit_result.ok is True
    runner.executor.place_order.assert_called_once()


@pytest.mark.asyncio
async def test_regular_hours_entry_reaches_executor() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    _configure_order_runtime(runner)
    with patch("robo_trader.runner_async.is_extended_hours", return_value=False):
        result = await runner._place_order_with_circuit_breaker(
            Order(symbol="AAPL", quantity=1, side="BUY", price=100.0)
        )
    assert result.ok is True
    runner.executor.place_order.assert_called_once()

"""Fail-closed event-time and symbol-cycle containment for PR 1A."""

from __future__ import annotations

import asyncio
import inspect
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

import pandas as pd
import pytest

import robo_trader.runner_async as runner_module
from robo_trader.clients.subprocess_ibkr_client import IBKRTimeoutError
from robo_trader.connection_health import HealthStatus
from robo_trader.execution import ExecutionResult, Order
from robo_trader.market_data_contract import (
    AdjustmentState,
    BarTimestampSemantics,
    BrokerProtectiveQuote,
    CanonicalBar,
    CanonicalBarBatch,
    HistoricalBarContract,
    MarketDataSource,
    MarketSession,
    MarketSessionPolicy,
)
from robo_trader.monitoring.performance import PerformanceMonitor
from robo_trader.paper_reduction_gateway import PaperReductionGateway
from robo_trader.protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    assert_producer_owned_protective_quote,
)
from robo_trader.runner_async import (
    AsyncRunner,
    MarketDataContractError,
    RecoverableBrokerDisconnectError,
    SymbolCycleAbortError,
    SymbolResult,
    run_continuous,
)
from robo_trader.reconciliation.runtime_integration import RuntimeReconciliationController
from robo_trader.stop_loss_monitor import StopLossMonitor


@pytest.fixture(autouse=True)
def _regular_hours_default():
    """Keep order-admission tests independent of the host wall clock.

    Tests for extended-hours behavior override this patch explicitly.
    """

    with patch("robo_trader.runner_async.is_extended_hours", return_value=False):
        yield


@pytest.fixture
def continuous_safety_args(monkeypatch):
    class TestCoordinator:
        started = True

    class TestRuntimeContext:
        pass

    resources = SimpleNamespace(
        database=object(),
        gateway=object(),
        reconciliation=SimpleNamespace(
            reconcile_periodic_if_due=AsyncMock(return_value=None),
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "SafetyRuntimeCoordinator",
        TestCoordinator,
    )
    monkeypatch.setattr(
        runner_module,
        "RuntimeSafetyContext",
        TestRuntimeContext,
    )
    monkeypatch.setattr(
        runner_module,
        "_start_paper_order_runtime",
        AsyncMock(return_value=resources),
    )
    monkeypatch.setattr(
        runner_module,
        "_close_paper_order_runtime",
        AsyncMock(),
    )
    return {
        "safety_runtime": TestCoordinator(),
        "runtime_context": TestRuntimeContext(),
    }


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
    runner.portfolio_id = "default"
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
async def test_unsealed_historical_frame_never_reaches_trading_state(
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
    assert result.message == "Execution blocked: canonical market-data evidence unavailable"
    assert runner.market_data_cache == {}
    assert runner.latest_prices == {}
    assert runner.latest_price_times == {}
    runner.advanced_risk.update_market_prices.assert_not_called()
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
    assert result.message == "Execution blocked: canonical market-data evidence unavailable"
    assert runner.latest_price_times == {}
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


@pytest.mark.asyncio
async def test_authoritative_pre_fetch_disconnect_is_recoverable_cycle_abort() -> None:
    runner = _runner_for_fetch(AssertionError("broker request must not start"))
    runner.ib.is_connected = False

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        with pytest.raises(RecoverableBrokerDisconnectError):
            await runner.fetch_and_store_data("AAPL")

    runner.health.record_failure.assert_called_once()
    runner._fetch_historical_bars.assert_not_awaited()
    runner.db.batch_store_market_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_with_inflight_broker_request_is_terminal() -> None:
    request_started = asyncio.Event()
    release_request = asyncio.Event()

    async def gated_fetch(**_kwargs):
        request_started.set()
        await release_request.wait()
        raise AssertionError("test request should be cancelled")

    runner = _runner_for_fetch(AssertionError("unused"))
    runner._fetch_historical_bars = AsyncMock(side_effect=gated_fetch)

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        inflight = asyncio.create_task(runner.fetch_and_store_data("AAPL"))
        await request_started.wait()
        runner.ib.is_connected = False
        with pytest.raises(SymbolCycleAbortError) as caught:
            await runner.fetch_and_store_data("MSFT")

    assert not isinstance(caught.value, RecoverableBrokerDisconnectError)
    assert "another broker request was active" in str(caught.value)
    inflight.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inflight
    assert runner._active_cycle_broker_requests == set()


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


async def _configure_order_runtime(runner: AsyncRunner, executor_result=None) -> None:
    runner.portfolio_id = "default"
    runner._baseline_entry_handle = object()
    runner._baseline_entry_intent = object()
    accepted_at = datetime.now(timezone.utc)
    accepted_monotonic = 1000.0
    protective_clock = {
        "utc": accepted_at,
        "monotonic": accepted_monotonic,
    }
    runner._order_admission_lock = asyncio.Lock()
    runner._symbol_cycle_abort_event = asyncio.Event()
    runner._cycle_worker_tasks = set()
    runner._order_admitted_tasks = set()
    runner._kill_switch_log_last = {}
    runner._kill_switch_log_throttle_seconds = 60
    reconciliation = object.__new__(RuntimeReconciliationController)
    reconciliation.entry_eligible = lambda: True
    runner.reconciliation_controller = reconciliation
    runner._protective_feed_status = {
        symbol: {
            "available": True,
            "live_grade": True,
            "source": "live_protective",
            "con_id": 265_000 + index,
            "transport_generation": "test-generation-1",
        }
        for index, symbol in enumerate(("AAPL", "MSFT", "TSLA", "NVDA"), start=1)
    }
    runner._test_protective_clock = protective_clock
    runner.stop_loss_monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=SimpleNamespace(),
        portfolio_id="default",
    )
    runner.stop_loss_monitor._utcnow = lambda: protective_clock["utc"]
    runner.stop_loss_monitor._monotonic = lambda: protective_clock["monotonic"]
    for index, symbol in enumerate(("AAPL", "MSFT", "TSLA", "NVDA"), start=1):
        assert await runner.stop_loss_monitor.update_price(
            symbol,
            100.0,
            source_timestamp=accepted_at,
            source=ProtectiveQuoteSource.LIVE_BROKER,
            con_id=265_000 + index,
            transport_generation="test-generation-1",
            source_event_id=f"event-{index}",
        )
    runner.risk = MagicMock(emergency_shutdown_triggered=False)
    runner.risk.validate_order.return_value = (True, "ok")
    runner.portfolio = MagicMock()
    runner.portfolio.equity = AsyncMock(return_value=Decimal("100000"))
    runner.positions = {}
    runner.latest_prices = {"AAPL": 100.0}
    runner.latest_price_times = {"AAPL": accepted_at}
    runner.latest_price_sources = {"AAPL": "live_protective"}
    runner.daily_pnl = 0.0
    runner.daily_executed_notional = 0.0
    contract = HistoricalBarContract(
        schema_version=1,
        symbol="AAPL",
        con_id=265001,
        exchange="SMART",
        primary_exchange="NASDAQ",
        timezone_name="UTC",
        timeframe="1 min",
        session_policy=MarketSessionPolicy.REGULAR_ONLY,
        source=MarketDataSource.IBKR_HISTORICAL_TRADES,
        retrieval_time=accepted_at,
        broker_time=accepted_at,
        adjustment_state=AdjustmentState.RAW,
        transport_generation="test-generation-1",
        timestamp_semantics=BarTimestampSemantics.BAR_START,
        use_rth=True,
        what_to_show="TRADES",
    )
    batch = CanonicalBarBatch(
        contract=contract,
        bars=(
            CanonicalBar(
                contract=contract,
                timestamp=accepted_at - timedelta(seconds=30),
                open=Decimal("99"),
                high=Decimal("101"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=100,
                session=MarketSession.REGULAR,
            ),
        ),
    )
    frame = batch.to_frame()
    runner._canonical_bar_batches = {"AAPL": (batch, frame)}
    broker_quote = BrokerProtectiveQuote(
        schema_version=1,
        symbol="AAPL",
        con_id=265001,
        exchange="SMART",
        primary_exchange="NASDAQ",
        currency="USD",
        security_type="STK",
        price=Decimal("100.0"),
        source_timestamp=accepted_at,
        retrieval_timestamp=accepted_at,
        session=MarketSession.REGULAR,
        source=MarketDataSource.IBKR_LIVE_LAST_TRADE,
        source_event_id="event-1",
        transport_generation="test-generation-1",
        market_data_type=1,
    )
    runner._broker_protective_quotes = {"AAPL": broker_quote}
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
    gateway = PaperReductionGateway.__new__(PaperReductionGateway)
    gateway._started = True

    @asynccontextmanager
    async def serialize_entry(symbol=None, *, portfolio_id=None):
        assert portfolio_id == "default"
        yield broker_quote if symbol == "AAPL" else None

    gateway.serialize_entry = serialize_entry
    gateway.submit_reduction = AsyncMock(
        return_value=runner.executor.place_order.return_value,
    )
    gateway.submit_baseline_entry = MagicMock(
        return_value=runner.executor.place_order.return_value,
    )
    gateway.issue_baseline_entry_intent = MagicMock(
        return_value=runner._baseline_entry_intent,
    )
    runner.paper_reduction_gateway = gateway
    runner.monitor = PerformanceMonitor()


async def _place_order(runner: AsyncRunner, order: Order):
    return await runner._place_order_with_circuit_breaker(
        order,
        _entry_intent=(runner._baseline_entry_intent if order.side == "BUY" else None),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "BUY_TO_COVER"])
@pytest.mark.parametrize(
    "status",
    [
        None,
        {"available": False, "live_grade": False, "source": "historical_bar"},
        {"available": True, "live_grade": False, "source": "live_protective"},
        {"available": True, "live_grade": True, "source": "historical_bar"},
    ],
)
async def test_dashboard_status_cannot_revoke_exact_live_protection(
    side: str,
    status: dict | None,
) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    runner._protective_feed_status = {} if status is None else {"AAPL": status}

    result = await _place_order(
        runner,
        Order(
            symbol="AAPL",
            quantity=1,
            side=side,
            price=100.0,
        ),
    )

    assert result.ok is True
    if side == "BUY":
        runner.paper_reduction_gateway.submit_baseline_entry.assert_called_once()
        runner.executor.place_order.assert_not_called()
    else:
        runner.paper_reduction_gateway.submit_reduction.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["BUY", "SELL", "BUY_TO_COVER"])
async def test_central_order_admission_accepts_exact_live_protection(side: str) -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)

    result = await _place_order(
        runner,
        Order(
            symbol="AAPL",
            quantity=1,
            side=side,
            price=100.0,
        ),
    )

    assert result.ok is True
    if side in {"SELL", "BUY_TO_COVER"}:
        runner.paper_reduction_gateway.submit_reduction.assert_awaited_once()
        submitted = runner.paper_reduction_gateway.submit_reduction.await_args.kwargs
        assert submitted["order"].price is None
        quote = submitted["protective_quote"]
        assert type(quote) is ProtectiveQuoteEvidence
        assert quote.source is ProtectiveQuoteSource.LIVE_BROKER
        assert (
            assert_producer_owned_protective_quote(
                quote,
                producer=runner.stop_loss_monitor,
            )
            is quote
        )
        runner.executor.place_order.assert_not_called()
    else:
        runner.paper_reduction_gateway.submit_baseline_entry.assert_called_once()
        runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("aged_clock", ["event", "receipt"])
@pytest.mark.parametrize("side", ["BUY", "SELL", "BUY_TO_COVER"])
async def test_order_admission_rechecks_monitor_owned_protective_freshness(
    aged_clock: str,
    side: str,
) -> None:
    """The current central feed gate applies to exits as well as entries."""
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    assert runner._has_live_protective_feed("AAPL") is True

    if aged_clock == "event":
        runner._test_protective_clock["utc"] += timedelta(seconds=11)
    else:
        runner._test_protective_clock["monotonic"] += 11

    result = await _place_order(
        runner,
        Order(
            symbol="AAPL",
            quantity=1,
            side=side,
            price=100.0,
        ),
    )

    assert runner._has_live_protective_feed("AAPL") is False
    assert result.ok is False
    assert any(
        phrase in result.message.lower()
        for phrase in ("live protective feed unavailable", "not authoritative")
    )
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_entry_quote_aging_during_equity_await_never_touches_executor() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)

    async def age_quote_during_equity(_prices):
        runner._test_protective_clock["monotonic"] += 11
        return Decimal("100000")

    runner.portfolio.equity = AsyncMock(side_effect=age_quote_during_equity)

    result = await _place_order(
        runner,
        Order(
            symbol="AAPL",
            quantity=1,
            side="BUY",
            price=100.0,
        ),
    )

    assert result.ok is False
    assert result.message == "Entry blocked: protective quote expired during final admission"
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_entry_fill_accounting_uses_exact_fills_for_cumulative_risk_and_marks() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.daily_executed_notional = 0.0
    runner.positions = {
        "AAPL": SimpleNamespace(quantity=2, avg_price=Decimal("400")),
        "MSFT": SimpleNamespace(quantity=-3, avg_price=Decimal("350")),
    }
    runner.db = SimpleNamespace(
        record_trade=AsyncMock(),
        update_position=AsyncMock(),
    )
    buy_fill = runner._exact_entry_fill_price(
        ExecutionResult(
            True,
            "filled",
            100.0,
            exact_fill_price=Decimal("400"),
        )
    )
    short_fill = runner._exact_entry_fill_price(
        ExecutionResult(
            True,
            "filled",
            100.0,
            exact_fill_price=Decimal("350"),
        )
    )

    await runner._record_entry_fill_accounting(
        symbol="AAPL",
        side="BUY",
        quantity=2,
        fill_price=buy_fill,
        strategy_reference_price=100.0,
    )
    await runner._record_entry_fill_accounting(
        symbol="MSFT",
        side="SELL_SHORT",
        quantity=3,
        fill_price=short_fill,
        strategy_reference_price=100.0,
    )

    assert runner.daily_executed_notional == 1850.0
    assert runner.daily_executed_notional > 500.0
    assert runner.db.update_position.await_args_list == [
        call("AAPL", 2, Decimal("400"), Decimal("400")),
        call("MSFT", -3, Decimal("350"), Decimal("350")),
    ]


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
    with pytest.raises(SymbolCycleAbortError, match="poisoned generation"):
        await runner.run_parallel(["AAPL", "MSFT"])

    assert reached_second is False
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_recoverable_disconnect_aborts_parallel_cycle_before_downstream() -> None:
    reached_second = False

    async def process(symbol: str):
        nonlocal reached_second
        if symbol == "AAPL":
            raise RecoverableBrokerDisconnectError("disconnected before request")
        reached_second = True
        raise AssertionError("remaining symbol must not run")

    runner = _runner_for_parallel(process)
    with pytest.raises(
        RecoverableBrokerDisconnectError,
        match="disconnected before request",
    ):
        await runner.run_parallel(["AAPL", "MSFT"])

    assert reached_second is False
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_abort_dominates_simultaneous_recoverable_disconnect() -> None:
    both_ready = asyncio.Event()
    ready = 0

    async def process(symbol: str):
        nonlocal ready
        ready += 1
        if ready == 2:
            both_ready.set()
        await both_ready.wait()
        if symbol == "AAPL":
            raise RecoverableBrokerDisconnectError("plain disconnect")
        raise SymbolCycleAbortError("request identity became ambiguous")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2

    with pytest.raises(SymbolCycleAbortError, match="request identity became ambiguous") as caught:
        await runner.run_parallel(["AAPL", "MSFT"])

    assert not isinstance(caught.value, RecoverableBrokerDisconnectError)
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_terminal_abort_discovered_during_drain_overrides_recoverable() -> None:
    terminal_task_admitted = asyncio.Event()

    async def process(symbol: str):
        if symbol == "AAPL":
            await terminal_task_admitted.wait()
            raise RecoverableBrokerDisconnectError("early recoverable disconnect")

        current_task = asyncio.current_task()
        assert current_task is not None
        runner._order_admitted_tasks.add(current_task)
        terminal_task_admitted.set()
        await asyncio.sleep(0.05)
        raise SymbolCycleAbortError("late terminal ambiguity")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2

    with pytest.raises(SymbolCycleAbortError, match="late terminal ambiguity") as caught:
        await runner.run_parallel(["AAPL", "MSFT"])

    assert not isinstance(caught.value, RecoverableBrokerDisconnectError)
    assert isinstance(runner._symbol_cycle_abort_error, SymbolCycleAbortError)
    assert not isinstance(
        runner._symbol_cycle_abort_error,
        RecoverableBrokerDisconnectError,
    )
    runner.update_position_market_prices.assert_not_awaited()


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
        result = await _place_order(
            runner,
            Order(
                symbol="MSFT",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
        assert result.ok is False
        sibling_settled.set()
        return SymbolResult("MSFT", 0, 100.0, 0, False, result.message)

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    await _configure_order_runtime(runner)
    with pytest.raises(SymbolCycleAbortError, match="poisoned generation"):
        await runner.run_parallel(["AAPL", "MSFT"])

    assert sibling_settled.is_set()
    runner.executor.place_order.assert_not_called()
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_transport_abort_cancels_hung_non_admitted_sibling() -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def process(symbol: str):
        if symbol == "AAPL":
            await sibling_started.wait()
            raise SymbolCycleAbortError("originating identity poison")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2

    with pytest.raises(SymbolCycleAbortError, match="originating identity poison"):
        await asyncio.wait_for(
            runner.run_parallel(["AAPL", "MSFT"]),
            timeout=1,
        )

    assert sibling_cancelled.is_set()
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_transport_abort_preserves_first_causal_error() -> None:
    secondary_latched = asyncio.Event()

    async def process(symbol: str):
        if symbol == "MSFT":
            raise SymbolCycleAbortError("originating identity poison")
        await runner._symbol_cycle_abort_event.wait()
        raise SymbolCycleAbortError("secondary already-poisoned observation")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    original_latch = runner._latch_symbol_cycle_abort

    async def latch_in_causal_order(cause=None):
        await original_latch(cause)
        if cause is not None and "originating" in str(cause):
            await secondary_latched.wait()
        elif cause is not None:
            secondary_latched.set()

    runner._latch_symbol_cycle_abort = latch_in_causal_order

    with pytest.raises(SymbolCycleAbortError, match="originating identity poison"):
        await runner.run_parallel(["AAPL", "MSFT"])

    assert secondary_latched.is_set()
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_abort_before_order_admission_has_no_fill() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    await runner._order_admission_lock.acquire()
    abort_task = asyncio.create_task(runner._latch_symbol_cycle_abort())
    await asyncio.sleep(0)
    order_task = asyncio.create_task(
        _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
    )
    runner._order_admission_lock.release()

    await abort_task
    result = await order_task
    assert result.ok is False
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_queued_order_cannot_beat_abort_intent_to_admission_lock() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    await runner._order_admission_lock.acquire()

    # Queue the order first so it is the lock's oldest waiter.
    order_task = asyncio.create_task(
        _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
    )
    await asyncio.sleep(0)
    abort_task = asyncio.create_task(
        runner._latch_symbol_cycle_abort(
            SymbolCycleAbortError("transport failed before latch admission")
        )
    )
    await asyncio.sleep(0)

    # Abort intent is visible without acquiring the held admission lock.
    assert runner._symbol_cycle_abort_event.is_set()
    runner._order_admission_lock.release()

    result = await order_task
    await abort_task
    assert result.ok is False
    assert "broker transport unavailable" in result.message.lower()
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
        result = await _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
        assert result.ok
        assert runner._symbol_cycle_abort_event.is_set()
        accounting_started.set()
        await release_accounting.wait()
        await accounting("AAPL", result.fill_price)
        return SymbolResult("AAPL", 1, 100.0, 1, True, "accounted")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    await _configure_order_runtime(runner)

    def fill(*, order, portfolio_id, intent):
        assert order.symbol == "AAPL"
        assert portfolio_id == "default"
        assert intent is runner._baseline_entry_intent
        fill_happened.set()
        return SimpleNamespace(ok=True, fill_price=100.0, msg="filled", message="filled")

    runner.paper_reduction_gateway.submit_baseline_entry.side_effect = fill

    async def hold_after_fill_until_abort():
        await runner._symbol_cycle_abort_event.wait()

    runner.circuit_breaker.record_success = AsyncMock(side_effect=hold_after_fill_until_abort)
    cycle = asyncio.create_task(runner.run_parallel(["AAPL", "MSFT"]))
    await accounting_started.wait()
    assert not cycle.done()
    release_accounting.set()
    with pytest.raises(SymbolCycleAbortError, match="poisoned generation"):
        await cycle

    runner.paper_reduction_gateway.submit_baseline_entry.assert_called_once()
    runner.executor.place_order.assert_not_called()
    accounting.assert_awaited_once_with("AAPL", 100.0)
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_parent_cancel_after_fill_waits_for_accounting() -> None:
    fill_happened = asyncio.Event()
    release_accounting = asyncio.Event()
    accounting_done = asyncio.Event()

    async def process(_symbol: str):
        result = await _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
        fill_happened.set()
        await release_accounting.wait()
        accounting_done.set()
        return SymbolResult("AAPL", 1, 100.0, 1, result.ok, "accounted")

    runner = _runner_for_parallel(process)
    await _configure_order_runtime(runner)
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
async def test_parent_cancel_during_abort_drain_preserves_cancellation() -> None:
    fill_happened = asyncio.Event()
    release_accounting = asyncio.Event()
    accounting_done = asyncio.Event()

    async def process(symbol: str):
        if symbol == "MSFT":
            await fill_happened.wait()
            raise SymbolCycleAbortError("originating transport poison")
        result = await _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
        fill_happened.set()
        await release_accounting.wait()
        accounting_done.set()
        return SymbolResult("AAPL", 1, 100.0, 1, result.ok, "accounted")

    runner = _runner_for_parallel(process)
    runner.max_concurrent_symbols = 2
    await _configure_order_runtime(runner)
    parent = asyncio.create_task(runner.run_parallel(["AAPL", "MSFT"]))

    await fill_happened.wait()
    while not any(
        task.get_name() == "symbol-cycle-accounting-drain" for task in asyncio.all_tasks()
    ):
        await asyncio.sleep(0)

    parent.cancel()
    await asyncio.sleep(0)
    assert not parent.done()
    release_accounting.set()

    with pytest.raises(asyncio.CancelledError):
        await parent
    assert parent.cancelled() is True
    assert accounting_done.is_set()
    runner.update_position_market_prices.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_admission_does_not_leak_cycle_lock() -> None:
    runner = _runner_for_parallel(AsyncMock())
    await _configure_order_runtime(runner)
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
    with pytest.raises(SymbolCycleAbortError, match="first cycle poisoned"):
        await runner.run_parallel(["AAPL"])
    second = await runner.run_parallel(["AAPL"])
    assert len(second) == 1
    assert second[0].message == "healthy next cycle"


@pytest.mark.asyncio
async def test_symbol_cycle_abort_suppresses_all_run_level_downstream_work() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock()
    runner._activate_after_setup = MagicMock()
    runner.teardown = AsyncMock()
    runner.positions = {}
    runner.cfg = SimpleNamespace(symbols=["AAPL"])
    runner.max_concurrent_symbols = 1
    runner.ai_analyst = None
    runner.run_parallel = AsyncMock(side_effect=SymbolCycleAbortError("shared transport poisoned"))
    runner.market_data_cache = OrderedDict(
        {
            "AAPL": pd.DataFrame({"close": [100.0]}),
            "MSFT": pd.DataFrame({"close": [200.0]}),
        }
    )
    runner.pairs_strategy = MagicMock()
    runner.pairs_strategy.pair_stats = {("AAPL", "MSFT"): object()}
    runner.pairs_strategy.analyze_pairs = AsyncMock()
    runner.stat_arb_strategy = MagicMock()
    runner.stat_arb_strategy.calculate_arbitrage_scores = AsyncMock()
    runner.update_account_summary = AsyncMock()
    runner.monitor = MagicMock()
    runner.monitor.log_performance_summary = AsyncMock()
    runner.use_correlation_sizing = False

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        with pytest.raises(SymbolCycleAbortError, match="shared transport poisoned"):
            await runner.run(["AAPL"])

    runner.pairs_strategy.analyze_pairs.assert_not_awaited()
    runner.stat_arb_strategy.calculate_arbitrage_scores.assert_not_awaited()
    runner.update_account_summary.assert_not_awaited()
    runner.monitor.log_performance_summary.assert_not_awaited()
    runner.teardown.assert_awaited_once()


@pytest.mark.asyncio
async def test_recoverable_disconnect_suppresses_all_run_level_downstream_work() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    runner.setup = AsyncMock()
    runner._ensure_health_monitor_for_activation = AsyncMock()
    runner._activate_after_setup = MagicMock()
    runner.teardown = AsyncMock()
    runner.positions = {}
    runner.cfg = SimpleNamespace(symbols=["AAPL"])
    runner.max_concurrent_symbols = 1
    runner.ai_analyst = None
    runner.run_parallel = AsyncMock(
        side_effect=RecoverableBrokerDisconnectError("plain disconnect")
    )
    runner.market_data_cache = OrderedDict()
    runner.pairs_strategy = MagicMock()
    runner.pairs_strategy.analyze_pairs = AsyncMock()
    runner.stat_arb_strategy = MagicMock()
    runner.stat_arb_strategy.calculate_arbitrage_scores = AsyncMock()
    runner.update_account_summary = AsyncMock()
    runner.monitor = MagicMock()
    runner.monitor.log_performance_summary = AsyncMock()
    runner.use_correlation_sizing = False

    with patch("robo_trader.runner_async.is_trading_allowed", return_value=True):
        with pytest.raises(RecoverableBrokerDisconnectError, match="plain disconnect"):
            await runner.run(["AAPL"])

    runner.pairs_strategy.analyze_pairs.assert_not_awaited()
    runner.stat_arb_strategy.calculate_arbitrage_scores.assert_not_awaited()
    runner.update_account_summary.assert_not_awaited()
    runner.monitor.log_performance_summary.assert_not_awaited()
    runner.teardown.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_authoritative_disconnect_recovers_without_watchdog_exit(
    continuous_safety_args,
) -> None:
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = MagicMock()
    runner.health.status = HealthStatus.HEALTHY
    runner.health._status = HealthStatus.HEALTHY
    runner.run = AsyncMock(
        side_effect=RecoverableBrokerDisconnectError("disconnected before request")
    )
    runner.recover_connection = AsyncMock(return_value=True)
    runner._safe_disconnect = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock()
    portfolio = SimpleNamespace(
        id="default",
        name="Default",
        starting_cash=100000,
        symbols=["AAPL"],
        active=True,
    )

    with (
        patch("signal.signal"),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
        patch("robo_trader.runner_async._setup_continuous_runner", new_callable=AsyncMock),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=[portfolio],
        ),
        patch(
            "robo_trader.runner_async.sleep_unless_shutdown",
            new_callable=AsyncMock,
            side_effect=asyncio.CancelledError,
        ),
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        await run_continuous(
            symbols=["AAPL"],
            interval_seconds=1,
            **continuous_safety_args,
        )

    runner.run.assert_awaited_once_with(["AAPL"])
    runner.recover_connection.assert_awaited_once()
    runner._safe_disconnect.assert_not_awaited()
    assert runner._recovery_exhausted is False
    # AsyncRunner.run owns its cycle-finalization hook. The continuous loop
    # must not invoke it a second time after a recovered cycle failure.
    runner.teardown.assert_not_awaited()
    runner.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_authoritative_disconnect_exits_when_recovery_exhausts(
    continuous_safety_args,
) -> None:
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = MagicMock()
    runner.health.status = HealthStatus.HEALTHY
    runner.health._status = HealthStatus.HEALTHY
    runner.run = AsyncMock(
        side_effect=RecoverableBrokerDisconnectError("disconnected before request")
    )
    runner.recover_connection = AsyncMock(return_value=False)
    runner._safe_disconnect = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock()
    portfolio = SimpleNamespace(
        id="default",
        name="Default",
        starting_cash=100000,
        symbols=["AAPL"],
        active=True,
    )

    with (
        patch("signal.signal"),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
        patch("robo_trader.runner_async._setup_continuous_runner", new_callable=AsyncMock),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=[portfolio],
        ),
        patch(
            "robo_trader.runner_async.sleep_unless_shutdown",
            new_callable=AsyncMock,
        ) as sleep,
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        await run_continuous(
            symbols=["AAPL"],
            interval_seconds=1,
            **continuous_safety_args,
        )

    runner.run.assert_awaited_once_with(["AAPL"])
    runner.recover_connection.assert_awaited_once()
    assert runner._recovery_exhausted is True
    runner._safe_disconnect.assert_not_awaited()
    runner.teardown.assert_not_awaited()
    sleep.assert_not_awaited()
    runner.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_authoritative_disconnect_exits_when_recovery_raises(
    continuous_safety_args,
) -> None:
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = MagicMock()
    runner.health.status = HealthStatus.HEALTHY
    runner.health._status = HealthStatus.HEALTHY
    runner.run = AsyncMock(
        side_effect=RecoverableBrokerDisconnectError("disconnected before request")
    )
    runner.recover_connection = AsyncMock(side_effect=RuntimeError("recovery infrastructure"))
    runner._safe_disconnect = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock()
    portfolio = SimpleNamespace(
        id="default",
        name="Default",
        starting_cash=100000,
        symbols=["AAPL"],
        active=True,
    )

    with (
        patch("signal.signal"),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
        patch("robo_trader.runner_async._setup_continuous_runner", new_callable=AsyncMock),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=[portfolio],
        ),
        patch(
            "robo_trader.runner_async.sleep_unless_shutdown",
            new_callable=AsyncMock,
        ) as sleep,
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        await run_continuous(
            symbols=["AAPL"],
            interval_seconds=1,
            **continuous_safety_args,
        )

    runner.run.assert_awaited_once_with(["AAPL"])
    runner.recover_connection.assert_awaited_once()
    assert runner._recovery_exhausted is True
    runner._safe_disconnect.assert_not_awaited()
    runner.teardown.assert_not_awaited()
    sleep.assert_not_awaited()
    runner.cleanup.assert_awaited_once()


@pytest.mark.asyncio
async def test_continuous_transport_abort_disconnects_and_exits_without_next_cycle(
    continuous_safety_args,
) -> None:
    runner = MagicMock()
    runner.recovery_in_progress = False
    runner._recovery_exhausted = False
    runner.health = MagicMock()
    runner.health.status = HealthStatus.HEALTHY
    runner.health._status = HealthStatus.HEALTHY
    runner.run = AsyncMock(side_effect=SymbolCycleAbortError("shared transport poisoned"))
    runner._safe_disconnect = AsyncMock()
    runner.teardown = AsyncMock()
    runner.cleanup = AsyncMock()
    portfolio = SimpleNamespace(
        id="default",
        name="Default",
        starting_cash=100000,
        symbols=["AAPL"],
        active=True,
    )

    with (
        patch("signal.signal"),
        patch("robo_trader.runner_async.AsyncRunner", return_value=runner),
        patch("robo_trader.runner_async._setup_continuous_runner", new_callable=AsyncMock),
        patch("robo_trader.runner_async.is_trading_allowed", return_value=True),
        patch(
            "robo_trader.multiuser.portfolio_config.load_portfolio_configs",
            return_value=[portfolio],
        ),
        patch("robo_trader.runner_async.sleep_unless_shutdown", new_callable=AsyncMock) as sleep,
        patch("robo_trader.runner_async._write_exit_audit"),
        patch("robo_trader.runner_async._fire_runner_exit_alert"),
    ):
        await run_continuous(
            symbols=["AAPL"],
            interval_seconds=1,
            **continuous_safety_args,
        )

    runner.run.assert_awaited_once_with(["AAPL"])
    runner._safe_disconnect.assert_awaited_once()
    assert runner.health._status is HealthStatus.UNHEALTHY
    runner.health.record_failure.assert_not_called()
    assert runner._recovery_exhausted is True
    runner.teardown.assert_not_awaited()
    runner.cleanup.assert_awaited_once()
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_extended_hours_blocks_entry_sides_but_not_exit() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    with patch("robo_trader.runner_async.is_extended_hours", return_value=True):
        for side in ("BUY", "SELL_SHORT"):
            result = await _place_order(
                runner,
                Order(
                    symbol="AAPL",
                    quantity=1,
                    side=side,
                    price=100.0,
                ),
            )
            assert result.ok is False
        exit_result = await _place_order(
            runner, Order(symbol="AAPL", quantity=1, side="SELL", price=100.0)
        )
    assert exit_result.ok is True
    runner.paper_reduction_gateway.submit_reduction.assert_awaited_once()
    submitted = runner.paper_reduction_gateway.submit_reduction.await_args.kwargs
    assert submitted["order"].price is None
    quote = submitted["protective_quote"]
    assert quote.source is ProtectiveQuoteSource.LIVE_BROKER
    assert (
        assert_producer_owned_protective_quote(
            quote,
            producer=runner.stop_loss_monitor,
        )
        is quote
    )
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_extended_hours_entry_requires_and_accepts_matching_exact_session() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    regular_batch, _ = runner._canonical_bar_batches["AAPL"]
    extended_contract = replace(
        regular_batch.contract,
        session_policy=MarketSessionPolicy.EXTENDED,
        use_rth=False,
    )
    extended_bar = replace(
        regular_batch.bars[-1],
        contract=extended_contract,
        session=MarketSession.PRE_MARKET,
    )
    extended_batch = CanonicalBarBatch(extended_contract, (extended_bar,))
    extended_frame = extended_batch.to_frame()
    runner._canonical_bar_batches["AAPL"] = (extended_batch, extended_frame)
    quote = replace(
        runner._broker_protective_quotes["AAPL"],
        session=MarketSession.PRE_MARKET,
    )

    @asynccontextmanager
    async def serialize_entry(symbol=None, *, portfolio_id=None):
        assert portfolio_id == "default"
        yield quote if symbol == "AAPL" else None

    runner.paper_reduction_gateway.serialize_entry = serialize_entry
    with (
        patch("robo_trader.runner_async.is_extended_hours", return_value=True),
        patch("robo_trader.runner_async.get_market_session", return_value="pre-market"),
    ):
        result = await _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=Decimal("1"),
            ),
        )

    assert result.ok is True
    submitted = runner.paper_reduction_gateway.submit_baseline_entry.call_args.kwargs["order"]
    assert submitted.price == Decimal("100.0")


@pytest.mark.asyncio
async def test_final_entry_admission_rejects_stale_canonical_bar_batch() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    current_batch, _ = runner._canonical_bar_batches["AAPL"]
    stale_time = datetime.now(timezone.utc) - timedelta(hours=1)
    stale_contract = replace(
        current_batch.contract,
        retrieval_time=stale_time,
        broker_time=stale_time,
    )
    stale_bar = replace(
        current_batch.bars[-1],
        contract=stale_contract,
        timestamp=stale_time - timedelta(minutes=1),
    )
    stale_batch = CanonicalBarBatch(stale_contract, (stale_bar,))
    stale_frame = stale_batch.to_frame()
    runner._canonical_bar_batches["AAPL"] = (stale_batch, stale_frame)

    result = await _place_order(
        runner,
        Order(
            symbol="AAPL",
            quantity=1,
            side="BUY",
            price=Decimal("100"),
        ),
    )

    assert result.ok is False
    assert "canonical session evidence unavailable" in result.message.lower()
    runner.executor.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_regular_hours_entry_reaches_executor() -> None:
    runner = AsyncRunner.__new__(AsyncRunner)
    await _configure_order_runtime(runner)
    with patch("robo_trader.runner_async.is_extended_hours", return_value=False):
        result = await _place_order(
            runner,
            Order(
                symbol="AAPL",
                quantity=1,
                side="BUY",
                price=100.0,
            ),
        )
    assert result.ok is True
    runner.paper_reduction_gateway.submit_baseline_entry.assert_called_once()
    runner.executor.place_order.assert_not_called()

"""Input, phase ordering, and sampling tests for the backtest engine."""

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from robo_trader.backtesting.engine import BacktestEngine
from tests.backtesting.test_engine_accounting import (
    ExactFillSimulator,
    ScriptedStrategy,
    _bars,
)


def _engine(strategy=None, **overrides) -> BacktestEngine:
    return BacktestEngine(
        strategy or ScriptedStrategy([{}]),
        ExactFillSimulator(commission=0),
        initial_capital=1_000,
        finalization_policy="mark_to_market",
        **overrides,
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda frame: frame.iloc[0:0], "must not be empty"),
        (lambda frame: frame.iloc[::-1], "sorted"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]]).sort_index(),
            "unique events",
        ),
        (lambda frame: frame.assign(close=np.nan), "finite"),
        (lambda frame: frame.assign(high=1), "OHLC bounds"),
        (lambda frame: frame.assign(volume=-1), "non-negative"),
    ],
)
def test_strict_market_data_validation(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _engine().run(mutate(_bars([10, 11])))


def test_single_asset_requires_datetime_index_and_exactly_one_symbol() -> None:
    frame = _bars([10, 11]).reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _engine().run(frame)
    with pytest.raises(ValueError, match="exactly one symbol"):
        _engine().run(_bars([10, 11]), symbols=["A", "B"])


@pytest.mark.parametrize("time_first", [False, True])
def test_multi_asset_normalizes_both_index_orientations(time_first: bool) -> None:
    times = pd.date_range("2026-01-05", periods=2, freq="1D")
    tuples = [(symbol, timestamp) for symbol in ("A", "B") for timestamp in times]
    names = ["symbol", "time"]
    if time_first:
        tuples = sorted((timestamp, symbol) for symbol, timestamp in tuples)
        names = ["time", "symbol"]
    index = pd.MultiIndex.from_tuples(tuples, names=names)
    data = pd.DataFrame(
        {
            "open": [10] * 4,
            "high": [11] * 4,
            "low": [9] * 4,
            "close": [10] * 4,
            "volume": [100] * 4,
        },
        index=index,
    )
    strategy = ScriptedStrategy([{}, {}])

    result = _engine(strategy).run(data)

    assert strategy.initialized_with == ["A", "B"]
    assert len(result.equity_curve) == 2


def test_no_signal_can_fill_on_decision_bar_even_at_first_event() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 1}}, {}])
    simulator = ExactFillSimulator(commission=0)
    engine = BacktestEngine(
        strategy,
        simulator,
        initial_capital=1_000,
        finalization_policy="mark_to_market",
    )
    data = _bars([10, 20])

    result = engine.run(data)

    assert simulator.calls[0][4] == data.index[1]
    assert result.positions[0].entry_price == 20


def test_pending_decision_waits_for_that_symbols_next_exact_bar() -> None:
    times = pd.date_range("2026-01-05", periods=3, freq="1D")
    index = pd.MultiIndex.from_tuples(
        [("A", times[0]), ("A", times[2]), ("B", times[1])], names=["symbol", "time"]
    ).sort_values()
    data = pd.DataFrame(
        {
            "open": [10, 12, 20],
            "high": [11, 13, 21],
            "low": [9, 11, 19],
            "close": [10, 12, 20],
            "volume": [100, 100, 100],
        },
        index=index,
    )
    strategy = ScriptedStrategy([{"A": {"action": "buy", "quantity": 1}}, {}, {}])
    simulator = ExactFillSimulator(commission=0)
    engine = BacktestEngine(
        strategy,
        simulator,
        initial_capital=1_000,
        finalization_policy="mark_to_market",
    )

    result = engine.run(data)

    assert simulator.calls[0][4] == times[2]
    assert result.positions[0].entry_price == 12


def test_nonzero_returns_and_intraday_sampling_frequency_are_used() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 1}}, {}, {}])
    result = _engine(strategy).run(_bars([10, 10, 10], [10, 11, 12], frequency="1h"))

    assert result.daily_returns.iloc[-1] != 0
    assert result.sampling_periods_per_year == pytest.approx(252 * 6.5)
    assert result.metrics["sampling_periods_per_year"] == pytest.approx(252 * 6.5)
    assert np.isfinite(result.metrics["sharpe_ratio"])


def test_daily_sampling_and_zero_denominators_produce_finite_metrics() -> None:
    result = _engine(ScriptedStrategy([{}, {}, {}])).run(_bars([10, 10, 10]))

    assert result.sampling_periods_per_year == 252
    for key in ("sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"):
        assert np.isfinite(result.metrics[key])
        assert result.metrics[key] == 0


def test_sortino_uses_downside_root_mean_square_not_negative_subset_variance() -> None:
    engine = _engine()
    engine._sampling_periods_per_year = 252
    engine.equity_curve = [1_000, 990, 980.1]
    engine.daily_returns = [-0.01, -0.01]

    metrics = engine.calculate_metrics()

    assert metrics["sortino_ratio"] == pytest.approx(-np.sqrt(252))


def test_strategy_receives_copies_not_engine_position_ownership() -> None:
    class MutatingStrategy:
        def __init__(self) -> None:
            self.call = 0

        def initialize(self, symbols: List[str]) -> None:
            self.call = 0

        def generate_signals(self, data: pd.DataFrame, positions: Dict) -> Dict:
            self.call += 1
            data.iloc[0, data.columns.get_loc("close")] = 999
            positions.clear()
            if self.call == 1:
                return {"SINGLE": {"action": "buy", "quantity": 1}}
            return {}

    result = _engine(MutatingStrategy()).run(_bars([10, 11]))

    assert result.positions[0].quantity == 1
    assert result.equity_curve.iloc[-1] == pytest.approx(1_000)


def test_intrabar_stop_and_take_profit_use_extremes_stop_first_and_fill_next_bar() -> None:
    class IntrabarExitStrategy:
        def initialize(self, symbols: List[str]) -> None:
            self.call = 0
            self.stop_prices = []
            self.take_profit_prices = []

        def generate_signals(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
            decision = {"SINGLE": {"action": "buy", "quantity": 1}} if self.call == 0 else {}
            self.call += 1
            return decision

        def check_stop_loss(self, _position, price: float) -> bool:
            self.stop_prices.append(price)
            return price <= 90

        def check_take_profit(self, _position, price: float) -> bool:
            self.take_profit_prices.append(price)
            return price >= 110

    index = pd.date_range("2026-01-05 09:30", periods=3, freq="1h")
    data = pd.DataFrame(
        {
            "open": [100, 100, 95],
            "high": [101, 120, 96],
            "low": [99, 80, 94],
            "close": [100, 100, 95],
            "volume": [10_000, 10_000, 10_000],
        },
        index=index,
    )
    strategy = IntrabarExitStrategy()
    simulator = ExactFillSimulator(commission=0)
    engine = BacktestEngine(
        strategy,
        simulator,
        initial_capital=1_000,
        finalization_policy="mark_to_market",
    )

    result = engine.run(data)

    assert strategy.stop_prices[0] == 80
    assert strategy.take_profit_prices[0] == 120
    assert [call[3] for call in simulator.calls] == ["buy", "sell"]
    assert simulator.calls[-1][4] == index[2]
    assert result.positions == []
    assert result.trades[0].exit_price == 95

    # Both thresholds were touched in the completed bar.  The hidden path is
    # ambiguous, so the adverse stop reason must win deterministically.
    strategy.initialize(["SINGLE"])
    direct = BacktestEngine(
        strategy,
        ExactFillSimulator(commission=0),
        initial_capital=1_000,
        finalization_policy="mark_to_market",
    )
    direct._execute_buy("SINGLE", 1, data.iloc[1], index[1])
    direct._update_positions(pd.DataFrame([data.iloc[1]], index=["SINGLE"]), index[1])
    assert direct._pending[0].reason == "risk-stop"


@pytest.mark.parametrize(
    ("frequency", "index", "expected_calls"),
    [
        ("daily", pd.date_range("2026-01-05 09:30", periods=3, freq="1h"), 1),
        (
            "weekly",
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-01-06 09:30"),
                    pd.Timestamp("2026-01-06 10:30"),
                    pd.Timestamp("2026-01-12 09:30"),
                ]
            ),
            2,
        ),
        (
            "monthly",
            pd.DatetimeIndex(
                [
                    pd.Timestamp("2026-02-02 09:30"),
                    pd.Timestamp("2026-02-02 10:30"),
                    pd.Timestamp("2026-03-02 09:30"),
                ]
            ),
            2,
        ),
    ],
)
def test_rebalance_runs_once_per_observed_period(
    frequency: str, index: pd.DatetimeIndex, expected_calls: int
) -> None:
    class TargetStrategy:
        def initialize(self, symbols: List[str]) -> None:
            self.calls = 0

        def get_target_weights(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
            self.calls += 1
            return {}

        def generate_signals(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
            return {}

    data = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10_000},
        index=index,
    )
    strategy = TargetStrategy()

    _engine(strategy, rebalance_frequency=frequency).run(data)

    assert strategy.calls == expected_calls


@pytest.mark.parametrize("target_order", [("B", "A"), ("A", "B")])
def test_rebalance_reductions_fund_increases_independent_of_mapping_order(
    target_order: tuple[str, str],
) -> None:
    class RotationStrategy:
        def initialize(self, symbols: List[str]) -> None:
            self.signal_call = 0
            self.rebalance_call = 0

        def get_target_weights(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
            self.rebalance_call += 1
            if self.rebalance_call == 1:
                return {}
            weights = {"A": 0.0, "B": 1.0}
            return {symbol: weights[symbol] for symbol in target_order}

        def generate_signals(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
            decision = {"A": {"action": "buy", "quantity": 10}} if self.signal_call == 0 else {}
            self.signal_call += 1
            return decision

    timestamps = pd.date_range("2026-01-05", periods=3, freq="1D")
    index = pd.MultiIndex.from_product([["A", "B"], timestamps], names=["symbol", "time"])
    data = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10_000},
        index=index,
    )

    result = _engine(
        RotationStrategy(),
        rebalance_frequency="daily",
    ).run(data)

    assert [(position.symbol, position.quantity) for position in result.positions] == [("B", 10)]


def test_rebalance_rejects_implicit_leverage_and_splits_direction_reversal() -> None:
    engine = _engine()
    timestamp = pd.Timestamp("2026-01-05")
    current = pd.DataFrame(
        {
            "open": [100, 100],
            "high": [101, 101],
            "low": [99, 99],
            "close": [100, 100],
            "volume": [10_000, 10_000],
        },
        index=["A", "B"],
    )
    current.attrs["timestamp"] = timestamp
    with pytest.raises(ValueError, match="unlevered gross exposure"):
        engine._queue_target_weights({"A": 0.6, "B": 0.6}, current, timestamp)

    engine._execute_buy("A", 5, current.loc["A"], timestamp)
    engine._queue_target_weights({"A": -0.5}, current, timestamp)

    assert [
        (order.side, order.remaining_quantity, order.reduce_only) for order in engine._pending
    ] == [
        ("sell", 5, True),
        ("sell", 5, False),
    ]


def test_fail_fast_is_default() -> None:
    strategy = ScriptedStrategy([RuntimeError("fail now")])
    with pytest.raises(RuntimeError, match="fail now"):
        _engine(strategy).run(_bars([10]))

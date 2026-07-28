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


def test_fail_fast_is_default() -> None:
    strategy = ScriptedStrategy([RuntimeError("fail now")])
    with pytest.raises(RuntimeError, match="fail now"):
        _engine(strategy).run(_bars([10]))

"""Accounting invariants for the deterministic backtest engine."""

from dataclasses import replace
from typing import Dict, List, Optional

import pandas as pd
import pytest

from robo_trader.backtesting.engine import BacktestEngine
from robo_trader.backtesting.execution_simulator import (
    ExecutionCost,
    ExecutionSimulator,
    MarketImpactModel,
    SimulatedOrder,
)


def _bars(
    opens: List[float],
    closes: Optional[List[float]] = None,
    *,
    start: str = "2026-01-05",
    frequency: str = "1D",
) -> pd.DataFrame:
    closes = closes or opens
    index = pd.date_range(start, periods=len(opens), freq=frequency)
    return pd.DataFrame(
        {
            "open": opens,
            "high": [max(open_, close) + 1 for open_, close in zip(opens, closes)],
            "low": [min(open_, close) - 1 for open_, close in zip(opens, closes)],
            "close": closes,
            "volume": [10_000] * len(opens),
        },
        index=index,
    )


class ScriptedStrategy:
    def __init__(self, decisions: List[Dict]) -> None:
        self.decisions = decisions
        self.call = 0
        self.initialized_with = None

    def initialize(self, symbols: List[str]) -> None:
        self.call = 0
        self.initialized_with = list(symbols)

    def generate_signals(self, _data: pd.DataFrame, _positions: Dict) -> Dict:
        decision = self.decisions[self.call] if self.call < len(self.decisions) else {}
        self.call += 1
        if isinstance(decision, Exception):
            raise decision
        return decision


class ExactFillSimulator:
    """Minimal deterministic simulator that exposes requested-vs-filled quantity."""

    def __init__(self, fill_limits: Optional[List[int]] = None, commission: float = 1.0) -> None:
        self.fill_limits = list(fill_limits or [])
        self._initial_limits = list(self.fill_limits)
        self.commission = commission
        self.calls = []

    def reset(self) -> None:
        self.fill_limits = list(self._initial_limits)
        self.calls = []

    def simulate_execution(
        self,
        *,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        price_data: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> SimulatedOrder:
        assert list(price_data.index) == [timestamp]
        limit = self.fill_limits.pop(0) if self.fill_limits else quantity
        filled = min(quantity, limit)
        price = float(price_data.loc[timestamp, "open"])
        fee = self.commission if filled else 0.0
        self.calls.append((symbol, quantity, filled, side, timestamp))
        return SimulatedOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=price,
            fill_price=price if filled else 0.0,
            execution_cost=ExecutionCost(0, 0, fee, 0, fee, price if filled else 0),
            filled=filled > 0,
            partial_fill=filled,
            filled_quantity=filled,
            remaining_quantity=quantity - filled,
        )


def _engine(strategy, simulator, **overrides) -> BacktestEngine:
    return BacktestEngine(
        strategy,
        simulator,
        initial_capital=1_000,
        finalization_policy="mark_to_market",
        **overrides,
    )


def test_decision_executes_on_next_bar_and_commission_is_charged_once() -> None:
    strategy = ScriptedStrategy(
        [
            {"SINGLE": {"action": "buy", "quantity": 10}},
            {"SINGLE": {"action": "sell", "quantity": 4}},
            {},
        ]
    )
    simulator = ExactFillSimulator(commission=1)

    result = _engine(strategy, simulator).run(_bars([9, 10, 12], [9, 11, 12]))

    assert [call[4] for call in simulator.calls] == list(result.equity_curve.index[1:])
    assert result.positions[0].quantity == 6
    assert result.trades[0].quantity == 4
    assert result.trades[0].entry_price == 10
    assert result.trades[0].exit_price == 12
    assert result.trades[0].commission == pytest.approx(1.4)
    assert result.trades[0].pnl == pytest.approx(6.6)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_018)


def test_partial_fill_remainder_persists_across_exact_future_bars() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 5}}, {}, {}, {}])
    simulator = ExactFillSimulator(fill_limits=[2, 2, 1], commission=0)

    result = _engine(strategy, simulator).run(_bars([8, 10, 11, 12]))

    assert [(call[1], call[2]) for call in simulator.calls] == [(5, 2), (3, 2), (1, 1)]
    assert result.positions[0].quantity == 5
    assert result.positions[0].entry_price == pytest.approx(10.8)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_006)


def test_partial_fill_parent_order_pays_minimum_commission_only_once() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 3}}, {}, {}, {}])
    simulator = ExecutionSimulator(
        spread_model="fixed",
        commission_per_share=0.005,
        min_commission=1.0,
        market_impact_model=MarketImpactModel(0, 0),
        slippage_factor=0,
        max_volume_participation=1,
    )
    data = _bars([100, 100, 100, 100])
    data["volume"] = 1
    engine = _engine(strategy, simulator)

    result = engine.run(data)

    assert result.positions[0].quantity == 3
    assert sum(float(lot.commission_remaining) for lot in engine._lots["SINGLE"]) == 1.0
    assert result.equity_curve.iloc[-1] == pytest.approx(998.985)


def test_next_open_fill_capacity_uses_only_decision_bar_known_volume() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 100}}, {}])
    simulator = ExecutionSimulator(
        spread_model="fixed",
        commission_per_share=0,
        min_commission=0,
        market_impact_model=MarketImpactModel(0, 0),
        slippage_factor=0,
        max_volume_participation=1,
    )
    data = _bars([100, 100])
    data["volume"] = [1, 10_000]

    result = _engine(strategy, simulator).run(data)

    assert result.positions[0].quantity == 1
    assert result.approval_eligible


def test_short_lots_and_partial_cover_use_fifo_and_signed_marking() -> None:
    strategy = ScriptedStrategy(
        [
            {"SINGLE": {"action": "short", "quantity": 10}},
            {"SINGLE": {"action": "cover", "quantity": 4}},
            {},
        ]
    )

    result = _engine(strategy, ExactFillSimulator()).run(_bars([11, 10, 8]))

    assert result.positions[0].quantity == -6
    assert result.trades[0].trade_type == "short"
    assert result.trades[0].quantity == 4
    assert result.trades[0].pnl == pytest.approx(6.6)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_018)


def test_fill_crossing_zero_closes_long_then_opens_exact_short_remainder() -> None:
    strategy = ScriptedStrategy(
        [
            {"SINGLE": {"action": "buy", "quantity": 3}},
            {"SINGLE": {"action": "short", "quantity": 5}},
            {},
        ]
    )

    result = _engine(strategy, ExactFillSimulator(commission=0)).run(_bars([9, 10, 12]))

    assert result.trades[0].trade_type == "long"
    assert result.trades[0].quantity == 3
    assert result.trades[0].pnl == pytest.approx(6)
    assert result.positions[0].quantity == -2
    assert result.positions[0].entry_price == 12


def test_reduce_only_sell_never_reverses_a_closed_long_into_a_short() -> None:
    strategy = ScriptedStrategy(
        [
            {"SINGLE": {"action": "buy", "quantity": 2}},
            {"SINGLE": {"action": "sell", "quantity": 2}},
            {"SINGLE": {"action": "sell", "quantity": 2}},
            {},
        ]
    )

    result = _engine(strategy, ExactFillSimulator(commission=0)).run(_bars([9, 10, 11, 12]))

    assert result.positions == []
    assert len(result.trades) == 1
    assert result.trades[0].quantity == 2


def test_forced_liquidation_replaces_final_mark_after_fill() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 10}}, {}, {}])
    engine = BacktestEngine(
        strategy,
        ExactFillSimulator(),
        initial_capital=1_000,
        finalization_policy="liquidate",
    )

    result = engine.run(_bars([9, 11, 12], [9, 11, 13]))

    assert result.positions == []
    assert result.trades[0].exit_price == 13
    assert result.trades[0].pnl == pytest.approx(18)
    assert result.equity_curve.iloc[-1] == pytest.approx(1_018)
    assert result.metrics["final_equity"] == pytest.approx(1_018)


def test_each_run_resets_state_and_previous_result_is_detached() -> None:
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 2}}, {}])
    engine = _engine(strategy, ExactFillSimulator(commission=0))
    data = _bars([10, 11])

    first = engine.run(data)
    second = engine.run(data)

    assert first.equity_curve.equals(second.equity_curve)
    assert first.positions == second.positions
    engine.equity_curve[-1] = -1
    assert first.equity_curve.iloc[-1] == pytest.approx(1_000)
    with pytest.raises(Exception):
        replace(first.positions[0], quantity=99).quantity = 1


def test_max_positions_counts_pending_openings() -> None:
    index = pd.date_range("2026-01-05", periods=2, freq="1D")
    rows = []
    for symbol in ("A", "B"):
        for timestamp in index:
            rows.append((symbol, timestamp, 10, 11, 9, 10, 10_000))
    data = pd.DataFrame(
        rows, columns=["symbol", "time", "open", "high", "low", "close", "volume"]
    ).set_index(["symbol", "time"])
    strategy = ScriptedStrategy(
        [{"A": {"action": "buy", "quantity": 1}, "B": {"action": "buy", "quantity": 1}}, {}]
    )

    result = _engine(strategy, ExactFillSimulator(commission=0), max_positions=1).run(data)

    assert [position.symbol for position in result.positions] == ["A"]


def test_dividend_and_split_hooks_adjust_cash_quantity_and_basis() -> None:
    data = _bars([9, 10, 6])
    data["split"] = [1, 1, 2]
    data["dividend"] = [0, 0, 1]
    strategy = ScriptedStrategy([{"SINGLE": {"action": "buy", "quantity": 2}}, {}, {}])

    result = _engine(strategy, ExactFillSimulator()).run(data)

    assert result.positions[0].quantity == 4
    assert result.positions[0].entry_price == 5
    assert result.equity_curve.iloc[-1] == pytest.approx(1_007)


def test_multi_asset_missing_quote_for_held_position_fails_closed() -> None:
    times = pd.date_range("2026-01-05", periods=3, freq="1D")
    index = pd.MultiIndex.from_tuples(
        [("A", times[0]), ("A", times[1]), ("B", times[2])],
        names=["symbol", "time"],
    )
    data = pd.DataFrame(
        {
            "open": [10, 10, 20],
            "high": [11, 11, 21],
            "low": [9, 9, 19],
            "close": [10, 10, 20],
            "volume": [100, 100, 100],
        },
        index=index,
    )
    strategy = ScriptedStrategy([{"A": {"action": "buy", "quantity": 1}}, {}, {}])

    with pytest.raises(RuntimeError, match="missing exact quote"):
        _engine(strategy, ExactFillSimulator(commission=0)).run(data)


def test_recorded_error_aborts_and_is_never_approval_eligible() -> None:
    strategy = ScriptedStrategy([{}, RuntimeError("broken strategy"), {}])
    engine = BacktestEngine(
        strategy,
        ExactFillSimulator(),
        initial_capital=1_000,
        finalization_policy="mark_to_market",
        error_policy="record",
    )

    result = engine.run(_bars([10, 11, 12]))

    assert result.approval_eligible is False
    assert len(result.errors) == 1
    assert "broken strategy" in result.errors[0]
    assert len(result.equity_curve) == 2

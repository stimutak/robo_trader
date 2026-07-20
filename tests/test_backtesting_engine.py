import pandas as pd

from robo_trader.backtesting.engine import BacktestEngine, Position
from robo_trader.backtesting.execution_simulator import ExecutionCost, SimulatedOrder


class NoSignalStrategy:
    def initialize(self, symbols):
        self.symbols = symbols

    def generate_signals(self, current_data, positions):
        return {}


class BuyOnceStopStrategy:
    def __init__(self):
        self.calls = 0

    def initialize(self, symbols):
        self.symbols = symbols

    def generate_signals(self, current_data, positions):
        self.calls += 1
        if self.calls == 1:
            return {"AAPL": {"action": "buy"}}
        return {}

    def check_stop_loss(self, position, current_price):
        return current_price <= position.entry_price * 0.95


class FillAtCloseExecution:
    def simulate_execution(
        self,
        symbol,
        quantity,
        side,
        order_type,
        price_data,
        timestamp,
        limit_price=None,
        stop_price=None,
    ):
        fill_price = float(price_data.iloc[0]["close"])
        return SimulatedOrder(
            symbol=symbol,
            quantity=quantity,
            side=side,
            order_type=order_type,
            timestamp=timestamp,
            requested_price=fill_price,
            fill_price=fill_price,
            execution_cost=ExecutionCost(
                spread_cost=0,
                market_impact=0,
                commission=0,
                slippage=0,
                total_cost=0,
                fill_price=fill_price,
            ),
            filled=True,
            partial_fill=quantity,
        )


def ohlcv(closes, lows=None, highs=None):
    lows = lows or closes
    highs = highs or closes
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1_000_000] * len(closes),
        },
        index=pd.MultiIndex.from_product(
            [["AAPL"], pd.date_range("2026-01-01", periods=len(closes), freq="D")]
        ),
    )


def test_backtest_daily_returns_use_previous_equity_point():
    engine = BacktestEngine(
        strategy=NoSignalStrategy(),
        execution_simulator=FillAtCloseExecution(),
        initial_capital=1000,
    )
    engine.positions["AAPL"] = Position(
        symbol="AAPL",
        quantity=10,
        entry_price=100.0,
        entry_time=pd.Timestamp("2026-01-01"),
    )

    result = engine.run(ohlcv([100.0, 110.0, 99.0]), symbols=["AAPL"])

    assert result.equity_curve.tolist()[:3] == [2000.0, 2100.0, 1990.0]
    assert result.daily_returns.tolist()[:2] == [0.05, -110.0 / 2100.0]


def test_backtest_stop_loss_uses_intrabar_low_for_long_positions():
    engine = BacktestEngine(
        strategy=BuyOnceStopStrategy(),
        execution_simulator=FillAtCloseExecution(),
        initial_capital=1000,
        max_positions=1,
    )

    result = engine.run(
        ohlcv(closes=[100.0, 101.0], lows=[100.0, 94.0], highs=[100.0, 102.0]),
        symbols=["AAPL"],
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_time == pd.Timestamp("2026-01-02")
    assert result.positions[0].is_open is False

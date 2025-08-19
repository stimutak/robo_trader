import pandas as pd

from robo_trader.strategies import sma_crossover_signals


def test_sma_crossover_signals_generates_columns():
    prices = pd.DataFrame({"close": [i for i in range(1, 51)]})
    out = sma_crossover_signals(prices, fast=5, slow=10)
    assert {"sma_fast", "sma_slow", "signal"}.issubset(out.columns)


def test_crossover_signal_positive():
    # Sequence that crosses up around index ~10
    data = [1] * 10 + [2] * 10 + [3] * 30
    df = pd.DataFrame({"close": data})
    out = sma_crossover_signals(df, fast=3, slow=8)
    assert (out["signal"] == 1).any()

from __future__ import annotations

import pandas as pd


def sma_crossover_signals(df: pd.DataFrame, fast: int = 10, slow: int = 20) -> pd.DataFrame:
    """Compute simple SMA crossover signals.

    Returns DataFrame with columns: close, sma_fast, sma_slow, signal
    signal: 1 for long entry, -1 for exit/short entry, 0 otherwise.
    """
    data = df.copy()
    if "close" not in data.columns and "close" not in data:
        # For IB df from util.df(bars), 'close' is present
        raise ValueError("DataFrame must contain 'close' column")
    # Compute with min_periods=1 so early values are available and crossings can be detected
    data["sma_fast"] = data["close"].rolling(fast, min_periods=1).mean()
    data["sma_slow"] = data["close"].rolling(slow, min_periods=1).mean()
    data["signal"] = 0
    # Count equality as potential cross to capture transitions robustly
    cross_up = (data["sma_fast"].shift(1) <= data["sma_slow"].shift(1)) & (data["sma_fast"] > data["sma_slow"])
    cross_down = (data["sma_fast"].shift(1) >= data["sma_slow"].shift(1)) & (data["sma_fast"] < data["sma_slow"])
    data.loc[cross_up, "signal"] = 1
    data.loc[cross_down, "signal"] = -1
    return data


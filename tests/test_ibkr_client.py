import pandas as pd

from robo_trader.clients.async_ibkr_client import normalize_bars_df


def test_normalize_bars_df_basic():
    raw = pd.DataFrame(
        {
            "Date": ["2024-01-01 09:30:00", "2024-01-01 09:35:00"],
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100.5, 101.2],
            "Volume": [1000, 1200],
        }
    )
    out = normalize_bars_df(raw)
    assert {"date", "open", "high", "low", "close", "volume"}.issubset(out.columns)
    assert len(out) == 2
    assert out["close"].dtype.kind in {"f", "i"}


def test_normalize_bars_df_drops_nan_close():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-01 09:30:00", "2024-01-01 09:35:00"],
            "open": [100, 101],
            "high": [101, 102],
            "low": [99, 100],
            "close": [None, 101.2],
            "volume": [1000, 1200],
        }
    )
    out = normalize_bars_df(raw)
    assert len(out) == 1
    assert float(out["close"].iloc[0]) == 101.2

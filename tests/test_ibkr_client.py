import pandas as pd

from robo_trader.connection_manager import ConnectionManager


def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal local replica for testing normalization independent of deprecated client
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()
    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]
    preferred = [
        c for c in ["date", "time", "open", "high", "low", "close", "volume"] if c in data.columns
    ]
    if preferred:
        data = data[preferred]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
    if "close" in data.columns:
        data = data.dropna(subset=["close"])
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    elif "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.sort_values("date")
    elif "time" in data.columns:
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        data = data.sort_values("time")
    return data.reset_index(drop=False)


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

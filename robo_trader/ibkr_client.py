from __future__ import annotations

import asyncio
from typing import Optional

from ib_insync import IB, Stock, util
import pandas as pd


class IBKRClient:
    """Thin async wrapper around ib_insync to standardize connection and data calls.

    This client is created in readonly mode by default. Use a separate executor
    for order placement to enforce risk controls before sending live orders.
    """

    def __init__(self, host: str, port: int, client_id: int) -> None:
        self._host = host
        self._port = port
        self._client_id = client_id
        self.ib = IB()

    async def connect(self, readonly: bool = True, timeout: float = 10.0) -> None:
        if self.ib.isConnected():
            return
        # connectAsync raises on failure; we propagate that to the caller
        await asyncio.wait_for(
            self.ib.connectAsync(self._host, self._port, clientId=self._client_id, readonly=readonly),
            timeout=timeout,
        )

    def qualify_stock(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        contract = Stock(symbol, exchange, currency)
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Unable to qualify contract for {symbol}")
        return qualified[0]

    async def fetch_recent_bars(self, symbol: str, duration: str = "2 D", bar_size: str = "5 mins"):
        """Fetch recent bars as a pandas DataFrame.

        Args:
            symbol: Equity ticker, e.g. "AAPL".
            duration: IB duration string, e.g. "2 D", "30 D".
            bar_size: IB bar size, e.g. "1 min", "5 mins", "1 hour".
        """
        contract = self.qualify_stock(symbol)
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        raw_df = util.df(bars)
        return normalize_bars_df(raw_df)


def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IB historical bars DataFrame for deterministic downstream use.

    - Lowercase column names
    - Keep standard OHLCV columns when present
    - Coerce numeric types where possible
    - Drop rows with NaN in close
    - Sort by index if datetime index, else by a 'date'/'time' column if present
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    data = df.copy()
    data.columns = [str(c).lower() for c in data.columns]

    # Standard column subset if present
    preferred_cols = [c for c in ["date", "time", "open", "high", "low", "close", "volume"] if c in data.columns]
    if preferred_cols:
        data = data[preferred_cols]

    # Coerce numerics
    for col in ["open", "high", "low", "close", "volume"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows without close
    if "close" in data.columns:
        data = data.dropna(subset=["close"])

    # Sort chronologically
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.sort_index()
    elif "date" in data.columns:
        try:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data = data.sort_values("date")
        except Exception:
            pass
    elif "time" in data.columns:
        try:
            data["time"] = pd.to_datetime(data["time"], errors="coerce")
            data = data.sort_values("time")
        except Exception:
            pass

    return data.reset_index(drop=False)


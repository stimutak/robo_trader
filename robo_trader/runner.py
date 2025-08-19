from __future__ import annotations

import asyncio
from typing import Dict

import pandas as pd

from .config import load_config
from .execution import Order, PaperExecutor
from .ibkr_client import IBKRClient
from .risk import Position, RiskManager
from .strategies import sma_crossover_signals


async def run_once() -> None:
    cfg = load_config()
    ib = IBKRClient(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
    await ib.connect(readonly=True)

    # Initialize helpers
    risk = RiskManager(
        max_daily_loss=cfg.max_daily_loss,
        max_position_risk_pct=cfg.max_position_risk_pct,
        max_symbol_exposure_pct=cfg.max_symbol_exposure_pct,
        max_leverage=cfg.max_leverage,
    )
    executor = PaperExecutor()

    equity = cfg.default_cash
    daily_pnl = 0.0
    positions: Dict[str, Position] = {}

    for symbol in cfg.symbols:
        df = await ib.fetch_recent_bars(symbol, duration="10 D", bar_size="30 mins")
        if df.empty:
            continue
        # Normalize columns
        if "close" not in df.columns and "close" in df:
            df = df.rename(columns={"close": "close"})
        signals = sma_crossover_signals(pd.DataFrame({"close": df["close"]}))
        last = signals.iloc[-1]
        price = float(last["close"]) if "close" in last else float(df["close"].iloc[-1])

        if last["signal"] == 1:
            qty = risk.position_size(equity, price)
            ok, msg = risk.validate_order(symbol, qty, price, equity, daily_pnl, positions)
            if ok and qty > 0:
                res = executor.place_order(Order(symbol=symbol, quantity=qty, side="BUY", price=price))
                if res.ok:
                    positions[symbol] = Position(symbol, qty, price)
        elif last["signal"] == -1 and symbol in positions:
            pos = positions[symbol]
            res = executor.place_order(Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=price))
            if res.ok:
                pnl = (price - pos.avg_price) * pos.quantity
                daily_pnl += pnl
                equity += pnl
                del positions[symbol]


def main() -> None:
    asyncio.run(run_once())


if __name__ == "__main__":
    main()


from __future__ import annotations

import asyncio
from typing import Dict, List

import pandas as pd

from .config import load_config
from .execution import Order, PaperExecutor
from .ibkr_client import IBKRClient
from .risk import Position, RiskManager
from .strategies import sma_crossover_signals
from .logger import get_logger

import argparse


logger = get_logger(__name__)


async def run_once(
    override_symbols: List[str] | None = None,
    duration: str = "10 D",
    bar_size: str = "30 mins",
) -> None:
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

    symbols = override_symbols if override_symbols else cfg.symbols
    logger.info(f"Processing symbols: {symbols}")

    for symbol in symbols:
        df = await ib.fetch_recent_bars(symbol, duration=duration, bar_size=bar_size)
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
    parser = argparse.ArgumentParser(description="Robo Trader runner (paper by default)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to override config", default="")
    parser.add_argument("--duration", type=str, help="IB duration string (e.g. '10 D')", default="10 D")
    parser.add_argument("--bar-size", type=str, help="IB bar size (e.g. '30 mins')", default="30 mins")
    parser.add_argument("--confirm-live", action="store_true", help="Required confirmation flag for live mode")
    args = parser.parse_args()

    cfg = load_config()
    if cfg.trading_mode.lower() == "live" and not args.confirm_live:
        raise SystemExit("Refusing to run in live mode without --confirm-live")

    override_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    asyncio.run(run_once(override_symbols, args.duration, args.bar_size))


if __name__ == "__main__":
    main()


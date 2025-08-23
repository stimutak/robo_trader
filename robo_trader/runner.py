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
from .portfolio import Portfolio
from .retry import retry_async

import argparse


logger = get_logger(__name__)


async def run_once(
    override_symbols: List[str] | None = None,
    duration: str = "10 D",
    bar_size: str = "30 mins",
    sma_fast: int = 10,
    sma_slow: int = 20,
    slippage_bps: float = 0.0,
    max_order_notional: float | None = None,
    max_daily_notional: float | None = None,
    default_cash_override: float | None = None,
) -> None:
    cfg = load_config()
    ib = IBKRClient(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
    await retry_async(lambda: ib.connect(readonly=True))

    # Initialize helpers
    risk = RiskManager(
        max_daily_loss=cfg.max_daily_loss,
        max_position_risk_pct=cfg.max_position_risk_pct,
        max_symbol_exposure_pct=cfg.max_symbol_exposure_pct,
        max_leverage=cfg.max_leverage,
        max_order_notional=max_order_notional,
        max_daily_notional=max_daily_notional,
    )
    executor = PaperExecutor(slippage_bps=slippage_bps)

    starting_cash = default_cash_override if default_cash_override is not None else cfg.default_cash
    portfolio = Portfolio(starting_cash)
    daily_pnl = 0.0
    positions: Dict[str, Position] = {}
    daily_executed_notional = 0.0

    symbols = override_symbols if override_symbols else cfg.symbols
    logger.info(f"Processing symbols: {symbols}")

    for symbol in symbols:
        df = await retry_async(lambda: ib.fetch_recent_bars(symbol, duration=duration, bar_size=bar_size))
        if df.empty:
            continue
        # Normalize columns
        if "close" not in df.columns and "close" in df:
            df = df.rename(columns={"close": "close"})
        signals = sma_crossover_signals(pd.DataFrame({"close": df["close"]}), fast=sma_fast, slow=sma_slow)
        last = signals.iloc[-1]
        price = float(last["close"]) if "close" in last else float(df["close"].iloc[-1])

        equity = portfolio.equity({symbol: price})

        if last["signal"] == 1:
            qty = risk.position_size(equity, price)
            ok, msg = risk.validate_order(symbol, qty, price, equity, daily_pnl, positions, daily_executed_notional)
            if ok and qty > 0:
                res = executor.place_order(Order(symbol=symbol, quantity=qty, side="BUY", price=price))
                if res.ok:
                    positions[symbol] = Position(symbol, qty, price)
                    portfolio.update_fill(symbol, "BUY", qty, res.fill_price or price)
                    daily_executed_notional += price * qty
        elif last["signal"] == -1 and symbol in positions:
            pos = positions[symbol]
            res = executor.place_order(Order(symbol=symbol, quantity=pos.quantity, side="SELL", price=price))
            if res.ok:
                portfolio.update_fill(symbol, "SELL", pos.quantity, res.fill_price or price)
                daily_pnl = portfolio.realized_pnl
                del positions[symbol]


def main() -> None:
    parser = argparse.ArgumentParser(description="Robo Trader runner (paper by default)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to override config", default="")
    parser.add_argument("--duration", type=str, help="IB duration string (e.g. '10 D')", default="10 D")
    parser.add_argument("--bar-size", type=str, help="IB bar size (e.g. '30 mins')", default="30 mins")
    parser.add_argument("--confirm-live", action="store_true", help="Required confirmation flag for live mode")
    parser.add_argument("--sma-fast", type=int, default=10, help="Fast SMA window")
    parser.add_argument("--sma-slow", type=int, default=20, help="Slow SMA window")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Paper slippage in basis points")
    parser.add_argument("--max-order-notional", type=float, default=None, help="Per-order notional ceiling")
    parser.add_argument("--max-daily-notional", type=float, default=None, help="Per-day notional ceiling")
    parser.add_argument("--default-cash", type=float, default=None, help="Override starting cash for paper run")
    args = parser.parse_args()

    cfg = load_config()
    if cfg.trading_mode.lower() == "live" and not args.confirm_live:
        raise SystemExit("Refusing to run in live mode without --confirm-live")

    override_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    asyncio.run(
        run_once(
            override_symbols,
            args.duration,
            args.bar_size,
            args.sma_fast,
            args.sma_slow,
            args.slippage_bps,
            args.max_order_notional,
            args.max_daily_notional,
            args.default_cash,
        )
    )


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    main()


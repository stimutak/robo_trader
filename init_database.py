#!/usr/bin/env python3
"""Initialize database with sample data for testing.

WARNING: This script writes FAKE trades, positions, signals, and account data.
It is intended for SAMPLE/TEST databases ONLY. It refuses to operate on the
production trading_data.db filename and refuses to overwrite an existing file
unless --force is passed.
"""

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

from robo_trader.database_async import AsyncTradingDatabase


async def init_sample_data(db_path: Path) -> None:
    """Initialize database with sample trading data."""
    db = AsyncTradingDatabase(db_path=db_path)
    await db.initialize()

    try:
        # Sample symbols
        symbols = ["AAPL", "NVDA", "TSLA", "PLTR", "SOFI"]

        # Add some sample trades (both BUY and SELL)
        base_time = datetime.now() - timedelta(days=7)

        for i, symbol in enumerate(symbols):
            # Buy trade
            buy_price = 100 + i * 20 + random.uniform(-5, 5)
            buy_qty = 100
            await db.record_trade(
                symbol=symbol,
                side="BUY",
                quantity=buy_qty,
                price=buy_price,
                slippage=0.01,
                commission=1.0,
            )

            # Add some market data
            for j in range(10):
                timestamp = base_time + timedelta(hours=j * 6)
                price = buy_price + random.uniform(-5, 5)
                await db.store_market_data(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=price,
                    high=price + random.uniform(0, 2),
                    low=price - random.uniform(0, 2),
                    close=price + random.uniform(-1, 1),
                    volume=random.randint(1000000, 5000000),
                )

            # Sell half the position (to show both BUY and SELL)
            if i < 3:  # Only sell for first 3 symbols
                sell_price = buy_price + random.uniform(-2, 10)
                await db.record_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=buy_qty // 2,
                    price=sell_price,
                    slippage=0.01,
                    commission=1.0,
                )

                # Update position
                await db.update_position(
                    symbol=symbol,
                    quantity=buy_qty // 2,
                    avg_cost=buy_price,
                    market_price=sell_price,
                )
            else:
                # Keep full position
                await db.update_position(
                    symbol=symbol,
                    quantity=buy_qty,
                    avg_cost=buy_price,
                    market_price=buy_price + random.uniform(-2, 5),
                )

        # Add some signals for strategy display.
        # Note: metadata is validated by DatabaseValidator and rejects SQL-like
        # tokens including double-quotes, so we keep it simple.
        for symbol in symbols[:3]:
            await db.record_signal(
                symbol=symbol,
                strategy="ML_Enhanced",
                signal_type="BUY" if random.random() > 0.5 else "HOLD",
                strength=random.uniform(0.5, 0.9),
            )

            await db.record_signal(
                symbol=symbol,
                strategy="OrderFlowImbalance",
                signal_type="BUY" if random.random() > 0.5 else "SELL",
                strength=random.uniform(0.4, 0.8),
            )

        # Update account with sample P&L
        await db.update_account(
            cash=95000, equity=105000, daily_pnl=500, realized_pnl=2000, unrealized_pnl=3000
        )

        print("Database initialized with sample data")
        print("   - 5 symbols with positions")
        print("   - Both BUY and SELL trades")
        print("   - Market data for each symbol")
        print("   - Strategy signals")
        print("   - Account P&L data")

    finally:
        await db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize a SAMPLE database for testing.",
    )
    parser.add_argument(
        "--db-path",
        required=True,
        help="Path to NEW database. Refuses to run if file exists (use --force to override).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwrite -- DANGEROUS. Will write fake trades over existing data.",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)

    # Refuse to clobber the production database name.
    if db_path.name == "trading_data.db":
        print(
            "REFUSING to run: this script is for SAMPLE/TEST DBs only. "
            "Use a different filename (e.g. sample.db).",
            file=sys.stderr,
        )
        return 2

    if db_path.exists() and not args.force:
        print(
            f"REFUSING to run: {db_path} already exists. Use --force to override "
            "(this will write fake trades).",
            file=sys.stderr,
        )
        return 2

    asyncio.run(init_sample_data(db_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Attempt to recover data from corrupted database"""

import asyncio
import sqlite3
from datetime import datetime

from robo_trader.database_async import AsyncTradingDatabase


def try_recover_data():
    """Try to recover data from backup database"""
    recovered_data = {
        "trades": [],
        "positions": [],
        "market_data": [],
        "signals": [],
        "account": None,
    }

    try:
        # Try to connect to backup database
        conn = sqlite3.connect("trading_data_backup_20250901_081014.db")
        conn.execute("PRAGMA integrity_check")

        # Try to recover trades
        try:
            cursor = conn.execute("SELECT * FROM trades LIMIT 1000")
            trades = cursor.fetchall()
            print(f"✅ Recovered {len(trades)} trades")
            recovered_data["trades"] = trades
        except Exception as e:
            print(f"❌ Could not recover trades: {e}")

        # Try to recover positions
        try:
            cursor = conn.execute("SELECT * FROM positions")
            positions = cursor.fetchall()
            print(f"✅ Recovered {len(positions)} positions")
            recovered_data["positions"] = positions
        except Exception as e:
            print(f"❌ Could not recover positions: {e}")

        # Try to recover recent market data
        try:
            cursor = conn.execute(
                """
                SELECT * FROM market_data 
                WHERE timestamp >= datetime('now', '-7 days')
                LIMIT 5000
            """
            )
            market_data = cursor.fetchall()
            print(f"✅ Recovered {len(market_data)} market data records")
            recovered_data["market_data"] = market_data
        except Exception as e:
            print(f"❌ Could not recover market data: {e}")

        # Try to recover signals
        try:
            cursor = conn.execute(
                """
                SELECT * FROM signals 
                WHERE timestamp >= datetime('now', '-1 day')
                LIMIT 1000
            """
            )
            signals = cursor.fetchall()
            print(f"✅ Recovered {len(signals)} signals")
            recovered_data["signals"] = signals
        except Exception as e:
            print(f"❌ Could not recover signals: {e}")

        # Try to recover account info
        try:
            cursor = conn.execute("SELECT * FROM account WHERE id = 1")
            account = cursor.fetchone()
            if account:
                print(f"✅ Recovered account info")
                recovered_data["account"] = account
        except Exception as e:
            print(f"❌ Could not recover account: {e}")

        conn.close()
        return recovered_data

    except Exception as e:
        print(f"❌ Failed to open backup database: {e}")
        return recovered_data


async def restore_to_new_database(recovered_data):
    """Restore recovered data to new database"""
    # Remove any existing database first
    import os

    if os.path.exists("trading_data.db"):
        os.remove("trading_data.db")
    if os.path.exists("trading_data.db-journal"):
        os.remove("trading_data.db-journal")

    db = AsyncTradingDatabase()
    await db.initialize()

    try:
        # Restore positions
        if recovered_data["positions"]:
            for pos in recovered_data["positions"]:
                try:
                    await db.update_position(
                        symbol=pos[1],  # symbol
                        quantity=pos[2],  # quantity
                        avg_cost=pos[3],  # avg_cost
                        market_price=pos[4] if len(pos) > 4 else None,  # market_price
                    )
                except Exception as e:
                    print(f"Failed to restore position: {e}")

        # Restore trades
        if recovered_data["trades"]:
            for trade in recovered_data["trades"][:100]:  # Limit to recent 100
                try:
                    await db.record_trade(
                        symbol=trade[1],  # symbol
                        side=trade[2],  # side
                        quantity=trade[3],  # quantity
                        price=trade[4],  # price
                        slippage=trade[5] if len(trade) > 5 else 0,
                        commission=trade[6] if len(trade) > 6 else 0,
                    )
                except Exception as e:
                    print(f"Failed to restore trade: {e}")

        # Restore account
        if recovered_data["account"]:
            acc = recovered_data["account"]
            await db.update_account(
                cash=acc[1],  # cash
                equity=acc[2],  # equity
                daily_pnl=acc[3] if len(acc) > 3 else 0,
                realized_pnl=acc[4] if len(acc) > 4 else 0,
                unrealized_pnl=acc[5] if len(acc) > 5 else 0,
            )

        print("\n✅ Data restoration complete!")

    finally:
        await db.close()


if __name__ == "__main__":
    print("Attempting to recover data from backup database...")
    print("-" * 50)

    recovered = try_recover_data()

    if any([recovered["trades"], recovered["positions"], recovered["market_data"]]):
        print("\n" + "=" * 50)
        print("Restoring recovered data to new database...")
        print("=" * 50)
        asyncio.run(restore_to_new_database(recovered))
    else:
        print("\n❌ No data could be recovered. Creating fresh database with sample data...")
        import init_database

        asyncio.run(init_database.init_sample_data())

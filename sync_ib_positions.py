#!/usr/bin/env python3
"""
Sync actual Interactive Brokers positions to local database.
This will clear old test data and replace with real positions.
"""

import asyncio
import logging
from datetime import datetime

from robo_trader.clients.async_ibkr_client import AsyncIBKRClient
from robo_trader.database_async import AsyncTradingDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def sync_positions():
    """Sync IB positions to database."""
    client = None
    db = None

    try:
        # Initialize IB client
        logger.info("Connecting to Interactive Brokers...")
        client = AsyncIBKRClient()
        await client.connect()

        # Initialize database
        logger.info("Connecting to database...")
        db = AsyncTradingDatabase()
        await db.initialize()

        # Get real positions from IB
        logger.info("Fetching positions from IB...")
        ib_positions = await client.get_positions()

        if not ib_positions:
            logger.warning("No positions found in IB account")
        else:
            logger.info(f"Found {len(ib_positions)} positions in IB:")
            for pos in ib_positions:
                logger.info(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")

        # Clear old test positions
        logger.info("\nClearing old test positions from database...")
        async with db.get_connection() as conn:
            await conn.execute("DELETE FROM positions")
            await conn.commit()
            logger.info("Old positions cleared")

        # Insert real IB positions
        if ib_positions:
            logger.info("\nSyncing IB positions to database...")
            for pos in ib_positions:
                await db.update_position(
                    symbol=pos["symbol"],
                    quantity=pos["quantity"],
                    avg_cost=pos["avg_cost"],
                    market_price=pos.get("market_value", 0) / pos["quantity"]
                    if pos["quantity"]
                    else 0,
                )
            logger.info(f"Synced {len(ib_positions)} positions to database")

        # Verify the sync
        logger.info("\nVerifying database positions:")
        db_positions = await db.get_positions()
        for pos in db_positions:
            logger.info(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")

        logger.info("\n✅ Position sync complete!")

    except Exception as e:
        logger.error(f"Error syncing positions: {e}")
        raise
    finally:
        if client:
            await client.disconnect()
        if db:
            await db.close()


if __name__ == "__main__":
    asyncio.run(sync_positions())

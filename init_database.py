#!/usr/bin/env python3
"""Initialize database with sample data for testing"""

import asyncio
from datetime import datetime, timedelta
import random
from robo_trader.database_async import AsyncTradingDatabase

async def init_sample_data():
    """Initialize database with sample trading data"""
    db = AsyncTradingDatabase()
    await db.initialize()
    
    try:
        # Sample symbols
        symbols = ['AAPL', 'NVDA', 'TSLA', 'PLTR', 'SOFI']
        
        # Add some sample trades (both BUY and SELL)
        base_time = datetime.now() - timedelta(days=7)
        
        for i, symbol in enumerate(symbols):
            # Buy trade
            buy_price = 100 + i * 20 + random.uniform(-5, 5)
            buy_qty = 100
            await db.record_trade(
                symbol=symbol,
                side='BUY',
                quantity=buy_qty,
                price=buy_price,
                slippage=0.01,
                commission=1.0
            )
            
            # Add some market data
            for j in range(10):
                timestamp = base_time + timedelta(hours=j*6)
                price = buy_price + random.uniform(-5, 5)
                await db.store_market_data(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=price,
                    high=price + random.uniform(0, 2),
                    low=price - random.uniform(0, 2),
                    close=price + random.uniform(-1, 1),
                    volume=random.randint(1000000, 5000000)
                )
            
            # Sell half the position (to show both BUY and SELL)
            if i < 3:  # Only sell for first 3 symbols
                sell_price = buy_price + random.uniform(-2, 10)
                await db.record_trade(
                    symbol=symbol,
                    side='SELL',
                    quantity=buy_qty // 2,
                    price=sell_price,
                    slippage=0.01,
                    commission=1.0
                )
                
                # Update position
                await db.update_position(
                    symbol=symbol,
                    quantity=buy_qty // 2,
                    avg_cost=buy_price,
                    market_price=sell_price
                )
            else:
                # Keep full position
                await db.update_position(
                    symbol=symbol,
                    quantity=buy_qty,
                    avg_cost=buy_price,
                    market_price=buy_price + random.uniform(-2, 5)
                )
        
        # Add some signals for strategy display
        for symbol in symbols[:3]:
            await db.record_signal(
                symbol=symbol,
                strategy='ML_Enhanced',
                signal_type='BUY' if random.random() > 0.5 else 'HOLD',
                strength=random.uniform(0.5, 0.9),
                metadata='{"confidence": 0.75}'
            )
            
            await db.record_signal(
                symbol=symbol,
                strategy='OrderFlowImbalance',
                signal_type='BUY' if random.random() > 0.5 else 'SELL',
                strength=random.uniform(0.4, 0.8),
                metadata='{"ofi": 0.65}'
            )
        
        # Update account with sample P&L
        await db.update_account(
            cash=95000,
            equity=105000,
            daily_pnl=500,
            realized_pnl=2000,
            unrealized_pnl=3000
        )
        
        print("✅ Database initialized with sample data")
        print("   - 5 symbols with positions")
        print("   - Both BUY and SELL trades")
        print("   - Market data for each symbol")
        print("   - Strategy signals")
        print("   - Account P&L data")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(init_sample_data())
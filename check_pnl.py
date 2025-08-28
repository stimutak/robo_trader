#!/usr/bin/env python
"""Check P&L calculations from database."""

import asyncio
from robo_trader.database_async import AsyncTradingDatabase

async def check_pnl():
    db = AsyncTradingDatabase()
    await db.initialize()
    try:
        # Get positions
        positions = await db.get_positions()
        print(f"Found {len(positions)} positions:")
        
        total_cost = 0
        total_value = 0
        
        for pos in positions:
            symbol = pos['symbol']
            qty = pos['quantity']
            avg_cost = pos['avg_cost']
            
            # Get latest price
            market_data = await db.get_latest_market_data(symbol, limit=1)
            if market_data:
                current_price = market_data[0]['close']
            else:
                # Try to get from IB
                from robo_trader.clients.async_ibkr_client import AsyncIBKRClient
                client = AsyncIBKRClient()
                await client.initialize()
                try:
                    price_data = await client.get_market_data(symbol)
                    current_price = price_data['close'] if price_data else avg_cost
                finally:
                    await client.close_all()
            
            cost = qty * avg_cost
            value = qty * current_price
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0
            
            total_cost += cost
            total_value += value
            
            print(f"  {symbol}: {qty} shares @ ${avg_cost:.2f} = ${cost:.2f}")
            print(f"    Current: ${current_price:.2f}, Value: ${value:.2f}")
            print(f"    P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
        
        total_unrealized = total_value - total_cost
        print(f"\nTotal Cost: ${total_cost:.2f}")
        print(f"Total Value: ${total_value:.2f}")
        print(f"Unrealized P&L: ${total_unrealized:.2f}")
        
        # Check trades for realized P&L
        trades = await db.get_trades_history(days=365)
        print(f"\nFound {len(trades)} trades")
        
        # Calculate realized P&L
        realized_pnl = 0
        symbol_trades = {}
        
        for trade in trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        for symbol, trades_list in symbol_trades.items():
            buys = []
            symbol_realized = 0
            
            for trade in sorted(trades_list, key=lambda x: x['timestamp']):
                if trade['side'] == 'buy':
                    buys.append({'price': trade['price'], 'quantity': trade['quantity']})
                elif trade['side'] == 'sell' and buys:
                    sell_qty = trade['quantity']
                    sell_price = trade['price']
                    
                    while sell_qty > 0 and buys:
                        buy = buys[0]
                        match_qty = min(sell_qty, buy['quantity'])
                        
                        pnl = (sell_price - buy['price']) * match_qty
                        symbol_realized += pnl
                        
                        sell_qty -= match_qty
                        buy['quantity'] -= match_qty
                        
                        if buy['quantity'] == 0:
                            buys.pop(0)
            
            if symbol_realized != 0:
                print(f"  {symbol}: Realized P&L: ${symbol_realized:.2f}")
                realized_pnl += symbol_realized
        
        print(f"\nTotal Realized P&L: ${realized_pnl:.2f}")
        print(f"Total P&L: ${realized_pnl + total_unrealized:.2f}")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(check_pnl())
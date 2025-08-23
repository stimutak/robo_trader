#!/usr/bin/env python3
"""Populate database with test price data for the last trading day."""

from robo_trader.database import TradingDatabase
from datetime import datetime, timedelta, date
import random

def populate_last_trading_day():
    """Add price data for yesterday's trading session."""
    db = TradingDatabase()
    
    # Get last trading day (Friday if today is weekend)
    today = date.today()
    if today.weekday() == 5:  # Saturday
        last_trading_day = today - timedelta(days=1)
    elif today.weekday() == 6:  # Sunday
        last_trading_day = today - timedelta(days=2)
    else:
        last_trading_day = today
    
    # Symbols to populate
    symbols = ['AAPL', 'NVDA', 'TSLA', 'PLTR', 'SOFI']
    
    # Base prices
    base_prices = {
        'AAPL': 225.0,
        'NVDA': 127.0,
        'TSLA': 215.0,
        'PLTR': 29.0,
        'SOFI': 7.5
    }
    
    print(f"Populating data for {last_trading_day}...")
    
    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate price data for full trading day (9:30 AM to 4:00 PM)
        for minute in range(0, 390, 5):  # Every 5 minutes
            # Calculate time
            hour = 9 + (30 + minute) // 60
            min_of_hour = (30 + minute) % 60
            
            timestamp = datetime.combine(
                last_trading_day,
                datetime.strptime(f"{hour:02d}:{min_of_hour:02d}", "%H:%M").time()
            )
            
            # Generate price with some random movement
            volatility = base_price * 0.002  # 0.2% volatility
            price = base_price + random.gauss(0, volatility)
            
            # Add trend based on time of day
            if minute < 60:  # First hour - usually volatile
                price += random.uniform(-base_price * 0.01, base_price * 0.01)
            elif minute < 180:  # Mid-day - trend up slightly
                price += base_price * 0.001 * (minute / 180)
            else:  # Afternoon - slight decline
                price -= base_price * 0.0005 * ((minute - 180) / 210)
            
            # Save to database
            db.save_price_point(symbol, price, timestamp, minute)
        
        print(f"  Added {symbol}: 78 data points")
    
    db.close()
    print(f"\nSuccessfully populated price data for {len(symbols)} symbols")
    print(f"Trading day: {last_trading_day}")

if __name__ == "__main__":
    populate_last_trading_day()
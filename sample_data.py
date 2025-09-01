#!/usr/bin/env python3
"""Sample data for dashboard when database is locked"""

def get_sample_positions():
    """Return sample positions data"""
    return [
        {'symbol': 'AAPL', 'quantity': 100, 'entry_price': 175.50, 'current_price': 178.20, 'ml_signal': 'hold'},
        {'symbol': 'NVDA', 'quantity': 50, 'entry_price': 455.00, 'current_price': 462.30, 'ml_signal': 'buy'},
        {'symbol': 'TSLA', 'quantity': 75, 'entry_price': 242.10, 'current_price': 238.50, 'ml_signal': 'sell'},
        {'symbol': 'PLTR', 'quantity': 200, 'entry_price': 15.25, 'current_price': 16.10, 'ml_signal': 'hold'},
        {'symbol': 'SOFI', 'quantity': 300, 'entry_price': 7.80, 'current_price': 8.15, 'ml_signal': 'buy'},
    ]

def get_sample_watchlist():
    """Return sample watchlist data"""
    symbols = [
        'AAPL', 'NVDA', 'TSLA', 'IXHL', 'NUAI', 'BZAI', 'ELTP', 'OPEN', 
        'CEG', 'VRT', 'PLTR', 'UPST', 'TEM', 'HTFL', 'SDGR', 'APLD', 
        'SOFI', 'CORZ', 'WULF'
    ]
    
    positions = {
        'AAPL': {'qty': 100, 'avg': 175.50, 'price': 178.20},
        'NVDA': {'qty': 50, 'avg': 455.00, 'price': 462.30},
        'TSLA': {'qty': 75, 'avg': 242.10, 'price': 238.50},
        'PLTR': {'qty': 200, 'avg': 15.25, 'price': 16.10},
        'SOFI': {'qty': 300, 'avg': 7.80, 'price': 8.15},
    }
    
    watchlist = []
    for symbol in symbols:
        if symbol in positions:
            pos = positions[symbol]
            pnl = (pos['price'] - pos['avg']) * pos['qty']
            watchlist.append({
                'symbol': symbol,
                'current_price': pos['price'],
                'quantity': pos['qty'],
                'avg_cost': pos['avg'],
                'pnl': pnl,
                'notes': 'Active',
                'has_position': True
            })
        else:
            import random
            price = random.uniform(10, 500)
            watchlist.append({
                'symbol': symbol,
                'current_price': price,
                'quantity': 0,
                'avg_cost': 0,
                'pnl': 0,
                'notes': 'Watching',
                'has_position': False
            })
    
    return watchlist

def get_sample_trades():
    """Return sample trades data"""
    return [
        {'id': 1, 'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 175.50, 
         'timestamp': '2025-08-28 09:30:00', 'slippage': 0.01, 'commission': 1.0,
         'notional': 17550, 'cash_impact': -17551},
        {'id': 2, 'symbol': 'NVDA', 'side': 'BUY', 'quantity': 50, 'price': 455.00,
         'timestamp': '2025-08-28 09:35:00', 'slippage': 0.02, 'commission': 1.0,
         'notional': 22750, 'cash_impact': -22751},
        {'id': 3, 'symbol': 'TSLA', 'side': 'BUY', 'quantity': 75, 'price': 242.10,
         'timestamp': '2025-08-28 10:00:00', 'slippage': 0.01, 'commission': 1.0,
         'notional': 18157.5, 'cash_impact': -18158.5},
        {'id': 4, 'symbol': 'AAPL', 'side': 'SELL', 'quantity': 50, 'price': 178.20,
         'timestamp': '2025-08-29 14:30:00', 'slippage': 0.01, 'commission': 1.0,
         'notional': 8910, 'cash_impact': 8909},
        {'id': 5, 'symbol': 'PLTR', 'side': 'BUY', 'quantity': 200, 'price': 15.25,
         'timestamp': '2025-08-29 15:00:00', 'slippage': 0.01, 'commission': 1.0,
         'notional': 3050, 'cash_impact': -3051},
    ]

def get_sample_pnl():
    """Return sample P&L data"""
    return {
        'daily': 523.45,
        'total': 2847.30,
        'realized': 1235.00,
        'unrealized': 1612.30
    }

def get_sample_account():
    """Return sample account data"""
    return {
        'cash': 94235.50,
        'equity': 97082.80,
        'daily_pnl': 523.45,
        'realized_pnl': 1235.00,
        'unrealized_pnl': 1612.30
    }
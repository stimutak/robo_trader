#!/usr/bin/env python3
"""Test database integration with the trading system."""

import asyncio
from datetime import datetime
from robo_trader.database import TradingDatabase
from robo_trader.logger import get_logger

logger = get_logger(__name__)

async def test_integration():
    """Test database integration."""
    
    # Initialize database
    db = TradingDatabase('test_integration.db')
    
    print("Testing Database Integration")
    print("=" * 50)
    
    # 1. Save a test trade
    print("\n1. Saving test trade...")
    trade_data = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 175.50,
        'ai_confidence': 85.0,
        'ai_reasoning': 'Test trade for database integration',
        'strategy': 'TEST',
        'pnl': None
    }
    trade_id = db.save_trade(trade_data)
    print(f"   ✓ Trade saved with ID: {trade_id}")
    
    # 2. Save options signal
    print("\n2. Saving options signal...")
    signal_data = {
        'symbol': 'SPY',
        'strike': 450.0,
        'expiry': datetime.now().date(),
        'option_type': 'CALL',
        'signal_type': 'SWEEP',
        'volume': 10000,
        'confidence': 90.0,
        'premium': 50000
    }
    signal_id = db.save_options_signal(signal_data)
    print(f"   ✓ Options signal saved with ID: {signal_id}")
    
    # 3. Save AI decision
    print("\n3. Saving AI decision...")
    decision_data = {
        'event_type': 'TEST',
        'event_data': {'test': True},
        'decision': 'BUY',
        'confidence': 75.0,
        'reasoning': 'Test decision for integration'
    }
    decision_id = db.save_ai_decision(decision_data)
    print(f"   ✓ AI decision saved with ID: {decision_id}")
    
    # 4. Save P&L snapshot
    print("\n4. Saving P&L snapshot...")
    pnl_data = {
        'total_pnl': 500.0,
        'daily_pnl': 200.0,
        'positions_count': 2
    }
    pnl_id = db.save_pnl_snapshot(pnl_data)
    print(f"   ✓ P&L snapshot saved with ID: {pnl_id}")
    
    # 5. Query recent data
    print("\n5. Querying recent data...")
    recent_trades = db.get_recent_trades(limit=5)
    print(f"   Found {len(recent_trades)} recent trades")
    
    today_pnl = db.get_today_pnl()
    print(f"   Today's P&L: ${today_pnl.get('total_pnl', 0):.2f}")
    
    metrics = db.get_performance_metrics(days=30)
    print(f"   30-day metrics: {metrics['total_trades']} trades, ${metrics['total_pnl']:.2f} P&L")
    
    print("\n" + "=" * 50)
    print("✅ All database integration tests passed!")
    
    # Clean up
    db.close()
    import os
    os.remove('test_integration.db')
    print("   Cleaned up test database")

if __name__ == "__main__":
    asyncio.run(test_integration())
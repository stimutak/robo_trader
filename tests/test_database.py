"""Tests for database module."""

import pytest
import tempfile
import os
from datetime import datetime, date, timedelta
from pathlib import Path

from robo_trader.database import TradingDatabase


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    db = TradingDatabase(db_path)
    yield db
    db.close()
    os.unlink(db_path)


def test_database_creation(temp_db):
    """Test that database and tables are created properly."""
    assert temp_db.conn is not None
    
    # Check tables exist
    cursor = temp_db.conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    
    expected_tables = {
        'trades', 'options_flow', 'pnl_history', 
        'news_archive', 'ai_decisions', 'market_regimes'
    }
    assert expected_tables.issubset(tables)


def test_save_and_retrieve_trade(temp_db):
    """Test saving and retrieving trades."""
    trade_data = {
        'symbol': 'AAPL',
        'action': 'BUY',
        'quantity': 100,
        'price': 150.50,
        'notional': 15050.00,
        'ai_confidence': 85.5,
        'ai_reasoning': 'Strong bullish signal from options flow',
        'strategy': 'OPTIONS_FLOW',
        'pnl': None,
        'commission': 1.00
    }
    
    trade_id = temp_db.save_trade(trade_data)
    assert trade_id > 0
    
    # Retrieve recent trades
    trades = temp_db.get_recent_trades(limit=1)
    assert len(trades) == 1
    
    saved_trade = trades[0]
    assert saved_trade['symbol'] == 'AAPL'
    assert saved_trade['action'] == 'BUY'
    assert saved_trade['quantity'] == 100
    assert saved_trade['price'] == 150.50
    assert saved_trade['ai_confidence'] == 85.5


def test_save_options_signal(temp_db):
    """Test saving options flow signals."""
    signal = {
        'symbol': 'SPY',
        'strike': 450.0,
        'expiry': date.today() + timedelta(days=30),
        'option_type': 'CALL',
        'signal_type': 'SWEEP',
        'volume': 5000,
        'open_interest': 10000,
        'confidence': 90.0,
        'premium': 2.50,
        'implied_volatility': 0.25
    }
    
    signal_id = temp_db.save_options_signal(signal)
    assert signal_id > 0
    
    # Retrieve recent options flow
    signals = temp_db.get_recent_options_flow(limit=1)
    assert len(signals) == 1
    assert signals[0]['symbol'] == 'SPY'
    assert signals[0]['signal_type'] == 'SWEEP'


def test_save_pnl_snapshot(temp_db):
    """Test saving P&L snapshots."""
    pnl_data = {
        'total_pnl': 1500.00,
        'daily_pnl': 500.00,
        'realized_pnl': 1000.00,
        'unrealized_pnl': 500.00,
        'positions_count': 3,
        'positions': [
            {'symbol': 'AAPL', 'quantity': 100, 'pnl': 200},
            {'symbol': 'GOOGL', 'quantity': 50, 'pnl': 300}
        ],
        'win_rate': 0.65,
        'sharpe_ratio': 1.8
    }
    
    snapshot_id = temp_db.save_pnl_snapshot(pnl_data)
    assert snapshot_id > 0


def test_save_news_deduplication(temp_db):
    """Test that duplicate news URLs are not saved."""
    news_item = {
        'title': 'Fed announces rate decision',
        'url': 'https://example.com/news/fed-decision',
        'source': 'Reuters',
        'sentiment': 0.2,
        'symbols': ['SPY', 'QQQ'],
        'category': 'MACRO',
        'ai_summary': 'Fed holds rates steady'
    }
    
    # First save should succeed
    news_id1 = temp_db.save_news(news_item)
    assert news_id1 is not None
    
    # Second save with same URL should return None
    news_id2 = temp_db.save_news(news_item)
    assert news_id2 is None


def test_save_and_update_ai_decision(temp_db):
    """Test saving AI decisions and updating outcomes."""
    decision = {
        'event_type': 'OPTIONS_FLOW',
        'event_data': {'symbol': 'TSLA', 'signal': 'CALL_SWEEP'},
        'decision': 'BUY',
        'confidence': 75.0,
        'reasoning': 'Large call sweep detected with high confidence'
    }
    
    decision_id = temp_db.save_ai_decision(decision)
    assert decision_id > 0
    
    # Update outcome
    temp_db.update_decision_outcome(decision_id, 'SUCCESS', 250.00)
    
    # Verify update
    cursor = temp_db.conn.cursor()
    cursor.execute('SELECT outcome, outcome_pnl FROM ai_decisions WHERE id = ?', (decision_id,))
    result = cursor.fetchone()
    assert result[0] == 'SUCCESS'
    assert result[1] == 250.00


def test_get_today_pnl(temp_db):
    """Test calculating today's P&L."""
    # Insert trades with explicit timestamp for today
    cursor = temp_db.conn.cursor()
    now = datetime.now()
    
    trades = [
        ('AAPL', 'BUY', 100, 150, 15000, 85, 'test', 'TEST', 200, 1),
        ('GOOGL', 'SELL', 50, 140, 7000, 75, 'test', 'TEST', -100, 1),
        ('MSFT', 'BUY', 75, 350, 26250, 80, 'test', 'TEST', 150, 1)
    ]
    
    for trade in trades:
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, action, quantity, price, notional, 
                              ai_confidence, ai_reasoning, strategy, pnl, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (now,) + trade)
    
    temp_db.conn.commit()
    
    today_pnl = temp_db.get_today_pnl()
    assert today_pnl['total_pnl'] == 250  # 200 - 100 + 150
    assert today_pnl['trade_count'] == 3
    assert today_pnl['wins'] == 2
    assert today_pnl['losses'] == 1
    assert today_pnl['win_rate'] == pytest.approx(0.667, rel=0.01)


def test_get_performance_metrics(temp_db):
    """Test calculating performance metrics."""
    # Add trades with different P&L values
    trades = [
        {'symbol': 'SPY', 'action': 'BUY', 'quantity': 100, 'price': 450, 'pnl': 500},
        {'symbol': 'QQQ', 'action': 'SELL', 'quantity': 50, 'price': 380, 'pnl': -200},
        {'symbol': 'TSLA', 'action': 'BUY', 'quantity': 10, 'price': 250, 'pnl': 300},
        {'symbol': 'AAPL', 'action': 'SELL', 'quantity': 100, 'price': 150, 'pnl': -100}
    ]
    
    for trade in trades:
        temp_db.save_trade(trade)
    
    metrics = temp_db.get_performance_metrics(days=30)
    assert metrics['total_trades'] == 4
    assert metrics['total_pnl'] == 500  # 500 - 200 + 300 - 100
    assert metrics['best_trade'] == 500
    assert metrics['worst_trade'] == -200
    assert metrics['winning_trades'] == 2
    assert metrics['losing_trades'] == 2
    assert metrics['win_rate'] == 0.5


def test_get_recent_trades_by_symbol(temp_db):
    """Test filtering trades by symbol."""
    # Add trades for different symbols
    temp_db.save_trade({'symbol': 'AAPL', 'action': 'BUY', 'quantity': 100, 'price': 150})
    temp_db.save_trade({'symbol': 'GOOGL', 'action': 'BUY', 'quantity': 50, 'price': 140})
    temp_db.save_trade({'symbol': 'AAPL', 'action': 'SELL', 'quantity': 100, 'price': 155})
    
    # Get only AAPL trades
    aapl_trades = temp_db.get_recent_trades(symbol='AAPL', limit=10)
    assert len(aapl_trades) == 2
    assert all(trade['symbol'] == 'AAPL' for trade in aapl_trades)
    
    # Get all trades
    all_trades = temp_db.get_recent_trades(limit=10)
    assert len(all_trades) == 3
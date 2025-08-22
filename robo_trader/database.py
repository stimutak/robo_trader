"""Database module for persistent storage of trading data."""

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TradingDatabase:
    """SQLite database for storing trading history and analytics."""
    
    def __init__(self, db_path: str = 'trading.db'):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(
            str(self.db_path), 
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Register adapters for date/datetime to avoid deprecation warnings
        sqlite3.register_adapter(datetime, lambda d: d.isoformat())
        sqlite3.register_adapter(date, lambda d: d.isoformat())
        sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode()))
        sqlite3.register_converter("DATE", lambda s: date.fromisoformat(s.decode()))
        
        self.create_tables()
        
    def create_tables(self):
        """Create all required database tables."""
        cursor = self.conn.cursor()
        
        # Trades table - records all executed trades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,  -- BUY, SELL
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                notional REAL NOT NULL,
                ai_confidence REAL,
                ai_reasoning TEXT,
                strategy TEXT,
                pnl REAL,
                commission REAL DEFAULT 0
            )
        ''')
        
        # Options flow table - unusual options activity
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry DATE NOT NULL,
                option_type TEXT NOT NULL,  -- CALL, PUT
                signal_type TEXT NOT NULL,  -- BLOCK, SWEEP, SPLIT
                volume INTEGER NOT NULL,
                open_interest INTEGER,
                confidence REAL,
                premium REAL,
                implied_volatility REAL
            )
        ''')
        
        # P&L history - track portfolio performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pnl_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_pnl REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                realized_pnl REAL,
                unrealized_pnl REAL,
                positions_count INTEGER,
                positions_json TEXT,  -- JSON of current positions
                win_rate REAL,
                sharpe_ratio REAL
            )
        ''')
        
        # News archive - store headlines and sentiment
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                title TEXT NOT NULL,
                url TEXT UNIQUE,
                source TEXT,
                sentiment REAL,  -- -1 to 1
                symbols TEXT,  -- Comma-separated list
                category TEXT,
                ai_summary TEXT
            )
        ''')
        
        # AI decisions - track all AI analysis and decisions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,  -- NEWS, OPTIONS_FLOW, TECHNICAL, REGIME_CHANGE
                event_data TEXT,  -- JSON of event details
                decision TEXT NOT NULL,  -- BUY, SELL, HOLD, WAIT
                confidence REAL NOT NULL,  -- 0 to 100
                reasoning TEXT,
                outcome TEXT,  -- SUCCESS, FAILURE, PENDING
                outcome_pnl REAL
            )
        ''')
        
        # Market regimes - track market conditions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_regimes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                regime TEXT NOT NULL,  -- BULLISH, BEARISH, NEUTRAL, VOLATILE
                vix_level REAL,
                trend_strength REAL,
                breadth REAL,
                risk_level TEXT
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_flow(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_archive(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_timestamp ON ai_decisions(timestamp)')
        
        self.conn.commit()
        
    def save_trade(self, trade_data: Dict[str, Any]) -> int:
        """Save a trade to the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, action, quantity, price, notional, 
                              ai_confidence, ai_reasoning, strategy, pnl, commission)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['symbol'],
            trade_data['action'],
            trade_data['quantity'],
            trade_data['price'],
            trade_data.get('notional', trade_data['quantity'] * trade_data['price']),
            trade_data.get('ai_confidence'),
            trade_data.get('ai_reasoning'),
            trade_data.get('strategy'),
            trade_data.get('pnl'),
            trade_data.get('commission', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid
        
    def save_options_signal(self, signal: Dict[str, Any]) -> int:
        """Save an options flow signal."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO options_flow (symbol, strike, expiry, option_type, signal_type,
                                    volume, open_interest, confidence, premium, implied_volatility)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            signal['strike'],
            signal['expiry'],
            signal['option_type'],
            signal['signal_type'],
            signal['volume'],
            signal.get('open_interest'),
            signal.get('confidence'),
            signal.get('premium'),
            signal.get('implied_volatility')
        ))
        self.conn.commit()
        return cursor.lastrowid
        
    def save_pnl_snapshot(self, pnl_data: Dict[str, Any]) -> int:
        """Save P&L snapshot."""
        cursor = self.conn.cursor()
        positions_json = json.dumps(pnl_data.get('positions', []))
        cursor.execute('''
            INSERT INTO pnl_history (total_pnl, daily_pnl, realized_pnl, unrealized_pnl,
                                   positions_count, positions_json, win_rate, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pnl_data['total_pnl'],
            pnl_data['daily_pnl'],
            pnl_data.get('realized_pnl'),
            pnl_data.get('unrealized_pnl'),
            pnl_data.get('positions_count', 0),
            positions_json,
            pnl_data.get('win_rate'),
            pnl_data.get('sharpe_ratio')
        ))
        self.conn.commit()
        return cursor.lastrowid
        
    def save_news(self, news_item: Dict[str, Any]) -> Optional[int]:
        """Save news item, skip if URL already exists."""
        cursor = self.conn.cursor()
        try:
            symbols = ','.join(news_item.get('symbols', []))
            cursor.execute('''
                INSERT INTO news_archive (title, url, source, sentiment, symbols, category, ai_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                news_item['title'],
                news_item.get('url'),
                news_item.get('source'),
                news_item.get('sentiment'),
                symbols,
                news_item.get('category'),
                news_item.get('ai_summary')
            ))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # URL already exists
            return None
            
    def save_ai_decision(self, decision: Dict[str, Any]) -> int:
        """Save AI decision and reasoning."""
        cursor = self.conn.cursor()
        event_data_json = json.dumps(decision.get('event_data', {}))
        cursor.execute('''
            INSERT INTO ai_decisions (event_type, event_data, decision, confidence,
                                    reasoning, outcome, outcome_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision['event_type'],
            event_data_json,
            decision['decision'],
            decision['confidence'],
            decision.get('reasoning'),
            decision.get('outcome', 'PENDING'),
            decision.get('outcome_pnl')
        ))
        self.conn.commit()
        return cursor.lastrowid
        
    def update_decision_outcome(self, decision_id: int, outcome: str, pnl: Optional[float] = None):
        """Update the outcome of a previous AI decision."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE ai_decisions 
            SET outcome = ?, outcome_pnl = ?
            WHERE id = ?
        ''', (outcome, pnl, decision_id))
        self.conn.commit()
        
    def get_recent_trades(self, symbol: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol."""
        cursor = self.conn.cursor()
        if symbol:
            cursor.execute('''
                SELECT * FROM trades 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def get_today_pnl(self) -> Dict[str, float]:
        """Get today's P&L summary."""
        cursor = self.conn.cursor()
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        cursor.execute('''
            SELECT 
                SUM(pnl) as total_pnl,
                COUNT(*) as trade_count,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses
            FROM trades 
            WHERE timestamp >= ? AND timestamp < ?
        ''', (today_start, today_end))
        
        result = cursor.fetchone()
        if result and result['total_pnl'] is not None:
            win_rate = result['wins'] / result['trade_count'] if result['trade_count'] > 0 else 0
            return {
                'total_pnl': result['total_pnl'],
                'trade_count': result['trade_count'],
                'win_rate': win_rate,
                'wins': result['wins'],
                'losses': result['losses']
            }
        return {'total_pnl': 0, 'trade_count': 0, 'win_rate': 0, 'wins': 0, 'losses': 0}
        
    def get_previous_day_pnl(self) -> Optional[Dict[str, Any]]:
        """Get the most recent P&L snapshot from a previous day."""
        cursor = self.conn.cursor()
        today = date.today().isoformat()
        
        cursor.execute('''
            SELECT * FROM pnl_history 
            WHERE DATE(timestamp) < ?
            ORDER BY timestamp DESC 
            LIMIT 1
        ''', (today,))
        
        result = cursor.fetchone()
        if result:
            data = dict(result)
            if data['positions_json']:
                data['positions'] = json.loads(data['positions_json'])
            return data
        return None
        
    def get_recent_options_flow(self, symbol: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent unusual options activity."""
        cursor = self.conn.cursor()
        if symbol:
            cursor.execute('''
                SELECT * FROM options_flow 
                WHERE symbol = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT * FROM options_flow 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
        
    def get_performance_metrics(self, days: int = 30) -> Dict[str, Any]:
        """Calculate performance metrics over specified days."""
        cursor = self.conn.cursor()
        cutoff_date = datetime.now().date()
        cutoff_date = cutoff_date.replace(day=max(1, cutoff_date.day - days))
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades
            FROM trades 
            WHERE DATE(timestamp) >= ?
        ''', (cutoff_date.isoformat(),))
        
        result = cursor.fetchone()
        if result and result['total_trades'] > 0:
            return {
                'total_trades': result['total_trades'],
                'total_pnl': result['total_pnl'] or 0,
                'avg_pnl': result['avg_pnl'] or 0,
                'best_trade': result['best_trade'] or 0,
                'worst_trade': result['worst_trade'] or 0,
                'win_rate': result['winning_trades'] / result['total_trades'],
                'winning_trades': result['winning_trades'],
                'losing_trades': result['losing_trades']
            }
        return {
            'total_trades': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'win_rate': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
    def close(self):
        """Close database connection."""
        self.conn.close()
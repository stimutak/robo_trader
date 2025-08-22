"""Database module for persistent storage of trading data."""

import sqlite3
import json
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def json_serialize_with_datetime(obj):
    """Custom JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


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
        
        # LLM decision tracking - detailed schema-based decisions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                prompt_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                mode TEXT NOT NULL,  -- trade, adjust, exit, neutral, watchlist
                symbol TEXT,
                direction TEXT,  -- long, short, flat
                conviction INTEGER NOT NULL,
                entry_price REAL,
                stop_price REAL,
                target_price REAL,
                position_size_bps INTEGER,
                expected_value_pct REAL,
                risk_reward_ratio REAL,
                p_win REAL,
                raw_decision_json TEXT NOT NULL,
                market_snapshot_json TEXT,
                latency_ms INTEGER,
                executed BOOLEAN DEFAULT FALSE,
                execution_id TEXT,
                actual_pnl REAL,
                actual_outcome TEXT  -- win, loss, scratch, timeout
            )
        ''')
        
        # Calibration tracking - for Brier scores and model performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                period_days INTEGER NOT NULL,  -- 30, 60, 90
                total_decisions INTEGER NOT NULL,
                trade_rate REAL,  -- % that resulted in trades
                win_rate REAL,
                avg_conviction REAL,
                brier_score REAL,  -- Calibration metric
                reliability REAL,  -- Slope of reliability plot
                resolution REAL,  -- Ability to discriminate
                avg_ev_error REAL,  -- Average error in EV prediction
                sharpe_ratio REAL,
                max_drawdown_pct REAL
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
        
        # Price history - store intraday price data for charts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                trading_day DATE NOT NULL,
                minute_index INTEGER,  -- Minutes since market open (0-389)
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_flow(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_timestamp ON news_archive(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_timestamp ON ai_decisions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_day ON price_history(symbol, trading_day)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)')
        
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
        
    def save_price_point(self, symbol: str, price: float, timestamp: datetime = None, minute_index: int = None) -> int:
        """Save a price point to the database."""
        if timestamp is None:
            timestamp = datetime.now()
        
        trading_day = timestamp.date()
        
        # Calculate minute index if not provided
        if minute_index is None:
            market_open = timestamp.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = int((timestamp - market_open).total_seconds() / 60)
            minute_index = max(0, min(389, minutes_since_open))
        
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO price_history (symbol, timestamp, price, trading_day, minute_index)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, timestamp, price, trading_day, minute_index))
            self.conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # Update if duplicate
            cursor.execute('''
                UPDATE price_history 
                SET price = ?, minute_index = ?
                WHERE symbol = ? AND timestamp = ?
            ''', (price, minute_index, symbol, timestamp))
            self.conn.commit()
            return cursor.lastrowid
    
    def save_llm_decision(self, decision_data: Dict[str, Any]) -> int:
        """Save an LLM trading decision to the database."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO llm_decisions (
                decision_id, prompt_hash, model_id, prompt_version,
                mode, symbol, direction, conviction,
                entry_price, stop_price, target_price, position_size_bps,
                expected_value_pct, risk_reward_ratio, p_win,
                raw_decision_json, market_snapshot_json, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision_data['decision_id'],
            decision_data['prompt_hash'],
            decision_data['model_id'],
            decision_data['prompt_version'],
            decision_data['mode'],
            decision_data.get('symbol'),
            decision_data.get('direction'),
            decision_data['conviction'],
            decision_data.get('entry_price'),
            decision_data.get('stop_price'),
            decision_data.get('target_price'),
            decision_data.get('position_size_bps'),
            decision_data.get('expected_value_pct'),
            decision_data.get('risk_reward_ratio'),
            decision_data.get('p_win'),
            json.dumps(decision_data['raw_decision'], default=json_serialize_with_datetime),
            json.dumps(decision_data.get('market_snapshot', {}), default=json_serialize_with_datetime),
            decision_data.get('latency_ms', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_llm_decision_outcome(
        self, 
        decision_id: str, 
        executed: bool, 
        execution_id: Optional[str] = None,
        actual_pnl: Optional[float] = None,
        actual_outcome: Optional[str] = None
    ):
        """Update an LLM decision with execution outcome."""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE llm_decisions 
            SET executed = ?, execution_id = ?, actual_pnl = ?, actual_outcome = ?
            WHERE decision_id = ?
        ''', (executed, execution_id, actual_pnl, actual_outcome, decision_id))
        self.conn.commit()
    
    def get_calibration_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get LLM decision data for calibration analysis."""
        cursor = self.conn.cursor()
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT 
                conviction, p_win, expected_value_pct,
                risk_reward_ratio, actual_pnl, actual_outcome,
                mode, direction, symbol, timestamp
            FROM llm_decisions
            WHERE timestamp >= ? AND executed = TRUE
            ORDER BY timestamp DESC
        ''', (cutoff,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def save_calibration_metrics(self, metrics: Dict[str, Any]) -> int:
        """Save calibration metrics for model performance tracking."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO calibration_metrics (
                period_days, total_decisions, trade_rate, win_rate,
                avg_conviction, brier_score, reliability, resolution,
                avg_ev_error, sharpe_ratio, max_drawdown_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['period_days'],
            metrics['total_decisions'],
            metrics.get('trade_rate', 0),
            metrics.get('win_rate', 0),
            metrics.get('avg_conviction', 0),
            metrics.get('brier_score', 0),
            metrics.get('reliability', 0),
            metrics.get('resolution', 0),
            metrics.get('avg_ev_error', 0),
            metrics.get('sharpe_ratio', 0),
            metrics.get('max_drawdown_pct', 0)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_last_trading_day_prices(self, symbol: str) -> List[Dict[str, Any]]:
        """Get the last full trading day's price data for a symbol."""
        cursor = self.conn.cursor()
        
        # Get the most recent trading day with data
        cursor.execute('''
            SELECT MAX(trading_day) as last_day 
            FROM price_history 
            WHERE symbol = ?
        ''', (symbol,))
        
        result = cursor.fetchone()
        if not result or not result['last_day']:
            return []
        
        last_day = result['last_day']
        
        # Get all prices from that day
        cursor.execute('''
            SELECT timestamp, price, minute_index 
            FROM price_history 
            WHERE symbol = ? AND trading_day = ?
            ORDER BY minute_index
        ''', (symbol, last_day))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_current_day_prices(self, symbol: str) -> List[Dict[str, Any]]:
        """Get today's price data for a symbol."""
        today = date.today()
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, price, minute_index 
            FROM price_history 
            WHERE symbol = ? AND trading_day = ?
            ORDER BY minute_index
        ''', (symbol, today))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def save_pnl_point(self, total_pnl: float, timestamp: datetime = None) -> int:
        """Save a P&L data point (simplified version for chart)."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Also save as regular P&L snapshot
        pnl_data = {
            'total_pnl': total_pnl,
            'daily_pnl': total_pnl,  # For now, can be refined later
        }
        return self.save_pnl_snapshot(pnl_data)
    
    def get_last_pnl_history(self, limit: int = 390) -> List[Dict[str, Any]]:
        """Get the most recent P&L history points."""
        cursor = self.conn.cursor()
        
        # Get today's P&L points first
        today_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        cursor.execute('''
            SELECT timestamp, total_pnl 
            FROM pnl_history 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (today_start, limit))
        
        results = cursor.fetchall()
        
        # If no data today, get the last trading day's data
        if not results:
            cursor.execute('''
                SELECT timestamp, total_pnl 
                FROM pnl_history 
                WHERE DATE(timestamp) = (
                    SELECT MAX(DATE(timestamp)) 
                    FROM pnl_history
                )
                ORDER BY timestamp
            ''')
            results = cursor.fetchall()
        
        return [dict(row) for row in results]
    
    def close(self):
        """Close database connection."""
        self.conn.close()
#!/Users/oliver/robo_trader/venv/bin/python
"""
Trading Dashboard Server

Provides real-time visualization of trading activity, positions, and P&L.
Runs on http://localhost:5555
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template_string
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("trading_data.db")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Robo Trader Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e6ed;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h1 { font-size: 2em; margin-bottom: 10px; }
        .status { 
            display: inline-block;
            padding: 5px 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 5px;
            margin-right: 10px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #1a1f3a;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #2d3561;
        }
        .card h2 {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #9ca3af;
        }
        .metric {
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .neutral { color: #6b7280; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #2d3561;
        }
        th { color: #9ca3af; font-weight: normal; }
        .chart {
            height: 200px;
            background: #0f1729;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6b7280;
            margin-top: 10px;
        }
        .timestamp {
            color: #6b7280;
            font-size: 0.9em;
            margin-top: 10px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Robo Trader Dashboard</h1>
        <span class="status">Mode: <strong id="mode">PAPER</strong></span>
        <span class="status">Status: <strong id="status">CONNECTING...</strong></span>
        <span class="status">Time: <strong id="time">--:--:--</strong></span>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Account Overview</h2>
            <div class="metric" id="equity">$0.00</div>
            <div>Total Equity</div>
            <table>
                <tr>
                    <td>Cash</td>
                    <td id="cash">$0.00</td>
                </tr>
                <tr>
                    <td>Positions</td>
                    <td id="position-value">$0.00</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Today's P&L</h2>
            <div class="metric neutral" id="daily-pnl">$0.00</div>
            <div id="daily-pnl-pct">0.00%</div>
            <table>
                <tr>
                    <td>Realized</td>
                    <td id="realized-pnl">$0.00</td>
                </tr>
                <tr>
                    <td>Unrealized</td>
                    <td id="unrealized-pnl">$0.00</td>
                </tr>
            </table>
        </div>

        <div class="card">
            <h2>Trading Activity</h2>
            <div class="metric" id="trade-count">0</div>
            <div>Trades Today</div>
            <table>
                <tr>
                    <td>Win Rate</td>
                    <td id="win-rate">0%</td>
                </tr>
                <tr>
                    <td>Avg Trade</td>
                    <td id="avg-trade">$0.00</td>
                </tr>
            </table>
        </div>
    </div>

    <div class="card">
        <h2>Active Positions</h2>
        <div id="positions-loading" class="loading">Loading positions...</div>
        <table id="positions-table" style="display:none;">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                    <th>Avg Cost</th>
                    <th>Market</th>
                    <th>P&L</th>
                    <th>P&L %</th>
                </tr>
            </thead>
            <tbody id="positions-body">
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Recent Trades</h2>
        <div id="trades-loading" class="loading">Loading trades...</div>
        <table id="trades-table" style="display:none;">
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Price</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody id="trades-body">
            </tbody>
        </table>
    </div>

    <div class="card">
        <h2>Equity Curve</h2>
        <div class="chart">Chart will be implemented in next iteration</div>
    </div>

    <div class="timestamp">Last updated: <span id="last-update">Never</span></div>

    <script>
        function formatCurrency(value) {
            const formatted = new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
            return formatted;
        }

        function formatPercent(value) {
            return (value * 100).toFixed(2) + '%';
        }

        function updateTime() {
            const now = new Date();
            document.getElementById('time').textContent = now.toLocaleTimeString();
        }

        async function fetchData() {
            try {
                const response = await fetch('/api/dashboard');
                const data = await response.json();
                
                // Update status
                document.getElementById('status').textContent = data.status || 'ACTIVE';
                document.getElementById('mode').textContent = data.mode || 'PAPER';
                
                // Update account overview
                document.getElementById('equity').textContent = formatCurrency(data.equity || 0);
                document.getElementById('cash').textContent = formatCurrency(data.cash || 0);
                document.getElementById('position-value').textContent = formatCurrency(data.position_value || 0);
                
                // Update P&L
                const dailyPnl = data.daily_pnl || 0;
                const dailyPnlElement = document.getElementById('daily-pnl');
                dailyPnlElement.textContent = formatCurrency(dailyPnl);
                dailyPnlElement.className = 'metric ' + (dailyPnl > 0 ? 'positive' : dailyPnl < 0 ? 'negative' : 'neutral');
                
                document.getElementById('daily-pnl-pct').textContent = formatPercent(data.daily_pnl_pct || 0);
                document.getElementById('realized-pnl').textContent = formatCurrency(data.realized_pnl || 0);
                document.getElementById('unrealized-pnl').textContent = formatCurrency(data.unrealized_pnl || 0);
                
                // Update trading activity
                document.getElementById('trade-count').textContent = data.trade_count || 0;
                document.getElementById('win-rate').textContent = formatPercent(data.win_rate || 0);
                document.getElementById('avg-trade').textContent = formatCurrency(data.avg_trade || 0);
                
                // Update positions
                if (data.positions && data.positions.length > 0) {
                    document.getElementById('positions-loading').style.display = 'none';
                    document.getElementById('positions-table').style.display = 'table';
                    const positionsBody = document.getElementById('positions-body');
                    positionsBody.innerHTML = data.positions.map(p => `
                        <tr>
                            <td>${p.symbol}</td>
                            <td>${p.quantity}</td>
                            <td>${formatCurrency(p.avg_cost)}</td>
                            <td>${formatCurrency(p.market_price)}</td>
                            <td class="${p.pnl > 0 ? 'positive' : p.pnl < 0 ? 'negative' : ''}">${formatCurrency(p.pnl)}</td>
                            <td class="${p.pnl_pct > 0 ? 'positive' : p.pnl_pct < 0 ? 'negative' : ''}">${formatPercent(p.pnl_pct)}</td>
                        </tr>
                    `).join('');
                }
                
                // Update recent trades
                if (data.recent_trades && data.recent_trades.length > 0) {
                    document.getElementById('trades-loading').style.display = 'none';
                    document.getElementById('trades-table').style.display = 'table';
                    const tradesBody = document.getElementById('trades-body');
                    tradesBody.innerHTML = data.recent_trades.map(t => `
                        <tr>
                            <td>${new Date(t.timestamp).toLocaleTimeString()}</td>
                            <td>${t.symbol}</td>
                            <td class="${t.side === 'BUY' ? 'positive' : 'negative'}">${t.side}</td>
                            <td>${t.quantity}</td>
                            <td>${formatCurrency(t.price)}</td>
                            <td>${formatCurrency(t.value)}</td>
                        </tr>
                    `).join('');
                }
                
                document.getElementById('last-update').textContent = new Date().toLocaleString();
            } catch (error) {
                console.error('Error fetching data:', error);
                document.getElementById('status').textContent = 'ERROR';
            }
        }

        // Update time every second
        setInterval(updateTime, 1000);
        
        // Fetch data every 5 seconds
        setInterval(fetchData, 5000);
        
        // Initial load
        updateTime();
        fetchData();
    </script>
</body>
</html>
"""


def init_database():
    """Initialize database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tables if they don't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            market_price REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY,
            cash REAL NOT NULL,
            equity REAL NOT NULL,
            daily_pnl REAL,
            realized_pnl REAL,
            unrealized_pnl REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Insert default account if not exists
    cursor.execute(
        """
        INSERT OR IGNORE INTO account (id, cash, equity) VALUES (1, 100000, 100000)
    """
    )

    conn.commit()
    conn.close()


def get_dashboard_data() -> Dict[str, Any]:
    """Fetch current dashboard data from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get account data
    cursor.execute("SELECT * FROM account WHERE id = 1")
    account = cursor.fetchone()

    # Get positions
    cursor.execute(
        """
        SELECT symbol, quantity, avg_cost, market_price 
        FROM positions 
        WHERE quantity != 0
    """
    )
    positions = cursor.fetchall()

    # Get today's trades
    today = datetime.now().date()
    cursor.execute(
        """
        SELECT symbol, side, quantity, price, timestamp 
        FROM trades 
        WHERE DATE(timestamp) = DATE('now')
        ORDER BY timestamp DESC
        LIMIT 20
    """
    )
    trades = cursor.fetchall()

    conn.close()

    # Calculate metrics
    position_value = sum(p[1] * (p[3] or p[2]) for p in positions)

    positions_data = []
    for symbol, qty, avg_cost, market_price in positions:
        market_price = market_price or avg_cost
        pnl = (market_price - avg_cost) * qty
        pnl_pct = (market_price / avg_cost - 1) if avg_cost > 0 else 0
        positions_data.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "avg_cost": avg_cost,
                "market_price": market_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        )

    trades_data = []
    for symbol, side, qty, price, timestamp in trades:
        trades_data.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": price,
                "value": qty * price,
                "timestamp": timestamp,
            }
        )

    # Calculate win rate
    winning_trades = sum(1 for t in trades if t[1] == "SELL")  # Simplified
    total_trades = len(trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0

    return {
        "status": "ACTIVE",
        "mode": "PAPER",
        "cash": account[1] if account else 100000,
        "equity": account[2] if account else 100000,
        "position_value": position_value,
        "daily_pnl": account[3] if account else 0,
        "daily_pnl_pct": (account[3] / account[2]) if account and account[2] > 0 else 0,
        "realized_pnl": account[4] if account else 0,
        "unrealized_pnl": account[5] if account else 0,
        "trade_count": len(trades),
        "win_rate": win_rate,
        "avg_trade": sum(t[2] * t[3] for t in trades) / len(trades) if trades else 0,
        "positions": positions_data,
        "recent_trades": trades_data,
    }


@app.route("/")
def index():
    """Serve the dashboard HTML."""
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/dashboard")
def dashboard_api():
    """API endpoint for dashboard data."""
    try:
        data = get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching dashboard data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_position", methods=["POST"])
def update_position():
    """Update position data (called by trading system)."""
    # This endpoint will be used by the trading system to update positions
    return jsonify({"status": "ok"})


def main():
    """Run the dashboard server."""
    logger.info("Initializing database...")
    init_database()

    logger.info("Starting dashboard server on http://localhost:5555")
    app.run(host="0.0.0.0", port=5555, debug=False)


if __name__ == "__main__":
    main()

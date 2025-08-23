#!/usr/bin/env python3
"""
Cursor-style Dashboard for Robo Trader
Ultra-minimal design inspired by cursor.com/dashboard
"""

from flask import Flask, render_template_string, jsonify, request, Response
import asyncio
import threading
import json
from datetime import datetime
import subprocess
import os
import signal
import glob
import time
import hashlib
from functools import wraps
from robo_trader.config import load_config
from robo_trader.logger import get_logger
from robo_trader.database import TradingDatabase

app = Flask(__name__)
logger = get_logger(__name__)

# Authentication configuration
AUTH_ENABLED = os.getenv('DASH_AUTH_ENABLED', 'false').lower() == 'true'
AUTH_USER = os.getenv('DASH_USER', 'admin')
AUTH_PASS_HASH = os.getenv('DASH_PASS_HASH', '')  # SHA256 hash of password

def check_auth(username, password):
    """Check if username/password is valid."""
    if not AUTH_ENABLED:
        return True
    if not AUTH_PASS_HASH:
        return True  # No password set, allow access
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return username == AUTH_USER and password_hash == AUTH_PASS_HASH

def authenticate():
    """Send 401 response that enables basic auth."""
    return Response(
        'Authentication required to access the trading dashboard.\n'
        'Please enter your username and password.',
        401,
        {'WWW-Authenticate': 'Basic realm="Robo Trader Dashboard"'}
    )

def requires_auth(f):
    """Decorator to require authentication for routes."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Initialize database
db = TradingDatabase()

# Global state
trading_process = None
trading_status = "stopped"
trading_log = []
positions = {}
pnl = {"daily": 0.0, "total": 0.0}
ai_decisions = []
news_feed = [
    {
        'title': 'Fed Minutes Show Officials Saw Inflation Progress',
        'source': 'Bloomberg',
        'sentiment': 0.3,
        'time': '10:05',
        'url': 'https://bloomberg.com'
    },
    {
        'title': 'Apple Announces New AI Features for iPhone',
        'source': 'Reuters',
        'sentiment': 0.6,
        'time': '09:45',
        'url': 'https://reuters.com'
    },
    {
        'title': 'Tech Stocks Rally on Strong Earnings',
        'source': 'CNBC',
        'sentiment': 0.8,
        'time': '09:30',
        'url': 'https://cnbc.com'
    }
]
trading_signals = []
options_flow = []
company_events = [
    {
        'symbol': 'NVDA',
        'type': '8-K',
        'description': 'Material Event - New Partnership Announced',
        'time': '09:15',
        'impact': 75
    },
    {
        'symbol': 'AAPL',
        'type': 'Form 4',
        'description': 'Insider Buying - CEO purchased 10,000 shares',
        'time': '08:30',
        'impact': 60
    },
    {
        'symbol': 'TSLA',
        'type': '10-Q',
        'description': 'Quarterly Report Filed',
        'time': '08:00',
        'impact': 50
    }
]  # Store SEC filings, earnings, FDA events
last_price_update_time = None  # Track when we last received price data

# Load user settings
USER_SETTINGS_FILE = "user_settings.json"

def load_user_settings():
    """Load saved user settings."""
    try:
        with open(USER_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {
            "default": {
                "symbols": ["AAPL", "NVDA", "TSLA", "IXHL", "NUAI", "BZAI", "ELTP", 
                           "OPEN", "CEG", "VRT", "PLTR", "UPST", 
                           "TEM", "HTFL", "SDGR", "APLD", "SOFI", "CORZ", "WULF",
                           "GLD", "BTC-USD", "ETH-USD"],
                "risk_level": "moderate",
                "max_daily_loss": 1000
            }
        }

def save_user_settings(settings):
    """Save user settings."""
    with open(USER_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

user_settings = load_user_settings()

def load_historical_data():
    """Load historical data from database on startup."""
    global pnl, positions, ai_decisions, trading_signals, options_flow
    
    try:
        # Load today's P&L
        today_pnl = db.get_today_pnl()
        pnl['daily'] = today_pnl.get('total_pnl', 0)
        
        # Load recent trades
        recent_trades = db.get_recent_trades(limit=50)
        for trade in recent_trades[:10]:  # Show last 10 trades
            trading_signals.append({
                'time': trade['timestamp'],
                'symbol': trade['symbol'],
                'action': trade['action'],
                'price': trade['price'],
                'confidence': trade.get('ai_confidence', 0)
            })
        
        # Load recent options flow
        recent_options = db.get_recent_options_flow(limit=20)
        for opt in recent_options:
            options_flow.append({
                'symbol': opt['symbol'],
                'type': opt['signal_type'],
                'strike': opt['strike'],
                'expiry': opt['expiry'],
                'confidence': opt.get('confidence', 0)
            })
        
        # Load performance metrics
        metrics = db.get_performance_metrics(days=30)
        pnl['total'] = metrics.get('total_pnl', 0)
        
        # If after hours, load previous day's P&L
        import datetime
        now = datetime.datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if now < market_open or now > market_close:
            prev_pnl = db.get_previous_day_pnl()
            if prev_pnl:
                pnl['daily'] = prev_pnl.get('daily_pnl', 0)
                pnl['total'] = prev_pnl.get('total_pnl', 0)
                
        logger.info(f"Loaded historical data: P&L ${pnl['total']:.2f}, {len(recent_trades)} trades")
        
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")

# Load historical data on startup
load_historical_data()

# Cursor-style HTML template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Robo Trader</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box;
        }
        
        :root {
            --bg: #0a0a0a;
            --surface: #111111;
            --surface-hover: #1a1a1a;
            --border: rgba(255, 255, 255, 0.06);
            --border-hover: rgba(255, 255, 255, 0.1);
            --text: #8b8b8b;
            --text-bright: #b4b4b4;
            --text-dim: #6b6b6b;
            --text-dimmer: #4a4a4a;
            --accent: #4a9eff;
            --accent-dim: rgba(74, 158, 255, 0.1);
            --accent-hover: #5eaaff;
            --green: #10b981;
            --red: #f87171;
            --blue: #4a9eff;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            font-size: 14px;
            -webkit-font-smoothing: antialiased;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 24px;
        }
        
        /* Header */
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 32px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .logo h1 {
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--text-bright);
        }
        
        .status-badge {
            padding: 4px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .status-badge.running {
            background: rgba(34, 197, 94, 0.1);
            color: var(--green);
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        
        .status-badge.stopped {
            background: rgba(239, 68, 68, 0.1);
            color: var(--red);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        
        /* Main Grid */
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }
        
        @media (max-width: 1024px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* Cards */
        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            transition: all 0.15s;
        }
        
        .card:hover {
            background: var(--surface-hover);
            border-color: var(--border-hover);
        }
        
        .card-title {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-dimmer);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 12px;
        }
        
        .card-value {
            font-size: 24px;
            font-weight: 500;
            margin-bottom: 4px;
            font-variant-numeric: tabular-nums;
            color: var(--text-bright);
        }
        
        .card-value.green { color: var(--green); }
        .card-value.red { color: var(--red); }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .market-open {
            background: rgba(34, 197, 94, 0.1);
            color: var(--green);
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        
        .market-closed {
            background: rgba(100, 100, 100, 0.1);
            color: #888;
            border: 1px solid rgba(100, 100, 100, 0.2);
        }
        
        .card-subtitle {
            font-size: 13px;
            color: var(--text-dim);
        }
        
        /* Control Buttons */
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
            margin-bottom: 24px;
        }
        
        .btn {
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid var(--border);
            background: var(--surface);
            color: var(--text);
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .btn:hover:not(:disabled) {
            background: var(--surface-hover);
            border-color: var(--border-hover);
        }
        
        .btn:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }
        
        .btn-primary {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }
        
        .btn-primary:hover:not(:disabled) {
            background: var(--accent-hover);
            border-color: var(--accent-hover);
        }
        
        .btn-danger {
            background: var(--red);
            border-color: var(--red);
            color: white;
        }
        
        .btn-danger:hover:not(:disabled) {
            opacity: 0.9;
        }
        
        /* Content Grid */
        .content-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
        }
        
        @media (max-width: 1024px) {
            .content-grid {
                grid-template-columns: 1fr;
            }
        }
        
        /* News Ticker */
        .ticker {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            overflow: hidden;
            position: relative;
        }
        
        .ticker-label {
            position: absolute;
            left: 12px;
            top: 12px;
            background: var(--surface);
            padding-right: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
            z-index: 1;
        }
        
        .ticker-content {
            white-space: nowrap;
            animation: scroll 60s linear infinite;
            padding-left: 60px;
            color: var(--text-dim);
            font-size: 13px;
        }
        
        @keyframes scroll {
            0% { transform: translateX(0); }
            100% { transform: translateX(-50%); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideUp {
            from { 
                opacity: 0;
                transform: translateY(20px);
            }
            to { 
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Lists */
        .list {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .list-header {
            padding: 16px 20px;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
            font-weight: 500;
            color: var(--text-dimmer);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .list-content {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .list-item {
            padding: 12px 20px;
            border-bottom: 1px solid var(--border);
            transition: background 0.15s;
            cursor: pointer;
        }
        
        .list-item:hover {
            background: var(--surface-hover);
        }
        
        .list-item:last-child {
            border-bottom: none;
        }
        
        .list-item-title {
            font-size: 13px;
            color: var(--text-bright);
            margin-bottom: 4px;
        }
        
        .list-item-meta {
            font-size: 12px;
            color: var(--text-dimmer);
            display: flex;
            gap: 12px;
        }
        
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }
        
        .badge-green {
            background: rgba(34, 197, 94, 0.1);
            color: var(--green);
        }
        
        .badge-red {
            background: rgba(239, 68, 68, 0.1);
            color: var(--red);
        }
        
        .badge-blue {
            background: var(--accent-dim);
            color: var(--accent);
        }
        
        /* Settings */
        .settings-grid {
            display: grid;
            gap: 16px;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .input-label {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-dimmer);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .input, .textarea, .select {
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text);
            font-size: 14px;
            font-family: inherit;
            transition: all 0.15s;
        }
        
        .input:focus, .textarea:focus, .select:focus {
            outline: none;
            border-color: var(--accent);
            background: var(--surface);
        }
        
        .textarea {
            resize: vertical;
            min-height: 60px;
            font-size: 12px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--border-hover);
        }
        
        /* Filter buttons */
        .filter-btn {
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 500;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 4px;
            color: var(--text-dimmer);
            cursor: pointer;
            transition: all 0.15s;
        }
        
        .filter-btn:hover {
            background: var(--surface-hover);
            border-color: var(--accent);
            color: var(--text);
        }
        
        .filter-btn.active {
            background: var(--accent);
            border-color: var(--accent);
            color: white;
        }
        
        /* Positions Table */
        .table {
            width: 100%;
            font-size: 13px;
        }
        
        .table th {
            text-align: left;
            padding: 8px 12px;
            font-size: 11px;
            font-weight: 500;
            color: var(--text-dimmer);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid var(--border);
        }
        
        .table td {
            padding: 12px;
            border-bottom: 1px solid var(--border);
            color: var(--text-dim);
        }
        
        .table tr:last-child td {
            border-bottom: none;
        }
        
        .table tr:hover td {
            background: var(--surface-hover);
        }
        
        /* Custom scrollbar for symbol list */
        #symbolSelectorContainer::-webkit-scrollbar {
            height: 6px;
        }
        
        #symbolSelectorContainer::-webkit-scrollbar-track {
            background: transparent;
        }
        
        #symbolSelectorContainer::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        #symbolSelectorContainer::-webkit-scrollbar-thumb:hover {
            background: var(--text-dimmer);
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <h1>Robo Trader</h1>
                <span id="status" class="status-badge stopped">Stopped</span>
                <span id="marketStatus" class="status-badge" style="margin-left: 10px;">Market Closed</span>
                <span id="liveIndicator" style="display: none; margin-left: 10px;">
                    <span style="display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: pulse 2s infinite;"></span>
                    <span style="color: #22c55e; font-size: 12px; margin-left: 5px;">LIVE DATA</span>
                </span>
            </div>
            <div style="color: var(--text-dim); font-size: 13px;">
                AI-Powered Trading • Claude 3.5 Sonnet
            </div>
        </div>
        
        <!-- Main Metrics -->
        <div class="main-grid">
            <div class="card">
                <div class="card-title">Daily P&L</div>
                <div id="dailyPnl" class="card-value">$0.00</div>
                <div class="card-subtitle">Today's Performance</div>
            </div>
            <div class="card">
                <div class="card-title">Total P&L</div>
                <div id="totalPnl" class="card-value">$0.00</div>
                <div class="card-subtitle">All Time</div>
            </div>
            <div class="card">
                <div class="card-title">Positions</div>
                <div id="posCount" class="card-value">0</div>
                <div class="card-subtitle">Active Trades</div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="controls">
            <button id="startBtn" class="btn btn-primary" onclick="startTrading()">
                Start Trading
            </button>
            <button id="stopBtn" class="btn btn-danger" onclick="stopTrading()" disabled>
                Stop Trading
            </button>
            <button class="btn" onclick="testAI()">
                Test AI Analysis
            </button>
        </div>
        
        <!-- News Ticker -->
        <div class="ticker">
            <div class="ticker-label">LIVE</div>
            <div id="tickerContent" class="ticker-content">
                Loading market news...
            </div>
        </div>
        
        <!-- Price Chart -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Price Chart <span id="assetTypeIndicator" style="margin-left: 8px;"></span></h2>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span id="marketHoursLabel" style="font-size: 11px; color: var(--text-dimmer);">Today (9:30 AM - 4:00 PM ET)</span>
                </div>
            </div>
            <!-- Symbol Selector List -->
            <div id="symbolSelectorContainer" style="
                padding: 8px 16px;
                background: var(--bg-darker);
                border-bottom: 1px solid var(--border);
                overflow-x: auto;
                overflow-y: hidden;
                white-space: nowrap;
                -webkit-overflow-scrolling: touch;
                scrollbar-width: thin;
                scrollbar-color: var(--border) transparent;
            ">
                <div id="symbolList" style="display: inline-flex; gap: 8px;">
                    <!-- Will be populated dynamically -->
                </div>
            </div>
            <div class="section-content" style="height: 250px; padding: 16px; position: relative;">
                <canvas id="priceChart"></canvas>
            </div>
        </div>
        
        <!-- Analytics Row -->
        <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 16px; margin: 16px 0;">
            <!-- AI Conviction Gauge -->
            <div class="section">
                <div class="section-header">
                    <h2 class="section-title">AI Conviction</h2>
                </div>
                <div class="section-content" style="height: 200px; display: flex; align-items: center; justify-content: center;">
                    <div style="position: relative; width: 180px; height: 180px;">
                        <canvas id="convictionGauge"></canvas>
                        <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center;">
                            <div id="convictionValue" style="font-size: 32px; font-weight: 600; color: var(--text-bright);">--</div>
                            <div id="convictionLabel" style="font-size: 12px; color: var(--text-dimmer); text-transform: uppercase; margin-top: 4px;">Waiting</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- P&L Chart -->
            <div class="section">
                <div class="section-header">
                    <h2 class="section-title">P&L History</h2>
                    <span style="font-size: 11px; color: var(--text-dimmer);">Today (9:30 AM - 4:00 PM ET)</span>
                </div>
                <div class="section-content" style="height: 200px; padding: 16px; position: relative;">
                    <canvas id="pnlChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Options Flow Section -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Options Flow</h2>
                <span id="flowSummary" style="font-size: 11px; color: var(--text-dimmer);">Scanning for unusual activity...</span>
            </div>
            <div class="section-content">
                <div id="optionsFlowContainer" style="min-height: 150px;">
                    <div class="empty-state">No unusual options activity detected</div>
                </div>
            </div>
        </div>
        
        <!-- Content Grid -->
        <div class="content-grid">
            <!-- Left Column -->
            <div style="display: flex; flex-direction: column; gap: 16px;">
                <!-- AI Decisions -->
                <div class="list">
                    <div class="list-header">AI Analysis</div>
                    <div id="aiDecisions" class="list-content">
                        <div class="list-item">
                            <div class="list-item-title">Waiting for market events...</div>
                            <div class="list-item-meta">
                                <span>No analysis yet</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- News Feed -->
                <div class="list">
                    <div class="list-header">Market News</div>
                    <div id="newsFeed" class="list-content">
                        <div class="list-item">
                            <div class="list-item-title">Loading news...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Positions -->
                <div class="list">
                    <div class="list-header">Positions</div>
                    <div id="positions" class="list-content">
                        <div class="list-item">
                            <div class="list-item-title">No positions</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Right Column -->
            <div style="display: flex; flex-direction: column; gap: 16px;">
                <!-- Settings -->
                <div class="card">
                    <div class="card-title">Settings</div>
                    <div class="settings-grid">
                        <div class="input-group">
                            <label class="input-label">Symbols</label>
                            <textarea id="symbols" class="textarea">AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,ADA,HBAR,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF</textarea>
                        </div>
                        <div class="input-group">
                            <label class="input-label">Risk Level</label>
                            <select id="riskLevel" class="select">
                                <option value="conservative">Conservative</option>
                                <option value="moderate" selected>Moderate</option>
                                <option value="aggressive">Aggressive</option>
                            </select>
                        </div>
                        <div class="input-group">
                            <label class="input-label">Max Daily Loss</label>
                            <input id="maxDailyLoss" class="input" value="1000" />
                        </div>
                        <button onclick="saveSettings()" class="btn">
                            Save Settings
                        </button>
                    </div>
                </div>
                
                <!-- Trading Signals -->
                <div class="list">
                    <div class="list-header">Trading Signals</div>
                    <div id="tradingSignals" class="list-content">
                        <div class="list-item">
                            <div class="list-item-title">No signals yet</div>
                        </div>
                    </div>
                </div>
                
                <!-- Company Events (SEC Filings, Earnings, FDA) -->
                <div class="list">
                    <div class="list-header" style="flex-direction: column; gap: 8px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; width: 100%;">
                            <span>Company Events</span>
                            <span style="font-size: 10px; color: var(--accent);">SEC • EARNINGS • FDA</span>
                        </div>
                        <div class="event-filters" style="display: flex; gap: 6px; flex-wrap: wrap;">
                            <button class="filter-btn active" data-filter="all" onclick="filterCompanyEvents('all')">All</button>
                            <button class="filter-btn" data-filter="SEC" onclick="filterCompanyEvents('SEC')">SEC</button>
                            <button class="filter-btn" data-filter="8-K" onclick="filterCompanyEvents('8-K')">8-K</button>
                            <button class="filter-btn" data-filter="10-" onclick="filterCompanyEvents('10-Q/K')">10-Q/K</button>
                            <button class="filter-btn" data-filter="Form 4" onclick="filterCompanyEvents('Form 4')">Form 4</button>
                            <button class="filter-btn" data-filter="Earnings" onclick="filterCompanyEvents('Earnings')">Earnings</button>
                            <button class="filter-btn" data-filter="FDA" onclick="filterCompanyEvents('FDA')">FDA</button>
                        </div>
                    </div>
                    <div id="companyEvents" class="list-content" style="max-height: 300px;">
                        <div class="list-item">
                            <div class="list-item-title">Monitoring SEC filings...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Activity Log -->
                <div class="list">
                    <div class="list-header">Activity</div>
                    <div id="log" class="list-content" style="max-height: 200px;">
                        <div class="list-item">
                            <div class="list-item-title">System ready</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Load settings on page load
        loadSettings();
        
        // Chart setup
        let priceChart = null;
        let convictionGauge = null;
        let pnlChart = null;
        let allCompanyEvents = [];  // Store all events for filtering
        let activeEventFilter = 'all';  // Track active filter
        let priceHistory = {};
        let pnlHistory = [];
        let currentChartSymbol = 'AAPL';  // Default to first watchlist symbol
        let watchlistSymbols = [];
        let symbolInfo = {};  // Store asset type info
        let currentConviction = 0;
        let currentDirection = 'neutral';
        
        function initConvictionGauge() {
            const ctx = document.getElementById('convictionGauge').getContext('2d');
            convictionGauge = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [0, 100],
                        backgroundColor: [
                            'rgba(74, 158, 255, 0.8)',  // Conviction color
                            'rgba(255, 255, 255, 0.03)' // Background
                        ],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    circumference: 180,
                    rotation: 270,
                    cutout: '75%',
                    plugins: {
                        legend: { display: false },
                        tooltip: { enabled: false }
                    }
                }
            });
        }
        
        function initPnLChart() {
            const ctx = document.getElementById('pnlChart').getContext('2d');
            
            // Pre-generate full trading day labels and empty data
            const fullDaySize = 390;
            const labels = [];
            const dataPoints = [];
            
            // Generate time labels for full trading day - ALL points need labels for chart.js
            const startHour = 9;
            const startMinute = 30;
            for (let i = 0; i < fullDaySize; i++) {
                const totalMinutes = startHour * 60 + startMinute + i;
                const hour = Math.floor(totalMinutes / 60);
                const minute = totalMinutes % 60;
                // Add label for every minute (chart.js will auto-skip as needed)
                labels.push(`${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`);
                dataPoints.push(null);  // Start with null data
            }
            
            pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,  // Start with full day labels
                    datasets: [{
                        label: 'P&L',
                        data: dataPoints,  // Start with null data
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34, 197, 94, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        spanGaps: false  // Don't connect null values
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(17, 17, 17, 0.95)',
                            titleColor: '#b4b4b4',
                            bodyColor: '#8b8b8b',
                            borderColor: '#2a2a2a',
                            borderWidth: 1,
                            padding: 10,
                            displayColors: false,
                            callbacks: {
                                label: function(context) {
                                    const value = context.parsed.y;
                                    return (value >= 0 ? '+' : '') + '$' + value.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.03)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#5a5a5a',
                                font: { size: 10 },
                                maxRotation: 0,
                                autoSkip: true,  // Let Chart.js auto-skip labels
                                maxTicksLimit: 13,  // Show about 13 ticks (every 30 mins)
                                callback: function(value, index) {
                                    // Only show time at 30-minute intervals
                                    if (index % 30 === 0) {
                                        return this.getLabelForValue(value);
                                    }
                                    return '';
                                }
                            }
                        },
                        y: {
                            position: 'right',
                            min: -120,  // Fixed initial scale
                            max: 120,   // Will be adjusted as data comes in
                            grid: {
                                color: 'rgba(255, 255, 255, 0.03)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#5a5a5a',
                                font: { size: 10 },
                                callback: function(value) {
                                    return '$' + value.toFixed(0);
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function initChart() {
            const ctx = document.getElementById('priceChart').getContext('2d');
            
            // Pre-generate full trading day labels
            const fullDaySize = 390;
            const labels = [];
            const dataPoints = [];
            
            // Generate time labels for full trading day
            const startHour = 9;
            const startMinute = 30;
            for (let i = 0; i < fullDaySize; i++) {
                const totalMinutes = startHour * 60 + startMinute + i;
                const hour = Math.floor(totalMinutes / 60);
                const minute = totalMinutes % 60;
                labels.push(`${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`);
                dataPoints.push(null);
            }
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,  // Start with full day labels
                    datasets: [{
                        label: 'Price',
                        data: dataPoints,  // Start with null data
                        borderColor: '#4a9eff',
                        backgroundColor: 'rgba(74, 158, 255, 0.05)',
                        borderWidth: 2,
                        tension: 0,  // No curve, straight lines
                        pointRadius: 3,  // Show points where we have data
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#4a9eff',
                        pointBorderColor: '#4a9eff',
                        pointHoverBackgroundColor: '#4a9eff',
                        pointHoverBorderColor: '#4a9eff',
                        spanGaps: false,  // Don't connect sparse points with lines
                        stepped: false,  // Remove stepped line
                        showLine: false  // Only show points, no connecting lines for sparse data
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: {
                            display: false
                        },
                        tooltip: {
                            backgroundColor: 'rgba(17, 17, 17, 0.95)',
                            titleColor: '#b4b4b4',
                            bodyColor: '#8b8b8b',
                            borderColor: '#2a2a2a',
                            borderWidth: 1,
                            padding: 10,
                            displayColors: false,
                            callbacks: {
                                label: function(context) {
                                    return '$' + context.parsed.y.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.03)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#5a5a5a',
                                font: {
                                    size: 10
                                },
                                maxRotation: 0,
                                autoSkip: true,
                                maxTicksLimit: 14  // Show hourly ticks for full trading day
                            }
                        },
                        y: {
                            position: 'right',
                            grid: {
                                color: 'rgba(255, 255, 255, 0.03)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#5a5a5a',
                                font: {
                                    size: 10
                                },
                                callback: function(value) {
                                    return '$' + value.toFixed(0);
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Make showAIDecisionDetail globally accessible for clicks
        window.showAIDecisionDetail = function(decision) {
            // Create modal for AI decision details
            const actionColor = decision.action === 'BUY' ? '#22c55e' : 
                              decision.action === 'SELL' ? '#ef4444' : 
                              decision.action === 'HOLD' ? '#f59e0b' : '#4a9eff';
            
            const confidenceColor = decision.confidence >= 80 ? '#22c55e' : 
                                  decision.confidence >= 60 ? '#4a9eff' :
                                  decision.confidence >= 40 ? '#f59e0b' : '#ef4444';
            
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: fadeIn 0.2s;
            `;
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
                max-width: 600px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                animation: slideUp 0.2s;
            `;
            
            // Parse additional details if available
            const watchlist = decision.watchlist || [];
            const thesis = decision.thesis || {};
            const targets = decision.targets || [];
            const stops = decision.stops || {};
            const raw_decision = decision.raw_decision || {};
            const compliance = raw_decision.compliance_checks || {};
            const risk_state = raw_decision.risk_state || {};
            const recommendation = raw_decision.recommendation || {};
            
            // Build decision tree HTML
            let decisionTreeHTML = '';
            if (raw_decision && Object.keys(raw_decision).length > 0) {
                decisionTreeHTML = `
                <div style="margin-bottom: 24px;">
                    <h4 style="color: var(--text-bright); font-size: 14px; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 16px;">🌳</span> Decision Tree Analysis
                    </h4>
                    
                    <!-- Step 1: Market Analysis -->
                    <div style="
                        background: linear-gradient(135deg, rgba(74, 158, 255, 0.1) 0%, rgba(74, 158, 255, 0.05) 100%);
                        border: 1px solid rgba(74, 158, 255, 0.3);
                        border-radius: 8px;
                        padding: 14px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 12px; font-weight: 600; color: #4a9eff; margin-bottom: 8px;">
                            1️⃣ Market Context
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">Trading Mode</span>
                                <div style="color: var(--text); font-size: 12px; font-weight: 500; margin-top: 2px;">
                                    ${raw_decision.mode || 'Analyzing'}
                                </div>
                            </div>
                            <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">Symbols Scanned</span>
                                <div style="color: var(--text); font-size: 12px; font-weight: 500; margin-top: 2px;">
                                    ${(raw_decision.universe_checked || []).length} tickers
                                </div>
                            </div>
                            ${risk_state.day_dd_bps !== undefined ? `
                            <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">Daily Drawdown</span>
                                <div style="color: ${risk_state.day_dd_bps > 100 ? '#ef4444' : '#22c55e'}; font-size: 12px; font-weight: 500; margin-top: 2px;">
                                    ${(risk_state.day_dd_bps / 100).toFixed(2)}%
                                </div>
                            </div>
                            ` : ''}
                            ${risk_state.cash_pct !== undefined ? `
                            <div style="background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">Cash Available</span>
                                <div style="color: var(--text); font-size: 12px; font-weight: 500; margin-top: 2px;">
                                    ${risk_state.cash_pct.toFixed(1)}%
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    <!-- Step 2: Compliance Checks -->
                    <div style="
                        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
                        border: 1px solid rgba(34, 197, 94, 0.3);
                        border-radius: 8px;
                        padding: 14px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 12px; font-weight: 600; color: #22c55e; margin-bottom: 8px;">
                            2️⃣ Risk & Compliance
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px;">
                            ${compliance.liquidity_ok !== undefined ? `
                            <div style="display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: ${compliance.liquidity_ok ? '#22c55e' : '#ef4444'}; font-size: 14px;">
                                    ${compliance.liquidity_ok ? '✓' : '✗'}
                                </span>
                                <span style="color: var(--text-dim); font-size: 11px;">Liquidity OK</span>
                            </div>
                            ` : ''}
                            ${compliance.spread_ok !== undefined ? `
                            <div style="display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: ${compliance.spread_ok ? '#22c55e' : '#ef4444'}; font-size: 14px;">
                                    ${compliance.spread_ok ? '✓' : '✗'}
                                </span>
                                <span style="color: var(--text-dim); font-size: 11px;">Spread OK</span>
                            </div>
                            ` : ''}
                            ${compliance.borrow_ok !== undefined ? `
                            <div style="display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: ${compliance.borrow_ok ? '#22c55e' : '#ef4444'}; font-size: 14px;">
                                    ${compliance.borrow_ok ? '✓' : '✗'}
                                </span>
                                <span style="color: var(--text-dim); font-size: 11px;">Borrow Available</span>
                            </div>
                            ` : ''}
                            ${compliance.correlation_ok !== undefined ? `
                            <div style="display: flex; align-items: center; gap: 6px; background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: ${compliance.correlation_ok ? '#22c55e' : '#ef4444'}; font-size: 14px;">
                                    ${compliance.correlation_ok ? '✓' : '✗'}
                                </span>
                                <span style="color: var(--text-dim); font-size: 11px;">Correlation OK</span>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                    
                    <!-- Step 3: Trade Analysis -->
                    ${recommendation.symbol ? `
                    <div style="
                        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%);
                        border: 1px solid rgba(245, 158, 11, 0.3);
                        border-radius: 8px;
                        padding: 14px;
                        margin-bottom: 10px;
                    ">
                        <div style="font-size: 12px; font-weight: 600; color: #f59e0b; margin-bottom: 8px;">
                            3️⃣ Trade Setup
                        </div>
                        ${recommendation.thesis ? `
                        <div style="
                            background: rgba(0, 0, 0, 0.2);
                            padding: 8px;
                            border-radius: 4px;
                            margin-bottom: 8px;
                        ">
                            <div style="color: var(--text-dimmer); font-size: 10px; margin-bottom: 4px;">Trade Thesis</div>
                            <div style="color: var(--text); font-size: 11px; line-height: 1.4;">
                                ${recommendation.thesis}
                            </div>
                        </div>
                        ` : ''}
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                            ${recommendation.risk_reward !== undefined ? `
                            <div style="background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">R/R Ratio</span>
                                <div style="color: ${recommendation.risk_reward >= 2 ? '#22c55e' : '#f59e0b'}; font-size: 11px; font-weight: 500;">
                                    1:${recommendation.risk_reward.toFixed(1)}
                                </div>
                            </div>
                            ` : ''}
                            ${recommendation.p_win !== undefined ? `
                            <div style="background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">Win %</span>
                                <div style="color: var(--text); font-size: 11px; font-weight: 500;">
                                    ${(recommendation.p_win * 100).toFixed(0)}%
                                </div>
                            </div>
                            ` : ''}
                            ${recommendation.expected_value_pct !== undefined ? `
                            <div style="background: rgba(0,0,0,0.2); padding: 6px 8px; border-radius: 4px;">
                                <span style="color: var(--text-dimmer); font-size: 10px;">EV</span>
                                <div style="color: ${recommendation.expected_value_pct > 0 ? '#22c55e' : '#ef4444'}; font-size: 11px; font-weight: 500;">
                                    ${recommendation.expected_value_pct > 0 ? '+' : ''}${recommendation.expected_value_pct.toFixed(1)}%
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                    ` : ''}
                    
                    <!-- Step 4: Final Decision -->
                    <div style="
                        background: linear-gradient(135deg, ${actionColor}22 0%, ${actionColor}11 100%);
                        border: 1px solid ${actionColor}66;
                        border-radius: 8px;
                        padding: 14px;
                    ">
                        <div style="font-size: 12px; font-weight: 600; color: ${actionColor}; margin-bottom: 8px;">
                            4️⃣ Decision Output
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-size: 18px; font-weight: bold; color: ${actionColor};">
                                    ${decision.action} ${decision.symbol}
                                </div>
                                <div style="color: var(--text-dim); font-size: 11px; margin-top: 2px;">
                                    Conviction: ${decision.confidence}%
                                </div>
                            </div>
                            ${raw_decision.aggressiveness_level !== undefined ? `
                            <div style="text-align: right;">
                                <div style="color: var(--text-dimmer); font-size: 10px; margin-bottom: 4px;">Aggression</div>
                                <div style="display: flex; gap: 3px;">
                                    ${[0, 1, 2, 3].map(i => `
                                        <div style="
                                            width: 6px;
                                            height: 6px;
                                            border-radius: 50%;
                                            background: ${i <= raw_decision.aggressiveness_level ? actionColor : 'var(--border)'};
                                        "></div>
                                    `).join('')}
                                </div>
                            </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                `;
            }
            
            content.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: var(--text-bright); font-size: 18px;">
                        ${decision.symbol} - AI Analysis
                    </h3>
                    <button onclick="this.closest('div[style*=fixed]').remove()" style="
                        background: transparent;
                        border: none;
                        color: var(--text-dimmer);
                        font-size: 24px;
                        cursor: pointer;
                        padding: 0;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">×</button>
                </div>
                
                ${decisionTreeHTML}
                
                <div style="background: ${actionColor}10; border: 1px solid ${actionColor}30; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 24px; font-weight: bold; color: ${actionColor};">
                            ${decision.action}
                        </span>
                        <span style="font-size: 20px; font-weight: bold; color: ${confidenceColor};">
                            ${decision.confidence}% Confidence
                        </span>
                    </div>
                    <div style="font-size: 14px; color: var(--text); line-height: 1.6;">
                        <strong>Reasoning:</strong> ${decision.reason}
                    </div>
                    ${decision.event_type ? `
                        <div style="font-size: 12px; color: var(--text-dimmer); margin-top: 8px;">
                            <strong>Trigger Event:</strong> ${decision.event_type}
                        </div>
                    ` : ''}
                </div>
                
                ${thesis.setup || thesis.catalyst ? `
                    <div style="margin-bottom: 20px;">
                        <h4 style="color: var(--text-bright); font-size: 14px; margin-bottom: 12px;">Trading Thesis</h4>
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; font-size: 13px; color: var(--text);">
                            ${thesis.setup ? `<div style="margin-bottom: 8px;"><strong>Setup:</strong> ${thesis.setup}</div>` : ''}
                            ${thesis.catalyst ? `<div style="margin-bottom: 8px;"><strong>Catalyst:</strong> ${thesis.catalyst}</div>` : ''}
                            ${thesis.risk_reward ? `<div style="margin-bottom: 8px;"><strong>Risk/Reward:</strong> ${thesis.risk_reward}</div>` : ''}
                            ${thesis.prob_win ? `<div><strong>Win Probability:</strong> ${thesis.prob_win}%</div>` : ''}
                        </div>
                    </div>
                ` : ''}
                
                ${decision.entry_price || targets.length > 0 || stops.technical ? `
                    <div style="margin-bottom: 20px;">
                        <h4 style="color: var(--text-bright); font-size: 14px; margin-bottom: 12px;">Price Levels</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            ${decision.entry_price ? `
                                <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                                    <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Entry Price</div>
                                    <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">$${decision.entry_price}</div>
                                </div>
                            ` : ''}
                            ${stops.technical ? `
                                <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                                    <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Stop Loss</div>
                                    <div style="font-size: 16px; font-weight: 600; color: #ef4444;">$${stops.technical}</div>
                                </div>
                            ` : ''}
                            ${targets.length > 0 ? `
                                <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; grid-column: span 2;">
                                    <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Targets</div>
                                    <div style="font-size: 14px; color: #22c55e;">${targets.map(t => '$' + t).join(', ')}</div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                ` : ''}
                
                ${watchlist.length > 0 ? `
                    <div style="margin-bottom: 20px;">
                        <h4 style="color: var(--text-bright); font-size: 14px; margin-bottom: 12px;">Watchlist Items</h4>
                        <div style="display: grid; gap: 8px;">
                            ${watchlist.map(item => `
                                <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; font-size: 13px;">
                                    <div style="color: var(--text-bright); margin-bottom: 4px;">
                                        <strong>${item.symbol}</strong>
                                        ${item.trigger_above ? ` - Above $${item.trigger_above}` : ''}
                                        ${item.trigger_below ? ` - Below $${item.trigger_below}` : ''}
                                    </div>
                                    ${item.notes ? `<div style="color: var(--text-dimmer); font-size: 12px;">${item.notes}</div>` : ''}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
                
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Analysis Time</div>
                        <div style="font-size: 14px; font-weight: 600; color: var(--text-bright);">${decision.time}</div>
                    </div>
                    <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Model</div>
                        <div style="font-size: 14px; font-weight: 600; color: var(--text-bright);">Claude 3</div>
                    </div>
                    <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px; text-align: center;">
                        <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Latency</div>
                        <div style="font-size: 14px; font-weight: 600; color: var(--text-bright);">${decision.latency || 'N/A'}</div>
                    </div>
                </div>
                
                <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);">
                    <button onclick="selectSymbol('${decision.symbol}'); this.closest('div[style*=fixed]').remove()" style="
                        width: 100%;
                        padding: 12px;
                        background: ${actionColor};
                        color: white;
                        border: none;
                        border-radius: 6px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: opacity 0.15s;
                    " onmouseover="this.style.opacity='0.9'" onmouseout="this.style.opacity='1'">
                        View ${decision.symbol} Chart
                    </button>
                </div>
            `;
            
            modal.appendChild(content);
            document.body.appendChild(modal);
            
            // Close on background click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                }
            });
        }
        
        // Make showOptionsDetail globally accessible for clicks
        window.showOptionsDetail = function(flow) {
            // Create modal content
            const isCall = flow.option_type === 'CALL';
            const color = flow.confidence >= 80 ? '#22c55e' : 
                         flow.confidence >= 60 ? '#4a9eff' :
                         flow.confidence >= 40 ? '#f59e0b' : '#ef4444';
            
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10000;
                animation: fadeIn 0.2s;
            `;
            
            const content = document.createElement('div');
            content.style.cssText = `
                background: var(--surface);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 24px;
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow-y: auto;
                animation: slideUp 0.2s;
            `;
            
            content.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <h3 style="margin: 0; color: var(--text-bright); font-size: 18px;">
                        ${flow.symbol} Options Flow Detail
                    </h3>
                    <button onclick="this.closest('div[style*=fixed]').remove()" style="
                        background: transparent;
                        border: none;
                        color: var(--text-dimmer);
                        font-size: 24px;
                        cursor: pointer;
                        padding: 0;
                        width: 30px;
                        height: 30px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    ">×</button>
                </div>
                
                <div style="background: ${color}10; border: 1px solid ${color}30; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <span style="font-size: 24px; font-weight: bold; color: ${color};">
                            ${isCall ? '📈 CALL' : '📉 PUT'}
                        </span>
                        <span style="font-size: 20px; font-weight: bold; color: ${color};">
                            ${flow.confidence}% Confidence
                        </span>
                    </div>
                    <div style="font-size: 14px; color: var(--text); line-height: 1.6;">
                        ${flow.interpretation || 'Unusual options activity detected'}
                    </div>
                </div>
                
                <div style="display: grid; gap: 12px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Strike Price</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">$${flow.strike}</div>
                        </div>
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Expiry</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">${flow.expiry}</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Volume</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">${flow.volume?.toLocaleString() || 'N/A'}</div>
                        </div>
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Open Interest</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">${flow.open_interest?.toLocaleString() || 'N/A'}</div>
                        </div>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Volume/OI Ratio</div>
                            <div style="font-size: 16px; font-weight: 600; color: ${flow.volume_oi_ratio > 2 ? '#22c55e' : 'var(--text-bright)'};">
                                ${flow.volume_oi_ratio?.toFixed(2)}x
                            </div>
                        </div>
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Premium</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">
                                $${(flow.premium / 1000).toFixed(1)}K
                            </div>
                        </div>
                    </div>
                    
                    <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                        <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Signal Type</div>
                        <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">
                            ${flow.signal_type?.toUpperCase() || 'UNUSUAL ACTIVITY'}
                        </div>
                    </div>
                    
                    ${flow.delta ? `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Delta</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">${flow.delta?.toFixed(3) || 'N/A'}</div>
                        </div>
                        <div style="background: var(--bg-darker); padding: 12px; border-radius: 6px;">
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-bottom: 4px;">Implied Volatility</div>
                            <div style="font-size: 16px; font-weight: 600; color: var(--text-bright);">
                                ${flow.implied_vol ? (flow.implied_vol * 100).toFixed(1) + '%' : 'N/A'}
                            </div>
                        </div>
                    </div>
                    ` : ''}
                    
                    <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);">
                        <button onclick="selectSymbol('${flow.symbol}'); this.closest('div[style*=fixed]').remove()" style="
                            width: 100%;
                            padding: 12px;
                            background: ${color};
                            color: white;
                            border: none;
                            border-radius: 6px;
                            font-weight: 600;
                            cursor: pointer;
                            transition: opacity 0.15s;
                        " onmouseover="this.style.opacity='0.9'" onmouseout="this.style.opacity='1'">
                            View ${flow.symbol} Chart
                        </button>
                    </div>
                </div>
            `;
            
            modal.appendChild(content);
            document.body.appendChild(modal);
            
            // Close on background click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                }
            });
        }
        
        // Make selectSymbol globally accessible for modals
        window.selectSymbol = function(symbol) {
            currentChartSymbol = symbol;
            
            // Update visual selection
            document.querySelectorAll('.symbol-link').forEach(link => {
                if (link.dataset.symbol === symbol) {
                    link.style.background = 'var(--primary)';
                    link.style.color = 'var(--bg-darkest)';
                    link.style.borderColor = 'var(--primary)';
                } else {
                    link.style.background = 'var(--bg-darkest)';
                    link.style.color = 'var(--text-dimmer)';
                    link.style.borderColor = 'var(--border)';
                }
            });
            
            // Update asset type indicator in header
            const assetType = symbolInfo[currentChartSymbol] || 'stocks';
            const indicator = document.getElementById('assetTypeIndicator');
            if (indicator) {
                if (assetType === 'gold') {
                    indicator.textContent = '🥇';
                    indicator.title = 'Gold';
                } else if (assetType === 'crypto') {
                    indicator.textContent = '₿';
                    indicator.title = 'Cryptocurrency';
                } else {
                    indicator.textContent = '';
                }
            }
            
            // Fetch price data and update chart
            if (!priceHistory[currentChartSymbol]) {
                fetch(`/api/prices/${currentChartSymbol}`)
                    .then(res => res.json())
                    .then(prices => {
                        priceHistory[currentChartSymbol] = prices;
                        updateChart();
                    })
                    .catch(err => {
                        console.log('Failed to fetch prices for', currentChartSymbol, err);
                    });
            } else {
                updateChart();
            }
            
            // Update market hours label for crypto (24/7)
            const hoursLabel = document.getElementById('marketHoursLabel');
            if (hoursLabel) {
                if (assetType === 'crypto') {
                    hoursLabel.textContent = '24/7 Trading';
                } else {
                    hoursLabel.textContent = 'Today (9:30 AM - 4:00 PM ET)';
                }
            }
            
            // Check if we have data for this symbol, if not fetch it
            if (!priceHistory[currentChartSymbol] || priceHistory[currentChartSymbol].length === 0) {
                fetch(`/api/prices/${currentChartSymbol}`)
                    .then(r => r.json())
                    .then(prices => {
                        priceHistory[currentChartSymbol] = prices;
                        updateChart();
                    })
                    .catch(err => {
                        console.error('Failed to fetch price data:', err);
                    });
            } else {
                updateChart();
            }
        }
        
        function updateChart() {
            if (!priceChart) return;
            
            const fullDaySize = 390;
            const dataPoints = new Array(fullDaySize).fill(null);
            
            const data = priceHistory[currentChartSymbol] || [];
            let nonNullCount = 0;
            let firstDataIndex = -1;
            let lastDataIndex = -1;
            
            // Place data at correct time positions and count non-null points
            for (const point of data) {
                if (point.index >= 0 && point.index < fullDaySize) {
                    dataPoints[point.index] = point.price;
                    nonNullCount++;
                    if (firstDataIndex === -1) firstDataIndex = point.index;
                    lastDataIndex = point.index;
                }
            }
            
            // If we have data, fill in the gaps to show a continuous line for the full day
            if (nonNullCount > 0 && firstDataIndex >= 0) {
                // Get first and last prices
                const firstPrice = dataPoints[firstDataIndex];
                const lastPrice = dataPoints[lastDataIndex];
                
                // Fill in before market open with first price (flat line)
                for (let i = 0; i < firstDataIndex; i++) {
                    dataPoints[i] = firstPrice;
                }
                
                // Fill in after last data point with last price (flat line)
                for (let i = lastDataIndex + 1; i < fullDaySize; i++) {
                    dataPoints[i] = lastPrice;
                }
                
                // Interpolate gaps in between if there are any
                let prevIndex = firstDataIndex;
                for (let i = firstDataIndex + 1; i <= lastDataIndex; i++) {
                    if (dataPoints[i] === null) {
                        // Find next non-null value
                        let nextIndex = i + 1;
                        while (nextIndex <= lastDataIndex && dataPoints[nextIndex] === null) {
                            nextIndex++;
                        }
                        
                        if (nextIndex <= lastDataIndex) {
                            // Linear interpolation
                            const prevPrice = dataPoints[prevIndex];
                            const nextPrice = dataPoints[nextIndex];
                            const steps = nextIndex - prevIndex;
                            
                            for (let j = prevIndex + 1; j < nextIndex; j++) {
                                const ratio = (j - prevIndex) / steps;
                                dataPoints[j] = prevPrice + (nextPrice - prevPrice) * ratio;
                            }
                            
                            prevIndex = nextIndex;
                            i = nextIndex - 1; // Skip to next known point
                        }
                    } else {
                        prevIndex = i;
                    }
                }
            }
            
            // Always show as a line chart for the full trading day
            priceChart.data.datasets[0].showLine = true;
            priceChart.data.datasets[0].spanGaps = false; // We've filled all gaps
            priceChart.data.datasets[0].pointRadius = 0; // Hide points for cleaner look
            priceChart.data.datasets[0].pointHoverRadius = 4;
            priceChart.data.datasets[0].borderWidth = 2;
            priceChart.data.datasets[0].tension = 0; // No curve for accurate representation
            
            priceChart.data.datasets[0].data = dataPoints;
            priceChart.update('none');
        }
        
        function addPricePoint(symbol, price) {
            if (!priceHistory[symbol]) {
                priceHistory[symbol] = [];
            }
            
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: false 
            });
            
            // Check if market is open
            const currentHour = now.getHours();
            const currentMinute = now.getMinutes();
            const currentTime = currentHour * 60 + currentMinute;
            const marketOpen = 9 * 60 + 30;  // 9:30 AM
            const marketClose = 16 * 60;     // 4:00 PM
            const dayOfWeek = now.getDay();
            const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
            const isMarketOpen = (currentTime >= marketOpen && currentTime <= marketClose) && isWeekday;
            
            if (isMarketOpen) {
                // Calculate position in trading day
                const minutesSinceOpen = currentTime - marketOpen;
                
                // Check if this is a new trading day
                if (minutesSinceOpen === 0 || 
                    (priceHistory[symbol].length > 0 && 
                     priceHistory[symbol][0].index !== undefined && 
                     minutesSinceOpen < priceHistory[symbol][priceHistory[symbol].length - 1].index)) {
                    priceHistory[symbol] = [];  // Clear for new trading day
                }
                
                // Only add new point if it's been at least 60 seconds since last point
                const lastPoint = priceHistory[symbol][priceHistory[symbol].length - 1];
                if (!lastPoint || timeStr !== lastPoint.time) {
                    const newPoint = {
                        time: timeStr,
                        price: price,
                        index: Math.max(0, Math.min(389, minutesSinceOpen))
                    };
                    priceHistory[symbol].push(newPoint);
                    // No need to save every price point - just keep in memory
                } else {
                    // Update the last point's price if within same minute
                    lastPoint.price = price;
                }
                
                // Keep full day of data (390 minutes = 6.5 hours of trading)
                if (priceHistory[symbol].length > 390) {
                    priceHistory[symbol].shift();
                }
                
                // Update chart if showing this symbol
                if (symbol === currentChartSymbol) {
                    updateChart();
                }
            }
            // After hours - keep showing last trading day's data without adding new points
        }
        
        // Initialize chart on load
        window.addEventListener('DOMContentLoaded', () => {
            initChart();
            initConvictionGauge();
            initPnLChart();
            // Load settings and populate symbols
            loadChartSymbols();
            // Load initial price data
            loadInitialPrices();
            // Load historical P&L data
            loadHistoricalPnL();
        });
        
        function loadChartSymbols() {
            // Get user symbols from settings
            fetch('/api/settings')
                .then(r => r.json())
                .then(settings => {
                    watchlistSymbols = settings.symbols || [
                        "AAPL", "NVDA", "TSLA", "IXHL", "NUAI", "BZAI", "ELTP", 
                        "OPEN", "CEG", "VRT", "PLTR", "UPST", 
                        "TEM", "HTFL", "SDGR", "APLD", "SOFI", "CORZ", "WULF",
                        "GLD", "BTC-USD", "ETH-USD"
                    ];
                    
                    symbolInfo = settings.symbol_info || {};
                    
                    // Populate symbol list with clickable links
                    const symbolList = document.getElementById('symbolList');
                    if (symbolList) {
                        symbolList.innerHTML = '';
                        watchlistSymbols.forEach((symbol, index) => {
                            const link = document.createElement('a');
                            link.href = '#';
                            link.dataset.symbol = symbol;
                            link.className = 'symbol-link';
                            link.style.cssText = `
                                padding: 6px 12px;
                                background: var(--bg-darkest);
                                border: 1px solid var(--border);
                                border-radius: 6px;
                                color: var(--text-dimmer);
                                text-decoration: none;
                                font-size: 12px;
                                font-weight: 500;
                                transition: all 0.2s ease;
                                display: inline-flex;
                                align-items: center;
                                gap: 4px;
                                white-space: nowrap;
                            `;
                            
                            const assetType = symbolInfo[symbol] || 'stocks';
                            // Show emoji in list
                            const typeEmoji = assetType === 'gold' ? '🥇 ' : assetType === 'crypto' ? '₿ ' : '';
                            link.innerHTML = typeEmoji + symbol;
                            
                            link.onclick = (e) => {
                                e.preventDefault();
                                selectSymbol(symbol);
                            };
                            
                            symbolList.appendChild(link);
                        });
                    }
                    
                    // Set initial symbol
                    currentChartSymbol = watchlistSymbols[0];
                    selectSymbol(currentChartSymbol);
                });
        }
        
        function loadInitialPrices() {
            // Load price data for default symbol
            setTimeout(() => {
                fetch(`/api/prices/${currentChartSymbol}`)
                    .then(r => r.json())
                    .then(prices => {
                        priceHistory[currentChartSymbol] = prices;
                        updateChart();
                    });
            }, 500);  // Small delay to ensure symbols are loaded
        }
        
        function loadHistoricalPnL() {
            // Load historical P&L data from database
            fetch('/api/pnl-history')
                .then(r => r.json())
                .then(data => {
                    if (data && data.length > 0) {
                        // Clear and reload with historical data
                        pnlHistory = data;
                        
                        // Display in chart with fixed axes
                        if (pnlChart) {
                            const fullDaySize = 390;
                            const labels = new Array(fullDaySize).fill('');
                            const dataPoints = new Array(fullDaySize).fill(null);
                            
                            // Generate full day labels
                            const startHour = 9;
                            const startMinute = 30;
                            for (let i = 0; i < fullDaySize; i++) {
                                const totalMinutes = startHour * 60 + startMinute + i;
                                const hour = Math.floor(totalMinutes / 60);
                                const minute = totalMinutes % 60;
                                if (i % 30 === 0) {
                                    labels[i] = `${hour.toString().padStart(2, '0')}:${minute.toString().padStart(2, '0')}`;
                                }
                            }
                            
                            // Place historical data at correct positions
                            for (const point of data) {
                                if (point.index >= 0 && point.index < fullDaySize) {
                                    dataPoints[point.index] = point.value;
                                }
                            }
                            
                            // Update chart
                            pnlChart.data.labels = labels;
                            pnlChart.data.datasets[0].data = dataPoints;
                            
                            // Set color based on last value
                            const lastValue = data[data.length - 1]?.value || 0;
                            const isPositive = lastValue >= 0;
                            pnlChart.data.datasets[0].borderColor = isPositive ? '#22c55e' : '#ef4444';
                            pnlChart.data.datasets[0].backgroundColor = isPositive ? 
                                'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
                            
                            // Set fixed y-axis scale
                            const values = data.map(p => p.value).filter(v => v !== null);
                            const maxVal = Math.max(...values, 100);
                            const minVal = Math.min(...values, -100);
                            const range = Math.max(Math.abs(maxVal), Math.abs(minVal), 100);
                            
                            pnlChart.options.scales.y.min = -range * 1.2;
                            pnlChart.options.scales.y.max = range * 1.2;
                            
                            pnlChart.update('none');
                        }
                    }
                })
                .catch(err => console.log('No historical P&L data:', err));
        }
        
        // Update dashboard every 2 seconds
        setInterval(updateDashboard, 2000);
        
        // Fetch price data every 10 seconds during market hours (more frequent updates)
        setInterval(() => {
            const now = new Date();
            const hour = now.getHours();
            const minute = now.getMinutes();
            const day = now.getDay();
            const currentTime = hour + minute / 60;
            
            const isWeekday = day >= 1 && day <= 5;
            const isMarketHours = isWeekday && currentTime >= 6.5 && currentTime <= 13;
            
            if (isMarketHours && currentChartSymbol) {
                fetch(`/api/prices/${currentChartSymbol}`)
                    .then(r => r.json())
                    .then(prices => {
                        priceHistory[currentChartSymbol] = prices;
                        updateChart();
                    })
                    .catch(err => console.log('Failed to fetch prices:', err));
            }
        }, 10000);  // Every 10 seconds
        
        function loadSettings() {
            fetch('/api/settings')
                .then(r => r.json())
                .then(data => {
                    if (data.symbols) {
                        document.getElementById('symbols').value = data.symbols.join(',');
                    }
                    if (data.risk_level) {
                        document.getElementById('riskLevel').value = data.risk_level;
                    }
                    if (data.max_daily_loss) {
                        document.getElementById('maxDailyLoss').value = data.max_daily_loss;
                    }
                });
        }
        
        function saveSettings() {
            const settings = {
                symbols: document.getElementById('symbols').value.split(',').map(s => s.trim()),
                risk_level: document.getElementById('riskLevel').value,
                max_daily_loss: parseInt(document.getElementById('maxDailyLoss').value)
            };
            
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'saved') {
                    // Brief visual feedback
                    const btn = event.target;
                    btn.textContent = 'Saved!';
                    setTimeout(() => btn.textContent = 'Save Settings', 1500);
                }
            });
        }
        
        let lastPriceUpdate = null;
        
        function checkMarketStatus() {
            const now = new Date();
            const hour = now.getHours();
            const minute = now.getMinutes();
            const day = now.getDay();
            
            // Market hours: 9:30 AM - 4:00 PM ET
            // Convert to ET for market hours check
            const etOffset = now.getTimezoneOffset() === 240 ? 0 : 3; // EDT is UTC-4, EST is UTC-5
            const etHour = hour + etOffset;
            const etTime = etHour + minute / 60;
            
            const marketOpen = 9.5;   // 9:30 AM ET
            const marketClose = 16;    // 4:00 PM ET
            
            const isWeekday = day >= 1 && day <= 5;
            const isMarketHours = isWeekday && etTime >= marketOpen && etTime < marketClose;
            
            const marketStatusEl = document.getElementById('marketStatus');
            const liveIndicatorEl = document.getElementById('liveIndicator');
            
            if (isMarketHours) {
                marketStatusEl.className = 'status-badge market-open';
                marketStatusEl.textContent = 'Market Open';
                
                // Show live indicator if we've received price updates recently
                if (lastPriceUpdate && (now - lastPriceUpdate) < 120000) {  // Within 2 minutes
                    liveIndicatorEl.style.display = 'inline-block';
                } else {
                    liveIndicatorEl.style.display = 'none';
                }
            } else {
                marketStatusEl.className = 'status-badge market-closed';
                marketStatusEl.textContent = 'Market Closed';
                liveIndicatorEl.style.display = 'none';
            }
        }
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.className = 'status-badge ' + data.status;
                    statusEl.textContent = data.status;
                    
                    // Track last price update
                    if (data.last_price_update) {
                        lastPriceUpdate = new Date(data.last_price_update);
                    }
                    
                    // Check market status
                    checkMarketStatus();
                    
                    // Update buttons
                    document.getElementById('startBtn').disabled = data.status === 'running';
                    document.getElementById('stopBtn').disabled = data.status === 'stopped';
                    
                    // Update P&L
                    updatePnL(data.pnl);
                    
                    // Update positions
                    updatePositions(data.positions);
                    
                    // Update AI decisions and conviction
                    updateAIDecisions(data.ai_decisions);
                    updateConvictionGauge(data.ai_decisions);
                    
                    // Update news feed
                    updateNewsFeed(data.news_feed);
                    
                    // Update trading signals
                    updateTradingSignals(data.trading_signals);
                    
                    // Update options flow
                    updateOptionsFlow(data.options_flow);
                    
                    // Update company events
                    updateCompanyEvents(data.company_events);
                    
                    // Update log
                    updateLog(data.log);
                    
                    // Update price data for charts
                    if (data.price_data) {
                        for (const [symbol, price] of Object.entries(data.price_data)) {
                            addPricePoint(symbol, price);
                        }
                    }
                    
                    // Update P&L history
                    if (data.pnl) {
                        updatePnLHistory(data.pnl);
                    }
                });
        }
        
        function updatePnL(pnl) {
            const dailyEl = document.getElementById('dailyPnl');
            const totalEl = document.getElementById('totalPnl');
            
            dailyEl.textContent = formatCurrency(pnl.daily);
            totalEl.textContent = formatCurrency(pnl.total);
            
            dailyEl.className = 'card-value ' + (pnl.daily >= 0 ? 'green' : 'red');
            totalEl.className = 'card-value ' + (pnl.total >= 0 ? 'green' : 'red');
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions');
            const count = document.getElementById('posCount');
            
            if (!positions || Object.keys(positions).length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">No positions</div></div>';
                count.textContent = '0';
                return;
            }
            
            count.textContent = Object.keys(positions).length;
            
            let html = '';
            for (const [symbol, pos] of Object.entries(positions)) {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'green' : 'red';
                html += `<div class="list-item">
                    <div class="list-item-title">${symbol}</div>
                    <div class="list-item-meta">
                        <span>${pos.quantity} shares</span>
                        <span>Avg: ${formatCurrency(pos.avg_price)}</span>
                        <span class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</span>
                    </div>
                </div>`;
            }
            container.innerHTML = html;
        }
        
        function updateAIDecisions(decisions) {
            const container = document.getElementById('aiDecisions');
            if (!decisions || decisions.length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">Waiting for market events...</div></div>';
                return;
            }
            
            // Store decisions globally for click handlers
            window.aiDecisionsData = decisions;
            
            let html = '';
            for (let i = 0; i < decisions.length; i++) {
                const decision = decisions[i];
                const badge = decision.confidence >= 75 ? 'badge-blue' : 
                             decision.confidence >= 50 ? 'badge-green' : '';
                html += `<div class="list-item" onclick="showAIDecisionDetail(window.aiDecisionsData[${i}])" style="cursor: pointer; transition: all 0.15s;" onmouseover="this.style.background='var(--surface-hover)'" onmouseout="this.style.background='transparent'">
                    <div class="list-item-title">${decision.symbol} - ${decision.action}</div>
                    <div class="list-item-meta">
                        <span class="badge ${badge}">${decision.confidence}%</span>
                        <span>${decision.reason}</span>
                        <span>${decision.time}</span>
                    </div>
                </div>`;
            }
            container.innerHTML = html;
        }
        
        function updateConvictionGauge(decisions) {
            if (!decisions || decisions.length === 0 || !convictionGauge) return;
            
            // Get latest AI decision
            const latest = decisions[decisions.length - 1];
            if (latest && latest.confidence !== undefined) {
                currentConviction = latest.confidence;
                currentDirection = latest.action || 'neutral';
                
                // Update gauge
                convictionGauge.data.datasets[0].data = [currentConviction, 100 - currentConviction];
                
                // Update color based on conviction level
                let color = '#4a9eff';  // Default blue
                if (currentConviction >= 80) {
                    color = '#22c55e';  // Green for high conviction
                } else if (currentConviction >= 60) {
                    color = '#4a9eff';  // Blue for medium
                } else if (currentConviction >= 40) {
                    color = '#f59e0b';  // Orange for low
                } else {
                    color = '#ef4444';  // Red for very low
                }
                
                convictionGauge.data.datasets[0].backgroundColor[0] = color;
                convictionGauge.update('none');
                
                // Update text
                document.getElementById('convictionValue').textContent = currentConviction + '%';
                
                // Update label
                let label = 'Neutral';
                if (currentDirection === 'buy' || currentDirection === 'BUY') label = 'Bullish';
                else if (currentDirection === 'sell' || currentDirection === 'SELL') label = 'Bearish';
                else if (currentDirection === 'hold' || currentDirection === 'HOLD') label = 'Hold';
                
                document.getElementById('convictionLabel').textContent = label;
            }
        }
        
        function updateOptionsFlow(flows) {
            const container = document.getElementById('optionsFlowContainer');
            const summary = document.getElementById('flowSummary');
            
            if (!flows || flows.length === 0) {
                container.innerHTML = '<div class="empty-state">No unusual options activity detected</div>';
                summary.textContent = 'Scanning for unusual activity...';
                return;
            }
            
            // Calculate summary stats
            let bullishCount = 0;
            let bearishCount = 0;
            let totalPremium = 0;
            
            flows.forEach(flow => {
                if (flow.option_type === 'CALL') bullishCount++;
                else if (flow.option_type === 'PUT') bearishCount++;
                totalPremium += flow.premium || 0;
            });
            
            // Update summary
            summary.textContent = `${flows.length} signals | Bullish: ${bullishCount} | Bearish: ${bearishCount}`;
            
            // Store flows data globally for click handlers
            window.optionsFlowData = flows;
            
            // Create flow visualization
            let html = '<div style="display: grid; gap: 8px;">';
            
            for (let i = 0; i < Math.min(flows.length, 10); i++) {  // Show top 10
                const flow = flows[i];
                const isCall = flow.option_type === 'CALL';
                const color = flow.confidence >= 80 ? '#22c55e' : 
                             flow.confidence >= 60 ? '#4a9eff' :
                             flow.confidence >= 40 ? '#f59e0b' : '#ef4444';
                
                const directionIcon = isCall ? '📈' : '📉';
                const signalIcon = flow.signal_type === 'sweep' ? '🔥' : 
                                  flow.signal_type === 'block' ? '🏢' :
                                  flow.signal_type === 'high_premium' ? '💰' : '⚡';
                
                html += `
                    <div onclick="showOptionsDetail(window.optionsFlowData[${i}])" style="
                        background: var(--surface);
                        border: 1px solid var(--border);
                        border-left: 3px solid ${color};
                        border-radius: 6px;
                        padding: 12px;
                        display: grid;
                        grid-template-columns: auto 1fr auto;
                        gap: 12px;
                        align-items: center;
                        transition: all 0.15s;
                        cursor: pointer;
                    " onmouseover="this.style.background='var(--surface-hover)'" onmouseout="this.style.background='var(--surface)'">
                        <div style="font-size: 20px;">${directionIcon}</div>
                        <div>
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                                <span style="font-weight: 600; color: var(--text-bright);">${flow.symbol}</span>
                                <span style="font-size: 11px; padding: 2px 6px; background: ${color}20; color: ${color}; border-radius: 4px;">
                                    ${flow.confidence}% ${signalIcon}
                                </span>
                                <span style="font-size: 11px; color: var(--text-dimmer);">${flow.signal_type}</span>
                            </div>
                            <div style="font-size: 12px; color: var(--text);">
                                ${flow.option_type} $${flow.strike} exp ${flow.expiry} 
                                | Vol: ${flow.volume} | V/OI: ${flow.volume_oi_ratio?.toFixed(1)}x
                            </div>
                            <div style="font-size: 11px; color: var(--text-dimmer); margin-top: 4px;">
                                ${flow.interpretation}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 14px; font-weight: 600; color: ${isCall ? '#22c55e' : '#ef4444'};">
                                $${(flow.premium / 1000).toFixed(0)}K
                            </div>
                            <div style="font-size: 10px; color: var(--text-dimmer);">
                                ${flow.time || 'now'}
                            </div>
                        </div>
                    </div>
                `;
            }
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        function updateCompanyEvents(events) {
            const container = document.getElementById('companyEvents');
            
            // Store all events globally
            if (events && events.length > 0) {
                allCompanyEvents = events;
            }
            
            // Filter events based on active filter
            let filteredEvents = allCompanyEvents;
            if (activeEventFilter !== 'all') {
                filteredEvents = allCompanyEvents.filter(event => {
                    const eventType = event.type || '';
                    // Special handling for different filter types
                    if (activeEventFilter === 'SEC') {
                        // Show all SEC filings
                        return eventType.includes('8-K') || eventType.includes('10-') || 
                               eventType.includes('Form 4') || eventType.includes('Insider');
                    } else if (activeEventFilter === '10-') {
                        return eventType.includes('10-Q') || eventType.includes('10-K');
                    } else if (activeEventFilter === 'Form 4') {
                        return eventType.includes('Form 4') || eventType.includes('Insider');
                    } else if (activeEventFilter === '8-K') {
                        return eventType.includes('8-K');
                    }
                    return eventType.includes(activeEventFilter);
                });
            }
            
            if (!filteredEvents || filteredEvents.length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">No events matching filter...</div></div>';
                return;
            }
            
            let html = '';
            for (const event of filteredEvents) {
                // Determine icon and color based on event type
                let icon = '📄';
                let color = '#4a9eff';
                
                if (event.type && event.type.includes('8-K')) {
                    icon = '📢';
                    color = '#f59e0b';
                } else if (event.type && event.type.includes('10-')) {
                    icon = '📊';
                    color = '#4a9eff';
                } else if (event.type && event.type.includes('Form 4')) {
                    icon = '👤';
                    color = '#8b42ff';
                } else if (event.type && event.type.includes('Earnings')) {
                    icon = '💰';
                    color = '#22c55e';
                } else if (event.type && event.type.includes('FDA')) {
                    icon = '💊';
                    color = '#ef4444';
                }
                
                const impactColor = event.impact >= 80 ? '#ef4444' :
                                   event.impact >= 70 ? '#f59e0b' :
                                   event.impact >= 60 ? '#4a9eff' : '#6b6b6b';
                
                html += `
                    <div class="list-item" style="border-left: 2px solid ${color}; padding-left: 10px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 16px;">${icon}</span>
                            <div style="flex: 1;">
                                <div class="list-item-title" style="display: flex; align-items: center; gap: 8px;">
                                    <span style="font-weight: 600; color: var(--text-bright);">${event.symbol}</span>
                                    <span style="font-size: 11px; color: ${color};">${event.type}</span>
                                    <span style="font-size: 10px; padding: 2px 6px; background: ${impactColor}20; color: ${impactColor}; border-radius: 3px;">
                                        ${event.impact}%
                                    </span>
                                </div>
                                <div class="list-item-subtitle" style="margin-top: 4px;">
                                    ${event.headline || event.description || 'Company event detected'}
                                </div>
                                ${event.url ? `<a href="${event.url}" target="_blank" style="font-size: 11px; color: var(--accent); text-decoration: none;">View Filing →</a>` : ''}
                            </div>
                            <div style="font-size: 11px; color: var(--text-dimmer);">
                                ${event.time || 'Now'}
                            </div>
                        </div>
                    </div>
                `;
            }
            
            container.innerHTML = html;
        }
        
        function filterCompanyEvents(filter) {
            // Update active filter
            activeEventFilter = filter;
            
            // Update button states
            document.querySelectorAll('.filter-btn').forEach(btn => {
                btn.classList.remove('active');
                if (btn.getAttribute('data-filter') === filter || 
                    (filter === '10-' && btn.textContent === '10-Q/K')) {
                    btn.classList.add('active');
                }
            });
            
            // Re-render events with filter
            updateCompanyEvents();
        }
        
        function updatePnLHistory(pnl) {
            const now = new Date();
            const timeStr = now.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit',
                hour12: false 
            });
            
            // Calculate minutes since market open (9:30 AM)
            const currentHour = now.getHours();
            const currentMinute = now.getMinutes();
            const currentTime = currentHour * 60 + currentMinute;
            const marketOpen = 9 * 60 + 30;  // 9:30 AM
            const marketClose = 16 * 60;     // 4:00 PM
            
            // Check if market is open
            const isMarketHours = currentTime >= marketOpen && currentTime <= marketClose;
            const dayOfWeek = now.getDay();
            const isWeekday = dayOfWeek >= 1 && dayOfWeek <= 5;
            const isMarketOpen = isMarketHours && isWeekday;
            
            if (isMarketOpen) {
                // During market hours, calculate position in trading day
                const minutesSinceOpen = currentTime - marketOpen;
                
                // Check if this is a new trading day (clear history at market open)
                if (minutesSinceOpen === 0 || (pnlHistory.length > 0 && minutesSinceOpen < pnlHistory[pnlHistory.length - 1].index)) {
                    pnlHistory = [];  // Clear for new trading day
                }
                
                // Add to history with index position
                pnlHistory.push({
                    time: timeStr,
                    value: pnl.total || 0,
                    index: Math.max(0, Math.min(389, minutesSinceOpen))  // Clamp to 0-389
                });
                
                // Keep only current trading day data
                if (pnlHistory.length > 390) {
                    pnlHistory.shift();
                }
            }
            // After hours - keep showing last trading day's data without adding new points
            
            // Update chart with full trading day window (390 points like price chart)
            if (pnlChart) {
                const fullDaySize = 390;  // Full trading day (9:30 AM - 4:00 PM)
                
                // Create fixed-size array for data
                const dataPoints = new Array(fullDaySize).fill(null);
                
                // Place actual P&L data at correct time positions
                for (let i = 0; i < pnlHistory.length; i++) {
                    const dataPoint = pnlHistory[i];
                    if (dataPoint.index !== undefined && dataPoint.index >= 0 && dataPoint.index < fullDaySize) {
                        dataPoints[dataPoint.index] = dataPoint.value;
                    }
                }
                
                // Update only the data, labels are already set in initPnLChart
                pnlChart.data.datasets[0].data = dataPoints;
                
                // Update color based on current P&L
                const isPositive = (pnl.total || 0) >= 0;
                pnlChart.data.datasets[0].borderColor = isPositive ? '#22c55e' : '#ef4444';
                pnlChart.data.datasets[0].backgroundColor = isPositive ? 
                    'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
                
                // Set fixed y-axis scale to prevent zooming
                // Start with a reasonable range and expand if needed
                const currentMax = Math.max(...pnlHistory.map(p => p.value), 100);
                const currentMin = Math.min(...pnlHistory.map(p => p.value), -100);
                const range = Math.max(Math.abs(currentMax), Math.abs(currentMin), 100);
                
                // Set symmetric scale with some padding
                pnlChart.options.scales.y.min = -range * 1.2;
                pnlChart.options.scales.y.max = range * 1.2;
                
                pnlChart.update('none');
            }
        }
        
        function updateNewsFeed(news) {
            const container = document.getElementById('newsFeed');
            const ticker = document.getElementById('tickerContent');
            
            if (!news || news.length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">No recent news</div></div>';
                ticker.textContent = 'Loading market news...';
                return;
            }
            
            // Update news feed with clickable links
            let html = '';
            for (const item of news) {
                const sentClass = item.sentiment > 0.2 ? 'badge-green' : 
                                 item.sentiment < -0.2 ? 'badge-red' : '';
                const titleHtml = item.url ? 
                    `<a href="${item.url}" target="_blank" style="color: inherit; text-decoration: none; cursor: pointer;" 
                       onmouseover="this.style.color='var(--accent)'" 
                       onmouseout="this.style.color='inherit'">${item.title}</a>` : 
                    item.title;
                    
                html += `<div class="list-item">
                    <div class="list-item-title">${titleHtml}</div>
                    <div class="list-item-meta">
                        <span>${item.source}</span>
                        ${sentClass ? `<span class="badge ${sentClass}">
                            ${item.sentiment > 0 ? 'Bullish' : 'Bearish'}
                        </span>` : ''}
                        <span>${item.time || 'Now'}</span>
                    </div>
                </div>`;
            }
            container.innerHTML = html;
            
            // Update ticker with clickable links
            let tickerHtml = '';
            for (const item of news) {
                const link = item.url ? 
                    `<a href="${item.url}" target="_blank" style="color: inherit; text-decoration: none;" 
                       onmouseover="this.style.textDecoration='underline'" 
                       onmouseout="this.style.textDecoration='none'">${item.title}</a>` : 
                    item.title;
                tickerHtml += link + ' • ';
            }
            ticker.innerHTML = tickerHtml + tickerHtml; // Duplicate for scrolling effect
        }
        
        function updateTradingSignals(signals) {
            const container = document.getElementById('tradingSignals');
            if (!signals || signals.length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">No signals yet</div></div>';
                return;
            }
            
            let html = '';
            for (const signal of signals) {
                const badge = signal.action === 'BUY' ? 'badge-green' : 'badge-red';
                html += `<div class="list-item">
                    <div class="list-item-title">${signal.action} ${signal.symbol}</div>
                    <div class="list-item-meta">
                        <span class="badge ${badge}">${signal.conviction}%</span>
                        <span>${signal.shares} @ ${formatCurrency(signal.price)}</span>
                    </div>
                </div>`;
            }
            container.innerHTML = html;
        }
        
        function updateLog(log) {
            const container = document.getElementById('log');
            if (!log || log.length === 0) return;
            
            let html = '';
            for (const entry of log.slice(-10)) {
                html += `<div class="list-item">
                    <div class="list-item-title">${entry}</div>
                </div>`;
            }
            container.innerHTML = html;
        }
        
        function formatCurrency(value) {
            const num = parseFloat(value) || 0;
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD',
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            }).format(num);
        }
        
        function startTrading() {
            const symbols = document.getElementById('symbols').value;
            fetch('/api/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbols: symbols})
            });
        }
        
        function stopTrading() {
            fetch('/api/stop', {method: 'POST'});
        }
        
        function testAI() {
            const event = prompt("Enter a market event to analyze:", 
                "Federal Reserve raises interest rates by 0.25%");
            if (event) {
                fetch('/api/test-ai', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({event: event})
                }).then(r => r.json()).then(result => {
                    alert(`AI Analysis:\n\nSignal: ${result.signal}\nConfidence: ${result.confidence}%\nReason: ${result.reason}`);
                });
            }
        }
        
        // Initial update
        updateDashboard();
    </script>
</body>
</html>
'''

# Copy all the API routes from original app.py
@app.route('/')
@requires_auth
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring - no auth required."""
    return jsonify({
        'status': 'healthy',
        'trading_active': trading_status == 'running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/status')
@requires_auth
def get_status():
    global trading_status, pnl, positions, ai_decisions, trading_log, news_feed, trading_signals, options_flow, company_events
    
    # Get real-time database metrics
    try:
        today_metrics = db.get_today_pnl()
        pnl['daily'] = today_metrics.get('total_pnl', pnl['daily'])
        
        # Get 30-day metrics for total P&L
        monthly_metrics = db.get_performance_metrics(days=30)
        pnl['total'] = monthly_metrics.get('total_pnl', pnl['total'])
    except:
        pass  # Use cached values if database fails
    
    # Extract current prices from positions
    price_data = {}
    if positions:
        for symbol, pos in positions.items():
            if 'current_price' in pos:
                price_data[symbol] = pos['current_price']
    
    return jsonify({
        'status': trading_status,
        'pnl': pnl,
        'positions': positions,
        'ai_decisions': ai_decisions[-10:],
        'news_feed': news_feed[-10:],
        'trading_signals': trading_signals[-5:],
        'options_flow': options_flow[-5:],
        'last_price_update': last_price_update_time.isoformat() if last_price_update_time else None,
        'company_events': company_events[-10:],
        'log': trading_log[-20:],
        'price_data': price_data
    })

@app.route('/api/start', methods=['POST'])
@requires_auth
def start_trading():
    global trading_process, trading_status, trading_log
    
    if trading_status == 'running':
        return jsonify({'error': 'Already running'}), 400
    
    symbols = request.json.get('symbols', '')
    
    # Start the AI trading system
    cmd = ['python', 'start_ai_trading.py']
    
    trading_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid
    )
    
    trading_status = 'running'
    trading_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Started trading")
    
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
@requires_auth
def stop_trading():
    global trading_process, trading_status, trading_log
    
    if trading_process and trading_status == 'running':
        os.killpg(os.getpgid(trading_process.pid), signal.SIGTERM)
        trading_process = None
        trading_status = 'stopped'
        trading_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Stopped trading")
    
    return jsonify({'status': 'stopped'})

@app.route('/api/news', methods=['POST'])
def update_news():
    """Receive news updates from AI runner."""
    global news_feed
    
    data = request.json
    if data and 'news' in data:
        news_feed = data['news']
        return jsonify({'status': 'updated'})
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/api/options', methods=['POST'])
def update_options():
    """Receive options flow updates from AI runner."""
    global options_flow
    
    data = request.json
    if data and 'options' in data:
        options_flow = data['options']
        return jsonify({'status': 'updated'})
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/api/ai_decision', methods=['POST'])
def receive_ai_decision():
    """Receive AI trading decision."""
    global ai_decisions
    data = request.json
    if data and 'decision' in data:
        decision = data['decision']
        ai_decisions.append(decision)
        # Keep only last 20 decisions
        if len(ai_decisions) > 20:
            ai_decisions.pop(0)
        return jsonify({'status': 'ok'})
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/api/company_event', methods=['POST'])
def add_company_event():
    """Receive company event from AI runner."""
    global company_events
    
    data = request.json
    if data and 'event' in data:
        event = data['event']
        # Keep only last 20 events
        company_events.insert(0, event)
        company_events = company_events[:20]
        logger.info(f"Received company event: {event.get('type')} for {event.get('symbol')}")
        return jsonify({'status': 'added'})
    return jsonify({'error': 'Invalid data'}), 400

@app.route('/api/company_events', methods=['GET'])
def get_company_events():
    """Get list of company events."""
    return jsonify(company_events)

@app.route('/api/prices/<symbol>')
def get_price_history(symbol):
    """Get historical price data for a symbol from database."""
    from robo_trader.database import TradingDatabase
    import datetime
    
    db = TradingDatabase()
    try:
        # Check if market is open
        now = datetime.datetime.now()
        current_time = now.hour * 60 + now.minute
        market_open = 9 * 60 + 30
        market_close = 16 * 60
        is_market_hours = current_time >= market_open and current_time <= market_close
        is_weekday = now.weekday() < 5
        
        if is_market_hours and is_weekday:
            # Get current day's prices
            prices = db.get_current_day_prices(symbol)
        else:
            # Get last trading day's prices
            prices = db.get_last_trading_day_prices(symbol)
        
        # Format for frontend
        formatted = []
        for p in prices:
            # Convert timestamp to time string
            ts = p['timestamp']
            if isinstance(ts, str):
                ts = datetime.datetime.fromisoformat(ts)
            time_str = ts.strftime('%H:%M')
            formatted.append({
                'time': time_str,
                'price': p['price'],
                'index': p.get('minute_index', 0)
            })
        
        return jsonify(formatted)
    finally:
        db.close()
    
    # TODO: Implement historical data storage
    # base_prices = {
    #     'AAPL': 175.0,
    #     'NVDA': 480.0,
    #     'TSLA': 250.0,
    #     'IXHL': 35.0,
    #     'NUAI': 3.50,
    #     'BZAI': 2.80,
    #     'ELTP': 4.20,
    #     'OPEN': 2.10,
    #     'CEG': 215.0,
    #     'VRT': 45.0,
    #     'PLTR': 22.0,
    #     'UPST': 48.0,
    #     'TEM': 280.0,
    #     'HTFL': 68.0,
    #     'SDGR': 65.0,
    #     'APLD': 12.0,
    #     'SOFI': 7.50,
    #     'CORZ': 8.20,
    #     'WULF': 3.90,
    #     'SPY': 450.0,
    #     'QQQ': 385.0
    # }
    # 
    # base_price = base_prices.get(symbol, 100.0)
    # # prices = []
    # current_price = base_price
    # 
    # # Generate 30 data points
    # now = datetime.datetime.now()
    # for i in range(30):
    #     time = now - datetime.timedelta(minutes=30-i)
    #     # Random walk
    #     change = random.uniform(-0.5, 0.5) * 0.01 * base_price
    #     current_price += change
    #     prices.append({
    #         'time': time.strftime('%H:%M'),
    #         'price': round(current_price, 2)
    #     })
    # 
    # return jsonify(prices)

@app.route('/api/pnl-history')
def get_pnl_history():
    """Get P&L history from database."""
    from robo_trader.database import TradingDatabase
    import datetime
    
    db = TradingDatabase()
    try:
        history = db.get_last_pnl_history()
        
        # Format for frontend
        formatted = []
        for h in history:
            ts = h['timestamp']
            if isinstance(ts, str):
                ts = datetime.datetime.fromisoformat(ts)
            
            # Calculate minute index
            market_open = ts.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = int((ts - market_open).total_seconds() / 60)
            minute_index = max(0, min(389, minutes_since_open))
            
            formatted.append({
                'time': ts.strftime('%H:%M'),
                'value': h['total_pnl'],
                'index': minute_index
            })
        
        return jsonify(formatted)
    finally:
        db.close()

@app.route('/api/save-price', methods=['POST'])
@requires_auth
def save_price():
    """Save price data to database and update live chart."""
    from robo_trader.database import TradingDatabase
    
    data = request.json
    db = TradingDatabase()
    try:
        # Save to database
        db.save_price_point(
            symbol=data['symbol'],
            price=data['price'],
            minute_index=data.get('minute_index')
        )
        
        # Track price update for live indicator
        global last_price_update_time
        last_price_update_time = datetime.now()
        
        return jsonify({'status': 'saved', 'timestamp': datetime.now().isoformat()})
    finally:
        db.close()

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get user settings with asset type information."""
    global user_settings
    settings = user_settings.get('default', {})
    
    # Add asset type information
    asset_types = {
        "stocks": ["AAPL", "NVDA", "TSLA", "IXHL", "NUAI", "BZAI", "ELTP", 
                   "OPEN", "CEG", "VRT", "PLTR", "UPST", 
                   "TEM", "HTFL", "SDGR", "APLD", "SOFI", "CORZ", "WULF"],
        "gold": ["GLD"],
        "crypto": ["BTC-USD", "ETH-USD"]
    }
    
    # Build symbol info mapping
    symbol_info = {}
    for asset_type, syms in asset_types.items():
        for sym in syms:
            symbol_info[sym] = asset_type
    
    settings['symbol_info'] = symbol_info
    return jsonify(settings)

@app.route('/api/settings', methods=['POST'])
@requires_auth
def save_settings():
    """Save user settings."""
    global user_settings
    
    data = request.json
    if data:
        user_settings['default'].update(data)
        save_user_settings(user_settings)
        return jsonify({'status': 'saved'})
    return jsonify({'error': 'Invalid data'}), 400


@app.route('/api/test-ai', methods=['POST'])
@requires_auth
def test_ai():
    event = request.json.get('event', '')
    
    # Simple mock for now
    import random
    signals = ['BUY', 'SELL', 'HOLD']
    signal = random.choice(signals)
    confidence = random.randint(40, 90)
    
    reasons = {
        'BUY': 'Bullish sentiment detected, positive market reaction expected',
        'SELL': 'Bearish implications, risk-off sentiment likely',
        'HOLD': 'Mixed signals, waiting for clearer direction'
    }
    
    global ai_decisions
    ai_decisions.append({
        'symbol': 'SPY',
        'action': signal,
        'confidence': confidence,
        'reason': reasons[signal],
        'time': datetime.now().strftime('%H:%M:%S')
    })
    
    return jsonify({
        'signal': signal,
        'confidence': confidence,
        'reason': reasons[signal]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=False)
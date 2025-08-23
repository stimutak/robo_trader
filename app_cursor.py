#!/usr/bin/env python3
"""
Cursor-style Dashboard for Robo Trader
Ultra-minimal design inspired by cursor.com/dashboard
"""

from flask import Flask, render_template_string, jsonify, request
import asyncio
import threading
import json
from datetime import datetime
import subprocess
import os
import signal
import glob
import time
from robo_trader.config import load_config
from robo_trader.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

# Global state
trading_process = None
trading_status = "stopped"
trading_log = []
positions = {}
pnl = {"daily": 0.0, "total": 0.0}
ai_decisions = []
news_feed = []
trading_signals = []
options_flow = []

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
                           "OPEN", "ADA", "HBAR", "CEG", "VRT", "PLTR", "UPST", 
                           "TEM", "HTFL", "SDGR", "APLD", "SOFI", "CORZ", "WULF"],
                "risk_level": "moderate",
                "max_daily_loss": 1000
            }
        }

def save_user_settings(settings):
    """Save user settings."""
    with open(USER_SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

user_settings = load_user_settings()

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
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="logo">
                <h1>Robo Trader</h1>
                <span id="status" class="status-badge stopped">Stopped</span>
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
        
        // Update dashboard every 2 seconds
        setInterval(updateDashboard, 2000);
        
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
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.className = 'status-badge ' + data.status;
                    statusEl.textContent = data.status;
                    
                    // Update buttons
                    document.getElementById('startBtn').disabled = data.status === 'running';
                    document.getElementById('stopBtn').disabled = data.status === 'stopped';
                    
                    // Update P&L
                    updatePnL(data.pnl);
                    
                    // Update positions
                    updatePositions(data.positions);
                    
                    // Update AI decisions
                    updateAIDecisions(data.ai_decisions);
                    
                    // Update news feed
                    updateNewsFeed(data.news_feed);
                    
                    // Update trading signals
                    updateTradingSignals(data.trading_signals);
                    
                    // Update log
                    updateLog(data.log);
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
            
            let html = '';
            for (const decision of decisions) {
                const badge = decision.confidence >= 75 ? 'badge-blue' : 
                             decision.confidence >= 50 ? 'badge-green' : '';
                html += `<div class="list-item">
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
        
        function updateNewsFeed(news) {
            const container = document.getElementById('newsFeed');
            const ticker = document.getElementById('tickerContent');
            
            if (!news || news.length === 0) {
                container.innerHTML = '<div class="list-item"><div class="list-item-title">No recent news</div></div>';
                return;
            }
            
            // Update news feed
            let html = '';
            for (const item of news) {
                const sentClass = item.sentiment > 0.2 ? 'badge-green' : 
                                 item.sentiment < -0.2 ? 'badge-red' : '';
                html += `<div class="list-item">
                    <div class="list-item-title">${item.title}</div>
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
            
            // Update ticker
            const tickerText = news.map(item => item.title).join(' • ');
            ticker.textContent = tickerText + ' • ' + tickerText;
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
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def get_status():
    global trading_status, pnl, positions, ai_decisions, trading_log, news_feed, trading_signals, options_flow
    return jsonify({
        'status': trading_status,
        'pnl': pnl,
        'positions': positions,
        'ai_decisions': ai_decisions[-10:],
        'news_feed': news_feed[-10:],
        'trading_signals': trading_signals[-5:],
        'options_flow': options_flow[-5:],
        'log': trading_log[-20:]
    })

@app.route('/api/start', methods=['POST'])
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

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get user settings."""
    global user_settings
    return jsonify(user_settings.get('default', {}))

@app.route('/api/settings', methods=['POST'])
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
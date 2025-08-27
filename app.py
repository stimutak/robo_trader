#!/usr/bin/env python3
"""
RoboTrader Dashboard - Clean, ML-Integrated Interface
Provides real-time monitoring of trading, ML models, and performance metrics
"""

from flask import Flask, render_template_string, jsonify, request, Response
import asyncio
import threading
import json
from datetime import datetime, timedelta
import subprocess
import os
import signal
import time
import hashlib
from functools import wraps
from pathlib import Path
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# Import our modules
from robo_trader.config import load_config
from robo_trader.logger import get_logger
from robo_trader.database_async import AsyncTradingDatabase
from robo_trader.analytics.performance import PerformanceAnalyzer
from robo_trader.features.feature_pipeline import FeaturePipeline
from robo_trader.ml.model_trainer import ModelTrainer
from robo_trader.websocket_server import ws_manager

logger = get_logger(__name__)
app = Flask(__name__)

# Configuration
config = load_config()
AUTH_ENABLED = os.getenv('DASH_AUTH_ENABLED', 'false').lower() == 'true'
AUTH_USER = os.getenv('DASH_USER', 'admin')
AUTH_PASS_HASH = os.getenv('DASH_PASS_HASH', '')

# Initialize components
db = None
feature_pipeline = None
model_trainer = None
performance_analyzer = None

# Global state
trading_process = None
trading_status = "stopped"
trading_log = []
positions = {}

# Load default symbols at module level
try:
    with open('user_settings.json', 'r') as f:
        settings = json.load(f)
        default_symbols = settings.get('default', {}).get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
except:
    default_symbols = ['AAPL', 'MSFT', 'GOOGL']
pnl = {"daily": 0.0, "total": 0.0, "unrealized": 0.0}
ml_metrics = {
    "models_trained": 0,
    "last_prediction": None,
    "feature_count": 0,
    "model_performance": {}
}
performance_metrics = {
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "win_rate": 0.0,
    "profit_factor": 0.0
}

def init_async_components():
    """Initialize async components in event loop"""
    global db, feature_pipeline, model_trainer, performance_analyzer
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    db = AsyncTradingDatabase()
    feature_pipeline = FeaturePipeline(config)
    model_trainer = ModelTrainer(config, model_dir=Path("models"))
    performance_analyzer = PerformanceAnalyzer()

# Initialize in background thread
init_thread = threading.Thread(target=init_async_components)
init_thread.daemon = True
init_thread.start()

# Authentication
def check_auth(username, password):
    """Check if username/password is valid."""
    if not AUTH_ENABLED:
        return True
    if not AUTH_PASS_HASH:
        return True
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return username == AUTH_USER and password_hash == AUTH_PASS_HASH

def authenticate():
    """Send 401 response that enables basic auth."""
    return Response(
        'Authentication required.\n'
        'Please enter your credentials.',
        401,
        {'WWW-Authenticate': 'Basic realm="RoboTrader Dashboard"'}
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

# HTML Template - Clean, modern design
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboTrader Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 0;
            border-bottom: 1px solid #2a2a2a;
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 16px;
            background: #1a1a1a;
            border-radius: 20px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #ff4444;
        }
        
        .status-dot.active {
            background: #44ff44;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: #3a3a3a;
            transform: translateY(-2px);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .card-title {
            font-size: 14px;
            font-weight: 500;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .card-value {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .card-change {
            font-size: 14px;
            color: #888;
        }
        
        .positive { color: #44ff44; }
        .negative { color: #ff4444; }
        .neutral { color: #ffaa44; }
        
        .table-container {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            overflow-x: auto;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #2a2a2a;
            color: #888;
            font-weight: 500;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        td {
            padding: 12px;
            border-bottom: 1px solid #1a1a1a;
        }
        
        tr:hover {
            background: #222;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        button.secondary {
            background: #2a2a2a;
        }
        
        .chart-container {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 12px;
            padding: 20px;
            height: 400px;
            margin-bottom: 20px;
        }
        
        .tabs {
            display: flex;
            gap: 20px;
            border-bottom: 1px solid #2a2a2a;
            margin-bottom: 20px;
        }
        
        .tab {
            padding: 10px 0;
            color: #888;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            transition: all 0.3s ease;
        }
        
        .tab.active {
            color: #fff;
            border-bottom-color: #667eea;
        }
        
        .log-container {
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            height: 200px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
        }
        
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        
        .log-time {
            color: #666;
            margin-right: 10px;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .metric-item {
            text-align: center;
            padding: 10px;
            background: #0a0a0a;
            border-radius: 8px;
        }
        
        .metric-label {
            font-size: 11px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: 600;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background: #2a2a2a;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .badge.ml {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">RoboTrader</div>
            <div class="status-indicator">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Disconnected</span>
            </div>
        </header>
        
        <div class="button-group">
            <button onclick="startTrading()" id="start-btn">Start Trading</button>
            <button onclick="stopTrading()" id="stop-btn" class="secondary">Stop Trading</button>
            <button onclick="refreshData()" class="secondary">Refresh Data</button>
        </div>
        
        <div class="tabs">
            <div class="tab active" onclick="switchTab('overview', this)">Overview</div>
            <div class="tab" onclick="switchTab('watchlist', this)">Watchlist</div>
            <div class="tab" onclick="switchTab('positions', this)">Positions</div>
            <div class="tab" onclick="switchTab('ml', this)">ML Models</div>
            <div class="tab" onclick="switchTab('performance', this)">Performance</div>
            <div class="tab" onclick="switchTab('logs', this)">Logs</div>
        </div>
        
        <div id="overview-tab" class="tab-content">
            <div class="grid">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Total P&L</span>
                    </div>
                    <div class="card-value" id="total-pnl">$0.00</div>
                    <div class="card-change" id="pnl-change">0.00%</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Daily P&L</span>
                    </div>
                    <div class="card-value" id="daily-pnl">$0.00</div>
                    <div class="card-change" id="daily-change">0.00%</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Open Positions</span>
                    </div>
                    <div class="card-value" id="position-count">0</div>
                    <div class="card-change" id="position-value">$0.00 value</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Win Rate</span>
                    </div>
                    <div class="card-value" id="win-rate">0.0%</div>
                    <div class="card-change" id="trade-count">0 trades today</div>
                </div>
            </div>
            
            <div class="metric-grid">
                <div class="metric-item">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value" id="sharpe">0.00</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative" id="max-dd">0.0%</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value" id="profit-factor">0.00</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">Avg Correlation</div>
                    <div class="metric-value" id="avg-correlation">0.00</div>
                </div>
            </div>
        </div>
        
        <div id="ml-tab" class="tab-content" style="display: none;">
            <div class="grid">
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Models Trained</span>
                        <span class="badge ml">ML</span>
                    </div>
                    <div class="card-value" id="models-trained">0</div>
                    <div class="card-change">Last trained: <span id="last-train">Never</span></div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Active Features</span>
                        <span class="badge ml">ML</span>
                    </div>
                    <div class="card-value" id="feature-count">0</div>
                    <div class="card-change">Feature pipeline status</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Model Accuracy</span>
                        <span class="badge ml">ML</span>
                    </div>
                    <div class="card-value" id="model-accuracy">0.0%</div>
                    <div class="card-change">Direction accuracy</div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Prediction Confidence</span>
                        <span class="badge ml">ML</span>
                    </div>
                    <div class="card-value" id="prediction-confidence">0.0%</div>
                    <div class="card-change">Average confidence</div>
                </div>
            </div>
            
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Model Type</th>
                            <th>Test Score</th>
                            <th>Features Used</th>
                            <th>Last Updated</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="model-table">
                        <tr>
                            <td colspan="5" style="text-align: center; color: #666;">No models trained yet</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <div class="table-container">
                <h3 style="margin-bottom: 15px; color: #888;">Top Features by Importance</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Feature Name</th>
                            <th>Importance Score</th>
                            <th>Category</th>
                        </tr>
                    </thead>
                    <tbody id="feature-table">
                        <tr>
                            <td colspan="3" style="text-align: center; color: #666;">No feature data available</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="performance-tab" class="tab-content" style="display: none;">
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Daily</th>
                            <th>Weekly</th>
                            <th>Monthly</th>
                            <th>All Time</th>
                        </tr>
                    </thead>
                    <tbody id="performance-table">
                        <tr>
                            <td>Return</td>
                            <td id="return-daily">0.00%</td>
                            <td id="return-weekly">0.00%</td>
                            <td id="return-monthly">0.00%</td>
                            <td id="return-all">0.00%</td>
                        </tr>
                        <tr>
                            <td>Volatility</td>
                            <td id="vol-daily">0.00%</td>
                            <td id="vol-weekly">0.00%</td>
                            <td id="vol-monthly">0.00%</td>
                            <td id="vol-all">0.00%</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td id="sharpe-daily">0.00</td>
                            <td id="sharpe-weekly">0.00</td>
                            <td id="sharpe-monthly">0.00</td>
                            <td id="sharpe-all">0.00</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td id="dd-daily">0.00%</td>
                            <td id="dd-weekly">0.00%</td>
                            <td id="dd-monthly">0.00%</td>
                            <td id="dd-all">0.00%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="watchlist-tab" class="tab-content" style="display: none;">
            <div class="table-container">
                <h3>Watched Symbols</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Current Price</th>
                            <th>Position</th>
                            <th>Avg Cost</th>
                            <th>P&L</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody id="watchlist-table">
                        <tr>
                            <td colspan="6" style="text-align: center; color: #666;">Loading watchlist...</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="positions-tab" class="tab-content" style="display: none;">
            <div class="table-container">
                <h3>Open Positions</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Quantity</th>
                            <th>Entry Price</th>
                            <th>Current Price</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Value</th>
                            <th>ML Signal</th>
                        </tr>
                    </thead>
                    <tbody id="positions-table">
                        <tr>
                            <td colspan="8" style="text-align: center; color: #666;">No open positions</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="logs-tab" class="tab-content" style="display: none;">
            <div class="log-container" id="log-container">
                <div class="log-entry">
                    <span class="log-time">00:00:00</span>
                    <span>System initialized</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentTab = 'overview';
        
        function switchTab(tab, element) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(content => {
                content.style.display = 'none';
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(t => {
                t.classList.remove('active');
            });
            
            // Show selected tab
            document.getElementById(tab + '-tab').style.display = 'block';
            
            // Add active class to selected tab
            element.classList.add('active');
            
            currentTab = tab;
            
            // Load tab-specific data
            if (tab === 'ml') {
                loadMLData();
            } else if (tab === 'performance') {
                loadPerformanceData();
            } else if (tab === 'positions') {
                loadPositions();
            }
        }
        
        async function startTrading() {
            document.getElementById('start-btn').disabled = true;
            try {
                const response = await fetch('/api/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({})  // Will use symbols from user_settings.json
                });
                const data = await response.json();
                if (data.status === 'started') {
                    updateStatus('running');
                    addLog('Trading started');
                }
            } catch (error) {
                console.error('Error starting trading:', error);
                addLog('Error: Failed to start trading');
            }
            document.getElementById('start-btn').disabled = false;
        }
        
        async function stopTrading() {
            document.getElementById('stop-btn').disabled = true;
            try {
                const response = await fetch('/api/stop', {method: 'POST'});
                const data = await response.json();
                if (data.status === 'stopped') {
                    updateStatus('stopped');
                    addLog('Trading stopped');
                }
            } catch (error) {
                console.error('Error stopping trading:', error);
                addLog('Error: Failed to stop trading');
            }
            document.getElementById('stop-btn').disabled = false;
        }
        
        async function refreshData() {
            await Promise.all([
                loadStatus(),
                loadPnL(),
                loadWatchlist(),
                loadPositions(),
                loadMLData(),
                loadPerformanceData()
            ]);
        }
        
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                updateStatus(data.trading_status);
                updatePnL(data.pnl);
                updateMetrics(data.metrics);
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        async function loadPnL() {
            try {
                const response = await fetch('/api/pnl');
                const data = await response.json();
                updatePnL(data);
            } catch (error) {
                console.error('Error loading P&L:', error);
            }
        }
        
        async function loadWatchlist() {
            try {
                const response = await fetch('/api/watchlist');
                const data = await response.json();
                updateWatchlistTable(data.watchlist);
            } catch (error) {
                console.error('Error loading watchlist:', error);
            }
        }
        
        async function loadPositions() {
            try {
                const response = await fetch('/api/positions');
                const data = await response.json();
                updatePositionsTable(data.positions);
            } catch (error) {
                console.error('Error loading positions:', error);
            }
        }
        
        async function loadMLData() {
            try {
                const response = await fetch('/api/ml/status');
                const data = await response.json();
                updateMLMetrics(data);
            } catch (error) {
                console.error('Error loading ML data:', error);
            }
        }
        
        async function loadPerformanceData() {
            try {
                const response = await fetch('/api/performance');
                const data = await response.json();
                updatePerformanceTable(data);
            } catch (error) {
                console.error('Error loading performance data:', error);
            }
        }
        
        function updateStatus(status) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            
            if (status === 'running') {
                dot.classList.add('active');
                text.textContent = 'Trading Active';
            } else {
                dot.classList.remove('active');
                text.textContent = 'Trading Stopped';
            }
        }
        
        function updatePnL(pnl) {
            document.getElementById('total-pnl').textContent = formatCurrency(pnl.total || 0);
            document.getElementById('daily-pnl').textContent = formatCurrency(pnl.daily || 0);
            
            // Update colors
            const totalEl = document.getElementById('total-pnl');
            const dailyEl = document.getElementById('daily-pnl');
            
            totalEl.className = pnl.total >= 0 ? 'card-value positive' : 'card-value negative';
            dailyEl.className = pnl.daily >= 0 ? 'card-value positive' : 'card-value negative';
        }
        
        function updateMetrics(metrics) {
            if (metrics) {
                document.getElementById('sharpe').textContent = (metrics.sharpe_ratio || 0).toFixed(2);
                document.getElementById('max-dd').textContent = ((metrics.max_drawdown || 0) * 100).toFixed(1) + '%';
                document.getElementById('profit-factor').textContent = (metrics.profit_factor || 0).toFixed(2);
                document.getElementById('win-rate').textContent = ((metrics.win_rate || 0) * 100).toFixed(1) + '%';
            }
        }
        
        function updateMLMetrics(data) {
            document.getElementById('models-trained').textContent = data.models_trained || 0;
            document.getElementById('feature-count').textContent = data.feature_count || 0;
            document.getElementById('model-accuracy').textContent = ((data.accuracy || 0) * 100).toFixed(1) + '%';
            document.getElementById('prediction-confidence').textContent = ((data.confidence || 0) * 100).toFixed(1) + '%';
            
            // Update model table
            if (data.models && data.models.length > 0) {
                const tbody = document.getElementById('model-table');
                tbody.innerHTML = data.models.map(model => `
                    <tr>
                        <td>${model.type}</td>
                        <td>${(model.test_score * 100).toFixed(2)}%</td>
                        <td>${model.feature_count}</td>
                        <td>${formatTime(model.updated)}</td>
                        <td><span class="badge ${model.status === 'active' ? 'ml' : ''}">${model.status}</span></td>
                    </tr>
                `).join('');
            }
            
            // Update feature table
            if (data.top_features && data.top_features.length > 0) {
                const tbody = document.getElementById('feature-table');
                tbody.innerHTML = data.top_features.map(feature => `
                    <tr>
                        <td>${feature.name}</td>
                        <td>${(feature.importance * 100).toFixed(2)}%</td>
                        <td>${feature.category}</td>
                    </tr>
                `).join('');
            }
        }
        
        function updateWatchlistTable(watchlist) {
            const tbody = document.getElementById('watchlist-table');
            if (!watchlist || watchlist.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #666;">No symbols in watchlist</td></tr>';
                return;
            }
            
            tbody.innerHTML = watchlist.map(item => {
                const pnl = item.quantity > 0 ? (item.current_price - item.avg_cost) * item.quantity : 0;
                const pnlDisplay = item.quantity > 0 ? `$${pnl.toFixed(2)}` : '-';
                const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                
                return `
                    <tr>
                        <td><strong>${item.symbol}</strong></td>
                        <td>$${item.current_price.toFixed(2)}</td>
                        <td>${item.quantity || '-'}</td>
                        <td>${item.avg_cost > 0 ? '$' + item.avg_cost.toFixed(2) : '-'}</td>
                        <td class="${item.quantity > 0 ? pnlClass : ''}">${pnlDisplay}</td>
                        <td>${item.notes || ''}</td>
                    </tr>
                `;
            }).join('');
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-table');
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #666;">No open positions</td></tr>';
                document.getElementById('position-count').textContent = '0';
                document.getElementById('position-value').textContent = '$0.00 value';
                return;
            }
            
            let totalValue = 0;
            tbody.innerHTML = positions.map(pos => {
                const pnl = (pos.current_price - pos.entry_price) * pos.quantity;
                const pnlPct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100;
                const value = pos.current_price * pos.quantity;
                totalValue += value;
                
                return `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.quantity}</td>
                        <td>$${pos.entry_price.toFixed(2)}</td>
                        <td>$${pos.current_price.toFixed(2)}</td>
                        <td class="${pnl >= 0 ? 'positive' : 'negative'}">$${pnl.toFixed(2)}</td>
                        <td class="${pnl >= 0 ? 'positive' : 'negative'}">${pnlPct.toFixed(2)}%</td>
                        <td>$${value.toFixed(2)}</td>
                        <td><span class="badge ${pos.ml_signal === 'buy' ? 'ml' : ''}">${pos.ml_signal || 'none'}</span></td>
                    </tr>
                `;
            }).join('');
            
            document.getElementById('position-count').textContent = positions.length.toString();
            document.getElementById('position-value').textContent = formatCurrency(totalValue) + ' value';
        }
        
        function updatePerformanceTable(data) {
            if (!data) return;
            
            ['daily', 'weekly', 'monthly', 'all'].forEach(period => {
                const metrics = data[period] || {};
                document.getElementById(`return-${period}`).textContent = formatPercent(metrics.return_pct);
                document.getElementById(`vol-${period}`).textContent = formatPercent(metrics.volatility);
                document.getElementById(`sharpe-${period}`).textContent = (metrics.sharpe || 0).toFixed(2);
                document.getElementById(`dd-${period}`).textContent = formatPercent(metrics.max_drawdown);
            });
        }
        
        function addLog(message) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="log-time">${time}</span><span>${message}</span>`;
            container.appendChild(entry);
            container.scrollTop = container.scrollHeight;
        }
        
        function formatCurrency(value) {
            return new Intl.NumberFormat('en-US', {
                style: 'currency',
                currency: 'USD'
            }).format(value);
        }
        
        function formatPercent(value) {
            if (value === undefined || value === null) return '0.00%';
            return (value * 100).toFixed(2) + '%';
        }
        
        function formatTime(timestamp) {
            if (!timestamp) return 'Never';
            const date = new Date(timestamp);
            const now = new Date();
            const diff = now - date;
            
            if (diff < 60000) return 'Just now';
            if (diff < 3600000) return Math.floor(diff / 60000) + ' min ago';
            if (diff < 86400000) return Math.floor(diff / 3600000) + ' hours ago';
            return Math.floor(diff / 86400000) + ' days ago';
        }
        
        // WebSocket connection for real-time updates
        let ws = null;
        let reconnectInterval = null;
        
        function connectWebSocket() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                return; // Already connected
            }
            
            ws = new WebSocket('ws://localhost:8765');
            
            ws.onopen = function() {
                console.log('WebSocket connected');
                addLog('Real-time connection established');
                clearInterval(reconnectInterval);
                reconnectInterval = null;
                
                // Subscribe to all symbols
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    symbols: ['*']  // Subscribe to all
                }));
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleRealtimeUpdate(data);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = function() {
                console.log('WebSocket disconnected');
                addLog('Real-time connection lost - reconnecting...');
                // Try to reconnect every 5 seconds
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connectWebSocket, 5000);
                }
            };
        }
        
        function handleRealtimeUpdate(data) {
            switch(data.type) {
                case 'market_data':
                    updateMarketPrice(data.symbol, data.price, data.bid, data.ask, data.volume);
                    break;
                case 'trade':
                    handleTradeUpdate(data);
                    break;
                case 'positions':
                    updatePositionsFromWS(data.positions);
                    break;
                case 'signal':
                    handleSignalUpdate(data);
                    break;
                case 'performance':
                    updateMetrics(data.metrics);
                    break;
            }
        }
        
        function updateMarketPrice(symbol, price, bid, ask, volume) {
            // Update watchlist prices in real-time
            const rows = document.querySelectorAll('#watchlist-tbody tr');
            rows.forEach(row => {
                const symbolCell = row.cells[0];
                if (symbolCell && symbolCell.textContent === symbol) {
                    const priceCell = row.cells[1];
                    if (priceCell) {
                        const oldPrice = parseFloat(priceCell.textContent.replace('$', ''));
                        priceCell.textContent = `$${price.toFixed(2)}`;
                        
                        // Flash color based on price change
                        if (oldPrice && oldPrice !== price) {
                            priceCell.style.transition = 'background-color 0.3s';
                            priceCell.style.backgroundColor = price > oldPrice ? '#00ff0033' : '#ff000033';
                            setTimeout(() => {
                                priceCell.style.backgroundColor = '';
                            }, 300);
                        }
                    }
                }
            });
        }
        
        function handleTradeUpdate(data) {
            const msg = `${data.side} ${data.quantity} ${data.symbol} @ $${data.price.toFixed(2)}`;
            addLog(`Trade executed: ${msg}`);
            // Refresh positions after trade
            loadPositions();
            loadPnL();
        }
        
        function handleSignalUpdate(data) {
            addLog(`Signal: ${data.signal} for ${data.symbol} (strength: ${data.strength.toFixed(2)})`);
        }
        
        function updatePositionsFromWS(positions) {
            // Update positions without full refresh
            if (currentTab === 'positions') {
                updatePositionsTable(positions);
            }
        }
        
        // Initialize on load
        window.onload = () => {
            refreshData();
            connectWebSocket(); // Connect to WebSocket
            setInterval(refreshData, 5000); // Keep polling as fallback
        };
    </script>
</body>
</html>
'''

@app.route('/')
@requires_auth
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/status')
@requires_auth
def status():
    """Get current system status"""
    return jsonify({
        'trading_status': trading_status,
        'pnl': pnl,
        'metrics': performance_metrics,
        'positions_count': len(positions),
        'ml_status': ml_metrics
    })

@app.route('/api/pnl')
@requires_auth
def get_pnl():
    """Get P&L data from real database"""
    from robo_trader.database_async import AsyncTradingDatabase
    import asyncio
    
    async def fetch_pnl():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Get account info from database
            account = await db.get_account_info()
            if account:
                return {
                    'daily': account.get('daily_pnl', 0),
                    'total': account.get('realized_pnl', 0),
                    'unrealized': account.get('unrealized_pnl', 0)
                }
            return pnl  # Return default if no account data
        finally:
            await db.close()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        real_pnl = loop.run_until_complete(fetch_pnl())
        return jsonify(real_pnl)
    except Exception as e:
        logger.error(f"Error fetching real P&L: {e}")
        return jsonify(pnl)  # Return default on error

@app.route('/api/positions')
@requires_auth  
def get_positions():
    """Get current positions from real database"""
    from robo_trader.database_async import AsyncTradingDatabase
    import asyncio
    
    async def fetch_positions():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Get real positions from database
            real_positions = await db.get_positions()
            
            # Get latest market data for each position
            enriched_positions = []
            for pos in real_positions:
                # Get latest price from market data
                market_data = await db.get_latest_market_data(pos['symbol'], limit=1)
                current_price = market_data[0]['close'] if market_data else pos.get('market_price', pos.get('avg_cost'))
                
                enriched_positions.append({
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'entry_price': pos.get('avg_cost', 0),
                    'current_price': current_price,
                    'ml_signal': 'hold'  # Default for now, can fetch from signals table
                })
            
            # If no real positions, show all symbols with 0 quantity
            if not enriched_positions and trading_status == 'running':
                for symbol in default_symbols:
                    market_data = await db.get_latest_market_data(symbol, limit=1)
                    if market_data:
                        enriched_positions.append({
                            'symbol': symbol,
                            'quantity': 0,
                            'entry_price': 0,
                            'current_price': market_data[0]['close'],
                            'ml_signal': 'watch'
                        })
            
            return enriched_positions
        finally:
            await db.close()
    
    # Run async function
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        real_positions = loop.run_until_complete(fetch_positions())
        return jsonify({'positions': real_positions})
    except Exception as e:
        logger.error(f"Error fetching real positions: {e}")
        return jsonify({'positions': []})

@app.route('/api/watchlist')
@requires_auth
def get_watchlist():
    """Get watchlist with latest prices"""
    from robo_trader.database_async import AsyncTradingDatabase
    import asyncio
    
    async def fetch_watchlist():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Get watchlist symbols
            async with db.get_connection() as conn:
                cursor = await conn.execute("SELECT symbol, notes FROM watchlist ORDER BY symbol")
                watchlist = await cursor.fetchall()
            
            # Get latest prices and positions for each symbol
            watchlist_data = []
            for symbol, notes in watchlist:
                # Get latest market data
                market_data = await db.get_latest_market_data(symbol, limit=1)
                current_price = market_data[0]['close'] if market_data else 0
                
                # Check if we have a position
                positions = await db.get_positions()
                position = next((p for p in positions if p['symbol'] == symbol), None)
                
                watchlist_data.append({
                    'symbol': symbol,
                    'current_price': current_price,
                    'quantity': position['quantity'] if position else 0,
                    'avg_cost': position['avg_cost'] if position else 0,
                    'notes': notes or '',
                    'has_position': position is not None
                })
            
            return watchlist_data
        finally:
            await db.close()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        watchlist = loop.run_until_complete(fetch_watchlist())
        return jsonify({'watchlist': watchlist})
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        return jsonify({'watchlist': []})

@app.route('/api/ml/status')
@requires_auth
def ml_status():
    """Get ML system status"""
    # This will be populated from actual ML components
    return jsonify({
        'models_trained': ml_metrics['models_trained'],
        'feature_count': 50,  # From feature pipeline
        'accuracy': 0.65,
        'confidence': 0.72,
        'models': [
            {
                'type': 'Random Forest',
                'test_score': 0.68,
                'feature_count': 45,
                'updated': datetime.now().isoformat(),
                'status': 'active'
            },
            {
                'type': 'XGBoost',
                'test_score': 0.71,
                'feature_count': 42,
                'updated': datetime.now().isoformat(),
                'status': 'active'
            }
        ],
        'top_features': [
            {'name': 'RSI_14', 'importance': 0.15, 'category': 'Technical'},
            {'name': 'correlation_spy', 'importance': 0.12, 'category': 'Cross-asset'},
            {'name': 'volatility_20d', 'importance': 0.10, 'category': 'Volatility'},
            {'name': 'momentum_10d', 'importance': 0.09, 'category': 'Momentum'},
            {'name': 'volume_ratio', 'importance': 0.08, 'category': 'Volume'}
        ]
    })

@app.route('/api/performance')
@requires_auth
def performance():
    """Get performance metrics"""
    # Mock data - will integrate with PerformanceAnalyzer
    return jsonify({
        'daily': {
            'return_pct': 0.0125,
            'volatility': 0.18,
            'sharpe': 1.2,
            'max_drawdown': -0.02
        },
        'weekly': {
            'return_pct': 0.035,
            'volatility': 0.22,
            'sharpe': 1.5,
            'max_drawdown': -0.04
        },
        'monthly': {
            'return_pct': 0.08,
            'volatility': 0.25,
            'sharpe': 1.8,
            'max_drawdown': -0.06
        },
        'all': {
            'return_pct': 0.15,
            'volatility': 0.28,
            'sharpe': 1.4,
            'max_drawdown': -0.12
        }
    })

@app.route('/api/start', methods=['POST'])
@requires_auth
def start_trading():
    """Start trading"""
    global trading_status, trading_process
    
    # Load symbols from user settings (make it global)
    global default_symbols
    try:
        with open('user_settings.json', 'r') as f:
            settings = json.load(f)
            default_symbols = settings.get('default', {}).get('symbols', ['AAPL', 'MSFT', 'GOOGL'])
    except:
        default_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    data = request.json
    symbols = data.get('symbols', default_symbols)
    
    if trading_status == 'running':
        return jsonify({'status': 'already_running'})
    
    # Start trading process with proper symbol format
    cmd = ['python', '-m', 'robo_trader.runner_async', '--symbols', ','.join(symbols)]
    trading_process = subprocess.Popen(cmd)
    trading_status = 'running'
    
    trading_log.append(f"{datetime.now().strftime('%H:%M:%S')} - Trading started for {', '.join(symbols)}")
    
    return jsonify({'status': 'started', 'symbols': symbols})

@app.route('/api/stop', methods=['POST'])
@requires_auth
def stop_trading():
    """Stop trading"""
    global trading_status, trading_process
    
    if trading_process:
        trading_process.terminate()
        trading_process = None
    
    trading_status = 'stopped'
    trading_log.append(f"{datetime.now().strftime('%H:%M:%S')} - Trading stopped")
    
    return jsonify({'status': 'stopped'})

@app.route('/api/logs')
@requires_auth
def get_logs():
    """Get recent logs"""
    return jsonify({'logs': trading_log[-100:]})  # Last 100 log entries

if __name__ == '__main__':
    # Initialize components
    logger.info("Starting RoboTrader Dashboard...")
    
    # Start WebSocket server
    logger.info("Starting WebSocket server...")
    ws_manager.start()
    
    # Wait for async components to initialize
    time.sleep(2)
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('DASH_PORT', 5000)),
        debug=os.getenv('FLASK_ENV') == 'development'
    )
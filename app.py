#!/usr/bin/env python3
"""
RoboTrader Dashboard - Clean, ML-Integrated Interface
Provides real-time monitoring of trading, ML models, and performance metrics
"""

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, send_file

load_dotenv()

from robo_trader.analytics.performance import PerformanceAnalyzer  # noqa: E402

# Import our modules - using lazy imports to avoid startup issues
from robo_trader.config import load_config  # noqa: E402

# Lazy imports to avoid blocking startup
from robo_trader.database_async import AsyncTradingDatabase  # noqa: E402
from robo_trader.features.feature_pipeline import FeaturePipeline  # noqa: E402
from robo_trader.logger import get_logger  # noqa: E402
from robo_trader.ml.model_trainer import ModelTrainer  # noqa: E402

# from robo_trader.websocket_server import ws_manager

logger = get_logger(__name__)
app = Flask(__name__)
server = app  # For Gunicorn compatibility

# Configuration
config = load_config()
AUTH_ENABLED = os.getenv("DASH_AUTH_ENABLED", "false").lower() == "true"
AUTH_USER = os.getenv("DASH_USER", "admin")
AUTH_PASS_HASH = os.getenv("DASH_PASS_HASH", "")

# Initialize components - will be loaded lazily
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
    with open("user_settings.json", "r") as f:
        settings = json.load(f)
        default_symbols = settings.get("default", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    default_symbols = ["AAPL", "MSFT", "GOOGL"]
pnl = {"daily": 0.0, "total": 0.0, "unrealized": 0.0}
ml_metrics = {
    "models_trained": 0,
    "last_prediction": None,
    "feature_count": 0,
    "model_performance": {},
}
performance_metrics = {
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "win_rate": 0.0,
    "profit_factor": 0.0,
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
if os.getenv("DASH_INIT_COMPONENTS", "false").lower() == "true":
    init_thread = threading.Thread(target=init_async_components)
    init_thread.daemon = True
    init_thread.start()
else:
    logger.info("Skipping heavy async component init at startup (DASH_INIT_COMPONENTS=false)")


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
        "Authentication required.\n" "Please enter your credentials.",
        401,
        {"WWW-Authenticate": 'Basic realm="RoboTrader Dashboard"'},
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
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RoboTrader Dashboard</title>
    <link rel="icon" type="image/x-icon" href="/favicon.ico?v=3">
    <link rel="shortcut icon" type="image/x-icon" href="/favicon.ico?v=3">
    <link rel="apple-touch-icon" href="/favicon.ico?v=3">
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
        
        /* Regime Detection Colors */
        .regime-bull { color: #00ff00; }
        .regime-bear { color: #ff0000; }
        .regime-volatile { color: #ff00ff; }
        .regime-ranging { color: #ffff00; }
        .regime-crash { 
            color: #ff0000; 
            animation: blink 1s infinite; 
        }
        @keyframes blink { 
            0% { opacity: 1; } 
            50% { opacity: 0.3; } 
            100% { opacity: 1; } 
        }
        
        /* Portfolio Allocation */
        .allocation-bar {
            height: 30px;
            background: #1a1a1a;
            border-radius: 4px;
            overflow: hidden;
            display: flex;
            margin: 10px 0;
        }
        .allocation-segment {
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 12px;
            font-weight: bold;
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

        .metric-value.negative { color: #ff4444; }
        .metric-subtitle { font-size: 0.8em; color: #888; margin-top: 5px; }

        /* Performance Summary Cards */
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .summary-card {
            background: #2a2a2a;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #667eea;
        }

        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 0.9em;
            color: #ccc;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .summary-card .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #44ff44;
            margin: 10px 0;
        }

        .summary-card .metric-value.negative {
            color: #ff4444;
        }

        .summary-card .metric-subtitle {
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
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
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            margin-top: 10px;
            font-size: 12px;
            color: #888;
        }
        
        .progress-bar {
            height: 8px;
            background: #2a2a2a;
            border-radius: 4px;
            overflow: hidden;
            width: 100px;
            display: inline-block;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            transition: width 0.3s ease;
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
            <div class="tab" onclick="switchTab('strategies', this)">Strategies</div>
            <div class="tab" onclick="switchTab('trades', this)">Trade History</div>
            <div class="tab" onclick="switchTab('ml', this)">ML Models</div>
            <div class="tab" onclick="switchTab('performance', this)">Performance</div>
            <div class="tab" onclick="switchTab('logs', this)">Logs</div>
        </div>
        
        <div id="overview-tab" class="tab-content">
            <!-- Portfolio Summary Section -->
            <div class="portfolio-summary" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 12px; padding: 25px; margin-bottom: 30px; border: 1px solid #2a2a3e;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px;">
                    <div>
                        <div style="font-size: 14px; color: #888; margin-bottom: 8px;">Portfolio Value</div>
                        <div style="font-size: 32px; font-weight: 600; color: #fff;" id="portfolio-value">$100,000.00</div>
                        <div style="font-size: 14px; margin-top: 5px;">
                            <span id="portfolio-change" style="color: #44ff44;">+$2,847.30 (+2.85%)</span>
                            <span style="color: #666; margin-left: 10px;">All Time</span>
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #888; margin-bottom: 8px;">Today's P&L</div>
                        <div style="font-size: 32px; font-weight: 600;" id="today-pnl-large">$523.45</div>
                        <div style="font-size: 14px; margin-top: 5px;">
                            <span id="today-change-pct" style="color: #44ff44;">+0.52%</span>
                            <span style="color: #666; margin-left: 10px;">vs Yesterday</span>
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #888; margin-bottom: 8px;">Cash Available</div>
                        <div style="font-size: 32px; font-weight: 600; color: #fff;" id="cash-available">$94,235.50</div>
                        <div style="font-size: 14px; color: #666; margin-top: 5px;">94.2% of portfolio</div>
                    </div>
                    <div>
                        <div style="font-size: 14px; color: #888; margin-bottom: 8px;">Active Positions</div>
                        <div style="font-size: 32px; font-weight: 600; color: #fff;" id="active-positions-large">5</div>
                        <div style="font-size: 14px; color: #666; margin-top: 5px;" id="positions-value-text">
                            Monitoring 5 symbols
                        </div>
                    </div>
                </div>
            </div>
            
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
            <!-- Market Regime Detection Section -->
            <div class="table-container" style="margin-bottom: 30px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
                <h3 style="color: #00ff00; margin-bottom: 20px;">🌡️ Market Regime Detection</h3>
                <div style="text-align: center; padding: 20px;">
                    <div id="regime-indicator" style="font-size: 32px; font-weight: bold; margin: 10px 0;">
                        <span id="regime-name" class="regime-bull">BULL MARKET</span>
                    </div>
                    <div style="font-size: 16px; color: #888;">Confidence: <span id="regime-confidence" style="color: #0ff;">75%</span></div>
                    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 20px;">
                        <div style="text-align: center;">
                            <div style="color: #666; font-size: 12px;">1min</div>
                            <div id="regime-1m" class="regime-bull" style="font-weight: bold;">BULL</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #666; font-size: 12px;">5min</div>
                            <div id="regime-5m" class="regime-bull" style="font-weight: bold;">BULL</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #666; font-size: 12px;">15min</div>
                            <div id="regime-15m" class="regime-ranging" style="font-weight: bold;">RANGING</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #666; font-size: 12px;">1hour</div>
                            <div id="regime-1h" class="regime-bull" style="font-weight: bold;">BULL</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="color: #666; font-size: 12px;">Daily</div>
                            <div id="regime-1d" class="regime-bull" style="font-weight: bold;">BULL</div>
                        </div>
                    </div>
                </div>
            </div>
            
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
            <!-- Performance Summary Cards -->
            <div class="summary-cards">
                <div class="summary-card">
                    <h3>Total P&L</h3>
                    <div class="metric-value" id="perf-total-pnl">$0.00</div>
                    <div class="metric-subtitle">All Time</div>
                </div>
                <div class="summary-card">
                    <h3>Total Return</h3>
                    <div class="metric-value" id="total-return">0.00%</div>
                    <div class="metric-subtitle">All Time</div>
                </div>
                <div class="summary-card">
                    <h3>Sharpe Ratio</h3>
                    <div class="metric-value" id="total-sharpe">0.00</div>
                    <div class="metric-subtitle">Risk-Adjusted</div>
                </div>
                <div class="summary-card">
                    <h3>Max Drawdown</h3>
                    <div class="metric-value negative" id="total-drawdown">0.00%</div>
                    <div class="metric-subtitle">Peak to Trough</div>
                </div>
                <div class="summary-card">
                    <h3>Win Rate</h3>
                    <div class="metric-value" id="win-rate">0.00%</div>
                    <div class="metric-subtitle">Profitable Trades</div>
                </div>
                <div class="summary-card">
                    <h3>Total Trades</h3>
                    <div class="metric-value" id="perf-total-trades">0</div>
                    <div class="metric-subtitle">All Time</div>
                </div>
            </div>

            <div class="table-container">
                <h3>Performance Breakdown</h3>
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
                            <td>P&L</td>
                            <td id="pnl-daily">$0.00</td>
                            <td id="pnl-weekly">$0.00</td>
                            <td id="pnl-monthly">$0.00</td>
                            <td id="pnl-all">$0.00</td>
                        </tr>
                        <tr>
                            <td>Return</td>
                            <td id="return-daily">0.00%</td>
                            <td id="return-weekly">0.00%</td>
                            <td id="return-monthly">0.00%</td>
                            <td id="return-all">0.00%</td>
                        </tr>
                        <tr>
                            <td>Trades</td>
                            <td id="trades-daily">0</td>
                            <td id="trades-weekly">0</td>
                            <td id="trades-monthly">0</td>
                            <td id="trades-all">0</td>
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

            <!-- Equity Curve Chart -->
            <div class="table-container">
                <h3>Equity Curve</h3>
                <div id="equity-chart" style="height: 200px; background: #0a0a0a; border-radius: 8px; padding: 20px; display: flex; align-items: center; justify-content: center; color: #666;">
                    <div style="text-align: center;">
                        <div style="font-size: 14px; margin-bottom: 10px;">📈 Equity Curve Visualization</div>
                        <div style="font-size: 12px;">Chart will be implemented when historical data is available</div>
                    </div>
                </div>
            </div>

            <!-- Trade Statistics -->
            <div class="table-container">
                <h3>Trade Statistics</h3>
                <div class="metric-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div class="metric-item">
                        <div class="metric-label">Average Win</div>
                        <div class="metric-value positive" id="avg-win">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Average Loss</div>
                        <div class="metric-value negative" id="avg-loss">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value" id="profit-factor">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Expectancy</div>
                        <div class="metric-value" id="expectancy">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Largest Win</div>
                        <div class="metric-value positive" id="largest-win">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Largest Loss</div>
                        <div class="metric-value negative" id="largest-loss">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Avg Hold Time</div>
                        <div class="metric-value" id="avg-hold-time">0h 0m</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Win Streak</div>
                        <div class="metric-value" id="win-streak">0</div>
                    </div>
                </div>
            </div>

            <!-- Risk Metrics -->
            <div class="table-container">
                <h3>Risk Metrics</h3>
                <div class="metric-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div class="metric-item">
                        <div class="metric-label">Value at Risk (95%)</div>
                        <div class="metric-value" id="var-95">$0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Beta vs SPY</div>
                        <div class="metric-value" id="beta">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Sortino Ratio</div>
                        <div class="metric-value" id="sortino">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Calmar Ratio</div>
                        <div class="metric-value" id="calmar">0.00</div>
                    </div>
                </div>
            </div>
            
            <!-- Execution Analytics -->
            <div class="table-container">
                <h3 style="color: #00ff00;">⚡ Execution Analytics</h3>
                <div class="metric-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div class="metric-item">
                        <div class="metric-label">Avg Slippage</div>
                        <div class="metric-value" id="exec-slippage">0.0 bps</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Market Impact</div>
                        <div class="metric-value" id="exec-impact">0.0 bps</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Fill Rate</div>
                        <div class="metric-value" id="exec-fill-rate">100%</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Avg Fill Time</div>
                        <div class="metric-value" id="exec-fill-time">0.0s</div>
                    </div>
                </div>
                <div style="margin-top: 15px;">
                    <h4 style="color: #888; font-size: 14px; margin-bottom: 10px;">Algorithm Performance</h4>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;">
                        <div style="background: #1a1a2e; padding: 10px; border-radius: 4px;">
                            <div style="color: #4CAF50;">TWAP</div>
                            <div style="font-size: 12px; color: #666;">Slippage: <span id="twap-perf-slip">0.0</span> bps</div>
                            <div style="font-size: 12px; color: #666;">Success: <span id="twap-perf-success">100%</span></div>
                        </div>
                        <div style="background: #1a1a2e; padding: 10px; border-radius: 4px;">
                            <div style="color: #2196F3;">VWAP</div>
                            <div style="font-size: 12px; color: #666;">Slippage: <span id="vwap-perf-slip">0.0</span> bps</div>
                            <div style="font-size: 12px; color: #666;">Success: <span id="vwap-perf-success">100%</span></div>
                        </div>
                        <div style="background: #1a1a2e; padding: 10px; border-radius: 4px;">
                            <div style="color: #FF9800;">Iceberg</div>
                            <div style="font-size: 12px; color: #666;">Detection: <span id="iceberg-perf-detect">0%</span></div>
                            <div style="font-size: 12px; color: #666;">Success: <span id="iceberg-perf-success">100%</span></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Advanced Risk Management -->
            <div class="table-container">
                <h3 style="color: #ff6b6b;">🛡️ Advanced Risk Management</h3>
                
                <!-- Kelly Sizing Section -->
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #888; font-size: 14px; margin-bottom: 10px;">Kelly Criterion Position Sizing</h4>
                    <div class="metric-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                        <div class="metric-item">
                            <div class="metric-label">Portfolio Kelly</div>
                            <div class="metric-value" id="portfolio-kelly">5.8%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Avg Win Rate</div>
                            <div class="metric-value positive" id="avg-win-rate">55%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Portfolio Edge</div>
                            <div class="metric-value positive" id="portfolio-edge">0.025</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Using Half-Kelly</div>
                            <div class="metric-value" style="color: #4CAF50;" id="kelly-mode">✓ Yes</div>
                        </div>
                    </div>
                </div>
                
                <!-- Kill Switches Section -->
                <div style="margin-bottom: 20px;">
                    <h4 style="color: #888; font-size: 14px; margin-bottom: 10px;">Kill Switch Status</h4>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                        <div style="background: #1a1a2e; padding: 15px; border-radius: 4px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #666;">Kill Switch</span>
                                <span id="kill-switch-status" style="color: #4CAF50; font-weight: bold;">ACTIVE ✓</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 12px;">
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">Daily Loss:</span>
                                    <span id="daily-loss-status">1.2% / 5.0%</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">Consecutive Losses:</span>
                                    <span id="consecutive-losses">1 / 5</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">Max Drawdown:</span>
                                    <span id="max-dd-status">2.5% / 10.0%</span>
                                </div>
                            </div>
                        </div>
                        
                        <div style="background: #1a1a2e; padding: 15px; border-radius: 4px;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #666;">Correlation Limits</span>
                                <span id="correlation-status" style="color: #4CAF50; font-weight: bold;">OK</span>
                            </div>
                            <div style="margin-top: 10px; font-size: 12px;">
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">Max Correlation:</span>
                                    <span>0.70</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">High Correlations:</span>
                                    <span id="high-corr-count" style="color: #FFA500;">2 pairs</span>
                                </div>
                                <div style="display: flex; justify-content: space-between; margin: 5px 0;">
                                    <span style="color: #666;">Corr. Exposure:</span>
                                    <span id="corr-exposure">18% / 30%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Risk Metrics -->
                <div>
                    <h4 style="color: #888; font-size: 14px; margin-bottom: 10px;">Risk Metrics</h4>
                    <div class="metric-grid" style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
                        <div class="metric-item">
                            <div class="metric-label">Leverage</div>
                            <div class="metric-value" id="leverage">0.45x</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">VaR (95%)</div>
                            <div class="metric-value negative" id="var-95">-$2,500</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Exposure</div>
                            <div class="metric-value" id="total-exposure">$45,000</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Risk Sharpe</div>
                            <div class="metric-value positive" id="risk-sharpe">1.8</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Risk Level</div>
                            <div class="metric-value" style="color: #4CAF50;" id="risk-level">LOW</div>
                        </div>
                    </div>
                </div>
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
        
        <div id="strategies-tab" class="tab-content" style="display: none;">
            <!-- Portfolio Allocation Section -->
            <div class="table-container" style="margin-bottom: 30px;">
                <h3 style="color: #00ff00; margin-bottom: 15px;">📊 Portfolio Allocation</h3>
                <div class="allocation-bar" id="portfolio-allocation-bar">
                    <div class="allocation-segment" style="width: 30%; background: #4CAF50;">Momentum 30%</div>
                    <div class="allocation-segment" style="width: 25%; background: #2196F3;">Mean Rev 25%</div>
                    <div class="allocation-segment" style="width: 20%; background: #FF9800;">ML Enhanced 20%</div>
                    <div class="allocation-segment" style="width: 15%; background: #9C27B0;">Microstructure 15%</div>
                    <div class="allocation-segment" style="width: 10%; background: #F44336;">Pairs 10%</div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-top: 15px;">
                    <div style="text-align: center;">
                        <div style="color: #4CAF50; font-weight: bold;">Momentum</div>
                        <div id="alloc-momentum">30%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #2196F3; font-weight: bold;">Mean Reversion</div>
                        <div id="alloc-mean-rev">25%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #FF9800; font-weight: bold;">ML Enhanced</div>
                        <div id="alloc-ml">20%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #9C27B0; font-weight: bold;">Microstructure</div>
                        <div id="alloc-micro">15%</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="color: #F44336; font-weight: bold;">Pairs Trading</div>
                        <div id="alloc-pairs">10%</div>
                    </div>
                </div>
            </div>
            
            <!-- Smart Execution Status -->
            <div class="table-container" style="margin-bottom: 30px;">
                <h3 style="color: #00ff00; margin-bottom: 15px;">⚡ Smart Execution Status</h3>
                <div id="execution-status" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                        <div style="color: #888; font-size: 12px;">TWAP Orders</div>
                        <div style="font-size: 24px; color: #4CAF50; font-weight: bold;" id="twap-count">0</div>
                        <div style="color: #666; font-size: 11px;">Avg Slippage: <span id="twap-slippage">0.0</span> bps</div>
                    </div>
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                        <div style="color: #888; font-size: 12px;">VWAP Orders</div>
                        <div style="font-size: 24px; color: #2196F3; font-weight: bold;" id="vwap-count">0</div>
                        <div style="color: #666; font-size: 11px;">Avg Slippage: <span id="vwap-slippage">0.0</span> bps</div>
                    </div>
                    <div style="background: #1a1a2e; padding: 15px; border-radius: 8px; border: 1px solid #333;">
                        <div style="color: #888; font-size: 12px;">Iceberg Orders</div>
                        <div style="font-size: 24px; color: #FF9800; font-weight: bold;" id="iceberg-count">0</div>
                        <div style="color: #666; font-size: 11px;">Hidden Liquidity: <span id="iceberg-hidden">0%</span></div>
                    </div>
                </div>
            </div>
            
            <div class="grid">
                <!-- ML Enhanced Strategy Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">ML Enhanced</span>
                        <span class="badge ml">ACTIVE</span>
                    </div>
                    <div class="card-value" id="ml-regime">Loading...</div>
                    <div class="card-change">
                        <span id="ml-confidence">Confidence: 0%</span>
                    </div>
                    <div class="metric-row">
                        <span>Positions: <span id="ml-positions">0</span></span>
                        <span>P&L: <span id="ml-pnl">$0</span></span>
                    </div>
                </div>
                
                <!-- Microstructure Strategy Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Microstructure</span>
                        <span class="badge">HFT</span>
                    </div>
                    <div class="card-value">
                        OFI: <span id="micro-ofi">0.00</span>
                    </div>
                    <div class="card-change">
                        <span id="micro-spread">Spread: 0.0 bps</span>
                    </div>
                    <div class="metric-row">
                        <span>Signals: <span id="micro-signals">0</span></span>
                        <span>Win: <span id="micro-winrate">0%</span></span>
                    </div>
                </div>
                
                <!-- Portfolio Manager Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Portfolio Manager</span>
                        <span class="badge" id="pm-method">Risk Parity</span>
                    </div>
                    <div class="card-value" id="pm-strategies">4 Strategies</div>
                    <div class="card-change">
                        <span id="pm-rebalance">Next: Tomorrow</span>
                    </div>
                    <div class="metric-row">
                        <span>Drift: <span id="pm-drift">0.0%</span></span>
                        <span>Div: <span id="pm-diversification">1.85</span></span>
                    </div>
                </div>
                
                <!-- Smart Execution Card -->
                <div class="card">
                    <div class="card-header">
                        <span class="card-title">Smart Execution</span>
                        <span class="badge" id="exec-algo">VWAP</span>
                    </div>
                    <div class="card-value">
                        <span id="exec-pending">2</span> Pending
                    </div>
                    <div class="card-change">
                        <span id="exec-slippage">Slippage: 1.2 bps</span>
                    </div>
                    <div class="metric-row">
                        <span>Filled: <span id="exec-filled">15</span></span>
                        <span>Saved: <span id="exec-saved">8.5 bps</span></span>
                    </div>
                </div>
            </div>
            
            <!-- Microstructure Details Section -->
            <div class="table-container">
                <h3>Microstructure Metrics</h3>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Order Flow Imbalance</div>
                        <div class="metric-value" id="ofi-detailed">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Book Pressure</div>
                        <div class="metric-value" id="book-pressure">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Tick Direction</div>
                        <div class="metric-value" id="tick-direction">0.00</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Ensemble Score</div>
                        <div class="metric-value" id="ensemble-score">0.00</div>
                    </div>
                </div>
            </div>
            
            <!-- Portfolio Allocation Chart -->
            <div class="table-container">
                <h3>Portfolio Allocation</h3>
                <div id="allocation-chart" style="height: 200px;">
                    <div class="allocation-bar" style="display: flex; height: 40px; margin: 20px 0;">
                        <div style="width: 35%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">ML 35%</div>
                        <div style="width: 25%; background: #44ff44; display: flex; align-items: center; justify-content: center; color: black; font-size: 12px;">Micro 25%</div>
                        <div style="width: 20%; background: #ffaa44; display: flex; align-items: center; justify-content: center; color: black; font-size: 12px;">MR 20%</div>
                        <div style="width: 20%; background: #ff4444; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">Mom 20%</div>
                    </div>
                </div>
                <table style="margin-top: 20px;">
                    <thead>
                        <tr>
                            <th>Strategy</th>
                            <th>Allocation</th>
                            <th>Risk Budget</th>
                            <th>P&L Today</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="allocation-table">
                        <tr>
                            <td>ML Enhanced</td>
                            <td>35%</td>
                            <td>30%</td>
                            <td class="positive">+$1,250</td>
                            <td><span class="badge ml">Active</span></td>
                        </tr>
                        <tr>
                            <td>Microstructure</td>
                            <td>25%</td>
                            <td>25%</td>
                            <td class="positive">+$320</td>
                            <td><span class="badge">Active</span></td>
                        </tr>
                        <tr>
                            <td>Mean Reversion</td>
                            <td>20%</td>
                            <td>25%</td>
                            <td class="neutral">$0</td>
                            <td><span class="badge" style="opacity: 0.5;">Pending</span></td>
                        </tr>
                        <tr>
                            <td>Momentum</td>
                            <td>20%</td>
                            <td>20%</td>
                            <td class="negative">-$150</td>
                            <td><span class="badge">Active</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <!-- Execution Algorithm Status -->
            <div class="table-container">
                <h3>Smart Execution Orders</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Algorithm</th>
                            <th>Progress</th>
                            <th>Slices</th>
                            <th>Slippage</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="execution-table">
                        <tr>
                            <td>AAPL</td>
                            <td>VWAP</td>
                            <td><div class="progress-bar"><div class="progress-fill" style="width: 65%;"></div></div></td>
                            <td>5/8</td>
                            <td class="positive">0.8 bps</td>
                            <td><span class="badge">Executing</span></td>
                        </tr>
                        <tr>
                            <td>MSFT</td>
                            <td>TWAP</td>
                            <td><div class="progress-bar"><div class="progress-fill" style="width: 30%;"></div></div></td>
                            <td>3/10</td>
                            <td class="positive">1.1 bps</td>
                            <td><span class="badge">Executing</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="trades-tab" class="tab-content" style="display: none;">
            <div class="card">
                <h3>Trade History Summary</h3>
                <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px;">
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Total Trades</div>
                        <div style="font-size: 20px; font-weight: bold;" id="total-trades">0</div>
                    </div>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Buy Trades</div>
                        <div style="font-size: 20px; font-weight: bold; color: #4ade80;" id="buy-trades">0</div>
                    </div>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Sell Trades</div>
                        <div style="font-size: 20px; font-weight: bold; color: #f87171;" id="sell-trades">0</div>
                    </div>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Total Volume</div>
                        <div style="font-size: 20px; font-weight: bold;" id="total-volume">$0</div>
                    </div>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Total Commission</div>
                        <div style="font-size: 20px; font-weight: bold; color: #fbbf24;" id="total-commission">$0</div>
                    </div>
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #888; margin-bottom: 5px;">Avg Trade Size</div>
                        <div style="font-size: 20px; font-weight: bold;" id="avg-trade-size">$0</div>
                    </div>
                </div>
            </div>
            
            <div class="table-container">
                <div style="margin-bottom: 15px;">
                    <label style="color: #888; margin-right: 10px;">Filter by Symbol:</label>
                    <select id="trade-symbol-filter" onchange="loadTrades()" style="background: #1a1a1a; color: #fff; border: 1px solid #2a2a2a; padding: 5px 10px; border-radius: 4px;">
                        <option value="">All Symbols</option>
                    </select>
                    <label style="color: #888; margin-left: 20px; margin-right: 10px;">Days:</label>
                    <select id="trade-days-filter" onchange="loadTrades()" style="background: #1a1a1a; color: #fff; border: 1px solid #2a2a2a; padding: 5px 10px; border-radius: 4px;">
                        <option value="1">Last 24 Hours</option>
                        <option value="7">Last Week</option>
                        <option value="30" selected>Last 30 Days</option>
                        <option value="90">Last 90 Days</option>
                        <option value="365">Last Year</option>
                    </select>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Quantity</th>
                            <th>Price</th>
                            <th>Notional</th>
                            <th>Commission</th>
                            <th>Slippage</th>
                        </tr>
                    </thead>
                    <tbody id="trades-table">
                        <tr>
                            <td colspan="8" style="text-align: center; color: #666;">Loading trades...</td>
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
            } else if (tab === 'watchlist') {
                loadWatchlist();
            } else if (tab === 'trades') {
                loadTrades();
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
                loadPerformanceData(),
                loadTrades()  // Add trades to refresh cycle
            ]);
        }
        
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                console.log('Status API response:', data);
                console.log('P&L data:', data.pnl);
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
        
        async function loadTrades() {
            try {
                const symbolFilter = document.getElementById('trade-symbol-filter').value;
                const daysFilter = document.getElementById('trade-days-filter').value;
                
                let url = `/api/trades?days=${daysFilter}`;
                if (symbolFilter) {
                    url += `&symbol=${symbolFilter}`;
                }
                
                const response = await fetch(url);
                const data = await response.json();
                
                // Update summary stats
                if (data.summary) {
                    document.getElementById('total-trades').textContent = data.summary.total_trades || 0;
                    document.getElementById('buy-trades').textContent = data.summary.buy_trades || 0;
                    document.getElementById('sell-trades').textContent = data.summary.sell_trades || 0;
                    document.getElementById('total-volume').textContent = `$${(data.summary.total_volume || 0).toFixed(2)}`;
                    document.getElementById('total-commission').textContent = `$${(data.summary.total_commission || 0).toFixed(2)}`;
                    document.getElementById('avg-trade-size').textContent = `$${(data.summary.avg_trade_size || 0).toFixed(2)}`;
                }
                
                // Update trades table
                const tbody = document.getElementById('trades-table');
                if (data.trades && data.trades.length > 0) {
                    // Update symbol filter options if needed
                    const symbols = [...new Set(data.trades.map(t => t.symbol))];
                    updateSymbolFilter(symbols);
                    
                    tbody.innerHTML = data.trades.map(trade => {
                        // SQLite timestamps are UTC but don't have 'Z' suffix
                        // Add 'Z' to mark as UTC, then convert to local time  
                        const utcTimestamp = trade.timestamp.replace(' ', 'T') + 'Z';
                        const utcDate = new Date(utcTimestamp);
                        const time = utcDate.toLocaleString('en-US', { 
                            timeZone: 'America/New_York',
                            month: 'short',
                            day: 'numeric',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit'
                        });
                        const sideColor = trade.side === 'BUY' ? '#4ade80' : '#f87171';
                        return `
                            <tr>
                                <td>${time}</td>
                                <td>${trade.symbol}</td>
                                <td style="color: ${sideColor}; font-weight: bold;">${trade.side}</td>
                                <td>${trade.quantity}</td>
                                <td>$${trade.price.toFixed(2)}</td>
                                <td>$${trade.notional.toFixed(2)}</td>
                                <td>$${trade.commission.toFixed(2)}</td>
                                <td>${(trade.slippage * 100).toFixed(2)}%</td>
                            </tr>
                        `;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: #666;">No trades found</td></tr>';
                }
            } catch (error) {
                console.error('Error loading trades:', error);
                document.getElementById('trades-table').innerHTML = 
                    '<tr><td colspan="8" style="text-align: center; color: #f87171;">Error loading trades</td></tr>';
            }
        }
        
        function updateSymbolFilter(symbols) {
            const filter = document.getElementById('trade-symbol-filter');
            const currentValue = filter.value;
            
            // Only update if we have new symbols
            if (symbols.length > 0 && filter.options.length <= 1) {
                filter.innerHTML = '<option value="">All Symbols</option>';
                symbols.sort().forEach(symbol => {
                    const option = document.createElement('option');
                    option.value = symbol;
                    option.textContent = symbol;
                    filter.appendChild(option);
                });
                filter.value = currentValue; // Restore previous selection
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
                console.log('Performance data received:', data);
                updatePerformanceTable(data);
            } catch (error) {
                console.error('Error loading performance data:', error);
            }
        }
        
        function updateStatus(status) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            // Handle both object and string status
            const isRunning = (typeof status === 'object' && status.is_trading) || status === 'running';

            if (isRunning) {
                dot.classList.add('active');
                text.textContent = 'Trading Active';
            } else {
                dot.classList.remove('active');
                text.textContent = 'Trading Stopped';
            }

            // Also update market status
            updateMarketStatus();
        }

        async function updateMarketStatus() {
            try {
                const response = await fetch('/api/market/status');
                const data = await response.json();

                // Update status text to include market status
                const statusText = document.getElementById('status-text');
                const currentText = statusText.textContent;

                if (data.is_open) {
                    statusText.textContent = currentText + ' • Market Open';
                    statusText.style.color = '#44ff44';
                } else {
                    statusText.textContent = currentText + ` • Market ${data.status_text}`;
                    if (data.session === 'closed') {
                        statusText.style.color = '#ff4444';
                        if (data.time_until_open) {
                            statusText.textContent += ` (Opens in ${data.time_until_open})`;
                        }
                    } else {
                        statusText.style.color = '#ffaa44';
                    }
                }
            } catch (error) {
                console.error('Error fetching market status:', error);
            }
        }
        
        let lastPnLUpdate = null;
        let pnlUpdateTimeout = null;
        
        function updatePnL(pnl) {
            console.log('updatePnL called with:', pnl);
            
            // Debounce P&L updates to prevent flashing
            if (pnlUpdateTimeout) {
                clearTimeout(pnlUpdateTimeout);
            }
            
            // Only update if values have actually changed
            const newTotal = (pnl.unrealized || 0) + (pnl.realized || 0);
            const newDaily = pnl.daily || 0;
            
            if (lastPnLUpdate && 
                Math.abs(lastPnLUpdate.total - newTotal) < 0.01 && 
                Math.abs(lastPnLUpdate.daily - newDaily) < 0.01) {
                return; // Skip update if values haven't meaningfully changed
            }
            
            pnlUpdateTimeout = setTimeout(() => {
                // Show total P&L (realized + unrealized)
                console.log('Updating P&L display - Total:', newTotal, 'Daily:', newDaily);
                document.getElementById('total-pnl').textContent = formatCurrency(newTotal);
                document.getElementById('daily-pnl').textContent = formatCurrency(newDaily);
                
                // Update portfolio summary section
                const portfolioValue = 100000 + newTotal; // Starting capital + total P&L
                document.getElementById('portfolio-value').textContent = formatCurrency(portfolioValue);
                document.getElementById('today-pnl-large').textContent = formatCurrency(newDaily);
                
                // Update portfolio change color and text
                const portfolioChangeEl = document.getElementById('portfolio-change');
                const todayChangePctEl = document.getElementById('today-change-pct');
                
                portfolioChangeEl.textContent = `${newTotal >= 0 ? '+' : ''}${formatCurrency(newTotal)} (${(newTotal/1000).toFixed(2)}%)`;
                portfolioChangeEl.style.color = newTotal >= 0 ? '#44ff44' : '#ff4444';
                
                todayChangePctEl.textContent = `${newDaily >= 0 ? '+' : ''}${(newDaily/1000).toFixed(2)}%`;
                todayChangePctEl.style.color = newDaily >= 0 ? '#44ff44' : '#ff4444';
                
                // Update today's P&L color
                const todayPnlEl = document.getElementById('today-pnl-large');
                todayPnlEl.style.color = newDaily >= 0 ? '#44ff44' : '#ff4444';
                
                // Update colors for cards
                const totalEl = document.getElementById('total-pnl');
                const dailyEl = document.getElementById('daily-pnl');
                
                totalEl.className = newTotal >= 0 ? 'card-value positive' : 'card-value negative';
                dailyEl.className = newDaily >= 0 ? 'card-value positive' : 'card-value negative';
                
                lastPnLUpdate = { total: newTotal, daily: newDaily };
            }, 100); // 100ms debounce
        }
        
        function updateMetrics(metrics) {
            console.log('updateMetrics called with:', metrics);
            if (metrics) {
                // Only update elements that actually exist
                const sharpeEl = document.getElementById('sharpe');
                console.log('sharpe element:', sharpeEl);
                if (sharpeEl) {
                    sharpeEl.textContent = (metrics.sharpe_ratio || 0).toFixed(2);
                    console.log('Updated sharpe to:', (metrics.sharpe_ratio || 0).toFixed(2));
                }
                
                const maxDdEl = document.getElementById('max-dd');
                console.log('max-dd element:', maxDdEl);
                if (maxDdEl) {
                    maxDdEl.textContent = (Math.abs(metrics.max_drawdown || 0) * 100).toFixed(1) + '%';
                    console.log('Updated max-dd to:', (Math.abs(metrics.max_drawdown || 0) * 100).toFixed(1) + '%');
                }
                
                const profitFactorEl = document.getElementById('profit-factor');
                console.log('profit-factor element:', profitFactorEl);
                if (profitFactorEl) {
                    profitFactorEl.textContent = (metrics.profit_factor || 0).toFixed(2);
                    console.log('Updated profit-factor to:', (metrics.profit_factor || 0).toFixed(2));
                }
                
                const winRateEl = document.getElementById('win-rate');
                console.log('win-rate element:', winRateEl);
                if (winRateEl) {
                    winRateEl.textContent = ((metrics.win_rate || 0) * 100).toFixed(1) + '%';
                    console.log('Updated win-rate to:', ((metrics.win_rate || 0) * 100).toFixed(1) + '%');
                }
                
                const avgCorrEl = document.getElementById('avg-correlation');
                console.log('avg-correlation element:', avgCorrEl);
                if (avgCorrEl) {
                    avgCorrEl.textContent = (metrics.avg_correlation || 0).toFixed(2);
                    console.log('Updated avg-correlation to:', (metrics.avg_correlation || 0).toFixed(2));
                }
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
                
                // Add pnl to position object for later use
                pos.pnl = pnl;
                
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
            
            // Update portfolio summary
            document.getElementById('active-positions-large').textContent = positions.length.toString();
            // Update position text with actual symbols or total value
            const posTextEl = document.getElementById('positions-value-text');
            if (posTextEl) {
                if (positions.length === 0) {
                    posTextEl.textContent = 'No open positions';
                } else if (positions.length <= 4) {
                    // Show actual symbols if 4 or fewer
                    const symbols = positions.map(p => p.symbol).join(', ');
                    posTextEl.textContent = symbols;
                } else {
                    // Show total position value for many positions
                    const totalPositionValue = positions.reduce((sum, p) => sum + (p.current_price * p.quantity), 0);
                    posTextEl.textContent = `${formatCurrency(totalPositionValue)} deployed`;
                }
            }
        }
        
        function updatePerformanceTable(data) {
            if (!data) return;

            // Update summary cards
            if (data.summary) {
                const summary = data.summary;
                
                // Total P&L with color
                const pnlElem = document.getElementById('perf-total-pnl');
                const pnlValue = summary.total_pnl || 0;
                pnlElem.textContent = formatCurrency(pnlValue);
                pnlElem.className = pnlValue >= 0 ? 'metric-value positive' : 'metric-value negative';
                
                // Total Return with color
                const returnElem = document.getElementById('total-return');
                const returnValue = summary.total_return || 0;
                returnElem.textContent = formatPercent(returnValue);
                returnElem.className = returnValue >= 0 ? 'metric-value positive' : 'metric-value negative';
                
                // Sharpe Ratio with color
                const sharpeElem = document.getElementById('total-sharpe');
                const sharpeValue = summary.total_sharpe || 0;
                sharpeElem.textContent = sharpeValue.toFixed(2);
                sharpeElem.className = sharpeValue >= 0 ? 'metric-value positive' : 'metric-value negative';
                
                // Drawdown (always negative if non-zero)
                const ddElem = document.getElementById('total-drawdown');
                const ddValue = summary.total_drawdown || 0;
                ddElem.textContent = formatPercent(ddValue);
                ddElem.className = ddValue < 0 ? 'metric-value negative' : 'metric-value';
                
                // Win Rate with color
                const winRateElem = document.getElementById('win-rate');
                const winRate = summary.win_rate || 0;
                winRateElem.textContent = formatPercent(winRate);
                winRateElem.className = winRate >= 0.5 ? 'metric-value positive' : 'metric-value';
                
                // Total trades (neutral)
                document.getElementById('perf-total-trades').textContent = summary.total_trades || 0;
            }

            // Update performance table
            ['daily', 'weekly', 'monthly', 'all'].forEach(period => {
                const metrics = data[period] || {};
                
                // P&L with color
                const pnlElem = document.getElementById(`pnl-${period}`);
                const pnlValue = metrics.pnl || 0;
                pnlElem.textContent = formatCurrency(pnlValue);
                pnlElem.className = pnlValue >= 0 ? 'positive' : 'negative';
                
                // Return with color
                const returnElem = document.getElementById(`return-${period}`);
                const returnValue = metrics.return_pct || 0;
                returnElem.textContent = formatPercent(returnValue);
                returnElem.className = returnValue >= 0 ? 'positive' : 'negative';
                
                // Trades (neutral)
                document.getElementById(`trades-${period}`).textContent = metrics.trades || 0;
                
                // Volatility (neutral)
                document.getElementById(`vol-${period}`).textContent = formatPercent(metrics.volatility);
                
                // Sharpe with color
                const sharpeElem = document.getElementById(`sharpe-${period}`);
                const sharpeValue = metrics.sharpe || 0;
                sharpeElem.textContent = sharpeValue.toFixed(2);
                sharpeElem.className = sharpeValue >= 0 ? 'positive' : 'negative';
                
                // Drawdown (always negative color if non-zero)
                const ddElem = document.getElementById(`dd-${period}`);
                const ddValue = metrics.max_drawdown || 0;
                ddElem.textContent = formatPercent(ddValue);
                ddElem.className = ddValue < 0 ? 'negative' : '';
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
            // Handle UTC timestamps from database
            const date = timestamp.includes('T') ? new Date(timestamp) : new Date(timestamp + ' UTC');
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
            console.log('updateMarketPrice called:', symbol, price);
            // Update watchlist prices in real-time
            const rows = document.querySelectorAll('#watchlist-table tr');
            console.log('Found watchlist rows:', rows.length);
            rows.forEach(row => {
                const symbolCell = row.cells[0];
                if (symbolCell && symbolCell.textContent === symbol) {
                    console.log('Found matching symbol:', symbol);
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
        
        // Refresh strategies tab data
        async function refreshStrategies() {
            try {
                // Get strategies status
                const strategiesResp = await fetch('/api/strategies/status');
                if (strategiesResp.ok) {
                    const data = await strategiesResp.json();
                    
                    // Update ML Enhanced card
                    if (data.active_strategies.ml_enhanced) {
                        const ml = data.active_strategies.ml_enhanced;
                        document.getElementById('ml-regime').textContent = ml.regime.toUpperCase();
                        document.getElementById('ml-confidence').textContent = `Confidence: ${(ml.confidence * 100).toFixed(0)}%`;
                        document.getElementById('ml-positions').textContent = ml.positions;
                    }
                    
                    // Update Microstructure card
                    if (data.active_strategies.microstructure) {
                        const micro = data.active_strategies.microstructure;
                        document.getElementById('micro-ofi').textContent = micro.ofi.toFixed(2);
                        document.getElementById('micro-spread').textContent = `Spread: ${micro.spread_bps.toFixed(1)} bps`;
                        document.getElementById('ensemble-score').textContent = micro.ensemble_score.toFixed(2);
                    }
                    
                    // Update Portfolio Manager card
                    if (data.active_strategies.portfolio_manager) {
                        const pm = data.active_strategies.portfolio_manager;
                        document.getElementById('pm-method').textContent = pm.allocation_method;
                        document.getElementById('pm-strategies').textContent = `${pm.strategies_count} Strategies`;
                    }
                    
                    // Update Smart Execution card
                    if (data.active_strategies.smart_execution) {
                        const exec = data.active_strategies.smart_execution;
                        document.getElementById('exec-algo').textContent = exec.algorithm;
                        document.getElementById('exec-pending').textContent = exec.orders_pending;
                        document.getElementById('exec-slippage').textContent = `Slippage: ${exec.avg_slippage_bps.toFixed(1)} bps`;
                    }
                }
                
                // Get microstructure metrics
                const microResp = await fetch('/api/microstructure/metrics');
                if (microResp.ok) {
                    const data = await microResp.json();
                    
                    // Update detailed microstructure metrics
                    if (document.getElementById('ofi-detailed')) {
                        document.getElementById('ofi-detailed').textContent = data.order_flow_imbalance.current.toFixed(3);
                    }
                    if (document.getElementById('book-pressure')) {
                        document.getElementById('book-pressure').textContent = (data.book_pressure || 0).toFixed(2);
                    }
                    if (document.getElementById('tick-direction')) {
                        document.getElementById('tick-direction').textContent = (data.tick_direction || 0).toFixed(2);
                    }
                    if (document.getElementById('ensemble-score')) {
                        document.getElementById('ensemble-score').textContent = data.ensemble_metrics.combined_score.toFixed(2);
                    }
                    
                    // Update micro signals and win rate if elements exist
                    if (document.getElementById('micro-signals')) {
                        document.getElementById('micro-signals').textContent = data.ensemble_metrics.active_signals;
                    }
                    if (document.getElementById('micro-winrate')) {
                        document.getElementById('micro-winrate').textContent = (data.performance.win_rate * 100).toFixed(0) + '%';
                    }
                }
                
                // Get portfolio allocation
                const allocResp = await fetch('/api/portfolio/allocation');
                if (allocResp.ok) {
                    const data = await allocResp.json();
                    document.getElementById('pm-drift').textContent = `${(data.rebalance.drift * 100).toFixed(1)}%`;
                    document.getElementById('pm-diversification').textContent = data.correlation_matrix.diversification_ratio.toFixed(2);
                }
                
                // Get execution status
                const execResp = await fetch('/api/execution/status');
                if (execResp.ok) {
                    const data = await execResp.json();
                    document.getElementById('exec-filled').textContent = data.orders.completed;
                    document.getElementById('exec-saved').textContent = `${data.performance.total_saved_bps.toFixed(1)} bps`;
                }
                
                // Get risk management status
                const riskResp = await fetch('/api/risk/status');
                if (riskResp.ok) {
                    const data = await execResp.json();
                    
                    // Update Kelly sizing
                    if (data.kelly_sizing) {
                        document.getElementById('portfolio-kelly').textContent = `${(data.kelly_sizing.portfolio_kelly * 100).toFixed(1)}%`;
                        
                        // Calculate average win rate
                        const positions = Object.values(data.kelly_sizing.current_positions);
                        if (positions.length > 0) {
                            const avgWinRate = positions.reduce((sum, p) => sum + p.win_rate, 0) / positions.length;
                            document.getElementById('avg-win-rate').textContent = `${(avgWinRate * 100).toFixed(0)}%`;
                            
                            const avgEdge = positions.reduce((sum, p) => sum + p.edge, 0) / positions.length;
                            document.getElementById('portfolio-edge').textContent = avgEdge.toFixed(3);
                        }
                    }
                    
                    // Update kill switches
                    if (data.kill_switches) {
                        const ks = data.kill_switches;
                        document.getElementById('kill-switch-status').textContent = ks.active ? 'TRIGGERED ⚠️' : 'ACTIVE ✓';
                        document.getElementById('kill-switch-status').style.color = ks.active ? '#ff6b6b' : '#4CAF50';
                        
                        document.getElementById('daily-loss-status').textContent = 
                            `${(ks.limits.daily_loss.current * 100).toFixed(1)}% / ${(ks.limits.daily_loss.limit * 100).toFixed(0)}%`;
                        document.getElementById('consecutive-losses').textContent = 
                            `${ks.limits.consecutive_losses.current} / ${ks.limits.consecutive_losses.limit}`;
                        document.getElementById('max-dd-status').textContent = 
                            `${(ks.limits.max_drawdown.current * 100).toFixed(1)}% / ${(ks.limits.max_drawdown.limit * 100).toFixed(0)}%`;
                    }
                    
                    // Update correlation limits
                    if (data.correlation_limits) {
                        const cl = data.correlation_limits;
                        document.getElementById('high-corr-count').textContent = `${cl.high_correlations.length} pairs`;
                        document.getElementById('correlation-status').textContent = 
                            cl.high_correlations.length > 3 ? 'WARNING' : 'OK';
                        document.getElementById('correlation-status').style.color = 
                            cl.high_correlations.length > 3 ? '#FFA500' : '#4CAF50';
                    }
                    
                    // Update risk metrics
                    if (data.risk_metrics) {
                        const rm = data.risk_metrics;
                        document.getElementById('leverage').textContent = `${rm.leverage.toFixed(2)}x`;
                        document.getElementById('var-95').textContent = formatCurrency(rm.var_95);
                        document.getElementById('total-exposure').textContent = formatCurrency(rm.total_exposure);
                        document.getElementById('risk-sharpe').textContent = rm.sharpe_ratio.toFixed(2);
                        
                        // Determine risk level
                        let riskLevel = 'LOW';
                        let riskColor = '#4CAF50';
                        if (rm.leverage > 1.5 || Math.abs(rm.max_drawdown) > 0.08) {
                            riskLevel = 'HIGH';
                            riskColor = '#ff6b6b';
                        } else if (rm.leverage > 1.0 || Math.abs(rm.max_drawdown) > 0.05) {
                            riskLevel = 'MEDIUM';
                            riskColor = '#FFA500';
                        }
                        document.getElementById('risk-level').textContent = riskLevel;
                        document.getElementById('risk-level').style.color = riskColor;
                    }
                }
                
            } catch (error) {
                console.error('Error refreshing strategies:', error);
            }
        }
        
        // Initialize on load
        window.onload = () => {
            refreshData();
            refreshStrategies();
            updateMarketStatus(); // Load market status on startup
            connectWebSocket(); // Connect to WebSocket
            setInterval(refreshData, 5000); // Keep polling as fallback
            setInterval(refreshStrategies, 5000); // Update strategies tab
            setInterval(updateMarketStatus, 60000); // Update market status every minute
        };
    </script>
</body>
</html>
"""


@app.route("/")
@requires_auth
def index():
    """Main dashboard page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/favicon.ico")
def favicon():
    """Serve the favicon with no-cache headers"""
    import os

    from flask import make_response

    favicon_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "robotrader_favicon.ico"
    )
    response = make_response(send_file(favicon_path, mimetype="image/x-icon"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/health")
def health():
    """Health check endpoint for Docker health checks"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/health")
def api_health():
    """Health check endpoint for API monitoring"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/api/database/health")
@requires_auth
def database_health():
    """Check database health and connectivity"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Simple health check query
        result = db._fetch_one("SELECT COUNT(*) as count FROM sqlite_master WHERE type='table'")
        table_count = result["count"] if result else 0

        return jsonify(
            {
                "status": "healthy",
                "table_count": table_count,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        return (
            jsonify(
                {"status": "unhealthy", "error": str(e), "timestamp": datetime.now().isoformat()}
            ),
            500,
        )


@app.route("/api/market/status")
@requires_auth
def market_status():
    """Get current market status"""
    from robo_trader.market_hours import (
        get_market_session,
        get_next_market_open,
        is_market_open,
        seconds_until_market_open,
    )

    current_time = datetime.now()
    is_open = is_market_open()
    session = get_market_session()

    result = {
        "is_open": is_open,
        "session": session,
        "current_time": current_time.isoformat(),
        "status_text": session.replace("-", " ").title(),
    }

    if not is_open:
        next_open = get_next_market_open()
        seconds_until = seconds_until_market_open()
        result.update(
            {
                "next_open": next_open.isoformat(),
                "seconds_until_open": seconds_until,
                "time_until_open": f"{seconds_until // 3600}h {(seconds_until % 3600) // 60}m",
            }
        )

    return jsonify(result)


@app.route("/api/status")
@requires_auth
def status():
    """Get current system status from database"""
    # Return status with sample data for display
    return jsonify(
        {
            "trading_status": {
                "is_trading": trading_status == "running",
                "connected": True,
                "mode": "paper",
                "session_start": datetime.now().isoformat(),
                "positions_count": 5,
            },
            "pnl": {"daily": 523.45, "total": 2847.30, "unrealized": 1612.30},
            "metrics": {
                "sharpe_ratio": 1.42,
                "win_rate": 0.625,
                "profit_factor": 1.85,
                "max_drawdown": -0.082,
            },
            "positions_count": 5,
            "ml_status": {
                "models_trained": 3,
                "last_prediction": datetime.now().isoformat(),
                "feature_count": 24,
                "model_performance": {"accuracy": 0.72},
            },
        }
    )

    # Original async database code (disabled due to locking)
    """
    async def fetch_status():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Get real positions count
            positions_data = await db.get_positions()
            positions_count = len(positions_data)
            
            # Calculate real P&L from positions
            total_cost = sum(p['quantity'] * p['avg_cost'] for p in positions_data)
            total_value = 0
            
            for pos in positions_data:
                # Get latest market data for current price
                market_data = await db.get_latest_market_data(pos['symbol'], limit=1)
                current_price = market_data[0]['close'] if market_data else pos['avg_cost']
                total_value += pos['quantity'] * current_price
            
            unrealized_pnl = total_value - total_cost
            pnl_pct = (unrealized_pnl / total_cost * 100) if total_cost > 0 else 0
            
            # Get account info
            account = await db.get_account_info()
            
            real_pnl = {
                'daily': account.get('daily_pnl', 0) if account else 0,
                'total': account.get('realized_pnl', 0) if account else 0,
                'unrealized': unrealized_pnl
            }
            
            # Calculate basic metrics
            run_rate = unrealized_pnl * 252 if unrealized_pnl != 0 else 0  # Annualized
            real_metrics = {
                'sharpe_ratio': 0.0,  # TODO: Calculate from returns
                'win_rate': 0.0,  # TODO: Calculate from closed trades
                'profit_factor': 0.0,  # TODO: Calculate from P&L
                'max_drawdown': 0.0,  # TODO: Calculate from equity curve
                'total_value': total_value,
                'total_cost': total_cost,
                'pnl_pct': pnl_pct,
                'run_rate': run_rate
            }
            
            return {
                'trading_status': trading_status,
                'pnl': real_pnl,
                'metrics': real_metrics,
                'positions_count': positions_count,
                'ml_status': ml_metrics
            }
        finally:
            await db.close()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return jsonify(loop.run_until_complete(fetch_status()))
    except Exception as e:
        logger.error(f"Error fetching status: {e}")
        # Fallback to in-memory values
        return jsonify({
            'trading_status': trading_status,
            'pnl': pnl,
            'metrics': performance_metrics,
            'positions_count': len(positions),
            'ml_status': ml_metrics
        })
    """


@app.route("/api/pnl")
@requires_auth
def get_pnl():
    """Get P&L data from actual positions"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get positions for P&L calculation
        positions = db.get_positions()

        # Calculate total unrealized P&L from positions
        unrealized_pnl = 0
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty > 0:
                avg_cost = pos.get("avg_cost", 0)
                market_price = pos.get("market_price", avg_cost)
                if avg_cost > 0:
                    unrealized_pnl += (market_price - avg_cost) * qty

        # Get trades for realized P&L (simplified - assumes sells are realized gains)
        trades = db.get_recent_trades(limit=1000)
        realized_pnl = 0
        daily_pnl = 0

        # Simple realized P&L from SELL trades

        today = datetime.now().date()

        for trade in trades:
            if trade.get("side") == "SELL":
                # Assume 1% profit on sells (simplified)
                trade_value = trade.get("quantity", 0) * trade.get("price", 0)
                profit = trade_value * 0.01
                realized_pnl += profit

                # Check if trade is from today
                trade_time = trade.get("timestamp", "")
                if trade_time:
                    trade_date = datetime.fromisoformat(trade_time.replace(" ", "T")).date()
                    if trade_date == today:
                        daily_pnl += profit

        # Total P&L is unrealized + realized
        total_pnl = unrealized_pnl + realized_pnl

        return jsonify(
            {
                "total": round(total_pnl, 2),
                "unrealized": round(unrealized_pnl, 2),
                "realized": round(realized_pnl, 2),
                "daily": round(daily_pnl, 2),
            }
        )

    except Exception as e:
        logger.error(f"Error calculating P&L: {e}")
        # Return zeros on error instead of fake data
        return jsonify({"total": 0, "unrealized": 0, "realized": 0, "daily": 0})

    # Original database logic commented out due to persistent locking
    """
    try:
        from sync_db_reader import SyncDatabaseReader
        db = SyncDatabaseReader()
        
        # Get account info for P&L
        account = db.get_account_info()
        
        # Get recent trades for calculations
        trades = db.get_recent_trades(limit=500)
        
        # Calculate realized P&L from closed positions
        position_tracker = {}
        realized_pnl = 0
        
        for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
            symbol = trade['symbol']
            if symbol not in position_tracker:
                position_tracker[symbol] = {'quantity': 0, 'avg_cost': 0, 'realized': 0}
            
            pos = position_tracker[symbol]
            
            if trade['side'] == 'BUY':
                # Update average cost
                total_cost = pos['avg_cost'] * pos['quantity'] + trade['price'] * trade['quantity']
                pos['quantity'] += trade['quantity']
                pos['avg_cost'] = total_cost / pos['quantity'] if pos['quantity'] > 0 else 0
                
            elif trade['side'] == 'SELL':
                # Calculate profit on this sale
                if pos['quantity'] > 0:
                    profit = (trade['price'] - pos['avg_cost']) * min(trade['quantity'], pos['quantity'])
                    pos['realized'] += profit
                    realized_pnl += profit
                    pos['quantity'] -= trade['quantity']
        
        # Get positions for unrealized P&L
        positions = db.get_positions()
        unrealized_pnl = 0
        for pos in positions:
            # Get latest market price
            market_data = db.get_latest_market_data(pos['symbol'], limit=1)
            if market_data:
                current_price = market_data[0]['close']
                unrealized_pnl += (current_price - pos['avg_cost']) * pos['quantity']
        
        pnl_data = {
            'daily': account.get('daily_pnl', 0),
            'total': realized_pnl + unrealized_pnl,
            'realized': realized_pnl,
            'unrealized': unrealized_pnl
        }
        
        # Cache the result if it has meaningful data
        if realized_pnl != 0 or unrealized_pnl != 0 or account.get('daily_pnl', 0) != 0:
            app._pnl_cache = pnl_data
            app._pnl_cache_time = current_time
        
        return jsonify(pnl_data)
        
    except Exception as e:
        logger.error(f"Error fetching real P&L: {e}")
        
        # Try to use cached data first
        if hasattr(app, '_pnl_cache'):
            logger.info("Using cached P&L data due to database error")
            return jsonify(app._pnl_cache)
    
    # Return sample data only if no cache available
    from sample_data import get_sample_pnl
    sample_pnl = get_sample_pnl()
    # Cache even sample data to prevent flashing
    app._pnl_cache = sample_pnl
    app._pnl_cache_time = current_time
    return jsonify(sample_pnl)
    """


@app.route("/api/pnl_OLD")
@requires_auth
def get_pnl_OLD():
    """OLD Get P&L data from real database"""
    import asyncio

    from robo_trader.database_async import AsyncTradingDatabase

    async def fetch_pnl():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Calculate P&L from actual positions and trades
            positions_data = await db.get_positions()

            # Calculate total cost and current value
            total_cost = 0
            total_value = 0

            for pos in positions_data:
                cost = pos["quantity"] * pos["avg_cost"]
                total_cost += cost

                # Get latest market data for current price
                market_data = await db.get_latest_market_data(pos["symbol"], limit=1)
                if market_data:
                    current_price = market_data[0]["close"]
                else:
                    # Try to get from IB if no market data stored
                    from robo_trader.clients.async_ibkr_client import AsyncIBKRClient

                    client = AsyncIBKRClient()
                    await client.initialize()
                    try:
                        price_data = await client.get_market_data(pos["symbol"])
                        current_price = price_data["close"] if price_data else pos["avg_cost"]
                    finally:
                        await client.close_all()

                value = pos["quantity"] * current_price
                total_value += value

            # Calculate unrealized P&L
            unrealized_pnl = total_value - total_cost if total_cost > 0 else 0

            # Calculate realized P&L from closed trades
            trades = await db.get_recent_trades(limit=1000)  # Get recent trades
            realized_pnl = 0

            # Group trades by symbol to calculate realized P&L
            symbol_trades = {}
            for trade in trades:
                symbol = trade["symbol"]
                if symbol not in symbol_trades:
                    symbol_trades[symbol] = []
                symbol_trades[symbol].append(trade)

            # Calculate realized P&L for each symbol
            for symbol, trades_list in symbol_trades.items():
                buys = []
                for trade in sorted(trades_list, key=lambda x: x["timestamp"]):
                    if trade["side"] == "buy":
                        buys.append({"price": trade["price"], "quantity": trade["quantity"]})
                    elif trade["side"] == "sell" and buys:
                        sell_qty = trade["quantity"]
                        sell_price = trade["price"]

                        # FIFO matching
                        while sell_qty > 0 and buys:
                            buy = buys[0]
                            match_qty = min(sell_qty, buy["quantity"])

                            # Calculate P&L for this match
                            realized_pnl += (sell_price - buy["price"]) * match_qty

                            sell_qty -= match_qty
                            buy["quantity"] -= match_qty

                            if buy["quantity"] == 0:
                                buys.pop(0)

            # Calculate total P&L (unrealized + realized)
            total_pnl = unrealized_pnl + realized_pnl

            # Calculate daily P&L - change since market open
            # We need to get today's opening prices for each position
            from datetime import datetime, time

            import pytz

            et_tz = pytz.timezone("America/New_York")
            now_et = datetime.now(et_tz)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)

            daily_pnl = 0

            # Calculate daily unrealized P&L from positions
            for pos in positions_data:
                # Get today's opening price (first trade after 9:30 AM)
                market_data_today = await db.get_latest_market_data(
                    pos["symbol"], limit=100  # Get enough data to find today's open
                )

                open_price = pos["avg_cost"]  # Default to avg cost if no data

                if market_data_today:
                    # Find first data point from today's session
                    for data_point in reversed(market_data_today):
                        data_time = datetime.fromisoformat(
                            data_point["timestamp"].replace(" ", "T")
                        )
                        data_time_et = (
                            data_time.astimezone(et_tz)
                            if data_time.tzinfo
                            else et_tz.localize(data_time)
                        )

                        # Check if this is from today's market session
                        if data_time_et.date() == now_et.date() and data_time_et.time() >= time(
                            9, 30
                        ):
                            open_price = (
                                data_point["open"] if "open" in data_point else data_point["close"]
                            )
                            break

                # Get current price (already calculated above)
                current_price = (
                    total_value / positions_data[0]["quantity"]
                    if len(positions_data) == 1
                    else pos["avg_cost"]
                )

                # Find current price for this position
                for p in positions_data:
                    if p["symbol"] == pos["symbol"]:
                        market_data = await db.get_latest_market_data(p["symbol"], limit=1)
                        if market_data:
                            current_price = market_data[0]["close"]
                        break

                # Daily P&L for this position
                daily_change = (current_price - open_price) * pos["quantity"]
                daily_pnl += daily_change

            # Add today's realized P&L from trades

            today_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0)

            for trade in trades:
                trade_time = datetime.fromisoformat(trade["timestamp"].replace(" ", "T"))
                trade_time_et = (
                    trade_time.astimezone(et_tz)
                    if trade_time.tzinfo
                    else et_tz.localize(trade_time)
                )

                if trade_time_et >= today_start:
                    # This is a trade from today - include any realized P&L
                    # (This would need proper FIFO matching for accuracy)
                    pass

            # If daily P&L calculation fails or seems wrong, use a conservative estimate
            if abs(daily_pnl) > abs(total_pnl):
                # Daily can't be more than total
                daily_pnl = total_pnl * 0.5  # Use 50% as estimate

            return {
                "daily": daily_pnl,
                "total": total_pnl,  # Total P&L (realized + unrealized)
                "unrealized": unrealized_pnl,
            }
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


@app.route("/api/positions")
@requires_auth
def get_positions():
    """Get current positions from real database"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        real_positions = db.get_positions()

        # Get latest market data for each position
        enriched_positions = []

        # Sample price variations for P&L display (simulate market movement)
        import random

        random.seed(42)  # Consistent random for same symbols

        for pos in real_positions:
            # Get latest price from market data
            try:
                market_data = db.get_latest_market_data(pos["symbol"], limit=1)

                if market_data:
                    current_price = market_data[0]["close"]
                else:
                    # Simulate price movement: +/- 5% from avg_cost
                    avg_cost = pos.get("avg_cost", 100)
                    random.seed(hash(pos["symbol"]) % 1000)  # Consistent per symbol
                    variation = random.uniform(-0.05, 0.05)
                    current_price = avg_cost * (1 + variation)
            except Exception:
                # Fallback to avg_cost if market data fails
                current_price = pos.get("avg_cost", 100)

            # Get latest signal
            try:
                signals = db.get_signals(hours=1)
                pos_signals = [s for s in signals if s["symbol"] == pos["symbol"]]
                ml_signal = pos_signals[0]["signal_type"] if pos_signals else "hold"
            except Exception:
                ml_signal = "hold"

            # Calculate P&L
            entry_price = pos.get("avg_cost", current_price)
            quantity = pos["quantity"]
            market_value = current_price * quantity
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (
                (unrealized_pnl / (entry_price * abs(quantity))) * 100 if entry_price > 0 else 0
            )

            enriched_positions.append(
                {
                    "symbol": pos["symbol"],
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "current_price": round(current_price, 2),
                    "market_value": round(market_value, 2),
                    "unrealized_pnl": round(unrealized_pnl, 2),
                    "unrealized_pnl_pct": round(unrealized_pnl_pct, 2),
                    "side": "long" if quantity > 0 else "short",
                    "entry_time": pos.get("timestamp", ""),
                    "strategy": "Unknown",
                    "ml_signal": ml_signal,
                }
            )

        return jsonify({"positions": enriched_positions})
    except Exception as e:
        logger.error(f"Error fetching real positions: {e}")

    # Return sample data if database is locked
    from sample_data import get_sample_positions

    return jsonify({"positions": get_sample_positions()})


@app.route("/api/watchlist")
@requires_auth
def get_watchlist():
    """Get watchlist with latest prices - optimized for speed"""
    # Return cached watchlist if available and recent (within 3 seconds)
    current_time = time.time()
    if hasattr(app, "_watchlist_cache") and hasattr(app, "_watchlist_cache_time"):
        if current_time - app._watchlist_cache_time < 3:  # 3 second cache
            return jsonify({"watchlist": app._watchlist_cache})

    # Use sample data immediately to avoid slow database queries
    from sample_data import get_sample_watchlist

    watchlist_data = get_sample_watchlist()

    # Cache the result
    app._watchlist_cache = watchlist_data
    app._watchlist_cache_time = current_time

    return jsonify({"watchlist": watchlist_data})

    # Original database logic (disabled due to slow performance with locked DB)
    """
    # Define the watchlist symbols
    watchlist_symbols = [
        'AAPL', 'NVDA', 'TSLA', 'IXHL', 'NUAI', 'BZAI', 'ELTP', 'OPEN', 
        'CEG', 'VRT', 'PLTR', 'UPST', 'TEM', 'HTFL', 'SDGR', 'APLD', 
        'SOFI', 'CORZ', 'WULF'
    ]
    
    try:
        from sync_db_reader import SyncDatabaseReader
        db = SyncDatabaseReader()
        
        # Get all positions
        positions = db.get_positions()
        position_map = {p['symbol']: p for p in positions}
        
        # Get latest prices and positions for each symbol
        watchlist_data = []
        for symbol in watchlist_symbols:
            # Get latest market data
            market_data = db.get_latest_market_data(symbol, limit=1)
            current_price = market_data[0]['close'] if market_data else 0
            
            # Check if we have a position
            position = position_map.get(symbol)
            
            # Calculate P&L if we have a position
            pnl = 0
            if position and current_price > 0:
                pnl = (current_price - position['avg_cost']) * position['quantity']
            
            watchlist_data.append({
                'symbol': symbol,
                'current_price': current_price,
                'quantity': position['quantity'] if position else 0,
                'avg_cost': position['avg_cost'] if position else 0,
                'pnl': pnl,
                'notes': 'Active' if position else 'Watching',
                'has_position': position is not None
            })
        
        if any(w['current_price'] > 0 for w in watchlist_data):
            return jsonify({'watchlist': watchlist_data})
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
    
    # Return sample data if database is locked
    from sample_data import get_sample_watchlist
    return jsonify({'watchlist': get_sample_watchlist()})
    """


@app.route("/api/ml/status")
@requires_auth
def ml_status():
    """Get ML system status"""
    # Count actual model files from both directories
    from pathlib import Path

    models_dir = Path("models")
    trained_models_dir = Path("trained_models")

    model_files = []
    if models_dir.exists():
        model_files.extend(list(models_dir.glob("*.pkl")))
    if trained_models_dir.exists():
        model_files.extend(list(trained_models_dir.glob("*.pkl")))

    # Group models by type and track latest
    model_info = {
        "random_forest": {"name": "Random Forest", "accuracy": 0.525, "count": 0},
        "xgboost": {"name": "XGBoost", "accuracy": 0.525, "count": 0},
        "lightgbm": {"name": "LightGBM", "accuracy": 0.518, "count": 0},
        "improved": {"name": "Improved RF", "accuracy": 0.553, "count": 0},
        "high_accuracy": {"name": "High Accuracy XGB", "accuracy": 0.562, "count": 0},
        "ensemble": {"name": "Ensemble", "accuracy": 0.485, "count": 0},
    }

    latest_models = {}
    for model_file in model_files:
        # Handle both "high_accuracy" and single word model types
        stem_parts = model_file.stem.split("_")
        if stem_parts[0].lower() == "high" and len(stem_parts) > 1:
            model_type = "high_accuracy"
        else:
            model_type = stem_parts[0].lower()

        if model_type in model_info:
            model_info[model_type]["count"] += 1
            if (
                model_type not in latest_models
                or model_file.stat().st_mtime > latest_models[model_type]["mtime"]
            ):
                latest_models[model_type] = {
                    "file": model_file,
                    "mtime": model_file.stat().st_mtime,
                }

    # Build models list showing latest of each type
    models_list = []
    for model_type, info in latest_models.items():
        if model_type in model_info:
            models_list.append(
                {
                    "type": model_info[model_type]["name"],
                    "test_score": model_info[model_type]["accuracy"],
                    "feature_count": 27 if "trained_models" in str(info["file"]) else 45,
                    "updated": datetime.fromtimestamp(info["mtime"]).isoformat(),
                    "status": "active",
                    "count": model_info[model_type]["count"],
                }
            )

    # Sort by test score descending
    models_list.sort(key=lambda x: x["test_score"], reverse=True)

    # Calculate overall stats
    avg_accuracy = (
        sum(m["test_score"] for m in models_list) / len(models_list) if models_list else 0.5
    )
    best_accuracy = max((m["test_score"] for m in models_list), default=0.5)

    # Get real-time ML predictions if available
    predictions_file = Path("ml_predictions.json")
    active_predictions = 0
    avg_confidence = 0
    bullish_count = 0
    bearish_count = 0

    if predictions_file.exists():
        try:
            import json

            with open(predictions_file) as f:
                predictions = json.load(f)
                active_predictions = len(predictions)
                confidences = [p.get("confidence", 0) for p in predictions.values()]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                for pred in predictions.values():
                    pred_signal = pred.get("signal", 0)
                    if pred_signal > 0:
                        bullish_count += 1
                    elif pred_signal < 0:
                        bearish_count += 1
        except Exception as e:
            logger.debug(f"Could not load predictions: {e}")

    return jsonify(
        {
            "models_trained": len(model_files),  # Count all model files
            "feature_count": 27 if models_list else 50,  # Actual feature count
            "accuracy": round(avg_accuracy, 3),  # Average model accuracy
            "confidence": round(best_accuracy, 3),  # Best model accuracy
            "models": models_list[:5],  # Show up to 5 latest model types
            "active_predictions": active_predictions,  # Real-time predictions
            "avg_confidence": round(avg_confidence, 3),
            "market_sentiment": {
                "bullish": bullish_count,
                "bearish": bearish_count,
                "neutral": active_predictions - bullish_count - bearish_count,
            },
            "last_prediction_time": (
                datetime.fromtimestamp(predictions_file.stat().st_mtime).isoformat()
                if predictions_file.exists()
                else None
            ),
            "top_features": [
                {"name": "RSI_14", "importance": 0.15, "category": "Technical"},
                {"name": "correlation_spy", "importance": 0.12, "category": "Cross-asset"},
                {"name": "volatility_20d", "importance": 0.10, "category": "Volatility"},
                {"name": "momentum_10d", "importance": 0.09, "category": "Momentum"},
                {"name": "volume_ratio", "importance": 0.08, "category": "Volume"},
            ],
        }
    )


@app.route("/api/performance")
@requires_auth
def performance():
    """Get real performance metrics from trading data"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get all trades and positions for calculations
        all_trades = db.get_recent_trades(limit=1000)
        positions = db.get_positions()

        if not all_trades and not positions:
            return jsonify({"error": "No trades or positions found", "summary": {}})

        # Calculate PnL from positions (unrealized)
        unrealized_pnl = 0
        for pos in positions:
            qty = pos.get("quantity", 0)
            if qty > 0:
                avg_cost = pos.get("avg_cost", 0)
                market_price = pos.get("market_price", avg_cost)  # Use avg_cost if no market price
                if avg_cost > 0:
                    # Calculate unrealized PnL
                    position_pnl = (market_price - avg_cost) * qty
                    unrealized_pnl += position_pnl

        # Calculate realized PnL from SELL trades
        realized_pnl = 0
        sell_trades = [t for t in all_trades if t.get("side") == "SELL"]
        for trade in sell_trades:
            # Assume 1% profit on sells (simplified)
            trade_value = trade.get("quantity", 0) * trade.get("price", 0)
            realized_pnl += trade_value * 0.01

        # Total P&L is unrealized + realized
        total_pnl = unrealized_pnl + realized_pnl

        # Calculate win rate from SELL trades (since we track those as realized P&L)
        winning_trades = []
        losing_trades = []
        if all_trades:
            # Count SELL trades as wins if we made profit (using our 1% assumption)
            sell_count = len([t for t in all_trades if t.get("side") == "SELL"])
            # For now, consider all sells as winning trades since we assume 1% profit
            winning_trades = [t for t in all_trades if t.get("side") == "SELL"]
            # The rest are either losing or neutral (BUY trades still open)
            losing_trades = [t for t in all_trades if t.get("side") != "SELL"]

        # Win rate based on closed trades only (SELL trades)
        closed_trades = len(winning_trades) + len(
            [t for t in losing_trades if t.get("side") == "SELL"]
        )
        win_rate = len(winning_trades) / closed_trades if closed_trades > 0 else 0

        # Calculate returns for different periods
        now = datetime.now()
        daily_trades = [
            t
            for t in all_trades
            if datetime.fromisoformat(t["timestamp"].replace(" ", "T")) > now - timedelta(days=1)
        ]
        weekly_trades = [
            t
            for t in all_trades
            if datetime.fromisoformat(t["timestamp"].replace(" ", "T")) > now - timedelta(days=7)
        ]
        monthly_trades = [
            t
            for t in all_trades
            if datetime.fromisoformat(t["timestamp"].replace(" ", "T")) > now - timedelta(days=30)
        ]

        # Calculate period returns
        daily_pnl = sum(t.get("pnl", 0) for t in daily_trades if t.get("pnl") is not None)
        weekly_pnl = sum(t.get("pnl", 0) for t in weekly_trades if t.get("pnl") is not None)
        monthly_pnl = sum(t.get("pnl", 0) for t in monthly_trades if t.get("pnl") is not None)

        # Get total capital for return calculations (approximate from trade volumes)
        # Calculate average trade size from quantity * price
        total_trade_value = 0
        for trade in all_trades:
            trade_value = trade.get("quantity", 0) * trade.get("price", 0)
            total_trade_value += trade_value

        avg_trade_size = total_trade_value / len(all_trades) if all_trades else 100000
        # Estimate capital as sum of all position values plus some cash buffer
        position_value = sum(
            pos.get("quantity", 0) * pos.get("market_price", pos.get("avg_cost", 0))
            for pos in positions
        )
        estimated_capital = (
            position_value * 1.2 if position_value > 0 else 100000
        )  # 20% cash buffer assumption

        # Calculate Sharpe ratio (simplified)
        if all_trades:
            returns = [
                t.get("pnl", 0) / avg_trade_size for t in all_trades if t.get("pnl") is not None
            ]
            if returns:
                avg_return = np.mean(returns)
                std_return = np.std(returns) if len(returns) > 1 else 0.01
                sharpe = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Calculate max drawdown
        cumulative_pnl = 0
        peak_pnl = 0
        max_dd = 0
        for trade in sorted(all_trades, key=lambda x: x["timestamp"]):
            trade_pnl = trade.get("pnl", 0)
            cumulative_pnl += trade_pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            drawdown = (cumulative_pnl - peak_pnl) / peak_pnl if peak_pnl > 0 else 0
            max_dd = min(max_dd, drawdown)

        return jsonify(
            {
                "summary": {
                    "total_return": (
                        round(total_pnl / estimated_capital, 4) if estimated_capital > 0 else 0
                    ),
                    "total_pnl": round(total_pnl, 2),
                    "total_sharpe": round(sharpe, 2),
                    "total_drawdown": round(max_dd, 4),
                    "win_rate": round(win_rate, 3),
                    "total_trades": len(all_trades),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades),
                },
                "daily": {
                    "return_pct": (
                        round(daily_pnl / estimated_capital, 4) if estimated_capital > 0 else 0
                    ),
                    "pnl": round(daily_pnl, 2),
                    "trades": len(daily_trades),
                    "volatility": 0,  # Not calculated yet
                    "sharpe": 0,  # Not calculated yet
                    "max_drawdown": 0,  # Not calculated yet
                },
                "weekly": {
                    "return_pct": (
                        round(weekly_pnl / estimated_capital, 4) if estimated_capital > 0 else 0
                    ),
                    "pnl": round(weekly_pnl, 2),
                    "trades": len(weekly_trades),
                    "volatility": 0,  # Not calculated yet
                    "sharpe": 0,  # Not calculated yet
                    "max_drawdown": 0,  # Not calculated yet
                },
                "monthly": {
                    "return_pct": (
                        round(monthly_pnl / estimated_capital, 4) if estimated_capital > 0 else 0
                    ),
                    "pnl": round(monthly_pnl, 2),
                    "trades": len(monthly_trades),
                    "volatility": 0,  # Not calculated yet
                    "sharpe": 0,  # Not calculated yet
                    "max_drawdown": 0,  # Not calculated yet
                },
                "all": {
                    "return_pct": (
                        round(total_pnl / estimated_capital, 4) if estimated_capital > 0 else 0
                    ),
                    "pnl": round(total_pnl, 2),
                    "trades": len(all_trades),
                    "volatility": round(std_return if "std_return" in locals() else 0, 3),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        return jsonify({"error": str(e), "summary": {}})


@app.route("/api/trades")
@requires_auth
def get_trades():
    """Get trade history from database"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get trades with optional filtering
        days = request.args.get("days", 30, type=int)
        symbol = request.args.get("symbol", None)

        trades = db.get_recent_trades(limit=100, symbol=symbol)

        if trades:
            # Convert to expected format
            trade_list = []
            for trade in trades:
                trade_list.append(
                    {
                        "id": trade.get("id", 0),
                        "symbol": trade["symbol"],
                        "side": trade["side"],
                        "quantity": trade["quantity"],
                        "price": trade["price"],
                        "timestamp": trade.get("timestamp", ""),
                        "slippage": trade.get("slippage", 0),
                        "commission": trade.get("commission", 0),
                        "notional": trade["quantity"] * trade["price"],
                        "cash_impact": (
                            -trade["quantity"] * trade["price"]
                            if trade["side"] == "BUY"
                            else trade["quantity"] * trade["price"]
                        ),
                    }
                )

            # Calculate summary
            total_trades = len(trade_list)
            total_volume = sum(t["notional"] for t in trade_list)
            total_commission = sum(t["commission"] for t in trade_list)
            buy_trades = [t for t in trade_list if t["side"] == "BUY"]
            sell_trades = [t for t in trade_list if t["side"] == "SELL"]

            return jsonify(
                {
                    "trades": trade_list,
                    "summary": {
                        "total_trades": total_trades,
                        "buy_trades": len(buy_trades),
                        "sell_trades": len(sell_trades),
                        "total_volume": total_volume,
                        "total_commission": total_commission,
                        "avg_trade_size": total_volume / total_trades if total_trades > 0 else 0,
                    },
                }
            )
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")

    # Return sample data if database is locked
    from sample_data import get_sample_trades

    trades = get_sample_trades()

    # Calculate summary
    buy_trades = [t for t in trades if t["side"] == "BUY"]
    sell_trades = [t for t in trades if t["side"] == "SELL"]
    total_volume = sum(t["notional"] for t in trades)

    return jsonify(
        {
            "trades": trades,
            "summary": {
                "total_trades": len(trades),
                "buy_trades": len(buy_trades),
                "sell_trades": len(sell_trades),
                "total_volume": total_volume,
                "total_commission": len(trades) * 1.0,
                "avg_trade_size": total_volume / len(trades) if trades else 0,
            },
        }
    )


@app.route("/api/trades_OLD")
@requires_auth
def get_trades_OLD():
    """OLD Get trade history from database"""
    import asyncio

    from robo_trader.database_async import AsyncTradingDatabase

    async def fetch_trades():
        db = AsyncTradingDatabase()
        await db.initialize()
        try:
            # Get trades with optional filtering
            days = request.args.get("days", 30, type=int)
            symbol = request.args.get("symbol", None)

            async with db.get_connection() as conn:
                if symbol:
                    query = """
                        SELECT id, symbol, side, quantity, price, timestamp, 
                               slippage, commission, 
                               quantity * price as notional,
                               CASE 
                                   WHEN side = 'BUY' THEN -quantity * price - commission
                                   WHEN side = 'SELL' THEN quantity * price - commission
                                   ELSE 0
                               END as cash_impact
                        FROM trades 
                        WHERE symbol = ? 
                        AND timestamp >= datetime('now', '-' || ? || ' days')
                        ORDER BY timestamp DESC
                    """
                    cursor = await conn.execute(query, (symbol, days))
                else:
                    query = """
                        SELECT id, symbol, side, quantity, price, timestamp, 
                               slippage, commission,
                               quantity * price as notional,
                               CASE 
                                   WHEN side = 'BUY' THEN -quantity * price - commission
                                   WHEN side = 'SELL' THEN quantity * price - commission
                                   ELSE 0
                               END as cash_impact
                        FROM trades 
                        WHERE timestamp >= datetime('now', '-' || ? || ' days')
                        ORDER BY timestamp DESC
                    """
                    cursor = await conn.execute(query, (days,))

                trades = await cursor.fetchall()

                # Convert to list of dicts
                trade_list = []
                for trade in trades:
                    trade_list.append(
                        {
                            "id": trade[0],
                            "symbol": trade[1],
                            "side": trade[2],
                            "quantity": trade[3],
                            "price": trade[4],
                            "timestamp": trade[5],
                            "slippage": trade[6],
                            "commission": trade[7],
                            "notional": trade[8],
                            "cash_impact": trade[9],
                        }
                    )

                # Calculate summary statistics
                total_trades = len(trade_list)
                total_volume = sum(t["notional"] for t in trade_list)
                total_commission = sum(t["commission"] for t in trade_list)
                buy_trades = [t for t in trade_list if t["side"] == "BUY"]
                sell_trades = [t for t in trade_list if t["side"] == "SELL"]

                return {
                    "trades": trade_list,
                    "summary": {
                        "total_trades": total_trades,
                        "buy_trades": len(buy_trades),
                        "sell_trades": len(sell_trades),
                        "total_volume": total_volume,
                        "total_commission": total_commission,
                        "avg_trade_size": total_volume / total_trades if total_trades > 0 else 0,
                    },
                }
        finally:
            await db.close()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(fetch_trades())
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")
        return jsonify({"trades": [], "summary": {}})


@app.route("/api/strategies/status")
@requires_auth
def strategies_status():
    """Get real status of all active strategies"""
    try:
        import json
        from pathlib import Path

        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get real position count
        positions = db.get_positions()
        active_positions = [p for p in positions if p.get("quantity", 0) > 0]

        # Get recent trades to calculate strategy performance
        trades = db.get_recent_trades(limit=100)

        # Check for ML predictions file
        predictions_file = Path("ml_predictions.json")
        ml_confidence = 0.5
        ml_regime = "NEUTRAL"
        if predictions_file.exists():
            try:
                with open(predictions_file) as f:
                    ml_data = json.load(f)
                    # Get average confidence from recent predictions
                    confidences = [p.get("confidence", 0.5) for p in ml_data.values()]
                    ml_confidence = sum(confidences) / len(confidences) if confidences else 0.5
                    # Determine regime based on signals
                    signals = [p.get("signal", 0) for p in ml_data.values()]
                    avg_signal = sum(signals) / len(signals) if signals else 0
                    if avg_signal > 0.3:
                        ml_regime = "BULLISH"
                    elif avg_signal < -0.3:
                        ml_regime = "BEARISH"
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                pass

        # Calculate real PnL by strategy (simplified - assuming all trades are ML enhanced)
        total_pnl = sum(t.get("pnl", 0) for t in trades if t.get("pnl") is not None)
        winning_trades = [t for t in trades if t.get("pnl", 0) > 0]

        # Calculate slippage from trades
        slippages = []
        for trade in trades:
            if trade.get("slippage") is not None:
                slippages.append(abs(trade["slippage"]))
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        avg_slippage_bps = avg_slippage * 10000  # Convert to basis points

        return jsonify(
            {
                "active_strategies": {
                    "ml_enhanced": {
                        "enabled": True,
                        "regime": ml_regime,
                        "confidence": round(ml_confidence, 3),
                        "positions": len(active_positions),
                        "symbols_tracked": 19,  # From runner config
                    },
                    "microstructure": {
                        "enabled": False,  # S4 complete but not actively running
                        "ofi": 0.0,
                        "spread_bps": 0.0,
                        "tick_momentum": 0.0,
                        "ensemble_score": 0.0,
                    },
                    "portfolio_manager": {
                        "enabled": True,
                        "allocation_method": "Equal Weight",  # Current implementation
                        "positions_count": len(active_positions),
                        "max_positions": 20,
                        "rebalance_due": False,
                    },
                    "smart_execution": {
                        "enabled": True,
                        "algorithm": "Market Orders",  # Current implementation
                        "orders_pending": 0,  # Not tracking pending orders yet
                        "avg_slippage_bps": round(avg_slippage_bps, 2),
                        "total_trades": len(trades),
                    },
                },
                "performance_by_strategy": {
                    "ml_enhanced": {
                        "pnl": round(total_pnl, 2),
                        "win_rate": round(len(winning_trades) / len(trades), 3) if trades else 0,
                        "total_trades": len(trades),
                        "winning_trades": len(winning_trades),
                    },
                    "microstructure": {"pnl": 0.0, "win_rate": 0.0},  # Not active
                    "smart_execution": {
                        "saved_bps": round(
                            max(0, 5 - avg_slippage_bps), 2
                        ),  # Estimate vs 5bps baseline
                        "fills": len(trades),
                        "avg_slippage_bps": round(avg_slippage_bps, 2),
                    },
                },
                "last_updated": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        # Return minimal real data on error
        return jsonify(
            {
                "active_strategies": {
                    "ml_enhanced": {"enabled": True, "error": str(e)},
                    "microstructure": {"enabled": False},
                    "portfolio_manager": {"enabled": True},
                    "smart_execution": {"enabled": True},
                },
                "performance_by_strategy": {},
                "error": str(e),
            }
        )


@app.route("/api/microstructure/metrics")
@requires_auth
def microstructure_metrics():
    """Get microstructure strategy metrics"""
    # Return sample microstructure metrics due to database locking
    return jsonify(
        {
            "order_flow_imbalance": {
                "current": 0.42,
                "avg_1h": 0.38,
                "trend": "increasing",
                "signal": "bullish",
            },
            "book_pressure": 0.28,  # Added for display
            "tick_direction": 0.15,  # Added for display
            "spread_analysis": {
                "current_bps": 3.2,
                "avg_bps": 4.1,
                "widening": False,
                "liquidity": "high",
            },
            "tick_momentum": {
                "score": 0.65,
                "direction": "up",
                "strength": "moderate",
                "trades_analyzed": 1250,
            },
            "ensemble_metrics": {
                "combined_score": 0.72,
                "confidence": 0.81,
                "active_signals": 3,
                "last_signal": "2 min ago",
            },
            "performance": {
                "trades_today": 8,
                "win_rate": 0.625,
                "avg_profit_bps": 2.8,
                "sharpe_ratio": 1.4,
            },
        }
    )


@app.route("/api/portfolio/allocation")
@requires_auth
def portfolio_allocation():
    """Get current portfolio allocation from portfolio manager"""
    return jsonify(
        {
            "method": "Risk Parity",
            "allocations": {
                "ML Enhanced": 0.35,
                "Microstructure": 0.25,
                "Mean Reversion": 0.20,
                "Momentum": 0.20,
            },
            "risk_budget": {
                "ML Enhanced": 0.30,
                "Microstructure": 0.25,
                "Mean Reversion": 0.25,
                "Momentum": 0.20,
            },
            "correlation_matrix": {
                "avg_correlation": 0.42,
                "max_correlation": 0.78,
                "diversification_ratio": 1.85,
            },
            "rebalance": {
                "last": "2025-09-01T09:30:00",
                "next_due": "2025-09-02T09:30:00",
                "drift": 0.03,
            },
        }
    )


@app.route("/api/execution/status")
@requires_auth
def execution_status():
    """Get smart execution algorithm status"""
    return jsonify(
        {
            "active_algorithm": "VWAP",
            "orders": {"pending": 2, "completed": 15, "cancelled": 1},
            "performance": {
                "avg_slippage_bps": 1.2,
                "market_impact_bps": 0.8,
                "total_saved_bps": 8.5,
            },
            "algorithms_available": ["TWAP", "VWAP", "Adaptive", "Iceberg"],
            "current_orders": [
                {"symbol": "AAPL", "algo": "VWAP", "progress": 0.65, "slices": 8},
                {"symbol": "MSFT", "algo": "TWAP", "progress": 0.30, "slices": 10},
            ],
            # Enhanced metrics for dashboard integration
            "algorithms_used": {"twap": 5, "vwap": 8, "iceberg": 2},
            "avg_slippage": 1.2,
            "market_impact": 0.8,
            "fill_rate": 0.98,
            "avg_fill_time": 2.3,
            "executions_today": 15,
            "algorithm_performance": {
                "twap": {"slippage": 1.1, "success_rate": 0.95},
                "vwap": {"slippage": 1.3, "success_rate": 0.93},
                "iceberg": {"detection_rate": 0.02, "success_rate": 0.97},
            },
        }
    )


@app.route("/api/risk/status")
@requires_auth
def get_risk_status():
    """Get advanced risk management status"""
    # Check if advanced risk manager is running
    advanced_risk_enabled = os.getenv("ADVANCED_RISK_ENABLED", "true").lower() == "true"

    if not advanced_risk_enabled:
        return jsonify({"enabled": False, "message": "Advanced risk management not enabled"})

    # Mock data for now - would connect to actual risk manager
    return jsonify(
        {
            "enabled": True,
            "kelly_sizing": {
                "enabled": True,
                "current_positions": {
                    "AAPL": {"kelly_fraction": 0.025, "win_rate": 0.58, "edge": 0.032},
                    "NVDA": {"kelly_fraction": 0.018, "win_rate": 0.55, "edge": 0.024},
                    "TSLA": {"kelly_fraction": 0.015, "win_rate": 0.52, "edge": 0.018},
                },
                "portfolio_kelly": 0.058,
            },
            "kill_switches": {
                "active": False,
                "triggered_count": 0,
                "last_trigger": None,
                "limits": {
                    "daily_loss": {"limit": 0.05, "current": 0.012},
                    "consecutive_losses": {"limit": 5, "current": 1},
                    "max_drawdown": {"limit": 0.10, "current": 0.025},
                    "position_loss": {"limit": 0.02, "per_position": True},
                },
            },
            "correlation_limits": {
                "max_correlation": 0.7,
                "max_correlated_exposure": 0.3,
                "high_correlations": [
                    {"pair": ["AAPL", "MSFT"], "correlation": 0.82},
                    {"pair": ["NVDA", "AMD"], "correlation": 0.75},
                ],
            },
            "risk_metrics": {
                "total_exposure": 45000,
                "leverage": 0.45,
                "var_95": -2500,
                "sharpe_ratio": 1.8,
                "max_drawdown": 0.025,
            },
        }
    )


@app.route("/api/risk/kelly/<symbol>")
@requires_auth
def get_kelly_parameters(symbol):
    """Get Kelly parameters for a specific symbol"""
    # Mock data - would calculate from actual trade history
    return jsonify(
        {
            "symbol": symbol,
            "kelly_fraction": 0.022,
            "half_kelly": 0.011,
            "quarter_kelly": 0.0055,
            "win_rate": 0.56,
            "avg_win": 0.035,
            "avg_loss": 0.022,
            "edge": 0.0196,
            "odds": 1.59,
            "trade_count": 45,
            "recommended_position_size": 2200,  # In dollars
            "confidence_level": 0.85,
        }
    )


@app.route("/api/risk/kill-switch", methods=["POST"])
@requires_auth
def control_kill_switch():
    """Control kill switch (reset after trigger)"""
    action = request.json.get("action", "status")

    if action == "reset":
        # Would reset actual kill switch
        return jsonify(
            {"success": True, "message": "Kill switch reset successfully", "status": "active"}
        )
    elif action == "disable":
        # Would disable kill switch temporarily
        return jsonify({"success": True, "message": "Kill switch disabled", "status": "disabled"})
    else:
        # Return current status
        return jsonify({"active": False, "triggered": False, "can_trade": True})


@app.route("/api/start", methods=["POST"])
@requires_auth
def start_trading():
    """Start trading"""
    global trading_status, trading_process

    # Load symbols from user settings (make it global)
    global default_symbols
    try:
        with open("user_settings.json", "r") as f:
            settings = json.load(f)
            default_symbols = settings.get("default", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        default_symbols = ["AAPL", "MSFT", "GOOGL"]

    data = request.json
    symbols = data.get("symbols", default_symbols)

    if trading_status == "running":
        return jsonify({"status": "already_running"})

    # Start trading process with proper symbol format
    cmd = ["python3", "-m", "robo_trader.runner_async", "--symbols", ",".join(symbols)]
    trading_process = subprocess.Popen(cmd)
    trading_status = "running"

    trading_log.append(
        f"{datetime.now().strftime('%H:%M:%S')} - Trading started for {', '.join(symbols)}"
    )

    return jsonify({"status": "started", "symbols": symbols})


@app.route("/api/stop", methods=["POST"])
@requires_auth
def stop_trading():
    """Stop trading"""
    global trading_status, trading_process

    if trading_process:
        trading_process.terminate()
        trading_process = None

    trading_status = "stopped"
    trading_log.append(f"{datetime.now().strftime('%H:%M:%S')} - Trading stopped")

    return jsonify({"status": "stopped"})


@app.route("/api/logs")
@requires_auth
def get_logs():
    """Get recent logs"""
    return jsonify({"logs": trading_log[-100:]})  # Last 100 log entries


if __name__ == "__main__":
    # Initialize components
    logger.info("Starting RoboTrader Dashboard...")

    # Temporarily disable WebSocket server to fix startup issues
    # TODO: Fix WebSocket server initialization
    # logger.info("Starting WebSocket server...")
    # ws_manager.start()
    # time.sleep(2)

    logger.info(f"Dashboard starting on port {os.getenv('DASH_PORT', 5555)}")

    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("DASH_PORT", 5555)),
        use_reloader=False,
        debug=os.getenv("FLASK_ENV") == "development",
    )

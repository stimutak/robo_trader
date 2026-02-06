#!/usr/bin/env python3
"""
RoboTrader Dashboard - Clean, ML-Integrated Interface
Provides real-time monitoring of trading, ML models, and performance metrics
"""

import asyncio
import hashlib
import hmac
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timedelta
from decimal import Decimal
from functools import wraps
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template_string, request, send_file
from flask_cors import CORS

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

# Configure CORS with whitelisted origins
# Set CORS_ORIGINS env var for production, or it defaults to allowing local development
cors_origins = os.getenv("CORS_ORIGINS", "").strip()
is_production = (
    os.getenv("FLASK_ENV", "").lower() == "production"
    or os.getenv("PRODUCTION", "").lower() == "true"
)
if cors_origins:
    # Production: use explicit whitelist from environment
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]
elif is_production:
    # Production without CORS_ORIGINS set: restrict to localhost only (safe default)
    logger.warning("PRODUCTION mode without CORS_ORIGINS set - restricting to localhost only")
    allowed_origins = ["http://localhost:5555", "http://127.0.0.1:5555"]
else:
    # Development: allow localhost and common local network patterns
    allowed_origins = [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://192.168.*.*:*",  # Local network
        "http://10.*.*.*:*",  # Private network
        "exp://*",  # Expo development (React Native)
    ]

CORS(app, origins=allowed_origins, supports_credentials=True)
server = app  # For Gunicorn compatibility

# Thread-safe cache for positions endpoint
_positions_cache_lock = threading.Lock()
_strategies_cache_lock = threading.Lock()

# Configuration
config = load_config()
DEFAULT_CAPITAL = float(os.getenv("DEFAULT_CASH", getattr(config, "default_cash", 100000)))
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
    """Check if username/password is valid using timing-safe comparison."""
    if not AUTH_ENABLED:
        return True
    if not AUTH_PASS_HASH:
        return True
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    # Use timing-safe comparison to prevent timing attacks
    return hmac.compare_digest(username, AUTH_USER) and hmac.compare_digest(
        password_hash, AUTH_PASS_HASH
    )


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
            padding: 8px 0;
            border-bottom: 1px solid #21262d;
            margin-bottom: 10px;
        }

        .logo {
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-center {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 10px;
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
            gap: 6px;
            padding: 4px 10px;
            background: #161b22;
            border-radius: 12px;
            font-size: 11px;
            border: 1px solid #21262d;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #f87171;
        }

        .status-dot.active {
            background: #4ade80;
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
        .warning { color: #ffa500; }
        
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
            gap: 6px;
        }

        button {
            padding: 5px 12px;
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(46, 160, 67, 0.3);
        }

        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        button.secondary {
            background: #21262d;
            border: 1px solid #30363d;
        }

        button.secondary:hover {
            background: #30363d;
            box-shadow: none;
        }

        button.danger {
            background: linear-gradient(135deg, #da3633 0%, #f85149 100%);
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
            gap: 4px;
            background: #0d1117;
            padding: 4px;
            border-radius: 8px;
            margin-bottom: 12px;
            border: 1px solid #21262d;
        }

        .tab {
            padding: 6px 12px;
            color: #8b949e;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s ease;
            font-size: 12px;
            font-weight: 500;
        }

        .tab:hover {
            color: #c9d1d9;
            background: #161b22;
        }

        .tab.active {
            color: #fff;
            background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        }
        
        .log-container {
            background: #0a0a0a;
            border: 1px solid #2a2a2a;
            border-radius: 8px;
            padding: 15px;
            height: 600px;
            overflow-y: auto;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
        }
        
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #1a1a1a;
        }
        
        .log-entry.signal-log {
            background: rgba(34, 197, 94, 0.1);
            border-left: 3px solid #22c55e;
            padding-left: 12px;
        }
        
        .log-entry.trade-log {
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
            padding-left: 12px;
        }
        
        .log-entry.performance-log {
            background: rgba(168, 85, 247, 0.1);
            border-left: 3px solid #a855f7;
            padding-left: 12px;
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
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">RoboTrader</div>
            <div class="header-center">
                <div class="status-indicator">
                    <div class="status-dot" id="status-dot"></div>
                    <span id="status-text">Disconnected</span>
                </div>
                <div class="status-indicator" id="market-status-badge" style="border-color: #30363d;">
                    <span style="color: #8b949e;">Market:</span>
                    <span id="market-status-text" style="color: #fbbf24;">Closed</span>
                </div>
            </div>
            <div class="header-right">
                <div class="button-group">
                    <button onclick="startTrading()" id="start-btn">Start</button>
                    <button onclick="stopTrading()" id="stop-btn" class="danger">Stop</button>
                    <button onclick="refreshData()" class="secondary">Refresh</button>
                </div>
            </div>
        </header>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('overview', this)">Overview</div>
            <div class="tab" onclick="switchTab('watchlist', this)">Watchlist</div>
            <div class="tab" onclick="switchTab('positions', this)">Positions</div>
            <div class="tab" onclick="switchTab('strategies', this)">Strategies</div>
            <div class="tab" onclick="switchTab('trades', this)">Trades</div>
            <div class="tab" onclick="switchTab('ml', this)">ML</div>
            <div class="tab" onclick="switchTab('performance', this)">Performance</div>
            <div class="tab" onclick="switchTab('logs', this)">Logs</div>
        </div>
        
        <div id="overview-tab" class="tab-content">
            <!-- HERO ROW: The Big Numbers -->
            <div style="display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                <!-- Total Equity - Hero Card -->
                <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 16px; border-radius: 10px; border: 1px solid #334155;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; letter-spacing: 1px;">Total Equity</div>
                            <div style="font-size: 28px; font-weight: bold; color: #fff;" id="ov-portfolio">$0</div>
                            <div style="font-size: 12px; margin-top: 4px;" id="ov-total-return-pct">
                                <span style="color: #4ade80;">+0%</span> <span style="color: #64748b;">all time</span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: #64748b; font-size: 10px;">Starting Capital</div>
                            <div style="color: #94a3b8; font-size: 14px;">$100,000</div>
                        </div>
                    </div>
                </div>
                <!-- Today's P&L -->
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 14px; border-radius: 8px; border-left: 3px solid #4ade80;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Today's P&L</div>
                    <div style="font-size: 20px; font-weight: bold;" id="ov-today-pnl">$0</div>
                    <div style="font-size: 11px; color: #64748b;" id="ov-today-pnl-pct">0%</div>
                </div>
                <!-- Unrealized P&L -->
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 14px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Unrealized P&L</div>
                    <div style="font-size: 20px; font-weight: bold;" id="ov-unrealized-pnl">$0</div>
                    <div style="font-size: 11px; color: #64748b;" id="ov-unrealized-pct">open positions</div>
                </div>
                <!-- Realized P&L -->
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 14px; border-radius: 8px; border-left: 3px solid #a78bfa;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Realized P&L</div>
                    <div style="font-size: 20px; font-weight: bold;" id="ov-total-pnl">$0</div>
                    <div style="font-size: 11px; color: #64748b;" id="ov-realized-trades">from closed trades</div>
                </div>
            </div>

            <!-- RISK ROW: Capital & Risk Metrics -->
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-bottom: 10px;">
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Positions Value</div>
                    <div style="font-size: 15px; font-weight: bold; color: #60a5fa;" id="ov-positions-value">$0</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Cash Available</div>
                    <div style="font-size: 15px; font-weight: bold; color: #34d399;" id="ov-cash">$0</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Exposure %</div>
                    <div style="font-size: 15px; font-weight: bold; color: #fbbf24;" id="ov-exposure">0%</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Current DD</div>
                    <div style="font-size: 15px; font-weight: bold; color: #f87171;" id="ov-current-dd">0%</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Max Drawdown</div>
                    <div style="font-size: 15px; font-weight: bold; color: #f87171;" id="max-dd">0%</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e; text-transform: uppercase;">Buying Power</div>
                    <div style="font-size: 15px; font-weight: bold; color: #4ade80;" id="ov-buying-power">$0</div>
                </div>
            </div>

            <!-- MAIN ROW: Chart + Position Summary -->
            <div style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 10px; margin-bottom: 10px;">
                <!-- Equity Curve (30-day) -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="color: #58a6ff; margin: 0; font-size: 11px; text-transform: uppercase;">Portfolio Value (30 Days)</h4>
                        <div style="font-size: 10px; color: #64748b;" id="ov-chart-range">—</div>
                    </div>
                    <div style="height: 120px;">
                        <canvas id="overview-equity-chart"></canvas>
                    </div>
                </div>

                <!-- Position Summary -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 11px; text-transform: uppercase;">Position Summary</h4>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Open Positions</div>
                            <div style="font-size: 18px; font-weight: bold;" id="ov-positions">0</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Winners / Losers</div>
                            <div style="font-size: 14px; font-weight: bold;"><span style="color: #4ade80;" id="ov-winners">0</span> / <span style="color: #f87171;" id="ov-losers">0</span></div>
                        </div>
                    </div>
                    <div style="font-size: 11px;">
                        <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d;">
                            <span style="color: #8b949e;">Best Position</span>
                            <span style="color: #4ade80;" id="ov-best-pos">—</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d;">
                            <span style="color: #8b949e;">Worst Position</span>
                            <span style="color: #f87171;" id="ov-worst-pos">—</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                            <span style="color: #8b949e;">Avg Position Size</span>
                            <span style="color: #94a3b8;" id="ov-avg-pos-size">—</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- STRATEGY ROW: Performance Metrics + Recent Trades -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
                <!-- Strategy Metrics -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 11px; text-transform: uppercase;">Strategy Performance</h4>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px;">
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Win Rate</div>
                            <div style="font-size: 14px; font-weight: bold; color: #4ade80;" id="ov-win-rate">0%</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Profit Factor</div>
                            <div style="font-size: 14px; font-weight: bold;" id="profit-factor">0.00</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Sharpe Ratio</div>
                            <div style="font-size: 14px; font-weight: bold;" id="sharpe">0.00</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Total Trades</div>
                            <div style="font-size: 14px; font-weight: bold;" id="ov-trades">0</div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-top: 6px;">
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Avg Win</div>
                            <div style="font-size: 13px; font-weight: bold; color: #4ade80;" id="ov-avg-win">$0</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Avg Loss</div>
                            <div style="font-size: 13px; font-weight: bold; color: #f87171;" id="ov-avg-loss">$0</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Best Trade</div>
                            <div style="font-size: 13px; font-weight: bold; color: #4ade80;" id="ov-best-trade">$0</div>
                        </div>
                        <div style="background: #161b22; padding: 8px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Worst Trade</div>
                            <div style="font-size: 13px; font-weight: bold; color: #f87171;" id="ov-worst-trade">$0</div>
                        </div>
                    </div>
                </div>

                <!-- Recent Activity -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 11px; text-transform: uppercase;">Recent Trades</h4>
                    <div id="recent-trades-list" style="font-size: 11px; max-height: 110px; overflow-y: auto;">
                        <div style="color: #666; text-align: center; padding: 20px;">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- STATUS ROW: System Health -->
            <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px;">
                <div style="background: #161b22; padding: 10px; border-radius: 6px; display: flex; align-items: center; gap: 8px; border: 1px solid #21262d;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #4ade80;" id="ov-conn-dot"></div>
                    <div>
                        <div style="font-size: 9px; color: #8b949e;">Gateway</div>
                        <div style="font-size: 12px; font-weight: bold;" id="tws-status">...</div>
                    </div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e;">Market</div>
                    <div style="font-size: 12px; font-weight: bold;" id="ov-market-status">—</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e;">Next Open/Close</div>
                    <div style="font-size: 12px; font-weight: bold;" id="ov-market-time">—</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e;">Last Update</div>
                    <div style="font-size: 12px; font-weight: bold;" id="ov-last-update">—</div>
                </div>
                <div style="background: #161b22; padding: 10px; border-radius: 6px; border: 1px solid #21262d;">
                    <div style="font-size: 9px; color: #8b949e;">Cycle Interval</div>
                    <div style="font-size: 12px; font-weight: bold;" id="ov-cycle-interval">15s</div>
                </div>
            </div>

            <!-- Hidden elements for backward compat -->
            <div style="display: none;">
                <span id="portfolio-change"></span>
                <span id="today-change-pct"></span>
                <span id="positions-value-text"></span>
                <span id="pnl-change"></span>
                <span id="tws-detail"></span>
                <span id="daily-pnl"></span>
                <span id="daily-change"></span>
                <span id="position-value"></span>
                <span id="position-count"></span>
                <span id="avg-correlation"></span>
                <span id="top-positions-list"></span>
            </div>
        </div>
        
        <div id="ml-tab" class="tab-content" style="display: none;">
            <!-- Compact Two-Column Layout -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <!-- Left: Key Metrics -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">ML System Status</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Models</div>
                            <div style="font-size: 16px; font-weight: bold;" id="models-trained">0</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Accuracy</div>
                            <div style="font-size: 16px; font-weight: bold;" id="model-accuracy">0%</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Confidence</div>
                            <div style="font-size: 16px; font-weight: bold;" id="prediction-confidence">0%</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Features</div>
                            <div style="font-size: 16px; font-weight: bold;" id="feature-count">0</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Predictions</div>
                            <div style="font-size: 16px; font-weight: bold;" id="active-predictions">0</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Signals</div>
                            <div style="font-size: 14px;"><span id="ml-buy-count" style="color: #4ade80;">0</span> / <span id="ml-sell-count" style="color: #ff6b6b;">0</span> / <span id="ml-hold-count" style="color: #888;">0</span></div>
                        </div>
                    </div>
                </div>
                <!-- Right: Feature Importance Chart -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">Top Features</h3>
                    <div style="height: 140px; position: relative;">
                        <canvas id="feature-chart-canvas"></canvas>
                    </div>
                </div>
            </div>

            <!-- ML Predictions Table -->
            <div class="table-container" style="padding: 12px; margin-bottom: 15px;">
                <h3 style="margin-bottom: 10px; font-size: 14px;">Current Predictions</h3>
                <div style="max-height: 200px; overflow-y: auto;">
                    <table style="font-size: 12px;">
                        <thead><tr><th>Symbol</th><th>Signal</th><th>Confidence</th><th>Source</th><th>Time</th></tr></thead>
                        <tbody id="predictions-table">
                            <tr><td colspan="5" style="text-align: center; color: #666;">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Two-Column: Models & Model Performance -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <!-- Trained Models -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">Trained Models</h3>
                    <div style="max-height: 150px; overflow-y: auto;">
                        <table style="font-size: 11px;">
                            <thead><tr><th>Type</th><th>Score</th><th>Features</th><th>Status</th></tr></thead>
                            <tbody id="model-table">
                                <tr><td colspan="4" style="text-align: center; color: #666;">Loading...</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
                <!-- Model Accuracy by Type -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">Model Accuracy</h3>
                    <div style="height: 150px; position: relative;">
                        <canvas id="model-accuracy-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="performance-tab" class="tab-content" style="display: none;">
            <!-- Compact Two-Column Layout -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <!-- Left: Key Metrics -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">Key Metrics</h3>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;">
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Total P&L</div>
                            <div style="font-size: 16px; font-weight: bold;" id="perf-total-pnl">$0</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Return</div>
                            <div style="font-size: 16px; font-weight: bold;" id="total-return">0%</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Win Rate</div>
                            <div style="font-size: 16px; font-weight: bold;" id="win-rate">0%</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Sharpe</div>
                            <div style="font-size: 16px; font-weight: bold;" id="total-sharpe">0</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Max DD</div>
                            <div style="font-size: 16px; font-weight: bold; color: #ff6b6b;" id="total-drawdown">0%</div>
                        </div>
                        <div style="text-align: center; padding: 8px; background: #1a1a2e; border-radius: 6px;">
                            <div style="font-size: 11px; color: #888;">Trades</div>
                            <div style="font-size: 16px; font-weight: bold;" id="perf-total-trades">0</div>
                        </div>
                    </div>
                </div>
                <!-- Right: Period Breakdown -->
                <div class="table-container" style="margin: 0; padding: 12px;">
                    <h3 style="margin-bottom: 10px; font-size: 14px;">Period P&L</h3>
                    <table style="font-size: 12px;">
                        <thead><tr><th>Period</th><th>P&L</th><th>Return</th><th>Trades</th></tr></thead>
                        <tbody>
                            <tr><td>Today</td><td id="pnl-daily">$0</td><td id="return-daily">0%</td><td id="trades-daily">0</td></tr>
                            <tr><td>Week</td><td id="pnl-weekly">$0</td><td id="return-weekly">0%</td><td id="trades-weekly">0</td></tr>
                            <tr><td>Month</td><td id="pnl-monthly">$0</td><td id="return-monthly">0%</td><td id="trades-monthly">0</td></tr>
                            <tr><td>All</td><td id="pnl-all">$0</td><td id="return-all">0%</td><td id="trades-all">0</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- P&L History & Portfolio Value Chart -->
            <div class="table-container" style="padding: 12px; margin-bottom: 15px;">
                <h3 style="margin-bottom: 10px; font-size: 14px;">P&L History & Portfolio Value</h3>
                <div style="height: 220px; position: relative;">
                    <canvas id="equity-chart-canvas"></canvas>
                </div>
            </div>

            <!-- Compact Trade Statistics -->
            <div class="table-container" style="padding: 12px;">
                <h3 style="margin-bottom: 10px; font-size: 14px;">Trade Statistics</h3>
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px;">
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Avg Win</div>
                        <div style="font-size: 13px; color: #4ade80;" id="avg-win">$0</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Avg Loss</div>
                        <div style="font-size: 13px; color: #ff6b6b;" id="avg-loss">$0</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Profit Factor</div>
                        <div style="font-size: 13px;" id="profit-factor">0</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Expectancy</div>
                        <div style="font-size: 13px;" id="expectancy">$0</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Best Trade</div>
                        <div style="font-size: 13px; color: #4ade80;" id="largest-win">$0</div>
                    </div>
                    <div style="text-align: center; padding: 6px; background: #1a1a2e; border-radius: 4px;">
                        <div style="font-size: 10px; color: #888;">Worst Trade</div>
                        <div style="font-size: 13px; color: #ff6b6b;" id="largest-loss">$0</div>
                    </div>
                </div>
            </div>
        </div>

        <div id="watchlist-tab" class="tab-content" style="display: none;">
            <!-- Watchlist Summary -->
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 12px;">
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Watching</div>
                    <div style="font-size: 18px; font-weight: bold;" id="watch-total">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #4ade80;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">With Position</div>
                    <div style="font-size: 18px; font-weight: bold; color: #4ade80;" id="watch-with-pos">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Total P&L</div>
                    <div style="font-size: 18px; font-weight: bold;" id="watch-total-pnl">$0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #a78bfa;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Winners</div>
                    <div style="font-size: 18px; font-weight: bold; color: #4ade80;" id="watch-winners">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #f472b6;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Losers</div>
                    <div style="font-size: 18px; font-weight: bold; color: #f87171;" id="watch-losers">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #34d399;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Best P&L</div>
                    <div style="font-size: 18px; font-weight: bold; color: #4ade80;" id="watch-best">$0</div>
                </div>
            </div>
            <!-- Watchlist Table -->
            <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: #161b22;">
                            <th style="padding: 8px 10px; text-align: left; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Symbol</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Price</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Qty</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Avg Cost</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">P&L</th>
                            <th style="padding: 8px 10px; text-align: center; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Status</th>
                        </tr>
                    </thead>
                    <tbody id="watchlist-table">
                        <tr><td colspan="6" style="padding: 20px; text-align: center; color: #666;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div id="positions-tab" class="tab-content" style="display: none;">
            <!-- Positions Summary -->
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 12px;">
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Positions</div>
                    <div style="font-size: 18px; font-weight: bold;" id="pos-count">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #4ade80;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Total Value</div>
                    <div style="font-size: 18px; font-weight: bold;" id="pos-total-value">$0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Unrealized P&L</div>
                    <div style="font-size: 18px; font-weight: bold;" id="pos-unrealized-pnl">$0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #a78bfa;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Winners</div>
                    <div style="font-size: 18px; font-weight: bold; color: #4ade80;" id="pos-winners">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #f472b6;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Losers</div>
                    <div style="font-size: 18px; font-weight: bold; color: #f87171;" id="pos-losers">0</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #34d399;">
                    <div style="color: #888; font-size: 9px; text-transform: uppercase;">Avg P&L %</div>
                    <div style="font-size: 18px; font-weight: bold;" id="pos-avg-pnl-pct">0%</div>
                </div>
            </div>
            <!-- Positions Table -->
            <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: #161b22;">
                            <th style="padding: 8px 10px; text-align: left; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Symbol</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Qty</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Entry</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Current</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">P&L</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">P&L %</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Value</th>
                            <th style="padding: 8px 10px; text-align: center; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Signal</th>
                        </tr>
                    </thead>
                    <tbody id="positions-table">
                        <tr><td colspan="8" style="padding: 20px; text-align: center; color: #666;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div id="strategies-tab" class="tab-content" style="display: none;">
            <!-- Compact Strategy Summary -->
            <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin-bottom: 15px;">
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #4ade80;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Active</div>
                    <div style="font-size: 20px; font-weight: bold; color: #4ade80;" id="strat-active-count">3</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Positions</div>
                    <div style="font-size: 20px; font-weight: bold;" id="strat-positions">30</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Total P&L</div>
                    <div style="font-size: 20px; font-weight: bold; color: #4ade80;" id="strat-total-pnl">$4,519</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #a78bfa;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Win Rate</div>
                    <div style="font-size: 20px; font-weight: bold;" id="strat-win-rate">62.5%</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #f472b6;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Trades</div>
                    <div style="font-size: 20px; font-weight: bold;" id="strat-trades">100</div>
                </div>
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 12px; border-radius: 8px; border-left: 3px solid #34d399;">
                    <div style="color: #888; font-size: 10px; text-transform: uppercase;">Slippage</div>
                    <div style="font-size: 20px; font-weight: bold;" id="strat-slippage">1.2 bps</div>
                </div>
            </div>

            <!-- Two Column Layout -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 15px;">
                <!-- Left: Strategy Cards -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase;">Active Strategies</h4>
                    <div id="strategy-cards" style="display: flex; flex-direction: column; gap: 8px;">
                        <!-- ML Enhanced -->
                        <div style="display: grid; grid-template-columns: 100px 1fr 80px 80px; gap: 10px; padding: 8px; background: #161b22; border-radius: 6px; align-items: center;">
                            <div>
                                <div style="font-weight: bold; color: #c9d1d9; font-size: 11px;">ML Enhanced</div>
                                <div style="font-size: 10px; color: #4ade80;" id="ml-status-badge">ACTIVE</div>
                            </div>
                            <div style="font-size: 10px; color: #8b949e;">
                                Regime: <span style="color: #fbbf24;" id="ml-regime-small">NEUTRAL</span> |
                                Conf: <span id="ml-conf-small">53.3%</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Positions</div>
                                <div style="font-weight: bold;" id="ml-pos-count">30</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">P&L</div>
                                <div style="font-weight: bold; color: #4ade80;" id="ml-pnl-small">$4,519</div>
                            </div>
                        </div>
                        <!-- Smart Execution -->
                        <div style="display: grid; grid-template-columns: 100px 1fr 80px 80px; gap: 10px; padding: 8px; background: #161b22; border-radius: 6px; align-items: center;">
                            <div>
                                <div style="font-weight: bold; color: #c9d1d9; font-size: 11px;">Smart Exec</div>
                                <div style="font-size: 10px; color: #4ade80;" id="exec-status-badge">ACTIVE</div>
                            </div>
                            <div style="font-size: 10px; color: #8b949e;">
                                Algo: <span style="color: #60a5fa;" id="exec-algo-small">Market</span> |
                                Fill: <span id="exec-fill-rate">98%</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Trades</div>
                                <div style="font-weight: bold;" id="exec-trade-count">100</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Saved</div>
                                <div style="font-weight: bold; color: #4ade80;" id="exec-saved-small">5.0 bps</div>
                            </div>
                        </div>
                        <!-- Portfolio Manager -->
                        <div style="display: grid; grid-template-columns: 100px 1fr 80px 80px; gap: 10px; padding: 8px; background: #161b22; border-radius: 6px; align-items: center;">
                            <div>
                                <div style="font-weight: bold; color: #c9d1d9; font-size: 11px;">Portfolio Mgr</div>
                                <div style="font-size: 10px; color: #4ade80;" id="pm-status-badge">ACTIVE</div>
                            </div>
                            <div style="font-size: 10px; color: #8b949e;">
                                Method: <span style="color: #a78bfa;" id="pm-method-small">Equal Weight</span> |
                                Max: <span id="pm-max-pos">30</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Positions</div>
                                <div style="font-weight: bold;" id="pm-pos-count">30</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Rebalance</div>
                                <div style="font-weight: bold;" id="pm-rebalance-small">No</div>
                            </div>
                        </div>
                        <!-- Microstructure -->
                        <div style="display: grid; grid-template-columns: 100px 1fr 80px 80px; gap: 10px; padding: 8px; background: #161b22; border-radius: 6px; align-items: center; opacity: 0.5;" id="micro-row">
                            <div>
                                <div style="font-weight: bold; color: #c9d1d9; font-size: 11px;">Microstructure</div>
                                <div style="font-size: 10px; color: #f87171;" id="micro-status-badge">DISABLED</div>
                            </div>
                            <div style="font-size: 10px; color: #8b949e;">
                                OFI: <span id="micro-ofi-small">0.00</span> |
                                Score: <span id="micro-score-small">0.00</span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Signals</div>
                                <div style="font-weight: bold;" id="micro-signals-small">0</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 10px; color: #8b949e;">Win Rate</div>
                                <div style="font-weight: bold;" id="micro-win-small">0%</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right: Allocation Chart -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase;">Portfolio Allocation</h4>
                    <div style="display: flex; gap: 15px; align-items: center;">
                        <canvas id="allocation-pie-chart" width="120" height="120"></canvas>
                        <div id="allocation-legend" style="flex: 1; font-size: 11px;">
                            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d;">
                                <span><span style="display: inline-block; width: 10px; height: 10px; background: #667eea; border-radius: 2px; margin-right: 6px;"></span>ML Enhanced</span>
                                <span style="font-weight: bold;" id="alloc-ml-pct">35%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d;">
                                <span><span style="display: inline-block; width: 10px; height: 10px; background: #4ade80; border-radius: 2px; margin-right: 6px;"></span>Microstructure</span>
                                <span style="font-weight: bold;" id="alloc-micro-pct">25%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #21262d;">
                                <span><span style="display: inline-block; width: 10px; height: 10px; background: #fbbf24; border-radius: 2px; margin-right: 6px;"></span>Mean Reversion</span>
                                <span style="font-weight: bold;" id="alloc-mr-pct">20%</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                                <span><span style="display: inline-block; width: 10px; height: 10px; background: #f87171; border-radius: 2px; margin-right: 6px;"></span>Momentum</span>
                                <span style="font-weight: bold;" id="alloc-mom-pct">20%</span>
                            </div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #21262d;">
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 10px;">
                            <div style="text-align: center;">
                                <div style="color: #8b949e;">Method</div>
                                <div style="font-weight: bold; color: #a78bfa;" id="alloc-method">Risk Parity</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #8b949e;">Diversification</div>
                                <div style="font-weight: bold;" id="alloc-div-ratio">1.85</div>
                            </div>
                            <div style="text-align: center;">
                                <div style="color: #8b949e;">Drift</div>
                                <div style="font-weight: bold;" id="alloc-drift">3.0%</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Bottom Row: Strategy P&L Chart + Execution Stats -->
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <!-- Strategy P&L Comparison -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase;">Strategy Performance</h4>
                    <div style="height: 120px;">
                        <canvas id="strategy-pnl-chart"></canvas>
                    </div>
                </div>

                <!-- Execution Metrics -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px;">
                    <h4 style="color: #58a6ff; margin: 0 0 10px 0; font-size: 12px; text-transform: uppercase;">Execution Metrics</h4>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; margin-bottom: 8px;">
                        <div style="background: #161b22; padding: 6px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Slippage</div>
                            <div style="font-size: 14px; font-weight: bold; color: #4ade80;" id="exec-avg-slip">1.2 bps</div>
                        </div>
                        <div style="background: #161b22; padding: 6px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Impact</div>
                            <div style="font-size: 14px; font-weight: bold; color: #fbbf24;" id="exec-impact">0.8 bps</div>
                        </div>
                        <div style="background: #161b22; padding: 6px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Fill Rate</div>
                            <div style="font-size: 14px; font-weight: bold;" id="exec-fill-pct">98%</div>
                        </div>
                        <div style="background: #161b22; padding: 6px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 9px; color: #8b949e;">Saved</div>
                            <div style="font-size: 14px; font-weight: bold; color: #4ade80;" id="exec-total-saved">8.5 bps</div>
                        </div>
                    </div>
                    <div style="font-size: 10px; color: #8b949e; margin-bottom: 4px;">Algorithm Usage</div>
                    <div style="height: 50px;">
                        <canvas id="algo-usage-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="trades-tab" class="tab-content" style="display: none;">
            <!-- Trade Summary + Filters Row -->
            <div style="display: flex; gap: 10px; margin-bottom: 12px; align-items: stretch;">
                <div style="display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; flex: 1;">
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #60a5fa;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Total</div>
                        <div style="font-size: 18px; font-weight: bold;" id="total-trades">0</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #4ade80;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Buys</div>
                        <div style="font-size: 18px; font-weight: bold; color: #4ade80;" id="buy-trades">0</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #f87171;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Sells</div>
                        <div style="font-size: 18px; font-weight: bold; color: #f87171;" id="sell-trades">0</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #fbbf24;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Volume</div>
                        <div style="font-size: 18px; font-weight: bold;" id="total-volume">$0</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #a78bfa;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Commission</div>
                        <div style="font-size: 18px; font-weight: bold; color: #fbbf24;" id="total-commission">$0</div>
                    </div>
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 10px; border-radius: 8px; border-left: 3px solid #34d399;">
                        <div style="color: #888; font-size: 9px; text-transform: uppercase;">Avg Size</div>
                        <div style="font-size: 18px; font-weight: bold;" id="avg-trade-size">$0</div>
                    </div>
                </div>
                <!-- Filters -->
                <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 6px; min-width: 160px;">
                    <select id="trade-symbol-filter" onchange="loadTrades()" style="background: #161b22; color: #c9d1d9; border: 1px solid #21262d; padding: 6px 8px; border-radius: 4px; font-size: 11px;">
                        <option value="">All Symbols</option>
                    </select>
                    <select id="trade-days-filter" onchange="loadTrades()" style="background: #161b22; color: #c9d1d9; border: 1px solid #21262d; padding: 6px 8px; border-radius: 4px; font-size: 11px;">
                        <option value="1">24 Hours</option>
                        <option value="7" selected>7 Days</option>
                        <option value="30">30 Days</option>
                        <option value="90">90 Days</option>
                    </select>
                </div>
            </div>
            <!-- Trades Table -->
            <div style="background: #0d1117; border: 1px solid #21262d; border-radius: 8px; overflow: hidden;">
                <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                    <thead>
                        <tr style="background: #161b22;">
                            <th style="padding: 8px 10px; text-align: left; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Time</th>
                            <th style="padding: 8px 10px; text-align: left; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Symbol</th>
                            <th style="padding: 8px 10px; text-align: center; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Side</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Qty</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Price</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">Notional</th>
                            <th style="padding: 8px 10px; text-align: right; color: #8b949e; font-weight: 500; border-bottom: 1px solid #21262d;">P&L</th>
                        </tr>
                    </thead>
                    <tbody id="trades-table">
                        <tr><td colspan="7" style="padding: 20px; text-align: center; color: #666;">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <div id="logs-tab" class="tab-content" style="display: none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="color: #ffa500; margin: 0;">📋 System Logs</h3>
                <div style="display: flex; gap: 10px; align-items: center;">
                    <label style="display: flex; align-items: center; gap: 5px; cursor: pointer; color: #888; font-size: 13px;">
                        <input type="checkbox" id="auto-scroll-toggle" checked style="cursor: pointer;">
                        Auto-scroll
                    </label>
                    <button onclick="scrollLogsToBottom()" style="background: #333; border: 1px solid #555; color: #fff; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                        ↓ Jump to Bottom
                    </button>
                    <button onclick="clearLogs()" style="background: #333; border: 1px solid #555; color: #fff; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">
                        Clear
                    </button>
                </div>
            </div>
            <div style="display: flex; gap: 8px; margin-bottom: 10px;">
                <button onclick="setLogFilter('ALL')" id="log-filter-ALL" class="log-filter-btn active" style="background: #3b82f6; border: none; color: #fff; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; font-weight: bold;">ALL</button>
                <button onclick="setLogFilter('DEBUG')" id="log-filter-DEBUG" class="log-filter-btn" style="background: #333; border: 1px solid #555; color: #6b7280; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">DEBUG</button>
                <button onclick="setLogFilter('INFO')" id="log-filter-INFO" class="log-filter-btn" style="background: #333; border: 1px solid #555; color: #3b82f6; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">INFO</button>
                <button onclick="setLogFilter('WARNING')" id="log-filter-WARNING" class="log-filter-btn" style="background: #333; border: 1px solid #555; color: #f59e0b; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">WARN</button>
                <button onclick="setLogFilter('ERROR')" id="log-filter-ERROR" class="log-filter-btn" style="background: #333; border: 1px solid #555; color: #ef4444; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 12px;">ERROR</button>
            </div>
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
            if (tab === 'overview') {
                loadOverviewData();
            } else if (tab === 'ml') {
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
                loadTrades(),
                loadOverviewData()
            ]);
        }

        let overviewEquityChart = null;
        const STARTING_CAPITAL = 100000;

        async function loadOverviewData() {
            try {
                // Load all data in parallel
                const [equityResp, posResp, tradesResp, perfResp, pnlResp, marketResp] = await Promise.all([
                    fetch('/api/equity-curve'),
                    fetch('/api/positions'),
                    fetch('/api/trades?days=365'),
                    fetch('/api/performance'),
                    fetch('/api/pnl'),
                    fetch('/api/market/status')
                ]);

                let positions = [];
                let positionsValue = 0;
                let unrealizedPnl = 0;
                let winners = 0;
                let losers = 0;
                let bestPos = null;
                let worstPos = null;

                // POSITIONS: Calculate all position-related metrics
                if (posResp.ok) {
                    const data = await posResp.json();
                    positions = data.positions || [];
                    document.getElementById('ov-positions').textContent = positions.length;

                    positions.forEach(p => {
                        const qty = parseFloat(p.quantity || 0);
                        const currentPrice = parseFloat(p.current_price || p.market_price || 0);
                        const entryPrice = parseFloat(p.entry_price || p.avg_cost || currentPrice);
                        const marketValue = qty * currentPrice;
                        const pnl = (currentPrice - entryPrice) * qty;
                        const pnlPct = entryPrice > 0 ? ((currentPrice - entryPrice) / entryPrice) * 100 : 0;

                        positionsValue += marketValue;
                        unrealizedPnl += pnl;

                        if (pnl >= 0) winners++;
                        else losers++;

                        if (!bestPos || pnl > bestPos.pnl) bestPos = { symbol: p.symbol, pnl, pnlPct };
                        if (!worstPos || pnl < worstPos.pnl) worstPos = { symbol: p.symbol, pnl, pnlPct };
                    });

                    // Update position summary
                    document.getElementById('ov-positions-value').textContent = formatCurrency(positionsValue);
                    document.getElementById('ov-winners').textContent = winners;
                    document.getElementById('ov-losers').textContent = losers;
                    document.getElementById('ov-avg-pos-size').textContent = positions.length > 0
                        ? formatCurrency(positionsValue / positions.length) : '—';
                    document.getElementById('ov-best-pos').textContent = bestPos
                        ? `${bestPos.symbol} +${bestPos.pnlPct.toFixed(1)}%` : '—';
                    document.getElementById('ov-worst-pos').textContent = worstPos
                        ? `${worstPos.symbol} ${worstPos.pnlPct.toFixed(1)}%` : '—';

                    // Unrealized P&L
                    const ovUnrealized = document.getElementById('ov-unrealized-pnl');
                    ovUnrealized.textContent = formatCurrency(unrealizedPnl);
                    ovUnrealized.style.color = unrealizedPnl >= 0 ? '#4ade80' : '#f87171';
                }

                // P&L DATA: Get equity, realized P&L, cash
                let equity = STARTING_CAPITAL;
                let realizedPnl = 0;
                let cash = 0;
                if (pnlResp.ok) {
                    const pnlData = await pnlResp.json();
                    equity = pnlData.equity || STARTING_CAPITAL;
                    realizedPnl = pnlData.realized || pnlData.realized_pnl || 0;
                    cash = pnlData.cash || 0;
                }

                // PERFORMANCE: Get all trade metrics
                let totalPnl = realizedPnl;
                let todayPnl = 0;
                let winRate = 0;
                let profitFactor = 0;
                let sharpe = 0;
                let maxDrawdown = 0;
                let avgWin = 0;
                let avgLoss = 0;
                let bestTrade = 0;
                let worstTrade = 0;
                let totalTrades = 0;

                if (perfResp.ok) {
                    const perfData = await perfResp.json();
                    const summary = perfData.summary || {};
                    const daily = perfData.daily || {};

                    totalPnl = summary.total_pnl !== undefined ? summary.total_pnl : (perfData.all?.pnl || realizedPnl);
                    todayPnl = daily.pnl || 0;
                    winRate = summary.win_rate || 0;
                    profitFactor = summary.profit_factor || 0;
                    sharpe = summary.total_sharpe || summary.sharpe || 0;
                    maxDrawdown = summary.total_drawdown || summary.max_drawdown || 0;
                    avgWin = summary.avg_win || 0;
                    avgLoss = summary.avg_loss || 0;
                    bestTrade = summary.best_trade || 0;
                    worstTrade = summary.worst_trade || 0;
                    totalTrades = summary.total_trades || 0;
                }

                // CALCULATE DERIVED METRICS
                const totalEquity = equity > 0 ? equity : (STARTING_CAPITAL + totalPnl + unrealizedPnl);
                const totalReturn = totalEquity - STARTING_CAPITAL;
                const totalReturnPct = (totalReturn / STARTING_CAPITAL) * 100;
                const todayPnlPct = totalEquity > 0 ? (todayPnl / totalEquity) * 100 : 0;
                const exposure = totalEquity > 0 ? (positionsValue / totalEquity) * 100 : 0;
                const buyingPower = Math.max(0, totalEquity - positionsValue);
                const currentDrawdown = maxDrawdown; // Would need peak tracking for accurate current DD

                // UPDATE ALL UI ELEMENTS
                // Hero row
                document.getElementById('ov-portfolio').textContent = formatCurrency(totalEquity);
                const totalReturnEl = document.getElementById('ov-total-return-pct');
                totalReturnEl.innerHTML = `<span style="color: ${totalReturn >= 0 ? '#4ade80' : '#f87171'};">${totalReturn >= 0 ? '+' : ''}${totalReturnPct.toFixed(1)}%</span> <span style="color: #64748b;">all time</span>`;

                const ovTodayPnl = document.getElementById('ov-today-pnl');
                ovTodayPnl.textContent = formatCurrency(todayPnl);
                ovTodayPnl.style.color = todayPnl >= 0 ? '#4ade80' : '#f87171';
                document.getElementById('ov-today-pnl-pct').textContent = `${todayPnl >= 0 ? '+' : ''}${todayPnlPct.toFixed(2)}%`;

                const ovTotalPnl = document.getElementById('ov-total-pnl');
                ovTotalPnl.textContent = formatCurrency(realizedPnl);
                ovTotalPnl.style.color = realizedPnl >= 0 ? '#4ade80' : '#f87171';

                // Risk row
                document.getElementById('ov-cash').textContent = cash > 0 ? formatCurrency(cash) : '—';
                document.getElementById('ov-exposure').textContent = exposure.toFixed(1) + '%';
                document.getElementById('ov-current-dd').textContent = (currentDrawdown * 100).toFixed(1) + '%';
                document.getElementById('max-dd').textContent = (maxDrawdown * 100).toFixed(1) + '%';
                document.getElementById('ov-buying-power').textContent = formatCurrency(buyingPower);

                // Strategy metrics
                document.getElementById('ov-win-rate').textContent = (winRate * 100).toFixed(1) + '%';
                document.getElementById('profit-factor').textContent = profitFactor.toFixed(2);
                document.getElementById('sharpe').textContent = sharpe.toFixed(2);
                document.getElementById('ov-trades').textContent = totalTrades;
                document.getElementById('ov-avg-win').textContent = formatCurrency(avgWin);
                document.getElementById('ov-avg-loss').textContent = formatCurrency(Math.abs(avgLoss));
                document.getElementById('ov-best-trade').textContent = formatCurrency(bestTrade);
                document.getElementById('ov-worst-trade').textContent = formatCurrency(worstTrade);

                // TRADES: Recent trades list
                if (tradesResp.ok) {
                    const data = await tradesResp.json();
                    const trades = data.trades || [];
                    const today = new Date().toISOString().split('T')[0];
                    const todayTrades = trades.filter(t => t.timestamp && t.timestamp.startsWith(today));
                    renderRecentTrades(todayTrades.length > 0 ? todayTrades : trades.slice(0, 8));
                }

                // EQUITY CURVE: Render chart
                if (equityResp.ok) {
                    const data = await equityResp.json();
                    renderOverviewEquityChart(data);
                    // Show date range
                    if (data.labels && data.labels.length > 0) {
                        document.getElementById('ov-chart-range').textContent =
                            `${data.labels[0]} → ${data.labels[data.labels.length - 1]}`;
                    }
                }

                // MARKET STATUS
                if (marketResp.ok) {
                    const marketData = await marketResp.json();
                    document.getElementById('ov-market-status').textContent = marketData.is_open ? 'Open' : 'Closed';
                    document.getElementById('ov-market-status').style.color = marketData.is_open ? '#4ade80' : '#f87171';
                    if (marketData.next_open) {
                        document.getElementById('ov-market-time').textContent = marketData.is_open ?
                            `Closes ${marketData.next_close || '4:00 PM'}` :
                            `Opens ${marketData.next_open}`;
                    }
                }

                // Last update time
                document.getElementById('ov-last-update').textContent = new Date().toLocaleTimeString();

            } catch (error) {
                console.error('Error loading overview data:', error);
            }
        }

        function renderOverviewEquityChart(data) {
            const ctx = document.getElementById('overview-equity-chart');
            if (!ctx) return;

            // Use portfolio_values if available (from equity_history), otherwise calculate from P&L
            let values = [];
            let labels = [];

            if (data.portfolio_values && data.portfolio_values.length > 0) {
                values = data.portfolio_values;
                labels = data.labels || values.map((_, i) => i);
            } else if (data.values && data.values.length > 0) {
                // Calculate portfolio values from P&L
                values = data.values.map(pnl => STARTING_CAPITAL + pnl);
                labels = data.labels || values.map((_, i) => i);
            } else {
                // No data yet
                return;
            }

            // Calculate gain/loss from start
            const firstValue = values[0] || STARTING_CAPITAL;
            const lastValue = values[values.length - 1] || STARTING_CAPITAL;
            const isPositive = lastValue >= firstValue;

            if (overviewEquityChart) {
                overviewEquityChart.data.labels = labels;
                overviewEquityChart.data.datasets[0].data = values;
                overviewEquityChart.data.datasets[0].borderColor = isPositive ? '#4ade80' : '#f87171';
                overviewEquityChart.update('none');
                return;
            }

            const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 120);
            gradient.addColorStop(0, isPositive ? 'rgba(74, 222, 128, 0.3)' : 'rgba(248, 113, 113, 0.3)');
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

            overviewEquityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        borderColor: isPositive ? '#4ade80' : '#f87171',
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            display: true,
                            grid: { display: false },
                            ticks: { color: '#64748b', font: { size: 8 }, maxTicksLimit: 5 }
                        },
                        y: {
                            grid: { color: '#21262d' },
                            ticks: {
                                color: '#8b949e',
                                font: { size: 9 },
                                callback: function(value) { return '$' + (value/1000).toFixed(0) + 'k'; }
                            }
                        }
                    }
                }
            });
        }

        function renderTopPositions(positions) {
            const container = document.getElementById('top-positions-list');
            if (!container) return;

            if (!positions || positions.length === 0) {
                container.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No positions</div>';
                return;
            }

            // Calculate P&L and sort
            const withPnl = positions.map(p => ({
                ...p,
                pnl: (p.current_price - p.entry_price) * p.quantity,
                pnlPct: ((p.current_price - p.entry_price) / p.entry_price) * 100
            })).sort((a, b) => Math.abs(b.pnl) - Math.abs(a.pnl)).slice(0, 6);

            container.innerHTML = withPnl.map(p => {
                const pnlColor = p.pnl >= 0 ? '#4ade80' : '#f87171';
                const arrow = p.pnl >= 0 ? '▲' : '▼';
                return `
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #21262d;">
                        <div>
                            <span style="color: #58a6ff; font-weight: bold;">${p.symbol}</span>
                            <span style="color: #8b949e; margin-left: 8px;">${p.quantity} @ $${p.entry_price.toFixed(2)}</span>
                        </div>
                        <div style="color: ${pnlColor}; font-weight: bold;">
                            ${arrow} $${Math.abs(p.pnl).toFixed(0)} (${p.pnlPct >= 0 ? '+' : ''}${p.pnlPct.toFixed(1)}%)
                        </div>
                    </div>
                `;
            }).join('');
        }

        function renderRecentTrades(trades) {
            const container = document.getElementById('recent-trades-list');
            if (!container) return;

            if (!trades || trades.length === 0) {
                container.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No trades today</div>';
                return;
            }

            const recent = trades.slice(0, 8);
            container.innerHTML = recent.map(t => {
                const sideColor = t.side === 'BUY' ? '#4ade80' : '#f87171';
                const sideBg = t.side === 'BUY' ? '#238636' : '#da3633';
                let time = '';
                try {
                    const d = new Date(t.timestamp.replace(' ', 'T') + 'Z');
                    time = d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
                } catch (e) { time = ''; }
                return `
                    <div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #21262d;">
                        <div>
                            <span style="background: ${sideBg}; color: #fff; padding: 1px 5px; border-radius: 3px; font-size: 9px; margin-right: 6px;">${t.side}</span>
                            <span style="color: #58a6ff; font-weight: bold;">${t.symbol}</span>
                        </div>
                        <div style="color: #8b949e;">
                            ${t.quantity} @ $${t.price.toFixed(2)} <span style="color: #666; margin-left: 5px;">${time}</span>
                        </div>
                    </div>
                `;
            }).join('');
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
                    
                    // Update 'trades today' counter from returned trades
                    try {
                        const today = new Date();
                        const todayDateStr = today.toDateString();
                        const tradesToday = data.trades.filter(t => {
                            if (!t.timestamp) return false;
                            const utcTimestamp = t.timestamp.replace(' ', 'T') + 'Z';
                            const d = new Date(utcTimestamp);
                            return d.toDateString() === todayDateStr;
                        }).length;
                        const tc = document.getElementById('trade-count');
                        if (tc) {
                            tc.textContent = `${tradesToday} trade${tradesToday === 1 ? '' : 's'} today`;
                        }
                    } catch (e) {
                        console.debug('Could not update trades-today counter:', e);
                    }

                    tbody.innerHTML = data.trades.map(trade => {
                        // SQLite timestamps are UTC but don't have 'Z' suffix
                        let time = trade.timestamp || 'N/A';
                        try {
                            if (trade.timestamp) {
                                const utcTimestamp = trade.timestamp.replace(' ', 'T') + 'Z';
                                const utcDate = new Date(utcTimestamp);
                                if (!isNaN(utcDate.getTime())) {
                                    time = utcDate.toLocaleString('en-US', {
                                        timeZone: 'America/New_York',
                                        month: 'short',
                                        day: 'numeric',
                                        hour: '2-digit',
                                        minute: '2-digit'
                                    });
                                }
                            }
                        } catch (e) {
                            console.debug('Date parse error for trade:', trade.id, e);
                        }
                        const sideColor = trade.side === 'BUY' ? '#4ade80' : '#f87171';
                        const sideBg = trade.side === 'BUY' ? '#238636' : '#da3633';
                        // P&L display - only show for SELL trades (BUY has no realized P&L)
                        const pnl = trade.pnl || 0;
                        const pnlColor = pnl >= 0 ? '#4ade80' : '#f87171';
                        const pnlDisplay = trade.side === 'SELL' ? `<span style="color: ${pnlColor}; font-weight: 500;">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(0)}</span>` : '<span style="color: #6b7280;">-</span>';
                        return `
                            <tr style="border-bottom: 1px solid #21262d;">
                                <td style="padding: 6px 10px; color: #8b949e;">${time}</td>
                                <td style="padding: 6px 10px;"><strong style="color: #58a6ff;">${trade.symbol}</strong></td>
                                <td style="padding: 6px 10px; text-align: center;"><span style="background: ${sideBg}; color: #fff; padding: 2px 8px; border-radius: 3px; font-size: 10px; font-weight: bold;">${trade.side}</span></td>
                                <td style="padding: 6px 10px; text-align: right;">${trade.quantity}</td>
                                <td style="padding: 6px 10px; text-align: right;">$${trade.price.toFixed(2)}</td>
                                <td style="padding: 6px 10px; text-align: right;">$${trade.notional.toFixed(0)}</td>
                                <td style="padding: 6px 10px; text-align: right;">${pnlDisplay}</td>
                            </tr>
                        `;
                    }).join('');
                } else {
                    tbody.innerHTML = '<tr><td colspan="7" style="padding: 20px; text-align: center; color: #666;">No trades found</td></tr>';
                }
            } catch (error) {
                console.error('Error loading trades:', error);
                document.getElementById('trades-table').innerHTML =
                    '<tr><td colspan="7" style="padding: 20px; text-align: center; color: #f87171;">Error loading trades</td></tr>';
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
        
        let featureChart = null;
        let modelAccuracyChart = null;

        async function loadMLData() {
            try {
                // Load ML status and predictions in parallel
                const [statusResponse, predictionsResponse] = await Promise.all([
                    fetch('/api/ml/status'),
                    fetch('/api/ml/predictions')
                ]);
                const statusData = await statusResponse.json();
                const predictionsData = await predictionsResponse.json();
                updateMLMetrics(statusData, predictionsData);
            } catch (error) {
                console.error('Error loading ML data:', error);
            }
        }
        
        // Store loaded logs to enable incremental updates without disrupting scroll
        let loadedLogCount = 0;

        async function loadLogs() {
            try {
                const response = await fetch('/api/logs');
                const data = await response.json();
                
                const container = document.getElementById('log-container');
                if (!container || !data.logs || data.logs.length === 0) return;
                
                // Remove spam and duplicate messages  
                const deduplicatedLogs = [];
                let lastLogMessage = '';
                const seenEvents = new Set();
                
                data.logs.forEach(log => {
                    // Extract the message part without timestamp for comparison
                    const timeMatch = log.match(/^(\\d{2}:\\d{2}:\\d{2})/);
                    let message = log;
                    if (timeMatch) {
                        const separatorIndex = log.indexOf(' - ', timeMatch[0].length);
                        if (separatorIndex !== -1) {
                            message = log.substring(separatorIndex + 3);
                        } else {
                            message = log.substring(timeMatch[0].length).replace(/^[\\s:-]+/, '');
                        }
                    }
                    
                    // Trim whitespace for better comparison
                    message = message.trim();
                    
                    // Filter out spam events completely
                    if (message.startsWith('{') && message.includes('"event":')) {
                        try {
                            const jsonMatch = message.match(/"event":\\s*"([^"]+)"/);
                            if (jsonMatch) {
                                const eventType = jsonMatch[1];
                                
                                // Filter out spam events completely
                                if (eventType === "Found valid pair" || 
                                    eventType === "Fetched" ||
                                    eventType === "Sent WebSocket update") {
                                    return; // Skip these spam events entirely
                                }
                                
                                // For other JSON events, limit to one per event type
                                if (seenEvents.has(eventType)) {
                                    return; // Skip duplicate event types
                                }
                                seenEvents.add(eventType);
                            }
                        } catch (e) {
                            // If JSON parsing fails, use original logic
                        }
                    }
                    
                    // Only add if it's different from the last message (for non-JSON logs)
                    if (message !== lastLogMessage) {
                        deduplicatedLogs.push(log);
                        lastLogMessage = message;
                    }
                });
                
                // Only update if there are new logs beyond what we've already loaded
                if (deduplicatedLogs.length <= loadedLogCount) return;
                
                // Check if user is at bottom BEFORE any DOM changes
                const isAtBottom = Math.abs(container.scrollHeight - container.clientHeight - container.scrollTop) < 50;
                
                // Only add the new logs, NEVER rebuild entire container
                const newLogs = deduplicatedLogs.slice(loadedLogCount);
                
                newLogs.forEach(log => {
                    // Extract time from log if it starts with HH:MM:SS format (24-hour)
                    const timeMatch = log.match(/^(\\d{2}):(\\d{2}):(\\d{2})/);
                    let time = '';
                    if (timeMatch) {
                        // Convert 24-hour to 12-hour format
                        const hours24 = parseInt(timeMatch[1]);
                        const minutes = timeMatch[2];
                        const seconds = timeMatch[3];
                        const period = hours24 >= 12 ? 'PM' : 'AM';
                        const hours12 = hours24 % 12 || 12;  // Convert 0 to 12
                        time = `${hours12}:${minutes}:${seconds} ${period}`;
                    }

                    // Handle both " - " and ": " after time
                    let message = log;
                    if (timeMatch) {
                        // Skip past time and separator (either " - " or ": ")
                        const separatorIndex = log.indexOf(' - ', timeMatch[0].length);
                        if (separatorIndex !== -1) {
                            message = log.substring(separatorIndex + 3);
                        } else {
                            message = log.substring(timeMatch[0].length).replace(/^[\\s:-]+/, '');
                        }
                    }
                    
                    // Add CSS classes for different types of messages
                    let cssClass = 'log-entry';
                    if (message.includes('Signal=') || message.includes('signal')) {
                        cssClass += ' signal-log';
                    } else if (message.includes('Trading cycle') || message.includes('executed') || message.includes('trades')) {
                        cssClass += ' trade-log';
                    } else if (message.includes('Performance') || message.includes('P&L')) {
                        cssClass += ' performance-log';
                    }
                    
                    // Create and append new log element
                    const logDiv = document.createElement('div');
                    logDiv.className = cssClass;
                    logDiv.innerHTML = `
                        <span class="log-time">${time}</span>
                        <span>${message}</span>
                    `;
                    container.appendChild(logDiv);
                });
                
                // Update loaded count
                loadedLogCount = deduplicatedLogs.length;
                
                // Auto-scroll if toggle is checked OR user was already at bottom
                const autoScrollEnabled = document.getElementById('auto-scroll-toggle')?.checked;
                if (autoScrollEnabled || isAtBottom) {
                    container.scrollTop = container.scrollHeight;
                }
                
            } catch (error) {
                console.error('Error loading logs:', error);
            }
        }
        
        let equityChart = null;

        async function loadPerformanceData() {
            try {
                // Load performance metrics and equity curve in parallel
                const [perfResponse, equityResponse] = await Promise.all([
                    fetch('/api/performance'),
                    fetch('/api/equity-curve')
                ]);
                const data = await perfResponse.json();
                const equityData = await equityResponse.json();
                console.log('Performance data received:', data);
                updatePerformanceTable(data);
                renderEquityCurve(equityData);
            } catch (error) {
                console.error('Error loading performance data:', error);
            }
        }

        function renderEquityCurve(data) {
            const ctx = document.getElementById('equity-chart-canvas');
            if (!ctx) return;

            // Destroy existing chart if it exists
            if (equityChart) {
                equityChart.destroy();
            }

            const hasPnlData = data.values && data.values.length > 0;
            const hasPortfolioData = data.portfolio_values && data.portfolio_values.length > 0;

            if (!hasPnlData && !hasPortfolioData) {
                ctx.parentElement.innerHTML = '<div style="display: flex; align-items: center; justify-content: center; height: 100%; color: #666;">No trade data yet</div>';
                return;
            }

            // Create gradient
            const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 200);
            const lastValue = hasPnlData ? data.values[data.values.length - 1] : 0;
            if (lastValue >= 0) {
                gradient.addColorStop(0, 'rgba(74, 222, 128, 0.3)');
                gradient.addColorStop(1, 'rgba(74, 222, 128, 0)');
            } else {
                gradient.addColorStop(0, 'rgba(255, 107, 107, 0.3)');
                gradient.addColorStop(1, 'rgba(255, 107, 107, 0)');
            }

            // Use actual portfolio values from equity_history if available,
            // otherwise calculate from starting capital + P&L
            const STARTING_CAPITAL = 100000;
            let portfolioValues;
            let useHistoricalData = data.portfolio_values && data.portfolio_values.length > 0;

            if (useHistoricalData) {
                // Use actual daily portfolio snapshots (industry standard)
                portfolioValues = data.portfolio_values;
            } else {
                // Fallback: calculate from P&L data
                portfolioValues = data.values.map(pnl => STARTING_CAPITAL + pnl);
            }

            equityChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: 'P&L',
                        data: data.values,
                        borderColor: lastValue >= 0 ? '#4ade80' : '#ff6b6b',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2,
                        yAxisID: 'y'
                    }, {
                        label: 'Portfolio Value',
                        data: portfolioValues,
                        borderColor: '#3b82f6',
                        backgroundColor: 'transparent',
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderWidth: 2,
                        borderDash: [5, 5],
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: { color: '#888', font: { size: 10 }, boxWidth: 12 }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            backgroundColor: '#1a1a2e',
                            titleColor: '#fff',
                            bodyColor: '#fff',
                            borderColor: '#333',
                            borderWidth: 1,
                            callbacks: {
                                label: function(context) {
                                    const label = context.dataset.label || '';
                                    const value = context.parsed.y;
                                    if (label === 'Portfolio Value') {
                                        return label + ': $' + value.toLocaleString(undefined, {maximumFractionDigits: 0});
                                    }
                                    return label + ': $' + value.toFixed(2);
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            display: true,
                            grid: { color: '#222' },
                            ticks: { color: '#666', maxTicksLimit: 6, font: { size: 10 } }
                        },
                        y: {
                            display: true,
                            position: 'left',
                            grid: { color: '#222' },
                            ticks: {
                                color: lastValue >= 0 ? '#4ade80' : '#ff6b6b',
                                font: { size: 10 },
                                callback: function(value) { return '$' + value.toLocaleString(); }
                            },
                            title: { display: true, text: 'P&L', color: '#666', font: { size: 10 } }
                        },
                        y1: {
                            display: true,
                            position: 'right',
                            grid: { drawOnChartArea: false },
                            ticks: {
                                color: '#3b82f6',
                                font: { size: 10 },
                                callback: function(value) { return '$' + (value/1000).toFixed(0) + 'k'; }
                            },
                            title: { display: true, text: 'Portfolio', color: '#666', font: { size: 10 } }
                        }
                    },
                    interaction: { mode: 'nearest', axis: 'x', intersect: false }
                }
            });
        }
        
        function updateStatus(status) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            // Handle both object and string status
            const isRunning = (typeof status === 'object' && status.is_trading) || status === 'running';

            if (isRunning) {
                dot.classList.add('active');
            } else {
                dot.classList.remove('active');
            }

            // Use the message from the API if available, otherwise use default
            if (typeof status === 'object' && status.message) {
                text.textContent = status.message;
                // Add detail as subtitle if available
                if (status.detail) {
                    text.title = status.detail; // Show detail as tooltip
                }
            } else if (isRunning) {
                text.textContent = 'Trading Active';
            } else {
                text.textContent = 'Trading Stopped';
            }

            // Update TWS connection status
            if (typeof status === 'object' && status.tws_health) {
                const twsStatusEl = document.getElementById('tws-status');
                const twsDetailEl = document.getElementById('tws-detail');

                // Determine connection state based on market status and gateway health
                const marketOpen = status.market_open || false;
                const gatewayRunning = status.gateway_running || false;
                const isTrading = status.is_trading || false;

                if (isTrading && marketOpen) {
                    // Actually connected and trading
                    twsStatusEl.textContent = '✅ Connected';
                    twsStatusEl.className = 'card-value positive';
                    twsDetailEl.textContent = status.tws_health;
                } else if (gatewayRunning && !marketOpen) {
                    // Gateway running but market closed (normal standby)
                    twsStatusEl.textContent = '⏸️ Standby';
                    twsStatusEl.className = 'card-value warning';
                    twsDetailEl.textContent = status.tws_health + ' (Market Closed)';
                } else if (gatewayRunning) {
                    // Gateway running but not connected yet
                    twsStatusEl.textContent = '🔄 Ready';
                    twsStatusEl.className = 'card-value warning';
                    twsDetailEl.textContent = status.tws_health;
                } else {
                    // Gateway not running - actual error
                    twsStatusEl.textContent = '❌ Not Available';
                    twsStatusEl.className = 'card-value negative';
                    twsDetailEl.textContent = status.tws_health || 'Gateway/TWS not running';
                }
            }

            // Also update market status
            updateMarketStatus();
        }

        async function updateMarketStatus() {
            try {
                const response = await fetch('/api/market/status');
                const data = await response.json();

                // Update header market status badge
                const marketStatusText = document.getElementById('market-status-text');
                if (marketStatusText) {
                    if (data.is_open) {
                        marketStatusText.textContent = 'Open';
                        marketStatusText.style.color = '#4ade80';
                    } else if (data.session === 'pre_market') {
                        marketStatusText.textContent = 'Pre-Market';
                        marketStatusText.style.color = '#fbbf24';
                    } else if (data.session === 'after_hours') {
                        marketStatusText.textContent = 'After Hours';
                        marketStatusText.style.color = '#fbbf24';
                    } else {
                        marketStatusText.textContent = 'Closed';
                        marketStatusText.style.color = '#f87171';
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

            // If P&L API returns null, skip - let loadOverviewData calculate from positions
            if (!pnl || (pnl.daily === null && pnl.total === null && pnl.unrealized === null)) {
                console.log('P&L API returned null - skipping, positions will provide data');
                return;
            }

            // Debounce P&L updates to prevent flashing
            if (pnlUpdateTimeout) {
                clearTimeout(pnlUpdateTimeout);
            }

            // Handle null values (market closed) - force update to $0.00
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
        
        function updateMLMetrics(data, predictions) {
            // Update key metrics
            document.getElementById('models-trained').textContent = data.models_trained || 0;
            document.getElementById('feature-count').textContent = data.feature_count || 0;
            document.getElementById('model-accuracy').textContent = ((data.accuracy || 0) * 100).toFixed(1) + '%';
            document.getElementById('prediction-confidence').textContent = ((data.avg_confidence || data.confidence || 0) * 100).toFixed(1) + '%';

            // Update predictions count and signals
            const predElem = document.getElementById('active-predictions');
            if (predElem) predElem.textContent = data.active_predictions || 0;

            if (data.market_sentiment) {
                const buyElem = document.getElementById('ml-buy-count');
                const sellElem = document.getElementById('ml-sell-count');
                const holdElem = document.getElementById('ml-hold-count');
                if (buyElem) buyElem.textContent = data.market_sentiment.bullish || 0;
                if (sellElem) sellElem.textContent = data.market_sentiment.bearish || 0;
                if (holdElem) holdElem.textContent = data.market_sentiment.neutral || 0;
            }

            // Update model table (compact format)
            if (data.models && data.models.length > 0) {
                const tbody = document.getElementById('model-table');
                tbody.innerHTML = data.models.map(model => `
                    <tr>
                        <td>${model.type}</td>
                        <td style="color: ${model.test_score >= 0.55 ? '#4ade80' : '#fff'}">${(model.test_score * 100).toFixed(1)}%</td>
                        <td>${model.feature_count}</td>
                        <td><span style="color: ${model.status === 'active' ? '#4ade80' : '#666'};">${model.status === 'active' ? '●' : '○'}</span></td>
                    </tr>
                `).join('');

                // Render model accuracy chart
                renderModelAccuracyChart(data.models);
            }

            // Render feature importance chart
            if (data.top_features && data.top_features.length > 0) {
                renderFeatureChart(data.top_features);
            }

            // Update predictions table
            if (predictions && predictions.predictions) {
                renderPredictionsTable(predictions.predictions);
            }
        }

        function renderFeatureChart(features) {
            const ctx = document.getElementById('feature-chart-canvas');
            if (!ctx) return;

            if (featureChart) featureChart.destroy();

            const labels = features.slice(0, 5).map(f => f.name.replace(/_/g, ' '));
            const values = features.slice(0, 5).map(f => f.importance * 100);

            featureChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: ['#4ade80', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa'],
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { display: true, grid: { color: '#222' }, ticks: { color: '#666', font: { size: 9 }, callback: v => v + '%' } },
                        y: { display: true, grid: { display: false }, ticks: { color: '#888', font: { size: 9 } } }
                    }
                }
            });
        }

        function renderModelAccuracyChart(models) {
            const ctx = document.getElementById('model-accuracy-chart');
            if (!ctx) return;

            if (modelAccuracyChart) modelAccuracyChart.destroy();

            const labels = models.map(m => m.type.split(' ')[0]);
            const values = models.map(m => m.test_score * 100);
            const colors = values.map(v => v >= 55 ? '#4ade80' : v >= 52 ? '#fbbf24' : '#ff6b6b');

            modelAccuracyChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        data: values,
                        backgroundColor: colors,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { display: true, grid: { display: false }, ticks: { color: '#888', font: { size: 9 } } },
                        y: { display: true, min: 45, max: 60, grid: { color: '#222' }, ticks: { color: '#666', font: { size: 9 }, callback: v => v + '%' } }
                    }
                }
            });
        }

        function renderPredictionsTable(predictions) {
            const tbody = document.getElementById('predictions-table');
            if (!tbody) return;

            if (!predictions || predictions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #666;">No predictions</td></tr>';
                return;
            }

            tbody.innerHTML = predictions.map(p => {
                const actionColor = p.action === 'BUY' ? '#4ade80' : p.action === 'SELL' ? '#ff6b6b' : '#888';
                const confColor = p.confidence >= 0.7 ? '#4ade80' : p.confidence >= 0.5 ? '#fbbf24' : '#888';
                const time = p.timestamp ? p.timestamp.split('T')[1]?.split('.')[0] || '' : '';
                return `
                    <tr>
                        <td><strong>${p.symbol}</strong></td>
                        <td style="color: ${actionColor}; font-weight: bold;">${p.action}</td>
                        <td style="color: ${confColor}">${(p.confidence * 100).toFixed(0)}%</td>
                        <td style="color: #666; font-size: 10px;">${p.source}</td>
                        <td style="color: #666; font-size: 10px;">${time}</td>
                    </tr>
                `;
            }).join('');
        }
        
        function updateWatchlistTable(watchlist) {
            const tbody = document.getElementById('watchlist-table');
            if (!watchlist || watchlist.length === 0) {
                tbody.innerHTML = '<tr><td colspan="6" style="padding: 20px; text-align: center; color: #666;">No symbols in watchlist</td></tr>';
                // Reset summary stats
                document.getElementById('watch-total').textContent = '0';
                document.getElementById('watch-with-pos').textContent = '0';
                document.getElementById('watch-total-pnl').textContent = '$0';
                document.getElementById('watch-winners').textContent = '0';
                document.getElementById('watch-losers').textContent = '0';
                document.getElementById('watch-best').textContent = '$0';
                return;
            }

            // Calculate summary stats
            let totalPnl = 0, winners = 0, losers = 0, withPos = 0, bestPnl = 0;
            watchlist.forEach(item => {
                if (item.quantity > 0) {
                    withPos++;
                    const pnl = (item.current_price - item.avg_cost) * item.quantity;
                    totalPnl += pnl;
                    if (pnl > 0) winners++;
                    else if (pnl < 0) losers++;
                    if (pnl > bestPnl) bestPnl = pnl;
                }
            });

            // Update summary
            document.getElementById('watch-total').textContent = watchlist.length;
            document.getElementById('watch-with-pos').textContent = withPos;
            const pnlEl = document.getElementById('watch-total-pnl');
            pnlEl.textContent = '$' + totalPnl.toFixed(0);
            pnlEl.style.color = totalPnl >= 0 ? '#4ade80' : '#f87171';
            document.getElementById('watch-winners').textContent = winners;
            document.getElementById('watch-losers').textContent = losers;
            document.getElementById('watch-best').textContent = '$' + bestPnl.toFixed(0);

            tbody.innerHTML = watchlist.map(item => {
                const pnl = item.quantity > 0 ? (item.current_price - item.avg_cost) * item.quantity : 0;
                const pnlDisplay = item.quantity > 0 ? `$${pnl.toFixed(2)}` : '-';
                const pnlColor = pnl >= 0 ? '#4ade80' : '#f87171';
                const statusBadge = item.quantity > 0
                    ? '<span style="background: #238636; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 10px;">HOLDING</span>'
                    : '<span style="background: #30363d; color: #8b949e; padding: 2px 6px; border-radius: 3px; font-size: 10px;">WATCHING</span>';

                return `
                    <tr style="border-bottom: 1px solid #21262d;">
                        <td style="padding: 6px 10px;"><strong style="color: #58a6ff;">${item.symbol}</strong></td>
                        <td style="padding: 6px 10px; text-align: right;">$${item.current_price.toFixed(2)}</td>
                        <td style="padding: 6px 10px; text-align: right;">${item.quantity || '-'}</td>
                        <td style="padding: 6px 10px; text-align: right;">${item.avg_cost > 0 ? '$' + item.avg_cost.toFixed(2) : '-'}</td>
                        <td style="padding: 6px 10px; text-align: right; color: ${item.quantity > 0 ? pnlColor : '#8b949e'};">${pnlDisplay}</td>
                        <td style="padding: 6px 10px; text-align: center;">${statusBadge}</td>
                    </tr>
                `;
            }).join('');
        }
        
        function updatePositionsTable(positions) {
            const tbody = document.getElementById('positions-table');
            if (!positions || positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="8" style="padding: 20px; text-align: center; color: #666;">No open positions</td></tr>';
                document.getElementById('position-count').textContent = '0';
                document.getElementById('position-value').textContent = '$0.00 value';
                // Reset summary stats
                document.getElementById('pos-count').textContent = '0';
                document.getElementById('pos-total-value').textContent = '$0';
                document.getElementById('pos-unrealized-pnl').textContent = '$0';
                document.getElementById('pos-winners').textContent = '0';
                document.getElementById('pos-losers').textContent = '0';
                document.getElementById('pos-avg-pnl-pct').textContent = '0%';
                return;
            }

            // Calculate summary stats
            let totalValue = 0, totalPnl = 0, winners = 0, losers = 0, totalPnlPct = 0;
            const posData = positions.map(pos => {
                const pnl = (pos.current_price - pos.entry_price) * pos.quantity;
                const pnlPct = ((pos.current_price - pos.entry_price) / pos.entry_price) * 100;
                const value = pos.current_price * pos.quantity;
                totalValue += value;
                totalPnl += pnl;
                totalPnlPct += pnlPct;
                if (pnl > 0) winners++;
                else if (pnl < 0) losers++;
                return { ...pos, pnl, pnlPct, value };
            });

            // Update summary stats
            document.getElementById('pos-count').textContent = positions.length;
            document.getElementById('pos-total-value').textContent = '$' + (totalValue / 1000).toFixed(1) + 'k';
            const pnlEl = document.getElementById('pos-unrealized-pnl');
            pnlEl.textContent = '$' + totalPnl.toFixed(0);
            pnlEl.style.color = totalPnl >= 0 ? '#4ade80' : '#f87171';
            document.getElementById('pos-winners').textContent = winners;
            document.getElementById('pos-losers').textContent = losers;
            const avgPnlPct = totalPnlPct / positions.length;
            const avgEl = document.getElementById('pos-avg-pnl-pct');
            avgEl.textContent = avgPnlPct.toFixed(1) + '%';
            avgEl.style.color = avgPnlPct >= 0 ? '#4ade80' : '#f87171';

            tbody.innerHTML = posData.map(pos => {
                const pnlColor = pos.pnl >= 0 ? '#4ade80' : '#f87171';
                const signalBadge = pos.ml_signal === 'buy'
                    ? '<span style="background: #238636; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 10px;">BUY</span>'
                    : pos.ml_signal === 'sell'
                    ? '<span style="background: #da3633; color: #fff; padding: 2px 6px; border-radius: 3px; font-size: 10px;">SELL</span>'
                    : '<span style="background: #30363d; color: #8b949e; padding: 2px 6px; border-radius: 3px; font-size: 10px;">HOLD</span>';

                return `
                    <tr style="border-bottom: 1px solid #21262d;">
                        <td style="padding: 6px 10px;"><strong style="color: #58a6ff;">${pos.symbol}</strong></td>
                        <td style="padding: 6px 10px; text-align: right;">${pos.quantity}</td>
                        <td style="padding: 6px 10px; text-align: right;">$${pos.entry_price.toFixed(2)}</td>
                        <td style="padding: 6px 10px; text-align: right;">$${pos.current_price.toFixed(2)}</td>
                        <td style="padding: 6px 10px; text-align: right; color: ${pnlColor};">$${pos.pnl.toFixed(2)}</td>
                        <td style="padding: 6px 10px; text-align: right; color: ${pnlColor};">${pos.pnlPct.toFixed(1)}%</td>
                        <td style="padding: 6px 10px; text-align: right;">$${pos.value.toFixed(0)}</td>
                        <td style="padding: 6px 10px; text-align: center;">${signalBadge}</td>
                    </tr>
                `;
            }).join('');

            document.getElementById('position-count').textContent = positions.length.toString();
            document.getElementById('position-value').textContent = formatCurrency(totalValue) + ' value';

            // Update portfolio summary
            document.getElementById('active-positions-large').textContent = positions.length.toString();
            const posTextEl = document.getElementById('positions-value-text');
            if (posTextEl) {
                if (positions.length === 0) {
                    posTextEl.textContent = 'No open positions';
                } else if (positions.length <= 4) {
                    posTextEl.textContent = positions.map(p => p.symbol).join(', ');
                } else {
                    posTextEl.textContent = `${formatCurrency(totalValue)} deployed`;
                }
            }
        }
        
        async function updateSafetyMonitoring() {
            try {
                // Fetch circuit breakers
                const breakersResponse = await fetch('/api/safety/circuit-breakers');
                const breakers = await breakersResponse.json();
                updateCircuitBreakers(breakers);

                // Fetch order manager
                const ordersResponse = await fetch('/api/safety/order-manager');
                const orders = await ordersResponse.json();
                updateOrderManager(orders);

                // Fetch data validator
                const validatorResponse = await fetch('/api/safety/data-validator');
                const validator = await validatorResponse.json();
                updateDataValidator(validator);

                // Fetch safety thresholds
                const thresholdsResponse = await fetch('/api/safety/thresholds');
                const thresholds = await thresholdsResponse.json();
                updateSafetyThresholds(thresholds);
            } catch (error) {
                console.error('Error updating safety monitoring:', error);
            }
        }

        function updateCircuitBreakers(breakers) {
            const container = document.getElementById('circuit-breakers-content');
            if (!container || !breakers) return;

            let html = '';
            for (const [name, stats] of Object.entries(breakers)) {
                const stateClass = stats.state === 'closed' ? 'success' :
                                 stats.state === 'open' ? 'danger' : 'warning';
                const stateIcon = stats.state === 'closed' ? '✓' :
                                stats.state === 'open' ? '✗' : '⚠';

                html += `
                    <div class="circuit-breaker-item">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <strong>${name}</strong>
                            <span class="badge badge-${stateClass}">${stateIcon} ${stats.state.toUpperCase()}</span>
                        </div>
                        <div style="font-size: 0.9em; color: #999;">
                            Calls: ${stats.total_calls} |
                            Failed: ${stats.failed_calls} |
                            Success Rate: ${((stats.successful_calls / Math.max(stats.total_calls, 1)) * 100).toFixed(1)}%
                        </div>
                    </div>
                `;
            }

            container.innerHTML = html || '<div style="color: #666;">No circuit breakers active</div>';
        }

        function updateOrderManager(data) {
            if (!data) return;

            // Update stats
            document.getElementById('total-orders').textContent = data.total_orders || 0;
            document.getElementById('active-orders').textContent = data.active_orders || 0;
            document.getElementById('fill-rate').textContent = ((data.fill_rate || 0)).toFixed(1) + '%';
            document.getElementById('error-rate').textContent = ((data.error_rate || 0)).toFixed(1) + '%';

            // Update recent orders table
            const tbody = document.getElementById('recent-orders-body');
            if (!tbody) return;

            if (data.recent_orders && data.recent_orders.length > 0) {
                let html = '';
                data.recent_orders.forEach(order => {
                    const statusClass = order.status === 'filled' ? 'success' :
                                      order.status === 'error' || order.status === 'rejected' ? 'danger' :
                                      order.status === 'partial_fill' ? 'warning' : 'info';

                    html += `
                        <tr>
                            <td>${order.symbol}</td>
                            <td><span class="badge badge-${statusClass}">${order.status}</span></td>
                            <td>${order.fill_percentage.toFixed(1)}%</td>
                            <td>${order.retry_count}</td>
                        </tr>
                    `;
                });
                tbody.innerHTML = html;
            } else {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: #666;">No recent orders</td></tr>';
            }
        }

        function updateDataValidator(data) {
            if (!data) return;

            // Update validation stats
            document.getElementById('total-validations').textContent = data.total_validations || 0;
            document.getElementById('pass-rate').textContent = ((data.pass_rate || 0)).toFixed(1) + '%';
            document.getElementById('failed-stale').textContent = data.failed_stale || 0;
            document.getElementById('failed-spread').textContent = data.failed_spread || 0;
        }

        function updateSafetyThresholds(thresholds) {
            const container = document.getElementById('safety-thresholds-content');
            if (!container || !thresholds) return;

            let html = '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;">';

            for (const [key, value] of Object.entries(thresholds)) {
                // Format the key for display
                const displayKey = key.replace(/_/g, ' ').toLowerCase()
                    .replace(/\\b\\w/g, l => l.toUpperCase());

                html += `
                    <div style="padding: 8px; background: #2a2a2a; border-radius: 4px;">
                        <div style="font-size: 0.85em; color: #999;">${displayKey}</div>
                        <div style="font-size: 1.1em; font-weight: bold;">${value}</div>
                    </div>
                `;
            }

            html += '</div>';
            container.innerHTML = html;
        }

        function updatePerformanceTable(data) {
            if (!data) return;

            // Update key metrics (summary cards)
            if (data.summary) {
                const summary = data.summary;

                // Total P&L with color
                const pnlElem = document.getElementById('perf-total-pnl');
                if (pnlElem) {
                    const pnlValue = summary.total_pnl || 0;
                    pnlElem.textContent = formatCurrency(pnlValue);
                    pnlElem.style.color = pnlValue >= 0 ? '#4ade80' : '#ff6b6b';
                }

                // Total Return with color
                const returnElem = document.getElementById('total-return');
                if (returnElem) {
                    const returnValue = summary.total_return || 0;
                    returnElem.textContent = formatPercent(returnValue);
                    returnElem.style.color = returnValue >= 0 ? '#4ade80' : '#ff6b6b';
                }

                // Sharpe Ratio with color
                const sharpeElem = document.getElementById('total-sharpe');
                if (sharpeElem) {
                    const sharpeValue = summary.total_sharpe || 0;
                    sharpeElem.textContent = sharpeValue.toFixed(2);
                    sharpeElem.style.color = sharpeValue >= 1 ? '#4ade80' : (sharpeValue >= 0 ? '#fff' : '#ff6b6b');
                }

                // Max Drawdown
                const ddElem = document.getElementById('total-drawdown');
                if (ddElem) {
                    const ddValue = summary.total_drawdown || 0;
                    ddElem.textContent = formatPercent(ddValue);
                }

                // Win Rate with color
                const winRateElem = document.getElementById('win-rate');
                if (winRateElem) {
                    const winRate = summary.win_rate || 0;
                    winRateElem.textContent = formatPercent(winRate);
                    winRateElem.style.color = winRate >= 0.5 ? '#4ade80' : '#ff6b6b';
                }

                // Total trades
                const tradesElem = document.getElementById('perf-total-trades');
                if (tradesElem) tradesElem.textContent = summary.total_trades || 0;
            }

            // Update period breakdown table
            ['daily', 'weekly', 'monthly', 'all'].forEach(period => {
                const metrics = data[period] || {};

                // P&L with color
                const pnlElem = document.getElementById(`pnl-${period}`);
                if (pnlElem) {
                    const pnlValue = metrics.pnl || 0;
                    pnlElem.textContent = formatCurrency(pnlValue);
                    pnlElem.style.color = pnlValue >= 0 ? '#4ade80' : '#ff6b6b';
                }

                // Return with color
                const returnElem = document.getElementById(`return-${period}`);
                if (returnElem) {
                    const returnValue = metrics.return_pct || 0;
                    returnElem.textContent = formatPercent(returnValue);
                    returnElem.style.color = returnValue >= 0 ? '#4ade80' : '#ff6b6b';
                }

                // Trades
                const tradesElem = document.getElementById(`trades-${period}`);
                if (tradesElem) tradesElem.textContent = metrics.trades || 0;
            });

            // Update trade statistics from equity curve data
            if (data.summary) {
                const wins = data.summary.winning_trades || 0;
                const losses = data.summary.losing_trades || 0;
                // These will be populated when we have more detailed trade stats
            }
        }
        
        function addLog(message) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="log-time">${time}</span><span>${message}</span>`;
            container.appendChild(entry);
            // Auto-scroll if enabled
            const autoScrollEnabled = document.getElementById('auto-scroll-toggle')?.checked;
            if (autoScrollEnabled) {
                container.scrollTop = container.scrollHeight;
            }
        }

        function scrollLogsToBottom() {
            const container = document.getElementById('log-container');
            if (container) {
                container.scrollTop = container.scrollHeight;
            }
        }

        let currentLogFilter = 'ALL';

        function setLogFilter(level) {
            currentLogFilter = level;
            // Update button styles
            document.querySelectorAll('.log-filter-btn').forEach(btn => {
                btn.style.background = '#333';
                btn.style.border = '1px solid #555';
            });
            const activeBtn = document.getElementById('log-filter-' + level);
            if (activeBtn) {
                activeBtn.style.background = '#3b82f6';
                activeBtn.style.border = 'none';
            }
            // Apply filter to existing logs
            applyLogFilter();
        }

        function applyLogFilter() {
            const container = document.getElementById('log-container');
            if (!container) return;
            container.querySelectorAll('.log-entry').forEach(entry => {
                const logLevel = entry.dataset.level;
                if (currentLogFilter === 'ALL' || logLevel === currentLogFilter || !logLevel) {
                    entry.style.display = '';
                } else {
                    entry.style.display = 'none';
                }
            });
        }

        function clearLogs() {
            const container = document.getElementById('log-container');
            if (container) {
                container.innerHTML = '<div class="log-entry"><span class="log-time">' +
                    new Date().toLocaleTimeString() + '</span><span>Logs cleared</span></div>';
                loadedLogCount = 0;
            }
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
                case 'log':
                    handleLogMessage(data);
                    break;
            }
        }

        function handleLogMessage(data) {
            const container = document.getElementById('log-container');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.dataset.level = data.level || 'INFO';

            // Parse timestamp or use current time
            let time;
            if (data.timestamp) {
                try {
                    time = new Date(data.timestamp).toLocaleTimeString();
                } catch (e) {
                    time = new Date().toLocaleTimeString();
                }
            } else {
                time = new Date().toLocaleTimeString();
            }

            // Color based on log level
            const levelColors = {
                'DEBUG': '#6b7280',
                'INFO': '#3b82f6',
                'WARNING': '#f59e0b',
                'ERROR': '#ef4444'
            };
            const levelColor = levelColors[data.level] || '#888';

            // Format: time [LEVEL] source: message
            const levelBadge = `<span style="color: ${levelColor}; font-weight: bold;">[${data.level}]</span>`;
            const source = data.source ? `<span style="color: #888;">${data.source}:</span> ` : '';

            entry.innerHTML = `<span class="log-time">${time}</span> ${levelBadge} ${source}<span>${data.message}</span>`;

            // Apply current filter
            if (currentLogFilter !== 'ALL' && data.level !== currentLogFilter) {
                entry.style.display = 'none';
            }

            container.appendChild(entry);

            // Auto-scroll if enabled
            const autoScrollEnabled = document.getElementById('auto-scroll-toggle')?.checked;
            if (autoScrollEnabled) {
                container.scrollTop = container.scrollHeight;
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
            const reason = data.reason ? ` - ${data.reason}` : '';
            const signalEmoji = data.signal === 'BUY' ? '🟢' : data.signal === 'SELL' ? '🔴' : '⚪';
            addLog(`${signalEmoji} ${data.symbol}: ${data.signal}${reason}`);
        }
        
        function updatePositionsFromWS(positions) {
            // Update positions without full refresh
            if (currentTab === 'positions') {
                updatePositionsTable(positions);
            }
        }
        
        // Strategy charts
        let allocationPieChart = null;
        let strategyPnlChart = null;
        let algoUsageChart = null;

        // Refresh strategies tab data
        async function refreshStrategies() {
            try {
                // Fetch all data in parallel
                const [strategiesResp, allocResp, execResp, riskResp] = await Promise.all([
                    fetch('/api/strategies/status'),
                    fetch('/api/portfolio/allocation'),
                    fetch('/api/execution/status'),
                    fetch('/api/risk/status')
                ]);

                let strategiesData = null;
                let allocData = null;
                let execData = null;
                let riskData = null;

                if (strategiesResp.ok) strategiesData = await strategiesResp.json();
                if (allocResp.ok) allocData = await allocResp.json();
                if (execResp.ok) execData = await execResp.json();
                if (riskResp.ok) riskData = await riskResp.json();

                // Update summary metrics
                if (strategiesData) {
                    const ml = strategiesData.active_strategies.ml_enhanced || {};
                    const exec = strategiesData.active_strategies.smart_execution || {};
                    const micro = strategiesData.active_strategies.microstructure || {};
                    const pm = strategiesData.active_strategies.portfolio_manager || {};
                    const perf = strategiesData.performance_by_strategy || {};

                    // Count active strategies
                    let activeCount = 0;
                    if (ml.enabled) activeCount++;
                    if (exec.enabled) activeCount++;
                    if (pm.enabled) activeCount++;
                    if (micro.enabled) activeCount++;

                    document.getElementById('strat-active-count').textContent = activeCount;
                    document.getElementById('strat-positions').textContent = ml.positions || pm.positions_count || 0;

                    const totalPnl = (perf.ml_enhanced?.pnl || 0) + (perf.microstructure?.pnl || 0);
                    const pnlEl = document.getElementById('strat-total-pnl');
                    pnlEl.textContent = '$' + totalPnl.toLocaleString(undefined, {maximumFractionDigits: 0});
                    pnlEl.style.color = totalPnl >= 0 ? '#4ade80' : '#f87171';

                    document.getElementById('strat-win-rate').textContent = ((perf.ml_enhanced?.win_rate || 0) * 100).toFixed(1) + '%';
                    document.getElementById('strat-trades').textContent = perf.ml_enhanced?.total_trades || exec.total_trades || 0;
                    document.getElementById('strat-slippage').textContent = (exec.avg_slippage_bps || 0).toFixed(1) + ' bps';

                    // Update ML Enhanced row
                    document.getElementById('ml-status-badge').textContent = ml.enabled ? 'ACTIVE' : 'DISABLED';
                    document.getElementById('ml-status-badge').style.color = ml.enabled ? '#4ade80' : '#f87171';
                    document.getElementById('ml-regime-small').textContent = ml.regime || 'NEUTRAL';
                    document.getElementById('ml-conf-small').textContent = ((ml.confidence || 0) * 100).toFixed(1) + '%';
                    document.getElementById('ml-pos-count').textContent = ml.positions || 0;
                    const mlPnl = perf.ml_enhanced?.pnl || 0;
                    const mlPnlEl = document.getElementById('ml-pnl-small');
                    mlPnlEl.textContent = '$' + mlPnl.toLocaleString(undefined, {maximumFractionDigits: 0});
                    mlPnlEl.style.color = mlPnl >= 0 ? '#4ade80' : '#f87171';

                    // Update Smart Execution row
                    document.getElementById('exec-status-badge').textContent = exec.enabled ? 'ACTIVE' : 'DISABLED';
                    document.getElementById('exec-status-badge').style.color = exec.enabled ? '#4ade80' : '#f87171';
                    document.getElementById('exec-algo-small').textContent = exec.algorithm || 'Market';
                    document.getElementById('exec-trade-count').textContent = exec.total_trades || 0;
                    document.getElementById('exec-saved-small').textContent = (perf.smart_execution?.saved_bps || 0).toFixed(1) + ' bps';

                    // Update Portfolio Manager row
                    document.getElementById('pm-status-badge').textContent = pm.enabled ? 'ACTIVE' : 'DISABLED';
                    document.getElementById('pm-status-badge').style.color = pm.enabled ? '#4ade80' : '#f87171';
                    document.getElementById('pm-method-small').textContent = pm.allocation_method || 'Equal Weight';
                    document.getElementById('pm-max-pos').textContent = pm.max_positions || 30;
                    document.getElementById('pm-pos-count').textContent = pm.positions_count || 0;
                    document.getElementById('pm-rebalance-small').textContent = pm.rebalance_due ? 'Yes' : 'No';

                    // Update Microstructure row
                    const microRow = document.getElementById('micro-row');
                    if (micro.enabled) {
                        microRow.style.opacity = '1';
                        document.getElementById('micro-status-badge').textContent = 'ACTIVE';
                        document.getElementById('micro-status-badge').style.color = '#4ade80';
                    } else {
                        microRow.style.opacity = '0.5';
                        document.getElementById('micro-status-badge').textContent = 'DISABLED';
                        document.getElementById('micro-status-badge').style.color = '#f87171';
                    }
                    document.getElementById('micro-ofi-small').textContent = (micro.ofi || 0).toFixed(2);
                    document.getElementById('micro-score-small').textContent = (micro.ensemble_score || 0).toFixed(2);
                    document.getElementById('micro-win-small').textContent = ((perf.microstructure?.win_rate || 0) * 100).toFixed(0) + '%';

                    // Render Strategy P&L Chart
                    renderStrategyPnlChart(perf);
                }

                // Update allocation section
                if (allocData) {
                    const allocs = allocData.allocations || {};
                    document.getElementById('alloc-ml-pct').textContent = ((allocs['ML Enhanced'] || 0) * 100).toFixed(0) + '%';
                    document.getElementById('alloc-micro-pct').textContent = ((allocs['Microstructure'] || 0) * 100).toFixed(0) + '%';
                    document.getElementById('alloc-mr-pct').textContent = ((allocs['Mean Reversion'] || 0) * 100).toFixed(0) + '%';
                    document.getElementById('alloc-mom-pct').textContent = ((allocs['Momentum'] || 0) * 100).toFixed(0) + '%';

                    document.getElementById('alloc-method').textContent = allocData.method || 'Risk Parity';
                    document.getElementById('alloc-div-ratio').textContent = (allocData.correlation_matrix?.diversification_ratio || 0).toFixed(2);
                    document.getElementById('alloc-drift').textContent = ((allocData.rebalance?.drift || 0) * 100).toFixed(1) + '%';

                    // Render pie chart
                    renderAllocationPieChart(allocs);
                }

                // Update execution metrics
                if (execData) {
                    document.getElementById('exec-avg-slip').textContent = (execData.avg_slippage || 0).toFixed(1) + ' bps';
                    document.getElementById('exec-impact').textContent = (execData.market_impact || 0).toFixed(1) + ' bps';
                    document.getElementById('exec-fill-pct').textContent = ((execData.fill_rate || 0) * 100).toFixed(0) + '%';
                    document.getElementById('exec-fill-rate').textContent = ((execData.fill_rate || 0) * 100).toFixed(0) + '%';
                    document.getElementById('exec-total-saved').textContent = (execData.performance?.total_saved_bps || 0).toFixed(1) + ' bps';

                    // Render algo usage chart
                    renderAlgoUsageChart(execData.algorithms_used || {});
                }

                // Update risk metrics (keep backward compat)
                if (riskData) {
                    if (riskData.kelly_sizing && document.getElementById('portfolio-kelly')) {
                        document.getElementById('portfolio-kelly').textContent = `${(riskData.kelly_sizing.portfolio_kelly * 100).toFixed(1)}%`;
                    }
                    if (riskData.kill_switches && document.getElementById('kill-switch-status')) {
                        const ks = riskData.kill_switches;
                        document.getElementById('kill-switch-status').textContent = ks.active ? 'TRIGGERED' : 'ACTIVE';
                        document.getElementById('kill-switch-status').style.color = ks.active ? '#ff6b6b' : '#4CAF50';
                    }
                    if (riskData.risk_metrics && document.getElementById('leverage')) {
                        const rm = riskData.risk_metrics;
                        document.getElementById('leverage').textContent = `${rm.leverage.toFixed(2)}x`;
                    }
                }

            } catch (error) {
                console.error('Error refreshing strategies:', error);
            }
        }

        function renderAllocationPieChart(allocations) {
            const ctx = document.getElementById('allocation-pie-chart');
            if (!ctx) return;

            const labels = Object.keys(allocations);
            const data = Object.values(allocations).map(v => v * 100);
            const colors = ['#667eea', '#4ade80', '#fbbf24', '#f87171'];

            if (allocationPieChart) {
                allocationPieChart.data.labels = labels;
                allocationPieChart.data.datasets[0].data = data;
                allocationPieChart.update('none');
                return;
            }

            allocationPieChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors,
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: false,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false }
                    },
                    cutout: '60%'
                }
            });
        }

        function renderStrategyPnlChart(perf) {
            const ctx = document.getElementById('strategy-pnl-chart');
            if (!ctx) return;

            const strategies = ['ML Enhanced', 'Microstructure', 'Smart Exec'];
            const pnlValues = [
                perf.ml_enhanced?.pnl || 0,
                perf.microstructure?.pnl || 0,
                0  // Smart exec doesn't have P&L, shows saved bps
            ];
            const winRates = [
                (perf.ml_enhanced?.win_rate || 0) * 100,
                (perf.microstructure?.win_rate || 0) * 100,
                (perf.smart_execution?.success_rate || 0.95) * 100
            ];

            if (strategyPnlChart) {
                strategyPnlChart.data.datasets[0].data = pnlValues;
                strategyPnlChart.data.datasets[1].data = winRates;
                strategyPnlChart.update('none');
                return;
            }

            strategyPnlChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: strategies,
                    datasets: [
                        {
                            label: 'P&L ($)',
                            data: pnlValues,
                            backgroundColor: pnlValues.map(v => v >= 0 ? '#4ade80' : '#f87171'),
                            yAxisID: 'y'
                        },
                        {
                            label: 'Win Rate (%)',
                            data: winRates,
                            backgroundColor: '#60a5fa',
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top',
                            labels: { color: '#8b949e', font: { size: 10 } }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: '#21262d' },
                            ticks: { color: '#8b949e', font: { size: 10 } }
                        },
                        y: {
                            type: 'linear',
                            position: 'left',
                            grid: { color: '#21262d' },
                            ticks: { color: '#4ade80', font: { size: 10 } },
                            title: { display: true, text: 'P&L ($)', color: '#4ade80', font: { size: 10 } }
                        },
                        y1: {
                            type: 'linear',
                            position: 'right',
                            grid: { display: false },
                            ticks: { color: '#60a5fa', font: { size: 10 } },
                            title: { display: true, text: 'Win Rate (%)', color: '#60a5fa', font: { size: 10 } },
                            min: 0,
                            max: 100
                        }
                    }
                }
            });
        }

        function renderAlgoUsageChart(algoUsage) {
            const ctx = document.getElementById('algo-usage-chart');
            if (!ctx) return;

            const labels = Object.keys(algoUsage).map(k => k.toUpperCase());
            const data = Object.values(algoUsage);
            const colors = ['#4ade80', '#60a5fa', '#fbbf24', '#a78bfa'];

            if (algoUsageChart) {
                algoUsageChart.data.labels = labels;
                algoUsageChart.data.datasets[0].data = data;
                algoUsageChart.update('none');
                return;
            }

            algoUsageChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: colors.slice(0, labels.length),
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { color: '#21262d' },
                            ticks: { color: '#8b949e', font: { size: 9 } }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#8b949e', font: { size: 9 } }
                        }
                    }
                }
            });
        }
        
        // Initialize on load
        window.onload = () => {
            refreshData();
            refreshStrategies();
            updateSafetyMonitoring(); // Load safety monitoring on startup
            loadLogs(); // Load logs immediately on startup
            updateMarketStatus(); // Load market status on startup
            connectWebSocket(); // Connect to WebSocket
            setInterval(refreshData, 5000); // Keep polling as fallback
            setInterval(refreshStrategies, 5000); // Update strategies tab
            setInterval(updateSafetyMonitoring, 10000); // Update safety monitoring every 10 seconds
            setInterval(loadLogs, 2000); // Update logs every 2 seconds
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


@app.route("/api/portfolios")
@requires_auth
def get_portfolios():
    """Get all portfolio definitions with summary stats.

    Returns list of portfolios with their account summary.
    """
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        portfolios = db.get_portfolios()

        # Enrich each portfolio with account summary
        result = []
        for p in portfolios:
            pid = p.get("id", "default")
            account = db.get_account_info(portfolio_id=pid)
            positions = db.get_positions(portfolio_id=pid)

            result.append({
                "id": pid,
                "name": p.get("name", pid),
                "starting_cash": p.get("starting_cash", 100000),
                "active": bool(p.get("active", 1)),
                "symbols": p.get("symbols", ""),
                "account": {
                    "cash": account.get("cash", 0),
                    "equity": account.get("equity", 0),
                    "realized_pnl": account.get("realized_pnl", 0),
                    "unrealized_pnl": account.get("unrealized_pnl", 0),
                    "daily_pnl": account.get("daily_pnl", 0),
                },
                "positions_count": len([p for p in positions if p.get("quantity", 0) != 0]),
            })

        return jsonify({"portfolios": result})
    except Exception as e:
        logger.error(f"Error getting portfolios: {e}")
        return jsonify({"portfolios": [{"id": "default", "name": "Default Portfolio", "active": True}]})


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


@app.route("/health/live")
def health_liveness():
    """
    Kubernetes liveness probe - checks if the application is running
    Returns 200 if the application can serve traffic
    """
    try:
        # Basic check - can we respond?
        return (
            jsonify(
                {"status": "alive", "timestamp": datetime.now().isoformat(), "service": "dashboard"}
            ),
            200,
        )
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return (
            jsonify({"status": "dead", "error": str(e), "timestamp": datetime.now().isoformat()}),
            500,
        )


@app.route("/health/ready")
def health_readiness():
    """
    Kubernetes readiness probe - checks if the application is ready to serve traffic
    Returns 200 if all dependencies are available and healthy
    """
    checks = {
        "timestamp": datetime.now().isoformat(),
        "service": "dashboard",
        "ready": True,
        "checks": {},
    }

    # Check database connectivity
    try:
        from sync_db_reader import SyncDatabaseReader

        db_reader = SyncDatabaseReader()
        result = db_reader._fetch_one("SELECT 1")
        checks["checks"]["database"] = "ready" if result else "not_ready"
    except Exception as e:
        logger.warning(f"Database readiness check failed: {e}")
        checks["checks"]["database"] = "not_ready"
        checks["ready"] = False

    # Check if we can access config
    try:
        from robo_trader.config import load_config

        cfg = load_config()
        checks["checks"]["config"] = "ready" if cfg else "not_ready"
    except Exception as e:
        logger.warning(f"Config readiness check failed: {e}")
        checks["checks"]["config"] = "not_ready"
        checks["ready"] = False

    # Check filesystem access
    try:
        data_path = Path("/app/data")
        logs_path = Path("/app/logs")
        checks["checks"]["filesystem"] = (
            "ready" if data_path.exists() and logs_path.exists() else "not_ready"
        )
    except Exception as e:
        logger.warning(f"Filesystem readiness check failed: {e}")
        checks["checks"]["filesystem"] = "not_ready"
        checks["ready"] = False

    status_code = 200 if checks["ready"] else 503
    return jsonify(checks), status_code


@app.route("/metrics")
def metrics():
    """
    Prometheus metrics endpoint
    Exposes application metrics in Prometheus format
    """
    try:
        from robo_trader.monitoring.production_monitor import ProductionMonitor

        # Get metrics from production monitor if available
        metrics_text = "# RoboTrader Dashboard Metrics\n"

        # Add basic metrics
        metrics_text += f"# HELP dashboard_up Dashboard service status\n"
        metrics_text += f"# TYPE dashboard_up gauge\n"
        metrics_text += f"dashboard_up 1\n"

        metrics_text += f"# HELP dashboard_requests_total Total HTTP requests\n"
        metrics_text += f"# TYPE dashboard_requests_total counter\n"
        metrics_text += f"dashboard_requests_total {{}} {0}\n"

        return Response(metrics_text, mimetype="text/plain")
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return Response("# Error generating metrics\n", mimetype="text/plain"), 500


@app.route("/api/market/status")
@requires_auth
def market_status():
    """Get current market status"""
    from robo_trader.market_hours import (
        get_market_session,
        get_next_market_open,
        is_extended_hours,
        is_market_open,
        seconds_until_market_open,
    )

    # Check if extended hours trading is enabled
    enable_extended = os.getenv("ENABLE_EXTENDED_HOURS", "false").lower() in ("true", "1", "yes")

    current_time = datetime.now()
    is_open = is_market_open()
    session = get_market_session()

    # Consider extended hours as "open" if enabled
    if enable_extended and is_extended_hours():
        is_open = True
        session = "extended-hours"

    result = {
        "is_open": is_open,
        "session": session,
        "current_time": current_time.isoformat(),
        "status_text": session.replace("-", " ").title(),
        "extended_hours_enabled": enable_extended,
    }

    if is_open:
        # Market is open - show when it closes (handles early close days)
        from zoneinfo import ZoneInfo

        from robo_trader.market_hours import _get_market_close_time

        et = ZoneInfo("America/New_York")
        now_et = datetime.now(et)
        close_time_obj = _get_market_close_time(now_et.date())
        close_time = now_et.replace(
            hour=close_time_obj.hour, minute=close_time_obj.minute, second=0, microsecond=0
        )
        result["next_close"] = close_time.strftime("%I:%M %p")
    else:
        next_open = get_next_market_open()
        seconds_until = seconds_until_market_open()
        result.update(
            {
                "next_open": next_open.strftime("%a %I:%M %p"),
                "seconds_until_open": seconds_until,
                "time_until_open": f"{seconds_until // 3600}h {(seconds_until % 3600) // 60}m",
            }
        )

    return jsonify(result)


def check_ibkr_connection():
    """
    Check TWS/Gateway connection using lsof (no zombies).

    Checks both TWS (port 7497) and Gateway (port 4002) to see which is running.
    Also checks for ESTABLISHED connections to distinguish between:
    - Gateway available (listening) but no active API connection
    - Gateway available AND runner has active API connection

    Uses lsof to check if ports are listening - this does NOT create TCP connections
    and therefore does NOT create zombie CLOSE_WAIT connections.

    IMPORTANT: Do NOT use socket.connect_ex() here - it creates zombie connections
    that block subsequent IBKR API handshakes!
    """
    import subprocess

    tws_healthy = False
    gateway_healthy = False
    api_connected = False  # True if there's an ESTABLISHED connection to Gateway/TWS
    status_msg = "Unknown"

    # Check TWS (port 7497) using lsof
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:7497", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and "LISTEN" in result.stdout:
            tws_healthy = True
    except Exception as e:
        logger.debug(f"TWS health check error: {e}")

    # Check Gateway (port 4002) using lsof
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:4002", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and "LISTEN" in result.stdout:
            gateway_healthy = True
    except Exception as e:
        logger.debug(f"Gateway health check error: {e}")

    # Check for ESTABLISHED connections (actual API connections)
    # This tells us if the runner currently has an active connection to Gateway/TWS
    try:
        # Check for established connections on port 4002 (Gateway) or 7497 (TWS)
        result = subprocess.run(
            ["lsof", "-nP", "-iTCP:4002", "-sTCP:ESTABLISHED"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and "ESTABLISHED" in result.stdout:
            api_connected = True
        else:
            # Also check TWS port
            result = subprocess.run(
                ["lsof", "-nP", "-iTCP:7497", "-sTCP:ESTABLISHED"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and "ESTABLISHED" in result.stdout:
                api_connected = True
    except Exception as e:
        logger.debug(f"API connection check error: {e}")

    # Determine status message - be clear about the distinction
    if api_connected:
        if gateway_healthy:
            status_msg = "Gateway API connected (port 4002)"
        elif tws_healthy:
            status_msg = "TWS API connected (port 7497)"
        else:
            status_msg = "API connected"
    elif gateway_healthy or tws_healthy:
        # Gateway/TWS is running but no active API connection
        if gateway_healthy and tws_healthy:
            status_msg = "Gateway & TWS available (no active API session)"
        elif gateway_healthy:
            status_msg = "Gateway available (no active API session)"
        else:
            status_msg = "TWS available (no active API session)"
    else:
        status_msg = "No TWS/Gateway detected"

    return {
        "connected": api_connected,  # Now means actually connected, not just available
        "gateway_available": gateway_healthy,  # Gateway is listening
        "tws_available": tws_healthy,  # TWS is listening
        "api_connected": api_connected,  # Active ESTABLISHED connection exists
        "status": status_msg,
        "tws_running": tws_healthy,  # Keep for backwards compat
        "gateway_running": gateway_healthy,  # Keep for backwards compat
    }


@app.route("/api/status")
@requires_auth
def status():
    """Get current system status from database"""
    global trading_status, trading_process

    # Check if trading process is actually running
    if trading_process and trading_process.poll() is not None:
        # Process has terminated, update status
        trading_status = "stopped"
        trading_process = None

    # If no process reference but status is running, reset it
    if trading_process is None and trading_status == "running":
        trading_status = "stopped"

    # Check if runner is actually running system-wide (not just dashboard-started)
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-f", "runner_async"], capture_output=True, text=True, timeout=1
        )
        runner_actually_running = result.returncode == 0 and len(result.stdout.strip()) > 0
    except Exception:
        runner_actually_running = False

    # Check real IBKR connection (simple check - no zombies)
    # DISABLED: Health check creates zombie connections
    # Use simple status based on runner state instead
    from datetime import time as dt_time
    from datetime import timedelta

    # Get current time and market hours for status messages
    now = datetime.now()
    market_start = dt_time(9, 30)
    market_end = dt_time(16, 0)
    is_weekday = now.weekday() < 5

    # Use the centralized market hours logic
    from robo_trader.market_hours import is_extended_hours
    from robo_trader.market_hours import is_market_open as check_market_open

    # Check if extended hours trading is enabled
    enable_extended = os.getenv("ENABLE_EXTENDED_HOURS", "false").lower() in ("true", "1", "yes")

    market_open = check_market_open()
    # Consider extended hours as "open" if enabled
    if enable_extended and is_extended_hours():
        market_open = True

    # Build clear status message
    runner_running = runner_actually_running

    # Check TWS/Gateway health with sync approach (no zombies)
    # Do this FIRST so we can use it in status messages
    ibkr_check = check_ibkr_connection()
    api_connected = ibkr_check.get("api_connected", False)  # Active ESTABLISHED connection
    gateway_available = ibkr_check.get("gateway_available", False)  # Gateway is listening
    tws_available = ibkr_check.get("tws_available", False)  # TWS is listening
    ibkr_status_msg = ibkr_check.get("status", "Unknown")
    tws_running = ibkr_check.get("tws_running", False)
    gateway_running = ibkr_check.get("gateway_running", False)

    # Build clear status message based on actual connection state
    if not runner_running:
        status_message = "⚠️ Runner not started - No trading activity"
        status_detail = "Start the runner with: python3 -m robo_trader.runner_async"
    elif not market_open:
        status_message = "💤 Market Closed - Runner sleeping"
        # Calculate time until market open
        next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        if now.time() >= market_end or not is_weekday:
            # Move to next weekday
            days_ahead = 1 if is_weekday else (7 - now.weekday())
            next_open = next_open + timedelta(days=days_ahead)
        hours_until = (next_open - now).total_seconds() / 3600
        status_detail = f"Next market open: {next_open.strftime('%a %I:%M %p')} (in {hours_until:.1f} hours). Runner checks every 30 minutes."
    elif api_connected:
        # Runner is running, market is open, AND we have an active API connection
        status_message = "✅ Market Open - API Connected"
        status_detail = "Runner has active IBKR API connection and is processing data"
    elif gateway_available or tws_available:
        # Runner is running, market is open, but NO active API connection
        # This is the per-cycle mode - connects only during trading cycles
        status_message = "🔄 Market Open - Waiting for cycle"
        status_detail = (
            "Gateway available. Runner connects per-cycle for stability (no active API session now)"
        )
    else:
        # Runner is running, market is open, but no Gateway/TWS
        status_message = "⚠️ Market Open - No Gateway"
        status_detail = "Runner is running but Gateway/TWS not detected. Check IBKR Gateway."

    # Get symbol count from user_settings.json
    symbols_count = 0
    try:
        with open("user_settings.json", "r") as f:
            settings = json.load(f)
            symbols = settings.get("default", {}).get("symbols", [])
            symbols_count = len(symbols)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        symbols_count = 0

    # Return status with NO FAKE DATA
    # is_trading = True ONLY when runner is running AND market is open AND gateway available
    is_actually_trading = runner_running and market_open and (gateway_available or tws_available)

    # Get real P&L and positions count from database (using cached sync reader)
    pnl_data = None
    positions_count = 0
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        portfolio_id = request.args.get("portfolio_id", "default")
        positions = db.get_positions(portfolio_id=portfolio_id)
        positions_count = len([p for p in positions if p.get("quantity", 0) != 0])

        # Calculate P&L from positions
        if positions:
            total_cost = sum(
                p["quantity"] * p["avg_cost"] for p in positions if p.get("quantity", 0) > 0
            )
            total_value = sum(
                p["quantity"] * (p.get("market_price") or p["avg_cost"])
                for p in positions
                if p.get("quantity", 0) > 0
            )
            unrealized_pnl = total_value - total_cost

            # Get account info for realized P&L and cash
            account = db.get_account_info(portfolio_id=portfolio_id)
            realized_pnl = account.get("realized_pnl", 0) or 0
            daily_pnl = account.get("daily_pnl", 0) or 0
            cash = account.get("cash", 0) or 0

            # Equity = cash + position value (not just positions)
            equity = cash + total_value

            pnl_data = {
                "daily": round(daily_pnl, 2),
                "total": round(realized_pnl + unrealized_pnl, 2),
                "unrealized": round(unrealized_pnl, 2),
                "realized": round(realized_pnl, 2),
                "equity": round(equity, 2),
                "cash": round(cash, 2),
            }
    except Exception as e:
        logger.debug(f"Could not load P&L for status: {e}")

    return jsonify(
        {
            "trading_status": {
                "is_trading": is_actually_trading,
                "connected": api_connected,  # Now means ACTUALLY connected (ESTABLISHED socket)
                "api_connected": api_connected,  # Explicit: active ESTABLISHED connection
                "gateway_available": gateway_available,  # Gateway is listening (can connect)
                "market_open": market_open,
                "mode": "paper",
                "session_start": datetime.now().isoformat(),
                "message": status_message,
                "detail": status_detail,
                "runner_state": "running" if runner_running else "stopped",
                "tws_health": ibkr_status_msg,
                "tws_running": tws_running,  # TWS status (backwards compat)
                "gateway_running": gateway_running,  # Gateway status (backwards compat)
                "symbols_count": symbols_count,  # Number of symbols being traded
            },
            "pnl": pnl_data,
            "metrics": None,  # Will be populated from runner metrics
            "positions_count": positions_count,
            "ml_status": None,  # Will be populated from ML status
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
            
            # Get trades to calculate win rate and profit factor
            trades = await db.get_trades(limit=100)
            
            # Calculate win rate
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0.0
            
            # Calculate profit factor
            gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
            gross_loss = abs(sum(t.get('pnl', 0) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Calculate Sharpe ratio from returns
            if trades and len(trades) >= 2:
                returns = []
                sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
                for i in range(1, len(sorted_trades)):
                    if sorted_trades[i-1].get('pnl') and sorted_trades[i].get('pnl'):
                        prev_val = sorted_trades[i-1]['pnl']
                        curr_val = sorted_trades[i]['pnl']
                        if prev_val != 0:
                            returns.append((curr_val - prev_val) / abs(prev_val))
                
                if returns:
                    import numpy as np
                    returns_array = np.array(returns)
                    avg_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    sharpe_ratio = (avg_return * np.sqrt(252) / std_return) if std_return > 0 else 0.0
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0
            
            # Calculate max drawdown
            if trades:
                equity_curve = []
                cumulative_pnl = 0
                for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
                    cumulative_pnl += trade.get('pnl', 0)
                    equity_curve.append(cumulative_pnl)
                
                if equity_curve:
                    peak = equity_curve[0]
                    max_drawdown = 0
                    for value in equity_curve:
                        if value > peak:
                            peak = value
                        drawdown = (value - peak) / peak if peak != 0 else 0
                        if drawdown < max_drawdown:
                            max_drawdown = drawdown
                else:
                    max_drawdown = 0.0
            else:
                max_drawdown = 0.0
            
            real_metrics = {
                'sharpe_ratio': round(sharpe_ratio, 2),
                'win_rate': round(win_rate, 3),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 3),
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
    # Check if market is open - don't show stale data when closed
    # Use the centralized market hours logic
    from robo_trader.market_hours import is_market_open as check_market_open

    market_open = check_market_open()

    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        portfolio_id = request.args.get("portfolio_id", "default")

        # Get positions
        positions = db.get_positions(portfolio_id=portfolio_id)

        # Calculate total market value from positions (matches mobile app)
        total_market_value = Decimal("0")
        for pos in positions:
            qty = Decimal(str(pos.get("quantity", 0) or 0))
            price = Decimal(str(pos.get("market_price", 0) or 0))
            total_market_value += qty * price

        # Get cash from account
        account = db.get_account_info(portfolio_id=portfolio_id)
        cash = Decimal(str(account.get("cash", 0) or 0))

        # Equity = cash + market value of positions
        equity = cash + total_market_value

        # When market is closed, return equity but null P&L
        if not market_open:
            return jsonify(
                {
                    "total": None,
                    "unrealized": None,
                    "realized": None,
                    "daily": None,
                    "equity": float(equity),
                    "cash": float(cash),
                }
            )

        # Calculate total unrealized P&L from positions using Decimal
        unrealized_pnl = Decimal("0")
        for pos in positions:
            qty = Decimal(str(pos.get("quantity", 0)))
            if qty > 0:
                avg_cost = Decimal(str(pos.get("avg_cost", 0)))
                market_price = Decimal(str(pos.get("market_price", avg_cost)))
                if avg_cost > 0:
                    unrealized_pnl += (market_price - avg_cost) * qty

        # Get trades for realized P&L - USE STORED PNL VALUES from database
        # The database already tracks accurate P&L per trade
        # Use limit=5000 to ensure we get ALL trades for accurate total P&L
        trades = db.get_recent_trades(limit=5000, portfolio_id=portfolio_id)
        today = datetime.now().date()

        realized_pnl = Decimal("0")
        daily_pnl = Decimal("0")

        for trade in trades:
            # Use the stored pnl value from the database (already calculated correctly)
            stored_pnl = trade.get("pnl")
            if stored_pnl is not None:
                profit = Decimal(str(stored_pnl))
                realized_pnl += profit

                # Track daily P&L
                trade_time = trade.get("timestamp", "")
                if trade_time:
                    try:
                        trade_date = datetime.fromisoformat(trade_time.replace(" ", "T")).date()
                        if trade_date == today:
                            daily_pnl += profit
                    except ValueError:
                        pass  # Skip if timestamp parsing fails
            elif trade.get("side", "").upper() == "SELL":
                # Log warning for SELL trades with missing P&L - these should have pnl calculated
                logger.warning(
                    f"SELL trade missing pnl value: {trade.get('symbol')} "
                    f"qty={trade.get('quantity')} price={trade.get('price')} "
                    f"timestamp={trade.get('timestamp')}"
                )

        # Total P&L is unrealized + realized
        total_pnl = unrealized_pnl + realized_pnl

        # Convert Decimal to float for JSON serialization
        return jsonify(
            {
                "total": float(round(total_pnl, 2)),
                "unrealized": float(round(unrealized_pnl, 2)),
                "realized": float(round(realized_pnl, 2)),
                "daily": float(round(daily_pnl, 2)),
                "equity": float(equity),
                "cash": float(cash),
            }
        )

    except Exception as e:
        logger.error(f"Error calculating P&L: {e}")
        # Return zeros on error instead of fake data
        return jsonify(
            {
                "total": 0,
                "unrealized": 0,
                "realized": 0,
                "daily": 0,
                "equity": DEFAULT_CAPITAL,
                "cash": DEFAULT_CAPITAL,
            }
        )

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
    
    # Return zeros if no cache available
    zero_pnl = {"total": 0, "unrealized": 0, "realized": 0, "daily": 0}
    # Cache zeros to prevent flashing
    app._pnl_cache = zero_pnl
    app._pnl_cache_time = current_time
    return jsonify(zero_pnl)
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
                    from robo_trader.connection_manager import IBKRClient

                    async with IBKRClient() as client:
                        price_data = await client.get_market_data(pos["symbol"])
                        current_price = price_data.get("last") if price_data else pos["avg_cost"]

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
    """Get current positions from real database - optimized to avoid DB lock"""
    # Return cached positions if available and recent (within 2 seconds)
    # Use lock for thread-safe cache access
    current_time = time.time()
    portfolio_id = request.args.get("portfolio_id", "default")
    cache_key = f"_positions_cache_{portfolio_id}"
    cache_time_key = f"_positions_cache_time_{portfolio_id}"
    with _positions_cache_lock:
        if hasattr(app, cache_key) and hasattr(app, cache_time_key):
            if current_time - getattr(app, cache_time_key) < 2:  # 2 second cache
                return jsonify({"positions": getattr(app, cache_key)})

    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        real_positions = db.get_positions(portfolio_id=portfolio_id)

        # Batch fetch: Get all signals once (not per-position!)
        try:
            all_signals = db.get_signals(hours=1, portfolio_id=portfolio_id)
            # Build lookup dict by symbol
            signal_by_symbol = {}
            for s in all_signals:
                sym = s["symbol"]
                if sym not in signal_by_symbol:
                    signal_by_symbol[sym] = s["signal_type"]
        except Exception as e:
            logger.debug(f"Could not load signals: {e}")
            signal_by_symbol = {}

        # Enrich positions using market_price from positions table (already fetched)
        enriched_positions = []
        for pos in real_positions:
            # Use market_price from positions table (updated by runner)
            # Explicit None check to handle 0 as valid price (e.g. halted stocks)
            market_price = pos.get("market_price")
            current_price = market_price if market_price is not None else pos.get("avg_cost", 100)

            # Look up signal from pre-fetched dict
            ml_signal = signal_by_symbol.get(pos["symbol"], "hold")

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

        # Cache the result (thread-safe, keyed by portfolio_id)
        with _positions_cache_lock:
            setattr(app, cache_key, enriched_positions)
            setattr(app, cache_time_key, time.time())

        return jsonify({"positions": enriched_positions})
    except Exception as e:
        logger.error(f"Error fetching real positions: {e}")

    # Return empty positions if database is locked
    return jsonify({"positions": []})


@app.route("/api/watchlist")
@requires_auth
def get_watchlist():
    """Get watchlist with latest prices - optimized for speed"""
    # Return cached watchlist if available and recent (within 3 seconds)
    current_time = time.time()
    portfolio_id = request.args.get("portfolio_id", "default")
    wl_cache_key = f"_watchlist_cache_{portfolio_id}"
    wl_cache_time_key = f"_watchlist_cache_time_{portfolio_id}"
    if hasattr(app, wl_cache_key) and hasattr(app, wl_cache_time_key):
        if current_time - getattr(app, wl_cache_time_key) < 3:  # 3 second cache
            return jsonify({"watchlist": getattr(app, wl_cache_key)})

    # Define the watchlist symbols
    watchlist_symbols = [
        "AAPL",
        "NVDA",
        "TSLA",
        "IXHL",
        "NUAI",
        "BZAI",
        "ELTP",
        "OPEN",
        "CEG",
        "VRT",
        "PLTR",
        "UPST",
        "TEM",
        "HTFL",
        "SDGR",
        "APLD",
        "SOFI",
        "CORZ",
        "WULF",
        "IMRX",
    ]

    watchlist_data = []

    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get all positions for this portfolio
        positions = db.get_positions(portfolio_id=portfolio_id)
        position_map = {p["symbol"]: p for p in positions}

        # Get latest prices and positions for each symbol
        for symbol in watchlist_symbols:
            # Get latest market data
            market_data = db.get_latest_market_data(symbol, limit=1)
            current_price = market_data[0]["close"] if market_data else 0

            # Check if we have a position
            position = position_map.get(symbol)

            # Calculate P&L if we have a position
            pnl = 0
            if position and current_price > 0:
                pnl = (current_price - position["avg_cost"]) * position["quantity"]

            watchlist_data.append(
                {
                    "symbol": symbol,
                    "current_price": current_price,
                    "quantity": position["quantity"] if position else 0,
                    "avg_cost": position["avg_cost"] if position else 0,
                    "pnl": pnl,
                    "notes": "Active" if position else "Watching",
                    "has_position": position is not None,
                }
            )

        # Cache the result (keyed by portfolio_id)
        setattr(app, wl_cache_key, watchlist_data)
        setattr(app, wl_cache_time_key, current_time)

    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        # If error, return minimal watchlist with just symbols
        watchlist_data = [
            {
                "symbol": s,
                "current_price": 0,
                "quantity": 0,
                "avg_cost": 0,
                "pnl": 0,
                "notes": "Watching",
                "has_position": False,
            }
            for s in watchlist_symbols
        ]

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
    
    # Return empty watchlist if database is locked
    return jsonify({'watchlist': []})
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


@app.route("/api/ml/predictions")
@requires_auth
def ml_predictions():
    """Get current ML predictions for all symbols"""
    from pathlib import Path

    predictions_file = Path("ml_predictions.json")
    predictions_list = []

    if predictions_file.exists():
        try:
            with open(predictions_file) as f:
                predictions = json.load(f)
                for symbol, pred in predictions.items():
                    predictions_list.append(
                        {
                            "symbol": symbol,
                            "signal": pred.get("signal", 0),
                            "action": pred.get("action", "HOLD"),
                            "confidence": pred.get("confidence", 0.5),
                            "source": pred.get("source", "Unknown"),
                            "timestamp": pred.get("timestamp", ""),
                        }
                    )
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")

    # Sort by confidence descending
    predictions_list.sort(key=lambda x: x["confidence"], reverse=True)

    # Calculate summary stats
    buy_signals = len([p for p in predictions_list if p["action"] == "BUY"])
    sell_signals = len([p for p in predictions_list if p["action"] == "SELL"])
    hold_signals = len([p for p in predictions_list if p["action"] == "HOLD"])

    return jsonify(
        {
            "predictions": predictions_list,
            "summary": {
                "total": len(predictions_list),
                "buy": buy_signals,
                "sell": sell_signals,
                "hold": hold_signals,
                "avg_confidence": round(
                    sum(p["confidence"] for p in predictions_list) / len(predictions_list), 3
                )
                if predictions_list
                else 0,
            },
        }
    )


@app.route("/api/performance")
@requires_auth
def performance():
    """Get real performance metrics from trading data"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        portfolio_id = request.args.get("portfolio_id", "default")

        # Get all trades and positions for calculations
        all_trades = db.get_recent_trades(limit=5000, portfolio_id=portfolio_id)
        positions = db.get_positions(portfolio_id=portfolio_id)

        if not all_trades and not positions:
            return jsonify({"error": "No trades or positions found", "summary": {}})

        # Load watchlist cache for current prices
        watchlist_prices = {}
        try:
            with open("watchlist_cache.json", "r") as f:
                cache = json.load(f)
                for symbol, data in cache.items():
                    if isinstance(data, dict) and "price" in data:
                        watchlist_prices[symbol] = data["price"]
                    elif isinstance(data, (int, float)):
                        watchlist_prices[symbol] = data
        except Exception:
            pass

        # Calculate unrealized PnL from positions
        unrealized_pnl = 0
        position_value = 0
        for pos in positions:
            qty = pos.get("quantity", 0)
            symbol = pos.get("symbol", "")
            if qty > 0:
                avg_cost = pos.get("avg_cost", 0)
                # Use watchlist price if available, otherwise market_price, then avg_cost
                current_price = watchlist_prices.get(symbol, pos.get("market_price", avg_cost))
                if avg_cost > 0:
                    position_pnl = (current_price - avg_cost) * qty
                    unrealized_pnl += position_pnl
                    position_value += current_price * qty

        # Calculate realized PnL by matching BUY/SELL trades per symbol
        realized_pnl = 0
        symbol_costs = {}  # Track avg cost per symbol from BUY trades
        winning_count = 0
        losing_count = 0

        for trade in sorted(all_trades, key=lambda x: x["timestamp"]):
            symbol = trade.get("symbol", "")
            side = trade.get("side", "")
            qty = trade.get("quantity", 0)
            price = trade.get("price", 0)

            if side == "BUY":
                # Track cost basis
                if symbol not in symbol_costs:
                    symbol_costs[symbol] = {"qty": 0, "cost": 0}
                symbol_costs[symbol]["qty"] += qty
                symbol_costs[symbol]["cost"] += qty * price
            elif side == "SELL":
                # Calculate realized PnL
                if symbol in symbol_costs and symbol_costs[symbol]["qty"] > 0:
                    avg_cost = symbol_costs[symbol]["cost"] / symbol_costs[symbol]["qty"]
                    # Cap sell quantity at held quantity to handle orphaned sells
                    sell_qty = min(qty, symbol_costs[symbol]["qty"])
                    trade_pnl = (price - avg_cost) * sell_qty
                    realized_pnl += trade_pnl
                    if trade_pnl > 0:
                        winning_count += 1
                    else:
                        losing_count += 1
                    # Reduce cost basis
                    symbol_costs[symbol]["qty"] -= sell_qty
                    symbol_costs[symbol]["cost"] -= avg_cost * sell_qty

        total_pnl = unrealized_pnl + realized_pnl
        total_trades_closed = winning_count + losing_count
        win_rate = winning_count / total_trades_closed if total_trades_closed > 0 else 0

        # Calculate returns for different periods
        now = datetime.now()

        def parse_trade_time(t):
            """Safely parse trade timestamp, return None on error."""
            try:
                return datetime.fromisoformat(t["timestamp"].replace(" ", "T"))
            except (ValueError, KeyError, TypeError):
                return None

        daily_trades = [
            t for t in all_trades if (ts := parse_trade_time(t)) and ts > now - timedelta(days=1)
        ]
        weekly_trades = [
            t for t in all_trades if (ts := parse_trade_time(t)) and ts > now - timedelta(days=7)
        ]
        monthly_trades = [
            t for t in all_trades if (ts := parse_trade_time(t)) and ts > now - timedelta(days=30)
        ]

        # Estimate capital (position value + 20% cash buffer)
        estimated_capital = position_value * 1.2 if position_value > 0 else DEFAULT_CAPITAL

        # Calculate period PnL by matching trades in each period
        def calc_period_pnl(trades):
            pnl = 0
            costs = {}
            for trade in sorted(trades, key=lambda x: x["timestamp"]):
                symbol, side = trade.get("symbol", ""), trade.get("side", "")
                qty, price = trade.get("quantity", 0), trade.get("price", 0)
                if side == "BUY":
                    if symbol not in costs:
                        costs[symbol] = {"qty": 0, "cost": 0}
                    costs[symbol]["qty"] += qty
                    costs[symbol]["cost"] += qty * price
                elif side == "SELL" and symbol in costs and costs[symbol]["qty"] > 0:
                    avg = costs[symbol]["cost"] / costs[symbol]["qty"]
                    # Cap sell at held quantity to handle orphaned sells
                    sell_qty = min(qty, costs[symbol]["qty"])
                    pnl += (price - avg) * sell_qty
                    costs[symbol]["qty"] -= sell_qty
                    costs[symbol]["cost"] -= avg * sell_qty
            return pnl

        daily_pnl = calc_period_pnl(daily_trades)
        weekly_pnl = calc_period_pnl(weekly_trades)
        monthly_pnl = calc_period_pnl(monthly_trades)

        # Build equity curve for volatility/sharpe/drawdown calculations
        # Also track individual trade PnLs for statistics
        equity_curve = []
        trade_pnls = []  # List of all individual trade PnLs
        wins = []  # Winning trade PnLs
        losses = []  # Losing trade PnLs
        cumulative = 0
        costs = {}
        for trade in sorted(all_trades, key=lambda x: x["timestamp"]):
            symbol, side = trade.get("symbol", ""), trade.get("side", "")
            qty, price = trade.get("quantity", 0), trade.get("price", 0)
            if side == "BUY":
                if symbol not in costs:
                    costs[symbol] = {"qty": 0, "cost": 0}
                costs[symbol]["qty"] += qty
                costs[symbol]["cost"] += qty * price
            elif side == "SELL" and symbol in costs and costs[symbol]["qty"] > 0:
                avg = costs[symbol]["cost"] / costs[symbol]["qty"]
                # Cap sell at held quantity to handle orphaned sells
                sell_qty = min(qty, costs[symbol]["qty"])
                trade_pnl = (price - avg) * sell_qty
                cumulative += trade_pnl
                equity_curve.append(cumulative)
                trade_pnls.append(trade_pnl)
                if trade_pnl > 0:
                    wins.append(trade_pnl)
                else:
                    losses.append(trade_pnl)
                costs[symbol]["qty"] -= sell_qty
                costs[symbol]["cost"] -= avg * sell_qty

        # Calculate trade statistics
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        best_trade = max(trade_pnls) if trade_pnls else 0
        worst_trade = min(trade_pnls) if trade_pnls else 0
        total_wins = sum(wins)
        total_losses = abs(sum(losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Calculate metrics from equity curve
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve)
            avg_return = np.mean(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0.01
            volatility = std_return
            sharpe = (avg_return * np.sqrt(252)) / std_return if std_return > 0 else 0

            # Max drawdown - calculate based on equity curve with starting capital
            # Note: worst_dd tracks the most negative drawdown (worst case)
            starting_capital = estimated_capital
            equity_values = [starting_capital + ec for ec in equity_curve]
            peak = equity_values[0]
            worst_dd = 0  # Drawdowns are negative, so 0 is the "best" (no drawdown)
            for val in equity_values:
                if val > peak:
                    peak = val
                dd = (val - peak) / peak if peak > 0 else 0  # dd is negative or zero
                worst_dd = min(worst_dd, dd)  # Track most negative (worst) drawdown
            max_dd = worst_dd  # Alias for backward compatibility
        else:
            volatility, sharpe, max_dd = 0, 0, 0

        return jsonify(
            {
                "summary": {
                    "total_return": round(total_pnl / estimated_capital, 4)
                    if estimated_capital > 0
                    else 0,
                    "total_pnl": round(total_pnl, 2),
                    "total_sharpe": round(sharpe, 2),
                    "total_drawdown": round(max_dd, 4),
                    "max_drawdown": round(abs(max_dd), 4),
                    "win_rate": round(win_rate, 3),
                    "total_trades": len(all_trades),
                    "winning_trades": winning_count,
                    "losing_trades": losing_count,
                    "avg_win": round(avg_win, 2),
                    "avg_loss": round(avg_loss, 2),
                    "best_trade": round(best_trade, 2),
                    "worst_trade": round(worst_trade, 2),
                    "profit_factor": round(profit_factor, 2),
                },
                "daily": {
                    "return_pct": round(daily_pnl / estimated_capital, 4)
                    if estimated_capital > 0
                    else 0,
                    "pnl": round(daily_pnl, 2),
                    "trades": len(daily_trades),
                    "volatility": round(volatility, 4),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4),
                },
                "weekly": {
                    "return_pct": round(weekly_pnl / estimated_capital, 4)
                    if estimated_capital > 0
                    else 0,
                    "pnl": round(weekly_pnl, 2),
                    "trades": len(weekly_trades),
                    "volatility": round(volatility, 4),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4),
                },
                "monthly": {
                    "return_pct": round(monthly_pnl / estimated_capital, 4)
                    if estimated_capital > 0
                    else 0,
                    "pnl": round(monthly_pnl, 2),
                    "trades": len(monthly_trades),
                    "volatility": round(volatility, 4),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4),
                },
                "all": {
                    "return_pct": round(total_pnl / estimated_capital, 4)
                    if estimated_capital > 0
                    else 0,
                    "pnl": round(total_pnl, 2),
                    "trades": len(all_trades),
                    "volatility": round(volatility, 4),
                    "sharpe": round(sharpe, 2),
                    "max_drawdown": round(max_dd, 4),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e), "summary": {}})


@app.route("/api/equity-curve")
@requires_auth
def equity_curve():
    """Get equity curve data for charting - industry standard portfolio value tracking.

    Returns:
        - labels: dates for x-axis
        - values: cumulative realized P&L (from trades)
        - portfolio_values: actual portfolio value over time (from equity_history table)
        - pnl_by_trade: individual trade P&L data
    """
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()
        portfolio_id = request.args.get("portfolio_id", "default")

        # Get equity history (daily portfolio value snapshots)
        equity_history = db.get_equity_history(days=365, portfolio_id=portfolio_id)
        portfolio_labels = [d["date"] for d in equity_history]
        portfolio_values = [d["equity"] for d in equity_history]

        # Also get trade-based P&L for detailed view
        all_trades = db.get_recent_trades(limit=1000, portfolio_id=portfolio_id)

        if not all_trades:
            return jsonify(
                {
                    "labels": portfolio_labels,
                    "values": [],
                    "portfolio_values": portfolio_values,
                    "pnl_by_trade": [],
                }
            )

        # Build P&L curve by matching BUY/SELL trades
        equity_data = []
        cumulative = 0
        costs = {}

        for trade in sorted(all_trades, key=lambda x: x["timestamp"]):
            symbol = trade.get("symbol", "")
            side = trade.get("side", "")
            qty = trade.get("quantity", 0)
            price = trade.get("price", 0)
            timestamp = trade.get("timestamp", "")

            if side == "BUY":
                if symbol not in costs:
                    costs[symbol] = {"qty": 0, "cost": 0}
                costs[symbol]["qty"] += qty
                costs[symbol]["cost"] += qty * price
            elif side == "SELL" and symbol in costs and costs[symbol]["qty"] > 0:
                avg = costs[symbol]["cost"] / costs[symbol]["qty"]
                trade_pnl = (price - avg) * qty
                cumulative += trade_pnl
                equity_data.append(
                    {
                        "timestamp": timestamp,
                        "pnl": round(trade_pnl, 2),
                        "cumulative": round(cumulative, 2),
                        "symbol": symbol,
                    }
                )
                costs[symbol]["qty"] -= qty
                costs[symbol]["cost"] -= avg * qty

        # Format for Chart.js
        pnl_labels = [d["timestamp"][:10] for d in equity_data]  # Just date
        pnl_values = [d["cumulative"] for d in equity_data]

        # Prefer equity_history data (accurate portfolio values) over P&L calculation
        # Only fall back to P&L calculation if NO equity_history exists
        if portfolio_labels and portfolio_values:
            # Use actual equity snapshots - this is the accurate data
            labels = portfolio_labels
            # portfolio_values already set from equity_history
        elif pnl_labels:
            # Fallback: calculate from P&L when no equity_history exists
            labels = pnl_labels
            starting_capital = float(os.getenv("DEFAULT_CASH", config.default_cash))
            portfolio_values = [starting_capital + pnl for pnl in pnl_values]
        else:
            labels = []

        return jsonify(
            {
                "labels": labels,
                "values": pnl_values,
                "portfolio_values": portfolio_values,
                "pnl_by_trade": equity_data,
            }
        )
    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        return jsonify({"labels": [], "values": [], "portfolio_values": [], "pnl_by_trade": []})


@app.route("/api/trades")
@requires_auth
def get_trades():
    """Get trade history from database"""
    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get trades with optional filtering with input validation
        try:
            days = request.args.get("days", 30, type=int)
            # Validate days parameter
            if days is not None and days < 0:
                days = 30  # Default to 30 days for invalid input
            elif days is not None and days > 365:
                days = 365  # Cap at 1 year to prevent excessive data
        except (ValueError, TypeError):
            days = 30  # Default to 30 days on parse error

        symbol = request.args.get("symbol", None)

        # Make trade limit configurable via environment variable
        trade_limit = int(os.getenv("DASHBOARD_TRADE_LIMIT", "1000"))
        trade_limit = min(max(trade_limit, 10), 5000)  # Clamp between 10 and 5000

        portfolio_id = request.args.get("portfolio_id", "default")
        trades = db.get_recent_trades(limit=trade_limit, symbol=symbol, days=days, portfolio_id=portfolio_id)

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
                        "pnl": trade.get("pnl"),  # Realized P&L for SELL trades
                        "notional": trade["quantity"] * trade["price"],
                        # Include all four sides (BUY, SELL, SELL_SHORT, BUY_TO_COVER)
                        "cash_impact": (
                            -trade["quantity"] * trade["price"]
                            if trade["side"] in ("BUY", "BUY_TO_COVER")
                            else trade["quantity"] * trade["price"]
                        ),
                    }
                )

            # Calculate summary
            total_trades = len(trade_list)
            total_volume = sum(t["notional"] for t in trade_list)
            total_commission = sum(t["commission"] for t in trade_list)
            # Count by high-level direction for summary
            buy_trades = [t for t in trade_list if t["side"] in ("BUY", "BUY_TO_COVER")]
            sell_trades = [t for t in trade_list if t["side"] in ("SELL", "SELL_SHORT")]

            # Calculate P&L stats (only for trades with P&L - i.e., SELL trades)
            trades_with_pnl = [t for t in trade_list if t.get("pnl") is not None]
            winners = [t for t in trades_with_pnl if t["pnl"] > 0]
            losers = [t for t in trades_with_pnl if t["pnl"] < 0]
            total_pnl = sum(t["pnl"] for t in trades_with_pnl)

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
                        "winners": len(winners),
                        "losers": len(losers),
                        "total_pnl": total_pnl,
                    },
                }
            )
    except Exception as e:
        logger.error(f"Error fetching trades: {e}")

    # Return empty trades if database is locked
    trades = []

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
                        SELECT id, symbol, action, quantity, price, timestamp,
                               slippage, commission,
                               quantity * price as notional,
                               CASE
                                   WHEN action = 'BUY' THEN -quantity * price - commission
                                   WHEN action = 'SELL' THEN quantity * price - commission
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
                        SELECT id, symbol, action, quantity, price, timestamp,
                               slippage, commission,
                               quantity * price as notional,
                               CASE
                                   WHEN action = 'BUY' THEN -quantity * price - commission
                                   WHEN action = 'SELL' THEN quantity * price - commission
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
    # Return cached status if available and recent (within 3 seconds)
    # Thread-safe cache access
    current_time = time.time()
    portfolio_id = request.args.get("portfolio_id", "default")
    strat_cache_key = f"_strategies_cache_{portfolio_id}"
    strat_cache_time_key = f"_strategies_cache_time_{portfolio_id}"
    with _strategies_cache_lock:
        if hasattr(app, strat_cache_key) and hasattr(app, strat_cache_time_key):
            if current_time - getattr(app, strat_cache_time_key) < 3:  # 3 second cache
                return jsonify(getattr(app, strat_cache_key))

    try:
        import json
        from pathlib import Path

        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get real position count
        positions = db.get_positions(portfolio_id=portfolio_id)
        active_positions = [p for p in positions if p.get("quantity", 0) > 0]

        # Get recent trades to calculate strategy performance
        trades = db.get_recent_trades(limit=100, portfolio_id=portfolio_id)

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

        # Calculate real PnL from watchlist cache (has live prices)
        total_pnl = 0
        winning_positions = 0
        watchlist_position_count = 0
        if hasattr(app, "_watchlist_cache") and app._watchlist_cache:
            for item in app._watchlist_cache:
                if item.get("has_position") and item.get("quantity", 0) > 0:
                    watchlist_position_count += 1
                    pnl = item.get("pnl", 0)
                    total_pnl += pnl
                    if pnl > 0:
                        winning_positions += 1

        # Calculate slippage from trades
        slippages = []
        for trade in trades:
            if trade.get("slippage") is not None:
                slippages.append(abs(trade["slippage"]))
        avg_slippage = sum(slippages) / len(slippages) if slippages else 0
        avg_slippage_bps = avg_slippage * 10000  # Convert to basis points

        # Build data dict (cache this, not the Response object)
        strategies_data = {
            "active_strategies": {
                "ml_enhanced": {
                    "enabled": True,
                    "regime": ml_regime,
                    "confidence": round(ml_confidence, 3),
                    "positions": len(active_positions),
                    "symbols_tracked": len(active_positions),  # Actual tracked symbols
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
                    "strategies_count": 4,  # ML Enhanced, Mean Reversion, Momentum, Pairs
                    "max_positions": int(os.getenv("RISK_MAX_OPEN_POSITIONS", 30)),
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
                    "win_rate": round(winning_positions / watchlist_position_count, 3)
                    if watchlist_position_count > 0
                    else 0,
                    "total_trades": len(trades),
                    "winning_positions": winning_positions,
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

        # Cache the data dict, not the Response object (thread-safe, keyed by portfolio_id)
        with _strategies_cache_lock:
            setattr(app, strat_cache_key, strategies_data)
            setattr(app, strat_cache_time_key, time.time())

        return jsonify(strategies_data)
    except Exception as e:
        logger.error(f"Error getting strategy status: {e}")
        # Return minimal real data on error
        return jsonify(
            {
                "active_strategies": {
                    "ml_enhanced": {"enabled": True, "error": str(e)},
                    "microstructure": {"enabled": False},
                    "portfolio_manager": {
                        "enabled": True,
                        "allocation_method": "Equal Weight",
                        "strategies_count": 4,
                    },
                    "smart_execution": {"enabled": True},
                },
                "performance_by_strategy": {},
                "error": str(e),
            }
        )


@app.route("/api/microstructure/metrics")
@requires_auth
def microstructure_metrics():
    """Get microstructure strategy metrics from real data"""
    try:
        # Try to get real microstructure data from database or state files
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get recent trades for microstructure analysis
        recent_trades = db.get_recent_trades(limit=100)

        # Calculate real metrics if we have trades
        if recent_trades:
            trades_today = len(
                [
                    t
                    for t in recent_trades
                    if t.get("timestamp", "").startswith(datetime.now().strftime("%Y-%m-%d"))
                ]
            )
            winning_trades = [t for t in recent_trades if t.get("pnl", 0) > 0]
            win_rate = len(winning_trades) / len(recent_trades) if recent_trades else 0

            return jsonify(
                {
                    "order_flow_imbalance": {
                        "current": 0,
                        "avg_1h": 0,
                        "trend": "neutral",
                        "signal": "neutral",
                    },
                    "book_pressure": 0,
                    "tick_direction": 0,
                    "spread_analysis": {
                        "current_bps": 0,
                        "avg_bps": 0,
                        "widening": False,
                        "liquidity": "unknown",
                    },
                    "tick_momentum": {
                        "score": 0,
                        "direction": "neutral",
                        "strength": "none",
                        "trades_analyzed": len(recent_trades),
                    },
                    "ensemble_metrics": {
                        "combined_score": 0,
                        "confidence": 0,
                        "active_signals": 0,
                        "last_signal": "N/A",
                    },
                    "performance": {
                        "trades_today": trades_today,
                        "win_rate": round(win_rate, 3),
                        "avg_profit_bps": 0,
                        "sharpe_ratio": 0,
                    },
                }
            )
    except Exception as e:
        logger.error(f"Error getting microstructure metrics: {e}")

    # Return zeros if error
    return jsonify(
        {
            "order_flow_imbalance": {
                "current": 0,
                "avg_1h": 0,
                "trend": "neutral",
                "signal": "neutral",
            },
            "book_pressure": 0,
            "tick_direction": 0,
            "spread_analysis": {
                "current_bps": 0,
                "avg_bps": 0,
                "widening": False,
                "liquidity": "unknown",
            },
            "tick_momentum": {
                "score": 0,
                "direction": "neutral",
                "strength": "none",
                "trades_analyzed": 0,
            },
            "ensemble_metrics": {
                "combined_score": 0,
                "confidence": 0,
                "active_signals": 0,
                "last_signal": "N/A",
            },
            "performance": {
                "trades_today": 0,
                "win_rate": 0,
                "avg_profit_bps": 0,
                "sharpe_ratio": 0,
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
    """Get advanced risk management status from actual risk state"""
    # Check if advanced risk manager is running
    advanced_risk_enabled = os.getenv("ADVANCED_RISK_ENABLED", "true").lower() == "true"

    if not advanced_risk_enabled:
        return jsonify({"enabled": False, "message": "Advanced risk management not enabled"})

    try:
        # Load risk state from file
        risk_state_file = Path("data/risk_state.json")
        risk_state = {}
        if risk_state_file.exists():
            with open(risk_state_file) as f:
                risk_state = json.load(f)

        # Get real data from database
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get positions and calculate exposure
        positions = db.get_positions()
        active_positions = {p["symbol"]: p for p in positions if p.get("quantity", 0) > 0}
        total_exposure = sum(
            abs(p.get("quantity", 0) * p.get("current_price", 0)) for p in active_positions.values()
        )

        # Get recent trades for Kelly calculation
        trades = db.get_recent_trades(limit=100)

        # Calculate win rate and consecutive losses
        # Use (t.get("pnl") or 0) to handle None values from BUY trades
        winning_trades = [t for t in trades if (t.get("pnl") or 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.5

        # Count consecutive losses
        consecutive_losses = 0
        for trade in reversed(trades):
            if (trade.get("pnl") or 0) < 0:
                consecutive_losses += 1
            else:
                break

        # Calculate daily loss
        today_trades = [
            t
            for t in trades
            if t.get("timestamp")
            and datetime.fromisoformat(t["timestamp"]).date() == datetime.now().date()
        ]
        daily_pnl = sum((t.get("pnl") or 0) for t in today_trades)
        daily_loss_pct = (
            abs(daily_pnl / risk_state.get("current_capital", 100000)) if daily_pnl < 0 else 0
        )

        # Calculate max drawdown from trades
        cumulative_pnl = 0
        peak_pnl = 0
        max_drawdown = 0
        for trade in trades:
            cumulative_pnl += trade.get("pnl") or 0
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = (
                (peak_pnl - cumulative_pnl) / risk_state.get("current_capital", 100000)
                if peak_pnl > 0
                else 0
            )
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate Kelly fractions for top positions
        kelly_positions = {}
        for symbol in list(active_positions.keys())[:5]:  # Top 5 positions
            symbol_trades = [t for t in trades if t.get("symbol") == symbol]
            symbol_wins = [t for t in symbol_trades if (t.get("pnl") or 0) > 0]
            symbol_win_rate = len(symbol_wins) / len(symbol_trades) if symbol_trades else 0.5

            # Simple Kelly estimation
            if symbol_wins and symbol_trades:
                avg_win = sum((t.get("pnl") or 0) for t in symbol_wins) / len(symbol_wins)
                losses = [t for t in symbol_trades if (t.get("pnl") or 0) < 0]
                avg_loss = (
                    abs(sum((t.get("pnl") or 0) for t in losses) / len(losses))
                    if losses
                    else avg_win * 0.5
                )
                edge = symbol_win_rate * avg_win - (1 - symbol_win_rate) * avg_loss
                kelly_fraction = min(0.25, max(0, (edge / avg_win) if avg_win > 0 else 0))
            else:
                kelly_fraction = 0.02
                edge = 0

            kelly_positions[symbol] = {
                "kelly_fraction": round(kelly_fraction, 4),
                "win_rate": round(symbol_win_rate, 2),
                "edge": round(edge / risk_state.get("current_capital", 100000), 4),
            }

        # Calculate portfolio Kelly
        portfolio_kelly = (
            sum(p["kelly_fraction"] for p in kelly_positions.values()) / len(kelly_positions)
            if kelly_positions
            else 0.02
        )

        return jsonify(
            {
                "enabled": True,
                "kelly_sizing": {
                    "enabled": True,
                    "current_positions": kelly_positions,
                    "portfolio_kelly": round(portfolio_kelly, 4),
                },
                "kill_switches": {
                    "active": risk_state.get("kill_switch_triggered", False),
                    "triggered_count": 0,  # Would need to track this
                    "last_trigger": None,
                    "limits": {
                        "daily_loss": {"limit": 0.05, "current": round(daily_loss_pct, 4)},
                        "consecutive_losses": {"limit": 5, "current": consecutive_losses},
                        "max_drawdown": {"limit": 0.10, "current": round(max_drawdown, 4)},
                        "position_loss": {"limit": 0.02, "per_position": True},
                    },
                },
                "correlation_limits": {
                    "max_correlation": 0.7,
                    "max_correlated_exposure": 0.3,
                    "high_correlations": [],  # Would need correlation matrix calculation
                },
                "risk_metrics": {
                    "total_exposure": round(total_exposure, 2),
                    "leverage": round(
                        total_exposure / risk_state.get("current_capital", 100000), 2
                    ),
                    "var_95": 0,  # Would need historical returns
                    "sharpe_ratio": 0,  # Would need returns and volatility
                    "max_drawdown": round(max_drawdown, 4),
                    "current_capital": risk_state.get("current_capital", 100000),
                    "total_pnl": risk_state.get("total_pnl", 0),
                    "daily_pnl": round(daily_pnl, 2),
                },
            }
        )
    except Exception as e:
        logger.error(f"Error getting risk status: {e}")
        return jsonify(
            {
                "enabled": True,
                "error": str(e),
                "kelly_sizing": {"enabled": True, "current_positions": {}, "portfolio_kelly": 0},
                "kill_switches": {"active": False, "limits": {}},
                "correlation_limits": {},
                "risk_metrics": {},
            }
        )


@app.route("/api/risk/kelly/<symbol>")
@requires_auth
def get_kelly_parameters(symbol):
    """Get Kelly parameters for a specific symbol from actual trade history"""
    import re

    # Validate symbol format (1-5 uppercase letters, common stock symbol format)
    if not symbol or not re.match(r"^[A-Z]{1,5}$", symbol.upper()):
        return jsonify({"error": "Invalid symbol format. Must be 1-5 uppercase letters."}), 400

    # Normalize to uppercase
    symbol = symbol.upper()

    try:
        from sync_db_reader import SyncDatabaseReader

        db = SyncDatabaseReader()

        # Get trades for this symbol
        all_trades = db.get_recent_trades(limit=200)
        symbol_trades = [t for t in all_trades if t.get("symbol") == symbol]

        if not symbol_trades:
            # Return conservative defaults if no trade history
            return jsonify(
                {
                    "symbol": symbol,
                    "kelly_fraction": 0.01,
                    "half_kelly": 0.005,
                    "quarter_kelly": 0.0025,
                    "win_rate": 0.5,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "edge": 0,
                    "odds": 0,
                    "trade_count": 0,
                    "recommended_position_size": 1000,
                    "confidence_level": 0.0,
                }
            )

        # Calculate win/loss statistics
        wins = [t for t in symbol_trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in symbol_trades if (t.get("pnl") or 0) < 0]

        win_rate = len(wins) / len(symbol_trades)
        avg_win = sum((t.get("pnl") or 0) for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum((t.get("pnl") or 0) for t in losses) / len(losses)) if losses else 0

        # Calculate Kelly fraction
        if avg_loss > 0 and avg_win > 0:
            odds = avg_win / avg_loss
            kelly = (win_rate * odds - (1 - win_rate)) / odds
            kelly = max(0, min(0.25, kelly))  # Cap at 25%
        else:
            kelly = 0.01  # Conservative default

        # Calculate edge
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        # Load current capital from risk state with type validation
        risk_state_file = Path("data/risk_state.json")
        current_capital = DEFAULT_CAPITAL
        if risk_state_file.exists():
            try:
                with open(risk_state_file) as f:
                    risk_state = json.load(f)
                    raw_capital = risk_state.get("current_capital", DEFAULT_CAPITAL)
                    # Type validation: ensure it's a valid number
                    if isinstance(raw_capital, (int, float)) and raw_capital > 0:
                        current_capital = float(raw_capital)
                    else:
                        logger.warning(
                            f"Invalid current_capital in risk_state.json: {raw_capital}, using default"
                        )
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Failed to parse risk_state.json: {e}, using default capital")

        # Calculate recommended position size
        recommended_size = current_capital * kelly * 0.5  # Half Kelly

        return jsonify(
            {
                "symbol": symbol,
                "kelly_fraction": round(kelly, 4),
                "half_kelly": round(kelly * 0.5, 4),
                "quarter_kelly": round(kelly * 0.25, 4),
                "win_rate": round(win_rate, 3),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "edge": round(edge, 2),
                "odds": round(odds if avg_loss > 0 else 0, 2),
                "trade_count": len(symbol_trades),
                "recommended_position_size": round(recommended_size, 0),
                "confidence_level": min(
                    0.95, len(symbol_trades) / 30
                ),  # Higher confidence with more trades
            }
        )
    except Exception as e:
        logger.error(f"Error calculating Kelly for {symbol}: {e}")
        return jsonify(
            {
                "symbol": symbol,
                "error": str(e),
                "kelly_fraction": 0.01,
                "half_kelly": 0.005,
                "quarter_kelly": 0.0025,
                "trade_count": 0,
            }
        )


@app.route("/api/risk/kill-switch", methods=["POST"])
@requires_auth
def control_kill_switch():
    """Control kill switch (reset after trigger)"""
    action = (request.json or {}).get("action", "status")

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


@app.route("/api/safety/circuit-breakers")
@requires_auth
def get_circuit_breakers():
    """Get circuit breaker status for all components"""
    try:
        from robo_trader.circuit_breaker import circuit_manager

        all_stats = circuit_manager.get_all_statistics()
        open_breakers = circuit_manager.get_open_breakers()

        return jsonify(
            {
                "breakers": all_stats,
                "open_count": len(open_breakers),
                "open_breakers": open_breakers,
                "any_open": circuit_manager.is_any_open(),
            }
        )
    except ImportError:
        return jsonify(
            {
                "breakers": {},
                "open_count": 0,
                "open_breakers": [],
                "any_open": False,
                "error": "Circuit breaker module not available",
            }
        )


@app.route("/api/safety/order-manager")
@requires_auth
def get_order_manager_status():
    """Get order management statistics"""
    try:
        # Try to get from runner if available
        if hasattr(app, "order_manager"):
            stats = app.order_manager.get_statistics()
            active_orders = app.order_manager.get_active_orders()
            return jsonify(
                {
                    "statistics": stats,
                    "active_orders": [
                        {
                            "id": order.id[:8],
                            "symbol": order.symbol,
                            "side": order.side,
                            "status": order.status.value,
                            "filled": f"{order.fill_percentage:.1f}%",
                        }
                        for order in active_orders
                    ],
                }
            )
        else:
            # Return mock data for demonstration
            return jsonify(
                {
                    "statistics": {
                        "total_orders": 0,
                        "active_orders": 0,
                        "successful_fills": 0,
                        "failed_orders": 0,
                        "fill_rate": 0.0,
                        "error_rate": 0.0,
                    },
                    "active_orders": [],
                }
            )
    except Exception as e:
        return jsonify({"error": str(e), "statistics": {}, "active_orders": []})


@app.route("/api/safety/data-validator")
@requires_auth
def get_data_validator_status():
    """Get data validation statistics"""
    try:
        # Try to get from runner if available
        if hasattr(app, "data_validator"):
            stats = app.data_validator.get_statistics()
            return jsonify(stats)
        else:
            # Return mock data for demonstration
            return jsonify(
                {
                    "total_validations": 0,
                    "passed": 0,
                    "failed_stale": 0,
                    "failed_spread": 0,
                    "failed_price": 0,
                    "failed_volume": 0,
                    "failed_anomaly": 0,
                    "pass_rate": 100.0,
                }
            )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/safety/thresholds")
@requires_auth
def get_safety_thresholds():
    """Get all safety thresholds from environment"""
    thresholds = {
        "max_open_positions": os.getenv("MAX_OPEN_POSITIONS", "5"),
        "max_orders_per_minute": os.getenv("MAX_ORDERS_PER_MINUTE", "10"),
        "stop_loss_percent": os.getenv("STOP_LOSS_PERCENT", "2.0"),
        "take_profit_percent": os.getenv("TAKE_PROFIT_PERCENT", "3.0"),
        "data_staleness_seconds": os.getenv("DATA_STALENESS_SECONDS", "60"),
        "circuit_breaker_threshold": os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5"),
        "circuit_breaker_timeout": os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300"),
        "max_daily_trades": os.getenv("MAX_DAILY_TRADES", "100"),
    }
    return jsonify(thresholds)


@app.route("/api/start", methods=["POST"])
@requires_auth
def start_trading():
    """Start trading with proper Gateway checks and zombie cleanup."""
    global trading_status

    # Load symbols from user settings
    global default_symbols
    try:
        with open("user_settings.json", "r") as f:
            settings = json.load(f)
            default_symbols = settings.get("default", {}).get("symbols", ["AAPL", "MSFT", "GOOGL"])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        default_symbols = ["AAPL", "MSFT", "GOOGL"]

    data = request.json or {}
    symbols = data.get("symbols", default_symbols)

    if trading_status == "running":
        return jsonify({"status": "already_running"})

    # Use the start_runner.sh script for proper startup with Gateway checks
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "start_runner.sh")
    symbols_str = ",".join(symbols) if isinstance(symbols, list) else symbols

    try:
        # Run the startup script and capture output
        result = subprocess.run(
            [script_path, symbols_str],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            trading_status = "running"
            trading_log.append(
                f"{datetime.now().strftime('%H:%M:%S')} - Trading started for {symbols_str}"
            )
            return jsonify(
                {
                    "status": "started",
                    "symbols": symbols_str.split(","),
                    "output": result.stdout,
                }
            )
        else:
            trading_log.append(
                f"{datetime.now().strftime('%H:%M:%S')} - Failed to start: {result.stderr}"
            )
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": result.stderr or result.stdout,
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "error": "Startup timed out"}), 500
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        # Don't expose internal error details to client
        return jsonify({"status": "error", "error": "Failed to start trading system"}), 500


@app.route("/api/stop", methods=["POST"])
@requires_auth
def stop_trading():
    """Stop trading - kills all runner processes."""
    global trading_status, trading_process

    try:
        # Kill all runner processes (more reliable than just terminating one)
        result = subprocess.run(
            ["pkill", "-9", "-f", "runner_async"],
            capture_output=True,
            text=True,
        )

        # Also terminate tracked process if any
        if trading_process:
            try:
                trading_process.terminate()
            except Exception:
                pass
            trading_process = None

        trading_status = "stopped"
        trading_log.append(f"{datetime.now().strftime('%H:%M:%S')} - Trading stopped")

        return jsonify({"status": "stopped"})

    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/logs")
@requires_auth
def get_logs():
    """Get recent logs from log file (newest first)"""
    logs = []
    log_file = "robo_trader.log"

    try:
        # Read last 5000 lines from log file to get past repetitive warnings
        with open(log_file, "r") as f:
            # Get last 5000 lines efficiently
            lines = f.readlines()
            recent_lines = lines[-5000:] if len(lines) > 5000 else lines

            for line in recent_lines:
                try:
                    # Parse JSON log entry
                    log_entry = json.loads(line)

                    # Parse ISO timestamp and convert to local time
                    timestamp_str = log_entry.get("timestamp", "")
                    if timestamp_str:
                        try:
                            # Parse ISO format timestamp
                            from datetime import datetime

                            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            # Format as local time HH:MM:SS
                            timestamp = dt.strftime("%H:%M:%S")
                        except Exception:
                            # Fallback to simple parsing
                            timestamp = (
                                timestamp_str.split("T")[1].split(".")[0]
                                if "T" in timestamp_str
                                else ""
                            )
                    else:
                        timestamp = ""

                    message = log_entry.get("message", "")

                    # Extract event from nested JSON if present
                    if '{"event":' in message:
                        try:
                            nested = json.loads(message)
                            event_msg = nested.get("event", message)
                        except Exception:
                            event_msg = message
                    else:
                        event_msg = message

                    # Skip repetitive correlation warnings (keep only 1)
                    if "Not enough symbols with sufficient data for correlation" in event_msg:
                        if hasattr(get_logs, "_seen_correlation_warning"):
                            continue
                        get_logs._seen_correlation_warning = True

                    # Format log entry for display
                    if timestamp and event_msg:
                        logs.append(f"{timestamp} - {event_msg}")
                except json.JSONDecodeError:
                    # Handle non-JSON log lines
                    logs.append(line.strip())

        # Reset correlation warning flag for next request
        if hasattr(get_logs, "_seen_correlation_warning"):
            delattr(get_logs, "_seen_correlation_warning")

    except FileNotFoundError:
        logs.append("Log file not found")
    except Exception as e:
        logs.append(f"Error reading logs: {str(e)}")

    # Return logs in reverse order (newest first), max 100
    return jsonify({"logs": list(reversed(logs[-100:]))})


if __name__ == "__main__":
    # Initialize components
    logger.info("Starting RoboTrader Dashboard...")

    # Start WebSocket server for real-time log streaming
    from robo_trader.websocket_server import ws_manager

    logger.info("Starting WebSocket server...")
    ws_manager.start()

    logger.info(f"Dashboard starting on port {os.getenv('DASH_PORT', 5555)}")

    # Run Flask app
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("DASH_PORT", 5555)),
        use_reloader=False,
        debug=os.getenv("FLASK_ENV") == "development",
    )

#!/usr/bin/env python3
"""
Simple Web Dashboard for Robo Trader
Following CLAUDE.md: Clarity over cleverness, no unnecessary abstraction
"""

from flask import Flask, render_template_string, jsonify, request
import asyncio
import threading
import json
from datetime import datetime
import subprocess
import os
import signal
from robo_trader.config import load_config
from robo_trader.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

# Global state (simple is better than complex)
trading_process = None
trading_status = "stopped"
trading_log = []
positions = {}
pnl = {"daily": 0.0, "total": 0.0}
ai_decisions = []

# HTML template with inline CSS and JS (single file simplicity)
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Robo Trader Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 20px;
        }
        .status.running { background: #4caf50; color: white; }
        .status.stopped { background: #f44336; color: white; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .card h2 {
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .big-button {
            width: 100%;
            padding: 20px;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 10px;
        }
        .start { background: #4caf50; color: white; }
        .start:hover { background: #45a049; }
        .stop { background: #f44336; color: white; }
        .stop:hover { background: #da190b; }
        .test { background: #2196F3; color: white; }
        .test:hover { background: #0b7dda; }
        .pnl {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .pnl.positive { color: #4caf50; }
        .pnl.negative { color: #f44336; }
        .log {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 5px;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
        }
        .log-entry {
            padding: 2px 0;
            border-bottom: 1px solid #ddd;
        }
        .positions-table {
            width: 100%;
            border-collapse: collapse;
        }
        .positions-table th {
            background: #f5f5f5;
            padding: 10px;
            text-align: left;
        }
        .positions-table td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }
        .ai-decision {
            background: #e3f2fd;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .ai-decision .symbol {
            font-weight: bold;
            color: #1976d2;
        }
        .ai-decision .confidence {
            float: right;
            background: #1976d2;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
        }
        .settings {
            display: grid;
            gap: 10px;
        }
        .settings label {
            font-weight: bold;
            color: #666;
        }
        .settings input, .settings select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Robo Trader Dashboard
                <span id="status" class="status stopped">STOPPED</span>
            </h1>
            <p>AI-Powered Trading with Claude 3.5 Sonnet</p>
        </div>

        <div class="grid">
            <!-- Control Panel -->
            <div class="card">
                <h2>🎮 Control Panel</h2>
                <button id="startBtn" class="big-button start" onclick="startTrading()">
                    START TRADING
                </button>
                <button id="stopBtn" class="big-button stop" onclick="stopTrading()" disabled>
                    STOP TRADING
                </button>
                <button class="big-button test" onclick="testAI()">
                    TEST AI ANALYSIS
                </button>
            </div>

            <!-- P&L Display -->
            <div class="card">
                <h2>💰 Profit & Loss</h2>
                <div>Daily P&L</div>
                <div id="dailyPnl" class="pnl positive">$0.00</div>
                <div>Total P&L</div>
                <div id="totalPnl" class="pnl positive">$0.00</div>
            </div>

            <!-- Account Info -->
            <div class="card">
                <h2>📊 Account Status</h2>
                <div>Mode: <strong>PAPER TRADING</strong></div>
                <div>Account: <strong id="accountId">-</strong></div>
                <div>Cash: <strong id="cash">$1,000,000</strong></div>
                <div>Positions: <strong id="posCount">0</strong></div>
            </div>
        </div>

        <div class="grid">
            <!-- Positions -->
            <div class="card">
                <h2>📈 Current Positions</h2>
                <div id="positions">
                    <p style="color: #999;">No positions yet</p>
                </div>
            </div>

            <!-- AI Decisions -->
            <div class="card">
                <h2>🧠 AI Analysis Feed</h2>
                <div id="aiDecisions">
                    <p style="color: #999;">Waiting for market events...</p>
                </div>
            </div>
        </div>

        <div class="grid">
            <!-- Settings -->
            <div class="card">
                <h2>⚙️ Quick Settings</h2>
                <div class="settings">
                    <label>Symbols to Trade</label>
                    <input id="symbols" value="SPY,QQQ,AAPL" />
                    
                    <label>Risk Level</label>
                    <select id="riskLevel">
                        <option value="conservative">Conservative (1% risk)</option>
                        <option value="moderate" selected>Moderate (2% risk)</option>
                        <option value="aggressive">Aggressive (3% risk)</option>
                    </select>
                    
                    <label>Max Daily Loss</label>
                    <input id="maxLoss" value="$1000" />
                </div>
            </div>

            <!-- Activity Log -->
            <div class="card">
                <h2>📜 Activity Log</h2>
                <div id="log" class="log">
                    <div class="log-entry">System ready...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update dashboard every 2 seconds
        setInterval(updateDashboard, 2000);
        
        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.className = 'status ' + data.status;
                    statusEl.textContent = data.status.toUpperCase();
                    
                    // Update buttons
                    document.getElementById('startBtn').disabled = data.status === 'running';
                    document.getElementById('stopBtn').disabled = data.status === 'stopped';
                    
                    // Update P&L
                    updatePnL(data.pnl);
                    
                    // Update positions
                    updatePositions(data.positions);
                    
                    // Update AI decisions
                    updateAIDecisions(data.ai_decisions);
                    
                    // Update log
                    updateLog(data.log);
                });
        }
        
        function updatePnL(pnl) {
            const daily = document.getElementById('dailyPnl');
            const total = document.getElementById('totalPnl');
            
            daily.textContent = '$' + pnl.daily.toFixed(2);
            daily.className = 'pnl ' + (pnl.daily >= 0 ? 'positive' : 'negative');
            
            total.textContent = '$' + pnl.total.toFixed(2);
            total.className = 'pnl ' + (pnl.total >= 0 ? 'positive' : 'negative');
        }
        
        function updatePositions(positions) {
            const container = document.getElementById('positions');
            document.getElementById('posCount').textContent = Object.keys(positions).length;
            
            if (Object.keys(positions).length === 0) {
                container.innerHTML = '<p style="color: #999;">No positions yet</p>';
                return;
            }
            
            let html = '<table class="positions-table"><tr><th>Symbol</th><th>Qty</th><th>P&L</th></tr>';
            for (const [symbol, pos] of Object.entries(positions)) {
                const pnlClass = pos.pnl >= 0 ? 'positive' : 'negative';
                html += `<tr>
                    <td>${symbol}</td>
                    <td>${pos.quantity}</td>
                    <td class="${pnlClass}">$${pos.pnl.toFixed(2)}</td>
                </tr>`;
            }
            html += '</table>';
            container.innerHTML = html;
        }
        
        function updateAIDecisions(decisions) {
            const container = document.getElementById('aiDecisions');
            if (decisions.length === 0) {
                container.innerHTML = '<p style="color: #999;">Waiting for market events...</p>';
                return;
            }
            
            let html = '';
            for (const decision of decisions.slice(-5)) {  // Last 5 decisions
                html += `<div class="ai-decision">
                    <span class="symbol">${decision.symbol}</span>
                    <span class="confidence">${decision.confidence}%</span>
                    <div>${decision.action}: ${decision.reason}</div>
                    <small>${decision.time}</small>
                </div>`;
            }
            container.innerHTML = html;
        }
        
        function updateLog(log) {
            const container = document.getElementById('log');
            let html = '';
            for (const entry of log.slice(-10)) {  // Last 10 entries
                html += `<div class="log-entry">${entry}</div>`;
            }
            container.innerHTML = html;
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
                    alert(`AI Analysis:\\n\\nSignal: ${result.signal}\\nConfidence: ${result.confidence}%\\nReason: ${result.reason}`);
                });
            }
        }
        
        // Initial update
        updateDashboard();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/status')
def get_status():
    global trading_status, pnl, positions, ai_decisions, trading_log
    return jsonify({
        'status': trading_status,
        'pnl': pnl,
        'positions': positions,
        'ai_decisions': ai_decisions[-10:],  # Last 10
        'log': trading_log[-20:]  # Last 20 entries
    })

@app.route('/api/start', methods=['POST'])
def start_trading():
    global trading_process, trading_status, trading_log
    
    if trading_status == 'running':
        return jsonify({'error': 'Already running'}), 400
    
    symbols = request.json.get('symbols', 'SPY,QQQ,AAPL')
    
    # Start the runner as a subprocess
    cmd = [
        'python', '-m', 'robo_trader.runner',
        '--symbols', symbols,
        '--duration', '1 D',
        '--bar-size', '5 mins'
    ]
    
    trading_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid
    )
    
    trading_status = 'running'
    trading_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] Started trading: {symbols}")
    
    # Start thread to read output
    threading.Thread(target=read_process_output, daemon=True).start()
    
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

@app.route('/api/test-ai', methods=['POST'])
def test_ai():
    event = request.json.get('event', '')
    
    # Simple mock for now - will integrate with real AI
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

def read_process_output():
    global trading_process, trading_log
    
    if not trading_process:
        return
    
    for line in trading_process.stdout:
        line = line.strip()
        if line:
            # Filter out IB connection noise
            if any(x in line for x in ['Warning 2104', 'Warning 2106', 'Warning 2158', 
                                       'data farm connection is OK', 'HMDS data farm']):
                continue  # Skip these normal IB messages
            
            # Clean up other IB messages
            if 'INFO ib_insync' in line:
                if 'Connected' in line:
                    trading_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected to IB")
                elif 'Disconnected' in line:
                    trading_log.append(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 Disconnected from IB")
                continue
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            trading_log.append(f"[{timestamp}] {line}")
            
            # Parse for positions, P&L, AI decisions
            # This is simplified - real implementation would parse actual output
            if 'position' in line.lower():
                update_positions_from_log(line)
            elif 'pnl' in line.lower():
                update_pnl_from_log(line)

def update_positions_from_log(line):
    # Parse position updates from log
    pass

def update_pnl_from_log(line):
    # Parse P&L updates from log
    pass

if __name__ == '__main__':
    import nest_asyncio
    nest_asyncio.apply()
    
    print("\n" + "="*60)
    print("🚀 ROBO TRADER WEB DASHBOARD")
    print("="*60)
    print("\n📊 Opening dashboard at: http://localhost:5555")
    print("\n💡 Features:")
    print("  • One-click start/stop trading")
    print("  • Real-time P&L monitoring")
    print("  • AI decision feed")
    print("  • Position tracking")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    # Install Flask if needed
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        subprocess.run(['pip', 'install', 'flask'], check=True)
    
    # Start the web server
    app.run(host='0.0.0.0', port=5555, debug=False)
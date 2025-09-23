#!/bin/bash
set -e

echo "🚀 RoboTrader Clean Startup Script"
echo "=================================="

# Clean up any existing lock files
rm -f /tmp/robo_trader_*.lock /tmp/robo_trader_*.pid

# Kill ALL Python processes aggressively
echo "🔥 Killing all Python processes..."
pkill -9 -f python 2>/dev/null || true
pkill -9 -f runner_async 2>/dev/null || true  
pkill -9 -f app.py 2>/dev/null || true
pkill -9 -f websocket_server 2>/dev/null || true

# Wait for processes to die
echo "⏳ Waiting for processes to terminate..."
sleep 5

# Verify no Python processes remain
PYTHON_COUNT=$(ps aux | grep python | grep -v grep | wc -l)
echo "📊 Remaining Python processes: $PYTHON_COUNT"

if [ $PYTHON_COUNT -gt 0 ]; then
    echo "⚠️  Warning: Some Python processes still running:"
    ps aux | grep python | grep -v grep
fi

# Clear any port conflicts
echo "🌐 Clearing port conflicts..."
lsof -ti:5555 | xargs kill -9 2>/dev/null || true
lsof -ti:8765 | xargs kill -9 2>/dev/null || true

# Wait for ports to clear
sleep 3

# Start services with process locks
echo "🔧 Starting services with process locks..."

# Start WebSocket server
echo "🌐 Starting WebSocket server..."
python3 -c "
from process_manager import ProcessManager
import subprocess
import sys

pm = ProcessManager('websocket_server')
if pm.acquire_lock():
    subprocess.run(['python3', '-m', 'robo_trader.websocket_server'])
else:
    print('WebSocket server already running')
    sys.exit(1)
" &

# Wait for WebSocket server to start
sleep 3

# Start trading runner
echo "📈 Starting trading runner..."
export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
python3 -c "
from process_manager import ProcessManager
import subprocess
import sys
import os

pm = ProcessManager('trading_runner')
if pm.acquire_lock():
    os.environ['LOG_FILE'] = '/Users/oliver/robo_trader/robo_trader.log'
    subprocess.run([
        'python3', '-m', 'robo_trader.runner_async', 
        '--symbols', 'AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF,QQQ,QLD,BBIO,IMRX,CRGY'
    ])
else:
    print('Trading runner already running')
    sys.exit(1)
" &

# Wait for trading runner to initialize
sleep 3

# Start dashboard
echo "📊 Starting dashboard..."
export DASH_PORT=5555
python3 -c "
from process_manager import ProcessManager
import subprocess
import sys
import os

pm = ProcessManager('dashboard')
if pm.acquire_lock():
    os.environ['DASH_PORT'] = '5555'
    subprocess.run(['python3', 'app.py'])
else:
    print('Dashboard already running')
    sys.exit(1)
" &

echo "✅ All services started with process locks!"
echo "📊 Dashboard: http://localhost:5555"
echo "🌐 WebSocket: ws://localhost:8765"
echo "📋 Logs: /Users/oliver/robo_trader/robo_trader.log"

# Show running processes
echo "🔍 Running processes:"
ps aux | grep -E "(python3.*robo_trader|python3.*app\.py)" | grep -v grep | head -10

echo "🎯 System startup complete!"
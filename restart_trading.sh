#!/bin/bash
# Clean restart script for AI trading system and dashboard
# Kills old instances and starts fresh

echo "🛑 Stopping existing processes..."

# Kill any existing Python trading processes
pkill -f "python.*ai_runner" 2>/dev/null
pkill -f "python.*start_ai_trading" 2>/dev/null
pkill -f "python.*runner" 2>/dev/null
pkill -f "python.*app.py" 2>/dev/null

# Wait a moment for processes to clean up
sleep 2

# Clear any IB connection on client ID 1
echo "📡 Clearing IB connections..."

# Activate virtual environment
source .venv/bin/activate

echo "🚀 Starting Dashboard..."
nohup python app.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "✅ Dashboard running at http://localhost:5555 (PID: $DASHBOARD_PID)"

# Wait for dashboard to start
sleep 3

# Check if dashboard is really running
if ! curl -s http://localhost:5555/api/status > /dev/null 2>&1; then
    echo "❌ Dashboard failed to start. Check dashboard.log for errors"
    exit 1
fi

echo ""
echo "🤖 Starting AI Trading System..."
echo "================================"

# Start with consistent client ID
export IBKR_CLIENT_ID=1
nohup python start_ai_trading.py > ai_trading.log 2>&1 &
AI_PID=$!
echo "✅ AI Trading System started (PID: $AI_PID)"

echo ""
echo "📊 Both systems are running!"
echo "   Dashboard: http://localhost:5555"
echo "   Logs: dashboard.log and ai_trading.log"
echo ""
echo "To stop: pkill -f 'python.*app.py' && pkill -f 'python.*start_ai_trading'"
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
python app.py &
DASHBOARD_PID=$!
echo "✅ Dashboard running at http://localhost:5555 (PID: $DASHBOARD_PID)"

echo ""
echo "🤖 Starting AI Trading System..."
echo "================================"

# Start with consistent client ID
export IBKR_CLIENT_ID=1
python start_ai_trading.py
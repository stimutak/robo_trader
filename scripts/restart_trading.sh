#!/bin/bash
#
# Restart Trading System
# 
# This script cleanly stops any existing trading processes and restarts
# both the dashboard and AI trading system.
#

set -e

echo "=========================================="
echo "🤖 Robo Trader System Restart"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    echo -n "Checking for $name on port $port... "
    
    local pid=$(lsof -ti:$port 2>/dev/null || true)
    if [ ! -z "$pid" ]; then
        echo -e "${YELLOW}Found PID $pid, killing...${NC}"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    else
        echo -e "${GREEN}Not running${NC}"
    fi
}

# Function to kill process by name
kill_process() {
    local pattern=$1
    local name=$2
    echo -n "Checking for $name processes... "
    
    local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}Found PIDs: $pids, killing...${NC}"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    else
        echo -e "${GREEN}Not running${NC}"
    fi
}

# Step 1: Stop existing processes
echo ""
echo "Step 1: Stopping existing processes..."
echo "--------------------------------------"
kill_port 5555 "Dashboard"
kill_process "app.py" "Dashboard"
kill_process "start_ai_trading.py" "AI Trading"
kill_process "robo_trader.runner" "Legacy Runner"

# Step 2: Check virtual environment
echo ""
echo "Step 2: Checking environment..."
echo "--------------------------------------"

if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -U pip setuptools wheel
    pip install -r requirements.txt
    pip install -e .
else
    echo -e "${GREEN}Virtual environment found${NC}"
    source .venv/bin/activate
fi

# Step 3: Check configuration
echo ""
echo "Step 3: Checking configuration..."
echo "--------------------------------------"

if [ -z "$IBKR_HOST" ] || [ -z "$IBKR_PORT" ] || [ -z "$IBKR_CLIENT_ID" ]; then
    echo -e "${YELLOW}Warning: IBKR environment variables not set${NC}"
    echo "Please ensure the following are set:"
    echo "  - IBKR_HOST (e.g., localhost)"
    echo "  - IBKR_PORT (e.g., 7497 for paper, 7496 for live)"
    echo "  - IBKR_CLIENT_ID (e.g., 1)"
    
    if [ ! -f ".env" ]; then
        echo ""
        echo "Creating default .env file..."
        cat > .env << EOF
# IBKR Configuration
IBKR_HOST=localhost
IBKR_PORT=7497
IBKR_CLIENT_ID=1

# Trading Mode (paper or live)
TRADING_MODE=paper

# Optional: LLM API Keys
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
EOF
        echo -e "${GREEN}.env file created with defaults${NC}"
    fi
    
    # Source .env file
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
fi

echo "Configuration:"
echo "  IBKR_HOST: ${IBKR_HOST:-not set}"
echo "  IBKR_PORT: ${IBKR_PORT:-not set}"
echo "  IBKR_CLIENT_ID: ${IBKR_CLIENT_ID:-not set}"
echo "  TRADING_MODE: ${TRADING_MODE:-paper}"

# Step 4: Start dashboard
echo ""
echo "Step 4: Starting dashboard..."
echo "--------------------------------------"

nohup python app.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard started with PID $DASHBOARD_PID"
echo "Waiting for dashboard to initialize..."
sleep 3

# Check if dashboard is running
if lsof -ti:5555 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dashboard running at http://localhost:5555${NC}"
else
    echo -e "${RED}✗ Dashboard failed to start. Check dashboard.log${NC}"
    exit 1
fi

# Step 5: Start AI trading
echo ""
echo "Step 5: Starting AI trading system..."
echo "--------------------------------------"

nohup python start_ai_trading.py > ai_trading.log 2>&1 &
TRADING_PID=$!
echo "AI Trading started with PID $TRADING_PID"
sleep 2

# Check if trading is running
if ps -p $TRADING_PID > /dev/null; then
    echo -e "${GREEN}✓ AI Trading system running${NC}"
else
    echo -e "${RED}✗ AI Trading failed to start. Check ai_trading.log${NC}"
    exit 1
fi

# Step 6: Summary
echo ""
echo "=========================================="
echo -e "${GREEN}🚀 System Started Successfully!${NC}"
echo "=========================================="
echo ""
echo "Dashboard:    http://localhost:5555"
echo "Dashboard PID: $DASHBOARD_PID (log: dashboard.log)"
echo "Trading PID:  $TRADING_PID (log: ai_trading.log)"
echo ""
echo "Commands:"
echo "  View dashboard log:  tail -f dashboard.log"
echo "  View trading log:    tail -f ai_trading.log"
echo "  Stop all:           ./restart_trading.sh"
echo "  Check status:       ps aux | grep -E 'app.py|start_ai_trading.py'"
echo ""
echo "Press Ctrl+C in the terminal running the scripts to stop them"
echo "Or run this script again to restart everything"
echo ""
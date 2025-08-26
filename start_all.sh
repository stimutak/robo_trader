#!/bin/bash
# Start all Robo Trader components
# This script starts both the dashboard and trading system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    Robo Trader Full System Startup${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if IB Gateway/TWS is running
echo -e "\n${YELLOW}Checking IB Gateway/TWS...${NC}"
if lsof -i :7497 &>/dev/null || lsof -i :7496 &>/dev/null; then
    echo -e "${GREEN}✓ IB Gateway/TWS detected${NC}"
else
    echo -e "${RED}✗ IB Gateway/TWS not detected${NC}"
    echo "Please start IB Gateway or TWS with API enabled on port 7497 (paper) or 7496 (live)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Kill any existing processes
echo -e "\n${YELLOW}Stopping existing processes...${NC}"
pkill -f "python.*app.py" 2>/dev/null || true
pkill -f "python.*start_ai_trading.py" 2>/dev/null || true
sleep 2

# Start dashboard in background
echo -e "\n${GREEN}Starting Dashboard...${NC}"
nohup python app.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID"

# Wait for dashboard to start
sleep 3

# Check if dashboard is running
if curl -s http://localhost:5555/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Dashboard running at http://localhost:5555${NC}"
else
    echo -e "${RED}✗ Dashboard failed to start. Check dashboard.log${NC}"
    exit 1
fi

# Start trading system
echo -e "\n${GREEN}Starting AI Trading System...${NC}"
nohup python start_ai_trading.py > ai_trading.log 2>&1 &
TRADING_PID=$!
echo "Trading System PID: $TRADING_PID"

# Save PIDs for later shutdown
echo "$DASHBOARD_PID" > .dashboard.pid
echo "$TRADING_PID" > .trading.pid

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}    All Systems Started Successfully${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Dashboard: http://localhost:5555"
echo "Logs:"
echo "  - Dashboard: tail -f dashboard.log"
echo "  - Trading: tail -f ai_trading.log"
echo ""
echo "To stop all systems: ./stop_all.sh"
echo ""
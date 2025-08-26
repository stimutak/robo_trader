#!/bin/bash
# Robo Trader Startup Script
# This script starts the AI trading system with the proper virtual environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    Robo Trader AI System Startup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check if IB Gateway/TWS is running
if ! lsof -i :7497 &>/dev/null && ! lsof -i :7496 &>/dev/null; then
    echo -e "${YELLOW}Warning: IB Gateway/TWS not detected on standard ports${NC}"
    echo "Please ensure IB Gateway or TWS is running with API enabled"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ IB Gateway/TWS detected${NC}"
fi

# Activate virtual environment and start trading
echo -e "${GREEN}Starting AI Trading System...${NC}"
source venv/bin/activate

# Kill any existing trading processes
pkill -f start_ai_trading.py 2>/dev/null || true

# Start the trading system
python start_ai_trading.py

# Note: The script will run in foreground. Use nohup or screen for background execution.
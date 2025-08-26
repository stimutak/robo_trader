#!/bin/bash
# Stop all Robo Trader components

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Robo Trader components...${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Stop using saved PIDs if available
if [ -f .dashboard.pid ]; then
    PID=$(cat .dashboard.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo -e "${GREEN}✓ Stopped dashboard (PID: $PID)${NC}"
    fi
    rm .dashboard.pid
fi

if [ -f .trading.pid ]; then
    PID=$(cat .trading.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo -e "${GREEN}✓ Stopped trading system (PID: $PID)${NC}"
    fi
    rm .trading.pid
fi

# Also kill by process name as backup
pkill -f "python.*app.py" 2>/dev/null && echo -e "${GREEN}✓ Stopped dashboard processes${NC}" || true
pkill -f "python.*start_ai_trading.py" 2>/dev/null && echo -e "${GREEN}✓ Stopped trading processes${NC}" || true

echo -e "${GREEN}All components stopped${NC}"
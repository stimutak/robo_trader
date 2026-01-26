#!/bin/bash
# Start the trading runner with Gateway checks and zombie cleanup
# This script is called by the dashboard Start button
# It does NOT restart the dashboard or WebSocket server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Get symbols from argument or user_settings.json
SYMBOLS="${1:-}"
if [ -z "$SYMBOLS" ]; then
    SYMBOLS=$(python3 -c "import json; print(','.join(json.load(open('user_settings.json')).get('default',{}).get('symbols',['AAPL','NVDA','TSLA'])))" 2>/dev/null || echo "AAPL,NVDA,TSLA")
fi

# Find Python
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON="venv/bin/python"
else
    PYTHON="python3"
fi

echo "=== Runner Startup ==="

# Step 1: Kill existing runner
echo "1. Stopping existing runner..."
pkill -9 -f "runner_async" 2>/dev/null && echo "   Killed existing runner" || echo "   No existing runner"

# Step 2: Check Gateway
echo "2. Checking Gateway..."
if ! lsof -nP -iTCP:4002 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "   ERROR: Gateway not listening on port 4002"
    echo "   Please start IB Gateway first"
    exit 1
fi
echo "   Gateway is listening"

# Step 3: Check for zombies
echo "3. Checking for zombie connections..."
ZOMBIES=$(lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT 2>/dev/null | grep -v "^COMMAND" | wc -l | tr -d ' ')
if [ "$ZOMBIES" -gt 0 ]; then
    echo "   WARNING: Found $ZOMBIES zombie connections"
    echo "   Attempting cleanup..."

    # Kill Python processes that might own zombies
    pkill -9 -f "ibkr_subprocess_worker" 2>/dev/null
    sleep 2

    # Recheck
    ZOMBIES=$(lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT 2>/dev/null | grep -v "^COMMAND" | wc -l | tr -d ' ')
    if [ "$ZOMBIES" -gt 0 ]; then
        echo "   WARNING: Still have $ZOMBIES zombies - Gateway restart may be needed"
    else
        echo "   Zombies cleared"
    fi
else
    echo "   No zombies found"
fi

# Step 4: Start runner
echo "4. Starting runner..."
echo "   Symbols: $SYMBOLS"

export LOG_FILE="$SCRIPT_DIR/robo_trader.log"
$PYTHON -m robo_trader.runner_async --symbols "$SYMBOLS" --force-connect &
RUNNER_PID=$!

sleep 3

if ps -p $RUNNER_PID > /dev/null 2>&1; then
    echo "   Runner started (PID: $RUNNER_PID)"
    echo "=== SUCCESS ==="
    exit 0
else
    echo "   ERROR: Runner failed to start"
    echo "   Check robo_trader.log for details"
    exit 1
fi

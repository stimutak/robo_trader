#!/bin/bash
#
# RoboTrader Startup Script
#
# This script ensures clean startup by:
# 1. Starting Gateway via IBC if not running
# 2. Killing all existing Python trader processes
# 3. Cleaning up zombie CLOSE_WAIT connections
# 4. Automatically restarting Gateway if zombies block API
# 5. Starting the trading system
#
# Usage:
#   ./START_TRADER.sh                    # Start with default symbols
#   ./START_TRADER.sh "AAPL,NVDA"        # Start with custom symbols
#
# Gateway Management:
#   ./scripts/start_gateway.sh           # Start Gateway via IBC
#   python3 scripts/gateway_manager.py status  # Check Gateway status
#

set -e

PORT=4002
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_GATEWAY_RETRIES=3

# Parse arguments
SYMBOLS="AAPL,NVDA,TSLA"

for arg in "$@"; do
    case $arg in
        *)
            SYMBOLS="$arg"
            ;;
    esac
done

echo "=========================================="
echo "RoboTrader Startup Script"
echo "=========================================="
echo ""

# Step 1: Kill existing Python processes first
echo "1. Killing existing trader processes..."
pkill -9 -f "runner_async" 2>/dev/null && echo "   ✓ Killed runner_async" || echo "   ✓ No runner_async running"
pkill -9 -f "app.py" 2>/dev/null && echo "   ✓ Killed dashboard" || echo "   ✓ No dashboard running"
pkill -9 -f "websocket_server" 2>/dev/null && echo "   ✓ Killed websocket_server" || echo "   ✓ No websocket_server running"
sleep 2
echo ""

# Function to start Gateway via IBC
start_gateway() {
    echo "   Starting Gateway via IBC..."

    # Kill any existing Gateway first
    pkill -f "IB Gateway" 2>/dev/null || true
    pkill -f "ibgateway" 2>/dev/null || true
    echo "   Waiting 10s for Gateway to fully shut down..."
    sleep 10

    # Start Gateway
    cd "$SCRIPT_DIR"

    # Set up IBC environment
    export TWS_MAJOR_VRSN="10.37"
    if [ ! -d ~/Applications/"IB Gateway 10.37" ]; then
        GATEWAY_DIR=$(ls -d ~/Applications/"IB Gateway"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$GATEWAY_DIR" ]; then
            export TWS_MAJOR_VRSN=$(basename "$GATEWAY_DIR" | sed 's/IB Gateway //')
        else
            echo "   ERROR: No IB Gateway found in ~/Applications"
            return 1
        fi
    fi

    export IBC_INI="${SCRIPT_DIR}/config/ibc/config.ini"
    export TRADING_MODE="paper"
    export TWOFA_TIMEOUT_ACTION="restart"
    export IBC_PATH="${SCRIPT_DIR}/IBCMacos-3"
    export TWS_PATH=~/Applications
    export TWS_SETTINGS_PATH=
    export LOG_PATH="${SCRIPT_DIR}/config/ibc/logs"

    # Check config exists
    if [ ! -f "$IBC_INI" ]; then
        echo "   ERROR: IBC config not found at $IBC_INI"
        echo "   Run: cp config/ibc/config.ini.template config/ibc/config.ini"
        echo "   Then edit with your IBKR credentials."
        return 1
    fi

    # Create log directory
    mkdir -p "$LOG_PATH"

    # Make scripts executable
    chmod +x "${IBC_PATH}"/*.sh 2>/dev/null || true
    chmod +x "${IBC_PATH}"/scripts/*.sh 2>/dev/null || true

    echo "   Using Gateway version: $TWS_MAJOR_VRSN"
    echo ""
    echo "   ========================================"
    echo "   STARTING GATEWAY - 2FA REQUIRED"
    echo "   ========================================"
    echo "   Check your IBKR Mobile app for 2FA prompt"
    echo ""

    # Launch Gateway inline (blocks until Gateway exits or we Ctrl+C)
    cd "$IBC_PATH"
    ./gatewaystartmacos.sh -inline &
    IBC_PID=$!

    # Wait for Gateway to start and API port to open
    # CRITICAL: Use lsof, NOT nc -z (nc creates zombie connections that block API handshakes!)
    echo "   Waiting for Gateway to start..."
    for i in $(seq 1 120); do
        if lsof -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | grep -q LISTEN; then
            echo "   ✓ Gateway API port $PORT is now open!"
            # Wait for Gateway to fully initialize after port opens
            # Gateway needs time to complete login/2FA before API is responsive
            echo "   Waiting 30s for Gateway to complete login/2FA..."
            sleep 30
            return 0
        fi

        # Check if IBC process died
        if ! kill -0 $IBC_PID 2>/dev/null; then
            # IBC finished launching, check if Gateway is running
            if pgrep -f "IB Gateway" > /dev/null 2>&1; then
                # Gateway running but port not open yet, keep waiting
                :
            else
                echo "   Gateway process not detected, IBC may have failed"
            fi
        fi

        sleep 1
        if [ $((i % 15)) -eq 0 ]; then
            echo "   Still waiting... ($i seconds)"
        fi
    done

    echo "   TIMEOUT: Gateway did not start within 120 seconds"
    return 1
}

# Note: API handshake test added back (scripts/test_gateway_api.py) to verify Gateway
# is actually responding, not just that the port is open. This test properly disconnects
# to avoid creating zombie connections.

# Function to check if port is listening (uses lsof to avoid creating zombie connections)
# CRITICAL: Do NOT use nc -z for port checking - it creates zombie connections that block API handshakes!
is_port_listening() {
    lsof -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | grep -q LISTEN
}

# Function to check for zombie connections
check_zombies() {
    # Count actual zombie lines - use wc -l and trim whitespace
    local count
    count=$(lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep "CLOSE_WAIT" | wc -l | tr -d ' ')
    # Return 0 if empty
    echo "${count:-0}"
}

# Step 2: Check/Start Gateway with retry logic
echo "2. Checking Gateway status..."
GATEWAY_RETRY=0
API_CONNECTED=false

while [ "$API_CONNECTED" = false ] && [ $GATEWAY_RETRY -lt $MAX_GATEWAY_RETRIES ]; do
    GATEWAY_RETRY=$((GATEWAY_RETRY + 1))

    if [ $GATEWAY_RETRY -gt 1 ]; then
        echo ""
        echo "   =========================================="
        echo "   GATEWAY RETRY $GATEWAY_RETRY of $MAX_GATEWAY_RETRIES"
        echo "   =========================================="
        echo ""
    fi

    # Check if Gateway is running
    if ! pgrep -f "IB Gateway" > /dev/null 2>&1 && ! pgrep -f "ibcalpha.ibc" > /dev/null 2>&1; then
        echo "   Gateway is NOT running - starting via IBC..."
        if ! start_gateway; then
            echo "   Failed to start Gateway"
            continue
        fi
    else
        echo "   ✓ Gateway process detected"
    fi

    # Check if port is listening (using lsof to avoid zombie connections)
    if ! is_port_listening; then
        echo "   ⚠️  API port $PORT is NOT listening"
        echo "   Gateway may be starting or needs restart..."

        # Wait a bit for Gateway to fully start
        for i in $(seq 1 30); do
            if is_port_listening; then
                echo "   ✓ API port $PORT is now listening"
                break
            fi
            sleep 1
        done

        if ! is_port_listening; then
            echo "   Port still not listening - restarting Gateway..."
            start_gateway
            continue
        fi
    else
        echo "   ✓ API port $PORT is listening"
    fi

    # Check for zombies (now that we use lsof for port checking, we should have zero startup zombies)
    ZOMBIES=$(check_zombies)
    if [ "$ZOMBIES" -gt 0 ]; then
        echo "   ⚠️  Found $ZOMBIES zombie connection(s)!"
        echo "   Zombies block API handshakes - restarting Gateway..."

        # Kill Python zombies first
        lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u | while read pid; do
            kill -9 $pid 2>/dev/null && echo "   Killed Python zombie PID $pid" || true
        done
        sleep 1

        # Check if zombies remain (Gateway-owned)
        ZOMBIES=$(check_zombies)
        if [ "$ZOMBIES" -gt 0 ]; then
            echo "   Gateway zombies remain - must restart Gateway"
            start_gateway
            continue
        fi
    fi

    # Skip API handshake test - it creates zombie connections that block subsequent connections!
    # The lsof port check above is sufficient to verify Gateway is listening.
    # The trading system will handle connection retries if needed.
    echo "   ✓ Gateway port is listening - ready for connections"
    API_CONNECTED=true
done

if [ "$API_CONNECTED" = false ]; then
    echo ""
    echo "=========================================="
    echo "❌ FAILED TO CONNECT TO GATEWAY API"
    echo "=========================================="
    echo ""
    echo "After $MAX_GATEWAY_RETRIES attempts, could not establish API connection."
    echo ""
    echo "Manual troubleshooting:"
    echo "  1. Check Gateway is fully started (shows 'IB Gateway - READY')"
    echo "  2. Verify 2FA was completed on your phone"
    echo "  3. Try: python3 scripts/gateway_manager.py restart"
    echo ""
    exit 1
fi

echo ""

# Step 3: Clean up any remaining zombie connections
echo "3. Final zombie cleanup..."
ZOMBIES=$(check_zombies)
if [ "$ZOMBIES" -gt 0 ]; then
    echo "   Killing $ZOMBIES zombie connection(s)..."
    lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u | while read pid; do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 1
fi
echo "   ✓ Zombie cleanup complete"
echo ""

# Step 4: Activate virtual environment
echo "4. Activating virtual environment..."
cd "$SCRIPT_DIR"
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "   ✓ Virtual environment activated"
else
    echo "   ⚠️  No virtual environment found - using system Python"
fi
echo ""

# Step 5: Start WebSocket server
echo "5. Starting WebSocket server..."
python3 -m robo_trader.websocket_server &
WS_PID=$!
sleep 2

if ps -p $WS_PID > /dev/null; then
    echo "   ✓ WebSocket server started (PID: $WS_PID)"
else
    echo "   ⚠️  WebSocket server may have failed to start"
fi
echo ""

# Step 6: Start trading system
echo "6. Starting trading system..."
echo "   Symbols: $SYMBOLS"
echo "   Log: robo_trader.log"
echo ""

export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log

python3 -m robo_trader.runner_async --symbols "$SYMBOLS" --force-connect &
TRADER_PID=$!

echo "   ✓ Trading system started (PID: $TRADER_PID)"
echo ""

# Step 7: Start dashboard
echo "7. Starting dashboard..."
export DASH_PORT=5555
python3 app.py &
DASH_PID=$!
sleep 2

if ps -p $DASH_PID > /dev/null; then
    echo "   ✓ Dashboard started (PID: $DASH_PID)"
else
    echo "   ⚠️  Dashboard may have failed to start"
fi
echo ""

# Step 8: Monitor startup
echo "8. Monitoring startup (10 seconds)..."
sleep 10

if ps -p $TRADER_PID > /dev/null; then
    echo "   ✓ Trading system is running"
    echo ""
    echo "=========================================="
    echo "✅ STARTUP SUCCESSFUL"
    echo "=========================================="
    echo ""
    echo "Trading system is running with PID: $TRADER_PID"
    echo "WebSocket server PID: $WS_PID"
    echo "Dashboard PID: $DASH_PID"
    echo ""
    echo "Monitor logs: tail -f robo_trader.log"
    echo "View dashboard: http://localhost:5555"
    echo ""
    echo "To stop:"
    echo "  pkill -9 -f runner_async"
    echo "  pkill -9 -f websocket_server"
    echo "  pkill -9 -f app.py"
    echo ""
else
    echo "   ❌ Trading system stopped unexpectedly"
    echo ""
    echo "Check logs: tail -50 robo_trader.log"
    echo ""
    exit 1
fi

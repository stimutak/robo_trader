#!/bin/bash
#
# RoboTrader Startup Script with Zombie Cleanup
#
# This script ensures clean startup by:
# 1. Auto-detecting Gateway or TWS and selecting correct port
# 2. Killing all existing Python trader processes
# 3. Cleaning up zombie CLOSE_WAIT connections
# 4. Testing IBKR connectivity
# 5. Starting the trading system
#

set -e

# Function to auto-detect IBKR service and port
detect_ibkr_port() {
    # Check for Gateway (preferred)
    if pgrep -f "ibgateway" > /dev/null 2>&1; then
        # Gateway is running
        if netstat -an 2>/dev/null | grep -q ":4002.*LISTEN"; then
            echo "4002"
            return 0
        elif netstat -an 2>/dev/null | grep -q ":4001.*LISTEN"; then
            echo "4001"
            return 0
        else
            # Gateway running but no port detected, default to paper
            echo "4002"
            return 0
        fi
    fi

    # Check for TWS
    if pgrep -f "tws" > /dev/null 2>&1; then
        # TWS is running
        if netstat -an 2>/dev/null | grep -q ":7497.*LISTEN"; then
            echo "7497"
            return 0
        elif netstat -an 2>/dev/null | grep -q ":7496.*LISTEN"; then
            echo "7496"
            return 0
        else
            # TWS running but no port detected, default to paper
            echo "7497"
            return 0
        fi
    fi

    # No process detected, check for listening ports
    if netstat -an 2>/dev/null | grep -q ":4002.*LISTEN"; then
        echo "4002"
        return 0
    elif netstat -an 2>/dev/null | grep -q ":4001.*LISTEN"; then
        echo "4001"
        return 0
    elif netstat -an 2>/dev/null | grep -q ":7497.*LISTEN"; then
        echo "7497"
        return 0
    elif netstat -an 2>/dev/null | grep -q ":7496.*LISTEN"; then
        echo "7496"
        return 0
    fi

    # Nothing detected, default to Gateway paper port
    echo "4002"
    return 0
}

# Auto-detect port
PORT=$(detect_ibkr_port)
SYMBOLS="${1:-AAPL,NVDA,TSLA}"

# Determine service name for display
if [ "$PORT" = "4002" ] || [ "$PORT" = "4001" ]; then
    SERVICE_NAME="Gateway"
    ENV_NAME=$( [ "$PORT" = "4002" ] && echo "Paper" || echo "Live" )
else
    SERVICE_NAME="TWS"
    ENV_NAME=$( [ "$PORT" = "7497" ] && echo "Paper" || echo "Live" )
fi

echo "=========================================="
echo "RoboTrader Startup Script"
echo "=========================================="
echo "Detected: $SERVICE_NAME $ENV_NAME (port $PORT)"
echo ""

# Step 1: Kill existing processes
echo "1. Killing existing trader processes..."
pkill -9 -f "runner_async" 2>/dev/null && echo "   ✓ Killed runner_async" || echo "   ✓ No runner_async running"
pkill -9 -f "app.py" 2>/dev/null && echo "   ✓ Killed dashboard" || echo "   ✓ No dashboard running"
pkill -9 -f "websocket_server" 2>/dev/null && echo "   ✓ Killed websocket_server" || echo "   ✓ No websocket_server running"
sleep 2
echo ""

# Step 2: Clean up zombie connections
echo "2. Cleaning up zombie connections..."
ZOMBIE_COUNT=$(netstat -an | grep "$PORT" | grep "CLOSE_WAIT" | wc -l | tr -d ' ')

if [ "$ZOMBIE_COUNT" -gt 0 ]; then
    echo "   Found $ZOMBIE_COUNT zombie connection(s)"
    
    # Kill Python zombies
    PYTHON_ZOMBIES=$(lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u || true)
    
    if [ -n "$PYTHON_ZOMBIES" ]; then
        echo "   Killing Python zombie processes..."
        for PID in $PYTHON_ZOMBIES; do
            kill -9 $PID 2>/dev/null && echo "   ✓ Killed PID $PID" || true
        done
        sleep 1
    fi
    
    # Check if Gateway zombies remain
    REMAINING=$(netstat -an | grep "$PORT" | grep "CLOSE_WAIT" | wc -l | tr -d ' ')
    if [ "$REMAINING" -gt 0 ]; then
        echo "   ⚠️  $REMAINING Gateway-owned zombie(s) remain"
        echo "   These may block connections - Gateway restart may be needed"
    else
        echo "   ✓ All zombies cleaned up"
    fi
else
    echo "   ✓ No zombie connections found"
fi
echo ""

# Step 3: Test IBKR connectivity
echo "3. Testing $SERVICE_NAME connectivity on port $PORT..."

# Create test script with detected port
cat > /tmp/test_ibkr.py << PYEOF
import asyncio
import sys
from ib_async import IB

async def test():
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", $PORT, clientId=999, readonly=True, timeout=5.0),
            timeout=7.0
        )
        print("   ✅ $SERVICE_NAME connection successful!")
        ib.disconnect()
        return 0
    except asyncio.TimeoutError:
        print("   ❌ $SERVICE_NAME connection TIMEOUT")
        print("")
        print("   $SERVICE_NAME is NOT responding to API requests on port $PORT.")
        print("   This will prevent the trading system from starting.")
        print("")
        print("   Possible solutions:")
        print("   1. Check $SERVICE_NAME API settings (File → Global Configuration → API)")
        print("   2. Ensure 'Enable ActiveX and Socket Clients' is checked")
        print("   3. Add 127.0.0.1 to Trusted IPs")
        print("   4. Restart $SERVICE_NAME (close and relaunch with 2FA)")
        print("   5. Check firewall/security software")
        print("")
        try:
            ib.disconnect()
        except:
            pass
        return 1
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        try:
            ib.disconnect()
        except:
            pass
        return 1

sys.exit(asyncio.run(test()))
PYEOF

python3 /tmp/test_ibkr.py
TEST_RESULT=$?
rm -f /tmp/test_ibkr.py /tmp/test_gateway.py

if [ $TEST_RESULT -ne 0 ]; then
    echo "=========================================="
    echo "❌ STARTUP ABORTED"
    echo "=========================================="
    echo ""
    echo "Cannot start trading system - $SERVICE_NAME not responding on port $PORT."
    echo ""
    echo "Run diagnostics: python3 diagnose_gateway_api.py"
    echo "Or force reconnect: ./force_gateway_reconnect.sh"
    echo ""
    exit 1
fi
echo ""

# Step 4: Activate virtual environment
echo "4. Activating virtual environment..."
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
echo "   Port: $PORT ($SERVICE_NAME $ENV_NAME)"
echo "   Log: robo_trader.log"
echo ""

export LOG_FILE="$(pwd)/robo_trader.log"
export IBKR_PORT=$PORT  # Pass detected port to runner

python3 -m robo_trader.runner_async --symbols "$SYMBOLS" &
TRADER_PID=$!

echo "   ✓ Trading system started (PID: $TRADER_PID)"
echo ""

# Step 7: Monitor startup
echo "7. Monitoring startup (10 seconds)..."
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
    echo "Connected to: $SERVICE_NAME $ENV_NAME (port $PORT)"
    echo ""
    echo "Monitor logs: tail -f robo_trader.log"
    echo "View dashboard: http://localhost:5555"
    echo ""
    echo "To stop:"
    echo "  pkill -9 -f runner_async"
    echo "  pkill -9 -f websocket_server"
    echo ""
else
    echo "   ❌ Trading system stopped unexpectedly"
    echo ""
    echo "Check logs: tail -50 robo_trader.log"
    echo ""
    exit 1
fi


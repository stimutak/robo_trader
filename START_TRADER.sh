#!/bin/bash
#
# RoboTrader Startup Script with Zombie Cleanup
#
# This script ensures clean startup by:
# 1. Killing all existing Python trader processes
# 2. Cleaning up zombie CLOSE_WAIT connections
# 3. Testing Gateway connectivity
# 4. Starting the trading system
#

set -e

PORT=4002
SYMBOLS="${1:-AAPL,NVDA,TSLA}"

echo "=========================================="
echo "RoboTrader Startup Script"
echo "=========================================="
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

# Step 3: Activate virtual environment
echo "3. Activating virtual environment..."
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "   ✓ Virtual environment activated"
else
    echo "   ⚠️  No virtual environment found - using system Python"
fi
echo ""

# Step 4: Test Gateway connectivity
echo "4. Testing Gateway connectivity..."

# Create test script
cat > /tmp/test_gateway.py << 'PYEOF'
import asyncio
import sys
from ib_async import IB

async def test():
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", 4002, clientId=999, readonly=True, timeout=5.0),
            timeout=7.0
        )
        print("   ✅ Gateway connection successful!")
        ib.disconnect()
        return 0
    except asyncio.TimeoutError:
        print("   ❌ Gateway connection TIMEOUT")
        print("")
        print("   Gateway is NOT responding to API requests.")
        print("   This will prevent the trading system from starting.")
        print("")
        print("   Possible solutions:")
        print("   1. Check Gateway API settings (File → Global Configuration → API)")
        print("   2. Restart Gateway (close and relaunch with 2FA)")
        print("   3. Check firewall/security software")
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

python3 /tmp/test_gateway.py
TEST_RESULT=$?
rm -f /tmp/test_gateway.py

if [ $TEST_RESULT -ne 0 ]; then
    echo "=========================================="
    echo "❌ STARTUP ABORTED"
    echo "=========================================="
    echo ""
    echo "Cannot start trading system - Gateway not responding."
    echo ""
    echo "Run diagnostics: python3 diagnose_gateway_api.py"
    echo "Or force reconnect: ./force_gateway_reconnect.sh"
    echo ""
    exit 1
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

python3 -m robo_trader.runner_async --symbols "$SYMBOLS" &
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


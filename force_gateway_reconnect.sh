#!/bin/bash
#
# Force Gateway to Accept New Connections
#
# This script attempts to clear Gateway's connection state without restarting it.
# It kills any zombie CLOSE_WAIT connections and tests if Gateway will accept new API connections.
#

set -e

PORT=4002
echo "=========================================="
echo "Gateway Connection Reset Tool"
echo "=========================================="
echo ""

# Step 1: Check if Gateway is running
echo "1. Checking Gateway process..."
GATEWAY_PID=$(ps aux | grep -i "gateway\|tws" | grep -i java | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$GATEWAY_PID" ]; then
    echo "❌ Gateway is not running!"
    echo "   Please start IB Gateway first."
    exit 1
fi

echo "✅ Gateway running (PID: $GATEWAY_PID)"
echo ""

# Step 2: Check for zombie connections
echo "2. Checking for zombie CLOSE_WAIT connections..."
ZOMBIE_COUNT=$(netstat -an | grep "$PORT" | grep "CLOSE_WAIT" | wc -l | tr -d ' ')

if [ "$ZOMBIE_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ZOMBIE_COUNT zombie connection(s)"
    echo ""
    echo "   Zombie connections owned by Gateway:"
    lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null || echo "   (lsof not available)"
    echo ""
    echo "   NOTE: Gateway-owned zombies CANNOT be killed without restarting Gateway."
    echo "   These zombies may block new API connections."
    echo ""
else
    echo "✅ No zombie connections found"
    echo ""
fi

# Step 3: Kill any Python zombie processes (safe to kill)
echo "3. Cleaning up Python zombie processes..."
PYTHON_ZOMBIES=$(lsof -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u)

if [ -n "$PYTHON_ZOMBIES" ]; then
    echo "   Killing Python zombie processes..."
    for PID in $PYTHON_ZOMBIES; do
        echo "   Killing PID $PID..."
        kill -9 $PID 2>/dev/null || echo "   (already dead)"
    done
    sleep 1
    echo "✅ Python zombies cleaned up"
else
    echo "✅ No Python zombies to clean up"
fi
echo ""

# Step 4: Test if Gateway accepts connections
echo "4. Testing if Gateway accepts new API connections..."
echo "   (This will attempt a connection with client_id=777)"
echo ""

# Create a simple Python test script
cat > /tmp/test_gateway_accept.py << 'PYEOF'
import asyncio
import sys
from ib_async import IB

async def test():
    ib = IB()
    try:
        await asyncio.wait_for(
            ib.connectAsync("127.0.0.1", 4002, clientId=777, readonly=True, timeout=5.0),
            timeout=7.0
        )
        print("✅ Gateway ACCEPTS connections - API handshake successful!")
        print(f"   Accounts: {ib.managedAccounts()}")
        ib.disconnect()
        return 0
    except asyncio.TimeoutError:
        print("❌ Gateway REJECTS connections - API handshake timeout")
        print("   This means Gateway is NOT responding to API protocol messages.")
        try:
            ib.disconnect()
        except:
            pass
        return 1
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        try:
            ib.disconnect()
        except:
            pass
        return 1

sys.exit(asyncio.run(test()))
PYEOF

python3 /tmp/test_gateway_accept.py
TEST_RESULT=$?
echo ""

# Step 5: Recommendations based on test result
if [ $TEST_RESULT -eq 0 ]; then
    echo "=========================================="
    echo "✅ SUCCESS - Gateway is accepting connections!"
    echo "=========================================="
    echo ""
    echo "Your trading system should now be able to connect."
    echo "Run: python3 -m robo_trader.runner_async --symbols AAPL"
    echo ""
else
    echo "=========================================="
    echo "❌ PROBLEM - Gateway is NOT accepting connections"
    echo "=========================================="
    echo ""
    echo "Possible causes:"
    echo ""
    echo "1. Gateway API settings not configured:"
    echo "   → Open Gateway: File → Global Configuration → API → Settings"
    echo "   → Check: ☑️ Enable ActiveX and Socket Clients"
    echo "   → Add: 127.0.0.1 to Trusted IPs"
    echo "   → Port: 4002"
    echo "   → Click Apply"
    echo ""
    echo "2. Gateway needs restart (zombie connections blocking):"
    if [ "$ZOMBIE_COUNT" -gt 0 ]; then
        echo "   → You have $ZOMBIE_COUNT zombie connection(s)"
        echo "   → Close Gateway completely"
        echo "   → Relaunch and login (requires 2FA)"
        echo "   → Run this script again to verify"
    else
        echo "   → No zombies detected, but Gateway may still need restart"
        echo "   → Close Gateway completely"
        echo "   → Relaunch and login (requires 2FA)"
    fi
    echo ""
    echo "3. Firewall/security software blocking API protocol:"
    echo "   → Check macOS Security & Privacy settings"
    echo "   → Check if any antivirus is blocking connections"
    echo ""
fi

# Cleanup
rm -f /tmp/test_gateway_accept.py

exit $TEST_RESULT


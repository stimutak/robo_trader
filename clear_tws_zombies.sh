#!/bin/bash
# Clear TWS zombie connections

echo "=== TWS Zombie Connection Cleaner ==="
echo ""

# Check for zombie connections
echo "Checking for zombie connections on TWS port 7497..."
ZOMBIES=$(lsof -i :7497 | grep -E "CLOSED|CLOSE_WAIT" | wc -l)

if [ $ZOMBIES -gt 0 ]; then
    echo "❌ Found $ZOMBIES zombie connection(s)"
    echo ""
    echo "Zombie connections detected:"
    lsof -i :7497 | grep -E "CLOSED|CLOSE_WAIT"
    echo ""
    echo "ACTION REQUIRED:"
    echo "1. In TWS: File → Exit (completely close TWS)"
    echo "2. Wait 5 seconds"
    echo "3. Restart TWS and log in"
    echo "4. The trading system will automatically reconnect"
else
    echo "✅ No zombie connections found"

    # Check if TWS is listening
    if lsof -i :7497 | grep LISTEN > /dev/null; then
        echo "✅ TWS is listening on port 7497"

        # Check active connections
        ACTIVE=$(lsof -i :7497 | grep ESTABLISHED | wc -l)
        if [ $ACTIVE -gt 0 ]; then
            echo "✅ $ACTIVE active connection(s) to TWS"
        else
            echo "⚠️  No active connections - trading system may not be connected"
        fi
    else
        echo "❌ TWS is not listening on port 7497"
        echo "   Please start TWS"
    fi
fi

echo ""
echo "Current TWS connections:"
lsof -i :7497
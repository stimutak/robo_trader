#!/bin/bash
# Monitor zombie CLOSE_WAIT connections on port 7497
# These are leftover from failed TWS API handshake attempts
# NO SIGNALS SENT - monitoring only

echo "Checking for zombie connections on port 7497..."
ZOMBIE_COUNT=$(netstat -an | grep 7497 | grep CLOSE_WAIT | wc -l | tr -d ' ')

if [ "$ZOMBIE_COUNT" -gt 0 ]; then
    echo "⚠️  Found $ZOMBIE_COUNT zombie CLOSE_WAIT connections"

    # Try to find TWS process (java process listening on 7497)
    TWS_PID=$(lsof -ti:7497 -sTCP:LISTEN 2>/dev/null | head -1)

    if [ -n "$TWS_PID" ]; then
        echo "ℹ️  TWS process running: PID $TWS_PID"
    else
        echo "⚠️  TWS process not found on port 7497"
    fi

    echo "ℹ️  Zombies are on TWS side (CLOSE_WAIT) - monitoring only, no action taken"
    echo "ℹ️  They are harmless and will clear on next TWS restart"
else
    echo "✅ No zombie connections found"
fi

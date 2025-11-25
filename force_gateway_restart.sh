#!/bin/bash

# Gateway Force Restart Procedure
# This script performs a complete Gateway restart to clear internal API state corruption

echo "=========================================="
echo "Gateway Force Restart Procedure"
echo "=========================================="
echo "This will completely restart Gateway to clear internal API state issues."
echo ""

# Step 1: Identify Gateway process
echo "Step 1: Identifying Gateway process..."
GATEWAY_PID=$(ps aux | grep -i "JavaAppli.*gateway\|JavaAppli.*tws" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$GATEWAY_PID" ]; then
    echo "❌ No Gateway process found"
    echo "Please start Gateway manually and run this script again"
    exit 1
fi

echo "✅ Found Gateway process: PID $GATEWAY_PID"

# Step 2: Check current connections
echo ""
echo "Step 2: Checking current connections..."
echo "Port 4002 connections:"
lsof -nP -iTCP:4002 2>/dev/null || echo "No connections found"

echo ""
echo "CLOSE_WAIT zombies:"
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT 2>/dev/null || echo "No zombies found"

# Step 3: Kill Python processes first
echo ""
echo "Step 3: Stopping Python trading processes..."
pkill -9 -f "runner_async" 2>/dev/null && echo "✅ Killed runner_async" || echo "ℹ️ No runner_async running"
pkill -9 -f "app.py" 2>/dev/null && echo "✅ Killed dashboard" || echo "ℹ️ No dashboard running"
pkill -9 -f "websocket_server" 2>/dev/null && echo "✅ Killed websocket_server" || echo "ℹ️ No websocket_server running"

sleep 2

# Step 4: Force kill Gateway
echo ""
echo "Step 4: Force killing Gateway process..."
echo "⚠️ WARNING: This will require 2FA re-authentication"
echo ""
read -p "Continue with Gateway force kill? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 1
fi

echo "Force killing Gateway PID $GATEWAY_PID..."
kill -9 $GATEWAY_PID

# Wait for process to die
sleep 3

# Verify it's dead
if ps -p $GATEWAY_PID > /dev/null 2>&1; then
    echo "❌ Gateway process still running - manual intervention required"
    exit 1
else
    echo "✅ Gateway process terminated"
fi

# Step 5: Clean up any remaining connections
echo ""
echo "Step 5: Cleaning up remaining connections..."
sleep 2

# Check for any remaining connections
REMAINING=$(lsof -nP -iTCP:4002 2>/dev/null | wc -l)
if [ $REMAINING -gt 0 ]; then
    echo "⚠️ Found $REMAINING remaining connections on port 4002"
    lsof -nP -iTCP:4002
else
    echo "✅ Port 4002 is clean"
fi

# Step 6: Instructions for restart
echo ""
echo "Step 6: Gateway restart instructions"
echo "=========================================="
echo "1. Launch IB Gateway application"
echo "2. Complete 2FA authentication"
echo "3. Wait for Gateway to show 'Connected' status"
echo "4. Verify API settings:"
echo "   - File → Global Configuration → API → Settings"
echo "   - ✅ Enable ActiveX and Socket Clients"
echo "   - Port: 4002 (paper trading)"
echo "   - Add 127.0.0.1 to Trusted IPs"
echo "5. Wait 1-2 minutes for full initialization"
echo ""
echo "After Gateway restart, run:"
echo "  python3 diagnose_gateway_internal_state.py"
echo ""
echo "If diagnostics pass, start trading system:"
echo "  ./START_TRADER.sh AAPL"
echo ""
echo "=========================================="
echo "Gateway force restart procedure complete"
echo "Manual restart and 2FA required"
echo "=========================================="

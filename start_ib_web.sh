#!/bin/bash
# Start IB Client Portal Gateway for Web API access

echo "Starting IB Client Portal Gateway..."
echo "=================================="
echo ""
echo "This provides the Web API for trading without TWS/IB Gateway desktop app."
echo ""
echo "Steps:"
echo "1. Gateway will start on https://localhost:5000"
echo "2. Your browser will open automatically"
echo "3. Login with your IB credentials (use paper trading account)"
echo "4. Keep this terminal open while trading"
echo ""
echo "Press Ctrl+C to stop the gateway"
echo ""

cd clientportal.gw

# Start the gateway
./bin/run.sh root/conf.yaml
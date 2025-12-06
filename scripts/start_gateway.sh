#!/bin/bash
#=============================================================================
# RoboTrader Gateway Start Script
#
# Starts IB Gateway using IBC with proper configuration.
# This script is cross-platform aware but this version is for macOS.
#
# Usage:
#   ./start_gateway.sh [paper|live]
#
# Default is paper trading mode.
#=============================================================================

set -e

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Trading mode (default: paper)
TRADING_MODE="${1:-paper}"

# Gateway version - auto-detect or set manually
# Prefer 10.37 (stable) over 10.41 if available
if [ -z "$TWS_MAJOR_VRSN" ]; then
    # Prefer 10.37 if available (known stable version)
    if [ -d ~/Applications/"IB Gateway 10.37" ]; then
        TWS_MAJOR_VRSN="10.37"
    else
        # Fall back to latest installed version
        GATEWAY_DIR=$(ls -d ~/Applications/"IB Gateway"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$GATEWAY_DIR" ]; then
            TWS_MAJOR_VRSN=$(basename "$GATEWAY_DIR" | sed 's/IB Gateway //')
        else
            echo "ERROR: No IB Gateway found in ~/Applications"
            echo "Please install IB Gateway first."
            exit 1
        fi
    fi
fi

echo "=========================================="
echo "RoboTrader Gateway Launcher"
echo "=========================================="
echo ""
echo "Gateway Version: $TWS_MAJOR_VRSN"
echo "Trading Mode: $TRADING_MODE"
echo ""

# IBC Configuration - stored in project for portability
export TWS_MAJOR_VRSN
export IBC_INI="${PROJECT_ROOT}/config/ibc/config.ini"
export TRADING_MODE
export TWOFA_TIMEOUT_ACTION=restart
export IBC_PATH="${PROJECT_ROOT}/IBCMacos-3"
export TWS_PATH=~/Applications
export TWS_SETTINGS_PATH=
export LOG_PATH="${PROJECT_ROOT}/config/ibc/logs"

# Check IBC config exists
if [ ! -f "$IBC_INI" ]; then
    echo "ERROR: IBC config not found at $IBC_INI"
    echo ""

    # Check for template
    TEMPLATE="${PROJECT_ROOT}/config/ibc/config.ini.template"
    if [ -f "$TEMPLATE" ]; then
        echo "Creating config from template..."
        mkdir -p "$(dirname "$IBC_INI")"
        cp "$TEMPLATE" "$IBC_INI"
        echo ""
        echo "Config created at: $IBC_INI"
        echo "Please edit this file and set your IBKR credentials:"
        echo "  - IbLoginId=YOUR_USERNAME"
        echo "  - IbPassword=YOUR_PASSWORD"
        echo ""
    else
        echo "Please create config from template:"
        echo "  cp config/ibc/config.ini.template config/ibc/config.ini"
        echo "Then edit config.ini with your IBKR credentials."
    fi
    exit 1
fi

# Update trading mode in config
if [ "$TRADING_MODE" = "paper" ]; then
    sed -i '' 's/^TradingMode=.*/TradingMode=paper/' "$IBC_INI" 2>/dev/null || true
elif [ "$TRADING_MODE" = "live" ]; then
    sed -i '' 's/^TradingMode=.*/TradingMode=live/' "$IBC_INI" 2>/dev/null || true
fi

# Check if Gateway is already running
if pgrep -f "ibgateway" > /dev/null 2>&1; then
    echo "WARNING: Gateway appears to be already running."
    echo ""

    # Check API port
    if [ "$TRADING_MODE" = "paper" ]; then
        PORT=4002
    else
        PORT=4001
    fi

    if nc -z 127.0.0.1 $PORT 2>/dev/null; then
        echo "API port $PORT is accepting connections."
        echo "Gateway is ready to use."
        echo ""
        echo "To restart Gateway: ./scripts/start_gateway.sh restart"
        exit 0
    else
        echo "API port $PORT is NOT accepting connections."
        echo "Gateway may need to be restarted."
        echo ""
        read -p "Restart Gateway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi

        # Stop existing Gateway
        echo "Stopping existing Gateway..."
        pkill -f "ibgateway" 2>/dev/null || true
        sleep 3
    fi
fi

# Create log directory
mkdir -p "$LOG_PATH"

# Make IBC scripts executable
chmod +x "${IBC_PATH}"/*.sh 2>/dev/null || true
chmod +x "${IBC_PATH}"/scripts/*.sh 2>/dev/null || true

echo "Starting IB Gateway via IBC..."
echo ""
echo "IMPORTANT: After Gateway starts:"
echo "  1. Complete 2FA authentication on your phone"
echo "  2. Wait for Gateway to show 'IB Gateway - READY'"
echo "  3. Then run: ./START_TRADER.sh"
echo ""

# Launch Gateway - run inline to preserve environment variables
cd "$IBC_PATH"
./gatewaystartmacos.sh -inline

echo ""
echo "Gateway launch initiated."
echo "Check the new Terminal window for login progress."

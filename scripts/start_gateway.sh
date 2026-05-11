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

# Restrict permissions on any files we create (IBC config contains credentials)
umask 077

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
# NEW-IB-L2: TODO — relocate IBC logs to ~/Library/Logs/RoboTrader/, the
# macOS-native location for per-user app logs. Deferred from Round 2 because
# the watchdog (scripts/com.robotrader.watchdog.plist) and any external log
# scrapers point at this in-project path; moving it requires a coordinated
# update of all consumers in the same commit.
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
        # IBC config contains plaintext IBKR credentials - restrict to user only
        chmod 600 "$IBC_INI"
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

# SECURITY: Refuse to start unless Gateway is configured for read-only API access.
# RoboTrader relies on Gateway-side ReadOnlyApi=yes as a primary safety net
# against any code path that might attempt to submit live orders.
# NEW-IB-H1.1: Anchored, case-insensitive regex with explicit end-anchor and
# tolerated whitespace — prevents 'ReadOnlyApi=yesno' bypass and accepts
# 'ReadOnlyApi=Yes' / 'readonlyapi=YES' (all of which IBC honors).
if ! grep -Eqi '^[[:space:]]*readonlyapi[[:space:]]*=[[:space:]]*yes[[:space:]]*$' "$IBC_INI"; then
    echo "FATAL: $IBC_INI does not have ReadOnlyApi=yes. Refusing to start." >&2
    echo "       RoboTrader requires Gateway-side read-only enforcement." >&2
    echo "       To fix: set 'ReadOnlyApi=yes' in $IBC_INI and re-run." >&2
    exit 3
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

# NEW-IB-M4: IBC log files contain account session details and were created
# world/group readable by IBC's default umask. Strip group/other access on
# both the directory and existing files. This is a no-op on a fresh install
# (umask 077 above already covered files we created), but it's needed when
# IBC writes logs after we've already returned, or when an older config is
# being upgraded.
if [ -n "${LOG_PATH:-}" ] && [ -d "$LOG_PATH" ]; then
    chmod -R go-rwx "$LOG_PATH" 2>/dev/null || true
fi

echo ""
echo "Gateway launch initiated."
echo "Check the new Terminal window for login progress."

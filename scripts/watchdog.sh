#!/bin/bash
#
# RoboTrader Watchdog - Auto-restart on stall detection
#
# Usage: ./scripts/watchdog.sh [stale_minutes]
#   stale_minutes: How long without log activity before restart (default: 5)
#
# Run in background: nohup ./scripts/watchdog.sh &
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$PROJECT_DIR/robo_trader.log"
LOG_FILE_1="$PROJECT_DIR/robo_trader.log.1"
WATCHDOG_LOG="$PROJECT_DIR/watchdog.log"
STALE_MINUTES="${1:-5}"  # Default 5 minutes
CHECK_INTERVAL=60        # Check every 60 seconds

cd "$PROJECT_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$WATCHDOG_LOG"
}

is_market_hours() {
    # Check if we're in market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
    local hour=$(TZ="America/New_York" date +%H)
    local minute=$(TZ="America/New_York" date +%M)
    local dow=$(TZ="America/New_York" date +%u)  # 1=Mon, 7=Sun

    # Weekend check
    if [ "$dow" -gt 5 ]; then
        return 1
    fi

    # Convert to minutes since midnight
    local now_minutes=$((10#$hour * 60 + 10#$minute))
    local open_minutes=$((9 * 60 + 30))   # 9:30 AM
    local close_minutes=$((16 * 60))       # 4:00 PM

    if [ "$now_minutes" -ge "$open_minutes" ] && [ "$now_minutes" -lt "$close_minutes" ]; then
        return 0
    fi
    return 1
}

is_extended_hours() {
    # Check if extended hours trading is enabled
    if grep -q "ENABLE_EXTENDED_HOURS=true" "$PROJECT_DIR/.env" 2>/dev/null; then
        local hour=$(TZ="America/New_York" date +%H)
        local dow=$(TZ="America/New_York" date +%u)

        # Weekend - no extended hours
        if [ "$dow" -gt 5 ]; then
            return 1
        fi

        # Pre-market: 4:00 AM - 9:30 AM or After-hours: 4:00 PM - 8:00 PM
        if [ "$hour" -ge 4 ] && [ "$hour" -lt 20 ]; then
            return 0
        fi
    fi
    return 1
}

is_trading_time() {
    is_market_hours || is_extended_hours
}

get_log_age_seconds() {
    # Get the most recent log modification time
    local newest_time=0

    for f in "$LOG_FILE" "$LOG_FILE_1"; do
        if [ -f "$f" ]; then
            local mtime=$(stat -f %m "$f" 2>/dev/null || stat -c %Y "$f" 2>/dev/null)
            if [ "$mtime" -gt "$newest_time" ]; then
                newest_time=$mtime
            fi
        fi
    done

    if [ "$newest_time" -eq 0 ]; then
        echo "999999"  # No log file
        return
    fi

    local now=$(date +%s)
    echo $((now - newest_time))
}

is_runner_alive() {
    pgrep -f "runner_async" > /dev/null 2>&1
}

restart_trader() {
    log "RESTARTING trader due to stall..."

    # Kill existing processes
    pkill -9 -f "runner_async" 2>/dev/null
    pkill -9 -f "websocket_server" 2>/dev/null
    sleep 2

    # Restart
    "$PROJECT_DIR/START_TRADER.sh" >> "$WATCHDOG_LOG" 2>&1

    log "Restart complete"
}

# Main loop
log "=========================================="
log "Watchdog started (stale threshold: ${STALE_MINUTES} minutes)"
log "=========================================="

while true; do
    if is_trading_time; then
        if is_runner_alive; then
            age_seconds=$(get_log_age_seconds)
            age_minutes=$((age_seconds / 60))

            if [ "$age_minutes" -ge "$STALE_MINUTES" ]; then
                log "STALL DETECTED: No log activity for ${age_minutes} minutes (threshold: ${STALE_MINUTES})"
                restart_trader
            fi
        else
            log "Runner not running during trading hours - starting..."
            restart_trader
        fi
    fi

    sleep "$CHECK_INTERVAL"
done

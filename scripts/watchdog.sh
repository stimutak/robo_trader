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
WATCHDOG_LOG_MAX_SIZE=10485760  # 10MB max log size
STALE_MINUTES="${1:-5}"  # Default 5 minutes
CHECK_INTERVAL=60        # Check every 60 seconds
LOCKFILE="$PROJECT_DIR/.watchdog.lock"

cd "$PROJECT_DIR"

# Validate STALE_MINUTES (2-30 range)
if ! [[ "$STALE_MINUTES" =~ ^[0-9]+$ ]] || [ "$STALE_MINUTES" -lt 2 ] || [ "$STALE_MINUTES" -gt 30 ]; then
    echo "Error: stale_minutes must be between 2 and 30"
    exit 1
fi

# Check for existing watchdog (lockfile)
if [ -f "$LOCKFILE" ]; then
    existing_pid=$(cat "$LOCKFILE" 2>/dev/null)
    if kill -0 "$existing_pid" 2>/dev/null; then
        echo "Watchdog already running (PID: $existing_pid)"
        exit 1
    fi
    # Stale lockfile, remove it
    rm -f "$LOCKFILE"
fi
echo $$ > "$LOCKFILE"
trap "rm -f '$LOCKFILE'" EXIT

rotate_log() {
    # Rotate watchdog log if too large
    if [ -f "$WATCHDOG_LOG" ]; then
        local size=$(stat -f %z "$WATCHDOG_LOG" 2>/dev/null || stat -c %s "$WATCHDOG_LOG" 2>/dev/null || echo 0)
        if [ "$size" -gt "$WATCHDOG_LOG_MAX_SIZE" ]; then
            mv "$WATCHDOG_LOG" "$WATCHDOG_LOG.old"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log rotated (was ${size} bytes)" > "$WATCHDOG_LOG"
        fi
    fi
}

log() {
    rotate_log
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$WATCHDOG_LOG"
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
    if grep -q "^ENABLE_EXTENDED_HOURS=true" "$PROJECT_DIR/.env" 2>/dev/null; then
        local hour=$(TZ="America/New_York" date +%H)
        local minute=$(TZ="America/New_York" date +%M)
        local dow=$(TZ="America/New_York" date +%u)

        # Weekend - no extended hours
        if [ "$dow" -gt 5 ]; then
            return 1
        fi

        # Convert to minutes since midnight for precise boundaries
        local now_minutes=$((10#$hour * 60 + 10#$minute))
        local premarket_start=$((4 * 60))      # 4:00 AM
        local premarket_end=$((9 * 60 + 30))   # 9:30 AM
        local afterhours_start=$((16 * 60))    # 4:00 PM
        local afterhours_end=$((20 * 60))      # 8:00 PM

        # Pre-market: 4:00 AM - 9:30 AM
        if [ "$now_minutes" -ge "$premarket_start" ] && [ "$now_minutes" -lt "$premarket_end" ]; then
            return 0
        fi

        # After-hours: 4:00 PM - 8:00 PM
        if [ "$now_minutes" -ge "$afterhours_start" ] && [ "$now_minutes" -lt "$afterhours_end" ]; then
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
            if [ -n "$mtime" ] && [ "$mtime" -gt "$newest_time" ]; then
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

get_last_cycle_age_seconds() {
    # Check for ACTUAL trading cycle completion, not just log spam
    # This catches connect/disconnect loops that don't produce real trading
    # Returns: age in seconds, or -1 if no cycle found (use file mtime as fallback)
    if [ ! -f "$LOG_FILE" ]; then
        echo "-1"  # Sentinel: no log file, caller should use file mtime
        return
    fi

    # Look for "Trading cycle complete" or "Cycle complete" in last 1000 lines
    local last_cycle_line=$(tail -1000 "$LOG_FILE" 2>/dev/null | grep -E "(Trading cycle complete|Cycle complete)" | tail -1)

    if [ -z "$last_cycle_line" ]; then
        echo "-1"  # Sentinel: no cycle found in recent logs, caller should use file mtime
        return
    fi

    # Extract timestamp from JSON log line
    local timestamp=$(echo "$last_cycle_line" | grep -o '"timestamp": "[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$timestamp" ]; then
        log "WARNING: Found cycle line but failed to extract timestamp, using file mtime"
        get_log_age_seconds
        return
    fi

    # Parse ISO timestamp (2026-02-03T16:43:50.332356)
    # Convert to epoch
    local date_part=$(echo "$timestamp" | cut -dT -f1)
    local time_part=$(echo "$timestamp" | cut -dT -f2 | cut -d. -f1)

    local cycle_epoch=$(date -j -f "%Y-%m-%d %H:%M:%S" "$date_part $time_part" +%s 2>/dev/null)

    if [ -z "$cycle_epoch" ]; then
        # macOS date format fallback
        cycle_epoch=$(date -j -f "%Y-%m-%dT%H:%M:%S" "${date_part}T${time_part}" +%s 2>/dev/null)
    fi

    if [ -z "$cycle_epoch" ]; then
        log "WARNING: Failed to parse timestamp '$timestamp', using file mtime"
        get_log_age_seconds
        return
    fi

    local now=$(date +%s)
    echo $((now - cycle_epoch))
}

is_runner_alive() {
    pgrep -f "python.*runner_async" > /dev/null 2>&1
}

restart_trader() {
    log "RESTARTING trader due to stall..."

    # Kill existing processes - use more specific patterns
    pkill -9 -f "python.*runner_async" 2>/dev/null
    pkill -9 -f "python.*websocket_server" 2>/dev/null
    sleep 2

    # Restart
    "$PROJECT_DIR/START_TRADER.sh" >> "$WATCHDOG_LOG" 2>&1

    log "Restart complete"
}

# Main loop
log "=========================================="
log "Watchdog started (PID: $$, stale threshold: ${STALE_MINUTES} minutes)"
log "=========================================="

while true; do
    if is_trading_time; then
        if is_runner_alive; then
            # Check for actual trading cycles, not just log file modification
            cycle_age_seconds=$(get_last_cycle_age_seconds)

            # Also check log modification as a fallback
            log_age_seconds=$(get_log_age_seconds)

            # Handle sentinel value: -1 means no cycle found, fall back to file mtime
            if [ "$cycle_age_seconds" -eq -1 ]; then
                # No cycle completion found - use file mtime instead
                # This handles first startup or if cycle messages aren't in recent logs
                age_seconds="$log_age_seconds"
                age_minutes=$(( ("$age_seconds" + 59) / 60 ))

                if [ "$age_minutes" -ge "$STALE_MINUTES" ]; then
                    log "STALL DETECTED: No log activity for ${age_seconds}s (~${age_minutes} min, threshold: ${STALE_MINUTES})"
                    log "Note: No 'Trading cycle complete' messages found in recent logs"
                    restart_trader
                fi
            else
                # We have cycle completion timestamps - use them for more accurate detection
                cycle_age_minutes=$(( ("$cycle_age_seconds" + 59) / 60 ))

                if [ "$cycle_age_minutes" -ge "$STALE_MINUTES" ]; then
                    # Detect connect/disconnect spam vs actual stall
                    if [ "$log_age_seconds" -lt 60 ] && [ "$cycle_age_seconds" -gt 300 ]; then
                        log "STALL DETECTED: Log updating but NO TRADING CYCLES for ${cycle_age_seconds}s (~${cycle_age_minutes} min)"
                        log "This indicates a connect/disconnect loop - restarting..."
                    else
                        log "STALL DETECTED: No trading cycles for ${cycle_age_seconds}s (~${cycle_age_minutes} min, threshold: ${STALE_MINUTES})"
                    fi
                    restart_trader
                fi
            fi
        else
            log "Runner not running during trading hours - starting..."
            restart_trader
        fi
    fi

    sleep "$CHECK_INTERVAL"
done

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
# launchd's StandardOutPath/StandardErrorPath target (see com.robotrader.watchdog.plist).
# launchd holds a persistent O_APPEND fd on this file, so it must be capped IN PLACE.
# Preserve at most one bounded tail before truncating; copying the whole file can
# consume the last free disk space during the failure mode this guard handles.
WATCHDOG_LAUNCHD_LOG="$PROJECT_DIR/watchdog_launchd.log"
WATCHDOG_LOG_MAX_SIZE=10485760  # 10MB max log size
STALE_MINUTES="${1:-5}"  # Default 5 minutes
CHECK_INTERVAL=60        # Check every 60 seconds
LOCKFILE="$PROJECT_DIR/.watchdog.lock"
RUNNER_EXIT_AUDIT="$PROJECT_DIR/data/runner_exit.json"
RESTART_POLICY="$SCRIPT_DIR/watchdog_restart_policy.py"
RESTART_GUARD="$SCRIPT_DIR/watchdog_restart_guard.sh"
PYTHON3_BIN="$PROJECT_DIR/.venv/bin/python3"
if [ ! -x "$PYTHON3_BIN" ]; then
    PYTHON3_BIN="$(command -v python3 2>/dev/null || true)"
fi
LAST_TERMINAL_SAFETY_REASON=""
# Session-bound restart authorization. This value is updated only after this
# watchdog has directly observed exactly one live runner process.
LAST_OBSERVED_RUNNER_PID=""
if [ -f "$RESTART_GUARD" ]; then
    # shellcheck source=watchdog_restart_guard.sh
    source "$RESTART_GUARD"
else
    # A missing policy guard must never turn an existing exit audit into
    # permission to restart.
    watchdog_restart_allowed_for_policy_rc() { return 1; }
fi

# Layer 6: escalation when restart attempts repeatedly fail (e.g., Gateway 2FA wall).
# Before this layer, the watchdog would silently retry forever — that's how the
# 2026-05-12 outage went unnoticed for 22 hours.
FAILURE_STATE_FILE="$PROJECT_DIR/.watchdog_failures"
ESCALATION_THRESHOLD=3        # Failures before first notification
REMINDER_INTERVAL=12          # Re-notify every N additional failures (with BACKOFF=300s → ~1h)
BACKOFF_INTERVAL=300          # 5 min between attempts after escalation
RESTART_VERIFY_WAIT=30        # Seconds to wait before checking if restart succeeded

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

cap_launchd_log() {
    # Cap the launchd stdout/stderr log IN PLACE. launchd holds a persistent
    # O_APPEND fd on this file, so `mv` does NOT work (writers follow the moved
    # inode — that was the 2026-07-10 incident's bypass). Save only a bounded
    # tail, then truncate in place with `: >`, which O_APPEND writers continue
    # past safely at the new EOF.
    if [ -f "$WATCHDOG_LAUNCHD_LOG" ]; then
        local size=$(stat -f %z "$WATCHDOG_LAUNCHD_LOG" 2>/dev/null || stat -c %s "$WATCHDOG_LAUNCHD_LOG" 2>/dev/null || echo 0)
        if [ "$size" -gt "$WATCHDOG_LOG_MAX_SIZE" ]; then
            local backup_tmp="$WATCHDOG_LAUNCHD_LOG.old.tmp"
            local backup_note="without backup (bounded tail copy failed)"
            if tail -c "$WATCHDOG_LOG_MAX_SIZE" "$WATCHDOG_LAUNCHD_LOG" > "$backup_tmp" 2>/dev/null; then
                mv -f "$backup_tmp" "$WATCHDOG_LAUNCHD_LOG.old"
                backup_note="saved final ${WATCHDOG_LOG_MAX_SIZE} bytes"
            else
                rm -f "$backup_tmp"
            fi
            : > "$WATCHDOG_LAUNCHD_LOG"
            log "launchd log capped in place (was ${size} bytes; ${backup_note})"
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

get_failure_count() {
    if [ -f "$FAILURE_STATE_FILE" ]; then
        cat "$FAILURE_STATE_FILE" 2>/dev/null | head -1 | tr -d '[:space:]' || echo "0"
    else
        echo "0"
    fi
}

set_failure_count() {
    echo "$1" > "$FAILURE_STATE_FILE"
}

reset_failures() {
    if [ -f "$FAILURE_STATE_FILE" ]; then
        log "Recovery: clearing failure counter (was $(get_failure_count))"
        rm -f "$FAILURE_STATE_FILE"
    fi
}

notify_user() {
    local title="$1"
    local msg="$2"
    # macOS native notification — works because launchd loads us under Aqua session
    osascript -e "display notification \"$msg\" with title \"$title\" sound name \"Basso\"" >/dev/null 2>&1 || true
    log "NOTIFICATION SENT [$title]: $msg"
}

restart_trader() {
    # Returns 0 if the restart appears successful (runner alive after wait), 1 otherwise.
    # A deliberate terminal safety exit must survive the supervisor boundary.
    # Manual START_TRADER.sh remains available after the missing protection is
    # deployed, but the watchdog must not loop through Gateway/2FA meanwhile.
    # Always ask the policy when the runner is absent. A missing audit is not
    # evidence that a restart is safe: the terminal-exit audit is deliberately
    # best-effort, so disk exhaustion, permissions, or SIGKILL can leave no
    # file. In that ambiguous state the watchdog stays stopped and requires a
    # manual START_TRADER.sh run. A successful manual startup clears stale
    # audit state after its safety setup completes.
    if ! is_runner_alive; then
        local policy_output
        local policy_rc
        if [ -n "$PYTHON3_BIN" ] && [ -x "$PYTHON3_BIN" ] && [ -f "$RESTART_POLICY" ]; then
            local expected_runner_pid="${LAST_OBSERVED_RUNNER_PID:-unavailable}"
            policy_output="$(
                "$PYTHON3_BIN" "$RESTART_POLICY" \
                    "$RUNNER_EXIT_AUDIT" "$expected_runner_pid" 2>/dev/null
            )"
            policy_rc=$?
        else
            policy_output="restart_policy_unavailable"
            policy_rc=21
        fi

        if ! watchdog_restart_allowed_for_policy_rc "$policy_rc"; then
            local policy_reason
            policy_reason=$(echo "$policy_output" | head -1 | tr -cd '[:alnum:]_.-')
            if [ -z "$policy_reason" ]; then
                policy_reason="restart_policy_invalid"
            fi
            # Log every supervisor restart request that terminates at the
            # policy boundary. This is intentionally distinct from the
            # "RESTARTING" message below because START_TRADER.sh was not
            # invoked. Notifications remain deduplicated to avoid alert spam.
            log "AUTOMATIC RESTART REQUEST DENIED: launcher not invoked (restart_rc=2, policy_rc=${policy_rc}, reason=${policy_reason})"
            if [ "$policy_reason" != "$LAST_TERMINAL_SAFETY_REASON" ]; then
                log "TERMINAL SAFETY BLOCK: watchdog restart suppressed (reason=$policy_reason)"
                notify_user "RoboTrader safety block" \
                    "Automatic restart suppressed: $policy_reason. Keep trader stopped until protection is restored, then run START_TRADER.sh manually."
                LAST_TERMINAL_SAFETY_REASON="$policy_reason"
            fi
            reset_failures
            return 2
        fi
    fi
    LAST_TERMINAL_SAFETY_REASON=""

    log "RESTARTING trader due to stall..."

    # Delegate the complete restart to the authoritative launcher. It validates
    # the paper/read-only contract and IBC configuration before terminating any
    # process, so the watchdog must not pre-kill a healthy runner itself.
    "$PROJECT_DIR/START_TRADER.sh" >> "$WATCHDOG_LOG" 2>&1

    log "Restart script finished; verifying runner came up..."
    sleep "$RESTART_VERIFY_WAIT"

    if is_runner_alive; then
        log "Restart verified: runner_async is alive"
        reset_failures
        return 0
    fi

    # Restart failed — likely a 2FA timeout on Gateway. Track and escalate.
    local prev_failures
    prev_failures=$(get_failure_count)
    local failures=$((prev_failures + 1))
    set_failure_count "$failures"
    log "Restart FAILED: runner_async not alive after ${RESTART_VERIFY_WAIT}s (consecutive failures: $failures)"

    # First-time escalation
    if [ "$failures" -eq "$ESCALATION_THRESHOLD" ]; then
        notify_user "RoboTrader watchdog: trader is DOWN" \
            "Restart failed ${failures}x — likely IBKR 2FA wall. Check IBKR Mobile app or run ./START_TRADER.sh manually."
    elif [ "$failures" -gt "$ESCALATION_THRESHOLD" ]; then
        # Periodic reminder
        local extra=$((failures - ESCALATION_THRESHOLD))
        if [ "$((extra % REMINDER_INTERVAL))" -eq 0 ]; then
            notify_user "RoboTrader still DOWN" \
                "Restart still failing ($failures attempts). Trader has not traded for ~$((failures * BACKOFF_INTERVAL / 60)) min."
        fi
    fi

    return 1
}

# Main loop
log "=========================================="
log "Watchdog started (PID: $$, stale threshold: ${STALE_MINUTES} minutes)"
log "=========================================="

while true; do
    # Keep the launchd stdout/stderr log from filling the disk (2026-07-10 incident).
    cap_launchd_log

    # Layer 6: if we're in an escalated-failure state, slow down to avoid
    # hammering Gateway and burning 2FA attempts.
    failure_count=$(get_failure_count)
    if [ "$failure_count" -ge "$ESCALATION_THRESHOLD" ]; then
        current_interval=$BACKOFF_INTERVAL
    else
        current_interval=$CHECK_INTERVAL
    fi

    if is_trading_time; then
        if is_runner_alive; then
            # Bind any subsequent exit audit to the exact process this
            # watchdog observed alive. Multiple matches are ambiguous and
            # intentionally clear the identity, forcing a manual restart if
            # they all disappear.
            observed_runner_pids=$(pgrep -f "python.*runner_async" 2>/dev/null || true)
            observed_runner_count=$(printf '%s\n' "$observed_runner_pids" | awk 'NF {count++} END {print count+0}')
            if [ "$observed_runner_count" -eq 1 ] && [[ "$observed_runner_pids" =~ ^[0-9]+$ ]]; then
                LAST_OBSERVED_RUNNER_PID="$observed_runner_pids"
            else
                LAST_OBSERVED_RUNNER_PID=""
                log "WARNING: runner identity ambiguous; automatic restart will remain blocked if runner exits"
            fi

            # Runner is alive — if we previously escalated, clear that state and
            # notify the user that recovery happened (so they know the alert is resolved).
            if [ "$failure_count" -ge "$ESCALATION_THRESHOLD" ]; then
                notify_user "RoboTrader recovered" \
                    "Trader is back up after ${failure_count} failed attempts."
            fi
            reset_failures

            # Check for actual trading cycles, not just log file modification
            cycle_age_seconds=$(get_last_cycle_age_seconds)

            # Also check log modification as a fallback
            log_age_seconds=$(get_log_age_seconds)

            # Handle sentinel value: -1 means no cycle found, fall back to file mtime
            if [ "$cycle_age_seconds" -eq -1 ]; then
                # No cycle completion found - use file mtime instead
                # This handles first startup or if cycle messages aren't in recent logs
                age_seconds="$log_age_seconds"
                age_minutes=$(( (age_seconds + 59) / 60 ))

                if [ "$age_minutes" -ge "$STALE_MINUTES" ]; then
                    log "STALL DETECTED: No log activity for ${age_seconds}s (~${age_minutes} min, threshold: ${STALE_MINUTES})"
                    log "Note: No 'Trading cycle complete' messages found in recent logs"
                    restart_trader
                fi
            else
                # We have cycle completion timestamps - use them for more accurate detection
                cycle_age_minutes=$(( (cycle_age_seconds + 59) / 60 ))

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
            restart_trader
        fi
    fi

    sleep "$current_interval"
done

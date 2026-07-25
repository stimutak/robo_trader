#!/bin/bash
#
# RoboTrader Startup Script
#
# This script ensures clean startup by:
# 1. Verifying the identity-bound paper safety journal without mutation
# 2. Gracefully stopping the existing trading runner
# 3. Starting Gateway via IBC if not running
# 4. Cleaning up zombie CLOSE_WAIT connections
# 5. Automatically restarting Gateway if zombies block API
# 6. Running the preflight safety gate
# 7. Replacing monitoring processes and starting the trading system
#
# Usage:
#   ./START_TRADER.sh                    # Start with default symbols
#   ./START_TRADER.sh "AAPL,NVDA"        # Start with custom symbols
#
# Gateway Management:
#   ./START_TRADER.sh                         # Supervised lifecycle entry
#   python3 scripts/gateway_manager.py status # Read-only diagnostics
#

set -e

PORT="${IBKR_PORT:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAX_GATEWAY_RETRIES=3

# Serialize every manual/watchdog invocation and internal Gateway recovery
# before any path can quiesce the runner or manipulate Gateway. On the first
# invocation Python acquires the atomic advisory lock and execs this script,
# leaving FD 200 owned by this Bash process for its complete lifetime.
LOCK_PYTHON=$(command -v python3 2>/dev/null || true)
if [ -z "$LOCK_PYTHON" ]; then
    echo "FATAL: python3 is required for the atomic runtime lifecycle lock." >&2
    exit 75
fi
if [ -z "${ROBOTRADER_RUNTIME_LIFECYCLE_FD:-}" ]; then
    exec "$LOCK_PYTHON" "$SCRIPT_DIR/robo_trader/runtime_lifecycle_lock.py" \
        --exec-launcher "$SCRIPT_DIR/START_TRADER.sh" -- "$@"
    echo "FATAL: could not enter the atomic runtime lifecycle wrapper." >&2
    exit 75
fi
if ! "$LOCK_PYTHON" "$SCRIPT_DIR/robo_trader/runtime_lifecycle_lock.py" \
    --validate-fd "$ROBOTRADER_RUNTIME_LIFECYCLE_FD"; then
    echo "FATAL: inherited runtime lifecycle lock validation failed." >&2
    exit 75
fi

# Resolve lsof to an absolute path and refuse to run without it.
# The watchdog launches us under launchd, whose PATH omits /usr/sbin (where lsof
# lives). A bare `lsof` there is "command not found"; the `2>/dev/null` on every
# port check silently turns that into "port not listening", so START_TRADER kills
# a perfectly healthy Gateway and the watchdog restart-storms. This was the real
# root cause of the 2026-05-13 → 2026-05-29 storms (472 failed restarts) — NOT
# 2FA timing. Fail loud instead of starting blind.
LSOF="$(command -v lsof 2>/dev/null || true)"
[ -z "$LSOF" ] && [ -x /usr/sbin/lsof ] && LSOF=/usr/sbin/lsof
if [ -z "$LSOF" ]; then
    echo "FATAL: lsof not found on PATH ($PATH) and not at /usr/sbin/lsof." >&2
    echo "       Port checks cannot run; refusing to start to avoid a Gateway-kill restart storm." >&2
    exit 3
fi

# Load defaults from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    # Read only the startup values we need instead of sourcing arbitrary shell.
    SYMBOLS=$(grep "^SYMBOLS=" "$SCRIPT_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | tr -d "'" | xargs)
    ENV_IBKR_PORT=$(grep "^IBKR_PORT=" "$SCRIPT_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2- | sed 's/[[:space:]]*#.*$//' | tr -d '"' | tr -d "'" | xargs)
fi

# Fallback default if .env doesn't have SYMBOLS
SYMBOLS="${SYMBOLS:-AAPL,NVDA,TSLA}"
PORT="${PORT:-${ENV_IBKR_PORT:-4002}}"
case "$PORT" in
    4002)
        ;;
    *)
        echo "FATAL: supervised paper remediation requires IB Gateway port 4002; got '$PORT'." >&2
        exit 4
        ;;
esac

# Parse arguments (override .env if provided)
# --force="<reason>" is forwarded to the preflight gate so the documented
# bypass (CLAUDE.md "Bypass mechanics") actually works through this entrypoint;
# previously the gate was always called with no args, so re-running START_TRADER
# could never clear a BLOCK.
PREFLIGHT_FORCE_REASON=""
for arg in "$@"; do
    case $arg in
        --force=*)
            PREFLIGHT_FORCE_REASON="${arg#--force=}"
            ;;
        *)
            SYMBOLS="$arg"
            ;;
    esac
done

echo "=========================================="
echo "RoboTrader Startup Script"
echo "=========================================="
echo ""

validate_ibc_safety_config() {
    local config_path="$1"
    local counts
    local readonly_count
    local readonly_valid
    local trading_mode_count
    local trading_mode_valid

    # Count every active assignment for each safety-critical key, not merely
    # the expected value. This rejects duplicate-good and good-plus-conflicting
    # entries instead of relying on whichever duplicate IBC happens to honor.
    counts=$(awk '
        function trim(value) {
            sub(/^[[:space:]]+/, "", value)
            sub(/[[:space:]]+$/, "", value)
            return value
        }
        BEGIN {
            readonly_count = 0
            readonly_valid = 0
            trading_mode_count = 0
            trading_mode_valid = 0
        }
        {
            line = $0
            sub(/\r$/, "", line)
            if (line ~ /^[[:space:]]*([#;]|$)/) {
                next
            }
            separator = index(line, "=")
            if (separator == 0) {
                next
            }
            key = tolower(trim(substr(line, 1, separator - 1)))
            value = tolower(trim(substr(line, separator + 1)))
            if (key == "readonlyapi") {
                readonly_count++
                if (value == "yes") {
                    readonly_valid++
                }
            } else if (key == "tradingmode") {
                trading_mode_count++
                if (value == "paper") {
                    trading_mode_valid++
                }
            }
        }
        END {
            print readonly_count, readonly_valid, trading_mode_count, trading_mode_valid
        }
    ' "$config_path") || return 1

    read -r readonly_count readonly_valid trading_mode_count trading_mode_valid <<< "$counts"
    if [ "$readonly_count" -ne 1 ] || [ "$readonly_valid" -ne 1 ]; then
        echo "IBC config must contain exactly one active ReadOnlyApi=yes assignment; found ${readonly_count:-0} ReadOnlyApi assignment(s)."
        return 1
    fi
    if [ "$trading_mode_count" -ne 1 ] || [ "$trading_mode_valid" -ne 1 ]; then
        echo "IBC config must contain exactly one active TradingMode=paper assignment; found ${trading_mode_count:-0} TradingMode assignment(s)."
        return 1
    fi
}

# SECURITY: Verify Gateway-side read-only enforcement is configured.
# RoboTrader relies on IBC's ReadOnlyApi=yes as a primary safety net against
# any code path (intentional or accidental) that might attempt to submit
# live orders. If the active config has been modified to permit writes, abort.
IBC_INI="${SCRIPT_DIR}/config/ibc/config.ini"
# Normalize key/value case and whitespace, then require exactly one active
# assignment for each safety setting. A duplicate expected value is ambiguous
# just like an explicitly conflicting value, so both fail closed.
if [ ! -f "$IBC_INI" ]; then
    echo "FATAL: IBC config not found." >&2
    echo "       File: $IBC_INI" >&2
    echo "       Copy config/ibc/config.ini.template and configure the paper account first." >&2
    exit 4
fi
IBC_VALIDATION_ERROR=""
if ! IBC_VALIDATION_ERROR="$(validate_ibc_safety_config "$IBC_INI")"; then
    echo "FATAL: invalid IBC paper/read-only safety configuration." >&2
    echo "       $IBC_VALIDATION_ERROR" >&2
    echo "       File: $IBC_INI" >&2
    echo "       Require one ReadOnlyApi=yes and one TradingMode=paper assignment." >&2
    exit 4
fi

# PR-01 containment contract. Export both the canonical variables and the
# temporary legacy alias so every child observes the same paper/read-only
# identity. load_dotenv() does not override these exported values.
export EXECUTION_MODE="paper"
export TRADING_MODE="paper"
export IBKR_PORT="$PORT"
export IBKR_READONLY="true"


# Stop a matching process group gracefully, then fail closed if even SIGKILL
# cannot remove it. The bounded TERM wait lets runner teardown disconnect its
# persistent IBKR socket before any Gateway/socket/zombie inspection occurs.
stop_processes_gracefully() {
    local label="$1"
    local pattern="$2"
    local wait_seconds="${3:-10}"
    local pids
    local remaining
    local waited=0

    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "   ✓ No $label running"
        return 0
    fi

    echo "   Requesting graceful stop for $label (SIGTERM)..."
    for pid in $pids; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    while [ "$waited" -lt "$wait_seconds" ]; do
        remaining=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -z "$remaining" ]; then
            echo "   ✓ Stopped $label gracefully"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    remaining=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -z "$remaining" ]; then
        echo "   ✓ Stopped $label gracefully"
        return 0
    fi

    if [ -n "$remaining" ]; then
        echo "   ⚠️  $label did not stop within ${wait_seconds}s; forcing SIGKILL"
        for pid in $remaining; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        sleep 1
    fi

    remaining=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$remaining" ]; then
        echo "FATAL: unable to stop $label process(es): $remaining" >&2
        return 1
    fi

    echo "   ✓ Forced stop complete for $label"
}


# Function to start Gateway via IBC
start_gateway() {
    echo "   Starting Gateway via IBC..."

    # Kill any existing Gateway first
    pkill -f "IB Gateway" 2>/dev/null || true
    pkill -f "ibgateway" 2>/dev/null || true
    echo "   Waiting 10s for Gateway to fully shut down..."
    sleep 10

    # Start Gateway
    cd "$SCRIPT_DIR"

    # Set up IBC environment
    export TWS_MAJOR_VRSN="10.37"
    if [ ! -d ~/Applications/"IB Gateway 10.37" ]; then
        GATEWAY_DIR=$(ls -d ~/Applications/"IB Gateway"* 2>/dev/null | sort -V | tail -1)
        if [ -n "$GATEWAY_DIR" ]; then
            export TWS_MAJOR_VRSN=$(basename "$GATEWAY_DIR" | sed 's/IB Gateway //')
        else
            echo "   ERROR: No IB Gateway found in ~/Applications"
            return 1
        fi
    fi

    export IBC_INI="${SCRIPT_DIR}/config/ibc/config.ini"
    export TRADING_MODE="paper"
    export TWOFA_TIMEOUT_ACTION="restart"
    export IBC_PATH="${SCRIPT_DIR}/IBCMacos-3"
    export TWS_PATH=~/Applications
    export TWS_SETTINGS_PATH=
    export LOG_PATH="${SCRIPT_DIR}/config/ibc/logs"

    # Check config exists
    if [ ! -f "$IBC_INI" ]; then
        echo "   ERROR: IBC config not found at $IBC_INI"
        echo "   Run: cp config/ibc/config.ini.template config/ibc/config.ini"
        echo "   Then edit with your IBKR credentials."
        return 1
    fi

    # Create log directory
    mkdir -p "$LOG_PATH"

    # Make scripts executable
    chmod +x "${IBC_PATH}"/*.sh 2>/dev/null || true
    chmod +x "${IBC_PATH}"/scripts/*.sh 2>/dev/null || true

    echo "   Using Gateway version: $TWS_MAJOR_VRSN"
    echo ""
    echo "   ========================================"
    echo "   STARTING GATEWAY - 2FA REQUIRED"
    echo "   ========================================"
    echo "   Check your IBKR Mobile app for 2FA prompt"
    echo ""

    # Launch Gateway inline (blocks until Gateway exits or we Ctrl+C)
    cd "$IBC_PATH"
    # Long-lived descendants must not inherit the launcher's lifecycle lock.
    ./gatewaystartmacos.sh -inline 200>&- &
    IBC_PID=$!

    # Wait for Gateway to start and API port to open
    # CRITICAL: Use lsof, NOT nc -z (nc creates zombie connections that block API handshakes!)
    # 240s window: IBC + Gateway cold-start + 2FA approval routinely needs >120s.
    # The previous 120s window was tight enough to lose 89 races over 13 days.
    echo "   Waiting for Gateway to start..."
    for i in $(seq 1 240); do
        if "$LSOF" -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | grep -q LISTEN; then
            echo "   ✓ Gateway API port $PORT is now open!"
            # Wait for Gateway to fully initialize after port opens
            # Gateway needs time to complete login/2FA before API is responsive
            echo "   Waiting 30s for Gateway to complete login/2FA..."
            sleep 30
            return 0
        fi

        # Check if IBC process died
        if ! kill -0 $IBC_PID 2>/dev/null; then
            # IBC finished launching, check if Gateway is running
            if pgrep -f "IB Gateway" > /dev/null 2>&1; then
                # Gateway running but port not open yet, keep waiting
                :
            else
                echo "   Gateway process not detected, IBC may have failed"
            fi
        fi

        sleep 1
        if [ $((i % 15)) -eq 0 ]; then
            echo "   Still waiting... ($i seconds)"
        fi
    done

    echo "   TIMEOUT: Gateway did not start within 240 seconds"
    return 1
}

# Note: API handshake test added back (scripts/test_gateway_api.py) to verify Gateway
# is actually responding, not just that the port is open. This test properly disconnects
# to avoid creating zombie connections.

# Function to check if port is listening (uses lsof to avoid creating zombie connections)
# CRITICAL: Do NOT use nc -z for port checking - it creates zombie connections that block API handshakes!
is_port_listening() {
    "$LSOF" -nP -iTCP:$PORT -sTCP:LISTEN 2>/dev/null | grep -q LISTEN
}

# Function to check for zombie connections
check_zombies() {
    # Count actual zombie lines - use wc -l and trim whitespace
    local count
    count=$("$LSOF" -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep "CLOSE_WAIT" | wc -l | tr -d ' ')
    # Return 0 if empty
    echo "${count:-0}"
}

# Function to rotate a child's stdout log before (re)launching it.
# In a restart storm (472 restarts on 2026-05-29) each start would otherwise
# truncate the previous attempt's crash output, leaving only the last try.
# Keep ONE prior generation as <name>.log.1 (gitignored via *.log.*), then let
# the caller's `> file` redirect truncate a fresh file. Guarded so a non-zero
# test/mv can't trip `set -e`. Uses only /bin,/usr/bin coreutils (test, mv).
rotate_log() {
    local f="$1"
    if [ -s "$f" ]; then
        mv -f "$f" "$f.1" || true
    fi
}

# Step 0.5: Verify the safety journal before changing any running process or
# Gateway state. Normal startup may replay this journal but must never create,
# repair, rebind, or bypass it.
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    SAFETY_VERIFY_PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
    SAFETY_VERIFY_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    SAFETY_VERIFY_PYTHON="$LOCK_PYTHON"
fi
echo "0.5. Verifying identity-bound paper safety journal..."
if ! "$SAFETY_VERIFY_PYTHON" \
    "$SCRIPT_DIR/scripts/manage_paper_safety_journal.py" verify; then
    echo "FATAL: paper safety journal verification blocked startup." >&2
    echo "Fail-closed: stopping the existing trading runner only..." >&2
    if ! stop_processes_gracefully \
        "runner_async" "robo_trader[./]runner_async"; then
        echo "FATAL: existing trading runner could not be quiesced." >&2
    fi
    # Write this only after the stop attempt: the old runner's SIGTERM/finally
    # handlers write their own exit audit and must not overwrite this sticky
    # terminal startup block.
    if ! "$SAFETY_VERIFY_PYTHON" \
        "$SCRIPT_DIR/scripts/write_paper_safety_terminal_audit.py"; then
        echo "FATAL: could not persist the terminal safety audit." >&2
    fi
    echo "Gateway, dashboard, and WebSocket processes were left untouched." >&2
    exit 7
fi
echo "   ✓ Paper safety journal replay passed"
echo ""

# Step 1: Quiesce the only process allowed to hold the trading connection.
# This MUST precede Gateway status, LISTEN, CLOSE_WAIT, and preflight checks:
# the old persistent runner can otherwise make healthy sockets look stale or
# continue trading after a safety-gate BLOCK. Keep dashboard/WebSocket alive so
# operators retain monitoring while Gateway recovery and preflight run.
echo "1. Stopping existing trading runner..."
stop_processes_gracefully "runner_async" "robo_trader[./]runner_async"
echo ""

# Step 2: Check/Start Gateway with retry logic
echo "2. Checking Gateway status..."
GATEWAY_RETRY=0
API_CONNECTED=false

while [ "$API_CONNECTED" = false ] && [ $GATEWAY_RETRY -lt $MAX_GATEWAY_RETRIES ]; do
    GATEWAY_RETRY=$((GATEWAY_RETRY + 1))

    if [ $GATEWAY_RETRY -gt 1 ]; then
        echo ""
        echo "   =========================================="
        echo "   GATEWAY RETRY $GATEWAY_RETRY of $MAX_GATEWAY_RETRIES"
        echo "   =========================================="
        echo ""
    fi

    # Check if Gateway is running
    if ! pgrep -f "IB Gateway" > /dev/null 2>&1 && ! pgrep -f "ibcalpha.ibc" > /dev/null 2>&1; then
        echo "   Gateway is NOT running - starting via IBC..."
        if ! start_gateway; then
            echo "   Failed to start Gateway"
            continue
        fi
    else
        echo "   ✓ Gateway process detected"
    fi

    # Check if port is listening (using lsof to avoid zombie connections)
    if ! is_port_listening; then
        echo "   ⚠️  API port $PORT is NOT listening"
        echo "   Gateway process is alive but API port not bound yet."
        echo "   Normal during startup/2FA — waiting up to 180s before assuming Gateway is stuck."
        echo "   (Killing Gateway prematurely triggers a fresh 2FA prompt and causes restart storms.)"

        # Wait for Gateway to bind the API port. Gateway+IBC+2FA routinely needs 60-180s,
        # especially after a SIGTERM cycle. The previous 30s window was the root cause of
        # the 2026-05-13 → 05-26 restart storm (89 consecutive failed restarts).
        for i in $(seq 1 180); do
            if is_port_listening; then
                echo "   ✓ API port $PORT is now listening (after ${i}s)"
                break
            fi
            # Early-exit if Gateway actually died during the wait — no point waiting longer.
            if ! pgrep -f "IB Gateway" > /dev/null 2>&1 && ! pgrep -f "ibcalpha.ibc" > /dev/null 2>&1; then
                echo "   Gateway process died during wait — restart needed"
                break
            fi
            sleep 1
            if [ $((i % 30)) -eq 0 ]; then
                echo "   Still waiting for port bind... (${i}s) — check IBKR Mobile for 2FA prompt"
            fi
        done

        if ! is_port_listening; then
            echo "   Port still not listening after 180s - restarting Gateway..."
            start_gateway
            continue
        fi
    else
        echo "   ✓ API port $PORT is listening"
    fi

    # Check for zombies (now that we use lsof for port checking, we should have zero startup zombies)
    ZOMBIES=$(check_zombies)
    if [ "$ZOMBIES" -gt 0 ]; then
        echo "   ⚠️  Found $ZOMBIES zombie connection(s)!"
        echo "   Zombies block API handshakes - restarting Gateway..."

        # Kill Python zombies first
        "$LSOF" -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u | while read pid; do
            kill -9 $pid 2>/dev/null && echo "   Killed Python zombie PID $pid" || true
        done
        sleep 1

        # Check if zombies remain (Gateway-owned)
        ZOMBIES=$(check_zombies)
        if [ "$ZOMBIES" -gt 0 ]; then
            echo "   Gateway zombies remain - must restart Gateway"
            start_gateway
            continue
        fi
    fi

    # Skip API handshake test - it creates zombie connections that block subsequent connections!
    # The lsof port check above is sufficient to verify Gateway is listening.
    # The trading system will handle connection retries if needed.
    echo "   ✓ Gateway port is listening - ready for connections"
    API_CONNECTED=true
done

if [ "$API_CONNECTED" = false ]; then
    echo ""
    echo "=========================================="
    echo "❌ FAILED TO CONNECT TO GATEWAY API"
    echo "=========================================="
    echo ""
    echo "After $MAX_GATEWAY_RETRIES attempts, could not establish API connection."
    echo ""
    echo "Manual troubleshooting:"
    echo "  1. Check Gateway is fully started (shows 'IB Gateway - READY')"
    echo "  2. Verify 2FA was completed on your phone"
    echo "  3. Re-run: ./START_TRADER.sh"
    echo ""
    exit 1
fi

echo ""

# Step 3: Clean up any remaining zombie connections
echo "3. Final zombie cleanup..."
ZOMBIES=$(check_zombies)
if [ "$ZOMBIES" -gt 0 ]; then
    echo "   Killing $ZOMBIES zombie connection(s)..."
    "$LSOF" -nP -iTCP:$PORT -sTCP:CLOSE_WAIT 2>/dev/null | grep -i python | awk '{print $2}' | sort -u | while read pid; do
        kill -9 $pid 2>/dev/null || true
    done
    sleep 1
fi
echo "   ✓ Zombie cleanup complete"
echo ""

# Step 4: Set up Python environment
echo "4. Setting up Python environment..."
cd "$SCRIPT_DIR"

# Determine Python path - prefer .venv, fall back to system python3
if [ -x ".venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
    echo "   ✓ Using virtualenv Python: $PYTHON"
elif [ -x "venv/bin/python" ]; then
    PYTHON="$SCRIPT_DIR/venv/bin/python"
    echo "   ✓ Using virtualenv Python: $PYTHON"
else
    # Check if we need to create virtualenv
    if [ -f "requirements.txt" ]; then
        echo "   Creating virtual environment..."
        python3 -m venv .venv
        PYTHON="$SCRIPT_DIR/.venv/bin/python"
        echo "   Installing dependencies..."
        $PYTHON -m pip install -r requirements.txt -q
        echo "   ✓ Virtual environment created and dependencies installed"
    else
        PYTHON="python3"
        echo "   ⚠️  No virtualenv found - using system Python"
    fi
fi

# Verify Python works and has required packages
if ! $PYTHON -c "import pandas" 2>/dev/null; then
    echo "   ⚠️  Missing dependencies - installing from requirements.txt..."
    $PYTHON -m pip install -r requirements.txt -q
    echo "   ✓ Dependencies installed"
fi
echo ""

# Step 4.5: Preflight safety gate
# Runs scripts/preflight_check.py which validates system state before launch.
# Exit codes:
#   0 -> all checks passed, proceed
#   2 -> operator used --force to bypass a BLOCK (audited); proceed with warning banner
#   1, 3, * -> BLOCKED or preflight itself failed; abort startup
echo "4.5. Running preflight safety gate..."
# preflight intentionally exits non-zero (1=BLOCK, 2=--force bypass-and-proceed).
# Under `set -e` a bare invocation aborts the script before the case below can
# interpret the code — capture it via `|| PREFLIGHT_RC=$?`, which is exempt from
# set -e. (Without this, the 2=proceed branch was dead code and --force could
# never actually start the trader.)
PREFLIGHT_RC=0
if [ -n "$PREFLIGHT_FORCE_REASON" ]; then
    echo "   (operator --force supplied — bypass will be audited to data/preflight_bypass.log)"
    $PYTHON scripts/preflight_check.py --force "$PREFLIGHT_FORCE_REASON" || PREFLIGHT_RC=$?
else
    $PYTHON scripts/preflight_check.py || PREFLIGHT_RC=$?
fi
case "$PREFLIGHT_RC" in
    0)
        echo "   ✓ Preflight gate passed"
        ;;
    2)
        echo ""
        echo "=========================================="
        echo "⚠️  PREFLIGHT BYPASSED VIA --force"
        echo "=========================================="
        echo "Operator override was used. See data/preflight_bypass.log for audit trail."
        echo "Proceeding with launch."
        echo ""
        ;;
    *)
        # 1 (BLOCKED), 3 (preflight failed), or anything else.
        echo ""
        echo "=========================================="
        echo "❌ PREFLIGHT SAFETY GATE BLOCKED STARTUP"
        echo "=========================================="
        echo ""
        echo "Preflight reported blocking issues above (exit code $PREFLIGHT_RC)."
        echo "Resolve each one and re-run ./START_TRADER.sh, or use the audited"
        echo "single-invocation bypass: ./START_TRADER.sh --force=\"<reason>\""
        echo "(the bypass is per invocation and does not persist)"
        echo ""
        exit 1
        ;;
esac
echo ""

# Replace monitoring only after a preflight pass or audited bypass. A BLOCK
# exits above with the runner stopped but the existing dashboard/WebSocket
# still available for diagnosis.
echo "4.6. Replacing monitoring processes..."
stop_processes_gracefully "dashboard" '(^|[/[:space:]])app[.]py([[:space:]]|$)'
stop_processes_gracefully "websocket_server" "robo_trader[./]websocket_server"
echo ""

# Step 5: Start dashboard (includes WebSocket server)
echo "5. Starting dashboard with WebSocket server..."
export DASH_PORT=5555
# Redirect stdout/stderr: backgrounded children otherwise inherit the
# caller's fds — under the watchdog/launchd that was watchdog.log, which
# bypassed all rotation and filled the disk on 2026-07-10. Rotate one
# generation, then truncate on each start so these can't grow unbounded
# across restarts while still preserving the prior attempt's crash output.
rotate_log "$SCRIPT_DIR/dashboard_stdout.log"
$PYTHON app.py > "$SCRIPT_DIR/dashboard_stdout.log" 2>&1 200>&- &
DASH_PID=$!
sleep 2

if ps -p $DASH_PID > /dev/null; then
    echo "   ✓ Dashboard started (PID: $DASH_PID)"
    echo "   ✓ WebSocket server running on ws://localhost:8765"
else
    echo "   ⚠️  Dashboard may have failed to start"
    echo "      Check logs: tail -50 $SCRIPT_DIR/dashboard_stdout.log"
fi
echo ""

# Step 6: Start trading system
echo "6. Starting trading system..."
echo "   Symbols: $SYMBOLS"
echo "   Log: robo_trader.log"
echo ""

export LOG_FILE="$SCRIPT_DIR/robo_trader.log"

# --force-connect removed 2026-07-10: it is a testing flag ("Force IBKR
# connection even when market is closed") that, combined with a health-gate
# skip, drove the zero-backoff spin loop in the disk-fill incident. Outside
# market hours the runner now sleeps until open (extended hours still
# covered by ENABLE_EXTENDED_HOURS via is_trading_allowed).
rotate_log "$SCRIPT_DIR/runner_stdout.log"
$PYTHON -m robo_trader.runner_async --symbols "$SYMBOLS" > "$SCRIPT_DIR/runner_stdout.log" 2>&1 200>&- &
TRADER_PID=$!

echo "   ✓ Trading system started (PID: $TRADER_PID)"
echo ""

# Step 7: Monitor startup
echo "7. Monitoring startup (10 seconds)..."
sleep 10

if ps -p $TRADER_PID > /dev/null; then
    echo "   ✓ Trading system is running"
    echo ""
    echo "=========================================="
    echo "✅ STARTUP SUCCESSFUL"
    echo "=========================================="
    echo ""
    echo "Trading system is running with PID: $TRADER_PID"
    echo "Dashboard PID: $DASH_PID (includes WebSocket server)"
    echo ""
    echo "Monitor logs: tail -f robo_trader.log"
    echo "  Pre-logger crash output (import/.env errors): runner_stdout.log, dashboard_stdout.log"
    echo "View dashboard: http://localhost:5555"
    echo "WebSocket: ws://localhost:8765"
    echo ""
    echo "To stop gracefully:"
    echo "  pkill -TERM -f 'robo_trader[./]runner_async'"
    echo "  pkill -TERM -f '(^|[/[:space:]])app[.]py([[:space:]]|$)'"
    echo ""

    # Step 8: Verify the launchd watchdog is loaded.
    # If it isn't, a transient crash (e.g. lsof timeout in the gateway pre-flight)
    # will leave the trader dead until a human notices. This is exactly what
    # happened on 2026-05-11.
    if ! launchctl list 2>/dev/null | grep -q "robotrader"; then
        echo "=========================================="
        echo "WARNING: launchd watchdog is NOT LOADED"
        echo "=========================================="
        echo ""
        echo "The watchdog auto-restarts the trader if it stalls during"
        echo "market hours. Without it, a crash leaves the system dead"
        echo "until someone notices. This caused an overnight outage on"
        echo "2026-05-11."
        echo ""
        echo "Run ONCE per machine to fix:"
        echo "  ./scripts/install_watchdog.sh"
        echo ""
        echo "See DEV_SETUP.md Section 2.6.1 and CLAUDE.md for details."
        echo "=========================================="
        echo ""
    fi
else
    echo "   ❌ Trading system stopped unexpectedly"
    echo ""
    echo "Check logs: tail -50 $SCRIPT_DIR/runner_stdout.log"
    echo "  (a pre-logger crash — import error, bad .env — lands ONLY there, not robo_trader.log)"
    echo "  Then also: tail -50 robo_trader.log"
    echo ""
    exit 1
fi

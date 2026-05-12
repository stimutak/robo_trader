#!/usr/bin/env bash
#
# install_watchdog.sh — One-command launchd watchdog installer for RoboTrader.
#
# Idempotent: safe to re-run. Loads (or reloads) the launchd agent that
# auto-restarts the trader if the log goes stale during market hours
# (or extended hours, if ENABLE_EXTENDED_HOURS=true in .env).
#
# Usage:
#   ./scripts/install_watchdog.sh
#
# Environment overrides (testing):
#   LAUNCH_AGENTS_DIR   Override target dir (default: ~/Library/LaunchAgents)
#   SKIP_LAUNCHCTL=1    Skip launchctl load/list assertions (for CI/tests).
#
# Exits non-zero on any error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_SRC="$SCRIPT_DIR/com.robotrader.watchdog.plist"
WATCHDOG_SH="$SCRIPT_DIR/watchdog.sh"
LABEL="com.robotrader.watchdog"

LAUNCH_AGENTS_DIR="${LAUNCH_AGENTS_DIR:-$HOME/Library/LaunchAgents}"
PLIST_DEST="$LAUNCH_AGENTS_DIR/${LABEL}.plist"

die() {
    echo "" >&2
    echo "ERROR: $*" >&2
    echo "" >&2
    echo "Next steps:" >&2
    echo "  1. Check that scripts/com.robotrader.watchdog.plist exists in the repo." >&2
    echo "  2. Run: plutil -lint $PLIST_SRC" >&2
    echo "  3. Check that scripts/watchdog.sh is executable: chmod +x $WATCHDOG_SH" >&2
    echo "  4. If launchctl errors, try: launchctl bootout gui/\$(id -u) $PLIST_DEST" >&2
    exit 1
}

echo "==> RoboTrader watchdog installer"
echo "    Project:       $PROJECT_DIR"
echo "    Source plist:  $PLIST_SRC"
echo "    Target plist:  $PLIST_DEST"
echo "    Watchdog log:  $PROJECT_DIR/watchdog.log"
echo ""

# 1. Sanity checks ---------------------------------------------------------
[[ -f "$PLIST_SRC" ]]    || die "plist source not found at $PLIST_SRC"
[[ -f "$WATCHDOG_SH" ]]  || die "watchdog.sh not found at $WATCHDOG_SH"
[[ -x "$WATCHDOG_SH" ]]  || {
    echo "    watchdog.sh not executable, fixing..."
    chmod +x "$WATCHDOG_SH" || die "could not chmod +x $WATCHDOG_SH"
}

# 2. Validate plist --------------------------------------------------------
echo "==> Validating plist with plutil -lint"
plutil -lint "$PLIST_SRC" || die "plist failed plutil -lint"

# 3. Validate watchdog.sh shell syntax ------------------------------------
echo "==> Validating watchdog.sh shell syntax"
bash -n "$WATCHDOG_SH" || die "watchdog.sh failed bash -n syntax check"

# 4. Ensure LaunchAgents dir exists ---------------------------------------
mkdir -p "$LAUNCH_AGENTS_DIR" || die "could not create $LAUNCH_AGENTS_DIR"

# 5. Copy plist (idempotent) ----------------------------------------------
echo "==> Copying plist to $PLIST_DEST"
cp "$PLIST_SRC" "$PLIST_DEST" || die "failed to copy plist to $PLIST_DEST"

if [[ "${SKIP_LAUNCHCTL:-0}" = "1" ]]; then
    echo "==> SKIP_LAUNCHCTL=1 set; skipping launchctl load/list."
    echo ""
    echo "Done (skipped launchctl)."
    exit 0
fi

# 6. Unload any prior version ---------------------------------------------
echo "==> Unloading any prior watchdog agent (ignored if not present)"
launchctl unload "$PLIST_DEST" 2>/dev/null || true

# Best-effort: drop stale lockfile so the new run does not exit early.
rm -f "$PROJECT_DIR/.watchdog.lock" 2>/dev/null || true

# 7. Load the new one ------------------------------------------------------
echo "==> Loading $PLIST_DEST"
launchctl load "$PLIST_DEST" || die "launchctl load failed for $PLIST_DEST"

# 8. Give launchd a beat, then assert the agent registered ----------------
sleep 2
echo "==> Verifying registration with launchctl list"
if ! launchctl list | grep -q "$LABEL"; then
    die "launchctl list does not show $LABEL after load"
fi

echo ""
echo "SUCCESS: $LABEL is registered with launchd."
launchctl list | grep "$LABEL" || true
echo ""
echo "What this does:"
echo "  - Checks robo_trader.log modification time every 60 seconds."
echo "  - If no log activity for 5+ minutes during market hours"
echo "    (or extended hours when ENABLE_EXTENDED_HOURS=true), kills"
echo "    stale processes and runs ./START_TRADER.sh automatically."
echo "  - Logs all restart actions to watchdog.log in the project root."
echo "  - Survives reboot (loaded automatically on next GUI login)."
echo ""
echo "Inspect:    tail -f $PROJECT_DIR/watchdog.log"
echo "Stop:       launchctl unload $PLIST_DEST"
echo "Re-install: $0"

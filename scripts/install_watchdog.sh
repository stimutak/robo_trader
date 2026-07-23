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
#   ROBOTRADER_USER     Override launchd UserName (default: current user).
#   PLUTIL_BIN          Override plutil executable (testing only).
#
# Exits non-zero on any error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PLIST_SRC="$SCRIPT_DIR/com.robotrader.watchdog.plist"
WATCHDOG_SH="$SCRIPT_DIR/watchdog.sh"
LABEL="com.robotrader.watchdog"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLUTIL_BIN="${PLUTIL_BIN:-plutil}"
ROBOTRADER_USER="${ROBOTRADER_USER:-$(id -un)}"

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
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python not found: $PYTHON_BIN"
[[ -x "$WATCHDOG_SH" ]]  || {
    echo "    watchdog.sh not executable, fixing..."
    chmod +x "$WATCHDOG_SH" || die "could not chmod +x $WATCHDOG_SH"
}

# 2. Validate watchdog.sh shell syntax ------------------------------------
echo "==> Validating watchdog.sh shell syntax"
bash -n "$WATCHDOG_SH" || die "watchdog.sh failed bash -n syntax check"

# 3. Ensure LaunchAgents dir exists ---------------------------------------
mkdir -p "$LAUNCH_AGENTS_DIR" || die "could not create $LAUNCH_AGENTS_DIR"

# 4. Render the machine-specific plist (idempotent) ------------------------
# The tracked template contains this checkout's defaults so it remains easy to
# inspect, but the installed plist must always match the actual checkout and
# current GUI user. In particular, StandardOutPath/StandardErrorPath must point
# beside watchdog.sh's WATCHDOG_LAUNCHD_LOG or launchd output escapes its cap.
echo "==> Rendering plist to $PLIST_DEST"
"$PYTHON_BIN" - "$PLIST_SRC" "$PLIST_DEST" "$PROJECT_DIR" "$ROBOTRADER_USER" <<'PY'
import plistlib
import sys
from pathlib import Path

source, destination, project_dir, user_name = sys.argv[1:]
project = Path(project_dir).resolve()

with open(source, "rb") as handle:
    plist = plistlib.load(handle)

plist["UserName"] = user_name
plist["ProgramArguments"][0] = str(project / "scripts" / "watchdog.sh")
plist["WorkingDirectory"] = str(project)
launchd_log = str(project / "watchdog_launchd.log")
plist["StandardOutPath"] = launchd_log
plist["StandardErrorPath"] = launchd_log

with open(destination, "wb") as handle:
    plistlib.dump(plist, handle, sort_keys=False)
PY
chmod 600 "$PLIST_DEST" || die "could not set safe permissions on $PLIST_DEST"

# 5. Validate the rendered plist ------------------------------------------
echo "==> Validating rendered plist with plutil -lint"
"$PLUTIL_BIN" -lint "$PLIST_DEST" || die "rendered plist failed plutil -lint"

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
# launchd registration can lag the `launchctl load` return, so poll instead of
# checking once (a single check raced registration on 2026-07-10 and printed a
# false ERROR while the job was in fact present). `grep` returning non-zero must
# not trip `set -e`, hence the `if` guard around it.
echo "==> Verifying registration with launchctl list"
registered=0
for _ in $(seq 1 10); do
    if launchctl list | grep -q "$LABEL"; then
        registered=1
        break
    fi
    sleep 1
done
if [[ "$registered" -ne 1 ]]; then
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

# Handoff: Watchdog Auto-Restarter

**Date:** 2026-02-03
**Session Focus:** Implement automatic stall detection and restart for the trading system

---

## Summary

Created a watchdog service that monitors the trading system and automatically restarts it if stalled during market hours.

## Problem Solved

- Trader was stalling with no log activity (last logs at 12:18, discovered at 13:12)
- No automatic recovery mechanism existed
- Manual intervention required to restart

## Files Created

### 1. `scripts/watchdog.sh`
Bash script that:
- Monitors log file modification time
- Detects stalls (no activity for 5+ minutes)
- Auto-restarts via `./START_TRADER.sh`
- Only active during market hours (respects `ENABLE_EXTENDED_HOURS`)
- Logs to `watchdog.log`

### 2. `scripts/com.robotrader.watchdog.plist`
macOS launchd service configuration:
- Runs at login
- Keeps alive (auto-restarts if watchdog crashes)
- Installed to `~/Library/LaunchAgents/`

## Files Modified

### `START_TRADER.sh`
- Now reads `SYMBOLS=` from `.env` by default
- Falls back to `AAPL,NVDA,TSLA` if not set
- Command-line argument still overrides

**Before:**
```bash
SYMBOLS="AAPL,NVDA,TSLA"
```

**After:**
```bash
# Load defaults from .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
    SYMBOLS=$(grep "^SYMBOLS=" "$SCRIPT_DIR/.env" 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")
fi
SYMBOLS="${SYMBOLS:-AAPL,NVDA,TSLA}"
```

## Installation

Watchdog installed as macOS service:
```bash
cp scripts/com.robotrader.watchdog.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist
```

## Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Stale threshold | 5 minutes | `watchdog.sh` arg or plist |
| Check interval | 60 seconds | `watchdog.sh` |
| Symbols | From `.env` | `SYMBOLS=...` |
| Extended hours | From `.env` | `ENABLE_EXTENDED_HOURS=true` |

## Management Commands

```bash
# Check status
launchctl list | grep robotrader

# View watchdog log
tail -f watchdog.log

# Stop watchdog
launchctl unload ~/Library/LaunchAgents/com.robotrader.watchdog.plist

# Start watchdog
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist

# Change threshold (edit plist, then reload)
```

## How It Works

1. Every 60 seconds, watchdog checks:
   - Is it market hours (or extended hours if enabled)?
   - Is runner_async process alive?
   - Has log file been modified in last 5 minutes?

2. If stall detected:
   - Kills existing runner_async and websocket_server
   - Calls `./START_TRADER.sh` (reads symbols from .env)
   - Logs restart to `watchdog.log`

3. Outside market hours:
   - No monitoring (conserves resources)

## Other Findings

- User has 65 open positions but `RISK_MAX_OPEN_POSITIONS=20`
- This blocks all new BUY signals (duplicate protection working correctly)
- User may want to increase position limit in `.env`

## Next Steps

- [ ] Consider increasing `RISK_MAX_OPEN_POSITIONS` if more buys desired
- [ ] Monitor `watchdog.log` for restart frequency
- [ ] Optionally add Slack/email notification on restart

---

**Watchdog Status:** ✅ Installed and running (PID in `launchctl list`)

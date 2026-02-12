---
description: Check trading system health: Gateway, zombies, risk params
---

# Verify Trading System

Comprehensive verification of the trading system health and readiness.

## Step 1: Process Status Check

```bash
ps aux | grep -E "(runner_async|app.py|websocket_server)" | grep -v grep
```

**Required processes:**
- `runner_async` - Main trading engine (CRITICAL)
- `websocket_server` - Real-time updates
- `app.py` - Dashboard

If `runner_async` is NOT running, start it:
```bash
./START_TRADER.sh
```

## Step 2: Gateway Status

```bash
python3 scripts/gateway_manager.py status
```

Expected output shows Gateway running on correct port.

## Step 3: Zombie Connection Check

```bash
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

- If output is empty: GOOD - no zombies
- If output shows connections: BAD - zombies blocking API

**Fix zombies:**
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py"
python3 scripts/gateway_manager.py restart
```

## Step 4: Risk Parameters Validation

Check `.env` for required safety configs:
- `MAX_OPEN_POSITIONS` - Should be set (default: 10)
- `STOP_LOSS_PERCENT` - Should be set (default: 2.0)
- `MAX_POSITION_SIZE` - Should be set
- `EXECUTION_MODE` - Should be `paper` for testing
- `USE_TRAILING_STOP` - Should be `true` (locks in profits)
- `TRAILING_STOP_PERCENT` - Should be set (default: 5.0)
- `ENABLE_EXTENDED_HOURS` - Set to `true` for pre/after market

## Step 5: Safety Feature Tests

```bash
.venv/bin/python3 -m pytest tests/test_safety_features.py -v
```

All tests should pass.

## Step 6: Market Hours Logic

```bash
python3 -c "from robo_trader.market_hours import is_market_open, get_market_session; print(f'Market open: {is_market_open()}'); print(f'Current session: {get_market_session()}')"
```

Verify market state matches actual NYSE hours:
- Regular: 9:30 AM - 4:00 PM ET
- Pre-market: 4:00 AM - 9:30 AM ET
- After-hours: 4:00 PM - 8:00 PM ET

## Step 7: Recent Logs Check

```bash
tail -50 robo_trader.log | grep -E "(ERROR|CRITICAL|Exception)"
```

- If no output: GOOD - no recent errors
- If errors found: Investigate each one

## Verification Summary

Report format:
```markdown
## Trading System Verification

| Check | Status | Notes |
|-------|--------|-------|
| Processes | ✅/❌ | runner_async, websocket_server, app.py |
| Gateway | ✅/❌ | Port 4002 |
| Zombies | ✅/❌ | Count: X |
| Risk Params | ✅/❌ | |
| Safety Tests | ✅/❌ | X/Y passed |
| Market Hours | ✅/❌ | State: X |
| Recent Errors | ✅/❌ | Count: X |

**Overall Status:** READY / NOT READY
```

**CRITICAL:** If `runner_async` is not running, status is NOT READY regardless of other checks.

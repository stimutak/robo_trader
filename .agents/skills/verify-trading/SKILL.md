---
name: verify-trading-skill
description: "Comprehensive trading system health check. Use when users ask to verify the trading system, check Gateway status, look for zombies, or validate risk parameters."
allowed-tools: "Read,Bash,Grep,Glob"
context: fork
---

# Verify Trading System (Skill)

Comprehensive verification of the trading system health and readiness.

## Quick Verification

Run the verification script:
```bash
python3 {baseDir}/scripts/verify_system.py
```

## Manual Steps (if script unavailable)

### Step 1: Process Status
```bash
ps aux | grep -E "(runner_async|app.py|websocket_server)" | grep -v grep
```

Required processes:
- `runner_async` - Main trading engine (CRITICAL)
- `websocket_server` - Real-time updates
- `app.py` - Dashboard

### Step 2: Gateway Status
```bash
python3 scripts/gateway_manager.py status
```

### Step 3: Zombie Connection Check
```bash
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

Empty = GOOD. Any output = BAD (zombies blocking API).

### Step 4: Risk Parameters
Check `.env` for:
- `MAX_OPEN_POSITIONS` (default: 10)
- `STOP_LOSS_PERCENT` (default: 2.0)
- `EXECUTION_MODE` (should be `paper` for testing)

### Step 5: Safety Tests
```bash
python3 test_safety_features.py
```

### Step 6: Market Hours
```bash
python3 -c "from robo_trader.market_hours import is_market_open, get_market_session; print(f'Market open: {is_market_open()}'); print(f'Session: {get_market_session()}')"
```

### Step 7: Recent Errors
```bash
tail -50 robo_trader.log | grep -E "(ERROR|CRITICAL|Exception)"
```

## Output Format

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

**CRITICAL:** If `runner_async` is not running, status is NOT READY.

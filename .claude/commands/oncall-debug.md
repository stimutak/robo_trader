---
description: Systematic production debugging workflow
---

# On-Call Debug Workflow

Systematic debugging for production issues in the trading system.

## Step 1: Gather Initial Context

```bash
# Check if system is running
pgrep -f "runner_async" && echo "Runner: RUNNING" || echo "Runner: STOPPED"
pgrep -f "app.py" && echo "Dashboard: RUNNING" || echo "Dashboard: STOPPED"
pgrep -f "websocket_server" && echo "WebSocket: RUNNING" || echo "WebSocket: STOPPED"
```

## Step 2: Check Gateway Status

```bash
python3 scripts/gateway_manager.py status
```

Common issues:
- Gateway not running → `./scripts/start_gateway.sh paper`
- Zombie connections → `python3 scripts/gateway_manager.py restart`

## Step 3: Review Recent Logs

```bash
# Last 100 lines, focus on errors
tail -100 robo_trader.log | grep -E "(ERROR|CRITICAL|Exception|Traceback)"

# Full recent context
tail -200 robo_trader.log
```

## Step 4: Check for Known Issues

Search CLAUDE.md for similar patterns:
- Type errors → Check Decimal/float usage
- Connection errors → Check for zombies
- Market hours → Verify 4:00 PM close time
- Async errors → Check await usage

Search handoff/ docs for past fixes:
```bash
grep -r "<error_pattern>" handoff/
```

## Step 5: Common Issue Patterns

### Connection Failures
```bash
# Check zombies
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# Clear and restart
pkill -9 -f "runner_async"
python3 scripts/gateway_manager.py restart
```

### Type Errors (Decimal/Float)
- Location: Usually in runner_async.py around line 1656
- Fix: Use `price_float` not `price` for calculations

### Market Hours Wrong
- Location: `robo_trader/market_hours.py` line 36
- Check: Close time should be `time(16, 0)` not `time(16, 30)`

### WebSocket Issues
- Check: `MONITORING_LOG_FORMAT=plain` in environment
- Restart: `pkill -9 -f "websocket_server"`

## Step 6: Quick Restart

If all else fails:
```bash
# Full restart
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
./START_TRADER.sh
```

## Step 7: Document Finding

If you find a new issue pattern, add it to:
1. CLAUDE.md → "Common Mistakes" section
2. Create handoff doc in `handoff/HANDOFF_<date>_<description>.md`

## Output Format

```markdown
## Debug Summary

**Issue:** Brief description
**Root Cause:** What caused it
**Fix Applied:** What was done
**Prevention:** How to prevent recurrence

### Logs Evidence
```
<relevant log snippets>
```

### Files Modified
- file1.py: Description of change
- file2.py: Description of change
```

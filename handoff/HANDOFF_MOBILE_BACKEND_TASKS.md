# Backend Tasks for Mobile App Support

**Created:** 2026-01-24
**Priority:** HIGH
**Source:** Mobile app development (feature/mobile-app branch)

---

## Overview

The mobile app is functional but needs these backend enhancements to be fully useful.

---

## Task 1: Per-Trade P&L Calculation (HIGH PRIORITY)

### Problem
The `/api/trades` endpoint returns `pnl: null` for all trades. This breaks:
- Trade P&L display (shows $0.00)
- Winners/Losers filter buttons
- Performance tracking

### Current Response
```json
{
  "trades": [
    {
      "id": 911,
      "symbol": "MGM",
      "side": "BUY",
      "quantity": 58,
      "price": 34.42,
      "pnl": null,  // <-- PROBLEM
      ...
    }
  ]
}
```

### Required Solution
Calculate realized P&L for SELL trades using FIFO matching.

### Files to Modify
- `robo_trader/database_async.py` - Add P&L calculation when recording trades
- `app.py` - Ensure `/api/trades` returns the calculated P&L

### Implementation Approach
```python
# When recording a SELL trade:
# 1. Find matching BUY trades for same symbol (FIFO order)
# 2. Calculate: realized_pnl = (sell_price - avg_buy_price) * quantity
# 3. Store realized_pnl with the trade record

# Example:
# BUY 100 AAPL @ $150
# SELL 50 AAPL @ $160
# realized_pnl = (160 - 150) * 50 = $500
```

### Acceptance Criteria
- [ ] SELL trades have accurate `pnl` values
- [ ] BUY trades show `pnl: 0` or `pnl: null` (unrealized)
- [ ] `/api/trades` returns P&L in response
- [ ] Mobile app Winners/Losers filters work

---

## Task 2: WebSocket Log Streaming (HIGH PRIORITY)

### Problem
Mobile app logs screen shows "Disconnected" or empty because logs only stream when trading runner is actively processing.

### Current Behavior
- WebSocket connects successfully
- But no log messages are sent unless runner is in a trading cycle
- Dashboard/API logs are not streamed

### Required Solution
Stream ALL application logs via WebSocket, not just runner logs.

### Files to Modify
- `robo_trader/websocket_server.py` - Ensure all logs are broadcast
- `robo_trader/logger.py` - Verify WebSocketLogProcessor captures all logs

### Implementation Approach
```python
# In logger.py - ensure WebSocketLogProcessor is registered globally
# Not just for runner module

# In websocket_server.py - verify send_log_message is called
# for all log levels from all sources
```

### Acceptance Criteria
- [ ] Logs stream when dashboard is running (even without runner)
- [ ] All log levels visible (DEBUG, INFO, WARNING, ERROR)
- [ ] Source field shows which module generated the log
- [ ] Mobile app receives logs in real-time

---

## Task 3: Production CORS (MEDIUM PRIORITY)

### Problem
Currently using `CORS(app)` which allows ALL origins. This is fine for development but not for production.

### Current Code (app.py:44)
```python
CORS(app)  # Enable CORS for mobile app access
```

### Required Solution
Whitelist specific origins for production.

### Implementation
```python
# For production:
CORS(app, origins=[
    'http://localhost:5555',  # Local dashboard
    'http://192.168.1.166:5555',  # Local network
    # Add production mobile app origin when known
])
```

### Acceptance Criteria
- [ ] Development still works (localhost)
- [ ] Mobile app on local network still works
- [ ] Unknown origins are rejected

---

## Testing

After implementing, test with mobile app:

```bash
# Terminal 1: Start backend
cd /Users/oliver/robo_trader
./START_TRADER.sh

# Terminal 2: Start mobile
cd /Users/oliver/robo_trader-mobile/mobile
npx expo start --lan

# Verify:
# 1. Trades show P&L values (not $0.00)
# 2. Winners/Losers filters work
# 3. Logs stream in real-time
```

---

## Mobile App Reference

The mobile app expecting these APIs is at:
- Location: `/Users/oliver/robo_trader-mobile/mobile/`
- Branch: `feature/mobile-app`
- API config: `mobile/lib/constants.ts`

After backend changes, sync to mobile worktree:
```bash
cd /Users/oliver/robo_trader-mobile
git fetch origin main
git merge origin/main
```

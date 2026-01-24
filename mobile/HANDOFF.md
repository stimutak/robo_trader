# Mobile App Handoff

**Date:** 2026-01-24
**Branch:** `feature/mobile-app`
**Location:** `/Users/oliver/robo_trader-mobile/mobile/`

---

## Current State

The mobile app is **fully functional** with real data from the backend API.

### Completed Features

| Feature | Status | Notes |
|---------|--------|-------|
| Home Screen | ✅ Working | Portfolio, positions, system status |
| Analytics | ✅ Working | SVG equity curve, performance metrics |
| Trades | ✅ Working | Summary card, filters, trade list |
| ML Screen | ✅ Working | Model status, predictions |
| Logs Screen | ⚠️ Partial | UI works, needs backend runner for data |
| Position Detail | ✅ Scaffold | Route exists, needs implementation |

---

## Session 2026-01-24 Changes

### Backend Changes (robo_trader repo)

1. **CORS Support** - Mobile app can now fetch API data
   - Added `flask-cors` package
   - File: `app.py` - `CORS(app)` enabled

2. **WebSocket Binding** - Changed from `localhost` to `0.0.0.0`
   - File: `robo_trader/websocket_server.py`

### Mobile App Changes

1. **Equity Chart** - Complete rewrite with SVG
   - Smooth bezier curve with gradient fill
   - Red/green coloring based on value
   - Fixed percentage to show displayed period change (not all-time)
   - File: `components/charts/EquityChart.tsx`

2. **Trade Summary Card** - Added at top of trades screen
   - Shows: Total, Buys, Sells, Volume
   - File: `app/(tabs)/trades.tsx`

3. **Filter Chips Fix** - Were stretching to full height
   - Added explicit height constraints
   - File: `app/(tabs)/trades.tsx`

4. **API Debugging** - Added console logging
   - File: `lib/api.ts`

---

## To Run

### Start Backend
```bash
cd /Users/oliver/robo_trader
./START_TRADER.sh
```

### Start Mobile App
```bash
cd /Users/oliver/robo_trader-mobile/mobile
npx expo start --lan
```

Then scan QR code with iPhone camera (not Expo Go app directly).

---

## Configuration

### API Endpoints (`lib/constants.ts`)
```typescript
const DEV_HOST = '192.168.1.166';  // Mac's local IP - UPDATE IF CHANGED
export const API_BASE = __DEV__ ? `http://${DEV_HOST}:5555` : 'http://localhost:5555';
export const WS_URL = __DEV__ ? `ws://${DEV_HOST}:8765` : 'ws://localhost:8765';
```

**Note:** If Mac's IP changes, update `DEV_HOST`.

---

## Known Issues

| Issue | Cause | Fix Location |
|-------|-------|--------------|
| Logs "Disconnected" | WebSocket connects but no logs stream | `main` branch |
| Trade P&L = $0.00 | API returns `pnl: null` | `main` branch |
| Winners/Losers filters broken | No P&L data to filter | `main` branch |

---

## Backend Tasks (Do in main branch)

These changes need to be made in `/Users/oliver/robo_trader` on the `main` branch:

### 1. Per-Trade P&L Calculation (HIGH PRIORITY)
**Files:** `robo_trader/database_async.py`, `app.py`

```python
# In database_async.py - add to record_trade():
# Calculate realized P&L for SELL trades using FIFO matching
# Store in trades table
```

**Steps:**
1. Add `realized_pnl` column to trades table
2. When recording SELL, match against BUY trades (FIFO)
3. Calculate: `realized_pnl = (sell_price - avg_buy_price) * quantity`
4. Return P&L in `/api/trades` response

### 2. WebSocket Log Streaming (HIGH PRIORITY)
**Files:** `robo_trader/websocket_server.py`, `robo_trader/logger.py`

**Problem:** Logs only stream when trading runner is active

**Solution:** Stream ALL application logs, not just runner logs

See: `handoff/HANDOFF_WEBSOCKET_LOG_STREAMING.md`

### 3. Production CORS (MEDIUM PRIORITY)
**File:** `app.py`

Currently using `CORS(app)` which allows all origins. For production:
```python
CORS(app, origins=['https://your-app-domain.com'])
```

---

## Mobile Tasks (Do in feature/mobile-app branch)

These changes go in `/Users/oliver/robo_trader-mobile/mobile/`:

### 1. Position Detail Screen (HIGH PRIORITY)
**File:** `app/position/[symbol].tsx`

- Show position stats (entry, current, P&L)
- Price chart for symbol
- Close position button (future)

### 2. Error/Loading States (HIGH PRIORITY)
- Add error boundaries
- Loading skeletons for all screens
- Retry buttons on failure

### 3. EAS Build Setup (MEDIUM PRIORITY)
- Configure `eas.json`
- Set production API URL
- Submit to TestFlight

---

## Parallel Development Workflow

See `IMPLEMENTATION_PLAN.md` for full details.

**Quick Reference:**

```bash
# Backend work (main branch):
cd /Users/oliver/robo_trader
# make changes, commit, push to main

# Mobile work (feature branch):
cd /Users/oliver/robo_trader-mobile
git merge origin/main  # sync backend changes first
# make changes in mobile/, commit, push to feature/mobile-app

# Final merge when ready:
cd /Users/oliver/robo_trader
git merge feature/mobile-app
```

**Key Rule:** Backend files → main branch only. Mobile files → feature branch only.

---

## Git Setup

This is a **git worktree**:
- **Main repo:** `/Users/oliver/robo_trader` (branch: `main`)
- **Worktree:** `/Users/oliver/robo_trader-mobile` (branch: `feature/mobile-app`)

To merge into main:
```bash
cd /Users/oliver/robo_trader
git merge feature/mobile-app
```

---

## Tech Stack

- React Native + Expo SDK 54
- Expo Router (file-based routing)
- react-native-svg (charts)
- TanStack Query (data fetching)
- Zustand (state management)
- TypeScript

---

**Ready to use. Data flows from backend API to mobile app.**

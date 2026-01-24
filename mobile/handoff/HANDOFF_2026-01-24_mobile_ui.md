# Mobile App Handoff - 2026-01-24

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

## This Session's Changes

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

## Known Issues (Backend - Already Handed Off)

| Issue | Cause | Fix Location |
|-------|-------|--------------|
| Logs "Disconnected" | WebSocket connects but no logs stream | `main` branch |
| Trade P&L = $0.00 | API returns `pnl: null` | `main` branch |
| Winners/Losers filters broken | No P&L data to filter | `main` branch |

Backend tasks are documented in: `/Users/oliver/robo_trader/handoff/HANDOFF_MOBILE_BACKEND_TASKS.md`

---

## Mobile Tasks (YOUR WORK)

### 1. Position Detail Screen (HIGH PRIORITY)
**File:** `app/position/[symbol].tsx`

- Show position stats (entry, current, P&L)
- Price chart for symbol
- Close position button (future)

### 2. Error/Loading States (HIGH PRIORITY)
- Add error boundaries
- Loading skeletons for all screens
- Retry buttons on failure

### 3. Pull-to-Refresh
- Already on trades screen, add to all screens

### 4. EAS Build Setup (MEDIUM PRIORITY)
- Configure `eas.json`
- Set production API URL
- Submit to TestFlight

---

## After Backend Changes Are Ready

When the main branch has P&L calculation and log streaming working:

```bash
cd /Users/oliver/robo_trader-mobile
git fetch origin main
git merge origin/main
```

Then test:
- Trades show real P&L values
- Winners/Losers filters work
- Logs stream in real-time

---

## Parallel Development Workflow

| Location | Branch | Purpose |
|----------|--------|---------|
| `/Users/oliver/robo_trader` | `main` | Everything except mobile |
| `/Users/oliver/robo_trader-mobile` | `feature/mobile-app` | Mobile app only |

### Key Rules

1. **You work in:** `/Users/oliver/robo_trader-mobile/mobile/`
2. **You edit:** Only files in `mobile/` directory
3. **Never edit:** Backend files (`app.py`, `robo_trader/*.py`)
4. **Sync backend changes:** `git merge origin/main` when needed

### Your Commit Flow
```bash
cd /Users/oliver/robo_trader-mobile
git add mobile/
git commit -m "feat: description"
git push origin feature/mobile-app
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

## File Structure

```
mobile/
├── app/                    # Expo Router screens
│   ├── (tabs)/            # Tab screens
│   │   ├── index.tsx      # Home
│   │   ├── analytics.tsx  # Analytics
│   │   ├── ml.tsx         # ML
│   │   ├── trades.tsx     # Trades
│   │   └── logs.tsx       # Logs
│   ├── position/
│   │   └── [symbol].tsx   # Position detail (needs work)
│   └── _layout.tsx        # Root layout
├── components/
│   ├── charts/
│   │   └── EquityChart.tsx # SVG equity chart
│   └── ui/
│       └── Card.tsx       # Reusable card component
├── hooks/
│   ├── useAPI.ts          # TanStack Query hooks
│   └── useWebSocket.ts    # WebSocket connection
├── lib/
│   ├── api.ts             # API client
│   ├── constants.ts       # API URLs, colors
│   ├── types.ts           # TypeScript types
│   └── store.ts           # Zustand stores
├── handoff/               # Handoff documents
│   └── HANDOFF_2026-01-24_mobile_ui.md  # This file
└── HANDOFF.md             # Links to latest handoff
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start dev | `npx expo start --lan` |
| Run backend | `cd /Users/oliver/robo_trader && ./START_TRADER.sh` |
| Sync backend | `git fetch origin main && git merge origin/main` |
| Push changes | `git add mobile/ && git commit -m "feat: ..." && git push` |

---

**Focus on: Position detail screen and error states.**

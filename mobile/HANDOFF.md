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

1. **Logs "Disconnected"** - WebSocket connects but no logs because runner not active
2. **Trade P&L = $0.00** - API returns `pnl: null` for trades
3. **Winners/Losers filters** - Don't work due to missing P&L data

---

## Future Work

### High Priority
1. **WebSocket log streaming** - Backend needs to send logs (see `handoff/HANDOFF_WEBSOCKET_LOG_STREAMING.md`)
2. **Per-trade P&L** - Backend should calculate P&L for trades
3. **Production build** - EAS Build for TestFlight

### Medium Priority
4. **Position detail screen** - Implement chart and actions
5. **Trade detail view** - Tap for full details
6. **Push notifications** - Trade alerts

### Low Priority
7. **Offline caching**
8. **Biometric auth**
9. **iOS widget**

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

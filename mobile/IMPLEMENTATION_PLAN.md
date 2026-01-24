# Mobile App Implementation Plan

**Created:** 2026-01-24
**Status:** Phase 1 Complete, Phase 2 Ready

---

## Overview

This plan covers completing the RoboTrader mobile app and required backend enhancements.

---

## Parallel Development Strategy

### Repository Setup

| Location | Branch | Purpose |
|----------|--------|---------|
| `/Users/oliver/robo_trader` | `main` | Backend + Web Dashboard |
| `/Users/oliver/robo_trader-mobile` | `feature/mobile-app` | Mobile App |

### Rules for Parallel Development

1. **Backend changes → main branch**
   - API endpoints, WebSocket server, database changes
   - These serve both web dashboard and mobile app
   - Work in: `/Users/oliver/robo_trader`

2. **Mobile-only changes → feature/mobile-app**
   - React Native components, screens, mobile UI
   - Work in: `/Users/oliver/robo_trader-mobile/mobile/`

3. **Sync regularly**
   ```bash
   # After backend changes in main, update worktree:
   cd /Users/oliver/robo_trader-mobile
   git fetch origin main
   git merge origin/main
   ```

4. **Final merge when mobile is ready**
   ```bash
   cd /Users/oliver/robo_trader
   git merge feature/mobile-app
   ```

### Avoiding Conflicts

- **Never edit the same file in both branches** at the same time
- Backend files (`app.py`, `robo_trader/*.py`) → main only
- Mobile files (`mobile/**`) → feature/mobile-app only
- Shared files (`requirements.txt`, `CLAUDE.md`) → main, then sync

---

## Phase 1: Foundation (COMPLETE ✅)

### Mobile App
- [x] Project setup with Expo SDK 54
- [x] Tab navigation (Home, Analytics, ML, Trades, Logs)
- [x] API client with TanStack Query
- [x] WebSocket hook for log streaming
- [x] Zustand stores for state management
- [x] Dark theme throughout

### Backend
- [x] CORS enabled for mobile API access
- [x] WebSocket bound to 0.0.0.0 for network access

---

## Phase 2: Data Quality (NEXT)

### 2.1 Per-Trade P&L Calculation
**Branch:** `main`
**Files:** `robo_trader/database_async.py`, `app.py`

**Problem:** Trades API returns `pnl: null` for all trades

**Solution:**
- Calculate realized P&L when SELL trades are recorded
- Match SELLs to BUYs using FIFO
- Store `realized_pnl` in trades table

**Tasks:**
- [ ] Add `realized_pnl` column to trades table
- [ ] Implement FIFO matching in trade recording
- [ ] Update `/api/trades` to return calculated P&L
- [ ] Backfill existing trades with P&L

### 2.2 WebSocket Log Streaming
**Branch:** `main`
**Files:** `robo_trader/websocket_server.py`, `robo_trader/logger.py`

**Problem:** Logs only stream when runner is active

**Solution:**
- Stream all application logs via WebSocket
- Include dashboard logs, not just runner logs

**Tasks:**
- [ ] Enhance `WebSocketLogProcessor` to capture all logs
- [ ] Add log level filtering on server side
- [ ] Test with mobile app connected

### 2.3 Mobile App Polish
**Branch:** `feature/mobile-app`
**Files:** `mobile/app/`, `mobile/components/`

**Tasks:**
- [ ] Position detail screen (`/position/[symbol]`)
- [ ] Pull-to-refresh on all screens
- [ ] Error states for failed API calls
- [ ] Loading skeletons

---

## Phase 3: Production Ready

### 3.1 Build & Deploy
**Branch:** `feature/mobile-app`

**Tasks:**
- [ ] Set up EAS Build
- [ ] Configure production API URL
- [ ] Create app icons and splash screen
- [ ] TestFlight submission
- [ ] App Store submission

### 3.2 Environment Configuration
**Branch:** `main`

**Tasks:**
- [ ] Production CORS whitelist (not allow-all)
- [ ] API rate limiting for mobile
- [ ] Authentication token support

---

## Phase 4: Enhanced Features

### 4.1 Push Notifications
**Branch:** Both (backend + mobile)

**Tasks:**
- [ ] Expo Push Notifications setup
- [ ] Backend notification triggers (trade executed, error, etc.)
- [ ] Notification preferences in app

### 4.2 Offline Support
**Branch:** `feature/mobile-app`

**Tasks:**
- [ ] Cache API responses locally
- [ ] Show cached data when offline
- [ ] Sync when connection restored

### 4.3 iOS Widget
**Branch:** `feature/mobile-app`

**Tasks:**
- [ ] Home screen widget showing P&L
- [ ] Lock screen widget

---

## Task Assignment by Branch

### main branch (Backend)
| Task | Priority | Effort |
|------|----------|--------|
| Per-trade P&L calculation | High | Medium |
| WebSocket log streaming | High | Low |
| Production CORS config | Medium | Low |
| Push notification backend | Low | Medium |

### feature/mobile-app (Mobile)
| Task | Priority | Effort |
|------|----------|--------|
| Position detail screen | High | Low |
| Error/loading states | High | Low |
| EAS Build setup | Medium | Medium |
| Offline caching | Low | High |
| iOS Widget | Low | High |

---

## Development Workflow

### Starting a Backend Task
```bash
cd /Users/oliver/robo_trader
git pull origin main
# Make changes
git add -A && git commit -m "feat: description"
git push origin main
```

### Starting a Mobile Task
```bash
cd /Users/oliver/robo_trader-mobile
git pull origin main  # Get latest backend
git merge origin/main  # Merge into feature branch
# Make changes in mobile/
git add -A && git commit -m "feat: description"
git push origin feature/mobile-app
```

### Testing Together
```bash
# Terminal 1: Backend
cd /Users/oliver/robo_trader
./START_TRADER.sh

# Terminal 2: Mobile
cd /Users/oliver/robo_trader-mobile/mobile
npx expo start --lan
```

---

## Success Criteria

### Phase 2 Complete When:
- [ ] All trades show accurate P&L
- [ ] Winners/Losers filters work correctly
- [ ] Logs stream in real-time when app connected
- [ ] Position detail screen implemented

### Phase 3 Complete When:
- [ ] App available on TestFlight
- [ ] Production API secured
- [ ] No crashes in 7-day testing period

### Phase 4 Complete When:
- [ ] Push notifications working
- [ ] App usable offline
- [ ] Widget shows live data

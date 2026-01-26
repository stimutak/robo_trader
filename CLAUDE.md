# RoboTrader Project Guidelines

## 🚨 CRITICAL: NEVER DELETE USER DATA 🚨

**ABSOLUTELY NEVER delete, wipe, or "clean up" database data without EXPLICIT user permission.**

This includes:
- NEVER run `DELETE FROM trades` or similar
- NEVER "start fresh" or "nuclear option" the database
- NEVER assume duplicate data should be removed
- NEVER wipe equity_history, positions, or account tables

**If you see data issues:**
1. STOP and explain the problem to the user
2. ASK what they want to do
3. ALWAYS make a backup FIRST if any changes are needed
4. Let the USER decide if data should be deleted

**The user's trading history is IRREPLACEABLE. Deleting it destroys months of records.**

Violation of this rule caused catastrophic data loss on 2026-01-26. DO NOT REPEAT THIS MISTAKE.

---

## Project Phase Plan
**IMPORTANT:** The authoritative phase plan is in `IMPLEMENTATION_PLAN.md`. This is the ML-focused 4-phase plan:
- Phase 1: Foundation & Quick Wins (Tasks F1-F5) - COMPLETE ✅
- Phase 2: ML Infrastructure & Backtesting (Tasks M1-M5) - COMPLETE ✅
- Phase 3: Advanced Strategy Development (Tasks S1-S5) - COMPLETE ✅
- Phase 4: Stabilization & Code Quality (Tasks P1-P10) - REVISED 2026-01-15

**Current Status:** Phase 4 IN PROGRESS - P1-P7, P9-P10 complete (80%), P8 in progress

**Note:** The older 9-phase plan in `archived_plans/PROJECT_PLAN_9PHASE.md` is deprecated and should NOT be used. Any references to "Phase 5", "Phase 6" etc. from older commits refer to the old plan and should be ignored.

## ⚠️ IBKR Gateway Management (Updated 2025-12-03)

### Automated Gateway Management via IBC

The system now uses **IBC (IB Controller)** for automated Gateway management. Gateway restarts are **fully automated** when zombie connections are detected.

**How it works:**
- `START_TRADER.sh` automatically starts Gateway via IBC if not running
- Detects zombie CLOSE_WAIT connections that block API handshakes
- Automatically restarts Gateway to clear zombies (up to 3 retries)
- Tests actual API connectivity before proceeding
- Only requires manual 2FA on your phone when Gateway starts

**IBC Configuration:**
- Config file: `config/ibc/config.ini` (gitignored - contains credentials)
- Template: `config/ibc/config.ini.template`
- IBC scripts: `IBCMacos-3/` (macOS) or `IBCWin-3/` (Windows)

### Gateway Startup Flow

1. `./START_TRADER.sh` is the **only command you need**
2. If Gateway not running → starts via IBC automatically
3. If zombies detected → kills Python zombies, restarts Gateway if needed
4. Tests API connectivity with actual handshake
5. Retries up to 3 times if connection fails
6. Only proceeds when API is confirmed working

### What You Need to Do

**First-time setup:**
```bash
cp config/ibc/config.ini.template config/ibc/config.ini
# Edit config.ini with your IBKR credentials:
#   IbLoginId=YOUR_USERNAME
#   IbPassword=YOUR_PASSWORD
```

**Starting the trader:**
```bash
./START_TRADER.sh
# Watch for 2FA prompt on your IBKR Mobile app
# The script handles everything else automatically
```

### Gateway Management Commands

```bash
# THE ONLY WAY TO START THE TRADING SYSTEM:
./START_TRADER.sh

# Debugging/diagnostic commands (not for normal startup):
./scripts/start_gateway.sh paper            # Start Gateway only (debugging)
python3 scripts/gateway_manager.py status   # Check Gateway status
python3 scripts/gateway_manager.py restart  # Force restart Gateway (clears zombies)
tail -f config/ibc/logs/*.txt               # View Gateway logs
```

### Zombie Connections

**What are zombies?**
- CLOSE_WAIT connections from failed/incomplete API handshakes
- They block ALL future API connections until cleared
- Gateway-owned zombies require Gateway restart to clear

**The startup script handles this automatically:**
1. Detects zombie connections via `lsof`
2. Kills Python-owned zombies
3. If Gateway-owned zombies remain → restarts Gateway
4. Verifies API works before proceeding

**Manual zombie check:**
```bash
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

### Safe Process Management

**Kill Python processes (always safe):**
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

**Kill Gateway (triggers auto-restart on next START_TRADER.sh):**
```bash
pkill -f "IB Gateway"
```

### Gateway API Notes

- **Port 4002**: Paper trading API
- **Port 4001**: Live trading API
- ActiveX/Socket Clients is **permanently enabled** in IB Gateway (cannot be disabled)
- API permissions are configured in your IBKR account settings
- System uses **readonly** connections (no order placement via API)

## Mobile App & Parallel Development

### Repository Structure

| Location | Branch | Purpose |
|----------|--------|---------|
| `/Users/oliver/robo_trader` | `main` | Backend, API, Web Dashboard |
| `/Users/oliver/robo_trader-mobile` | `feature/mobile-app` | React Native Mobile App |

The mobile app lives in a **git worktree** linked to this repo.

### Parallel Development Rules

**CRITICAL: Follow these rules to avoid conflicts**

1. **ALL changes EXCEPT mobile → THIS repo (main branch)**
   - Trading system (`robo_trader/`)
   - Web dashboard (`app.py`, templates)
   - Scripts (`scripts/`)
   - Configuration files
   - Documentation (`*.md`, `handoff/`)
   - Everything that isn't in `mobile/`

2. **Mobile app ONLY → worktree (feature/mobile-app)**
   - React Native code (`mobile/**`)
   - Mobile UI components
   - Mobile-specific configs (`mobile/lib/constants.ts`)

3. **Never edit the same file in both branches simultaneously**

### Syncing Between Repos

```bash
# After making backend changes here, sync to mobile worktree:
cd /Users/oliver/robo_trader-mobile
git fetch origin main
git merge origin/main

# When mobile app is ready to merge:
cd /Users/oliver/robo_trader
git merge feature/mobile-app
```

### Mobile App Status

See `mobile/HANDOFF.md` and `mobile/IMPLEMENTATION_PLAN.md` in the worktree for current status.

**Quick start mobile dev:**
```bash
cd /Users/oliver/robo_trader-mobile/mobile
npx expo start --lan
```

---

## Key Project Files
- `IMPLEMENTATION_PLAN.md` - The active project roadmap
- `handoff/LATEST_HANDOFF.md` - Latest session handoff document
- `robo_trader/runner_async.py` - Main trading system with async parallel processing
- `robo_trader/runner/` - Extracted runner modules (Phase 4 P8 - IN PROGRESS)
  - `data_fetcher.py` - Market data retrieval and caching
  - `subprocess_manager.py` - IBKR subprocess health monitoring
- `robo_trader/exceptions.py` - Custom exception hierarchy (Phase 4 P9)
- `robo_trader/data_providers/` - Data provider abstraction (Polygon.io ready)
- `robo_trader/database_async.py` - Async database with equity_history table for portfolio tracking
- `sync_db_reader.py` - Sync database reader for dashboard access
- `app.py` - Dashboard with comprehensive professional overview
- `robo_trader/websocket_server.py` - WebSocket server for real-time updates
- `robo_trader/features/` - Feature engineering pipeline (Phase 2 - COMPLETE)
- `robo_trader/ml/` - ML model training & selection (Phase 2 - COMPLETE)
- `robo_trader/analytics/` - Performance analytics (Phase 2 - COMPLETE)
- `robo_trader/backtesting/` - Walk-forward backtesting (Phase 2 - COMPLETE)
- `scripts/gateway_manager.py` - Cross-platform Gateway management
- `scripts/start_gateway.sh` - Gateway launcher script
- `config/ibc/config.ini` - IBC credentials (gitignored)

## IMPORTANT: Python Command on macOS
**ALWAYS USE `python3` NOT `python` - THIS SYSTEM USES macOS WITH NO `python` COMMAND**

## Starting the Trading System

### The Only Command You Need

```bash
./START_TRADER.sh
# Or with custom symbols:
./START_TRADER.sh "AAPL,NVDA,TSLA"
```

**⚠️ IMPORTANT: Always use `./START_TRADER.sh` - NEVER start components manually!**

Do NOT run these directly:
- ❌ `python3 runner_async.py` - Use START_TRADER.sh instead
- ❌ `scripts/start_gateway.sh` - Use START_TRADER.sh instead
- ❌ `python3 app.py` - Use START_TRADER.sh instead
- ❌ `python3 websocket_server.py` - Use START_TRADER.sh instead

The script handles Gateway startup, zombie cleanup, connectivity testing, and proper sequencing. Starting components manually bypasses these safety checks.

**What the script does:**
1. ✅ Kills existing Python trader processes
2. ✅ Starts Gateway via IBC if not running
3. ✅ Detects and cleans up zombie connections
4. ✅ Restarts Gateway automatically if zombies block API
5. ✅ Tests actual API connectivity (not just port open)
6. ✅ Retries up to 3 times on failure
7. ✅ Starts WebSocket server, trading system, and dashboard
8. ✅ Monitors startup for 10 seconds

### Default Symbols (from user_settings.json)
```
AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF
```

### Diagnostic Commands

```bash
# Check Gateway status
python3 scripts/gateway_manager.py status

# Check for zombie connections
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# View trader logs
tail -f robo_trader.log

# View Gateway/IBC logs
tail -f config/ibc/logs/*.txt
```

### Testing Commands

```bash
# Activate virtual environment first
source .venv/bin/activate

# Run tests
pytest

# Test ML pipeline
python3 test_ml_pipeline.py

# Test safety features
python3 test_safety_features.py
```

## Current Issues Status
1. ✅ WebSocket connection handler signature error - FIXED
2. ✅ JSON serialization error with ServerConnection object - FIXED
3. ✅ Phase 1 F2: Upgrade config to Pydantic - COMPLETED
4. ✅ WebSocket stability - Fixed with client/server separation
5. ✅ TWS API Connection Issues - RESOLVED with subprocess approach
6. ✅ Zombie connection cleanup - AUTOMATED with Gateway restart (2025-12-03)
7. ✅ Gateway connectivity testing - BUILT INTO startup script with auto-restart
8. ✅ Subprocess worker connection failure - RESOLVED (2025-11-24)
9. ✅ IBC Integration - Gateway auto-start and zombie handling (2025-12-03)
10. ✅ **Socket Zombie Creation Bug - FIXED (2025-12-06)** - See below
11. ✅ **Dashboard Connection Status Accuracy - IMPROVED (2025-12-10)** - See below
12. ✅ **Subprocess Pipe Blocking - FIXED (2025-12-24)** - See below
13. ✅ **Near Real-Time Trading - IMPLEMENTED (2025-12-24)** - See below
14. ✅ **Decimal/Float Type Mismatch - FIXED (2025-12-29, enhanced 2026-01-15)** - See below
15. ✅ **Market Close Time Wrong - FIXED (2025-12-29)** - Was 4:30 PM, now 4:00 PM
16. ✅ **Int/Datetime Comparison Error - FIXED (2026-01-15)** - Added try/except fallback in correlation.py
17. ✅ **Missing Market Holidays - FIXED (2026-01-15)** - Added MLK, Presidents, Good Friday, Memorial, Labor, Thanksgiving, Juneteenth
18. ✅ **Dashboard Overview Redesign - IMPLEMENTED (2026-01-24)** - Professional-grade overview with all key metrics
19. ✅ **Equity History Tracking - IMPLEMENTED (2026-01-24)** - Daily portfolio snapshots in `equity_history` table

## Dashboard Overview (2026-01-24)

The dashboard overview now shows comprehensive professional trading metrics:

**Hero Row:**
- Total Equity (prominent, with % return since inception)
- Today's P&L ($ and %)
- Unrealized P&L (open positions)
- Realized P&L (closed trades)

**Risk Row:**
- Positions Value, Cash Available, Exposure %, Current Drawdown, Max Drawdown, Buying Power

**Main Row:**
- 30-day Portfolio Value chart (uses `equity_history` table)
- Position Summary (count, winners/losers, best/worst, avg size)

**Strategy Row:**
- Win Rate, Profit Factor, Sharpe Ratio, Total Trades
- Avg Win, Avg Loss, Best Trade, Worst Trade
- Recent Trades list

**Status Row:**
- Gateway connection, Market status, Next open/close, Last update, Cycle interval

## Equity History Tracking (2026-01-24)

Portfolio value is tracked daily using the `equity_history` table (industry standard approach).

**Table Schema:**
```sql
CREATE TABLE equity_history (
    date TEXT NOT NULL UNIQUE,
    equity REAL NOT NULL,
    cash REAL DEFAULT 0,
    positions_value REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

**How it works:**
- Snapshots saved at end of each trading cycle via `save_equity_snapshot()`
- One snapshot per day (updates if same day)
- Used by `/api/equity-curve` endpoint for portfolio value chart
- Accessible via `SyncDatabaseReader.get_equity_history()`

## AI-Driven Symbol Discovery (2026-01-14)

**The system discovers new trading opportunities from news - DO NOT manually expand symbol lists.**

### How It Works
1. **Base Symbols** in `.env` `SYMBOLS=` - Your watchlist (20 stocks)
2. **Existing Positions** - Automatically added for SELL signal monitoring
3. **AI Discovery** - Scans 50+ news headlines per cycle, finds new opportunities

### News Sources (12 RSS feeds)
- Yahoo Finance (top stories + tech)
- Reuters (markets + business)
- CNBC (top + investing)
- MarketWatch (top + market pulse)
- Seeking Alpha (currents + news)
- TechCrunch, Benzinga

### AI Discovery Flow
```
fetch_rss_news(50 headlines) → AI finds opportunities → Adds to processing queue
```

**Example discoveries:** OKTA (Cantor upgrade), SNPS (Loop Capital AI bullish)

### Important Behavior
- **"Already have long position"** = CORRECT behavior, not a bug
- System prevents duplicate positions in same stock
- New positions only opened for stocks NOT already owned
- AI confidence threshold: 50%+ required

### DO NOT
- Arbitrarily add stocks to SYMBOLS list
- Expand symbol list without AI/news basis
- Confuse "no new buys" with "system broken" (may already own those stocks)

## Critical Safety Features (2025-09-27) ✅
**Added to address audit findings:**
- **Order Management** (`order_manager.py`) - Full lifecycle tracking with retry logic
- **Data Validation** (`data_validator.py`) - Market data quality checks
- **Circuit Breaker** (`circuit_breaker.py`) - Fault tolerance system
- **Safety configs in `.env`** - MAX_OPEN_POSITIONS, STOP_LOSS_PERCENT, etc.
- Run `python3 test_safety_features.py` to validate all safety features

## Security Enhancements (2025-09-28) ✅
**Critical security vulnerability fixed:**
- **Secure Configuration** (`utils/secure_config.py`) - Validates and masks sensitive data
- **API Key Masking** - All sensitive values masked in logs (shows `1234****` instead of full value)
- **Required Config Validation** - Fails fast if critical configs missing
- **Port/Mode Consistency** - Prevents accidental live trading with paper ports
- All IBKR client IDs, accounts, and API keys now properly secured

## Decimal Precision Fix (2025-09-28) ✅
**PR #39 merged - Float precision errors eliminated:**
- **Portfolio** uses `Decimal` for cash, realized_pnl, avg_price
- **RiskManager** uses `Decimal` for position calculations
- **PrecisePricing** utilities handle all financial arithmetic
- Eliminates order rejections due to float precision errors
- See `DECIMAL_PRECISION_FIX.md` for details

## Major Fixes Completed

### Near Real-Time Trading System (2025-12-24) ✅

**System now runs with ~15-second latency during market hours.**

**Polling Intervals by Market State:**
| Market State | Polling Interval | Notes |
|-------------|------------------|-------|
| **Market Open** | 15 seconds | Near real-time with 1-minute bars |
| **Pre/After Hours** | 2 minutes | Extended hours data still available |
| **Near Open (<1hr)** | 5 minutes | Preparing for market open |
| **Closed (overnight/weekend)** | 30 minutes max | Conserve resources |

**Configuration Changes:**
- `bar_size`: 30 min → **1 minute** (finer granularity)
- `duration`: 10 days → **1 day** (faster data fetch)
- `interval_seconds`: 300 → **15** (near real-time)

**Files Modified:** `robo_trader/runner_async.py`

**Future Enhancement:** True streaming with `reqMktData()` for <1 second latency (Phase 2 planned)

### Trading Bug Fixes (2025-12-29) ✅

**Fixed two critical bugs preventing trade execution:**

1. **Decimal/Float Type Mismatch**
   - **File:** `runner_async.py` line 1656
   - **Error:** `unsupported operand type(s) for /: 'float' and 'decimal.Decimal'`
   - **Fix:** Changed `current_price=price` to `current_price=price_float`

2. **Market Close Time Wrong**
   - **File:** `market_hours.py` line 36
   - **Error:** Market showing as "open" after 4:00 PM
   - **Fix:** Changed `time(16, 30)` to `time(16, 0)` (4:30 PM → 4:00 PM)

**Known Issue (Open):**
- **Int/Datetime Comparison Error** affecting GM/GOLD symbols
- Error: `'>=' not supported between instances of 'int' and 'datetime.datetime'`
- Needs further investigation

**See:** `handoff/HANDOFF_2025-12-29_trading_bugs_fixed.md`

### Subprocess Pipe Blocking Fix (2025-12-24) ✅

**CRITICAL FIX:** Resolved race condition where data fetch commands were lost.

**Root Cause:** Worker used `run_in_executor` with timeout for `stdin.readline()`. When timeout fired, asyncio cancelled the future BUT the thread pool thread continued blocking. Next iteration spawned another thread, causing orphaned threads to consume data that was never returned.

**Fix:**
- Worker: Dedicated stdin reader thread with queue (no race condition)
- Parent: Direct `stdin.write()` without executor wrapper

**Files Modified:**
- `robo_trader/clients/ibkr_subprocess_worker.py` - Added `_stdin_reader()` thread
- `robo_trader/clients/subprocess_ibkr_client.py` - Direct stdin writes

**See:** `handoff/HANDOFF_2025-12-24_subprocess_pipe_fix_complete.md`

### Dashboard Connection Status Accuracy (2025-12-10) ✅

**Issue:** Dashboard showed "Connected" when Gateway was merely listening, even if no active API session existed.

**Fix:** Enhanced `check_ibkr_connection()` to distinguish between:
- **Gateway available** (port listening) - Gateway is running and can accept connections
- **API connected** (ESTABLISHED socket) - Runner has an active API session

**New Status Messages:**
- `✅ Market Open - API Connected` - Active IBKR API connection
- `🔄 Market Open - Waiting for cycle` - Gateway available, runner uses per-cycle connections
- `⚠️ Market Open - No Gateway` - Gateway/TWS not detected

**API Response Changes:**
- `connected` now means actual ESTABLISHED socket (was: just port listening)
- `api_connected` - explicit field for active connection status
- `gateway_available` - Gateway is listening and can accept connections

**File Modified:** `app.py` - `check_ibkr_connection()` function

### Socket Zombie Creation Bug Fix (2025-12-06) ✅

**CRITICAL FIX:** System now stable for 1+ hour continuous IBKR connection.

**Root Cause:** Three code locations used `socket.connect_ex()` to check if Gateway port was open. This creates a full TCP 3-way handshake, and when the socket is closed without completing the IBKR API handshake, Gateway sees it as an improperly disconnected client, creating CLOSE_WAIT zombie connections that block ALL future API connections.

**Fix:** Replaced `socket.connect_ex()` with `lsof -nP -iTCP:PORT -sTCP:LISTEN` which queries the kernel's socket table without creating any TCP connections.

**Files Fixed:**
- `app.py` - `check_ibkr_connection()` function (lines 2983-3041)
- `robo_trader/runner_async.py` - `test_port_open_lsof()` function (lines 460-497)
- `robo_trader/utils/tws_health.py` - `is_port_listening()` function (lines 141-180)
- `robo_trader/runner_async.py` - Fixed Python 3.12+ variable scoping error (removed redundant Path import)

**See:** `handoff/HANDOFF_2025-12-06_ZOMBIE_FIX.md` for complete details.

### IBC Gateway Integration (2025-12-03) ✅

**Automated Gateway Management:**
- `START_TRADER.sh` now handles everything automatically
- Gateway started via IBC with credentials from `config/ibc/config.ini`
- Zombie connections detected and Gateway restarted automatically
- API connectivity verified before proceeding
- Up to 3 retry attempts on failure

**Files Added/Modified:**
- `config/ibc/config.ini.template` - IBC config template
- `scripts/gateway_manager.py` - Cross-platform Gateway management
- `scripts/start_gateway.sh` - Gateway launcher
- `START_TRADER.sh` - Full auto-start with retry logic
- `IBCMacos-3/gatewaystartmacos.sh` - Environment variable support

### TWS API Connection Resolution ✅
**Problem:** Async context (`patchAsyncio()`) caused TWS API handshake timeouts and stuck connections
**Solution:** Implemented subprocess-based IBKR operations for complete async isolation

**Key Changes:**
- Created `SyncIBKRWrapper` class for thread-based operations
- Implemented subprocess approach for complete process isolation
- Fixed connection pooling complexity (removed, simplified to direct connections)
- Enhanced client ID management (unique timestamp + PID based IDs)
- Comprehensive error handling and cleanup

### Library Migration Notes (2025-09-27)
- **MIGRATION COMPLETE:** Successfully migrated from `ib_insync` to `ib_async` v2.0.1
- ib_insync author passed away early 2024, library archived March 2024
- ib_async is the community-maintained fork, drop-in replacement
- All imports updated: `from ib_insync` → `from ib_async`
- Old ib_insync library has been uninstalled
- System tested and running successfully with ib_async

### Subprocess Worker Connection Fix (2025-11-24) ✅

**CRITICAL FIX COMPLETED**: Resolved timing race condition in subprocess worker that caused connection failures.

**Problem Resolved:**
- **Issue**: Worker responded `{"connected": false, "accounts": []}` in ~163ms before handshake completed
- **Root Cause**: Subprocess client read response before worker finished IBKR connection process
- **Impact**: 0% connection success rate, system unable to start

**Solution Implemented:**
- **Synchronization Fix**: Added explicit `ib.isConnected()` polling loop with 2.0s stabilization wait
- **Zombie Prevention**: Pre-connection check detects CLOSE_WAIT zombies and aborts with clear instructions
- **Enhanced Debugging**: Worker stderr captured to `/tmp/worker_debug.log` for troubleshooting
- **Response Filtering**: JSON response filtering prevents ib_async log pollution

**Documentation:** See `docs/SUBPROCESS_WORKER_CONNECTION_FIX.md` for complete technical details.

### Gateway API Notes

- ActiveX/Socket Clients is **permanently enabled** in IB Gateway ≥10.41 and cannot be disabled
- System uses **READONLY mode** for TWS connections (no order placement via API)
- PaperExecutor handles orders separately
- No TWS security dialog popups with read-only connections

## WebSocket Fix Notes (2025-08-28)
- Fixed handler signature by adding `path` parameter
- Fixed JSON serialization by using structlog properly
- Disabled websockets library debug logging to prevent ServerConnection serialization
- Set MONITORING_LOG_FORMAT=plain when running dashboard to avoid JSON issues
- Created WebSocket client (`websocket_client.py`) for proper client/server separation
- Runner now uses client to connect to existing server instead of direct import

## Development Guidelines
- Always refer to IMPLEMENTATION_PLAN.md for phase objectives
- Maintain backward compatibility with existing trading logic
- Test all changes with paper trading before live
- Document major changes in handoff documents
- Use `./START_TRADER.sh` for all startup - it handles Gateway automatically

---

## Common Mistakes (Auto-Updated)

**When Claude makes an error, add it here so it won't repeat.**

### 🚨 CRITICAL - Data Destruction (NEVER DO THESE)
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Deleting trades/positions to "clean up" duplicates | ASK USER FIRST, make backup, let user decide | 2026-01-26 |
| "Nuclear option" / "start fresh" on database | NEVER - user data is irreplaceable | 2026-01-26 |
| Wiping equity_history to fix graphs | Add missing data, don't delete existing | 2026-01-26 |
| Assuming bad data should be removed | Explain problem to user, let them decide | 2026-01-26 |

### Type Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Using `price` (Decimal) in float division | Use `price_float` for calculations | 2025-12-29 |
| Market close at 4:30 PM | Close is 4:00 PM ET (`time(16, 0)`) | 2025-12-29 |
| Int/datetime comparison | Ensure both operands are same type | 2025-12-29 |
| `portfolio.equity()` returns Decimal | Always `float(equity)` before math ops | 2026-01-15 |
| `portfolio.realized_pnl` is Decimal | Convert to float: `float(portfolio.realized_pnl)` | 2026-01-15 |
| Passing Decimal to `db.update_account()` | Convert all values to float first | 2026-01-15 |

### Connection & Socket Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Using `socket.connect_ex()` for port check | Use `lsof -nP -iTCP:PORT -sTCP:LISTEN` | 2025-12-06 |
| Not checking for zombies before connect | Check CLOSE_WAIT connections first | 2025-11-24 |
| Reading subprocess stdout too early | Wait for `isConnected()` polling loop | 2025-11-24 |

### Async/Await Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| `run_in_executor` with timeout for stdin | Use dedicated reader thread with queue | 2025-12-24 |
| Cancelling futures with blocking threads | Thread continues even after cancel | 2025-12-24 |

### Database Performance Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Querying DB per-item in loop (N+1 queries) | Batch fetch all data once, then loop over dict | 2026-01-26 |
| `/api/positions` made 198 DB queries | Use `market_price` from positions table + batch signals | 2026-01-26 |
| No caching on frequently-hit API endpoints | Add 2-3 second cache to avoid DB contention | 2026-01-26 |

### Config Attribute Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| `self.cfg.portfolio.initial_capital` | Use `self.cfg.default_cash` - no portfolio.initial_capital | 2026-01-26 |

### Trading Logic Errors
| Mistake | Correct Approach | Date |
|---------|-----------------|------|
| Float precision in position sizing | Use `Decimal` for all financial math | 2025-09-28 |
| Hardcoding market hours | Use `MarketHours` class | 2025-12-29 |
| Assuming fixed % profit on sells | Track cost basis with position tracker (FIFO) | 2026-01-06 |
| P&L dashboard using float | Use `Decimal` for all P&L calculations | 2026-01-06 |
| Arbitrarily expanding symbol list | Let AI discover from news, don't add random stocks | 2026-01-14 |
| "Already have position" = bug | This is CORRECT behavior - prevents duplicate buys | 2026-01-14 |
| Missing dynamic market holidays | Use `_is_market_holiday()` - includes MLK, Presidents, etc. | 2026-01-15 |
| `db.update_position(qty)` uses order qty | Use `self.positions[symbol].quantity` (accumulated) | 2026-01-16 |
| `self.positions` not synced to Portfolio | Must sync DB positions to `self.portfolio.positions` on startup | 2026-01-26 |
| Portfolio shows only cash, no positions | `portfolio.equity()` uses its own positions dict - sync it! | 2026-01-26 |
| Parallel BUY race condition - duplicate buys | Use `_pending_orders` set with lock before `symbol not in positions` check | 2026-01-26 |
| API `get_recent_trades(limit=1000)` misses old trades | Use `limit=5000` to ensure ALL trades included in P&L calc | 2026-01-26 |
| SELL trades with NULL pnl column | Update NULL pnls with estimated value from avg buy price | 2026-01-26 |
| P&L API recalculating instead of using stored values | Use stored `pnl` column from trades table, not FIFO recalc | 2026-01-26 |

---

## Verification Checklist

**Before any PR, verify:**
- [ ] `python3 -m pytest tests/` passes
- [ ] `python3 -m black --check .` passes
- [ ] `python3 -m flake8 .` passes
- [ ] `./scripts/run_bugbot.sh` finds no critical issues
- [ ] `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT` shows no zombies
- [ ] Manual test of affected functionality

---

## Parallel Claude Workflow (5 Terminal Tabs)

See `docs/PARALLEL_CLAUDE_SETUP.md` for full setup guide.

### Terminal Layout

| Tab | Name | Purpose | Key Command |
|-----|------|---------|-------------|
| 1 | MAIN | Primary development | (work happens here) |
| 2 | TEST | Continuous testing | `/test-and-commit` |
| 3 | REVIEW | Code review | `/review` |
| 4 | DOCS | Research/documentation | (research) |
| 5 | HOTFIX | Quick fixes | (emergency fixes) |

### Standard Workflow

```
Tab 1 (MAIN):   Implement changes
Tab 3 (REVIEW): /review         → 6 subagents check code
Tab 1 (MAIN):   Fix issues found
Tab 2 (TEST):   /test-and-commit → Tests pass? → Commit
User:           git push
```

### Slash Commands

| Command | What It Does | Commits? |
|---------|--------------|----------|
| `/review` | 6 parallel subagents review for bugs, security, style | NO |
| `/test-and-commit` | Run pytest, fix failures, then commit | YES |
| `/commit` | Quick commit with proper message format | YES |
| `/pr` | Full PR workflow (tests + lint + PR) | YES |
| `/verify-trading` | Check Gateway, zombies, risk params | NO |
| `/oncall-debug` | Systematic production debugging | NO |
| `/code-simplifier` | Review and simplify recent code | NO |

### CRITICAL: DO NOT Auto-Commit

**After completing work, Claude must:**
1. Report what was changed
2. Say: "Run `/review` in REVIEW tab, then `/test-and-commit` in TEST tab"
3. **STOP** - wait for user to run the slash commands

**Claude CAN commit only when:**
- User runs `/commit` or `/test-and-commit`
- User explicitly says "commit" or "commit and push"
- User runs `/pr`

### Handoff Between Instances

When one Claude instance needs to share context with another:
```bash
# Source Claude writes:
handoff/HANDOFF_<date>_<topic>.md

# Target Claude reads:
Read handoff/LATEST_HANDOFF.md
```

---

## Adding New Mistakes

When Claude makes an error:

1. **Identify the root cause**
2. **Add to table above** with:
   - What went wrong
   - Correct approach
   - Date discovered
3. **Commit the update** so all future sessions learn

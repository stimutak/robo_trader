# RoboTrader Project Guidelines

## Project Phase Plan
**IMPORTANT:** The authoritative phase plan is in `IMPLEMENTATION_PLAN.md`. This is the ML-focused 4-phase plan over 16 weeks:
- Phase 1: Foundation & Quick Wins (Tasks F1-F5) - COMPLETE ✅
- Phase 2: ML Infrastructure & Backtesting (Tasks M1-M5) - COMPLETE ✅
- Phase 3: Advanced Strategy Development (Tasks S1-S5) - COMPLETE ✅
- Phase 4: Production Hardening & Deployment (Tasks P1-P6)

**Current Status:** Phase 3 COMPLETE ✅ - Phase 4 IN PROGRESS (P1-P2 complete, 33%)

**Note:** The older 9-phase plan in `archived_plans/PROJECT_PLAN_9PHASE.md` is deprecated and should NOT be used. Any references to "Phase 5", "Phase 6" etc. from older commits refer to the old plan and should be ignored.

## ⚠️ CRITICAL: TWS/IBKR Gateway Management

**NEVER KILL TWS OR IBKR GATEWAY PROCESSES**

TWS (Trader Workstation) and IBKR Gateway require manual login with credentials and 2FA. They CANNOT be automatically restarted.

**Allowed:**
- ✅ Kill Python processes: `runner_async`, `app.py`, `websocket_server`
- ✅ Restart our application services
- ✅ Check TWS connection status
- ✅ Diagnose TWS health

**FORBIDDEN:**
- ❌ `pkill -f "tws"` or any TWS process killing
- ❌ `pkill -f "ibgateway"` or Gateway process killing
- ❌ Killing any Java processes related to IBKR
- ❌ Automatic TWS restart attempts

**If TWS needs restart:** User must do it manually with login credentials.

**Safe Process Kill Command (Python only):**
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

## Key Project Files
- `IMPLEMENTATION_PLAN.md` - The active project roadmap
- `handoff/LATEST_HANDOFF.md` - Latest session handoff document
- `robo_trader/runner_async.py` - Main trading system with async parallel processing
- `app.py` - Dashboard with monitoring interface
- `robo_trader/websocket_server.py` - WebSocket server for real-time updates
- `robo_trader/features/` - Feature engineering pipeline (Phase 2 - COMPLETE)
- `robo_trader/ml/` - ML model training & selection (Phase 2 - COMPLETE)
- `robo_trader/analytics/` - Performance analytics (Phase 2 - COMPLETE)
- `robo_trader/backtesting/` - Walk-forward backtesting (Phase 2 - COMPLETE)

## IMPORTANT: Python Command on macOS
**ALWAYS USE `python3` NOT `python` - THIS SYSTEM USES macOS WITH NO `python` COMMAND**

## Testing Commands

**CRITICAL (2025-09-25): ALWAYS USE VIRTUAL ENVIRONMENT**
After macOS upgrades, Python paths reset. You MUST activate `.venv` first!

# KILL ALL PROCESSES (run this first to prevent duplicates)
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
```

# START SYSTEM (clean start with .venv)
```bash
# ACTIVATE VIRTUAL ENVIRONMENT (REQUIRED!)
cd /Users/oliver/robo_trader
source .venv/bin/activate

# Start WebSocket server (REQUIRED FIRST)
python3 -m robo_trader.websocket_server &

# Start trading runner with logging
export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF,QQQ,QLD,BBIO,IMRX,CRGY &

# Start dashboard on port 5555 (ALWAYS USE PORT 5555)
export DASH_PORT=5555
python3 app.py &
```

# If missing dependencies (ib_async, pandas, etc.)
```bash
source .venv/bin/activate
pip install ib_async pandas
```

# RESTART DASHBOARD ONLY (when code changes)
```bash
pkill -9 -f "app.py" && sleep 2 && export DASH_PORT=5555 && python3 app.py &

# Run trading system
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF

# Check market hours
python3 test_market_hours.py

# Run tests
pytest

# Test ML pipeline (Phase 2)
python3 test_ml_pipeline.py

# Test model training
python3 test_m3_complete.py

# Test performance analytics
python3 test_m4_performance.py

# Test critical safety features (added 2025-09-27)
python3 test_safety_features.py
```

## Current Issues Status
1. ✅ WebSocket connection handler signature error - FIXED
2. ✅ JSON serialization error with ServerConnection object - FIXED
3. ✅ Phase 1 F2: Upgrade config to Pydantic - COMPLETED
4. ✅ WebSocket stability - Fixed with client/server separation
5. ✅ TWS API Connection Issues - RESOLVED with subprocess approach

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

## Major Fixes Completed (2025-09-23)

### TWS API Connection Resolution ✅
**Problem:** Async context (`patchAsyncio()`) caused TWS API handshake timeouts and stuck connections
**Solution:** Implemented subprocess-based IBKR operations for complete async isolation

**Key Changes:**
- Created `SyncIBKRWrapper` class for thread-based operations
- Implemented subprocess approach for complete process isolation
- Fixed connection pooling complexity (removed, simplified to direct connections)
- Enhanced client ID management (unique timestamp + PID based IDs)
- Comprehensive error handling and cleanup

**Files Modified:**
- `robo_trader/clients/async_ibkr_client.py` - Simplified connection architecture
- `robo_trader/clients/sync_ibkr_wrapper.py` - New subprocess-based wrapper
- `robo_trader/runner_async.py` - Updated to use new client approach

### Library Migration Notes (2025-09-27)
- **MIGRATION COMPLETE:** Successfully migrated from `ib_insync` to `ib_async` v2.0.1
- ib_insync author passed away early 2024, library archived March 2024
- ib_async is the community-maintained fork, drop-in replacement
- All imports updated: `from ib_insync` → `from ib_async`
- Old ib_insync library has been uninstalled
- System tested and running successfully with ib_async

### TWS Connection Requirements
- TWS must be restarted periodically to clear stuck connections
- Check TWS: File → Global Configuration → API → Settings
- Set Master API client ID = 0, Enable Socket Clients, Add 127.0.0.1 to Trusted IPs
- Monitor for CLOSE_WAIT connections: `netstat -an | grep 7497`
- Restart TWS if stuck connections accumulate

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
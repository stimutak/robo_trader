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
- ❌ **NEVER USE `lsof -ti:7497 | xargs kill` - THIS KILLS TWS!**

**If TWS needs restart:** User must do it manually with login credentials.

**CRITICAL - ZOMBIE CONNECTION HANDLING:**
- Zombie CLOSE_WAIT connections ARE HARMFUL - they cause connection timeouts
- They must be killed to restore connectivity
- **DO NOT KILL TWS - USE SAFE ZOMBIE KILL COMMAND:**
```bash
# SAFE - Kills ONLY zombies, NOT TWS
lsof -ti tcp:7497 -sTCP:CLOSE_WAIT | xargs kill -9

# DANGEROUS - Kills EVERYTHING including TWS (NEVER USE)
lsof -ti:7497 | xargs kill
```
- The system has `kill_tws_zombie_connections()` in `robust_connection.py:161`
- Zombies accumulate from failed handshakes and prevent new connections
- See commits bd87fe5, f55015c for zombie connection bug fixes

**CRITICAL - GATEWAY API CONFIGURATION (2025-11-10):**
- **NEVER** suggest checking or enabling "ActiveX and Socket Clients" in Gateway configuration
- This setting is **permanently enabled** in IB Gateway and **cannot be disabled**
- Gateway is always listening on the configured API port (4002 for paper, 4001 for live)
- If API connections fail, the issue is NOT this setting - look elsewhere
- Common actual causes: IBKR account API permissions, Gateway version issues, library incompatibility

**CRITICAL - API DISCONNECT SAFEGUARD (2025-11-02):**
- `ib.disconnect()` now defaults to a no-op to protect the Gateway API layer.
- Use the new helper `robo_trader.utils.ibkr_safe.safe_disconnect()` when you absolutely must disconnect.
- Override the guard only by exporting `IBKR_FORCE_DISCONNECT=1` for isolated tests.
- If you see `Gateway API layer is unresponsive. Manual restart required.`: stop Python processes and restart IB Gateway (full exit + 2FA login), then rerun `./START_TRADER.sh`.

**CRITICAL - DISCONNECT ZOMBIE FIX (2025-11-20):**
- **FIXED:** `START_TRADER.sh` and `force_gateway_reconnect.sh` now use `safe_disconnect()` with `IBKR_FORCE_DISCONNECT=1`
- Previously, test scripts called `ib.disconnect()` which was a no-op, leaving orphaned connections that became Gateway zombies
- These zombies blocked subsequent API handshakes, requiring Gateway restart
- Now test scripts properly disconnect using `safe_disconnect()` with force flag enabled
- This prevents zombie accumulation during startup connectivity tests
- See commits for zombie connection bug investigation

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

## Starting the Trading System

**RECOMMENDED: Use the automated startup script (2025-10-23)**

The `START_TRADER.sh` script provides clean startup with automatic zombie cleanup and Gateway connectivity testing.

### Default Symbols (from user_settings.json)
```
AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF
```

### Quick Start (Recommended)
```bash
# Start with default symbols (from user_settings.json)
./START_TRADER.sh

# Start with custom symbols
./START_TRADER.sh "AAPL,NVDA,TSLA,QQQ"
```

**What the script does:**
1. ✅ Kills existing Python trader processes
2. ✅ Cleans up zombie CLOSE_WAIT connections
3. ✅ Tests Gateway connectivity (aborts if Gateway not responding)
4. ✅ Starts WebSocket server
5. ✅ Starts trading system with logging
6. ✅ Monitors startup for 10 seconds

**If startup fails:** The script will show clear error messages and diagnostic commands.

### Manual Startup (Advanced)

**CRITICAL (2025-09-25): ALWAYS USE VIRTUAL ENVIRONMENT**
After macOS upgrades, Python paths reset. You MUST activate `.venv` first!

```bash
# KILL ALL PROCESSES (run this first to prevent duplicates)
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"

# ACTIVATE VIRTUAL ENVIRONMENT (REQUIRED!)
cd /Users/oliver/robo_trader
source .venv/bin/activate

# Start WebSocket server (REQUIRED FIRST)
python3 -m robo_trader.websocket_server &

# Start trading runner with logging
export LOG_FILE=/Users/oliver/robo_trader/robo_trader.log
python3 -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF &

# Start dashboard on port 5555 (ALWAYS USE PORT 5555)
export DASH_PORT=5555
python3 app.py &
```

### Diagnostic Commands

```bash
# Test Gateway connectivity
./force_gateway_reconnect.sh

# Full diagnostics (checks Gateway, zombies, config)
python3 diagnose_gateway_api.py

# Check for zombie connections
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT

# View logs
tail -f robo_trader.log
```

### Testing Commands

```bash
# If missing dependencies (ib_async, pandas, etc.)
source .venv/bin/activate
pip install ib_async pandas

# RESTART DASHBOARD ONLY (when code changes)
pkill -9 -f "app.py" && sleep 2 && export DASH_PORT=5555 && python3 app.py &

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
6. ✅ Zombie connection cleanup - AUTOMATED at startup (2025-10-23)
7. ✅ Gateway connectivity testing - BUILT INTO startup script (2025-10-23)
8. ✅ Subprocess worker connection failure - RESOLVED (2025-11-24)

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

**Results:**
- **Connection Time**: 2-3 seconds (vs previous ~163ms failure)
- **Success Rate**: 100% when Gateway clean (vs 0% before)
- **Error Detection**: Immediate zombie detection vs 30s timeout
- **User Experience**: Clear error messages with specific restart instructions

**Files Modified:**
- `robo_trader/clients/ibkr_subprocess_worker.py` - Synchronization fix
- `robo_trader/clients/subprocess_ibkr_client.py` - Zombie prevention + debug capture
- `test_subprocess_connection_fix.py` - Comprehensive test suite

**Documentation:** See `docs/SUBPROCESS_WORKER_CONNECTION_FIX.md` for complete technical details.

### Gateway Connection Management (2025-10-23)

**Automated Zombie Cleanup:**
- `runner_async.py` now automatically cleans up zombie connections at startup
- `START_TRADER.sh` tests Gateway connectivity before starting trader
- Zombie connections are detected and killed (Python processes only)
- Gateway-owned zombies are logged but cannot be killed (require Gateway restart)

**Gateway Connectivity Testing:**
- `force_gateway_reconnect.sh` - Test if Gateway accepts API connections
- `diagnose_gateway_api.py` - Comprehensive diagnostics (Gateway, port, TCP, API, zombies)
- Startup script aborts with clear error if Gateway not responding

**Gateway API Requirements:**
- Gateway must have API socket clients enabled
- Check Gateway: File → Global Configuration → API → Settings
- ☑️ Enable ActiveX and Socket Clients (CRITICAL)
- Add 127.0.0.1 to Trusted IPs
- Socket port: 4002 (paper) or 4001 (live)
- Master API client ID: 0 (or blank)

**Troubleshooting:**
- If API handshake times out: Check Gateway API settings (above)
- If zombies accumulate: Use `START_TRADER.sh` for automatic cleanup
- If Gateway not responding: Restart Gateway (requires 2FA login)
- Monitor connections: `netstat -an | grep 4002`

### TWS Readonly Connection (2025-10-05) ✅
**Important: System uses READONLY mode for TWS connections**
- `readonly=True` in all IBKR connections
- No order placement through TWS API (PaperExecutor handles orders)
- Only data access: historical bars, positions, account info
- **Benefit**: No TWS security dialog popups (read-only doesn't require approval)
- **Client ID Strategy**:
  - First attempt: Fixed client_id (TWS remembers, no dialog)
  - Retry attempts: Random client_id (zombie fix, prevent confusion)

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
- it is not the socket and activex param in gateway config that is at fault as it is perminantly on and cannot be disabled in gateway.
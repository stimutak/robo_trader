# RoboTrader Project Guidelines

## Project Phase Plan
**IMPORTANT:** The authoritative phase plan is in `IMPLEMENTATION_PLAN.md`. This is the ML-focused 4-phase plan over 16 weeks:
- Phase 1: Foundation & Quick Wins (Tasks F1-F5) - COMPLETE ✅
- Phase 2: ML Infrastructure & Backtesting (Tasks M1-M5) - COMPLETE ✅
- Phase 3: Advanced Strategy Development (Tasks S1-S5) - IN PROGRESS 🚧
- Phase 4: Production Hardening & Deployment (Tasks P1-P6)

**Current Status:** Phase 3 IN PROGRESS - S1-S4 COMPLETE ✅ (80%), S5 remaining.

**Note:** The older 9-phase plan in `archived_plans/PROJECT_PLAN_9PHASE.md` is deprecated and should NOT be used. Any references to "Phase 5", "Phase 6" etc. from older commits refer to the old plan and should be ignored.

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

## Testing Commands
```bash
# Run trading system
python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF

# Run dashboard
export DASH_PORT=5555
python app.py

# Check market hours
python test_market_hours.py

# Run tests
pytest

# Test ML pipeline (Phase 2)
python test_ml_pipeline.py

# Test model training
python test_m3_complete.py

# Test performance analytics
python test_m4_performance.py
```

## Current Issues to Fix
1. ✅ WebSocket connection handler signature error - FIXED
2. ✅ JSON serialization error with ServerConnection object - FIXED  
3. ✅ Phase 1 F2: Upgrade config to Pydantic - COMPLETED
4. ✅ WebSocket stability - Fixed with client/server separation

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
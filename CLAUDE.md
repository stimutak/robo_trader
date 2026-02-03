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

---

## ⚠️ IBKR Gateway Management

### Automated Gateway Management via IBC

The system uses **IBC (IB Controller)** for automated Gateway management. Gateway restarts are **fully automated** when zombie connections are detected.

**How it works:**
- `START_TRADER.sh` automatically starts Gateway via IBC if not running
- Detects zombie CLOSE_WAIT connections that block API handshakes
- Automatically restarts Gateway to clear zombies (up to 3 retries)
- Tests actual API connectivity before proceeding
- Only requires manual 2FA on your phone when Gateway starts

**IBC Configuration:**
- Config file: `config/ibc/config.ini` (gitignored - contains credentials)
- Template: `config/ibc/config.ini.template`

### Gateway Commands

```bash
# THE ONLY WAY TO START THE TRADING SYSTEM:
./START_TRADER.sh

# Debugging/diagnostic commands:
python3 scripts/gateway_manager.py status   # Check Gateway status
python3 scripts/gateway_manager.py restart  # Force restart Gateway
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT        # Check for zombies
tail -f config/ibc/logs/*.txt               # View Gateway logs
```

### Kill Processes (Safe)
```bash
pkill -9 -f "runner_async" && pkill -9 -f "app.py" && pkill -9 -f "websocket_server"
pkill -f "IB Gateway"  # Triggers auto-restart on next START_TRADER.sh
```

### Gateway API Notes
- **Port 4002**: Paper trading API
- **Port 4001**: Live trading API
- System uses **readonly** connections (no order placement via API)

---

## Watchdog Auto-Restarter

- Monitors log file modification time every 60 seconds
- If no log activity for 5+ minutes during market hours → auto-restart
- Respects `ENABLE_EXTENDED_HOURS` setting (monitors 4 AM - 8 PM if enabled)
- Runs as macOS launchd service (survives reboot)

### Watchdog Commands
```bash
launchctl list | grep robotrader                                    # Check if running
tail -f watchdog.log                                                # View log
launchctl unload ~/Library/LaunchAgents/com.robotrader.watchdog.plist  # Stop
launchctl load ~/Library/LaunchAgents/com.robotrader.watchdog.plist    # Start
```

---

## Mobile App & Parallel Development

| Location | Branch | Purpose |
|----------|--------|---------|
| `/Users/oliver/robo_trader` | `main` | Backend, API, Web Dashboard |
| `/Users/oliver/robo_trader-mobile` | `feature/mobile-app` | React Native Mobile App |

**CRITICAL:** Never edit the same file in both branches simultaneously.

```bash
# Sync to mobile worktree after backend changes:
cd /Users/oliver/robo_trader-mobile && git fetch origin main && git merge origin/main
```

---

## Key Project Files
- `IMPLEMENTATION_PLAN.md` - The active project roadmap
- `handoff/LATEST_HANDOFF.md` - Latest session handoff document
- `robo_trader/runner_async.py` - Main trading system with async parallel processing
- `robo_trader/database_async.py` - Async database with equity_history table
- `robo_trader/stop_loss_monitor.py` - Stop-loss protection (recreated on startup)
- `app.py` - Dashboard with comprehensive professional overview
- `robo_trader/websocket_server.py` - WebSocket server for real-time updates
- `scripts/gateway_manager.py` - Cross-platform Gateway management
- `config/ibc/config.ini` - IBC credentials (gitignored)

## IMPORTANT: Python Command on macOS
**ALWAYS USE `python3` NOT `python` - THIS SYSTEM USES macOS WITH NO `python` COMMAND**

---

## Starting the Trading System

```bash
./START_TRADER.sh
# Or with custom symbols:
./START_TRADER.sh "AAPL,NVDA,TSLA"
```

**⚠️ IMPORTANT: Always use `./START_TRADER.sh` - NEVER start components manually!**

Do NOT run these directly:
- ❌ `python3 runner_async.py`
- ❌ `python3 app.py`
- ❌ `scripts/start_gateway.sh`

### Diagnostic Commands
```bash
python3 scripts/gateway_manager.py status    # Gateway status
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT         # Check zombies
tail -f robo_trader.log                       # Trader logs
```

### Testing Commands
```bash
pytest                           # Run tests
python3 test_ml_pipeline.py      # Test ML pipeline
python3 test_safety_features.py  # Test safety features
```

---

## Current Issues Status
1. ✅ WebSocket stability - Fixed with client/server separation
2. ✅ TWS API Connection Issues - RESOLVED with subprocess approach
3. ✅ Zombie connection cleanup - AUTOMATED with Gateway restart
4. ✅ Socket Zombie Creation Bug - FIXED (use lsof not socket.connect_ex)
5. ✅ Subprocess Pipe Blocking - FIXED (dedicated stdin reader thread)
6. ✅ Near Real-Time Trading - IMPLEMENTED (15-second cycles)
7. ✅ Decimal/Float Type Mismatch - FIXED (use price_float)
8. ✅ Extended Hours Trading - ENABLED (`ENABLE_EXTENDED_HOURS=true`)
9. ✅ Dashboard Overview Redesign - IMPLEMENTED
10. ✅ Equity History Tracking - IMPLEMENTED (`equity_history` table)
11. ✅ Duplicate BUY Race Condition - FIXED (3-layer protection)
12. ✅ ML/MTF Disagreement Threshold - FIXED (adaptive threshold)
13. ✅ Stop-losses on restart - FIXED (recreated from DB positions)

---

## AI-Driven Symbol Discovery

**The system discovers new trading opportunities from news - DO NOT manually expand symbol lists.**

### How It Works
1. **Base Symbols** in `.env` `SYMBOLS=` - Your watchlist
2. **Existing Positions** - Automatically added for SELL signal monitoring
3. **AI Discovery** - Scans 50+ news headlines per cycle, finds new opportunities

### Important Behavior
- **"Already have long position"** = CORRECT behavior, not a bug
- System prevents duplicate positions in same stock
- AI confidence threshold: 50%+ required

### DO NOT
- Arbitrarily add stocks to SYMBOLS list
- Expand symbol list without AI/news basis
- Confuse "no new buys" with "system broken" (may already own those stocks)

---

## Key Configuration (.env)

| Setting | Purpose |
|---------|---------|
| `USE_TRAILING_STOP=true` | **Trailing stops enabled** - lets winners run! |
| `TRAILING_STOP_PERCENT=5.0` | Trailing stop at 5% below high water mark |
| `STOP_LOSS_PERCENT=2.0` | Fixed stop-loss (only used if trailing disabled) |
| `ENABLE_EXTENDED_HOURS=true` | Trade 4AM-8PM ET (pre/after market) |
| `MAX_OPEN_POSITIONS` | Limit concurrent positions |
| `SYMBOLS=` | Base watchlist (AI discovers more) |

### Trailing Stops (Added 2026-02-03)

**Problem solved:** System was selling at loss with avg loss $718 vs avg win $444. Losses were 60% bigger than wins.

**Solution:** Trailing stops that follow price UP:
- Initial stop at 5% below entry price
- As price rises, stop rises with it (ratchets up)
- Never moves down - only up
- Locks in profits while letting winners run

**Example:**
```
Buy at $100 → initial stop at $95 (5% below)
Price rises to $120 → stop moves to $114 (5% below $120)
Price drops to $114 → SELL triggered at $114 (locked in $14 profit!)
```

**Configuration:**
```bash
USE_TRAILING_STOP=true          # Enable trailing (recommended)
TRAILING_STOP_PERCENT=5.0       # 5% trail (adjust based on volatility)
```

**Disable trailing (use fixed stops):**
```bash
USE_TRAILING_STOP=false
STOP_LOSS_PERCENT=2.0           # Fixed 2% stop
```

---

## Critical Safety Features
- **Order Management** (`order_manager.py`) - Full lifecycle tracking with retry logic
- **Data Validation** (`data_validator.py`) - Market data quality checks
- **Circuit Breaker** (`circuit_breaker.py`) - Fault tolerance system
- **Stop-Loss Monitor** - Recreated on startup for ALL existing positions
- **Decimal Precision** - Portfolio uses `Decimal` for all financial math

---

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
| Arbitrarily expanding symbol list | Let AI discover from news, don't add random stocks | 2026-01-14 |
| "Already have position" = bug | This is CORRECT behavior - prevents duplicate buys | 2026-01-14 |
| Missing dynamic market holidays | Use `_is_market_holiday()` - includes MLK, Presidents, etc. | 2026-01-15 |
| `db.update_position(qty)` uses order qty | Use `self.positions[symbol].quantity` (accumulated) | 2026-01-16 |
| `self.positions` not synced to Portfolio | Must sync DB positions to `self.portfolio.positions` on startup | 2026-01-26 |
| Portfolio shows only cash, no positions | `portfolio.equity()` uses its own positions dict - sync it! | 2026-01-26 |
| Parallel BUY race condition - duplicate buys | Use 3-layer protection: cycle set + pending lock + DB check | 2026-01-27 |
| Pairs trading records trades but not positions | After pairs BUY: update `self.positions` AND `db.update_position()` | 2026-02-03 |
| Pairs trading `has_recent_buy` uses 120s | Use 600s (10 min) to catch trades across multiple cycles | 2026-02-03 |
| Pairs trading missing `portfolio.update_fill()` | Call `await self.portfolio.update_fill()` after pairs BUY | 2026-02-03 |
| Pairs trading missing stop-loss creation | Add stop-loss via `stop_loss_monitor.add_stop_loss()` after pairs BUY | 2026-02-03 |
| `db.get_positions()` inside pairs loop (N+1) | Fetch once before loop into `db_positions_list`, reuse inside | 2026-02-03 |
| In-memory duplicate checks reset each cycle | Add DATABASE check inside lock - `await self.db.get_positions()` | 2026-01-27 |
| Market shows "closed" at 4:00 PM when user wants to trade | Set `ENABLE_EXTENDED_HOURS=true` in .env | 2026-01-27 |
| Using `is_market_open()` for trading checks | Use `is_trading_allowed()` which includes extended hours | 2026-01-27 |
| Fixed 0.8 confidence threshold for ML/MTF disagreement | Use adaptive threshold: `model_test_score + margin`, lower in range-bound | 2026-01-27 |
| Ignoring market regime in signal resolution | Range-bound regime → MTF trend signals are noise, trust ML more | 2026-01-27 |
| Positions table not accumulating quantities | Check `db.update_position()` receives accumulated qty, not order qty | 2026-01-27 |
| Stale positions → wrong equity calculation | Rebuild positions from trades if mismatch detected | 2026-01-27 |
| Pairs trading bypasses duplicate protection | Add `has_recent_buy_trade()` check before pairs BUY orders | 2026-01-28 |
| Pairs trading position check `> 100` instead of `> 0` | Check `quantity > 0` to block any existing position | 2026-01-28 |
| Fresh AsyncRunner each cycle resets in-memory protection | Use DB-level checks (`has_recent_buy_trade`) that persist | 2026-01-28 |
| DB column `action` vs `side` mismatch | `trading_data.db` uses `side`, not `action` - check schema first | 2026-01-29 |
| Pairs trading bypasses MAX_OPEN_POSITIONS | Add position count check before opening pairs trades | 2026-01-29 |
| Missing parameter validation on new DB methods | Add symbol validation + seconds bounds check (1-86400) | 2026-01-29 |
| Stop-losses not created for existing positions on restart | `load_existing_positions()` must call `stop_loss_monitor.add_stop_loss()` for each loaded position | 2026-02-03 |
| In-memory stop-loss orders lost on restart | Stop-losses are ephemeral - ALWAYS recreate on startup from DB positions | 2026-02-03 |

---

## Verification Checklist

**Before any PR, verify:**
- [ ] `python3 -m pytest tests/` passes
- [ ] `python3 -m black --check .` passes
- [ ] `python3 -m flake8 .` passes
- [ ] `lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT` shows no zombies
- [ ] Manual test of affected functionality

---

## Parallel Claude Workflow (5 Terminal Tabs)

See `docs/PARALLEL_CLAUDE_SETUP.md` for full setup guide.

### Terminal Layout

| Tab | Name | Purpose |
|-----|------|---------|
| 1 | MAIN | Primary development |
| 2 | TEST | Continuous testing (`/test-and-commit`) |
| 3 | REVIEW | Code review (`/review`) |
| 4 | DOCS | Research/documentation |
| 5 | HOTFIX | Quick fixes |

### Slash Commands

| Command | What It Does | Commits? |
|---------|--------------|----------|
| `/review` | 6 parallel subagents review for bugs, security, style | NO |
| `/two-phase-review` | Review + challenger phase (filters false positives) | NO |
| `/verify` | Run verification loop (tests, lint, trading checks) | NO |
| `/test-and-commit` | Run pytest, fix failures, then commit | YES |
| `/commit` | Quick commit with proper message format | YES |
| `/pr` | Full PR workflow (tests + lint + PR) | YES |
| `/verify-trading` | Check Gateway, zombies, risk params | NO |
| `/retrospective` | Extract session learnings, update CLAUDE.md | YES |

### When to Suggest Commands

| Trigger | Suggest |
|---------|---------|
| After implementing a feature | `/verify` then `/review` |
| After fixing a bug | `/verify` |
| Before committing significant changes | `/review` or `/two-phase-review` |
| User says "commit" or "done" | `/test-and-commit` |
| Session with errors/fixes | `/retrospective` |

### CRITICAL: DO NOT Auto-Commit

**After completing work, Claude must:**
1. Report what was changed
2. Suggest verification: "Run `/verify` to validate"
3. Suggest review: "Then run `/review`"
4. Suggest commit: "Finally `/test-and-commit` when ready"
5. **STOP** - wait for user to run the slash commands

**Claude CAN commit only when:**
- User runs `/commit` or `/test-and-commit`
- User explicitly says "commit" or "commit and push"
- User runs `/pr`

### Handoff Between Instances

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
2. **Add to table above** with date
3. **Commit the update** so all future sessions learn

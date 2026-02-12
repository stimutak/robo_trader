# Multi-Portfolio / Multiuser Implementation Plan

## Overview

Transform the single-user RoboTrader into a multi-portfolio system where multiple independent portfolios can trade simultaneously through a shared IBKR Gateway connection. Start with 2 portfolios, architect for N.

## Design Principles

1. **Shared broker, isolated portfolios** - All portfolios use the same IBKR Gateway connection (port 4002) but have fully independent positions, trades, cash, risk parameters, and equity tracking
2. **Single database, partitioned by portfolio_id** - One `trading_data.db` with a `portfolio_id` column on all user-scoped tables (not separate DB files)
3. **Single process** - One runner process manages all portfolios sequentially within each cycle (no need for multiple OS processes)
4. **Backward compatible** - Existing data migrates seamlessly as `portfolio_id = 'default'`
5. **Minimal surface area** - Change only what's needed; market data, features, ticks remain global (shared across portfolios)

## Architecture

```
                    ┌─────────────────────────┐
                    │    IBKR Gateway :4002    │
                    └────────────┬────────────┘
                                 │ (shared connection)
                    ┌────────────┴────────────┐
                    │      AsyncRunner         │
                    │  (orchestrates cycles)   │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────┴──────┐  ┌───────┴────────┐  ┌──────┴─────────┐
    │  Portfolio "A"  │  │ Portfolio "B"  │  │ Portfolio "N"  │
    │  ────────────── │  │ ────────────── │  │ ──────────────  │
    │  Cash: $50,000  │  │ Cash: $50,000  │  │ Cash: ...      │
    │  Symbols: AAPL  │  │ Symbols: TSLA  │  │ Symbols: ...   │
    │  Risk: conserv. │  │ Risk: aggress. │  │ Risk: ...      │
    │  Positions: own │  │ Positions: own │  │ Positions: own │
    │  Stop-losses    │  │ Stop-losses    │  │ Stop-losses    │
    └────────────────┘  └────────────────┘  └────────────────┘
              │                  │                   │
              └──────────────────┼──────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │    trading_data.db       │
                    │  (partitioned by         │
                    │   portfolio_id)          │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────┴──────┐  ┌───────┴────────┐  ┌──────┴─────────┐
    │  Dashboard :5000│  │ WebSocket :8765│  │ Mobile App     │
    │  Portfolio picker│  │ Per-portfolio  │  │ Portfolio picker│
    │  in header       │  │ channels      │  │                │
    └────────────────┘  └────────────────┘  └────────────────┘
```

## What Changes vs What Stays the Same

### STAYS THE SAME (Global / Shared)
- IBKR Gateway connection (single port 4002)
- Market data tables: `market_data`, `ticks`, `features` (prices are prices)
- ML models and feature store (models are symbol-level, not portfolio-level)
- AI Analyst / news fetcher (news is global)
- Gateway manager, IBC, watchdog
- Logger infrastructure

### CHANGES (Per-Portfolio Isolation)
- **Database schema**: Add `portfolio_id` to `positions`, `trades`, `account`, `equity_history`, `signals`
- **New table**: `portfolios` (configuration for each portfolio)
- **AsyncRunner**: Loop over portfolios each cycle instead of single portfolio
- **Portfolio class**: Accept and carry `portfolio_id`
- **Config**: Support per-portfolio overrides (symbols, cash, risk params)
- **Dashboard API**: All endpoints accept `?portfolio_id=` query param
- **WebSocket**: Include `portfolio_id` in messages; clients subscribe to specific portfolio
- **Stop-loss monitor**: Tag stops with `portfolio_id`
- **Order manager**: Tag orders with `portfolio_id`
- **Start script**: Load portfolio configs on startup

---

## Phase 1: Database Schema & Migration (Foundation)

### 1.1 New `portfolios` table

```sql
CREATE TABLE IF NOT EXISTS portfolios (
    id TEXT PRIMARY KEY,           -- e.g., 'aggressive', 'conservative', 'default'
    name TEXT NOT NULL,            -- Display name: "Aggressive Growth"
    starting_cash REAL NOT NULL DEFAULT 100000,
    symbols TEXT NOT NULL,         -- Comma-separated: "AAPL,NVDA,TSLA"
    active INTEGER NOT NULL DEFAULT 1,
    -- Per-portfolio risk overrides (NULL = use global default)
    max_position_pct REAL,
    max_daily_loss_pct REAL,
    max_open_positions INTEGER,
    stop_loss_pct REAL,
    trailing_stop_pct REAL,
    use_trailing_stop INTEGER,
    -- Strategy overrides
    enabled_strategies TEXT,       -- NULL = use global, else comma-separated
    min_confidence REAL,
    -- Metadata
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### 1.2 Add `portfolio_id` to existing tables

Migration adds `portfolio_id TEXT NOT NULL DEFAULT 'default'` to:

| Table | Current unique constraint | New unique constraint |
|-------|--------------------------|----------------------|
| `positions` | `UNIQUE(symbol)` | `UNIQUE(portfolio_id, symbol)` |
| `trades` | none | none (add index on portfolio_id) |
| `account` | `id = 1` hardcoded | `UNIQUE(portfolio_id)` |
| `equity_history` | `UNIQUE(date)` | `UNIQUE(portfolio_id, date)` |
| `signals` | none | none (add index on portfolio_id) |

Tables that stay global (no `portfolio_id`):
- `market_data`, `ticks`, `features` — market data is shared

### 1.3 Migration strategy

Since SQLite doesn't support `ALTER TABLE ADD CONSTRAINT`, use the rename-recreate pattern:

```python
async def migrate_to_multiuser(self):
    """One-time migration to add portfolio_id support."""
    # 1. Check if already migrated
    # 2. Create portfolios table
    # 3. Insert 'default' portfolio from current config
    # 4. For each table needing portfolio_id:
    #    a. CREATE new_table with portfolio_id column
    #    b. INSERT INTO new_table SELECT *, 'default' FROM old_table
    #    c. DROP old_table
    #    d. ALTER TABLE new_table RENAME TO table
    # 5. Update account table: portfolio_id instead of id=1
```

**Backward compatibility**: All existing data gets `portfolio_id = 'default'`. The system works identically to before if only one portfolio exists.

### 1.4 Database method changes

Every method that touches these tables gets an optional `portfolio_id` parameter:

```python
# Before:
async def get_positions(self) -> List[Dict]:

# After:
async def get_positions(self, portfolio_id: str = 'default') -> List[Dict]:
```

Methods affected:
- `update_position()` → add `portfolio_id` param
- `get_positions()` → filter by `portfolio_id`
- `record_trade()` → add `portfolio_id` param
- `get_recent_trades()` → filter by `portfolio_id`
- `has_recent_buy_trade()` → filter by `portfolio_id`
- `has_recent_sell_trade()` → filter by `portfolio_id`
- `update_account()` → use `portfolio_id` instead of `WHERE id = 1`
- `get_account_info()` → filter by `portfolio_id`
- `record_equity_history()` → add `portfolio_id` param
- `get_equity_history()` → filter by `portfolio_id`
- `_calculate_fifo_pnl()` → filter by `portfolio_id`
- `record_signal()` → add `portfolio_id` param

---

## Phase 2: Portfolio Configuration System

### 2.1 Portfolio config in `.env`

```env
# Portfolio definitions (JSON-encoded list)
PORTFOLIOS=[{"id":"aggressive","name":"Aggressive Growth","starting_cash":50000,"symbols":"NVDA,TSLA,AMD,MARA","max_position_pct":0.04,"use_trailing_stop":true,"trailing_stop_pct":0.07},{"id":"conservative","name":"Conservative Income","starting_cash":50000,"symbols":"AAPL,MSFT,JNJ,PG,KO","max_position_pct":0.02,"use_trailing_stop":true,"trailing_stop_pct":0.04}]
```

If `PORTFOLIOS` is not set, auto-create a single `default` portfolio from the existing `SYMBOLS` and `DEFAULT_CASH` env vars (backward compatible).

### 2.2 PortfolioConfig dataclass

```python
@dataclass
class PortfolioConfig:
    id: str
    name: str
    starting_cash: float
    symbols: List[str]
    active: bool = True
    # Risk overrides (None = use global)
    max_position_pct: Optional[float] = None
    max_daily_loss_pct: Optional[float] = None
    max_open_positions: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    use_trailing_stop: Optional[bool] = None
    enabled_strategies: Optional[List[str]] = None
    min_confidence: Optional[float] = None
```

### 2.3 Config loading

`load_config()` gains a `portfolios: List[PortfolioConfig]` field. If no `PORTFOLIOS` env var, creates `[PortfolioConfig(id='default', ...)]` from existing settings.

---

## Phase 3: Runner Multi-Portfolio Loop

### 3.1 Core loop change

Currently:
```python
# Single portfolio per cycle
async def run_cycle():
    runner = AsyncRunner(...)
    await runner.setup()
    await runner.process_symbols(symbols)
    await runner.update_account()
```

After:
```python
async def run_cycle():
    for portfolio_config in active_portfolios:
        runner = AsyncRunner(portfolio_id=portfolio_config.id, ...)
        await runner.setup()  # Loads portfolio-specific positions/state
        await runner.process_symbols(portfolio_config.symbols)
        await runner.update_account()
```

### 3.2 AsyncRunner changes

- Accept `portfolio_id` in constructor
- Pass `portfolio_id` to all DB calls
- `load_existing_positions()` only loads positions for this portfolio
- `self.portfolio = Portfolio(starting_cash, portfolio_id=...)`
- Stop-loss monitor tags stops with portfolio_id
- WebSocket updates include portfolio_id

### 3.3 Position isolation

Critical: Two portfolios CAN hold the same symbol independently.
- Portfolio A: 100 shares AAPL @ $180
- Portfolio B: 50 shares AAPL @ $185

The `positions` table uses `UNIQUE(portfolio_id, symbol)` to allow this.

Duplicate buy protection (`has_recent_buy_trade`) must be scoped to portfolio_id.

---

## Phase 4: Dashboard Multi-Portfolio Support

### 4.1 API changes

All endpoints gain `portfolio_id` query parameter:

```
GET /api/positions?portfolio_id=aggressive
GET /api/pnl?portfolio_id=aggressive
GET /api/status?portfolio_id=aggressive
GET /api/account?portfolio_id=aggressive
GET /api/equity-history?portfolio_id=aggressive
GET /api/trades?portfolio_id=aggressive
```

Default: If `portfolio_id` not specified, return data for `'default'` portfolio.

New endpoint:
```
GET /api/portfolios  → List all portfolios with summary stats
```

### 4.2 Dashboard UI

- Add portfolio selector dropdown in header (next to existing controls)
- All dashboard panels update when portfolio is selected
- "All Portfolios" aggregate view shows combined equity, all positions
- Color-code or badge positions by portfolio

### 4.3 SyncDatabaseReader

The synchronous DB reader used by Flask also needs `portfolio_id` filtering on all queries.

---

## Phase 5: WebSocket Per-Portfolio Updates

### 5.1 Message format change

```json
{
    "type": "trade_update",
    "portfolio_id": "aggressive",
    "data": { "symbol": "NVDA", "side": "BUY", ... }
}
```

### 5.2 Client subscription

Clients send a subscribe message on connect:
```json
{"action": "subscribe", "portfolio_ids": ["aggressive", "conservative"]}
```

Or subscribe to all: `{"action": "subscribe", "portfolio_ids": ["*"]}`

The WebSocket manager routes messages only to subscribed clients.

---

## Phase 6: Stop-Loss & Order Manager Updates

### 6.1 Stop-loss monitor

- `StopLossOrder` gets `portfolio_id` field
- `add_stop_loss(symbol, ..., portfolio_id=...)`
- Stop triggers only affect the correct portfolio
- On restart, loads stops per-portfolio from DB positions

### 6.2 Order manager

- `OrderDetails` gets `portfolio_id` field
- Order callbacks include portfolio_id for routing

---

## Implementation Order (Recommended)

| Step | What | Risk | Effort |
|------|------|------|--------|
| **1** | Database migration + `portfolios` table | LOW - additive, backward compat | Medium |
| **2** | DB method signatures (add `portfolio_id` param) | LOW - default='default' | Medium |
| **3** | PortfolioConfig dataclass + config loading | LOW - new code | Small |
| **4** | AsyncRunner accepts portfolio_id, passes to DB | MEDIUM - core logic | Large |
| **5** | Multi-portfolio loop in `run_continuous()` | MEDIUM - orchestration | Medium |
| **6** | Dashboard API `portfolio_id` support | LOW - additive | Medium |
| **7** | Dashboard UI portfolio selector | LOW - frontend only | Medium |
| **8** | WebSocket per-portfolio routing | LOW - additive | Small |
| **9** | Stop-loss monitor portfolio isolation | MEDIUM - safety critical | Medium |
| **10** | Start script + .env config | LOW - configuration | Small |
| **11** | Testing & validation | - | Medium |

## Risk Mitigation

1. **Data safety**: Migration creates backup before any schema changes
2. **Backward compat**: `portfolio_id='default'` means system works identically if only one portfolio
3. **Incremental**: Each step is independently testable and deployable
4. **Same IBKR account**: No broker-level changes needed; portfolios are virtual subdivisions of the same account
5. **No data deletion**: Migration only adds columns and creates new tables

## Future Extensions (Out of Scope for Now)

- Full user authentication with JWT tokens
- Per-user login and role-based access
- Separate IBKR sub-accounts per user
- Independent processes per user
- User management admin panel
- API key-based access for programmatic users

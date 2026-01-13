# Polygon.io Data Integration Plan

**Branch:** `feature/polygon-data-integration`
**Created:** 2026-01-13
**Status:** Phase 1 Complete - Foundation Built

## Overview

Integrate Polygon.io (now Massive.com) for market data while keeping IBKR Gateway for order execution. This separates concerns and eliminates the zombie connection issues that plague IBKR's market data API.

## Architecture

```
┌──────────────────────────┐     ┌──────────────────────────┐
│      Polygon.io API      │     │     IBKR Gateway API     │
│   (api.polygon.io)       │     │      (localhost:4002)    │
├──────────────────────────┤     ├──────────────────────────┤
│ REST API:                │     │ Order Execution:         │
│ • Historical bars        │     │ • placeOrder()           │
│ • Previous close         │     │ • cancelOrder()          │
│ • Ticker details         │     │ • modifyOrder()          │
│                          │     │                          │
│ WebSocket (Advanced):    │     │ Account Data:            │
│ • Real-time quotes       │     │ • positions()            │
│ • Trade ticks            │     │ • accountSummary()       │
│ • 1-min aggregates       │     │ • portfolio()            │
└────────────┬─────────────┘     └────────────┬─────────────┘
             │                                │
             └──────────────┬─────────────────┘
                            │
               ┌────────────▼────────────┐
               │    DataProviderManager  │
               │  (Abstraction Layer)    │
               └────────────┬────────────┘
                            │
               ┌────────────▼────────────┐
               │     runner_async.py     │
               │    (Trading Logic)      │
               └─────────────────────────┘
```

## Current Status

### Phase 1: Foundation ✅ COMPLETE

- [x] **P1.1** Create `robo_trader/data_providers/` module structure
- [x] **P1.2** Create abstract `DataProvider` interface (`base.py`)
- [x] **P1.3** Implement `PolygonDataProvider` with REST API
- [x] **P1.4** Add config for Polygon API key (`.env`)
- [x] **P1.5** Write test script (`scripts/test_polygon_provider.py`)

**Files Created:**
- `robo_trader/data_providers/__init__.py`
- `robo_trader/data_providers/base.py` - Abstract interface
- `robo_trader/data_providers/polygon_provider.py` - Polygon implementation
- `scripts/test_polygon_provider.py` - Test script

**Test Results (2026-01-13):**
```
✅ Connection to Polygon API
✅ Historical bars (1min, 5min, 1day)
✅ Current price retrieval
✅ Quote data (last, volume, timestamp)
✅ Rate limiting (12.5s between calls on free tier)
⚠️ 429 rate limit hit after ~6 calls (expected on free tier)
```

### Phase 2: Historical Data Integration (TODO)

- [ ] **P2.1** Replace `_fetch_historical_bars()` to use Polygon REST
- [ ] **P2.2** Map Polygon bar format to existing DataFrame format
- [ ] **P2.3** Add caching layer to reduce API calls
- [ ] **P2.4** Test with backtesting framework

### Phase 3: Real-Time Streaming (TODO - Requires Advanced Tier)

- [ ] **P3.1** Implement `PolygonWebSocketClient` for streaming
- [ ] **P3.2** Create price cache updated by WebSocket
- [ ] **P3.3** Replace `get_market_data()` to use cached prices
- [ ] **P3.4** Handle reconnection and error recovery

### Phase 4: IBKR Order-Only Mode (TODO)

- [ ] **P4.1** Create `IBKRExecutionClient` (orders only)
- [ ] **P4.2** Simplify connection flow (no market data subscriptions)
- [ ] **P4.3** Test order execution with Polygon prices
- [ ] **P4.4** Handle price discrepancy alerts (Polygon vs IBKR fill)

### Phase 5: Cleanup & Optimization (TODO)

- [ ] **P5.1** Remove unused IBKR market data code
- [ ] **P5.2** Update dashboard to show data source
- [ ] **P5.3** Add monitoring for both connections
- [ ] **P5.4** Documentation and handoff

## Polygon.io Tier Comparison

| Tier | Price | Data Delay | WebSocket | Recommendation |
|------|-------|------------|-----------|----------------|
| Free | $0 | End of day | No | Development/Testing |
| Starter | $29 | 15-min | Yes (delayed) | Not for trading |
| Developer | $79 | 15-min | Yes (delayed) | Not for trading |
| **Advanced** | $199 | **Real-time** | Yes | **Production** |

**Current Setup:** Free tier (testing only)
**Recommended for Production:** Advanced tier ($199/mo)

## Usage

### Test the Provider

```bash
source venv/bin/activate
python3 scripts/test_polygon_provider.py
```

### Use in Code

```python
from robo_trader.data_providers import PolygonDataProvider

async def main():
    provider = PolygonDataProvider(tier="free")
    await provider.connect()

    # Get historical bars
    bars = await provider.get_historical_bars("AAPL", timeframe="1min", limit=100)
    print(bars)

    # Get current price
    price = await provider.get_current_price("AAPL")
    print(f"AAPL: ${price}")

    await provider.disconnect()
```

## Configuration

Added to `.env`:
```bash
POLYGON_API_KEY=1wpMtF8j...grRx
POLYGON_TIER=free  # free|starter|developer|advanced
```

## Rate Limiting

Free tier: 5 API calls per minute
- Provider enforces 12.5 second delay between calls
- Batch operations where possible
- Use caching to reduce API calls

## Next Steps

1. **For Backtesting:** Use Polygon for historical data (free tier works)
2. **For Live Trading:** Upgrade to Advanced tier ($199/mo) for real-time
3. **Integration:** Wire up to `runner_async.py` with fallback to IBKR

## References

- [Polygon Python Client](https://github.com/polygon-io/client-python)
- [Polygon REST API Docs](https://polygon.io/docs/stocks)
- [PyPI: polygon-api-client](https://pypi.org/project/polygon-api-client/)

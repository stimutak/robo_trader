# Handoff: AI Discovery System & Symbol Management
**Date:** 2026-01-14
**Session Focus:** Understanding why only 23 positions despite "tons of buy signals"

## Summary

User asked why the system had only 23 positions when there were many buy signals being generated. Investigation revealed the system is working correctly - "Already have long position" messages are expected behavior, not bugs.

## Key Findings

### 1. "Already Have Long Position" Is Correct Behavior
The system correctly prevents buying more of a stock you already own:
```
AI BUY signal for XOM → "Already have long position" → SKIP (correct)
AI BUY signal for CVX → "Already have long position" → SKIP (correct)
```

This is risk management, not a bug. The system uses a "one position per symbol" strategy.

### 2. AI Discovery System IS Working
The AI news scanner found real opportunities:
- **OKTA** - "Cantor Fitzgerald sees 30% upside for this cybersecurity stock"
- **SNPS** - "Loop Capital remains bullish on Synopsys due to AI tailwinds"

These were discovered from news headlines, not from a predefined list.

### 3. Symbol Sources (Order of Priority)
1. `.env` `SYMBOLS=` - Base watchlist (20 stocks)
2. Existing positions - Added automatically for SELL monitoring
3. AI-discovered opportunities - From news headline scanning

## Changes Made

### 1. Expanded News Sources (`robo_trader/news_fetcher.py`)
**Before:** 3 RSS feeds (Yahoo, Reuters, CNBC)
**After:** 12 RSS feeds including:
- MarketWatch (top stories + market pulse)
- Seeking Alpha (currents + news feed)
- TechCrunch
- Benzinga
- Additional Yahoo/Reuters/CNBC feeds

### 2. Increased Headline Scanning
- Per-source entries: 5 → 8
- Total headlines scanned: 20 → 50

### 3. Restored Original Symbol List (`.env`)
Reverted to user's original 20-symbol watchlist:
```
SYMBOLS=AAPL,NVDA,TSLA,IXHL,NUAI,BZAI,ELTP,OPEN,CEG,VRT,PLTR,UPST,TEM,HTFL,SDGR,APLD,SOFI,CORZ,WULF,IMRX
```

**Mistake made and corrected:** I initially added ~77 random blue chips which was wrong. The correct approach is to let AI discover opportunities from news, not manually expand the list.

### 4. Updated Documentation (`CLAUDE.md`)
- Added "AI-Driven Symbol Discovery" section
- Added common mistakes:
  - "Arbitrarily expanding symbol list" → Let AI discover from news
  - "'Already have position' = bug" → This is CORRECT behavior

## Architecture: How AI Discovery Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Trading Cycle                             │
├─────────────────────────────────────────────────────────────┤
│ 1. fetch_rss_news(max_items=50)                             │
│    └── 12 RSS feeds × 8 entries = ~96 headlines             │
│                                                              │
│ 2. ai_analyst.find_opportunities(headlines, exclude=owned)  │
│    └── Claude scans headlines for stock mentions            │
│    └── Returns symbols with confidence > 50%                │
│                                                              │
│ 3. symbols_to_process = base + owned + ai_discovered        │
│                                                              │
│ 4. For each symbol:                                         │
│    └── If BUY signal AND NOT owned → Execute BUY            │
│    └── If BUY signal AND owned → "Already have position"    │
│    └── If SELL signal AND owned → Execute SELL              │
└─────────────────────────────────────────────────────────────┘
```

## Files Modified

| File | Change |
|------|--------|
| `robo_trader/news_fetcher.py` | Added 9 RSS feeds, increased per-source entries |
| `robo_trader/runner_async.py` | Changed max_items from 20 to 50 |
| `.env` | Restored original SYMBOLS list |
| `CLAUDE.md` | Added AI Discovery section, common mistakes |
| `user_settings.json` | Left unchanged (was temporarily modified) |

## Current System State

- **Runner:** Running (PID 61293)
- **Gateway:** Connected on port 4002
- **Positions:** ~23 (existing holdings)
- **Symbols Processing:** Base 20 + positions + AI discoveries
- **AI Scanner:** Active, scanning 50 headlines per cycle

## Lessons Learned

1. **Don't arbitrarily expand symbol lists** - The AI should discover opportunities
2. **"Already have position" is not a bug** - It's correct risk management
3. **Symbol sources are `.env` not `user_settings.json`** - Config loads from environment
4. **News-driven discovery works** - OKTA and SNPS were real finds

## Next Steps (Optional)

1. Monitor AI discovery rate over next few days
2. Consider adding more stock-specific news sources (e.g., earnings calendars)
3. Investigate the Int/Datetime comparison error affecting GM/GOLD
4. Consider lowering AI confidence threshold if too few discoveries (currently 50%)

## Commands to Verify

```bash
# Check runner status
pgrep -f "runner_async" && echo "Running"

# Check AI discoveries in logs
grep "AI OPPORTUNITY" robo_trader.log | tail -10

# Check news scanning
grep "AI scanning" robo_trader.log | tail -5

# Verify symbol count
grep "Processing.*symbols" robo_trader.log | tail -3
```

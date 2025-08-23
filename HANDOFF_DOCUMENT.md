# 🤖 Robo Trader - System Handoff Document

## Executive Summary
AI-powered trading system using Claude 3.5 Sonnet to analyze markets and execute trades. Currently monitoring 21 symbols during market hours with a 75% conviction threshold for trades.

**Status**: ✅ Running | **Mode**: Paper Trading | **Dashboard**: http://localhost:5555

---

## 🚀 Quick Start Guide

### Start Everything (One Command)
```bash
./restart_trading.sh
```
This launches both the dashboard and AI trading system.

### Access Dashboard
Open browser to: **http://localhost:5555**

### Monitor Status
- News ticker should be scrolling
- AI status shows "Running"
- Positions update in real-time

---

## 📊 Current Configuration

### Watched Symbols (21 Total)
```
AAPL, NVDA, TSLA    # Major tech
NUAI, BZAI, PLTR    # AI plays
ADA, HBAR           # Crypto/blockchain
CORZ, WULF          # Bitcoin miners
CEG                 # Clean energy
SOFI, UPST          # Fintech
IXHL, ELTP          # Biotech/pharma
OPEN, VRT, TEM,     # Various sectors
HTFL, SDGR, APLD
```

### Trading Parameters
- **Conviction Threshold**: 75% (only trades on high confidence)
- **Position Size**: 2% of capital per trade
- **Max Daily Loss**: $1,000
- **Target Returns**: 2-5% monthly
- **Max Drawdown**: 15%

---

## ⚠️ Critical Missing Components

### 1. Company-Specific News Sources (URGENT)
**Problem**: Currently only monitoring general market news
**Impact**: Missing earnings, SEC filings, FDA approvals, insider trading

**Need to Add**:
- SEC EDGAR API for 8-K filings
- Earnings calendar integration
- FDA approval calendar (for IXHL, ELTP)
- Insider trading alerts (Form 4)
- Company press releases
- Crypto regulatory news

### 2. Modern UI Design
**Problem**: Dashboard looks dated, hard to read
**Solution**: Implement Cursor-style dark theme with:
- Glass morphism effects
- Smooth animations
- Real-time charts
- Better data visualization

---

## 🛠 Technical Architecture

### Core Components
```
start_ai_trading.py          # Main entry point
├── ai_runner.py            # Orchestrates everything
├── intelligence.py         # Claude AI integration
├── news.py                # RSS aggregation (NEEDS MORE SOURCES)
├── options_flow.py        # Options scanner
├── events.py              # Event processing
├── kelly.py               # Position sizing
└── app.py                 # Web dashboard (NEEDS UI UPDATE)
```

### Data Flow
```
News/Options → AI Analysis → Signal Generation → Risk Check → Order Execution
                   ↓
            Claude 3.5 Sonnet
            (75% conviction threshold)
```

### Active News Sources
- Bloomberg Markets
- Yahoo Finance  
- CNBC Top Stories
- MarketWatch
- Wall Street Journal

### Missing News Sources (Priority)
- SEC filings
- Earnings releases
- FDA calendar
- Insider trading
- Social sentiment (Reddit, StockTwits)

---

## 📈 Why No Trades Yet?

The system is working but being appropriately selective:

1. **General News Only**: Missing company-specific catalysts
2. **High Conviction Bar**: 75% threshold (by design)
3. **Market Hours Only**: Runs 9:30 AM - 4:00 PM ET

**Expected**: 10-30 quality trades per month, not daily

---

## 🔧 Maintenance & Operations

### Daily Tasks
- Verify system running during market hours
- Check dashboard for errors
- Monitor news ticker flowing

### Weekly Tasks
- Review AI decision logs
- Update watchlist if needed
- Check for executed trades

### Emergency Stop
```bash
pkill -f python  # Kills everything
```

---

## 🔑 API Keys & Services

### Currently Active
- ✅ **ANTHROPIC_API_KEY**: Claude AI (in .env)
- ✅ **Interactive Brokers TWS**: Port 7497

### Need to Add
- **SEC_API_KEY**: EDGAR filings
- **ALPHAVANTAGE_API_KEY**: Earnings data
- **TWITTER_API_KEY**: Social sentiment
- **REDDIT_CLIENT_ID**: WSB monitoring

---

## 📝 Configuration Files

### user_settings.json
Stores symbols, risk preferences, saved automatically

### .env
Contains API keys (never commit!)
```
ANTHROPIC_API_KEY=your_key_here
```

### CLAUDE.md
Project philosophy - **READ THIS FIRST**
Core principle: "One intelligent trade beats 1000 fast trades"

---

## 🐛 Known Issues

### Event Loop Error
- **Symptom**: "This event loop is already running"
- **Impact**: Minor, some data updates skip
- **Status**: Non-critical, mitigated

### Limited News Coverage  
- **Symptom**: No company-specific news
- **Impact**: Missing trading opportunities
- **Fix**: Add SEC, earnings, insider feeds

### Basic Dashboard UI
- **Symptom**: Dated appearance, poor UX
- **Impact**: Hard to monitor effectively
- **Fix**: Implement modern dark theme

---

## 📋 Handoff Checklist

### System Health
- [ ] TWS running on port 7497
- [ ] Paper trading account selected
- [ ] ANTHROPIC_API_KEY in .env
- [ ] Run ./restart_trading.sh
- [ ] Dashboard loads at localhost:5555
- [ ] News ticker scrolling
- [ ] No red errors in console

### Next Steps (Priority Order)
1. [ ] Add SEC EDGAR integration
2. [ ] Add earnings calendar API  
3. [ ] Implement modern UI theme
4. [ ] Add FDA calendar for biotechs
5. [ ] Integrate social sentiment
6. [ ] Add insider trading alerts
7. [ ] Create trade journal
8. [ ] Build performance metrics

---

## 💡 Quick Wins

### Test the System
Add a high-impact test news item to trigger AI:
```python
# Would likely generate >75% conviction
"NVDA Announces 50% Guidance Raise on AI Demand"
```

### Add Company RSS
Many companies have investor RSS feeds:
```
https://investor.nvidia.com/rss
https://investor.apple.com/rss
```

### Enable More Logging
Set logger level to DEBUG in logger.py for more detail

---

## 📞 Support Resources

### Documentation
- **CLAUDE.md**: Core philosophy and rules
- **TODO.md**: Development roadmap  
- **README.md**: Setup instructions

### File Locations
- Logs: Console output
- Settings: user_settings.json
- Code: /robo_trader/

### Philosophy
Per CLAUDE.md: **"Intelligence over speed"**
- Quality > Quantity
- High conviction only
- Capital preservation first

---

## 🎯 Success Metrics

### Working Correctly If:
- News updating every 5 minutes
- AI analyzing high-impact stories
- Options flow scanning active
- No trades = System being selective (good!)

### Needs Attention If:
- No news in ticker for 10+ minutes
- Red errors in console
- Dashboard not updating
- IB disconnection warnings

---

*Last Updated: August 21, 2025*
*Built with Claude 3.5 Sonnet for intelligent market analysis*
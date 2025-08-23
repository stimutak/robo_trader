## Robo Trader - Intelligent Autonomous Trading Bot

An AI-powered trading system that uses Claude AI to understand market events and make profitable trades. Combines real-time news analysis, options flow tracking, and risk management for intelligent trading decisions.

> **✅ Current Status**: System RUNNING during market hours | Dashboard at http://localhost:5555 | Paper Trading Active

### 🎯 Mission
Build a profitable trading bot that thinks like a master trader - focusing on **intelligence over speed**. The system analyzes market events using AI, detects institutional positioning, and adapts strategies to market conditions.

### 🚀 Key Features
- **Decisive AI Trading**: Professional quant-level prompt with forced JSON decisions
- **AI Market Analysis**: Claude 3.5 Sonnet for understanding news, earnings, Fed speeches
- **Event-Driven Trading**: React to catalysts before retail traders
- **Smart Money Tracking**: Options flow and institutional footprint detection
- **Multi-Asset Support**: Trade stocks, gold (GLD ETF), and crypto (BTC, ETH)*
- **Advanced Risk Controls**: 
  - ATR-based position sizing
  - 0.50% per-trade risk cap
  - 2% daily / 5% weekly drawdown limits
  - Mandatory stop losses
- **Liquidity Requirements**: $3M minimum ADV, 1% max spread
- **Correlation Control**: 35% max exposure per sector/theme
- **Expected Value Engine**: EV calculation, Kelly sizing, 1.8:1 min RR
- **Calibration Tracking**: Brier scores, reliability metrics, decision persistence
- **Risk-First Design**: Paper trading default with strict risk controls
- **Profit Focused**: Target 2-5% monthly returns with <15% max drawdown

*Note: Crypto requires IB live account, not available in paper trading

### Architecture
```
Market Data → AI Analysis → Decision Engine → Risk Management → Execution
     ↓              ↓              ↓                ↓              ↓
Historical     News/Events    Strategy         Position        IB API
  Context       + LLM         Selection         Sizing
```

### Project Layout
```
robo_trader/
├── robo_trader/
│   ├── config.py              # ✅ Enhanced with 15+ new parameters
│   ├── ibkr_client.py         # IB API wrapper for market data
│   ├── execution.py           # Paper/live execution
│   ├── risk.py                # ✅ ATR-based sizing, weekly DD limits
│   ├── strategies.py          # Trading strategies
│   ├── portfolio.py           # Portfolio & PnL tracking
│   ├── logger.py              # Centralized logging
│   ├── runner.py              # Basic runner (SMA strategy)
│   ├── ai_runner.py           # ✅ AI-powered trading system
│   ├── intelligence.py        # ✅ Claude integration (uses new LLM client)
│   ├── llm_client.py          # ✅ NEW: Decisive LLM with forced JSON
│   ├── schemas.py             # ✅ NEW: Pydantic decision schemas
│   ├── market_meta.py         # ✅ NEW: Liquidity & spread checks
│   ├── correlation.py         # ✅ NEW: Sector exposure control
│   ├── edge.py                # ✅ NEW: EV calculation & Kelly sizing
│   ├── calibration.py         # ✅ NEW: Brier scores & reliability
│   ├── news.py               # ✅ RSS news aggregation (9+ feeds)
│   ├── events.py             # ✅ Event-driven framework
│   ├── kelly.py              # ✅ Kelly Criterion sizing
│   ├── sentiment.py          # ✅ Sentiment analysis
│   ├── options_flow.py       # ✅ Options flow analysis
│   ├── company_intelligence.py # ✅ SEC filings, earnings, FDA tracking
│   └── database.py           # ✅ Enhanced with decision tracking
├── tests/
│   ├── test_risk.py
│   ├── test_strategies.py
│   ├── test_portfolio.py
│   └── test_retry.py
├── requirements.txt
├── PROJECT_PLAN.md           # Development roadmap
├── TODO.md                   # Current tasks
└── CLAUDE.md                 # AI assistant context
```

### 🚀 Quick Start - AI Trading in 3 Steps

```bash
# 1) Setup (one time)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure (already done if you have .env)
# Make sure .env has:
#   - ANTHROPIC_API_KEY (for Claude AI)
#   - IBKR_PORT=7497 (TWS paper trading)

# 3) START EVERYTHING! 🤖
./restart_trading.sh
# This starts both dashboard and AI trading system
```

That's it! The bot will:
- ✅ Fetch news from 9+ sources every 5 minutes
- ✅ Analyze with Claude AI for trading opportunities  
- ✅ Size positions optimally with Kelly Criterion
- ✅ Execute trades through Interactive Brokers
- ✅ Manage risk automatically

### 📊 Monitor Your Bot

**Web Dashboard** (http://localhost:5555):
- Automatically starts with `./restart_trading.sh`
- Or run separately: `python app.py`

**Test Components**:
```bash
# Test complete pipeline
python test_integration.py

# Test individual components
python -m robo_trader.news       # Test news feeds
python -m robo_trader.events     # Test event processing
python -m robo_trader.kelly      # Test position sizing
```

### 🎯 How to Use the AI Trading Bot

#### Test AI Analysis (No Trading)
```bash
# Test Claude's ability to analyze market events
python test_claude.py

# Example output:
# Direction: bullish
# Conviction: 75%
# Entry Price: $450.25
# Stop Loss: $444.75
# Rationale: Fed pivot confirmation creates sustained bullish setup...
```

#### Run AI-Powered Trading
```bash
# Run the example that analyzes events and makes paper trades
python ai_trading_example.py

# This will:
# 1. Connect to IB Gateway
# 2. Analyze sample events (Fed, earnings, news)
# 3. Get Claude's trading signals
# 4. Size positions using Kelly Criterion
# 5. Execute paper trades with risk management
# 6. Show portfolio P&L
```

#### Manual Event Analysis
```python
from robo_trader.intelligence import ClaudeTrader

claude = ClaudeTrader()
signal = await claude.analyze_market_event(
    "Tesla beats Q4 delivery expectations",
    "TSLA",
    {"price": 250, "volume": 100000000}
)
print(f"Trade {signal['direction']} with {signal['conviction']}% conviction")
```

#### Full Documentation
See `USAGE.md` for comprehensive guide including:
- Detailed examples
- Trading strategies
- Configuration options
- Safety features
- Troubleshooting

### 📦 Multi-Asset Trading Support

The system now supports multiple asset classes:

**Supported Assets:**
- **Stocks**: All US equities (AAPL, NVDA, TSLA, etc.)
- **Gold**: GLD ETF for gold exposure
- **Crypto**: BTC-USD, ETH-USD (requires IB live account)

**Default Watchlist** (21 symbols):
```python
STOCKS = ["AAPL", "NVDA", "TSLA", "IXHL", "NUAI", "BZAI", "ELTP", 
          "OPEN", "CEG", "VRT", "PLTR", "UPST", "TEM", "HTFL", 
          "SDGR", "APLD", "SOFI", "CORZ", "WULF"]
GOLD = ["GLD"]  # Gold ETF
CRYPTO = ["BTC-USD", "ETH-USD"]  # Major cryptocurrencies
```

**Asset Type Indicators** in Dashboard:
- 🥇 Gold assets
- ₿ Cryptocurrency
- Regular stocks (no indicator)

### Configuration
All values are read in `robo_trader/config.py` via environment variables. Defaults are conservative.

**Required for IBKR connectivity:**
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`

**AI/ML Integration:**
- `ANTHROPIC_API_KEY` for Claude AI analysis (required)
- News feeds configured automatically

**Risk and trading mode:**
- `TRADING_MODE` defaults to `paper` (live requires explicit flags)
- `MAX_DAILY_LOSS`, `MAX_POSITION_RISK_PCT`, `MAX_SYMBOL_EXPOSURE_PCT`, `MAX_LEVERAGE`

### 📈 Performance Targets
- **Monthly Returns**: 2-5%
- **Max Drawdown**: <15%
- **Sharpe Ratio**: >1.5
- **Win Rate**: 45-55% (focus on risk/reward)

### 🛡️ Safety & Risk Management
- **Paper First**: 30 days minimum paper trading before live
- **Position Limits**: Max 10% per position, 30% per sector
- **Daily Loss Limit**: 3% max daily loss
- **Live Trading Gate**: Requires `TRADING_MODE=live` + `--confirm-live` flag
- **Emergency Stop**: Manual kill switch for all positions

### 🧠 Intelligence Features (Live Now!)
1. **✅ LLM Market Analysis**: Claude 3.5 Sonnet analyzes all market events
2. **✅ Event Detection**: Real-time SEC filings, earnings, FDA approvals
3. **✅ Sentiment Analysis**: Market psychology from 9+ news sources
4. **✅ Options Flow**: Track unusual options activity and institutional moves
5. **✅ Company Intelligence**: Monitor 8-K, 10-Q, Form 4 filings automatically
6. **🔄 Regime Detection**: Adapt strategies to market conditions (in progress)

### 📊 Development Roadmap
See `PROJECT_PLAN.md` for detailed phases:
- **Phase 1**: Intelligence Layer (LLM, news, events) - *Current Priority*
- **Phase 2**: Smart Money Analysis (options flow, microstructure)
- **Phase 3**: Adaptive Strategies (regime detection, multi-strategy)
- **Phase 4**: ML Enhancement (price prediction, sentiment)
- **Phase 5**: Production Readiness (monitoring, live gate)

### CI/CD
GitHub Actions workflow at `.github/workflows/ci.yml` runs tests on push/PR.

### Contributing
1. Read `CLAUDE.md` for project philosophy and constraints
2. Check `TODO.md` for current tasks
3. Follow the NEVER/ALWAYS rules in development
4. Keep tests green and add tests for new features
5. Document trade rationale in commits



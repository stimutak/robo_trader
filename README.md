## Robo Trader - Intelligent Autonomous Trading Bot

An AI-powered trading system that uses LLMs to understand market events and make profitable trades. Combines real-time news analysis, institutional flow tracking, and adaptive strategies to generate consistent returns.

> **⚠️ Current Status**: AI Intelligence Layer ✅ COMPLETE | IB Broker Connection ❌ BLOCKED (authentication issue)

### 🎯 Mission
Build a profitable trading bot that thinks like a master trader - focusing on **intelligence over speed**. The system analyzes market events using AI, detects institutional positioning, and adapts strategies to market conditions.

### 🚀 Key Features
- **AI Market Analysis**: LLM integration for understanding news, earnings, Fed speeches
- **Event-Driven Trading**: React to catalysts before retail traders
- **Smart Money Tracking**: Options flow and institutional footprint detection
- **Adaptive Strategies**: Dynamic strategy selection based on market regime
- **Risk-First Design**: Paper trading default with strict risk controls
- **Profit Focused**: Target 2-5% monthly returns with <15% max drawdown

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
│   ├── config.py              # Env-driven configuration
│   ├── ibkr_client.py         # IB API wrapper for market data
│   ├── execution.py           # Paper/live execution
│   ├── risk.py                # Position sizing & risk management
│   ├── strategies.py          # Trading strategies
│   ├── portfolio.py           # Portfolio & PnL tracking
│   ├── logger.py              # Centralized logging
│   ├── runner.py              # Main orchestrator
│   └── (planned)
│       ├── intelligence.py    # LLM market analysis
│       ├── news.py           # News ingestion pipeline
│       ├── events.py         # Event-driven framework
│       ├── options_flow.py   # Options analysis
│       └── regime.py         # Market regime detection
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

### Quickstart
```bash
# 1) Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# 3) Configure env
cp .env.example .env
# Edit .env - MUST add your ANTHROPIC_API_KEY for AI features
# Also configure IBKR connection settings

# 4) Run tests
pytest -q

# 5) Test AI analysis
python test_claude.py  # Test Claude market analysis
python test_sentiment.py  # Test sentiment analysis

# 6) Run AI-powered trading example
python ai_trading_example.py  # Analyzes events and makes trades

# 7) Run basic paper trading (no AI yet)
python -m robo_trader.runner
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

### Configuration
All values are read in `robo_trader/config.py` via environment variables. Defaults are conservative.

**Required for IBKR connectivity:**
- `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`

**AI/ML Integration (planned):**
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for LLM analysis
- News API keys for data feeds

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

### 🧠 Intelligence Features (Coming Soon)
1. **LLM Market Analysis**: Understand Fed speeches, earnings calls, breaking news
2. **Event Detection**: Identify high-impact catalysts in real-time
3. **Sentiment Analysis**: Gauge market psychology from news and social media
4. **Options Flow**: Track smart money and institutional positioning
5. **Regime Detection**: Adapt strategies to market conditions

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



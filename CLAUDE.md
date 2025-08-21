## Project Intelligence Guide — Robo Trader

### Project: Intelligent Autonomous Trading Bot
**Mission**: Build an intelligent autonomous trading bot that makes profitable trades based on real-time market awareness, news analysis, and master-level understanding of finance. Focus on **intelligence over speed** - smart trades beat fast trades.

**Goal**: Create a profitable trading system for 1-2 accounts that:
- Uses LLMs to understand market events like a master trader
- Reacts to news, earnings, Fed announcements before retail traders
- Adapts strategies based on market conditions
- Generates consistent returns (2-5% monthly target)

### Core Philosophy
- **Intelligence-Driven**: Use AI/LLMs for market event understanding
- **Capital Preservation First**: Risk limits are non-negotiable
- **Event-Driven Trading**: React to catalysts with high conviction
- **Paper-First Development**: Thoroughly test before risking capital
- **Clarity Over Cleverness**: Prefer straightforward, readable code
- **No Unnecessary Abstraction**: Keep modules small and purpose-driven
- **Fix In Place**: Improve existing modules before adding new ones
- **Test Everything**: Especially risk logic, strategies, and integrations
- **Determinism & Reproducibility**: Same inputs and environment → same results

### Critical Constraints
#### NEVER DO
- ❌ Execute live trades by default
- ❌ Commit API keys, credentials, or PII
- ❌ Create duplicate files (e.g., `*-v2.py`, `*-enhanced.py`)
- ❌ Add complex frameworks/abstractions for simple tasks
- ❌ Bypass or weaken risk checks (daily loss, symbol exposure, leverage)
- ❌ Promise profits or rely on unverified predictive claims
- ❌ Use `print` debugging in library code (prefer structured logging/tests)

#### ALWAYS DO
- ✅ Search for existing code before adding new modules
- ✅ Keep paper trading as default; guard any live path behind config and explicit approval
- ✅ Enforce strict type hints and readable naming
- ✅ Write tests for new features; keep the test suite green
- ✅ Document the "why" for meaningful changes
- ✅ Validate inputs; handle error/edge cases first
- ✅ Keep functions small with clear responsibilities

### Architecture Overview
```
Market Data → AI Analysis → Decision Engine → Risk Management → Execution
     ↓              ↓              ↓                ↓              ↓
Historical     News/Events    Strategy         Position        IB API
  Context       + LLM         Selection         Sizing
```

### Key Components

#### 1. Intelligence Layer (PRIORITY)
- **LLM Integration**: Claude/GPT-4 for market event analysis
- **News Pipeline**: RSS feeds, financial APIs, social sentiment
- **Event Understanding**: Earnings, Fed speeches, economic data
- **Impact Assessment**: Predict market reaction before humans

#### 2. Smart Money Following
- **Options Flow**: Detect unusual activity and institutional positioning
- **Dark Pools**: Track large block trades
- **Volume Analysis**: Identify accumulation/distribution

#### 3. Adaptive Strategies
- **Market Regime Detection**: Risk-on/off, trending/ranging, volatile/calm
- **Dynamic Selection**: Choose strategy based on current conditions
- **Event-Driven**: Trade catalysts with asymmetric risk/reward

#### 4. Risk Management
- **Kelly Criterion**: Size positions based on edge
- **Max Drawdown**: Never exceed 15% account drawdown
- **Per-Trade Risk**: 1-2% max risk per position
- **Daily Loss Limits**: Enforce strict daily and per-symbol caps

### Trading Edge Sources
1. **Information Asymmetry**: LLM understands implications faster
2. **Sentiment Analysis**: Gauge market psychology from news/social
3. **Technical + Fundamental**: Combine price action with catalysts
4. **Regime Adaptation**: Switch strategies as market changes

### Profit Targets
- **Monthly**: 2-5% return (24-80% annually)
- **Win Rate**: 45-55% (focus on risk/reward ratio)
- **Sharpe Ratio**: > 1.5
- **Max Drawdown**: < 15%
- **Trades/Month**: 10-30 quality setups

### Current Structure
```
robo_trader/
├── robo_trader/
│   ├── __init__.py
│   ├── config.py              # Env-driven configuration
│   ├── ibkr_client.py         # Async ib_insync client wrapper
│   ├── execution.py           # Paper execution simulator
│   ├── risk.py                # Position sizing & exposure checks
│   ├── strategies.py          # SMA crossover (to be enhanced)
│   ├── portfolio.py           # Position and PnL tracking
│   ├── logger.py              # Centralized logging
│   ├── retry.py               # Retry/backoff utilities
│   └── runner.py              # Main orchestrator
├── tests/
│   ├── test_risk.py
│   ├── test_strategies.py
│   ├── test_portfolio.py
│   └── test_retry.py
├── requirements.txt
└── pyproject.toml
```

### Tech Stack
- **Language**: Python (>=3.10; dev target 3.13)
- **Data & IO**: pandas, numpy
- **Broker API**: ib_insync (IBKR)
- **AI/ML**: OpenAI/Anthropic APIs, transformers (planned)
- **News**: feedparser, requests, newsapi (planned)
- **Config**: python-dotenv
- **Tests**: pytest

### Development Priorities

#### Phase 1: Intelligence Foundation (IMMEDIATE)
- [ ] Add LLM integration (OpenAI/Anthropic API)
- [ ] Build news ingestion pipeline (RSS + APIs)
- [ ] Create event-driven trading framework
- [ ] Implement intelligent position sizing (Kelly criterion)
- [ ] Backtest on recent Fed meetings/earnings

#### Phase 2: Smart Strategies (Week 2)
- [ ] Add options flow analysis from IB
- [ ] Implement market regime classifier
- [ ] Build institutional footprint detection
- [ ] Create multi-strategy ensemble
- [ ] Test on paper account with real events

#### Phase 3: Production Trading (Week 3-4)
- [ ] Complete 30-day paper trading validation
- [ ] Implement performance monitoring
- [ ] Add strategy optimization
- [ ] Start live trading with small size ($1k-10k positions)
- [ ] Scale based on performance

### Development Commands
```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
pip install -e .

# Tests
pytest -q

# PREFERRED: Start complete system (dashboard + AI trading)
./restart_trading.sh
# This script:
# - Kills any existing processes cleanly
# - Starts dashboard at http://localhost:5555
# - Starts AI trading with all 21 symbols
# - Activates virtual environment automatically

# Alternative: Manual startup
python app.py &  # Start dashboard
python start_ai_trading.py  # Start AI trading with 21 symbols

# Legacy: Direct runner (requires manual symbol list)
python -m robo_trader.runner --symbols SPY,QQQ,TSLA

# Run with custom parameters
python -m robo_trader.runner --sma-fast 10 --sma-slow 20 --slippage-bps 5 --max-order-notional 10000
```

### Configuration
- All runtime configuration via `robo_trader/config.py` from environment
- Required: `IBKR_HOST`, `IBKR_PORT`, `IBKR_CLIENT_ID`
- Trading mode: `TRADING_MODE=paper` by default
- Add: `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for LLM
- News sources: Configure RSS feeds and API keys

### Testing Requirements
- Run unit tests before any commit; keep test suite fast and deterministic
- Add/extend tests for risk logic, order validation, and new strategies
- Cover edge cases: zero/negative prices, invalid quantities, exposure limits
- Integration tests for IB API and LLM calls
- Backtesting framework for historical events
- Paper trading: 30 days minimum before live

### Security Checklist
- [ ] API keys only via environment variables
- [ ] Never commit `.env` or secrets
- [ ] Sanitize logs (no PII/secrets)
- [ ] Respect IBKR pacing/rate limits
- [ ] Respect API rate limits for LLM/news
- [ ] Pin or vet dependencies regularly
- [ ] Secure LLM prompts (no injection)

### Git Workflow
1. Branch from main: `feature/short-description`
2. Use semantic commits with concise rationale
3. Keep tests green locally before PR
4. Update TODO.md and this file
5. Squash merge to main

### Planning & TODO Discipline
- Maintain `PROJECT_PLAN.md` (phases, tasks, estimates) and `TODO.md` (next steps)
- These files must be reviewed and updated at every session start and end, and on every commit
- Track immediate tasks in TodoWrite tool
- Document completed trades and learnings

### File Creation Policy
- Prefer in-place edits; create new files only when necessary for clarity
- If adding a new module, document why it exists, expected inputs/outputs, and tests

### Logging & Observability
- Prefer structured logging (centralized in logger.py)
- Keep library code quiet; rely on tests until logging module is introduced
- When adding logging, use levels and contextual metadata

### Live Trading Safeguards
- Live execution paths must:
  - Check `TRADING_MODE == "live"` and require explicit user confirmation
  - Pass all tests and include integration tests or dry-run harnesses
  - Enforce the same risk checks as paper mode
  - Provide clear rollback switches and maximum notional limits

### Code Style
- Descriptive names, explicit types, early returns, narrow responsibilities
- Validate inputs and handle errors first; avoid deep nesting
- Keep formatting consistent; do not reformat unrelated code in edits

### Non-Goals
- Predicting markets with certainty or guaranteeing returns
- Building UI or web services unless explicitly requested
- Competing on microsecond latency (intelligence > speed)

### Session Discipline
- Review this file at start/end of each session
- Update TODO.md with completed/new tasks
- Commit with clear messages including:
  - "🤖 Generated with Claude Code"
  - Co-Authored-By: Claude <noreply@anthropic.com>

### Handoff Document Convention
- Location: `/handoff/` directory in project root
- Naming: `YYYY-MM-DD_HHMM_handoff.md` (24-hour format, local time)
- Latest: Symlink `LATEST_HANDOFF.md` → most recent handoff doc
- Archive: Keep last 5 handoffs, archive older ones to `/handoff/archive/`
- Content: Status, running processes, next steps, blockers, session notes

### Remember
**The Goal**: Build a profitable trading bot that thinks like a master trader. One intelligent, high-conviction trade beats 1000 fast but dumb trades. Focus on understanding market events through AI, not competing on speed.

---
This document merges the original project discipline with the new intelligence-first trading approach. Paper mode and risk guardrails remain the default, but the focus shifts from speed to intelligence and profitability.
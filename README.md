# Robo Trader - ML-Driven Production Trading Platform

> ## ⚠️ CRITICAL: SYSTEM NOT READY FOR LIVE TRADING ⚠️
> **This system has critical safety issues that MUST be resolved before any live trading.**  
> **See [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md) for the mandatory action plan.**  
> **Current Production Readiness Score: 5/10 - PAPER TRADING ONLY**
>
> All developers MUST follow the production readiness plan until completion.

> ## 🛡️ Security Audit Completed 2026-05-10
> A comprehensive multi-agent security audit (commit `e528431`) addressed 53 of 57 findings across six attack surfaces. **If you are returning to dev work or onboarding a new machine, read [DEV_SETUP.md](DEV_SETUP.md) first** — particularly **[Section 3: What CHANGED for developers post-audit](DEV_SETUP.md#3-what-changed-for-developers-post-audit)**.
>
> | Doc | Purpose |
> |---|---|
> | [DEV_SETUP.md](DEV_SETUP.md) | Per-machine setup + dev habit changes (read this first) |
> | [SECURITY_AUDIT_2026-05-10.md](SECURITY_AUDIT_2026-05-10.md) | Full audit: 57 findings with file:line, exploit, fix, test |
> | [SECURITY_TEST_PLAN.md](SECURITY_TEST_PLAN.md) | 7-phase validation walkthrough; run before live trading |
>
> **Headlines (things that now behave differently):** stop-losses actually trigger (keying bug fixed); IBKR Gateway is now read-only at the broker level; AI BUY requires ML corroboration; pairs trading goes through full risk validation; dashboard auth is fail-closed with CSRF protection; HMAC integrity verification on all model artifact loads.

A production-grade algorithmic trading system with IBKR integration, featuring advanced ML infrastructure, async architecture, and comprehensive risk management. Paper trading by default with strict capital preservation controls.

## 🚀 Current Status

**⛔ SAFETY FIRST: Production readiness issues being addressed - see [PRODUCTION_READINESS_PLAN.md](PRODUCTION_READINESS_PLAN.md)**

**Phase 3 In Progress!** Advanced Strategy Development with ML Enhanced Framework.
- ✅ Phase 1: Foundation & Quick Wins (100% Complete)
- ✅ Phase 2: ML Infrastructure & Backtesting (100% Complete)
- 🚧 Phase 3: Advanced Strategy Development (S1 Complete - ML Enhanced Strategy)
- ⏳ Phase 4: Production Hardening & Deployment

See `IMPLEMENTATION_PLAN.md` for the complete 16-week roadmap.

## ✨ Key Features

### Core Infrastructure (Phase 1 - Complete)
- ✅ **Async IBKR Client**: Non-blocking operations with subprocess isolation
- ✅ **Pydantic Configuration**: Type-safe config with validation
- ✅ **Parallel Processing**: 3x throughput improvement
- ✅ **Async Database**: SQLite with connection pooling
- ✅ **Performance Monitoring**: Real-time metrics dashboard
- ✅ **TWS Connection Issues**: Resolved with subprocess-based approach

### ML Infrastructure (Phase 2 - Complete)
- ✅ **Feature Engineering**: 25+ technical indicators with time series
- ✅ **ML Model Training**: RF, XGBoost, LightGBM, Neural Networks
- ✅ **Walk-Forward Backtesting**: Realistic execution simulation
- ✅ **Performance Analytics**: Comprehensive metrics and attribution
- ✅ **Correlation Analysis**: Risk-aware position sizing

### Advanced Strategies (Phase 3 - In Progress)
- ✅ **ML Enhanced Strategy**: Market regime detection with multi-timeframe analysis
- ✅ **Regime Detection**: Identifies trending, volatile, ranging, and crash markets
- ✅ **Multi-Timeframe Analysis**: Confirms signals across 1m, 5m, 15m, 1h, 1d
- ✅ **Dynamic Position Sizing**: Adapts to market conditions and confidence
- ✅ **Multi-Strategy Portfolio Manager (S3)**: Dynamic allocation across strategies (Equal Weight, Risk Parity, Adaptive)
- 🚧 **Smart Execution**: TWAP, VWAP, Iceberg algorithms (in progress)

### Risk Management
- **Trailing Stops**: Follows price up, locks in profits (5% default)
- **Position Sizing**: Fixed, ATR-based, or Kelly Criterion
- **Portfolio Heat**: Maximum 6% risk exposure
- **Emergency Shutdown**: Auto-triggers on violations
- **Correlation Limits**: 0.7 maximum between positions
- **Daily Loss Limits**: Configurable stop-loss
- **10 Risk Violation Types**: Comprehensive monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ (tested on 3.13)
- Interactive Brokers TWS or IB Gateway
- IBKR Paper Trading Account (recommended for testing)

### Installation

```bash
# 1) Clone repository
git clone https://github.com/stimutak/robo_trader.git
cd robo_trader

# 2) Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4) Configure environment
cp .env.example .env
# Edit .env with your IBKR credentials and risk settings
```

### Running the System

**Recommended: Use the automated startup script**

```bash
# Start with default symbols (from user_settings.json)
./START_TRADER.sh

# Start with custom symbols
./START_TRADER.sh "AAPL,NVDA,TSLA,QQQ"
```

The startup script automatically:
- ✅ Kills existing processes
- ✅ Cleans up zombie connections
- ✅ Tests Gateway connectivity
- ✅ Starts WebSocket server
- ✅ Starts trading system
- ✅ Monitors startup health

**Manual startup (advanced users):**

```bash
# Always activate virtual environment first
source venv/bin/activate

# Option 1: Run async trading system with parallel processing
python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA

# Option 2: Run with ML Enhanced Strategy (regime detection + multi-timeframe)
python -m robo_trader.runner_async --symbols AAPL,NVDA --use-ml-enhanced

# Option 3: Run with dashboard (port 5555)
export DASH_PORT=5555
python app.py

# Option 4: Test ML pipeline
python test_ml_pipeline.py

# Option 5: Test ML Enhanced Strategy
python test_ml_enhanced_strategy.py
```

**Diagnostic tools:**

```bash
# Test Gateway connectivity
./force_gateway_reconnect.sh

# Full system diagnostics
python diagnose_gateway_api.py

# Check for zombie connections
lsof -nP -iTCP:4002 -sTCP:CLOSE_WAIT
```

## 📊 ML Capabilities

### Feature Engineering Pipeline
```python
# Generate features for any symbol
from robo_trader.features.feature_pipeline import FeaturePipeline

pipeline = FeaturePipeline()
features = pipeline.generate_features(df, symbol='AAPL')
# Returns 25+ technical indicators across multiple timeframes
```

### Model Training & Selection
```python
# Train and compare multiple models
from robo_trader.ml.model_selector import ModelSelector

selector = ModelSelector()
best_model = selector.select_best_model(X_train, y_train, X_test, y_test)
# Trains RF, XGBoost, LightGBM, and Neural Networks with hyperparameter tuning
```

### ML Enhanced Strategy (NEW!)
```python
# Use advanced ML strategy with regime detection
from robo_trader.strategies.ml_enhanced_strategy import MLEnhancedStrategy

strategy = MLEnhancedStrategy(config)
await strategy.initialize()

# Analyzes market regime and multi-timeframe signals
signal = await strategy.analyze('AAPL', market_data)
# Returns signal with dynamic position sizing and risk parameters
```

Key features:
- **Market Regime Detection**: Identifies bull, bear, volatile, ranging, and crash markets
- **Multi-Timeframe Analysis**: Confirms signals across 5 timeframes
- **Dynamic Position Sizing**: Adjusts based on regime and confidence (0.5% - 5% of capital)
- **Adaptive Risk Management**: Stop-loss and take-profit adapt to market conditions

### Performance Analytics
```python
# Analyze strategy performance
from robo_trader.analytics.strategy_performance import StrategyPerformanceTracker

tracker = StrategyPerformanceTracker()
metrics = tracker.calculate_metrics(returns, benchmark_returns)
# Returns Sharpe, Sortino, Calmar, attribution, and 20+ other metrics
```

## 📁 Project Structure

```
robo_trader/
├── robo_trader/
│   ├── config.py                    # Pydantic configuration
│   ├── runner_async.py              # Async parallel trading system
│   ├── clients/
│   │   └── async_ibkr_client.py    # Non-blocking IBKR operations
│   ├── features/                    # ML Feature Engineering
│   │   ├── feature_pipeline.py     # Main feature generation
│   │   └── technical_indicators.py # 25+ indicators
│   ├── ml/                         # Machine Learning
│   │   ├── model_trainer.py        # Model training pipeline
│   │   └── model_selector.py       # Model selection & tuning
│   ├── backtesting/                # Backtesting Framework
│   │   ├── walk_forward_optimization.py
│   │   └── execution_simulator.py
│   ├── analytics/                  # Performance Analytics
│   │   └── strategy_performance.py # Comprehensive metrics
│   ├── portfolio/                  # Multi-Strategy Portfolio Manager (S3)
│   │   └── portfolio_manager.py    # Allocation + metrics + rebalancing
│   └── websocket_server.py         # Real-time updates
├── app.py                           # Monitoring dashboard
├── tests/                           # Test suites
├── performance_results/             # Strategy performance JSON
├── trained_models/                  # Saved ML models
└── IMPLEMENTATION_PLAN.md           # Development roadmap
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# IBKR Connection
IBKR_HOST=127.0.0.1
IBKR_PORT=7497              # TWS Paper: 7497, Live: 7496
IBKR_CLIENT_ID=123
IBKR_READONLY=true          # Keep Gateway/TWS in read-only mode
IBKR_TIMEOUT=10.0           # Handshake timeout (seconds)
IBKR_SSL_MODE=auto          # auto | require (TLS only) | disabled (plain TCP)
IBKR_FORCE_DISCONNECT=0     # Leave disabled; set to 1 only for controlled test scripts

# Trading Mode
EXECUTION_MODE=paper         # paper or live
ENVIRONMENT=dev              # dev, staging, or production

# Risk Management
RISK_MAX_POSITION_PCT=0.02   # 2% per position
RISK_MAX_DAILY_LOSS_PCT=0.005 # 0.5% daily stop
RISK_MAX_LEVERAGE=2.0        # Maximum leverage
USE_TRAILING_STOP=true       # Enable trailing stops (recommended)
TRAILING_STOP_PERCENT=5.0    # 5% below high water mark
STOP_LOSS_PERCENT=2.0        # Fixed stop (if trailing disabled)

# ML Configuration
ML_FEATURE_LOOKBACK=60       # Days of history for features
ML_RETRAIN_FREQUENCY=7       # Days between model retraining
ML_MIN_TRAIN_SAMPLES=100     # Minimum samples for training

# Monitoring
MONITORING_LOG_FORMAT=plain   # plain or json
DASH_PORT=5555               # Dashboard port
```

> ℹ️ **IBKR SSL mode**: leave `IBKR_SSL_MODE=auto` to try plain TCP first and fall back to TLS automatically. If your Gateway is configured to require TLS, set it to `require` to skip the initial plain attempt.

## 🧪 Testing

### Run Complete Test Suite
```bash
# Phase 1 tests (infrastructure)
python test_phase1_complete.py

# ML pipeline tests
python test_ml_pipeline.py

# Model training tests
python test_m3_complete.py

# Performance analytics tests
python test_m4_performance.py

# All tests
pytest
```

### Portfolio Manager Tests (S3)
```bash
# Quick sanity tests for S3 Portfolio Manager
python test_portfolio_simple.py
python test_portfolio_manager.py
```

## 🧭 Portfolio Manager (S3) Usage

```python
from robo_trader.portfolio.portfolio_manager import (
    AllocationMethod, MultiStrategyPortfolioManager
)

pm = MultiStrategyPortfolioManager(
    config, risk_manager, allocation_method=AllocationMethod.ADAPTIVE,
    rebalance_frequency="daily", max_strategy_weight=0.4, min_strategy_weight=0.1,
)

# Register strategies (objects with a .name attribute)
pm.register_strategy(ml_enhanced_strategy, initial_weight=0.6)
pm.register_strategy(SimpleNamespace(name="Baseline_SMA"), initial_weight=0.4)

# Update capital and get weights
pm.update_capital(1_000_000)
weights = await pm.allocate_capital()

# Periodically consider rebalancing
if await pm.should_rebalance():
    await pm.rebalance()

# Record performance per strategy
pm.update_strategy_performance("ML_Enhanced", return_pct=0.004,
                               metrics={"trades": 5, "win_rate": 0.6})

# Portfolio metrics
metrics = pm.get_portfolio_metrics()
```

### Test Coverage
- Configuration validation
- Risk management systems
- ML feature generation
- Model training & selection
- Backtesting framework
- Performance analytics
- WebSocket connections
- Async operations

## 📈 Performance Metrics

### System Performance
- **Throughput**: 3x improvement with parallel processing
- **Latency**: Sub-100ms trade execution
- **Feature Generation**: 171 features in ~100ms
- **Model Training**: <1 second per model
- **Backtesting**: 1 year simulation in <5 seconds

### Trading Performance (Target)
- **Sharpe Ratio**: >1.5
- **Max Drawdown**: <10%
- **Win Rate**: >55%
- **Profit Factor**: >1.5

## 🔄 Development Phases

### ✅ Phase 1: Foundation & Quick Wins (Complete)
- Async IBKR client with retry logic
- Pydantic configuration system
- Parallel symbol processing
- Async database operations
- Performance monitoring

### ✅ Phase 2: ML Infrastructure (Complete)
- Feature engineering pipeline
- ML model training & selection
- Walk-forward backtesting
- Performance analytics
- Correlation integration

### 🚧 Phase 3: Advanced Strategies (Current)
- ML-driven strategy framework
- Smart execution algorithms
- Multi-strategy portfolio management
- Microstructure strategies
- Mean reversion suite

### ⏳ Phase 4: Production Hardening
- Advanced risk management
- Production monitoring stack
- Docker deployment
- Security & compliance
- CI/CD pipeline

## 🛡️ Safety Features

### Capital Preservation
- **Paper Trading Default**: Live requires explicit config
- **Pre-Trade Validation**: 10 risk rule checks
- **Emergency Shutdown**: Auto-triggers on violations
- **Position Limits**: Configurable per-trade risk
- **Market Hours**: Only trades during configured hours

### Risk Violations Tracked
1. Daily loss limit
2. Position size limit
3. Leverage limit
4. Correlation limit
5. Sector exposure limit
6. Portfolio heat limit
7. Order notional limit
8. Daily notional limit
9. Volume limit
10. Market cap limit

## 🚦 Production Checklist

Before going live:
- [ ] All tests passing
- [ ] 30+ days profitable paper trading
- [ ] Risk limits configured
- [ ] Monitoring & alerts setup
- [ ] ML models validated
- [ ] Backtesting complete
- [ ] Emergency procedures tested
- [ ] Capital allocation approved

## 🐛 Troubleshooting

### Common Issues

**WebSocket Connection Issues**
```bash
# Set plain text logging for dashboard
export MONITORING_LOG_FORMAT=plain
python app.py
```

**ML Pipeline Errors**
```bash
# Check data availability
python test_ml_pipeline.py
# Verify model files exist
ls trained_models/
```

**IBKR Connection Failed**
```bash
# Use automated diagnostics (RECOMMENDED)
./force_gateway_reconnect.sh
python diagnose_gateway_api.py

# Or manual checks:
# Verify TWS/Gateway running
# Check port (Paper: 4002, Live: 4001)
# Enable API in Gateway settings:
#   File → Global Configuration → API → Settings
#   ☑️ Enable ActiveX and Socket Clients
#   Add 127.0.0.1 to Trusted IPs
# Check for stuck connections: netstat -an | grep 4002
# Clean up zombies: ./START_TRADER.sh
```

**Gateway API Handshake Timeout (2025-10-23)**
- **Symptom:** TCP connection succeeds but API handshake times out
- **Cause:** Gateway API socket clients not enabled or Gateway in bad state
- **Solution:**
  1. Check Gateway API settings (see above)
  2. Run `./force_gateway_reconnect.sh` to test
  3. Use `./START_TRADER.sh` for automatic zombie cleanup
  4. Restart Gateway if needed (requires 2FA)
- **Prevention:** Always use `START_TRADER.sh` for clean startup

**Gateway API layer reported down (2025-11-02 guard)**
- **Symptom:** Worker logs show `Gateway API layer is unresponsive. Manual restart required.`
- **Cause:** Gateway API crashed after a previous disconnect or failed handshake.
- **Solution:** Restart IB Gateway (full exit + login) before trying again.
- **Guardrail:** `robo_trader` now short-circuits retries and surfaces this message instead of looping.
- **Optional:** Set `IBKR_FORCE_DISCONNECT=1` only when you explicitly want to call `ib.disconnect()` (e.g., in isolated tests).

**TWS API Timeout Issues (RESOLVED 2025-09-23)**
- Issue: `patchAsyncio()` caused API handshake timeouts
- Solution: Subprocess-based IBKR operations for async isolation
- Files: `robo_trader/clients/sync_ibkr_wrapper.py`
- Monitor: Check for stuck connections, restart TWS periodically

## 📚 Documentation

- `IMPLEMENTATION_PLAN.md`: Complete development roadmap
- `handoff/LATEST_HANDOFF.md`: Latest session notes
- `CLAUDE.md`: Project guidelines and startup commands
- `REAL_ISSUE_ANALYSIS.md`: Gateway connection troubleshooting
- `START_TRADER.sh`: Automated startup script with zombie cleanup
- `force_gateway_reconnect.sh`: Gateway connectivity testing
- `diagnose_gateway_api.py`: Comprehensive diagnostics
- `CRITICAL_BUG_FIXES_SUMMARY.md`: Summary of recent critical fixes
- `tests/test_critical_bug_fixes.py`: Regression suite for critical fixes

- `IMPLEMENTATION_PLAN.md`: Complete development roadmap
- `handoff/LATEST_HANDOFF.md`: Latest session notes
- `CLAUDE.md`: Project guidelines
- `performance_results/`: Strategy metrics
- `trained_models/`: Saved ML models

### Cloud Dev & CI
- Codespaces setup: `docs/codespaces.md`
- External agent integration: `docs/codex_cloud_integration.md`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Ensure tests pass
4. Submit pull request

## ⚠️ Disclaimer

This software is for educational purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before risking real capital.

## 📞 Support

- GitHub Issues: https://github.com/stimutak/robo_trader/issues
- Documentation: See project docs
- Session Notes: Check `/handoff` folder

---

**System Status**: ✅ TWS Connection Issues Resolved, Phase 3 Continuing
**Version**: 2.1.0 (TWS Connection Fix)
**Last Updated**: 2025-09-23

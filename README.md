# Robo Trader - ML-Driven Production Trading Platform

A production-grade algorithmic trading system with IBKR integration, featuring advanced ML infrastructure, async architecture, and comprehensive risk management. Paper trading by default with strict capital preservation controls.

## 🚀 Current Status

**Phase 2 Complete!** ML Infrastructure & Backtesting framework is fully operational.
- ✅ Phase 1: Foundation & Quick Wins (100% Complete)
- ✅ Phase 2: ML Infrastructure & Backtesting (100% Complete)
- 🚧 Phase 3: Advanced Strategy Development (Starting)
- ⏳ Phase 4: Production Hardening & Deployment

See `IMPLEMENTATION_PLAN.md` for the complete 16-week roadmap.

## ✨ Key Features

### Core Infrastructure (Phase 1 - Complete)
- ✅ **Async IBKR Client**: Non-blocking operations with retry logic
- ✅ **Pydantic Configuration**: Type-safe config with validation
- ✅ **Parallel Processing**: 3x throughput improvement
- ✅ **Async Database**: SQLite with connection pooling
- ✅ **Performance Monitoring**: Real-time metrics dashboard

### ML Infrastructure (Phase 2 - Complete)
- ✅ **Feature Engineering**: 25+ technical indicators with time series
- ✅ **ML Model Training**: RF, XGBoost, LightGBM, Neural Networks
- ✅ **Walk-Forward Backtesting**: Realistic execution simulation
- ✅ **Performance Analytics**: Comprehensive metrics and attribution
- ✅ **Correlation Analysis**: Risk-aware position sizing

### Risk Management
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

```bash
# Always activate virtual environment first
source venv/bin/activate

# Option 1: Run async trading system with parallel processing
python -m robo_trader.runner_async --symbols AAPL,NVDA,TSLA

# Option 2: Run with dashboard (port 5555)
export DASH_PORT=5555
python app.py

# Option 3: Test ML pipeline
python test_ml_pipeline.py
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

# Trading Mode
EXECUTION_MODE=paper         # paper or live
ENVIRONMENT=dev              # dev, staging, or production

# Risk Management
RISK_MAX_POSITION_PCT=0.02   # 2% per position
RISK_MAX_DAILY_LOSS_PCT=0.005 # 0.5% daily stop
RISK_MAX_LEVERAGE=2.0        # Maximum leverage

# ML Configuration
ML_FEATURE_LOOKBACK=60       # Days of history for features
ML_RETRAIN_FREQUENCY=7       # Days between model retraining
ML_MIN_TRAIN_SAMPLES=100     # Minimum samples for training

# Monitoring
MONITORING_LOG_FORMAT=plain   # plain or json
DASH_PORT=5555               # Dashboard port
```

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
# Verify TWS/Gateway running
# Check port (Paper: 7497, Live: 7496)
# Enable API in TWS settings
```

## 📚 Documentation

- `IMPLEMENTATION_PLAN.md`: Complete development roadmap
- `handoff/LATEST_HANDOFF.md`: Latest session notes
- `CLAUDE.md`: Project guidelines
- `performance_results/`: Strategy metrics
- `trained_models/`: Saved ML models

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

**System Status**: ✅ ML Infrastructure Complete, Phase 3 Starting
**Version**: 2.0.0 (Phase 2 Complete)
**Last Updated**: 2025-08-29
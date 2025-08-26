# Robo Trader - Production-Ready Equity Trading System

Safe, testable, risk-managed algorithmic trading system with IBKR integration. Features advanced risk management, async architecture, and comprehensive monitoring. Paper trading by default with strict capital preservation controls.

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

# 5) Run system tests
python test_phase1.py
# All 6 tests should pass
```

### Starting the System

```bash
# Always activate virtual environment first
source venv/bin/activate

# Option 1: Run trading system
python start_ai_trading.py

# Option 2: Run with enhanced dashboard
python app_enhanced.py
# Open browser to http://localhost:5555

# Option 3: Run basic test loop (paper only)
python -m robo_trader.runner
```

## ✨ Key Features

### Phase 1 Complete (Foundation)
- ✅ **Pydantic Configuration**: Type-safe config with validation
- ✅ **Advanced Risk Management**: ATR sizing, portfolio heat, emergency shutdown
- ✅ **Correlation Tracking**: Position correlation limits
- ✅ **Async Architecture**: Event-driven with health monitoring
- ✅ **Structured Logging**: JSON format with audit trail

### Risk Management
- **Position Sizing**: Fixed, ATR-based, or Kelly Criterion
- **Portfolio Heat**: Maximum 6% risk exposure
- **Emergency Shutdown**: Auto-triggers on 5 violations in 5 minutes
- **Correlation Limits**: 0.7 maximum between positions
- **Daily Loss Limits**: 0.5% default stop
- **10 Risk Violation Types**: Comprehensive monitoring

### System Architecture
- **Async Event Loops**: 5 concurrent monitoring tasks
- **Health Checks**: IBKR, database, risk status
- **Graceful Shutdown**: SIGTERM/SIGINT handling
- **Market Hours Aware**: Regular + extended hours
- **Auto-Recovery**: Exponential backoff retry logic

## 📁 Project Structure

```
robo_trader/
├── robo_trader/
│   ├── config.py              # Pydantic configuration with validation
│   ├── risk.py                # Advanced risk management
│   ├── correlation.py         # Correlation tracking
│   ├── logger.py              # Structured JSON logging
│   ├── core/
│   │   ├── __init__.py
│   │   └── engine.py          # Async trading engine
│   ├── ibkr_client.py         # IBKR connection wrapper
│   ├── execution.py           # Order execution
│   ├── portfolio.py           # Portfolio tracking
│   ├── strategies.py          # Trading strategies
│   ├── strategy_manager.py    # Multi-strategy orchestration
│   ├── database.py            # SQLite data persistence
│   └── runner.py              # Main entry point
├── tests/
│   ├── test_*.py              # Unit tests
│   └── test_phase1.py         # Integration test suite
├── venv/                      # Virtual environment (created)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your settings:

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
RISK_MIN_VOLUME=1000000      # Minimum daily volume
RISK_POSITION_SIZING=atr     # fixed, atr, or kelly

# Monitoring
MONITORING_LOG_FORMAT=json   # json or text
MONITORING_LOG_LEVEL=INFO    # DEBUG, INFO, WARNING, ERROR

# Symbols to Trade
SYMBOLS=AAPL,MSFT,SPY,QQQ
DEFAULT_CASH=100000          # Starting cash for paper trading
```

### Configuration Validation

The system validates all configuration on startup:
- Paper mode requires paper trading ports
- Production environment requires alerts enabled
- Risk limits must be within acceptable ranges

## 🧪 Testing

### Run Complete Test Suite
```bash
source venv/bin/activate
python test_phase1.py
```

### Test Categories
1. **Configuration**: Pydantic validation
2. **Risk Management**: Position sizing, violations
3. **Correlation Tracking**: Matrix calculations
4. **Structured Logging**: JSON output, censoring
5. **Trading Engine**: Async operations
6. **Backward Compatibility**: Legacy support

All tests must pass before running the system.

## 📊 Monitoring

### Dashboard
```bash
python app_enhanced.py
```
Visit http://localhost:5555 for:
- Real-time price charts
- Portfolio P&L tracking
- Risk metrics display
- AI market analysis
- Trade execution log

### Logs
JSON structured logs include:
- Trade execution details
- Risk violations
- Performance metrics
- System health status

Example log entry:
```json
{
  "timestamp": "2025-08-25T20:00:00.000Z",
  "event": "trade.executed",
  "symbol": "AAPL",
  "quantity": 100,
  "price": 150.00,
  "side": "BUY",
  "notional": 15000,
  "strategy": "momentum"
}
```

## 🛡️ Safety Features

### Capital Preservation
- **Paper Trading Default**: Live trading requires explicit configuration
- **Pre-Trade Validation**: Every order checked against 10 risk rules
- **Emergency Shutdown**: Automatic on critical violations
- **Position Limits**: Max 20 positions, 2% risk per trade
- **Daily Loss Limits**: Trading stops at 0.5% loss

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

## 🔄 System States

### Market Hours
- **Regular**: 9:30 AM - 4:00 PM EST
- **Extended**: 4:00 AM - 8:00 PM EST (paper only)
- **Closed**: Weekends and holidays

### Engine States
- `INITIALIZING`: Starting up
- `READY`: Initialized, waiting to start
- `RUNNING`: Active trading
- `PAUSED`: Temporarily suspended
- `STOPPING`: Graceful shutdown
- `STOPPED`: Fully stopped
- `ERROR`: Error state

## 📈 Trading Strategies

### Available Strategies
1. **SMA Crossover**: Simple moving average signals
2. **Momentum**: Price momentum with volume
3. **Mean Reversion**: RSI oversold bounce
4. **Breakout**: High-tight flag patterns

### Adding Custom Strategies
1. Extend base strategy class
2. Implement signal generation
3. Register with strategy manager
4. Backtest before deployment

## 🚦 Production Checklist

Before going live:
- [ ] All tests passing (6/6)
- [ ] Paper trading profitable for 30+ days
- [ ] Risk limits configured appropriately
- [ ] Monitoring and alerts configured
- [ ] Emergency contacts established
- [ ] Backup and recovery procedures tested
- [ ] Capital allocation approved

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**IBKR Connection Failed**
```bash
# Check TWS/Gateway is running
# Verify port numbers (Paper: 7497, Live: 7496)
# Enable API connections in TWS settings
```

**Tests Failing**
```bash
# Clean Python cache
find . -type d -name __pycache__ -exec rm -rf {} +
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

## 📚 Documentation

- `IMPLEMENTATION_NOTES.md`: Technical implementation details
- `PHASE1_COMPLETE.md`: Phase 1 completion summary
- `COMMIT_NOTES.md`: Git commit guidelines
- `docs/`: Additional documentation
- `handoff/`: Session handoff documents

## 🔮 Roadmap

### ✅ Phase 1: Foundation (Complete)
- Pydantic configuration
- Advanced risk management
- Async architecture
- Structured logging

### 🔄 Phase 2: Data Pipeline (Next)
- Real-time tick streaming
- Technical indicators
- Feature engineering
- Data validation

### 📅 Phase 3: Strategy Framework
- Backtesting engine
- Strategy optimization
- Walk-forward analysis
- Performance attribution

### 📅 Phase 4: Production Hardening
- Monitoring dashboard
- Alerting system
- Docker deployment
- Performance optimization

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Ensure tests pass
4. Submit pull request

## ⚖️ License

Private repository - all rights reserved

## ⚠️ Disclaimer

This software is for educational purposes. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly with paper trading before risking real capital.

## 📞 Support

- GitHub Issues: https://github.com/stimutak/robo_trader/issues
- Documentation: See `/docs` folder
- Logs: Check `/handoff` for session notes

---

**System Status**: ✅ Production-Ready Foundation
**Version**: 1.0.0 (Phase 1 Complete)
**Last Updated**: 2025-08-25
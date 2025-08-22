# Robo Trader - Action Plan & Next Steps

## Current Status (Aug 22, 2025)
✅ **Completed**: Comprehensive LLM transformation with decisive prompt
🟡 **Running**: System operational with schema validation issues
📊 **Dashboard**: Active at http://localhost:5555
💾 **Database**: Tracking decisions (10+ saved)

## Immediate Actions (Today/Tomorrow)

### 1. Fix Schema Validation Issues 🔴 PRIORITY
**Problem**: Claude is responding but field names don't match our schema exactly
**Solution Options**:
- [ ] Simplify schema to match Claude's natural output
- [ ] Add prompt engineering to force exact field names
- [ ] Create adapter layer to map Claude's fields to our schema
- [ ] Consider using OpenAI's function calling instead

### 2. Test New Risk Controls 🟡
- [ ] Verify ATR-based sizing is working correctly
- [ ] Test daily/weekly drawdown limits
- [ ] Confirm liquidity checks are blocking illiquid symbols
- [ ] Validate correlation bucket tracking

### 3. Monitor First 100 Decisions 📊
- [ ] Track conviction distribution
- [ ] Calculate initial Brier scores
- [ ] Identify if prompt is too conservative/aggressive
- [ ] Tune aggressiveness level based on results

## Week 1 Actions

### Performance Monitoring
- [ ] Set up calibration dashboard page
- [ ] Add real-time Brier score tracking
- [ ] Create P&L attribution by decision type
- [ ] Implement alerts for degraded calibration

### Integration Improvements
- [ ] Update `ai_runner.py` to use new LLM client directly
- [ ] Add market metadata fetching for all symbols
- [ ] Implement correlation tracking in real-time
- [ ] Connect EV calculations to position sizing

### Testing & Validation
- [ ] Run full test suite: `pytest tests/test_llm_system.py -v`
- [ ] Paper trade for 5 full trading days
- [ ] Document all trades and decisions
- [ ] Compare old vs new system performance

## Week 2-3 Actions

### Optimization
- [ ] Tune conviction thresholds based on results
- [ ] Adjust aggressiveness for different market regimes
- [ ] Optimize sector mappings for correlation control
- [ ] Fine-tune EV calculation parameters

### Backtesting
- [ ] Build backtest framework using same risk code
- [ ] Test on recent Fed meetings
- [ ] Validate on earnings seasons
- [ ] Compare with buy-and-hold baseline

### Enhanced Features
- [ ] Add options-specific position sizing
- [ ] Implement regime detection (trending/ranging/volatile)
- [ ] Create multi-strategy ensemble
- [ ] Add performance attribution analytics

## Month 1 Goals

### Calibration Targets
- [ ] Achieve Brier score ≤ 0.20
- [ ] Reach 40%+ trade rate (decisions resulting in trades)
- [ ] Maintain 0 liquidity violations
- [ ] Keep correlation exposures under control

### Performance Metrics
- [ ] Generate positive returns in paper trading
- [ ] Beat SPY benchmark on risk-adjusted basis
- [ ] Maintain Sharpe ratio > 1.0
- [ ] Keep max drawdown < 10%

### Production Readiness
- [ ] Complete 30-day paper trading validation
- [ ] Document all edge cases and failure modes
- [ ] Create operational runbook
- [ ] Set up monitoring and alerting

## Future Enhancements

### Advanced Event Impact Analysis 🎯
- [ ] **Primary Impact Detection**: Analyze major news events to identify most directly affected companies
  - Fed rate decisions → Banks (JPM, BAC), REITs (O, SPG), Utilities (NEE, SO)
  - Oil price moves → Energy (XOM, CVX), Airlines (DAL, UAL), Shipping (FDX, UPS)
  - Regulatory changes → Affected sectors and specific companies
  
- [ ] **Collateral Impact Mapping**: Identify partner/supplier chain effects
  - Apple news → Suppliers (QCOM, TSM, SWKS), Competitors (GOOGL, MSFT)
  - Tesla news → Battery suppliers (ALBM, LAC), Charging networks (CHPT, BLNK)
  - AI breakthroughs → Chip makers (NVDA, AMD), Cloud providers (AMZN, MSFT)
  
- [ ] **Daily Impact Summary**: Generate end-of-day report connecting:
  - Top news events of the day
  - Companies most affected (with conviction scores)
  - Expected multi-day ripple effects
  - Correlation clusters to watch
  
- [ ] **Supply Chain Intelligence**: Build knowledge graph of relationships
  - Customer/supplier relationships
  - Partnership agreements
  - Competitive dynamics
  - Sector interdependencies
  
- [ ] **Event Magnitude Scoring**: Quantify expected price impact
  - Historical event impact database
  - Sector-specific multipliers
  - Market regime adjustments
  - Volatility scaling

## Technical Debt to Address

### Code Quality
- [ ] Add comprehensive docstrings to new modules
- [ ] Increase test coverage to >80%
- [ ] Remove deprecated code paths
- [ ] Standardize error handling

### Infrastructure
- [ ] Set up proper log rotation
- [ ] Implement database backups
- [ ] Add health checks and monitoring
- [ ] Create deployment automation

### Documentation
- [ ] Write API documentation for new modules
- [ ] Create trader's guide for the system
- [ ] Document all configuration options
- [ ] Add troubleshooting guide

## Risk Mitigation

### Before Live Trading
1. **Mandatory 30-day paper validation**
2. **Calibration metrics must meet targets**
3. **Risk controls tested under stress**
4. **Backup plans for all failure modes**
5. **Capital allocation strategy defined**

### Safety Checklist
- [ ] Panic button implemented and tested
- [ ] Daily notional limits enforced
- [ ] Stop loss validation working
- [ ] Drawdown brakes trigger correctly
- [ ] Live trading requires explicit confirmation

## Configuration Tuning Guide

### Aggressiveness Levels
- **Level 0**: Very conservative (60+ conviction required)
- **Level 1**: Balanced (55+ conviction) - CURRENT
- **Level 2**: Assertive (52+ conviction)
- **Level 3**: Opportunistic (50+ conviction)

### Key Parameters to Monitor
```bash
AGGRESSIVENESS_LEVEL     # Start at 1, adjust based on results
CONVICTION_THRESHOLD     # Lower if too few trades
PER_TRADE_RISK_BPS      # Increase carefully if edge proven
MIN_ADV                 # May need to lower for small caps
MAX_BUCKET_EXPOSURE_PCT # Adjust based on diversification needs
```

## Success Metrics

### Week 1
- 20+ decisions logged
- Schema validation fixed
- No system crashes
- Dashboard showing decisions

### Month 1
- 100+ decisions analyzed
- Positive paper trading P&L
- Brier score < 0.25
- Clear edge identified

### Quarter 1
- Consistent profitability
- Sharpe > 1.5
- Ready for small live allocation
- Fully calibrated system

## Resources & Support

### Documentation
- Anthropic API: https://docs.anthropic.com
- IB API: https://interactivebrokers.github.io
- Project: `/handoff/LATEST_HANDOFF.md`

### Monitoring
- Dashboard: http://localhost:5555
- Logs: `tail -f ai_trading.log`
- Database: `sqlite3 trading.db`

### Emergency Procedures
1. Stop trading: `pkill -f start_ai_trading`
2. Check logs: `tail -100 ai_trading.log`
3. Review decisions: `sqlite3 trading.db "SELECT * FROM llm_decisions ORDER BY timestamp DESC LIMIT 10;"`
4. Restart: `./restart_trading.sh`

---
*Last Updated: August 22, 2025*
*Next Review: August 29, 2025*
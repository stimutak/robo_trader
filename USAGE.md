# Robo Trader - Usage Guide

## \ud83c\udf89 NEW: Web Dashboard (Easiest Way!)

### 1. Start the Dashboard
```bash
# Just run this one command:
python app.py
```

### 2. Open Your Browser
Go to: **http://localhost:5555**

### 3. Start Trading
- Click the big green **START TRADING** button
- Watch AI analyze markets in real-time
- Monitor your P&L and positions
- Click **STOP TRADING** when done

That's it! No more command line needed.

## Prerequisites

### TWS Setup (One Time)
1. Open TWS (Trader Workstation)
2. Login with **Paper Trading** selected
3. Enable API: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Add 127.0.0.1 to Trusted IPs
4. Your .env file should have:
```bash
ANTHROPIC_API_KEY=sk-ant-...  # Your Claude API key
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
```

## Alternative: Command Line Usage
```bash
# Run with default settings
python -m robo_trader.runner

# With custom parameters
python -m robo_trader.runner \
  --symbols SPY,QQQ,TSLA \
  --duration "5 D" \
  --bar-size "5 mins"
```

## 🧠 Using the Intelligence Layer

### Testing Claude Analysis
```bash
# Test Claude's market analysis capabilities
python test_claude.py

# Test sentiment analysis
python test_sentiment.py
```

### Manual Trading with AI Analysis

Create a script like this:

```python
import asyncio
from robo_trader.intelligence import ClaudeTrader, KellyCriterion
from robo_trader.sentiment import SimpleSentimentAnalyzer

async def analyze_and_trade():
    # Initialize AI
    claude = ClaudeTrader()
    sentiment = SimpleSentimentAnalyzer()
    
    # Example: Breaking news
    news = "Tesla announces record Q4 deliveries, beating estimates by 10%"
    
    # Quick sentiment check (instant, free)
    quick_sentiment = sentiment.analyze(news)
    print(f"Quick sentiment: {quick_sentiment.sentiment} ({quick_sentiment.confidence:.0%} confidence)")
    
    # If high-impact, get deep analysis from Claude
    if sentiment.is_high_impact(news):
        market_data = {
            "price": 250.50,
            "volume": 120_000_000,
            "rsi": 65,
            "support": 245,
            "resistance": 255
        }
        
        # Get Claude's analysis (~10-15 seconds, ~$0.02)
        signal = await claude.analyze_market_event(
            event_text=news,
            symbol="TSLA",
            market_data=market_data
        )
        
        print(f"\nClaude's Analysis:")
        print(f"Direction: {signal['direction']}")
        print(f"Conviction: {signal['conviction']}%")
        print(f"Entry: ${signal.get('entry_price', 'N/A')}")
        print(f"Stop Loss: ${signal.get('stop_loss', 'N/A')}")
        print(f"Take Profit: ${signal.get('take_profit', 'N/A')}")
        print(f"Rationale: {signal['rationale'][:200]}...")
        
        # Calculate position size
        if signal['conviction'] >= 50:
            position_size = KellyCriterion.size_from_conviction(signal['conviction'])
            print(f"Recommended Position: {position_size*100:.1f}% of portfolio")
            
            # TODO: Execute trade via IB API
            # if signal['direction'] == 'bullish':
            #     place_order(symbol, position_size, signal['entry_price'])

asyncio.run(analyze_and_trade())
```

## 📰 News-Driven Trading (Coming Soon)

### Phase 1: Manual News Analysis
```python
# Analyze specific events manually
from robo_trader.intelligence import ClaudeTrader

claude = ClaudeTrader()

# Fed announcement
await claude.analyze_fed_event(
    "Fed keeps rates unchanged, signals 3 cuts in 2024",
    current_rate=5.5,
    rate_expectations="Market expected pause"
)

# Earnings
await claude.analyze_earnings(
    symbol="AAPL",
    actual_eps=1.46,
    expected_eps=1.39,
    actual_rev=89.5e9,
    expected_rev=89.3e9,
    guidance="Slightly below expectations"
)
```

### Phase 2: Automated News Pipeline (TODO)
```python
# This will be implemented next
from robo_trader.news import NewsMonitor
from robo_trader.events import EventProcessor

monitor = NewsMonitor(symbols=['SPY', 'AAPL', 'TSLA'])
processor = EventProcessor(claude_trader)

# Monitor news feeds
await monitor.start()
# Process high-impact events
await processor.run()
```

## 🎯 Trading Strategies

### Current Strategies
1. **SMA Crossover** (Basic, no AI)
   - Simple moving average crossover
   - Configurable windows (--sma-fast, --sma-slow)

### AI-Enhanced Strategies (In Development)
1. **Event-Driven Trading**
   - Fed announcements
   - Earnings reports
   - Breaking news

2. **Sentiment-Based Trading**
   - News sentiment analysis
   - Social media monitoring
   - Market regime detection

3. **Smart Money Following**
   - Options flow analysis
   - Institutional positioning
   - Dark pool activity

## 📊 Performance Monitoring

### View Logs
```bash
# Check trading decisions
tail -f robo_trader.log  # (when logging to file is enabled)

# Check specific module
python -c "from robo_trader.portfolio import Portfolio; print(Portfolio.__doc__)"
```

### Test Strategies
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_intelligence.py -v

# Check test coverage
pytest --cov=robo_trader
```

## ⚙️ Configuration

### Risk Management (.env)
```bash
# Position limits
MAX_POSITION_RISK_PCT=0.02  # 2% per position
MAX_SYMBOL_EXPOSURE_PCT=0.10  # 10% per symbol
MAX_DAILY_LOSS=0.03  # 3% daily loss limit

# Order limits
MAX_ORDER_NOTIONAL=10000  # Max $10k per order
MAX_DAILY_NOTIONAL=50000  # Max $50k per day
```

### AI Configuration
```bash
# Claude settings (in code)
temperature=0.3  # Lower = more consistent
max_tokens=2000  # Response length

# Conviction thresholds
MIN_CONVICTION=50  # Don't trade below 50%
HIGH_CONVICTION=70  # Increase position size
```

## 🚨 Safety Features

### Paper Trading (Default)
- Always starts in paper mode
- No real money at risk
- Test strategies safely

### Live Trading Gates
```bash
# Requires ALL of these:
TRADING_MODE=live  # In .env
--confirm-live  # CLI flag
# Plus 30 days paper trading history
```

### Risk Limits
- Max 10% position size (Kelly-limited)
- Max 3% daily loss
- Stop losses on all trades
- Automatic position sizing based on conviction

## 📈 Typical Workflow

### Daily Trading
1. **Morning Prep**
   ```bash
   # Check market events
   python test_claude.py  # Test with today's news
   ```

2. **Run Trading Bot**
   ```bash
   # Paper trade with AI signals
   python -m robo_trader.runner --symbols SPY,QQQ,AAPL
   ```

3. **Monitor Performance**
   ```bash
   # Check positions and P&L
   # (Portfolio tracking in progress)
   ```

### Event Trading
1. **Major Announcement** (Fed, Earnings)
2. **Bot detects via news feed** (or manual input)
3. **Sentiment pre-filter** (<1ms)
4. **Claude deep analysis** (10-15s) if high-impact
5. **Generate trading signal**
6. **Size position via Kelly**
7. **Execute with risk controls**

## 🔧 Troubleshooting

### Common Issues

**Claude API Errors**
```bash
# Check API key
echo $ANTHROPIC_API_KEY

# Test connection
python -c "from robo_trader.intelligence import ClaudeTrader; ct = ClaudeTrader()"
```

**IB Connection Issues**
```bash
# Verify IB Gateway is running
# Check port (7497 for paper, 7496 for live)
# Ensure clientId is unique
```

**Low Conviction Signals**
- Normal for unclear events
- Bot only trades >50% conviction
- Better to skip than force trades

## 📚 Further Documentation

- `CLAUDE.md` - Project philosophy and development guidelines
- `PROJECT_PLAN.md` - Development roadmap
- `TODO.md` - Current tasks and priorities
- `CLAUDE_TRADING_PROMPT.md` - AI prompt engineering details

## 🎓 Examples

### Example 1: Fed Day Trading
```python
# When: FOMC announcement days
# Strategy: Trade SPY based on Fed tone
# Risk: 2-5% of portfolio
# Timeframe: Minutes to hours
```

### Example 2: Earnings Plays
```python
# When: After market earnings
# Strategy: Fade overreactions or ride momentum
# Risk: 1-3% of portfolio  
# Timeframe: Next day to few days
```

### Example 3: News Arbitrage
```python
# When: Breaking news hits
# Strategy: Trade before full market reaction
# Risk: 1-2% of portfolio
# Timeframe: Minutes to hours
```

## 🚀 Next Steps

1. **Complete news pipeline** for automated event detection
2. **Integrate AI with runner.py** for live signals
3. **Add backtesting** on historical events
4. **Build performance dashboard**
5. **Add more trading strategies**

---

For questions or issues, check the logs or run tests to diagnose problems. The bot is designed to be conservative - it's better to miss trades than lose money on bad signals.
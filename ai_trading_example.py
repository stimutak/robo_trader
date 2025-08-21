#!/usr/bin/env python3
"""
Example of AI-powered trading with Claude and IB
This shows how to use the intelligence layer for real trading decisions
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from robo_trader.intelligence import ClaudeTrader, KellyCriterion
from robo_trader.sentiment import SimpleSentimentAnalyzer, NewsFilter
from robo_trader.ibkr_client import IBKRClient
from robo_trader.config import load_config
from robo_trader.risk import RiskManager
from robo_trader.execution import PaperExecutor, Order
from robo_trader.portfolio import Portfolio
from robo_trader.logger import get_logger

# Load environment variables
load_dotenv()
logger = get_logger(__name__)


async def get_market_data(ib_client: IBKRClient, symbol: str) -> Dict[str, Any]:
    """Fetch current market data for a symbol"""
    try:
        # Get recent bars for technical indicators
        df = await ib_client.fetch_recent_bars(symbol, duration="1 D", bar_size="5 mins")
        
        if df.empty:
            return {}
        
        current_price = float(df['close'].iloc[-1])
        volume = float(df['volume'].sum())
        
        # Simple technical indicators
        close_prices = df['close'].values
        
        # RSI calculation (simplified)
        gains = [close_prices[i] - close_prices[i-1] for i in range(1, len(close_prices)) if close_prices[i] > close_prices[i-1]]
        losses = [close_prices[i-1] - close_prices[i] for i in range(1, len(close_prices)) if close_prices[i] < close_prices[i-1]]
        
        avg_gain = sum(gains) / 14 if len(gains) > 0 else 0
        avg_loss = sum(losses) / 14 if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # Support and resistance (simplified - using recent highs/lows)
        support = float(df['low'].min())
        resistance = float(df['high'].max())
        
        # Price change
        price_change_pct = ((current_price - float(df['close'].iloc[0])) / float(df['close'].iloc[0])) * 100
        
        return {
            "price": current_price,
            "volume": volume,
            "avg_volume": volume,  # Would need historical avg
            "price_change_pct": round(price_change_pct, 2),
            "rsi": round(rsi, 0),
            "support": round(support, 2),
            "resistance": round(resistance, 2)
        }
    except Exception as e:
        logger.error(f"Failed to get market data for {symbol}: {e}")
        return {}


async def analyze_and_trade_event(
    event_text: str,
    symbol: str,
    claude: ClaudeTrader,
    ib_client: IBKRClient,
    risk_mgr: RiskManager,
    executor: PaperExecutor,
    portfolio: Portfolio
):
    """Analyze an event and potentially trade on it"""
    
    print(f"\n{'='*60}")
    print(f"Analyzing Event for {symbol}")
    print(f"{'='*60}")
    print(f"Event: {event_text[:100]}...")
    
    # Get current market data
    market_data = await get_market_data(ib_client, symbol)
    if not market_data:
        print("❌ Failed to get market data")
        return
    
    print(f"Current Price: ${market_data['price']:.2f}")
    print(f"RSI: {market_data['rsi']:.0f}")
    print(f"Support/Resistance: ${market_data['support']:.2f} / ${market_data['resistance']:.2f}")
    
    # Get Claude's analysis
    print("\n🤖 Getting Claude's analysis...")
    signal = await claude.analyze_market_event(
        event_text=event_text,
        symbol=symbol,
        market_data=market_data
    )
    
    print(f"\n📊 Analysis Results:")
    print(f"Direction: {signal.get('direction', 'neutral')}")
    print(f"Conviction: {signal.get('conviction', 0)}%")
    print(f"Entry Price: ${signal.get('entry_price', market_data['price']):.2f}")
    print(f"Stop Loss: ${signal.get('stop_loss', 0):.2f}")
    print(f"Take Profit: ${signal.get('take_profit', 0):.2f}")
    print(f"Rationale: {signal.get('rationale', 'No rationale')[:200]}...")
    
    # Check if we should trade
    if signal.get('conviction', 0) < 50:
        print("\n⚠️ Conviction too low (<50%), skipping trade")
        return
    
    if signal.get('direction') == 'neutral':
        print("\n⚠️ Neutral signal, no trade")
        return
    
    # Calculate position size
    conviction = signal.get('conviction', 50)
    position_size_pct = KellyCriterion.size_from_conviction(conviction)
    
    equity = portfolio.equity({symbol: market_data['price']})
    position_value = equity * position_size_pct
    shares = int(position_value / market_data['price'])
    
    print(f"\n💰 Position Sizing:")
    print(f"Portfolio Equity: ${equity:,.2f}")
    print(f"Position Size: {position_size_pct*100:.1f}% (${position_value:,.2f})")
    print(f"Shares to Trade: {shares}")
    
    if shares == 0:
        print("❌ Position too small, skipping")
        return
    
    # Risk check
    side = "BUY" if signal['direction'] == 'bullish' else "SELL"
    positions = {}  # Would need actual positions
    daily_pnl = portfolio.realized_pnl
    daily_notional = 0  # Would need to track
    
    ok, msg = risk_mgr.validate_order(
        symbol, shares, market_data['price'], 
        equity, daily_pnl, positions, daily_notional
    )
    
    if not ok:
        print(f"\n❌ Risk check failed: {msg}")
        return
    
    # Execute trade (paper only for now)
    print(f"\n📝 Placing {side} order for {shares} shares of {symbol}")
    
    order = Order(
        symbol=symbol,
        quantity=shares,
        side=side,
        price=signal.get('entry_price', market_data['price'])
    )
    
    result = executor.place_order(order)
    
    if result.ok:
        print(f"✅ Order executed at ${result.fill_price:.2f}")
        
        # Update portfolio
        portfolio.update_fill(symbol, side, shares, result.fill_price)
        
        # Show stop loss and take profit
        if signal.get('stop_loss'):
            print(f"🛑 Stop Loss set at ${signal['stop_loss']:.2f}")
        if signal.get('take_profit'):
            print(f"🎯 Take Profit set at ${signal['take_profit']:.2f}")
    else:
        print(f"❌ Order failed: {result.error}")
    
    print(f"\n{'='*60}\n")


async def main():
    """Main trading loop with AI analysis"""
    
    print("🤖 AI-Powered Trading Example")
    print("=" * 60)
    
    # Load configuration
    cfg = load_config()
    
    # Initialize components
    print("Initializing components...")
    
    # IB connection
    ib = IBKRClient(cfg.ibkr_host, cfg.ibkr_port, cfg.ibkr_client_id)
    await ib.connect(readonly=True)
    print("✅ Connected to IB")
    
    # AI components
    claude = ClaudeTrader()
    sentiment = SimpleSentimentAnalyzer()
    news_filter = NewsFilter(['SPY', 'AAPL', 'TSLA', 'QQQ'])
    print("✅ AI systems initialized")
    
    # Trading components
    risk_mgr = RiskManager(
        max_daily_loss=cfg.max_daily_loss,
        max_position_risk_pct=cfg.max_position_risk_pct,
        max_symbol_exposure_pct=cfg.max_symbol_exposure_pct,
        max_leverage=cfg.max_leverage
    )
    executor = PaperExecutor()
    portfolio = Portfolio(cfg.default_cash)
    print("✅ Trading systems initialized")
    
    # Example events to analyze
    events = [
        {
            "symbol": "SPY",
            "text": "Federal Reserve leaves interest rates unchanged at 5.25-5.50% as widely expected. "
                   "Chair Powell indicates the Fed is likely done raising rates and sees three rate cuts in 2024. "
                   "Markets initially rise on the dovish tone."
        },
        {
            "symbol": "AAPL",
            "text": "Apple reports Q1 earnings: EPS $2.18 vs $2.10 expected, Revenue $119.6B vs $117.9B expected. "
                   "iPhone revenue up 6% YoY. However, company provides weaker than expected guidance for next quarter "
                   "citing ongoing challenges in China market."
        },
        {
            "symbol": "TSLA",
            "text": "Tesla announces plans to cut prices on Model 3 and Model Y by up to 6% in the US market. "
                   "This is the fourth price cut this year as the company aims to boost demand amid increasing competition."
        }
    ]
    
    # Process each event
    for event in events:
        # Quick sentiment check first
        quick_sentiment = sentiment.analyze(event['text'])
        print(f"\n🔍 Quick Sentiment for {event['symbol']}: {quick_sentiment.sentiment} "
              f"(confidence: {quick_sentiment.confidence:.0%})")
        
        # Check if high impact
        if sentiment.is_high_impact(event['text']) or quick_sentiment.confidence > 0.3:
            await analyze_and_trade_event(
                event['text'],
                event['symbol'],
                claude,
                ib,
                risk_mgr,
                executor,
                portfolio
            )
        else:
            print(f"⚠️ Low impact event for {event['symbol']}, skipping deep analysis")
    
    # Final portfolio summary
    print("\n" + "="*60)
    print("📊 Portfolio Summary")
    print("="*60)
    print(f"Starting Cash: ${cfg.default_cash:,.2f}")
    print(f"Current Equity: ${portfolio.equity({}):,.2f}")
    print(f"Realized P&L: ${portfolio.realized_pnl:,.2f}")
    print(f"Unrealized P&L: ${portfolio.unrealized_pnl:,.2f}")
    print(f"Total P&L: ${portfolio.total_pnl:,.2f}")
    
    # Disconnect
    ib.disconnect()
    print("\n✅ Done! (This was paper trading only)")


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY not found in environment")
        print("Please add it to your .env file")
        exit(1)
    
    # Run the example
    asyncio.run(main())
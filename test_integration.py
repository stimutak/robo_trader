#!/usr/bin/env python3
"""
Integration test for the complete AI trading pipeline.
Tests: News → Sentiment → Claude AI → Signal → Kelly Sizing
"""

import asyncio
import os
from datetime import datetime, timezone
from robo_trader.news import NewsAggregator
from robo_trader.intelligence import ClaudeTrader
from robo_trader.events import EventProcessor, NewsEvent, EventType
from robo_trader.kelly import KellyCalculator
from robo_trader.logger import get_logger

logger = get_logger(__name__)


async def test_pipeline():
    """Test the complete trading intelligence pipeline."""
    
    print("\n" + "="*60)
    print("🤖 ROBO TRADER - AI PIPELINE INTEGRATION TEST")
    print("="*60)
    
    # 1. Initialize components
    print("\n📦 Initializing components...")
    
    symbols = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]
    
    # News aggregator
    news_agg = NewsAggregator(symbols, lookback_hours=12)
    print("✓ News aggregator ready")
    
    # AI trader (check for API key)
    ai_trader = None
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            ai_trader = ClaudeTrader()
            print("✓ Claude AI ready")
        except Exception as e:
            print(f"⚠️  Claude AI not available: {e}")
    else:
        print("⚠️  No ANTHROPIC_API_KEY found - AI analysis disabled")
    
    # Kelly calculator
    kelly_calc = KellyCalculator(capital=100000)
    print("✓ Kelly calculator ready")
    
    # Event processor
    class MockRiskManager:
        pass
    
    event_processor = EventProcessor(
        symbols=symbols,
        risk_manager=MockRiskManager(),
        ai_trader=ai_trader,
        news_aggregator=news_agg
    )
    print("✓ Event processor ready")
    
    # 2. Fetch news
    print("\n📰 Fetching latest market news...")
    new_count, total_count = await news_agg.update()
    print(f"✓ Fetched {new_count} news items ({total_count} total)")
    
    # 3. Get high-impact news
    high_impact = news_agg.get_high_impact_news(min_relevance=0.3)
    print(f"\n🎯 Found {len(high_impact)} high-impact news items")
    
    if high_impact:
        # Show top 3
        for i, item in enumerate(high_impact[:3], 1):
            age = datetime.now(timezone.utc) - item.published
            age_str = f"{int(age.total_seconds() / 60)}m" if age.total_seconds() < 3600 else f"{int(age.total_seconds() / 3600)}h"
            print(f"\n{i}. [{item.source}] {age_str} ago")
            print(f"   📰 {item.title[:80]}...")
            print(f"   🎯 Relevance: {item.relevance_score:.2f} | Sentiment: {item.sentiment_score:+.2f}")
            if item.symbols:
                print(f"   📊 Symbols: {', '.join(item.symbols)}")
    
    # 4. Process through AI (if available)
    if ai_trader and high_impact:
        print("\n🤖 Analyzing with Claude AI...")
        
        # Take the most relevant news item
        top_news = high_impact[0]
        
        # Create news event
        news_event = NewsEvent(
            news_item=top_news,
            requires_analysis=True
        )
        
        # Process it
        await event_processor._handle_news(news_event)
        
        # Check if any signals were generated
        if event_processor.event_queue.size() > 0:
            print("✓ AI generated trading signals!")
            
            # Process signals
            while event_processor.event_queue.size() > 0:
                event = event_processor.event_queue.pop()
                if event.event_type == EventType.SIGNAL:
                    print(f"\n📈 SIGNAL: {event.data['signal']} {event.data['symbol']}")
                    print(f"   Conviction: {event.data['conviction']:.0%}")
                    print(f"   Reasoning: {event.data['reasoning'][:100]}...")
                    
                    # Calculate position size
                    shares, value = kelly_calc.calculate_position_size(
                        conviction=event.data['conviction'],
                        expected_return=0.03,  # 3% expected move
                        current_price=100.0
                    )
                    
                    if shares > 0:
                        print(f"   📊 Position: {shares} shares (${value:,.2f})")
        else:
            print("ℹ️  No trading signals generated (conviction too low or no actionable news)")
    
    # 5. Show event statistics
    print("\n📊 Event Processing Statistics:")
    stats = event_processor.get_event_stats()
    print(f"   Queue size: {stats['queue_size']}")
    print(f"   Processed: {stats['processed_total']}")
    print(f"   History: {stats['history_size']} events")
    
    # 6. Test Kelly sizing with different scenarios
    print("\n💰 Kelly Position Sizing Examples:")
    
    scenarios = [
        ("High conviction news", 0.75, 0.04),
        ("Medium conviction", 0.60, 0.03),
        ("Low conviction", 0.55, 0.02),
    ]
    
    for desc, conviction, expected_return in scenarios:
        shares, value = kelly_calc.calculate_position_size(
            conviction=conviction,
            expected_return=expected_return,
            current_price=100.0
        )
        
        if shares > 0:
            print(f"   {desc}: {shares} shares (${value:,.2f}) at {conviction:.0%} conviction")
        else:
            print(f"   {desc}: No position (conviction {conviction:.0%} too low)")
    
    print("\n" + "="*60)
    print("✅ INTEGRATION TEST COMPLETE")
    print("="*60)
    print("\n💡 Next steps:")
    print("   1. Connect to live IB data feed")
    print("   2. Enable in web dashboard")
    print("   3. Start paper trading with AI signals")
    print("   4. Monitor performance and adjust")


async def main():
    """Run the integration test."""
    try:
        await test_pipeline()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
Test script for Claude trading integration
"""

import asyncio
import os
from dotenv import load_dotenv
from robo_trader.intelligence import ClaudeTrader, KellyCriterion

# Load environment variables
load_dotenv()

async def test_market_analysis():
    """Test Claude's market event analysis"""
    print("=" * 60)
    print("Testing Claude Trading Integration")
    print("=" * 60)
    
    # Initialize Claude
    try:
        claude = ClaudeTrader()
        print("✅ Claude initialized successfully\n")
    except Exception as e:
        print(f"❌ Failed to initialize Claude: {e}")
        return
    
    # Test 1: Fed announcement
    print("Test 1: Analyzing Fed Announcement")
    print("-" * 40)
    
    fed_event = """
    Federal Reserve keeps interest rates unchanged at 5.25-5.50% as expected.
    Powell signals potential rate cuts in 2024 if inflation continues to moderate.
    Dot plot shows three rate cuts projected for next year.
    """
    
    market_data = {
        "price": 450.25,
        "volume": 85_000_000,
        "avg_volume": 75_000_000,
        "price_change_pct": 0.5,
        "rsi": 58,
        "support": 445,
        "resistance": 455
    }
    
    try:
        signal = await claude.analyze_market_event(
            event_text=fed_event,
            symbol="SPY",
            market_data=market_data
        )
        
        print(f"Direction: {signal.get('direction', 'N/A')}")
        print(f"Conviction: {signal.get('conviction', 0)}%")
        print(f"Entry Price: ${signal.get('entry_price', 'N/A')}")
        print(f"Stop Loss: ${signal.get('stop_loss', 'N/A')}")
        print(f"Take Profit: ${signal.get('take_profit', 'N/A')}")
        print(f"Rationale: {signal.get('rationale', 'N/A')[:200]}...")
        
        # Calculate position size using Kelly
        if signal.get('conviction', 0) >= 50:
            position_size = KellyCriterion.size_from_conviction(signal['conviction'])
            print(f"Recommended Position Size: {position_size*100:.1f}% of portfolio")
        
        print("\n✅ Fed analysis completed successfully\n")
        
    except Exception as e:
        print(f"❌ Fed analysis failed: {e}\n")
    
    # Test 2: Earnings announcement
    print("Test 2: Analyzing Earnings Report")
    print("-" * 40)
    
    earnings_event = """
    Apple reports Q4 earnings: EPS $1.46 vs $1.39 expected, Revenue $89.5B vs $89.3B expected.
    iPhone revenue up 3% YoY. Services revenue grows 16% to new record.
    Guidance for holiday quarter slightly below analyst expectations due to China weakness.
    """
    
    aapl_data = {
        "price": 185.50,
        "volume": 65_000_000,
        "avg_volume": 55_000_000,
        "price_change_pct": -2.1,
        "rsi": 45,
        "support": 180,
        "resistance": 190
    }
    
    try:
        signal = await claude.analyze_market_event(
            event_text=earnings_event,
            symbol="AAPL",
            market_data=aapl_data
        )
        
        print(f"Direction: {signal.get('direction', 'N/A')}")
        print(f"Conviction: {signal.get('conviction', 0)}%")
        print(f"Timeframe: {signal.get('timeframe', 'N/A')}")
        print(f"Key Risks: {signal.get('key_risks', [])}")
        
        print("\n✅ Earnings analysis completed successfully\n")
        
    except Exception as e:
        print(f"❌ Earnings analysis failed: {e}\n")
    
    # Test 3: Breaking news
    print("Test 3: Analyzing Breaking News")
    print("-" * 40)
    
    news_event = """
    BREAKING: Tesla announces $5 billion share buyback program.
    CEO Elon Musk says the company has "excessive cash reserves" and 
    sees current valuation as attractive for buybacks.
    """
    
    tsla_data = {
        "price": 245.80,
        "volume": 120_000_000,
        "avg_volume": 100_000_000,
        "price_change_pct": 3.5,
        "rsi": 62,
        "support": 240,
        "resistance": 250
    }
    
    try:
        signal = await claude.analyze_market_event(
            event_text=news_event,
            symbol="TSLA",
            market_data=tsla_data
        )
        
        print(f"Direction: {signal.get('direction', 'N/A')}")
        print(f"Conviction: {signal.get('conviction', 0)}%")
        print(f"Alternative Scenario: {signal.get('alternative_scenario', 'N/A')}")
        
        print("\n✅ News analysis completed successfully\n")
        
    except Exception as e:
        print(f"❌ News analysis failed: {e}\n")
    
    print("=" * 60)
    print("Testing Complete!")
    print("=" * 60)

async def test_kelly_sizing():
    """Test Kelly Criterion position sizing"""
    print("\nTesting Kelly Criterion Position Sizing")
    print("-" * 40)
    
    # Test conviction-based sizing
    convictions = [30, 50, 70, 85, 100]
    for conviction in convictions:
        size = KellyCriterion.size_from_conviction(conviction)
        print(f"Conviction {conviction}%: Position size {size*100:.1f}%")
    
    # Test Kelly formula
    print("\nKelly Formula Test:")
    win_prob = 0.60  # 60% win rate
    avg_win = 0.05   # 5% average win
    avg_loss = -0.02  # 2% average loss
    
    full_kelly = KellyCriterion.calculate_position_size(win_prob, avg_win, avg_loss, kelly_fraction=1.0)
    quarter_kelly = KellyCriterion.calculate_position_size(win_prob, avg_win, avg_loss, kelly_fraction=0.25)
    
    print(f"Win Rate: {win_prob*100}%, Avg Win: {avg_win*100}%, Avg Loss: {avg_loss*100}%")
    print(f"Full Kelly: {full_kelly*100:.1f}% of portfolio")
    print(f"Quarter Kelly (safer): {quarter_kelly*100:.1f}% of portfolio")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_market_analysis())
    asyncio.run(test_kelly_sizing())
#!/usr/bin/env python3
"""
Test the Company Intelligence module
"""

import asyncio
from robo_trader.company_intelligence import CompanyIntelligence, EventType

async def main():
    print("🔍 Testing Company Intelligence Module\n")
    print("=" * 60)
    
    # Test with your actual portfolio symbols
    symbols = [
        'AAPL', 'NVDA', 'TSLA',  # Major tech
        'IXHL', 'NUAI', 'BZAI', 'ELTP',  # Biotech/pharma
        'PLTR', 'SOFI', 'UPST',  # Fintech/tech
        'CEG', 'CORZ', 'WULF'  # Energy/miners
    ]
    
    print(f"📊 Monitoring {len(symbols)} symbols for company events")
    print(f"Symbols: {', '.join(symbols[:5])}...")
    print()
    
    # Initialize company intelligence
    intel = CompanyIntelligence(symbols)
    await intel.initialize()
    
    print("Fetching company-specific events...")
    print("-" * 60)
    
    # Fetch all events - look back 7 days for testing
    # Temporarily patch the method to look back further
    intel.sec_client.get_recent_filings = lambda symbol: intel.sec_client.get_recent_filings(symbol, hours=168)
    
    # Fetch all events
    events = await intel.fetch_all_events()
    
    if not events:
        print("❌ No events found (market may be closed or no recent filings)")
    else:
        print(f"\n✅ Found {len(events)} total events!\n")
        
        # Group by event type
        event_types = {}
        for event in events:
            event_type = event.event_type.value
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Display summary
        print("📈 Event Summary by Type:")
        print("-" * 40)
        for event_type, type_events in event_types.items():
            print(f"  {event_type}: {len(type_events)} events")
        
        # Show high-impact events
        high_impact = [e for e in events if e.impact_score >= 70]
        
        if high_impact:
            print(f"\n🎯 HIGH IMPACT EVENTS (Score >= 70):")
            print("=" * 60)
            
            for event in high_impact[:10]:  # Show top 10
                print(f"\n{'🔴' if event.impact_score >= 90 else '🟡'} {event.symbol} - {event.event_type.value}")
                print(f"   📰 {event.headline}")
                print(f"   📊 Impact Score: {event.impact_score}/100")
                print(f"   📝 {event.description[:150]}...")
                if event.url:
                    print(f"   🔗 {event.url}")
                print(f"   ⏰ {event.timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Show events by symbol
        print(f"\n📊 Events by Symbol:")
        print("-" * 40)
        symbol_events = {}
        for event in events:
            if event.symbol not in symbol_events:
                symbol_events[event.symbol] = 0
            symbol_events[event.symbol] += 1
        
        for symbol, count in sorted(symbol_events.items(), key=lambda x: x[1], reverse=True):
            print(f"  {symbol}: {count} events")
    
    # Clean up
    await intel.close()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("\nNote: In production, this will:")
    print("  • Check for new filings every 5 minutes")
    print("  • Send high-impact events to AI for analysis")
    print("  • Execute trades on 75%+ conviction signals")
    print("  • Store all events in database for analysis")

if __name__ == "__main__":
    asyncio.run(main())
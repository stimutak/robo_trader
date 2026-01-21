#!/usr/bin/env python3
"""
Test the PolygonDataProvider implementation.

This script validates:
1. Connection to Polygon API
2. Historical bar fetching
3. Current price/quote retrieval
4. Rate limiting behavior
5. Error handling
"""

import asyncio
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, "/Users/oliver/robo_trader")

from robo_trader.data_providers import PolygonDataProvider


async def test_polygon_provider():
    """Run comprehensive tests on PolygonDataProvider."""

    print("=" * 60)
    print("POLYGON DATA PROVIDER TEST SUITE")
    print("=" * 60)

    # Initialize provider
    print("\n[1/6] Initializing provider...")
    try:
        provider = PolygonDataProvider(tier="free")
        print(f"  Provider: {provider}")
        print(f"  Supports streaming: {provider.supports_streaming}")
        print("  SUCCESS")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # Connect
    print("\n[2/6] Connecting to Polygon...")
    try:
        connected = await provider.connect()
        if connected:
            print("  SUCCESS - Connected to Polygon.io")
        else:
            print("  FAILED - Connection returned False")
            return False
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

    # Test historical bars
    print("\n[3/6] Fetching historical bars (AAPL, 1min, 10 bars)...")
    try:
        bars = await provider.get_historical_bars(
            symbol="AAPL",
            timeframe="1min",
            limit=10,
        )
        if not bars.empty:
            print(f"  Retrieved {len(bars)} bars")
            print(f"  Columns: {list(bars.columns)}")
            print(f"  Date range: {bars['date'].min()} to {bars['date'].max()}")
            print(f"  Latest close: ${bars['close'].iloc[-1]:.2f}")
            print("  SUCCESS")
        else:
            print("  WARNING - Empty DataFrame returned (may be outside market hours)")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Test different timeframes
    print("\n[4/6] Testing different timeframes...")
    timeframes = ["5min", "1hour", "1day"]
    for tf in timeframes:
        try:
            bars = await provider.get_historical_bars(
                symbol="NVDA",
                timeframe=tf,
                limit=5,
            )
            if not bars.empty:
                print(f"  {tf}: {len(bars)} bars, latest ${bars['close'].iloc[-1]:.2f}")
            else:
                print(f"  {tf}: No data (may be rate limited or outside hours)")
        except Exception as e:
            print(f"  {tf}: FAILED - {e}")

    # Test current price
    print("\n[5/6] Getting current prices...")
    symbols = ["AAPL", "NVDA", "TSLA"]
    for symbol in symbols:
        try:
            price = await provider.get_current_price(symbol)
            if price:
                print(f"  {symbol}: ${price:.2f}")
            else:
                print(f"  {symbol}: No price available")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    # Test quote
    print("\n[6/6] Getting detailed quote (AAPL)...")
    try:
        quote = await provider.get_quote("AAPL")
        if quote:
            print(f"  Symbol: {quote.symbol}")
            print(f"  Last: ${quote.last:.2f}" if quote.last else "  Last: N/A")
            print(f"  Bid: ${quote.bid:.2f}" if quote.bid else "  Bid: N/A (free tier)")
            print(f"  Ask: ${quote.ask:.2f}" if quote.ask else "  Ask: N/A (free tier)")
            print(f"  Volume: {quote.volume:,}" if quote.volume else "  Volume: N/A")
            print(f"  Timestamp: {quote.timestamp}")
            print("  SUCCESS")
        else:
            print("  WARNING - No quote returned")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Disconnect
    await provider.disconnect()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

    return True


async def test_comparison_with_symbols():
    """Test with actual trading symbols from user_settings."""
    print("\n" + "=" * 60)
    print("TESTING WITH TRADING SYMBOLS")
    print("=" * 60)

    # Sample of user's actual symbols
    symbols = ["AAPL", "NVDA", "TSLA", "PLTR", "SOFI", "CEG"]

    provider = PolygonDataProvider(tier="free")
    await provider.connect()

    print("\nFetching 1-day bars for trading symbols...")
    print("(Rate limited - this will take ~1 min on free tier)\n")

    results = {}
    for symbol in symbols:
        try:
            bars = await provider.get_historical_bars(
                symbol=symbol,
                timeframe="1day",
                limit=5,
            )
            if not bars.empty:
                latest = bars.iloc[-1]
                results[symbol] = {
                    "close": latest["close"],
                    "volume": latest["volume"],
                    "date": latest["date"],
                }
                print(
                    f"  {symbol}: ${latest['close']:.2f} "
                    f"(vol: {latest['volume']:,.0f}) "
                    f"@ {latest['date'].strftime('%Y-%m-%d')}"
                )
            else:
                print(f"  {symbol}: No data")
        except Exception as e:
            print(f"  {symbol}: ERROR - {e}")

    await provider.disconnect()

    print(f"\nSuccessfully fetched data for {len(results)}/{len(symbols)} symbols")
    return results


if __name__ == "__main__":
    print("Starting Polygon Provider Tests...\n")

    # Run main test suite
    success = asyncio.run(test_polygon_provider())

    if success:
        # Run symbol comparison test
        print("\n")
        asyncio.run(test_comparison_with_symbols())

    print("\nDone!")

#!/usr/bin/env python3
"""Test market hours functionality and WebSocket updates."""

from datetime import datetime
import pytz
import asyncio
import time
from robo_trader.market_hours import (
    is_market_open,
    is_extended_hours,
    get_market_session,
    get_next_market_open,
    seconds_until_market_open
)
from robo_trader.websocket_server import ws_manager


def test_market_hours():
    """Test market hours functions with various times."""
    
    eastern = pytz.timezone('US/Eastern')
    
    # Test cases: (datetime, expected_session, expected_open, expected_extended)
    test_cases = [
        # Monday 9:00 AM ET (before open)
        (datetime(2025, 8, 25, 9, 0, tzinfo=eastern), "pre-market", False, True),
        # Monday 9:30 AM ET (market open)
        (datetime(2025, 8, 25, 9, 30, tzinfo=eastern), "regular", True, False),
        # Monday 3:00 PM ET (during trading)
        (datetime(2025, 8, 25, 15, 0, tzinfo=eastern), "regular", True, False),
        # Monday 4:00 PM ET (market close)
        (datetime(2025, 8, 25, 16, 0, tzinfo=eastern), "after-hours", False, True),
        # Monday 8:00 PM ET (after extended hours)
        (datetime(2025, 8, 25, 20, 0, tzinfo=eastern), "closed", False, False),
        # Saturday 12:00 PM ET (weekend)
        (datetime(2025, 8, 30, 12, 0, tzinfo=eastern), "closed", False, False),
    ]
    
    print("Market Hours Test Results:")
    print("-" * 60)
    
    for dt, expected_session, expected_open, expected_extended in test_cases:
        session = get_market_session(dt)
        is_open = is_market_open(dt)
        is_extended = is_extended_hours(dt)
        
        print(f"\nTime: {dt.strftime('%A %Y-%m-%d %I:%M %p %Z')}")
        print(f"  Session: {session} (expected: {expected_session}) {'✓' if session == expected_session else '✗'}")
        print(f"  Is Open: {is_open} (expected: {expected_open}) {'✓' if is_open == expected_open else '✗'}")
        print(f"  Extended: {is_extended} (expected: {expected_extended}) {'✓' if is_extended == expected_extended else '✗'}")
    
    # Test current status
    print("\n" + "=" * 60)
    print("Current Market Status:")
    print("-" * 60)
    
    now = datetime.now(eastern)
    print(f"Current time: {now.strftime('%A %Y-%m-%d %I:%M %p %Z')}")
    print(f"Market session: {get_market_session()}")
    print(f"Market open: {is_market_open()}")
    print(f"Extended hours: {is_extended_hours()}")
    
    if not is_market_open():
        next_open = get_next_market_open()
        seconds = seconds_until_market_open()
        hours = seconds / 3600
        print(f"Next market open: {next_open.strftime('%A %Y-%m-%d %I:%M %p %Z')}")
        print(f"Time until open: {hours:.1f} hours ({seconds:,} seconds)")


def test_websocket_updates():
    """Test WebSocket functionality by sending market updates."""
    print("\n" + "=" * 60)
    print("Testing WebSocket Updates:")
    print("-" * 60)
    
    # Send test market data updates
    symbols = ["AAPL", "NVDA", "TSLA"]
    prices = [195.50, 185.00, 245.75]
    
    for symbol, price in zip(symbols, prices):
        ws_manager.send_market_update(
            symbol=symbol,
            price=price,
            bid=price - 0.02,
            ask=price + 0.02,
            volume=1000000
        )
        print(f"✓ Sent market update for {symbol} @ ${price:.2f}")
        time.sleep(0.5)
    
    # Send a trade update
    ws_manager.send_trade_update("NVDA", "BUY", 10, 185.00, "executed")
    print("✓ Sent trade update: BUY 10 NVDA @ $185.00")
    
    # Send a signal update
    ws_manager.send_signal_update("AAPL", "BUY", 0.75)
    print("✓ Sent signal update: BUY signal for AAPL (strength: 0.75)")
    
    # Send performance metrics
    ws_manager.send_performance_update({
        "total_pnl": 1820.00,
        "daily_pnl": 320.00,
        "win_rate": 0.65,
        "sharpe_ratio": 1.45
    })
    print("✓ Sent performance update: Daily P&L: $320.00")
    
    print("\nWebSocket updates queued successfully!")
    print("(Updates will be broadcast to any connected dashboard clients)")


if __name__ == "__main__":
    test_market_hours()
    test_websocket_updates()
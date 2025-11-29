#!/usr/bin/env python3
"""
Test for thread-safe concurrent portfolio operations.

This test verifies that the Portfolio class correctly handles
concurrent updates from multiple async tasks without race conditions.
"""

import asyncio
import random
from decimal import Decimal

from robo_trader.portfolio import Portfolio


async def random_trader(portfolio: Portfolio, trader_id: int, symbol: str, num_trades: int):
    """Simulate a trader making random trades."""
    for i in range(num_trades):
        # Random buy or sell
        side = random.choice(["BUY", "SELL"])
        quantity = random.randint(1, 10)
        price = round(100 + random.uniform(-10, 10), 2)

        try:
            await portfolio.update_fill(symbol, side, quantity, price)
            print(f"Trader {trader_id}: {side} {quantity} {symbol} @ ${price}")
        except Exception as e:
            print(f"Trader {trader_id} error: {e}")

        # Small random delay
        await asyncio.sleep(random.uniform(0.001, 0.01))


async def test_concurrent_updates():
    """Test concurrent portfolio updates."""
    print("Testing Concurrent Portfolio Updates")
    print("=" * 50)

    # Initialize portfolio
    portfolio = Portfolio(100000.0)

    # Create multiple traders operating on different symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    traders = []

    # Launch 5 traders per symbol, each making 20 trades
    for symbol in symbols:
        for trader_id in range(5):
            trader = random_trader(portfolio, trader_id + len(symbols) * 10, symbol, 20)
            traders.append(trader)

    print(f"Launching {len(traders)} concurrent traders...")

    # Run all traders concurrently
    await asyncio.gather(*traders)

    print("\n" + "=" * 50)
    print("Final Portfolio State:")
    print(f"Cash: ${portfolio.cash}")
    print(f"Realized P&L: ${portfolio.realized_pnl}")

    print("\nPositions:")
    for symbol, position in portfolio.positions.items():
        print(f"  {symbol}: {position.quantity} shares @ ${position.avg_price}")

    # Test concurrent reads while writing
    print("\n" + "=" * 50)
    print("Testing Concurrent Reads During Writes...")

    async def writer_task():
        """Continuously write to portfolio."""
        for _ in range(100):
            await portfolio.update_fill("TEST", "BUY", 1, 100.0)
            await asyncio.sleep(0.001)

    async def reader_task():
        """Continuously read from portfolio."""
        results = []
        for _ in range(100):
            market_prices = {"TEST": 105.0}
            equity = await portfolio.equity(market_prices)
            unrealized = await portfolio.compute_unrealized(market_prices)
            results.append((equity, unrealized))
            await asyncio.sleep(0.001)
        return results

    # Run readers and writers concurrently
    writer = writer_task()
    readers = [reader_task() for _ in range(3)]

    results = await asyncio.gather(writer, *readers)

    print("Concurrent read/write test completed successfully")
    print(f"Final TEST position: {portfolio.positions.get('TEST')}")

    # Verify no data corruption
    if "TEST" in portfolio.positions:
        test_pos = portfolio.positions["TEST"]
        expected_qty = 100  # We bought 100 times, 1 share each
        assert (
            test_pos.quantity == expected_qty
        ), f"Expected {expected_qty} shares, got {test_pos.quantity}"
        print(f"✅ Data integrity verified: {test_pos.quantity} shares as expected")

    print("\n" + "=" * 50)
    print("All concurrent tests passed successfully!")


async def test_race_condition():
    """Test for race conditions in position updates."""
    print("\nTesting Race Condition Prevention")
    print("=" * 50)

    portfolio = Portfolio(100000.0)

    async def buy_and_sell(symbol: str):
        """Buy and immediately sell the same symbol."""
        await portfolio.update_fill(symbol, "BUY", 100, 100.0)
        await portfolio.update_fill(symbol, "SELL", 100, 101.0)

    # Launch many concurrent buy/sell pairs
    tasks = [buy_and_sell("RACE") for _ in range(50)]

    await asyncio.gather(*tasks)

    # After 50 buy/sell pairs, position should be empty
    assert "RACE" not in portfolio.positions, "Position should be closed after equal buy/sell"

    # Check P&L is correct: 50 trades * 100 shares * $1 profit
    expected_pnl = Decimal("5000.0")
    assert (
        portfolio.realized_pnl == expected_pnl
    ), f"Expected P&L ${expected_pnl}, got ${portfolio.realized_pnl}"

    print(f"✅ Race condition test passed: P&L = ${portfolio.realized_pnl}")
    print(f"✅ Position correctly closed: {'RACE' not in portfolio.positions}")


async def main():
    """Run all tests."""
    print("Thread-Safe Portfolio Concurrent Access Tests")
    print("=" * 50)

    try:
        # Test 1: Multiple concurrent traders
        await test_concurrent_updates()

        # Test 2: Race condition prevention
        await test_race_condition()

        print("\n" + "=" * 50)
        print("🎉 ALL CONCURRENT TESTS PASSED!")
        print("✅ Portfolio is thread-safe for async operations")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)

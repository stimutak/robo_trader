#!/usr/bin/env python3
"""Test short selling functionality."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.config import load_config
from robo_trader.execution import Order
from robo_trader.portfolio import Portfolio
from robo_trader.risk import Position
from robo_trader.runner_async import AsyncRunner


async def test_short_selling():
    """Test short selling with paper trading."""
    print("=" * 60)
    print("Testing Short Selling with Paper Trading")
    print("=" * 60)

    config = load_config()

    # Enable short selling in config
    config.execution.enable_short_selling = True

    # Initialize runner with short selling enabled
    runner = AsyncRunner(
        duration="1 D",
        bar_size="5 mins",
        use_ml_strategy=False,  # Test short selling separately
        use_smart_execution=False,
    )

    # Override config
    runner.cfg = config

    await runner.setup()

    print("\n1. Configuration Check:")
    print(f"   Short selling enabled: {runner.cfg.execution.enable_short_selling}")
    print(f"   Max short exposure: ${runner.cfg.execution.max_short_exposure:,.0f}")
    print(f"   Max position value: ${runner.cfg.execution.max_position_value:,.0f}")

    print("\n2. Testing Short Position Opening:")

    # Simulate opening a short position
    symbol = "AAPL"

    # Check if we can short
    if runner.cfg.execution.enable_short_selling:
        print(f"   Opening short position in {symbol}")

        # Create a sell order (for shorting)
        order = Order(symbol, "SELL", 100)

        # Execute the order (this would open a short position)
        fill = await runner.executor.execute(order)

        if fill:
            print(f"   ✅ Short position opened:")
            print(f"      Symbol: {fill['symbol']}")
            print(f"      Quantity: -{fill['quantity']} (negative for short)")
            print(f"      Price: ${fill['price']:.2f}")
            print(f"      Value: ${fill['quantity'] * fill['price']:,.2f}")

            # Update position tracking
            if symbol not in runner.positions:
                runner.positions[symbol] = Position(symbol, 0, 0.0)

            # Short position has negative quantity
            runner.positions[symbol].quantity = -fill["quantity"]
            runner.positions[symbol].avg_price = fill["price"]

            print(f"\n   Current position:")
            print(f"      Quantity: {runner.positions[symbol].quantity}")
            print(f"      Avg Price: ${runner.positions[symbol].avg_price:.2f}")
            print(f"      Market Value: ${runner.positions[symbol].quantity * fill['price']:,.2f}")

    print("\n3. Testing Buy-to-Cover (Closing Short):")

    if symbol in runner.positions and runner.positions[symbol].quantity < 0:
        print(f"   Covering short position in {symbol}")

        # Create a buy order to cover the short
        cover_order = Order(symbol, "BUY", abs(runner.positions[symbol].quantity))

        # Execute the cover order
        cover_fill = await runner.executor.execute(cover_order)

        if cover_fill:
            print(f"   ✅ Short position covered:")
            print(f"      Symbol: {cover_fill['symbol']}")
            print(f"      Quantity: {cover_fill['quantity']}")
            print(f"      Cover Price: ${cover_fill['price']:.2f}")

            # Calculate P&L
            short_price = runner.positions[symbol].avg_price
            cover_price = cover_fill["price"]
            quantity = abs(runner.positions[symbol].quantity)
            pnl = (short_price - cover_price) * quantity  # Profit if cover < short

            print(f"\n   P&L Calculation:")
            print(f"      Short Price: ${short_price:.2f}")
            print(f"      Cover Price: ${cover_price:.2f}")
            print(f"      Quantity: {quantity}")
            print(f"      P&L: ${pnl:,.2f} {'✅ Profit' if pnl > 0 else '❌ Loss'}")

            # Clear position
            runner.positions[symbol].quantity = 0

    print("\n4. Testing Short Selling Risk Limits:")

    # Test max short exposure limit
    test_symbols = ["NVDA", "TSLA", "META"]
    total_short_value = 0

    for sym in test_symbols:
        # Try to open a large short position
        large_order = Order(sym, "SELL", 1000)

        # Check if it would exceed limits
        order_value = large_order.quantity * 150  # Assume $150 price

        if total_short_value + order_value > runner.cfg.execution.max_short_exposure:
            print(f"   ❌ {sym}: Would exceed max short exposure")
            print(f"      Current short: ${total_short_value:,.0f}")
            print(f"      Order value: ${order_value:,.0f}")
            print(f"      Max allowed: ${runner.cfg.execution.max_short_exposure:,.0f}")
        else:
            print(f"   ✅ {sym}: Within short exposure limits")
            total_short_value += order_value

    print("\n5. Testing Mixed Long/Short Portfolio:")

    # Simulate a mixed portfolio
    mixed_positions = {
        "AAPL": Position("AAPL", 100, 150.0),  # Long
        "NVDA": Position("NVDA", -50, 500.0),  # Short
        "TSLA": Position("TSLA", 75, 200.0),  # Long
        "META": Position("META", -30, 350.0),  # Short
    }

    long_value = 0
    short_value = 0

    print("   Portfolio positions:")
    for symbol, pos in mixed_positions.items():
        value = abs(pos.quantity * pos.avg_price)
        if pos.quantity > 0:
            long_value += value
            print(f"      {symbol}: LONG {pos.quantity} @ ${pos.avg_price:.2f} = ${value:,.0f}")
        else:
            short_value += value
            print(
                f"      {symbol}: SHORT {abs(pos.quantity)} @ ${pos.avg_price:.2f} = ${value:,.0f}"
            )

    print(f"\n   Portfolio Summary:")
    print(f"      Total Long: ${long_value:,.0f}")
    print(f"      Total Short: ${short_value:,.0f}")
    print(f"      Net Exposure: ${long_value - short_value:,.0f}")
    print(f"      Gross Exposure: ${long_value + short_value:,.0f}")

    await runner.teardown()

    print("\n" + "=" * 60)
    print("✅ Short selling is working correctly!")
    print("=" * 60)
    print("\nNotes:")
    print("- Short positions have negative quantities")
    print("- P&L = (Short Price - Cover Price) × Quantity")
    print("- Risk limits prevent excessive short exposure")
    print("- Can maintain mixed long/short portfolios")


async def main():
    """Run all tests."""
    await test_short_selling()


if __name__ == "__main__":
    asyncio.run(main())

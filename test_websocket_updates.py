#!/usr/bin/env python
"""
Test script to simulate WebSocket price updates during after-hours.
This helps verify the dashboard updates are working correctly.
"""

import asyncio
import random
import time

from robo_trader.websocket_client import ws_client

# Test symbols with their base prices
TEST_SYMBOLS = {
    "AAPL": 232.90,
    "NVDA": 180.67,
    "TSLA": 346.12,
    "MSFT": 505.18,
    "QQQ": 571.86,
    "PLTR": 157.55,
    "SOFI": 26.03,
    "VRT": 134.24,
    "UPST": 73.61,
    "TEM": 73.51,
    "WULF": 9.44,
    "OPEN": 4.28,
    "CORZ": 14.38,
    "CEG": 320.11,
    "BZAI": 3.76,
    "ELTP": 0.66,
    "IXHL": 0.62,
    "NUAI": 0.49,
    "APLD": 16.62,
    "HTFL": 31.79,
    "SDGR": 19.72,
}


async def simulate_price_updates():
    """Simulate real-time price updates."""
    print("Starting WebSocket price update simulation...")
    print(f"Simulating updates for {len(TEST_SYMBOLS)} symbols")
    print("Open the dashboard to see real-time updates!")
    print("Press Ctrl+C to stop\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- Update cycle {iteration} ---")

        # Update each symbol with a small random price change
        for symbol, base_price in TEST_SYMBOLS.items():
            # Generate random price movement (-2% to +2%)
            change_pct = random.uniform(-0.02, 0.02)
            new_price = base_price * (1 + change_pct)

            # Calculate bid/ask spread
            spread = base_price * 0.001  # 0.1% spread
            bid = new_price - spread / 2
            ask = new_price + spread / 2

            # Random volume
            volume = random.randint(1000, 100000) * 100

            # Send the update
            ws_client.send_market_update(
                symbol=symbol, price=new_price, bid=bid, ask=ask, volume=volume
            )

            # Show what we sent
            print(f"  {symbol}: ${new_price:.2f} (was ${base_price:.2f}, {change_pct*100:+.2f}%)")

            # Small delay between symbols to avoid overwhelming
            await asyncio.sleep(0.1)

        # Wait before next update cycle
        print(f"\nWaiting 5 seconds before next update...")
        await asyncio.sleep(5)


def main():
    """Main entry point."""
    # Ensure WebSocket client is started
    ws_client.start()
    time.sleep(1)  # Give it time to connect

    try:
        # Run the simulation
        asyncio.run(simulate_price_updates())
    except KeyboardInterrupt:
        print("\n\nSimulation stopped.")


if __name__ == "__main__":
    main()

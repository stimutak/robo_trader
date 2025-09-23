#!/usr/bin/env python3
"""Test WebSocket connection and listen for updates."""

import asyncio
import json
import sys

import websockets


async def test_and_listen():
    """Test WebSocket connection and listen for updates."""
    uri = "ws://localhost:8765"

    try:
        print(f"Connecting to {uri}...")
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully")

            # Wait for initial message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✓ Received initial message: {data}")

            # Send a subscribe message to get all updates
            subscribe_msg = json.dumps(
                {"type": "subscribe", "symbols": ["*"]}  # Subscribe to all symbols
            )
            await websocket.send(subscribe_msg)
            print("✓ Sent subscribe message for all symbols")

            # Listen for updates for 10 seconds
            print("\nListening for updates (10 seconds)...")
            print("-" * 40)

            try:
                end_time = asyncio.get_event_loop().time() + 10
                update_count = 0

                while asyncio.get_event_loop().time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        update_count += 1

                        # Display update based on type
                        if data.get("type") == "market_data":
                            print(f"📊 Market: {data.get('symbol')} @ ${data.get('price'):.2f}")
                        elif data.get("type") == "trade":
                            print(
                                f"💰 Trade: {data.get('side')} {data.get('quantity')} {data.get('symbol')} @ ${data.get('price'):.2f}"
                            )
                        elif data.get("type") == "signal":
                            print(
                                f"🔔 Signal: {data.get('signal')} {data.get('symbol')} (strength: {data.get('strength')})"
                            )
                        elif data.get("type") == "performance":
                            metrics = data.get("metrics", {})
                            print(f"📈 Performance: P&L=${metrics.get('daily_pnl', 0):.2f}")
                        else:
                            print(f"📨 {data.get('type', 'unknown')}: {data}")

                    except asyncio.TimeoutError:
                        # No message received in 1 second, continue
                        pass

                print("-" * 40)
                print(f"\n✅ Received {update_count} updates")

            except Exception as e:
                print(f"Error while listening: {e}")

            print("\n✅ WebSocket test complete")
            return True

    except ConnectionRefusedError:
        print(f"❌ Connection refused - server not running on {uri}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_and_listen())
    sys.exit(0 if success else 1)

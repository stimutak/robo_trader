#!/usr/bin/env python3
"""
2-minute IBKR API connection stability test
"""
import asyncio
import time
from datetime import datetime

from robo_trader.utils.robust_connection import connect_ibkr_robust


async def test_2min_stability():
    """Test connection stability for 2 minutes"""

    print("=" * 60)
    print("2-Minute IBKR API Connection Stability Test")
    print("=" * 60)

    client = None
    start_time = time.time()

    try:
        # Connect
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connecting to IBKR Gateway...")
        client = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=1,
            readonly=True,
            timeout=15.0,
            max_retries=2,
            use_subprocess=True,
        )

        connect_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Connected in {connect_time:.3f}s")

        # Get initial data
        accounts = await client.get_accounts()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Accounts: {accounts}")

        positions = await client.get_positions()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Positions: {len(positions)}")

        # Test for 2 minutes
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting 2-minute stability test...")
        print("Testing: ping every 5 seconds, data fetch every 15 seconds\n")

        test_duration = 120  # 2 minutes
        ping_interval = 5
        data_interval = 15

        ping_count = 0
        ping_failures = 0
        data_count = 0
        data_failures = 0

        last_ping = time.time()
        last_data = time.time()

        while time.time() - start_time < test_duration:
            current_time = time.time()
            elapsed = current_time - start_time

            # Ping test every 5 seconds
            if current_time - last_ping >= ping_interval:
                try:
                    pong = await client.ping()
                    ping_count += 1
                    if pong:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] [{elapsed:5.1f}s] Ping #{ping_count}: ✅"
                        )
                    else:
                        print(
                            f"[{datetime.now().strftime('%H:%M:%S')}] [{elapsed:5.1f}s] Ping #{ping_count}: ❌ (returned False)"
                        )
                        ping_failures += 1
                except Exception as e:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [{elapsed:5.1f}s] Ping #{ping_count + 1}: ❌ {e}"
                    )
                    ping_failures += 1
                    ping_count += 1

                last_ping = current_time

            # Data fetch test every 15 seconds
            if current_time - last_data >= data_interval:
                try:
                    accounts = await client.get_accounts()
                    positions = await client.get_positions()
                    data_count += 1
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [{elapsed:5.1f}s] Data fetch #{data_count}: ✅ (accounts={len(accounts)}, positions={len(positions)})"
                    )
                except Exception as e:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] [{elapsed:5.1f}s] Data fetch #{data_count + 1}: ❌ {e}"
                    )
                    data_failures += 1
                    data_count += 1

                last_data = current_time

            # Sleep briefly
            await asyncio.sleep(0.5)

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("2-Minute Test Complete!")
        print("=" * 60)
        print(f"Total time: {total_time:.1f}s")
        print(f"\nPing tests:")
        print(f"  Total: {ping_count}")
        print(f"  Successful: {ping_count - ping_failures}")
        print(f"  Failed: {ping_failures}")
        print(
            f"  Success rate: {((ping_count - ping_failures) / ping_count * 100) if ping_count > 0 else 0:.1f}%"
        )

        print(f"\nData fetch tests:")
        print(f"  Total: {data_count}")
        print(f"  Successful: {data_count - data_failures}")
        print(f"  Failed: {data_failures}")
        print(
            f"  Success rate: {((data_count - data_failures) / data_count * 100) if data_count > 0 else 0:.1f}%"
        )

        if ping_failures == 0 and data_failures == 0:
            print("\n✅ PERFECT! No failures during 2-minute test!")
        elif ping_failures + data_failures <= 2:
            print(f"\n⚠️  Minor issues: {ping_failures + data_failures} failures")
        else:
            print(f"\n❌ UNSTABLE: {ping_failures + data_failures} failures")

        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

    finally:
        if client:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Disconnecting...")
            try:
                await client.disconnect()
                await client.stop()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Disconnected cleanly")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Disconnect error: {e}")


if __name__ == "__main__":
    asyncio.run(test_2min_stability())

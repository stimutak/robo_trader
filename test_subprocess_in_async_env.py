#!/usr/bin/env python3
"""
Test subprocess IBKR client in complex async environment
"""
import asyncio

from robo_trader.utils.robust_connection import connect_ibkr_robust


async def background_task():
    """Simulate background tasks running in runner_async"""
    while True:
        await asyncio.sleep(1)


async def test_with_background_tasks():
    """Test connection with background tasks running"""
    print("Starting background tasks...")

    # Start some background tasks like runner_async does
    tasks = [
        asyncio.create_task(background_task()),
        asyncio.create_task(background_task()),
        asyncio.create_task(background_task()),
    ]

    await asyncio.sleep(0.5)

    print("Connecting to IBKR with background tasks running...")
    try:
        client = await connect_ibkr_robust(
            host="127.0.0.1",
            port=4002,
            client_id=1,
            readonly=True,
            timeout=15.0,
            max_retries=1,
            use_subprocess=True,
        )

        accounts = await client.get_accounts()
        print(f"✅ Connected! Accounts: {accounts}")

        await client.disconnect()
        await client.stop()

    except Exception as e:
        print(f"❌ Failed: {e}")
    finally:
        # Cancel background tasks
        for task in tasks:
            task.cancel()


if __name__ == "__main__":
    asyncio.run(test_with_background_tasks())

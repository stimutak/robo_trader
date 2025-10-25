#!/usr/bin/env python3
"""
Test threading-based subprocess client
"""
import asyncio

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient


async def test():
    print("Creating subprocess client...")
    client = SubprocessIBKRClient()

    print("Starting subprocess...")
    await client.start()
    print(f"✅ Subprocess started, PID: {client.process.pid}")

    print("Sending ping command...")
    try:
        response = await client.ping()
        print(f"✅ Ping response: {response}")
    except Exception as e:
        print(f"❌ Ping failed: {e}")

    print("Stopping subprocess...")
    await client.stop()
    print("✅ Done")


if __name__ == "__main__":
    asyncio.run(test())

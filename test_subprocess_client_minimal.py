#!/usr/bin/env python3
"""
Minimal test of subprocess client with debug output
"""
import asyncio
import logging
import sys

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
import structlog

from robo_trader.clients.subprocess_ibkr_client import SubprocessIBKRClient


async def test_minimal():
    """Minimal test with debug output"""

    print("=" * 60)
    print("Testing SubprocessIBKRClient - Minimal")
    print("=" * 60)

    client = SubprocessIBKRClient()

    try:
        print("\n1. Starting subprocess...")
        await client.start()
        print(f"   Subprocess PID: {client.process.pid}")
        print(f"   Python: {sys.executable}")

        print("\n2. Sending connect command...")
        print("   (This is where it usually times out)")

        connected = await client.connect(
            host="127.0.0.1", port=4002, client_id=1, readonly=True, timeout=15.0
        )

        print(f"\n✅ Connected: {connected}")

        accounts = await client.get_accounts()
        print(f"✅ Accounts: {accounts}")

        await client.disconnect()
        print("✅ Disconnected")

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()

        # Check if subprocess is still running
        if client.process:
            print(f"\nSubprocess status:")
            print(f"  PID: {client.process.pid}")
            print(f"  Return code: {client.process.returncode}")

            # Try to read stderr
            try:
                stderr = await asyncio.wait_for(client.process.stderr.read(1024), timeout=1.0)
                if stderr:
                    print(f"  Stderr: {stderr.decode()}")
            except Exception:
                pass

    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(test_minimal())

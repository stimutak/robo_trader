#!/usr/bin/env python3
"""
Test pure async connection to IBKR without nested event loops
"""
import asyncio
import logging

from ib_async import IB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pure_async_connection():
    """Test connection using pure async approach"""
    ib = IB()

    try:
        logger.info("Attempting pure async connection to TWS...")
        # Use connectAsync directly without nested event loops
        await ib.connectAsync(host="127.0.0.1", port=7497, clientId=99, timeout=10)
        logger.info("✅ Successfully connected!")

        # Test basic functionality
        logger.info("Testing account summary...")
        summary = await ib.accountSummaryAsync()
        logger.info(f"Account summary items: {len(summary)}")

        # Clean disconnect
        ib.disconnect()
        logger.info("✅ Disconnected successfully")
        return True

    except asyncio.TimeoutError:
        logger.error("❌ Connection timed out")
        if ib.isConnected():
            ib.disconnect()
        return False
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        if ib.isConnected():
            ib.disconnect()
        return False


async def main():
    """Main test runner"""
    success = await test_pure_async_connection()
    if success:
        logger.info("\n✅ Pure async connection works!")
        logger.info("The issue is with nested event loops from patchAsyncio()")
    else:
        logger.info("\n❌ Even pure async fails - check TWS configuration")


if __name__ == "__main__":
    # Run without any nested event loop patches
    asyncio.run(main())

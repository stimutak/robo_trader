#!/Users/oliver/robo_trader/venv/bin/python
"""
AI Trading System Launcher

Starts the intelligent trading system with 21 pre-configured symbols.
Integrates with the dashboard for real-time monitoring.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from robo_trader.logger import get_logger
from robo_trader.runner import run_once

logger = get_logger(__name__)

# Default symbols - should match dashboard settings
DEFAULT_SYMBOLS = [
    "AAPL",  # Apple
    "NVDA",  # Nvidia
    "TSLA",  # Tesla
    "IXHL",  # iShares Healthcare Innovation ETF
    "NUAI",  # Nu Holdings
    "BZAI",  # Baidu AI
    "ELTP",  # Elite Pharma
    "OPEN",  # Opendoor
    "CEG",  # Constellation Energy
    "VRT",  # Vertiv Holdings
    "PLTR",  # Palantir
    "UPST",  # Upstart
    "TEM",  # Tempus AI
    "HTFL",  # HTF Holdings
    "SDGR",  # Schrodinger
    "APLD",  # Applied Digital
    "SOFI",  # SoFi Technologies
    "CORZ",  # Core Scientific
    "WULF",  # TeraWulf
]


def load_symbols_from_settings():
    """Load symbols from user settings file if it exists."""
    import json
    from pathlib import Path

    settings_file = Path("user_settings.json")
    if settings_file.exists():
        try:
            with open(settings_file, "r") as f:
                settings = json.load(f)
                symbols = settings.get("default", {}).get("symbols", DEFAULT_SYMBOLS)
                logger.info(f"Loaded {len(symbols)} symbols from user settings")
                return symbols
        except Exception as e:
            logger.warning(f"Failed to load user settings: {e}, using defaults")

    return DEFAULT_SYMBOLS


shutdown_event = asyncio.Event()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    shutdown_event.set()


async def trading_loop(
    symbols: List[str],
    duration: str = "2 D",
    bar_size: str = "5 mins",
    sma_fast: int = 10,
    sma_slow: int = 20,
    default_cash: float = 100000.0,
    max_order_notional: float = 10000.0,
    slippage_bps: float = 0.0,
):
    """
    Main trading loop that continuously monitors and trades symbols.

    Args:
        symbols: List of symbols to trade
        duration: Historical data duration
        bar_size: Bar size for historical data
        sma_fast: Fast SMA period
        sma_slow: Slow SMA period
        default_cash: Starting cash amount
        max_order_notional: Maximum order size
        slippage_bps: Slippage in basis points
    """
    logger.info(f"Starting AI trading system with {len(symbols)} symbols")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Configuration: SMA {sma_fast}/{sma_slow}, Max Order: ${max_order_notional:,.0f}")

    iteration = 0
    while not shutdown_event.is_set():
        iteration += 1
        start_time = datetime.now()

        try:
            logger.info(f"=== Trading Iteration {iteration} starting at {start_time} ===")

            # Run trading logic for all symbols
            await run_once(
                symbols=symbols,
                duration=duration,
                bar_size=bar_size,
                sma_fast=sma_fast,
                sma_slow=sma_slow,
                default_cash=default_cash,
                max_order_notional=max_order_notional,
                slippage_bps=slippage_bps,
            )

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Iteration {iteration} completed in {elapsed:.1f} seconds")

        except Exception as e:
            logger.error(f"Error in trading iteration {iteration}: {e}", exc_info=True)

        # Wait before next iteration (5 minutes default)
        wait_time = 300  # 5 minutes
        logger.info(f"Waiting {wait_time} seconds before next iteration...")

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
            break  # Shutdown requested
        except asyncio.TimeoutError:
            continue  # Continue to next iteration


async def main():
    """Main entry point for AI trading system."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 60)
    logger.info("🤖 Robo Trader AI System Starting")
    logger.info("=" * 60)
    logger.info("Mode: PAPER TRADING (Default)")
    logger.info("Dashboard: http://localhost:5555")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Check if dashboard is running
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dashboard_running = sock.connect_ex(("localhost", 5555)) == 0
    sock.close()

    if not dashboard_running:
        logger.warning("Dashboard not detected at http://localhost:5555")
        logger.warning("Start the dashboard with: python app.py")
    else:
        logger.info("✓ Dashboard detected at http://localhost:5555")

    # Load symbols from settings or use defaults
    symbols = load_symbols_from_settings()

    # Start trading loop
    try:
        await trading_loop(
            symbols=symbols,
            duration="2 D",
            bar_size="5 mins",
            sma_fast=10,
            sma_slow=20,
            default_cash=100000.0,
            max_order_notional=10000.0,
            slippage_bps=0.0,
        )
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in trading system: {e}", exc_info=True)
    finally:
        logger.info("AI trading system shutdown complete")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("ai_trading.log"), logging.StreamHandler(sys.stdout)],
    )

    # Run the async main function
    asyncio.run(main())

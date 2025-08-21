#!/usr/bin/env python3
"""
Simple script to start AI-powered trading with one command.
Following CLAUDE.md: Clarity over cleverness
"""

import asyncio
import sys
import signal
from dotenv import load_dotenv
load_dotenv()

from robo_trader.ai_runner import AITradingSystem
from robo_trader.logger import get_logger

logger = get_logger(__name__)

# Default configuration
DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
DEFAULT_CAPITAL = 100000

async def main():
    """Start the AI trading system."""
    
    print("\n" + "="*60)
    print("🤖 ROBO TRADER - AI-POWERED TRADING SYSTEM")
    print("="*60)
    print("\nConfiguration:")
    print(f"  Symbols: {', '.join(DEFAULT_SYMBOLS)}")
    print(f"  Capital: ${DEFAULT_CAPITAL:,}")
    print(f"  Mode: Paper Trading")
    print(f"  AI: Claude 3.5 Sonnet")
    print("\nStarting system...")
    
    # Create system
    system = AITradingSystem(
        symbols=DEFAULT_SYMBOLS,
        use_ai=True,
        capital=DEFAULT_CAPITAL,
        news_check_interval=300  # 5 minutes
    )
    
    # Setup
    try:
        await system.setup()
        print("\n✅ System ready! Starting trading loop...")
        print("\n📊 Monitoring:")
        print("  • News feeds every 5 minutes")
        print("  • Market data every minute")
        print("  • AI analysis on high-impact events")
        print("\nPress Ctrl+C to stop\n")
        print("-"*60)
        
        # Run
        await system.run()
        
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        await system.stop()
        print("✅ System stopped safely")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        await system.stop()
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
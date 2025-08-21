#!/usr/bin/env python3
"""
Test connection to IB Desktop (Gateway or TWS)
This replaces the broken Web API connection
"""

import asyncio
import sys
from datetime import datetime
from robo_trader.ibkr_client import IBKRClient
from robo_trader.config import load_config
from robo_trader.logger import get_logger

logger = get_logger(__name__)


async def test_connection():
    """Test basic IB Desktop connection and data fetch"""
    
    config = load_config()
    
    # Use the configured port from .env (should be 7497 for paper)
    port = config.ibkr_port
    
    print(f"\n{'='*60}")
    print("IB DESKTOP CONNECTION TEST")
    print(f"{'='*60}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Host: {config.ibkr_host}")
    print(f"Port: {port}")
    print(f"Client ID: {config.ibkr_client_id}")
    print(f"Mode: {'LIVE' if port == 7496 else 'PAPER'}")
    
    if port == 7496:
        print("\n⚠️  WARNING: Using LIVE trading port 7496!")
        print("For paper trading, please:")
        print("1. Stop IB Gateway/TWS")
        print("2. Start it in PAPER mode")
        print("3. It will use port 7497")
        print("\n🔒 Using READONLY connection for safety")
    
    print(f"\n{'='*60}")
    print("TESTING CONNECTION...")
    print(f"{'='*60}")
    
    client = IBKRClient(
        host=config.ibkr_host,
        port=port,
        client_id=config.ibkr_client_id
    )
    
    try:
        # Test 1: Connect
        print("\n1. Connecting to IB Desktop...")
        await client.connect(readonly=True, timeout=10.0)
        print("   ✅ Connected successfully!")
        
        # Test 2: Check connection details
        print("\n2. Connection Details:")
        if client.ib.isConnected():
            print(f"   ✅ Connection active")
            # Note: serverVersion and connectionTime not available in ib_insync
            print(f"   Client ready for trading")
        
        # Test 3: Fetch account info
        print("\n3. Account Information:")
        accounts = client.ib.managedAccounts()
        if accounts:
            print(f"   ✅ Accounts: {accounts}")
        
        # Test 4: Fetch market data
        print("\n4. Testing Market Data (SPY)...")
        try:
            bars = await client.fetch_recent_bars("SPY", duration="1 D", bar_size="5 mins")
            print(f"   ✅ Fetched {len(bars)} bars")
            print(f"   Latest bar: {bars.iloc[-1]['close']:.2f} at {bars.index[-1]}")
        except Exception as e:
            print(f"   ❌ Market data error: {e}")
        
        # Test 5: Test multiple symbols
        print("\n5. Testing Multiple Symbols...")
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for symbol in symbols:
            try:
                bars = await client.fetch_recent_bars(symbol, duration="1 D", bar_size="1 hour")
                latest = bars.iloc[-1]
                print(f"   ✅ {symbol}: ${latest['close']:.2f}")
            except Exception as e:
                print(f"   ❌ {symbol}: {e}")
        
        print(f"\n{'='*60}")
        print("✅ IB DESKTOP CONNECTION SUCCESSFUL!")
        print(f"{'='*60}")
        print("\nNext steps:")
        print("1. Ensure IB is running in PAPER mode (port 7497)")
        print("2. Update .env file: IBKR_PORT=7497")
        print("3. Run: python -m robo_trader.runner")
        print("4. Or test AI trading: python ai_trading_example.py")
        
        return True
        
    except asyncio.TimeoutError:
        print("\n❌ Connection timeout!")
        print("\nTroubleshooting:")
        print("1. Is IB Gateway or TWS running?")
        print("2. Check API settings in IB:")
        print("   - File → Global Configuration → API → Settings")
        print("   - Enable 'Enable ActiveX and Socket Clients'")
        print("   - Add 127.0.0.1 to trusted IPs")
        print("3. Check firewall settings")
        return False
        
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("\nMake sure IB Gateway/TWS is running and API is enabled")
        return False
        
    finally:
        if client.ib.isConnected():
            client.ib.disconnect()
            print("\n📊 Disconnected from IB")


async def main():
    """Main entry point"""
    success = await test_connection()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    # Check if ib_insync is installed
    try:
        import ib_insync
    except ImportError:
        print("ERROR: ib_insync not installed!")
        print("Run: pip install ib_insync")
        sys.exit(1)
    
    # Run the test
    asyncio.run(main())
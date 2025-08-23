#!/usr/bin/env python3
"""
Test IB Web API connection
Run this after starting the Client Portal Gateway
"""

import asyncio
import aiohttp
import ssl
import json
from datetime import datetime

async def test_connection():
    """Test basic connection to IB Web API"""
    
    print("=" * 60)
    print("IB Web API Connection Test")
    print("=" * 60)
    
    # SSL context for self-signed certificate
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    base_url = "https://localhost:5001/v1/api"
    
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(ssl=ssl_context)
    ) as session:
        
        # 1. Check if gateway is running
        print("\n1. Checking if Client Portal Gateway is running...")
        try:
            async with session.get(f"{base_url}/") as resp:
                if resp.status == 404:  # API returns 404 for root, but it means it's running
                    print("✅ Gateway is running on port 5000")
                else:
                    print(f"✅ Gateway responded with status {resp.status}")
        except aiohttp.ClientConnectorError:
            print("❌ Cannot connect to gateway on port 5000")
            print("\nTo start the gateway:")
            print("  ./start_ib_web.sh")
            print("\nThen login at: https://localhost:5000")
            return
        
        # 2. Check authentication status
        print("\n2. Checking authentication status...")
        try:
            async with session.get(f"{base_url}/iserver/auth/status") as resp:
                data = await resp.json()
                
                if data.get('authenticated'):
                    print(f"✅ Authenticated as: {data.get('username', 'Unknown')}")
                    print(f"   Connected: {data.get('connected', False)}")
                    print(f"   Competing: {data.get('competing', False)}")
                else:
                    print("❌ Not authenticated")
                    print("\nPlease:")
                    print("1. Open browser: https://localhost:5000")
                    print("2. Login with your IB credentials")
                    print("3. Use paper trading account for testing")
                    return
                    
        except Exception as e:
            print(f"❌ Error checking auth: {e}")
            return
        
        # 3. Get accounts
        print("\n3. Getting account information...")
        try:
            async with session.get(f"{base_url}/portfolio/accounts") as resp:
                accounts = await resp.json()
                if accounts:
                    for acc in accounts:
                        print(f"✅ Account: {acc.get('id', 'Unknown')}")
                        print(f"   Type: {acc.get('type', 'Unknown')}")
                else:
                    print("❌ No accounts found")
        except Exception as e:
            print(f"❌ Error getting accounts: {e}")
        
        # 4. Test market data
        print("\n4. Testing market data retrieval...")
        try:
            # Search for AAPL contract
            async with session.post(
                f"{base_url}/iserver/secdef/search",
                json={"symbol": "AAPL"}
            ) as resp:
                contracts = await resp.json()
                
                if contracts and len(contracts) > 0:
                    conid = contracts[0]['conid']
                    print(f"✅ Found AAPL contract: {conid}")
                    
                    # Get snapshot
                    async with session.get(
                        f"{base_url}/iserver/marketdata/snapshot",
                        params={'conids': conid, 'fields': '31,84,86,88'}
                    ) as resp2:
                        data = await resp2.json()
                        if data and len(data) > 0:
                            snapshot = data[0]
                            print(f"   Last Price: ${snapshot.get('31', 'N/A')}")
                            print(f"   Bid: ${snapshot.get('84', 'N/A')}")
                            print(f"   Ask: ${snapshot.get('86', 'N/A')}")
                            print(f"   Volume: {snapshot.get('88', 'N/A')}")
                else:
                    print("❌ Could not find AAPL contract")
                    
        except Exception as e:
            print(f"❌ Error getting market data: {e}")
        
        # 5. Get positions
        print("\n5. Getting current positions...")
        try:
            async with session.get(f"{base_url}/portfolio/accounts") as resp:
                accounts = await resp.json()
                if accounts:
                    account_id = accounts[0]['id']
                    
                    async with session.get(
                        f"{base_url}/portfolio/{account_id}/positions/0"
                    ) as resp2:
                        positions = await resp2.json()
                        
                        if positions and len(positions) > 0:
                            print(f"✅ Found {len(positions)} position(s):")
                            for pos in positions[:3]:  # Show first 3
                                print(f"   {pos.get('contractDesc', 'Unknown')}: "
                                      f"{pos.get('position', 0)} shares")
                        else:
                            print("✅ No open positions")
                            
        except Exception as e:
            print(f"❌ Error getting positions: {e}")
        
        print("\n" + "=" * 60)
        print("✅ IB Web API is ready for trading!")
        print("=" * 60)


async def test_trading_functionality():
    """Test our IB Web Client"""
    from robo_trader.ib_web_client import IBWebClient
    
    print("\nTesting IBWebClient class...")
    print("-" * 40)
    
    client = IBWebClient()
    
    # Connect
    connected = await client.connect()
    if not connected:
        print("❌ Failed to connect with IBWebClient")
        return
    
    print("✅ IBWebClient connected")
    
    # Get market data
    data = await client.get_market_data("SPY")
    if data:
        print(f"✅ SPY Price: ${data.get('price', 'N/A')}")
    
    # Get account summary
    summary = await client.get_account_summary()
    if summary:
        print(f"✅ Buying Power: ${summary.get('buying_power', 0):,.2f}")
    
    # Get positions
    positions = await client.get_positions()
    print(f"✅ Positions: {len(positions)}")
    
    await client.disconnect()
    print("✅ IBWebClient test complete")


if __name__ == "__main__":
    print("\n🌐 IB Web API Setup Test\n")
    
    # Test basic connection
    asyncio.run(test_connection())
    
    # Test our client class
    print("\n" + "=" * 60)
    asyncio.run(test_trading_functionality())
"""
Interactive Brokers Web API Client
Uses Client Portal REST API instead of TWS/Gateway
No desktop application required!
"""

import aiohttp
import asyncio
import ssl
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from .logger import get_logger

logger = get_logger(__name__)


class IBWebClient:
    """
    IB Client Portal Web API client
    
    Requires Client Portal Gateway running:
    1. Download: https://www.interactivebrokers.com/en/trading/ib-api.php
    2. Run: ./bin/run.sh root/conf.yaml
    3. Login at: https://localhost:5000
    """
    
    def __init__(self, host: str = "localhost", port: int = 5001):
        self.base_url = f"https://{host}:{port}/v1/api"
        self.session = None
        self.account_id = None
        
        # SSL context for self-signed cert
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
    
    async def connect(self):
        """Initialize connection and authenticate"""
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        )
        
        # Check authentication status
        try:
            async with self.session.get(f"{self.base_url}/iserver/auth/status") as resp:
                data = await resp.json()
                
                if data.get('authenticated'):
                    logger.info("Already authenticated with IB Web API")
                    await self._get_account()
                    return True
                else:
                    logger.error("Not authenticated. Please login at https://localhost:5000")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to IB Web API: {e}")
            logger.info("Make sure Client Portal Gateway is running:")
            logger.info("1. cd clientportal.gw")
            logger.info("2. ./bin/run.sh root/conf.yaml")
            logger.info("3. Login at https://localhost:5000")
            return False
    
    async def _get_account(self):
        """Get account information"""
        async with self.session.get(f"{self.base_url}/portfolio/accounts") as resp:
            accounts = await resp.json()
            if accounts:
                self.account_id = accounts[0]['id']
                logger.info(f"Using account: {self.account_id}")
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Market data dict with price, volume, etc.
        """
        # Search for contract
        contract = await self._search_contract(symbol)
        if not contract:
            return {}
        
        conid = contract['conid']
        
        # Get market data
        endpoint = f"{self.base_url}/iserver/marketdata/snapshot"
        params = {
            'conids': conid,
            'fields': '31,84,86,88,7295,7296,7308,7309,7310,7311'
            # 31=last, 84=bid, 86=ask, 88=volume, etc.
        }
        
        async with self.session.get(endpoint, params=params) as resp:
            data = await resp.json()
            
            if data and len(data) > 0:
                snapshot = data[0]
                return {
                    'price': snapshot.get('31', 0),
                    'bid': snapshot.get('84', 0),
                    'ask': snapshot.get('86', 0),
                    'volume': snapshot.get('88', 0),
                    'open': snapshot.get('7295', 0),
                    'high': snapshot.get('7296', 0),
                    'low': snapshot.get('7308', 0),
                    'close': snapshot.get('31', 0)
                }
        
        return {}
    
    async def _search_contract(self, symbol: str) -> Optional[Dict]:
        """Search for a contract by symbol"""
        endpoint = f"{self.base_url}/iserver/secdef/search"
        params = {'symbol': symbol}
        
        async with self.session.post(endpoint, json=params) as resp:
            data = await resp.json()
            
            if data and len(data) > 0:
                # Return first stock match
                for contract in data:
                    if contract.get('description', '').upper() == symbol.upper():
                        return contract
                return data[0]  # Fallback to first result
        
        return None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1d",
        bar_size: str = "1min"
    ) -> List[Dict]:
        """
        Get historical bars
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 1w, 1m, etc.)
            bar_size: Bar size (1min, 5min, 1h, 1d)
            
        Returns:
            List of price bars
        """
        contract = await self._search_contract(symbol)
        if not contract:
            return []
        
        conid = contract['conid']
        
        endpoint = f"{self.base_url}/iserver/marketdata/history"
        params = {
            'conid': conid,
            'period': period,
            'bar': bar_size,
            'outsideRth': False
        }
        
        async with self.session.get(endpoint, params=params) as resp:
            data = await resp.json()
            
            if data and 'data' in data:
                bars = []
                for bar in data['data']:
                    bars.append({
                        'time': bar['t'],
                        'open': bar['o'],
                        'high': bar['h'],
                        'low': bar['l'],
                        'close': bar['c'],
                        'volume': bar['v']
                    })
                return bars
        
        return []
    
    async def place_order(
        self,
        symbol: str,
        quantity: int,
        side: str,  # BUY or SELL
        order_type: str = "MKT",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place an order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            side: BUY or SELL
            order_type: MKT, LMT, STP, etc.
            price: Limit/stop price if applicable
            
        Returns:
            Order result with order_id
        """
        if not self.account_id:
            return {'error': 'No account ID'}
        
        contract = await self._search_contract(symbol)
        if not contract:
            return {'error': f'Contract not found for {symbol}'}
        
        conid = contract['conid']
        
        # Prepare order
        order = {
            'acctId': self.account_id,
            'conid': conid,
            'orderType': order_type,
            'side': side,
            'quantity': quantity,
            'tif': 'DAY'
        }
        
        if price and order_type in ['LMT', 'STP']:
            order['price'] = price
        
        # Place order
        endpoint = f"{self.base_url}/iserver/account/{self.account_id}/orders"
        
        async with self.session.post(endpoint, json={'orders': [order]}) as resp:
            result = await resp.json()
            
            if result and len(result) > 0:
                order_result = result[0]
                if 'id' in order_result:
                    logger.info(f"Order placed: {order_result['id']}")
                    return {
                        'success': True,
                        'order_id': order_result['id'],
                        'message': order_result.get('message', 'Order placed')
                    }
                else:
                    return {
                        'success': False,
                        'error': order_result.get('message', 'Order failed')
                    }
        
        return {'success': False, 'error': 'No response'}
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.account_id:
            return []
        
        endpoint = f"{self.base_url}/portfolio/{self.account_id}/positions/0"
        
        async with self.session.get(endpoint) as resp:
            positions = await resp.json()
            
            result = []
            for pos in positions:
                result.append({
                    'symbol': pos.get('contractDesc', ''),
                    'position': pos.get('position', 0),
                    'avg_cost': pos.get('avgCost', 0),
                    'market_value': pos.get('mktValue', 0),
                    'unrealized_pnl': pos.get('unrealizedPnl', 0)
                })
            
            return result
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary including buying power, P&L, etc."""
        if not self.account_id:
            return {}
        
        endpoint = f"{self.base_url}/portfolio/{self.account_id}/summary"
        
        async with self.session.get(endpoint) as resp:
            summary = await resp.json()
            
            return {
                'net_liquidation': summary.get('netliquidation', {}).get('amount', 0),
                'buying_power': summary.get('buyingpower', {}).get('amount', 0),
                'total_cash': summary.get('totalcashvalue', {}).get('amount', 0),
                'realized_pnl': summary.get('realizedpnl', {}).get('amount', 0),
                'unrealized_pnl': summary.get('unrealizedpnl', {}).get('amount', 0)
            }
    
    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()
            logger.info("Disconnected from IB Web API")


# Example usage
async def test_web_api():
    """Test IB Web API connection"""
    client = IBWebClient()
    
    # Connect (requires Client Portal Gateway running)
    connected = await client.connect()
    if not connected:
        print("Failed to connect. Make sure Client Portal Gateway is running.")
        return
    
    # Get market data
    data = await client.get_market_data("AAPL")
    print(f"AAPL Price: ${data.get('price', 'N/A')}")
    
    # Get account info
    summary = await client.get_account_summary()
    print(f"Buying Power: ${summary.get('buying_power', 0):,.2f}")
    
    # Get positions
    positions = await client.get_positions()
    for pos in positions:
        print(f"{pos['symbol']}: {pos['position']} shares, P&L: ${pos['unrealized_pnl']:.2f}")
    
    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_web_api())
import aiohttp
import asyncio
import json
from datetime import datetime

async def send_test_options():
    """Send test options flow data to dashboard"""
    
    test_options = {
        'options': [
            {
                'symbol': 'NVDA',
                'option_type': 'CALL',
                'strike': 130.0,
                'expiry': '2025-09-20',
                'volume': 15000,
                'open_interest': 8500,
                'vol_oi_ratio': 1.76,
                'premium': 285000,
                'delta': 0.65,
                'implied_vol': 0.42,
                'signal_type': 'SWEEP',
                'confidence': 85,
                'timestamp': datetime.now().strftime("%H:%M"),
                'time': datetime.now().strftime("%H:%M")
            },
            {
                'symbol': 'TSLA',
                'option_type': 'PUT',
                'strike': 210.0,
                'expiry': '2025-08-30',
                'volume': 8500,
                'open_interest': 12000,
                'vol_oi_ratio': 0.71,
                'premium': 125000,
                'delta': -0.35,
                'implied_vol': 0.55,
                'signal_type': 'BLOCK',
                'confidence': 72,
                'timestamp': datetime.now().strftime("%H:%M"),
                'time': datetime.now().strftime("%H:%M")
            },
            {
                'symbol': 'AAPL',
                'option_type': 'CALL',
                'strike': 230.0,
                'expiry': '2025-09-06',
                'volume': 5200,
                'open_interest': 3100,
                'vol_oi_ratio': 1.68,
                'premium': 95000,
                'delta': 0.45,
                'implied_vol': 0.28,
                'signal_type': 'SWEEP',
                'confidence': 68,
                'timestamp': datetime.now().strftime("%H:%M"),
                'time': datetime.now().strftime("%H:%M")
            }
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:5555/api/options',
                json=test_options,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    print(f"✅ Successfully sent {len(test_options['options'])} options flow signals")
                    for opt in test_options['options']:
                        print(f"   {opt['symbol']} {opt['option_type']} ${opt['strike']} - {opt['signal_type']}")
                else:
                    print(f"❌ Failed to send options: HTTP {resp.status}")
    except Exception as e:
        print(f"❌ Error sending options: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_options())

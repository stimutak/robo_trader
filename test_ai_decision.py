import aiohttp
import asyncio
import json
from datetime import datetime

async def send_test_decision():
    """Send a test AI decision with full decision tree data to dashboard"""
    
    test_decision = {
        'decision': {
            'symbol': 'NVDA',
            'action': 'BUY',
            'confidence': 75,
            'reason': 'Strong bullish options flow detected with unusual call volume at $130 strike. Smart money positioning for upside move.',
            'time': datetime.now().strftime('%H:%M:%S'),
            'event_type': 'OPTIONS_FLOW',
            'latency': 245,
            'entry_price': 127.50,
            'stops': {'stop_loss': 125.00, 'technical': 124.50},
            'targets': [130.00, 132.50, 135.00],
            'thesis': {
                'setup': 'Unusual call sweep activity detected',
                'catalyst': 'AI earnings momentum continuation',
                'risk_reward': '1:3.2',
                'prob_win': 65
            },
            'watchlist': [
                {'symbol': 'AMD', 'trigger_above': 165, 'notes': 'Sympathy play'},
                {'symbol': 'SMCI', 'trigger_above': 38, 'notes': 'AI infrastructure'}
            ],
            'raw_decision': {
                'mode': 'trade',
                'universe_checked': ['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT'],
                'conviction': 75,
                'aggressiveness_level': 2,
                'compliance_checks': {
                    'liquidity_ok': True,
                    'spread_ok': True,
                    'borrow_ok': True,
                    'correlation_ok': True
                },
                'risk_state': {
                    'day_dd_bps': 45,
                    'week_dd_bps': 120,
                    'cash_pct': 65.5,
                    'open_positions': 3,
                    'total_exposure_pct': 34.5
                },
                'recommendation': {
                    'symbol': 'NVDA',
                    'direction': 'long',
                    'entry_type': 'limit',
                    'entry_price': 127.50,
                    'position_size_bps': 150,
                    'stop_loss': 125.00,
                    'time_stop_hours': 48,
                    'targets': [130.00, 132.50, 135.00],
                    'thesis': 'Unusual options activity suggests institutional accumulation ahead of next catalyst',
                    'risk_reward': 3.2,
                    'p_win': 0.65,
                    'expected_value_pct': 12.5
                }
            }
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:5555/api/ai_decision',
                json=test_decision,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                if resp.status == 200:
                    print(f"✅ Successfully sent test AI decision for {test_decision['decision']['symbol']}")
                    print(f"   Action: {test_decision['decision']['action']}")
                    print(f"   Confidence: {test_decision['decision']['confidence']}%")
                    print(f"   Has decision tree: {'raw_decision' in test_decision['decision']}")
                else:
                    print(f"❌ Failed to send decision: HTTP {resp.status}")
    except Exception as e:
        print(f"❌ Error sending decision: {e}")

if __name__ == "__main__":
    asyncio.run(send_test_decision())

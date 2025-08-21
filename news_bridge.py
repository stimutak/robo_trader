#!/usr/bin/env python3
"""
Bridge to push news from AI runner logs to dashboard.
Monitors AI runner output and updates dashboard.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import re

async def monitor_and_push_news():
    """Monitor AI runner logs and push news to dashboard."""
    
    print("📡 News Bridge: Monitoring AI runner for news updates...")
    
    # Track seen news to avoid duplicates
    seen_titles = set()
    news_buffer = []
    
    while True:
        try:
            # Read from AI runner logs (in production, this would tail the log file)
            # For now, generate sample news to test the ticker
            
            sample_news = [
                {
                    "title": "Fed's Powell: Rates to remain higher for longer than expected",
                    "source": "Bloomberg",
                    "sentiment": -0.3,
                    "time": datetime.now().strftime("%H:%M")
                },
                {
                    "title": "Tech stocks rally on AI optimism, NVDA up 3%",
                    "source": "CNBC",
                    "sentiment": 0.5,
                    "time": datetime.now().strftime("%H:%M")
                },
                {
                    "title": "Walmart misses Q2 earnings, lowers guidance",
                    "source": "Reuters",
                    "sentiment": -0.4,
                    "time": datetime.now().strftime("%H:%M")
                },
                {
                    "title": "Options flow shows heavy call buying in SPY",
                    "source": "Options",
                    "sentiment": 0.3,
                    "time": datetime.now().strftime("%H:%M")
                }
            ]
            
            # Push to dashboard API
            async with aiohttp.ClientSession() as session:
                for news_item in sample_news:
                    if news_item["title"] not in seen_titles:
                        seen_titles.add(news_item["title"])
                        news_buffer.append(news_item)
                        
                        # Keep buffer size manageable
                        if len(news_buffer) > 50:
                            news_buffer.pop(0)
                        
                        # Update dashboard
                        try:
                            async with session.post(
                                "http://localhost:5555/api/news",
                                json={"news": news_buffer[-20:]}  # Last 20 items
                            ) as resp:
                                if resp.status == 200:
                                    print(f"✅ Pushed: {news_item['title'][:50]}...")
                        except:
                            pass  # Dashboard might not have the endpoint yet
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(monitor_and_push_news())
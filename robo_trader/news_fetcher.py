"""
News fetcher using RSS feeds - no API keys required.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import feedparser

logger = logging.getLogger(__name__)

# Free RSS feeds for financial news - diverse sources for AI discovery
RSS_FEEDS = {
    # Major financial news
    "Yahoo Finance": "https://finance.yahoo.com/rss/topfinstories",
    "Yahoo Tech": "https://finance.yahoo.com/rss/industry?s=technology",
    "Reuters Markets": "https://feeds.reuters.com/reuters/USMarketsNews",
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "CNBC Top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC Investing": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
    # Stock-specific sources
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Stocks": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
    "Seeking Alpha": "https://seekingalpha.com/market_currents.xml",
    "Seeking Alpha News": "https://seekingalpha.com/feed.xml",
    # Tech and growth stocks
    "TechCrunch": "https://techcrunch.com/feed/",
    "Benzinga": "https://www.benzinga.com/feed",
}


def fetch_rss_news(max_items: int = 20) -> List[Dict]:
    """Fetch news from RSS feeds."""
    all_news = []

    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:8]:
                published = datetime.now()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                title = entry.title if hasattr(entry, "title") else "No title"
                title = title.replace("&apos;", "'").replace("&quot;", '"').replace("&amp;", "&")

                news_item = {
                    "title": title[:100],
                    "source": source,
                    "time": published.strftime("%H:%M"),
                    "url": entry.link if hasattr(entry, "link") else "#",
                    "sentiment": analyze_sentiment(title),
                }

                all_news.append(news_item)

        except Exception as e:
            logger.warning(f"Failed to fetch RSS from {source}: {e}")

    return all_news[:max_items]


def analyze_sentiment(text: str) -> float:
    """Simple sentiment analysis."""
    text_lower = text.lower()

    positive = [
        "gains",
        "rises",
        "jumps",
        "surges",
        "rallies",
        "climbs",
        "soars",
        "beat",
        "upgrade",
    ]
    negative = [
        "falls",
        "drops",
        "plunges",
        "crashes",
        "declines",
        "slumps",
        "miss",
        "downgrade",
    ]

    pos_count = sum(1 for word in positive if word in text_lower)
    neg_count = sum(1 for word in negative if word in text_lower)

    if pos_count > neg_count:
        return min(0.8, pos_count * 0.3)
    elif neg_count > pos_count:
        return max(-0.8, -neg_count * 0.3)
    return 0.0

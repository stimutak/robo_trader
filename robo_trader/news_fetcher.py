"""
News fetcher using RSS feeds - no API keys required.
"""

import feedparser
from datetime import datetime, timedelta
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Free RSS feeds for financial news
RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/rss/topfinstories",
    "Reuters Markets": "https://feeds.reuters.com/reuters/USMarketsNews",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
}


def fetch_rss_news(max_items: int = 20) -> List[Dict]:
    """Fetch news from RSS feeds."""
    all_news = []

    for source, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)

            for entry in feed.entries[:5]:
                published = datetime.now()
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                title = entry.title if hasattr(entry, "title") else "No title"
                title = (
                    title.replace("&apos;", "'")
                    .replace("&quot;", '"')
                    .replace("&amp;", "&")
                )

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

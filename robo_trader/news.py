"""
News ingestion pipeline for real-time market intelligence.

Fetches and processes news from multiple sources:
- RSS feeds (Yahoo Finance, Reuters, Bloomberg)
- Financial APIs (when configured)
- Filters for relevance and deduplicates
"""

import asyncio
import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import feedparser
import aiohttp
from robo_trader.logger import get_logger
from robo_trader.sentiment import SimpleSentimentAnalyzer

logger = get_logger(__name__)


@dataclass
class NewsItem:
    """Represents a single news item."""
    title: str
    summary: str
    url: str
    source: str
    published: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    hash_id: str = field(init=False)
    
    def __post_init__(self):
        """Generate unique hash ID for deduplication."""
        content = f"{self.title}{self.summary}"
        self.hash_id = hashlib.md5(content.encode()).hexdigest()[:12]


class NewsAggregator:
    """Aggregates news from multiple sources with deduplication and filtering."""
    
    # Major RSS feeds for financial news
    RSS_FEEDS = {
        "yahoo_market": "https://finance.yahoo.com/news/rssindex",
        "yahoo_top": "https://finance.yahoo.com/rss/topfinstories",
        "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
        "reuters_markets": "https://feeds.reuters.com/reuters/marketsNews",
        "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
        "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "wsj_markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
    }
    
    # Symbol-specific RSS (template URLs)
    SYMBOL_RSS_TEMPLATE = {
        "yahoo": "https://finance.yahoo.com/rss/headline?s={symbol}",
    }
    
    def __init__(self, symbols: List[str], lookback_hours: int = 24):
        """
        Initialize news aggregator.
        
        Args:
            symbols: List of stock symbols to track
            lookback_hours: How many hours of news to fetch initially
        """
        self.symbols = [s.upper() for s in symbols]
        self.lookback_hours = lookback_hours
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        self.seen_hashes: Set[str] = set()
        self.news_cache: List[NewsItem] = []
        
    def _parse_published_date(self, entry: Dict) -> Optional[datetime]:
        """Parse various date formats from RSS feeds."""
        try:
            if hasattr(entry, 'published_parsed'):
                import time
                return datetime.fromtimestamp(time.mktime(entry.published_parsed), tz=timezone.utc)
            elif hasattr(entry, 'published'):
                from dateutil import parser
                return parser.parse(entry.published)
            elif hasattr(entry, 'updated'):
                from dateutil import parser
                return parser.parse(entry.updated)
        except Exception as e:
            logger.debug(f"Could not parse date: {e}")
        return datetime.now(timezone.utc)
    
    def _extract_symbols_from_text(self, text: str) -> List[str]:
        """Extract stock symbols mentioned in text."""
        found_symbols = []
        
        # Look for exact symbol matches (case-insensitive)
        text_upper = text.upper()
        for symbol in self.symbols:
            # Check for symbol with word boundaries
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text_upper):
                found_symbols.append(symbol)
        
        # Also look for common patterns like $TSLA or (NASDAQ:TSLA)
        ticker_patterns = [
            r'\$([A-Z]{1,5})\b',  # $TSLA
            r'\(([A-Z]{1,5})\)',   # (TSLA)
            r'(?:NYSE|NASDAQ|NYSEARCA)[:\s]+([A-Z]{1,5})',  # NYSE:TSLA
        ]
        
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text_upper)
            for match in matches:
                if match in self.symbols and match not in found_symbols:
                    found_symbols.append(match)
        
        return found_symbols
    
    def _calculate_relevance(self, item: NewsItem) -> float:
        """
        Calculate relevance score for a news item.
        
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        
        # Symbol mentions (most important)
        if item.symbols:
            score += 0.5 * min(len(item.symbols) / 2, 1.0)
        
        # Keywords that indicate market-moving news
        important_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'fda', 'approval', 'merger', 'acquisition', 'buyout', 'ipo',
            'fed', 'federal reserve', 'interest rate', 'inflation', 'gdp',
            'breaking', 'alert', 'urgent', 'exclusive', 'investigation',
            'recall', 'lawsuit', 'sec', 'probe', 'bankruptcy', 'layoff'
        ]
        
        text_lower = f"{item.title} {item.summary}".lower()
        keyword_count = sum(1 for kw in important_keywords if kw in text_lower)
        score += 0.3 * min(keyword_count / 3, 1.0)
        
        # Recency bonus
        age_hours = (datetime.now(timezone.utc) - item.published).total_seconds() / 3600
        if age_hours < 1:
            score += 0.2
        elif age_hours < 6:
            score += 0.1
        
        return min(score, 1.0)
    
    async def fetch_rss_feed(self, url: str, source_name: str) -> List[NewsItem]:
        """Fetch and parse a single RSS feed."""
        items = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        feed = feedparser.parse(content)
                        
                        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
                        
                        for entry in feed.entries[:50]:  # Limit entries per feed
                            published = self._parse_published_date(entry)
                            
                            # Skip old news
                            if published < cutoff_time:
                                continue
                            
                            title = entry.get('title', '').strip()
                            summary = entry.get('summary', entry.get('description', '')).strip()
                            
                            # Clean HTML from summary
                            summary = re.sub('<[^<]+?>', '', summary)[:500]
                            
                            if not title:
                                continue
                            
                            item = NewsItem(
                                title=title,
                                summary=summary,
                                url=entry.get('link', ''),
                                source=source_name,
                                published=published
                            )
                            
                            # Extract symbols
                            full_text = f"{title} {summary}"
                            item.symbols = self._extract_symbols_from_text(full_text)
                            
                            # Calculate scores
                            sentiment_result = self.sentiment_analyzer.analyze(full_text)
                            item.sentiment_score = sentiment_result.score
                            item.relevance_score = self._calculate_relevance(item)
                            
                            items.append(item)
                            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {source_name}")
        except Exception as e:
            logger.error(f"Error fetching {source_name}: {e}")
        
        return items
    
    async def fetch_all_feeds(self) -> List[NewsItem]:
        """Fetch news from all configured RSS feeds."""
        tasks = []
        
        # General market feeds
        for source_name, url in self.RSS_FEEDS.items():
            tasks.append(self.fetch_rss_feed(url, source_name))
        
        # Symbol-specific feeds
        for symbol in self.symbols[:5]:  # Limit to avoid too many requests
            yahoo_url = self.SYMBOL_RSS_TEMPLATE["yahoo"].format(symbol=symbol)
            tasks.append(self.fetch_rss_feed(yahoo_url, f"yahoo_{symbol}"))
        
        # Fetch all feeds concurrently
        results = await asyncio.gather(*tasks)
        
        # Flatten results and deduplicate
        all_items = []
        for items in results:
            for item in items:
                if item.hash_id not in self.seen_hashes:
                    self.seen_hashes.add(item.hash_id)
                    all_items.append(item)
        
        # Sort by relevance and recency
        all_items.sort(key=lambda x: (x.relevance_score, -x.published.timestamp()), reverse=True)
        
        return all_items
    
    def get_high_impact_news(self, min_relevance: float = 0.3) -> List[NewsItem]:
        """
        Get high-impact news items for AI analysis.
        
        Args:
            min_relevance: Minimum relevance score threshold
        
        Returns:
            List of high-impact news items
        """
        return [item for item in self.news_cache if item.relevance_score >= min_relevance]
    
    def get_symbol_news(self, symbol: str) -> List[NewsItem]:
        """Get news for a specific symbol."""
        symbol = symbol.upper()
        return [item for item in self.news_cache if symbol in item.symbols]
    
    async def update(self) -> Tuple[int, int]:
        """
        Fetch latest news and update cache.
        
        Returns:
            Tuple of (new_items_count, total_items_count)
        """
        logger.info(f"Fetching news for symbols: {self.symbols}")
        
        # Fetch new items
        new_items = await self.fetch_all_feeds()
        
        # Add to cache and keep recent items only
        self.news_cache.extend(new_items)
        
        # Remove old items (keep last 24 hours)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        self.news_cache = [item for item in self.news_cache if item.published > cutoff]
        
        # Re-sort by relevance
        self.news_cache.sort(key=lambda x: (x.relevance_score, -x.published.timestamp()), reverse=True)
        
        logger.info(f"News update: {len(new_items)} new, {len(self.news_cache)} total in cache")
        
        # Log some high-impact news
        high_impact = self.get_high_impact_news(min_relevance=0.5)
        if high_impact:
            logger.info(f"High-impact news ({len(high_impact)} items):")
            for item in high_impact[:3]:
                logger.info(f"  - [{item.source}] {item.title[:80]}... (relevance: {item.relevance_score:.2f})")
        
        return len(new_items), len(self.news_cache)
    
    def format_for_ai(self, max_items: int = 10) -> str:
        """
        Format top news items for AI analysis.
        
        Args:
            max_items: Maximum number of items to include
        
        Returns:
            Formatted string for AI prompt
        """
        high_impact = self.get_high_impact_news(min_relevance=0.3)[:max_items]
        
        if not high_impact:
            return "No significant market news in the last 24 hours."
        
        formatted = "Recent Market News:\n\n"
        for i, item in enumerate(high_impact, 1):
            age = datetime.now(timezone.utc) - item.published
            age_str = f"{int(age.total_seconds() / 60)}m ago" if age.total_seconds() < 3600 else f"{int(age.total_seconds() / 3600)}h ago"
            
            formatted += f"{i}. [{item.source}] {age_str}\n"
            formatted += f"   Title: {item.title}\n"
            if item.summary:
                formatted += f"   Summary: {item.summary[:200]}...\n"
            if item.symbols:
                formatted += f"   Symbols: {', '.join(item.symbols)}\n"
            formatted += f"   Sentiment: {item.sentiment_score:.2f} | Relevance: {item.relevance_score:.2f}\n\n"
        
        return formatted


async def main():
    """Test the news aggregator."""
    import sys
    
    # Test with some popular symbols
    symbols = ["AAPL", "TSLA", "NVDA", "SPY", "QQQ"]
    aggregator = NewsAggregator(symbols, lookback_hours=6)
    
    print("Fetching news feeds...")
    new_count, total_count = await aggregator.update()
    print(f"\nFetched {new_count} news items, {total_count} total in cache")
    
    # Show high-impact news
    high_impact = aggregator.get_high_impact_news(min_relevance=0.4)
    print(f"\nHigh-impact news ({len(high_impact)} items):")
    for item in high_impact[:5]:
        print(f"\n[{item.source}] {item.title}")
        print(f"  Symbols: {', '.join(item.symbols) if item.symbols else 'None'}")
        print(f"  Relevance: {item.relevance_score:.2f} | Sentiment: {item.sentiment_score:.2f}")
        print(f"  Published: {item.published.strftime('%Y-%m-%d %H:%M UTC')}")
    
    # Show formatted for AI
    print("\n" + "="*60)
    print("Formatted for AI Analysis:")
    print("="*60)
    print(aggregator.format_for_ai(max_items=5))


if __name__ == "__main__":
    asyncio.run(main())
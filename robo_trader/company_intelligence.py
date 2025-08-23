#!/usr/bin/env python3
"""
Company Intelligence Module - SEC Filings, Earnings, FDA, Insider Trading
Provides real-time company-specific event monitoring for trading signals
"""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import re
from dataclasses import dataclass
from enum import Enum

from .logger import get_logger
from .database import TradingDatabase

logger = get_logger(__name__)


class EventType(Enum):
    """Types of company events we monitor"""
    SEC_8K = "8-K Filing"
    SEC_10Q = "10-Q Filing"
    SEC_10K = "10-K Filing"
    FORM_4 = "Insider Trade"
    EARNINGS = "Earnings Report"
    FDA_APPROVAL = "FDA Decision"
    PRESS_RELEASE = "Press Release"
    ANALYST_UPGRADE = "Analyst Action"
    GUIDANCE_UPDATE = "Guidance Update"


@dataclass
class CompanyEvent:
    """Represents a company-specific event"""
    symbol: str
    event_type: EventType
    headline: str
    description: str
    timestamp: datetime
    impact_score: float  # 0-100 importance
    url: Optional[str] = None
    metadata: Optional[Dict] = None


class SECEdgarClient:
    """Fetches SEC filings from EDGAR system"""
    
    BASE_URL = "https://www.sec.gov"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions"
    
    def __init__(self):
        self.session = None
        self.cik_map = {}  # symbol -> CIK mapping
        # SEC requires a User-Agent with contact email
        self.headers = {
            'User-Agent': 'RoboTrader/1.0 (contact@roboai.trading)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
    
    async def initialize(self, symbols: List[str]):
        """Initialize with company symbols"""
        self.session = aiohttp.ClientSession(headers=self.headers)
        await self._load_cik_mapping(symbols)
    
    async def _load_cik_mapping(self, symbols: List[str]):
        """Load CIK (Central Index Key) for each symbol"""
        try:
            # SEC provides a ticker to CIK mapping file
            url = f"{self.BASE_URL}/files/company_tickers.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Build symbol -> CIK map
                    for item in data.values():
                        ticker = item.get('ticker', '').upper()
                        cik = str(item.get('cik_str', '')).zfill(10)
                        if ticker in symbols:
                            self.cik_map[ticker] = cik
                            logger.info(f"Mapped {ticker} -> CIK {cik}")
                    
                    logger.info(f"Loaded CIK mappings for {len(self.cik_map)} symbols")
        except Exception as e:
            logger.error(f"Error loading CIK mappings: {e}")
    
    async def get_recent_filings(self, symbol: str, hours: int = 24) -> List[CompanyEvent]:
        """Get recent SEC filings for a symbol"""
        events = []
        
        if symbol not in self.cik_map:
            return events
        
        cik = self.cik_map[symbol]
        
        try:
            # Get company submissions
            url = f"{self.SUBMISSIONS_URL}/CIK{cik}.json"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    recent_filings = data.get('filings', {}).get('recent', {})
                    
                    # Process recent filings
                    forms = recent_filings.get('form', [])
                    filing_dates = recent_filings.get('filingDate', [])
                    primary_docs = recent_filings.get('primaryDocument', [])
                    descriptions = recent_filings.get('primaryDocDescription', [])
                    
                    cutoff_date = datetime.now() - timedelta(hours=hours)
                    
                    for i in range(min(20, len(forms))):  # Check last 20 filings
                        form_type = forms[i]
                        filing_date_str = filing_dates[i] if i < len(filing_dates) else ""
                        
                        # Parse filing date
                        try:
                            filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                        except:
                            continue
                        
                        if filing_date < cutoff_date:
                            continue
                        
                        # Determine event type and importance
                        event_type = None
                        impact_score = 50
                        
                        if form_type == "8-K":
                            event_type = EventType.SEC_8K
                            impact_score = 80  # 8-Ks are important events
                        elif form_type == "10-Q":
                            event_type = EventType.SEC_10Q
                            impact_score = 70
                        elif form_type == "10-K":
                            event_type = EventType.SEC_10K
                            impact_score = 75
                        elif form_type == "4":
                            event_type = EventType.FORM_4
                            impact_score = 60  # Insider trading
                        else:
                            continue  # Skip other form types for now
                        
                        description = descriptions[i] if i < len(descriptions) else form_type
                        
                        # Build filing URL
                        accession = recent_filings.get('accessionNumber', [])[i].replace('-', '')
                        doc = primary_docs[i] if i < len(primary_docs) else ""
                        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"
                        
                        event = CompanyEvent(
                            symbol=symbol,
                            event_type=event_type,
                            headline=f"{symbol} files {form_type}",
                            description=description,
                            timestamp=filing_date,
                            impact_score=impact_score,
                            url=filing_url,
                            metadata={'form_type': form_type, 'cik': cik}
                        )
                        
                        events.append(event)
                        logger.info(f"Found {form_type} filing for {symbol}: {description}")
        
        except Exception as e:
            logger.error(f"Error fetching SEC filings for {symbol}: {e}")
        
        return events
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class EarningsCalendar:
    """Monitors earnings announcements and surprises"""
    
    def __init__(self):
        self.session = None
        # We'll use Yahoo Finance earnings calendar as it's free
        self.base_url = "https://query1.finance.yahoo.com/v1/finance"
    
    async def initialize(self):
        """Initialize the earnings calendar"""
        self.session = aiohttp.ClientSession()
    
    async def get_upcoming_earnings(self, symbol: str, days: int = 7) -> List[CompanyEvent]:
        """Get upcoming earnings for a symbol"""
        events = []
        
        try:
            # Yahoo Finance calendar endpoint
            url = f"{self.base_url}/calendar/earnings"
            params = {
                'symbol': symbol,
                'from': datetime.now().strftime('%Y-%m-%d'),
                'to': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Parse earnings data
                    earnings = data.get('finance', {}).get('result', [])
                    for item in earnings:
                        if item.get('symbol') == symbol:
                            earnings_date = item.get('startdatetime')
                            if earnings_date:
                                event = CompanyEvent(
                                    symbol=symbol,
                                    event_type=EventType.EARNINGS,
                                    headline=f"{symbol} Earnings Report",
                                    description=f"Earnings call scheduled",
                                    timestamp=datetime.fromisoformat(earnings_date),
                                    impact_score=90,  # Earnings are high impact
                                    metadata={'eps_estimate': item.get('epsestimate')}
                                )
                                events.append(event)
                                logger.info(f"Found earnings date for {symbol}: {earnings_date}")
        
        except Exception as e:
            logger.error(f"Error fetching earnings for {symbol}: {e}")
        
        return events
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class FDACalendar:
    """Monitors FDA approval dates for biotech/pharma stocks"""
    
    BIOTECH_SYMBOLS = ['IXHL', 'ELTP', 'BZAI', 'NUAI']  # Our biotech holdings
    
    def __init__(self):
        self.session = None
    
    async def initialize(self):
        """Initialize FDA calendar"""
        self.session = aiohttp.ClientSession()
    
    async def get_fda_events(self, symbol: str) -> List[CompanyEvent]:
        """Get FDA-related events for biotech stocks"""
        events = []
        
        if symbol not in self.BIOTECH_SYMBOLS:
            return events
        
        try:
            # For now, we'll use press releases and news feeds
            # In production, you'd want to use BioPharmCatalyst API or similar
            
            # FDA RSS feed for drug approvals
            rss_url = "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-approvals-and-databases/rss.xml"
            
            feed = await self._fetch_rss_feed(rss_url)
            
            for entry in feed.entries[:10]:  # Check recent entries
                # Look for company mentions in FDA news
                if symbol in entry.title.upper() or symbol in entry.summary.upper():
                    event = CompanyEvent(
                        symbol=symbol,
                        event_type=EventType.FDA_APPROVAL,
                        headline=entry.title,
                        description=entry.summary[:200],
                        timestamp=datetime.now(),  # Parse entry.published
                        impact_score=95,  # FDA decisions are very high impact
                        url=entry.link
                    )
                    events.append(event)
                    logger.info(f"Found FDA event for {symbol}: {entry.title}")
        
        except Exception as e:
            logger.error(f"Error fetching FDA events for {symbol}: {e}")
        
        return events
    
    async def _fetch_rss_feed(self, url: str):
        """Fetch and parse RSS feed"""
        async with self.session.get(url) as response:
            content = await response.text()
            return feedparser.parse(content)
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()


class CompanyIntelligence:
    """Main orchestrator for company-specific intelligence"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.sec_client = SECEdgarClient()
        self.earnings_calendar = EarningsCalendar()
        self.fda_calendar = FDACalendar()
        self.db = TradingDatabase()
        self.events_cache = {}  # Prevent duplicate alerts
    
    async def initialize(self):
        """Initialize all data sources"""
        logger.info(f"Initializing Company Intelligence for {len(self.symbols)} symbols")
        
        await self.sec_client.initialize(self.symbols)
        await self.earnings_calendar.initialize()
        await self.fda_calendar.initialize()
    
    async def fetch_all_events(self) -> List[CompanyEvent]:
        """Fetch events from all sources"""
        all_events = []
        
        tasks = []
        for symbol in self.symbols:
            # SEC filings - look back 7 days for now to get more data
            tasks.append(self.sec_client.get_recent_filings(symbol, hours=168))
            # Earnings
            tasks.append(self.earnings_calendar.get_upcoming_earnings(symbol, days=7))
            # FDA (only for biotech)
            if symbol in FDACalendar.BIOTECH_SYMBOLS:
                tasks.append(self.fda_calendar.get_fda_events(symbol))
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_events.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching events: {result}")
        
        # Filter duplicates
        unique_events = []
        for event in all_events:
            event_key = f"{event.symbol}_{event.event_type.value}_{event.headline}"
            if event_key not in self.events_cache:
                self.events_cache[event_key] = True
                unique_events.append(event)
        
        # Sort by impact score
        unique_events.sort(key=lambda x: x.impact_score, reverse=True)
        
        logger.info(f"Found {len(unique_events)} unique company events")
        
        return unique_events
    
    async def monitor_continuously(self, callback=None):
        """Monitor for new events continuously"""
        logger.info("Starting continuous company event monitoring")
        
        while True:
            try:
                events = await self.fetch_all_events()
                
                # Process high-impact events
                high_impact_events = [e for e in events if e.impact_score >= 70]
                
                if high_impact_events:
                    logger.info(f"Found {len(high_impact_events)} high-impact events")
                    
                    for event in high_impact_events:
                        logger.info(
                            f"🎯 {event.symbol} - {event.event_type.value}: "
                            f"{event.headline} (Impact: {event.impact_score})"
                        )
                        
                        # Store in database
                        self._store_event(event)
                        
                        # Callback for AI processing
                        if callback:
                            await callback(event)
                
                # Check every 5 minutes for SEC filings
                await asyncio.sleep(300)
            
            except Exception as e:
                logger.error(f"Error in event monitoring: {e}")
                await asyncio.sleep(60)
    
    def _store_event(self, event: CompanyEvent):
        """Store event in database for historical analysis"""
        try:
            # Store as AI decision for now (we can add a dedicated events table later)
            self.db.save_ai_decision(
                symbol=event.symbol,
                analysis=event.description,
                confidence=event.impact_score,
                decision=event.event_type.value,
                reasoning=f"Company event detected: {event.headline}",
                outcome="pending"
            )
        except Exception as e:
            logger.error(f"Error storing event: {e}")
    
    async def close(self):
        """Clean up resources"""
        await self.sec_client.close()
        await self.earnings_calendar.close()
        await self.fda_calendar.close()


async def test_company_intelligence():
    """Test the company intelligence module"""
    
    # Test with a few symbols
    test_symbols = ['AAPL', 'NVDA', 'IXHL', 'TSLA']
    
    intel = CompanyIntelligence(test_symbols)
    await intel.initialize()
    
    # Fetch events
    events = await intel.fetch_all_events()
    
    print(f"\n📊 Found {len(events)} total events")
    
    for event in events[:10]:  # Show top 10
        print(f"\n{event.symbol} - {event.event_type.value}")
        print(f"  📰 {event.headline}")
        print(f"  📊 Impact Score: {event.impact_score}")
        print(f"  🔗 {event.url}")
    
    await intel.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_company_intelligence())
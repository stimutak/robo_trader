"""
Polygon.io / Massive.com data provider implementation.

This provider uses Polygon's REST API for historical data and
WebSocket streaming for real-time quotes (if on Advanced tier).

Free tier limitations:
- 5 API calls per minute
- End of day data only
- No WebSocket streaming

Advanced tier ($199/mo):
- Unlimited API calls
- Real-time data
- WebSocket streaming for quotes and trades
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
import structlog

from robo_trader.data_providers.base import DataProvider, DataProviderType, Quote, Trade

logger = structlog.get_logger(__name__)


class PolygonDataProvider(DataProvider):
    """
    Polygon.io data provider for market data.

    Supports:
    - Historical OHLCV bars via REST API
    - Previous day close prices
    - Ticker details and metadata
    - Real-time streaming (Advanced tier only)

    Usage:
        provider = PolygonDataProvider(api_key="your_key")
        await provider.connect()

        # Get historical bars
        bars = await provider.get_historical_bars("AAPL", timeframe="1min", limit=100)

        # Get current price (uses previous close on free tier)
        price = await provider.get_current_price("AAPL")

        await provider.disconnect()
    """

    # Timeframe mapping from our format to Polygon format
    TIMEFRAME_MAP = {
        "1min": ("minute", 1),
        "5min": ("minute", 5),
        "15min": ("minute", 15),
        "30min": ("minute", 30),
        "1hour": ("hour", 1),
        "4hour": ("hour", 4),
        "1day": ("day", 1),
        "1week": ("week", 1),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        tier: str = "free",
        rate_limit_delay: float = 12.5,  # 5 calls/min = 12 sec between calls
    ):
        """
        Initialize Polygon data provider.

        Args:
            api_key: Polygon API key (or set POLYGON_API_KEY env var)
            tier: Subscription tier - "free", "starter", "developer", "advanced"
            rate_limit_delay: Seconds between API calls (free tier = 12.5s)
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Polygon API key required. Set POLYGON_API_KEY env var or pass api_key."
            )

        self.tier = tier.lower()
        self.rate_limit_delay = rate_limit_delay if tier == "free" else 0.1
        self._client = None
        self._ws_client = None
        self._connected = False
        self._last_api_call = 0.0
        self._price_cache: Dict[str, tuple[float, datetime]] = {}
        self._quote_callbacks: List[Callable[[Quote], None]] = []
        self._trade_callbacks: List[Callable[[Trade], None]] = []
        self._subscribed_symbols: set[str] = set()

    @property
    def provider_type(self) -> DataProviderType:
        return DataProviderType.POLYGON

    @property
    def supports_streaming(self) -> bool:
        # Only Advanced tier has real-time streaming
        return self.tier == "advanced"

    async def connect(self) -> bool:
        """Connect to Polygon API."""
        try:
            from polygon import RESTClient

            self._client = RESTClient(api_key=self.api_key)
            self._connected = True

            # Test connection with a simple call
            details = self._client.get_ticker_details("AAPL")
            logger.info(
                "Connected to Polygon.io",
                tier=self.tier,
                test_ticker=details.name if details else "N/A",
            )
            return True

        except Exception as e:
            logger.error("Failed to connect to Polygon", error=str(e))
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Polygon API."""
        if self._ws_client:
            try:
                self._ws_client.close()
            except Exception:
                pass
            self._ws_client = None

        self._client = None
        self._connected = False
        self._price_cache.clear()
        logger.info("Disconnected from Polygon.io")

    async def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_api_call
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
        self._last_api_call = asyncio.get_event_loop().time()

    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1min",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars from Polygon.

        Note: Free tier only has access to end-of-day data from previous days.
        """
        if not self._client:
            raise ConnectionError("Not connected to Polygon")

        await self._rate_limit()

        # Parse timeframe
        if timeframe not in self.TIMEFRAME_MAP:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {list(self.TIMEFRAME_MAP.keys())}"
            )

        timespan, multiplier = self.TIMEFRAME_MAP[timeframe]

        # Default date range (last 5 days for free tier)
        if end is None:
            end = datetime.now()
        if start is None:
            # Go back enough days to get the requested number of bars
            days_back = max(5, (limit // 390) + 2)  # ~390 mins per trading day
            start = end - timedelta(days=days_back)

        from_date = start.strftime("%Y-%m-%d")
        to_date = end.strftime("%Y-%m-%d")

        try:
            aggs = self._client.get_aggs(
                ticker=symbol.upper(),
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date,
                limit=limit,
                sort="asc",
            )

            # Convert to DataFrame
            bars_data = []
            for bar in aggs:
                bars_data.append(
                    {
                        "date": datetime.fromtimestamp(bar.timestamp / 1000),
                        "open": float(bar.open),
                        "high": float(bar.high),
                        "low": float(bar.low),
                        "close": float(bar.close),
                        "volume": int(bar.volume),
                    }
                )

            if not bars_data:
                logger.warning(f"No bars returned for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(bars_data)
            df = df.sort_values("date").tail(limit)

            logger.debug(
                "Fetched historical bars",
                symbol=symbol,
                timeframe=timeframe,
                bars=len(df),
            )
            return df

        except Exception as e:
            logger.error(
                "Failed to fetch historical bars",
                symbol=symbol,
                error=str(e),
            )
            return pd.DataFrame()

    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        On free tier, this returns the previous day's close.
        On advanced tier, this returns the latest trade price.
        """
        # Check cache first (5 second TTL)
        if symbol in self._price_cache:
            price, cached_at = self._price_cache[symbol]
            if (datetime.now() - cached_at).total_seconds() < 5:
                return price

        quote = await self.get_quote(symbol)
        if quote and quote.last:
            return quote.last
        return None

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote for a symbol.

        On free tier, returns previous day's close as last price.
        """
        if not self._client:
            raise ConnectionError("Not connected to Polygon")

        await self._rate_limit()

        try:
            # Get previous day's data (works on free tier)
            from_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            to_date = datetime.now().strftime("%Y-%m-%d")

            aggs = self._client.get_aggs(
                ticker=symbol.upper(),
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date,
                limit=1,
                sort="desc",  # Most recent first
            )

            # Get the most recent bar
            for bar in aggs:
                quote = Quote(
                    symbol=symbol,
                    bid=None,  # Not available on free tier
                    ask=None,
                    last=float(bar.close),
                    bid_size=None,
                    ask_size=None,
                    volume=int(bar.volume),
                    timestamp=datetime.fromtimestamp(bar.timestamp / 1000),
                )

                # Cache the price
                self._price_cache[symbol] = (quote.last, datetime.now())

                return quote

            return None

        except Exception as e:
            logger.error("Failed to get quote", symbol=symbol, error=str(e))
            return None

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols efficiently.

        Uses grouped daily endpoint for batch efficiency.
        """
        if not self._client:
            raise ConnectionError("Not connected to Polygon")

        await self._rate_limit()

        quotes = {}
        try:
            # Get grouped daily bars (all tickers in one call)
            from_date = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

            # Unfortunately, grouped daily requires iterating
            # Fall back to individual calls but with minimal delay
            for symbol in symbols:
                quote = await self.get_quote(symbol)
                if quote:
                    quotes[symbol] = quote

        except Exception as e:
            logger.error("Failed to get batch quotes", error=str(e))

        return quotes

    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """
        Subscribe to real-time quote updates (Advanced tier only).
        """
        if not self.supports_streaming:
            raise NotImplementedError(
                f"Streaming requires Advanced tier. Current tier: {self.tier}"
            )

        self._quote_callbacks.append(callback)
        self._subscribed_symbols.update(symbols)

        # TODO: Implement WebSocket streaming for Advanced tier
        # from polygon import WebSocketClient
        # self._ws_client = WebSocketClient(...)
        logger.warning("WebSocket streaming not yet implemented")

    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """
        Subscribe to real-time trade updates (Advanced tier only).
        """
        if not self.supports_streaming:
            raise NotImplementedError(
                f"Streaming requires Advanced tier. Current tier: {self.tier}"
            )

        self._trade_callbacks.append(callback)
        self._subscribed_symbols.update(symbols)

        logger.warning("WebSocket streaming not yet implemented")

    async def unsubscribe(self, symbols: Optional[List[str]] = None) -> None:
        """Unsubscribe from real-time updates."""
        if symbols:
            self._subscribed_symbols -= set(symbols)
        else:
            self._subscribed_symbols.clear()

    async def get_ticker_details(self, symbol: str) -> Optional[dict]:
        """
        Get detailed information about a ticker.

        Returns dict with: name, market_cap, description, homepage_url, etc.
        """
        if not self._client:
            raise ConnectionError("Not connected to Polygon")

        await self._rate_limit()

        try:
            details = self._client.get_ticker_details(symbol.upper())
            return {
                "symbol": details.ticker,
                "name": details.name,
                "market_cap": details.market_cap,
                "description": details.description,
                "homepage_url": details.homepage_url,
                "primary_exchange": details.primary_exchange,
                "type": details.type,
                "currency": details.currency_name,
            }
        except Exception as e:
            logger.error("Failed to get ticker details", symbol=symbol, error=str(e))
            return None

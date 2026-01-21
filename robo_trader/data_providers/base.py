"""
Abstract base class for market data providers.

This defines the interface that all data providers must implement,
allowing the trading system to work with different data sources
(Polygon.io, IBKR, Alpaca, etc.) interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

import pandas as pd


class DataProviderType(Enum):
    """Supported data provider types."""

    POLYGON = "polygon"
    IBKR = "ibkr"
    ALPACA = "alpaca"


@dataclass
class Quote:
    """Real-time quote data."""

    symbol: str
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    bid_size: Optional[int]
    ask_size: Optional[int]
    volume: Optional[int]
    timestamp: datetime

    @property
    def mid(self) -> Optional[float]:
        """Calculate mid price from bid/ask."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate spread as percentage of mid price."""
        if self.spread is not None and self.mid is not None and self.mid > 0:
            return self.spread / self.mid
        return None


@dataclass
class Trade:
    """Individual trade tick data."""

    symbol: str
    price: float
    size: int
    timestamp: datetime
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    Implementations must provide methods for:
    - Fetching historical OHLCV bars
    - Getting current/latest prices
    - Subscribing to real-time quote streams (if supported)
    """

    @property
    @abstractmethod
    def provider_type(self) -> DataProviderType:
        """Return the provider type identifier."""
        pass

    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return True if provider supports real-time streaming."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to the data provider.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean disconnection from the data provider."""
        pass

    @abstractmethod
    async def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1min",
        limit: int = 100,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV bars for a symbol.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            timeframe: Bar timeframe - "1min", "5min", "15min", "1hour", "1day"
            limit: Maximum number of bars to return
            start: Start datetime for historical range
            end: End datetime for historical range

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
            Sorted by date ascending (oldest first).
        """
        pass

    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current/latest price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Latest price or None if unavailable.
        """
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get current quote (bid/ask/last) for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Quote object or None if unavailable.
        """
        pass

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Default implementation calls get_quote() for each symbol.
        Providers can override for batch efficiency.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dict mapping symbol to Quote (missing symbols not included).
        """
        quotes = {}
        for symbol in symbols:
            quote = await self.get_quote(symbol)
            if quote is not None:
                quotes[symbol] = quote
        return quotes

    async def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[Quote], None],
    ) -> None:
        """
        Subscribe to real-time quote updates.

        Args:
            symbols: List of symbols to subscribe to
            callback: Function called with Quote on each update

        Raises:
            NotImplementedError: If provider doesn't support streaming.
        """
        if not self.supports_streaming:
            raise NotImplementedError(f"{self.provider_type.value} does not support streaming")

    async def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[Trade], None],
    ) -> None:
        """
        Subscribe to real-time trade updates.

        Args:
            symbols: List of symbols to subscribe to
            callback: Function called with Trade on each update

        Raises:
            NotImplementedError: If provider doesn't support streaming.
        """
        if not self.supports_streaming:
            raise NotImplementedError(f"{self.provider_type.value} does not support streaming")

    async def unsubscribe(self, symbols: Optional[List[str]] = None) -> None:
        """
        Unsubscribe from real-time updates.

        Args:
            symbols: Specific symbols to unsubscribe, or None for all.
        """
        pass

    def __repr__(self) -> str:
        streaming = "streaming" if self.supports_streaming else "polling"
        return f"<{self.__class__.__name__} ({self.provider_type.value}, {streaming})>"

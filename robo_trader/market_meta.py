"""
Market metadata provider for liquidity and spread checks.
Ensures only tradeable symbols with sufficient liquidity.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
import asyncio

from .logger import get_logger
from .ibkr_client import IBKRClient

logger = get_logger(__name__)


@dataclass
class MarketMetadata:
    """Market metadata for a symbol."""
    symbol: str
    adv: float  # Average Daily Volume in dollars
    spread_pct: float  # Bid-ask spread as % of mid
    spread_dollars: float  # Absolute spread in dollars
    bid: float
    ask: float
    last_price: float
    volume: int
    avg_volume_20d: int
    atr: float  # Average True Range
    shortable: bool
    borrow_rate: float  # Annual borrow rate for shorts
    last_updated: datetime
    
    def is_liquid(self, min_adv: float = 3_000_000, max_spread_pct: float = 0.01) -> bool:
        """Check if symbol meets liquidity requirements."""
        return self.adv >= min_adv and self.spread_pct <= max_spread_pct
    
    def get_liquidity_score(self) -> float:
        """Get liquidity score 0-100."""
        adv_score = min(100, (self.adv / 10_000_000) * 50)  # 50 points for $10M+ ADV
        spread_score = max(0, 50 - (self.spread_pct * 5000))  # 50 points for tight spread
        return adv_score + spread_score


class MarketMetaProvider:
    """
    Provider for market metadata with caching.
    Checks liquidity, spreads, and shortability.
    """
    
    def __init__(
        self,
        ib_client: IBKRClient,
        min_adv: float = 3_000_000,  # $3M minimum ADV
        max_spread_pct: float = 0.01,  # 1% maximum spread
        max_option_spread_pct: float = 0.08,  # 8% for options
        cache_duration_minutes: int = 5
    ):
        """
        Initialize market metadata provider.
        
        Args:
            ib_client: IB client for market data
            min_adv: Minimum average daily volume in dollars
            max_spread_pct: Maximum bid-ask spread as % of mid
            max_option_spread_pct: Maximum spread for options
            cache_duration_minutes: Cache duration in minutes
        """
        self.ib_client = ib_client
        self.min_adv = min_adv
        self.max_spread_pct = max_spread_pct
        self.max_option_spread_pct = max_option_spread_pct
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        
        # Cache for metadata
        self._cache: Dict[str, MarketMetadata] = {}
        
        logger.info(
            f"MarketMetaProvider initialized: "
            f"min_adv=${min_adv/1e6:.1f}M, max_spread={max_spread_pct:.1%}"
        )
    
    async def get_metadata(self, symbol: str, force_refresh: bool = False) -> Optional[MarketMetadata]:
        """
        Get market metadata for a symbol.
        
        Args:
            symbol: Stock symbol
            force_refresh: Force refresh from market
            
        Returns:
            MarketMetadata or None if unavailable
        """
        # Check cache first
        if not force_refresh and symbol in self._cache:
            cached = self._cache[symbol]
            if datetime.now() - cached.last_updated < self.cache_duration:
                return cached
        
        try:
            # Fetch market data
            market_data = await self._fetch_market_data(symbol)
            if not market_data:
                logger.warning(f"No market data available for {symbol}")
                return None
            
            # Create metadata
            metadata = MarketMetadata(
                symbol=symbol,
                adv=market_data['adv'],
                spread_pct=market_data['spread_pct'],
                spread_dollars=market_data['spread_dollars'],
                bid=market_data['bid'],
                ask=market_data['ask'],
                last_price=market_data['last'],
                volume=market_data['volume'],
                avg_volume_20d=market_data['avg_volume'],
                atr=market_data['atr'],
                shortable=market_data['shortable'],
                borrow_rate=market_data['borrow_rate'],
                last_updated=datetime.now()
            )
            
            # Cache it
            self._cache[symbol] = metadata
            
            # Log if fails liquidity check
            if not metadata.is_liquid(self.min_adv, self.max_spread_pct):
                logger.info(
                    f"{symbol} fails liquidity: ADV=${metadata.adv/1e6:.1f}M, "
                    f"Spread={metadata.spread_pct:.2%}"
                )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching metadata for {symbol}: {e}")
            return None
    
    async def _fetch_market_data(self, symbol: str) -> Optional[Dict]:
        """Fetch market data from IB."""
        try:
            # Get recent bars for volume and ATR
            bars = await self.ib_client.fetch_recent_bars(
                symbol=symbol,
                duration="20 D",
                bar_size="1 day"
            )
            
            if bars.empty:
                return None
            
            # Calculate metrics
            avg_volume = bars['volume'].mean()
            avg_price = bars['close'].mean()
            adv = avg_volume * avg_price
            
            # Calculate ATR
            high_low = bars['high'] - bars['low']
            high_close = abs(bars['high'] - bars['close'].shift())
            low_close = abs(bars['low'] - bars['close'].shift())
            tr = high_low.combine(high_close, max).combine(low_close, max)
            atr = tr.rolling(14).mean().iloc[-1]
            
            # Get current quote for spread
            # This would need IB quote data - simplified for now
            last_price = bars.iloc[-1]['close']
            bid = last_price * 0.9995  # Approximate 5bps spread
            ask = last_price * 1.0005
            
            spread_dollars = ask - bid
            spread_pct = spread_dollars / ((bid + ask) / 2)
            
            return {
                'symbol': symbol,
                'adv': adv,
                'spread_pct': spread_pct,
                'spread_dollars': spread_dollars,
                'bid': bid,
                'ask': ask,
                'last': last_price,
                'volume': bars.iloc[-1]['volume'],
                'avg_volume': int(avg_volume),
                'atr': atr,
                'shortable': True,  # Would check with IB
                'borrow_rate': 0.0  # Would get from IB
            }
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    async def check_liquidity_batch(
        self,
        symbols: List[str]
    ) -> Dict[str, bool]:
        """
        Check liquidity for multiple symbols.
        
        Args:
            symbols: List of symbols to check
            
        Returns:
            Dict mapping symbol to liquidity status
        """
        results = {}
        
        # Fetch metadata for all symbols
        tasks = [self.get_metadata(symbol) for symbol in symbols]
        metadata_list = await asyncio.gather(*tasks)
        
        for symbol, metadata in zip(symbols, metadata_list):
            if metadata:
                results[symbol] = metadata.is_liquid(self.min_adv, self.max_spread_pct)
            else:
                results[symbol] = False
        
        # Log summary
        liquid_count = sum(1 for is_liquid in results.values() if is_liquid)
        logger.info(
            f"Liquidity check: {liquid_count}/{len(symbols)} symbols pass "
            f"(ADV>${self.min_adv/1e6:.1f}M, Spread<{self.max_spread_pct:.1%})"
        )
        
        return results
    
    def validate_order_liquidity(
        self,
        symbol: str,
        order_size: int,
        price: float
    ) -> Tuple[bool, str]:
        """
        Validate that an order meets liquidity requirements.
        
        Args:
            symbol: Stock symbol
            order_size: Number of shares
            price: Order price
            
        Returns:
            Tuple of (is_valid, message)
        """
        if symbol not in self._cache:
            return False, "No metadata available - fetch first"
        
        metadata = self._cache[symbol]
        
        # Check basic liquidity
        if not metadata.is_liquid(self.min_adv, self.max_spread_pct):
            return False, f"Fails liquidity: ADV=${metadata.adv/1e6:.1f}M, Spread={metadata.spread_pct:.2%}"
        
        # Check order size vs ADV
        order_value = order_size * price
        if order_value > metadata.adv * 0.1:  # Order > 10% of ADV
            return False, f"Order too large: ${order_value/1e6:.1f}M vs ADV ${metadata.adv/1e6:.1f}M"
        
        # Check if shortable for sell orders
        if order_size < 0 and not metadata.shortable:
            return False, "Symbol not shortable"
        
        return True, "Liquidity OK"
    
    def get_execution_cost_estimate(
        self,
        symbol: str,
        order_size: int,
        is_aggressive: bool = False
    ) -> float:
        """
        Estimate execution costs in basis points.
        
        Args:
            symbol: Stock symbol
            order_size: Number of shares
            is_aggressive: True for market orders
            
        Returns:
            Estimated cost in basis points
        """
        if symbol not in self._cache:
            return 10.0  # Default 10bps if no data
        
        metadata = self._cache[symbol]
        
        # Base cost is half the spread
        base_cost_bps = (metadata.spread_pct / 2) * 10000
        
        # Add impact based on order size vs ADV
        order_value = abs(order_size) * metadata.last_price
        adv_ratio = order_value / max(metadata.adv, 1)
        
        # Linear impact: 1bp per 1% of ADV
        impact_bps = adv_ratio * 100
        
        # Market orders cost more
        if is_aggressive:
            base_cost_bps *= 1.5
        
        # Add commission estimate
        commission_bps = 0.5
        
        total_bps = base_cost_bps + impact_bps + commission_bps
        
        return min(total_bps, 50.0)  # Cap at 50bps
    
    def get_all_metadata(self) -> Dict[str, MarketMetadata]:
        """Get all cached metadata."""
        return self._cache.copy()
    
    def clear_cache(self):
        """Clear metadata cache."""
        self._cache.clear()
        logger.info("Metadata cache cleared")
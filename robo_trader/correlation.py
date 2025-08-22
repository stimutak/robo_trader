"""
Correlation budget and sector exposure control.
Prevents concentration risk in correlated positions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)


class Sector(str, Enum):
    """Market sectors for correlation grouping."""
    TECHNOLOGY = "technology"
    FINANCIALS = "financials"
    HEALTHCARE = "healthcare"
    CONSUMER = "consumer"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"
    CRYPTO = "crypto"
    PRECIOUS_METALS = "precious_metals"
    INDEXES = "indexes"
    VOLATILITY = "volatility"
    BONDS = "bonds"


@dataclass
class BucketExposure:
    """Exposure tracking for a correlation bucket."""
    bucket_name: str
    symbols: List[str]
    total_notional: float
    total_shares: Dict[str, int]
    percent_of_portfolio: float
    
    def add_position(self, symbol: str, shares: int, price: float):
        """Add a position to this bucket."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
        
        if symbol not in self.total_shares:
            self.total_shares[symbol] = 0
        
        self.total_shares[symbol] += shares
        self.total_notional += shares * price
    
    def remove_position(self, symbol: str, shares: int, price: float):
        """Remove a position from this bucket."""
        if symbol in self.total_shares:
            self.total_shares[symbol] -= shares
            self.total_notional -= shares * price
            
            if self.total_shares[symbol] <= 0:
                del self.total_shares[symbol]
                if symbol in self.symbols:
                    self.symbols.remove(symbol)


class CorrelationBudget:
    """
    Manages correlation buckets to prevent concentration risk.
    Enforces maximum exposure per bucket (sector/theme/factor).
    """
    
    # Default sector mappings - can be overridden
    DEFAULT_SECTOR_MAP = {
        # Tech giants
        "AAPL": Sector.TECHNOLOGY,
        "MSFT": Sector.TECHNOLOGY,
        "GOOGL": Sector.TECHNOLOGY,
        "GOOG": Sector.TECHNOLOGY,
        "META": Sector.TECHNOLOGY,
        "AMZN": Sector.TECHNOLOGY,
        "NVDA": Sector.TECHNOLOGY,
        "TSM": Sector.TECHNOLOGY,
        "AVGO": Sector.TECHNOLOGY,
        "ORCL": Sector.TECHNOLOGY,
        "CSCO": Sector.TECHNOLOGY,
        "CRM": Sector.TECHNOLOGY,
        "ADBE": Sector.TECHNOLOGY,
        "INTC": Sector.TECHNOLOGY,
        "AMD": Sector.TECHNOLOGY,
        "QCOM": Sector.TECHNOLOGY,
        "TXN": Sector.TECHNOLOGY,
        "MU": Sector.TECHNOLOGY,
        
        # AI/ML focused
        "PLTR": Sector.TECHNOLOGY,
        "AI": Sector.TECHNOLOGY,
        "UPST": Sector.TECHNOLOGY,
        "NUAI": Sector.TECHNOLOGY,
        "BZAI": Sector.TECHNOLOGY,
        "IXHL": Sector.TECHNOLOGY,
        "APLD": Sector.TECHNOLOGY,
        
        # EVs and related
        "TSLA": Sector.TECHNOLOGY,  # Could also be CONSUMER
        "RIVN": Sector.CONSUMER,
        "LCID": Sector.CONSUMER,
        "NIO": Sector.CONSUMER,
        
        # Financials
        "JPM": Sector.FINANCIALS,
        "BAC": Sector.FINANCIALS,
        "WFC": Sector.FINANCIALS,
        "GS": Sector.FINANCIALS,
        "MS": Sector.FINANCIALS,
        "C": Sector.FINANCIALS,
        "SCHW": Sector.FINANCIALS,
        "BLK": Sector.FINANCIALS,
        "SOFI": Sector.FINANCIALS,
        "UPST": Sector.FINANCIALS,
        "OPEN": Sector.FINANCIALS,
        
        # Healthcare
        "JNJ": Sector.HEALTHCARE,
        "UNH": Sector.HEALTHCARE,
        "PFE": Sector.HEALTHCARE,
        "LLY": Sector.HEALTHCARE,
        "ABBV": Sector.HEALTHCARE,
        "MRK": Sector.HEALTHCARE,
        "TMO": Sector.HEALTHCARE,
        "ABT": Sector.HEALTHCARE,
        "CVS": Sector.HEALTHCARE,
        "ELTP": Sector.HEALTHCARE,
        "HTFL": Sector.HEALTHCARE,
        
        # Energy
        "XOM": Sector.ENERGY,
        "CVX": Sector.ENERGY,
        "COP": Sector.ENERGY,
        "SLB": Sector.ENERGY,
        "EOG": Sector.ENERGY,
        "CEG": Sector.ENERGY,  # Clean energy
        "TEM": Sector.ENERGY,
        
        # Consumer
        "WMT": Sector.CONSUMER,
        "HD": Sector.CONSUMER,
        "PG": Sector.CONSUMER,
        "KO": Sector.CONSUMER,
        "PEP": Sector.CONSUMER,
        "COST": Sector.CONSUMER,
        "MCD": Sector.CONSUMER,
        "NKE": Sector.CONSUMER,
        "SBUX": Sector.CONSUMER,
        "VRT": Sector.CONSUMER,
        
        # Industrials
        "BA": Sector.INDUSTRIALS,
        "CAT": Sector.INDUSTRIALS,
        "HON": Sector.INDUSTRIALS,
        "UPS": Sector.INDUSTRIALS,
        "RTX": Sector.INDUSTRIALS,
        "LMT": Sector.INDUSTRIALS,
        "GE": Sector.INDUSTRIALS,
        "MMM": Sector.INDUSTRIALS,
        "SDGR": Sector.INDUSTRIALS,
        
        # Crypto miners
        "MARA": Sector.CRYPTO,
        "RIOT": Sector.CRYPTO,
        "COIN": Sector.CRYPTO,
        "MSTR": Sector.CRYPTO,
        "CORZ": Sector.CRYPTO,
        "WULF": Sector.CRYPTO,
        
        # ETFs and Indexes
        "SPY": Sector.INDEXES,
        "QQQ": Sector.INDEXES,
        "IWM": Sector.INDEXES,
        "DIA": Sector.INDEXES,
        "VOO": Sector.INDEXES,
        "VTI": Sector.INDEXES,
        
        # Precious Metals
        "GLD": Sector.PRECIOUS_METALS,
        "SLV": Sector.PRECIOUS_METALS,
        "GDX": Sector.PRECIOUS_METALS,
        "GDXJ": Sector.PRECIOUS_METALS,
        
        # Bonds
        "TLT": Sector.BONDS,
        "IEF": Sector.BONDS,
        "SHY": Sector.BONDS,
        "AGG": Sector.BONDS,
        
        # Volatility
        "VXX": Sector.VOLATILITY,
        "UVXY": Sector.VOLATILITY,
        "SVXY": Sector.VOLATILITY,
    }
    
    def __init__(
        self,
        max_bucket_exposure_pct: float = 0.35,  # 35% max per bucket
        sector_map: Optional[Dict[str, Sector]] = None,
        custom_buckets: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize correlation budget manager.
        
        Args:
            max_bucket_exposure_pct: Maximum exposure per bucket as decimal
            sector_map: Optional custom sector mappings
            custom_buckets: Optional custom correlation buckets
        """
        self.max_bucket_exposure_pct = max_bucket_exposure_pct
        self.sector_map = sector_map or self.DEFAULT_SECTOR_MAP.copy()
        
        # Track exposures by bucket
        self.buckets: Dict[str, BucketExposure] = {}
        
        # Add custom buckets if provided
        self.custom_buckets = custom_buckets or {}
        
        # Current portfolio value (updated externally)
        self.portfolio_value = 100000.0
        
        logger.info(
            f"CorrelationBudget initialized: max_bucket={max_bucket_exposure_pct:.0%}, "
            f"sectors={len(set(self.sector_map.values()))}"
        )
    
    def get_symbol_bucket(self, symbol: str) -> str:
        """Get the correlation bucket for a symbol."""
        # Check custom buckets first
        for bucket_name, symbols in self.custom_buckets.items():
            if symbol in symbols:
                return bucket_name
        
        # Check sector map
        if symbol in self.sector_map:
            return self.sector_map[symbol].value
        
        # Default to symbol itself if unknown
        return f"single_{symbol}"
    
    def update_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        portfolio_value: float
    ):
        """
        Update position in correlation tracking.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares (negative for short)
            price: Current price
            portfolio_value: Total portfolio value
        """
        self.portfolio_value = portfolio_value
        bucket_name = self.get_symbol_bucket(symbol)
        
        # Get or create bucket
        if bucket_name not in self.buckets:
            self.buckets[bucket_name] = BucketExposure(
                bucket_name=bucket_name,
                symbols=[],
                total_notional=0,
                total_shares={},
                percent_of_portfolio=0
            )
        
        bucket = self.buckets[bucket_name]
        
        # Update bucket
        if shares != 0:
            bucket.add_position(symbol, shares, price)
        else:
            # Remove position
            if symbol in bucket.total_shares:
                bucket.remove_position(symbol, bucket.total_shares[symbol], price)
        
        # Update percentage
        bucket.percent_of_portfolio = bucket.total_notional / portfolio_value if portfolio_value > 0 else 0
        
        # Clean up empty buckets
        if not bucket.symbols:
            del self.buckets[bucket_name]
    
    def check_new_position(
        self,
        symbol: str,
        shares: int,
        price: float,
        portfolio_value: Optional[float] = None
    ) -> Tuple[bool, str, float]:
        """
        Check if a new position would violate correlation limits.
        
        Args:
            symbol: Stock symbol
            shares: Number of shares to add
            price: Entry price
            portfolio_value: Optional portfolio value update
            
        Returns:
            Tuple of (is_allowed, reason, current_bucket_exposure_pct)
        """
        if portfolio_value:
            self.portfolio_value = portfolio_value
        
        bucket_name = self.get_symbol_bucket(symbol)
        position_value = abs(shares * price)
        
        # Get current bucket exposure
        current_exposure = 0
        if bucket_name in self.buckets:
            current_exposure = self.buckets[bucket_name].total_notional
        
        # Calculate new exposure
        new_exposure = current_exposure + position_value
        new_exposure_pct = new_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
        
        # Check limit
        if new_exposure_pct > self.max_bucket_exposure_pct:
            return (
                False,
                f"Would exceed {self.max_bucket_exposure_pct:.0%} limit for {bucket_name} "
                f"(current: {current_exposure/self.portfolio_value:.1%}, "
                f"new: {new_exposure_pct:.1%})",
                current_exposure / self.portfolio_value
            )
        
        return True, "Within correlation limits", current_exposure / self.portfolio_value
    
    def get_bucket_summary(self) -> Dict[str, Dict]:
        """Get summary of all bucket exposures."""
        summary = {}
        
        for bucket_name, bucket in self.buckets.items():
            summary[bucket_name] = {
                "symbols": bucket.symbols,
                "notional": bucket.total_notional,
                "percent": bucket.percent_of_portfolio,
                "shares": bucket.total_shares,
                "at_limit": bucket.percent_of_portfolio >= self.max_bucket_exposure_pct * 0.9
            }
        
        return summary
    
    def get_available_buckets(self) -> List[str]:
        """Get buckets with room for more exposure."""
        available = []
        
        all_buckets = set(self.sector_map.values())
        all_buckets.update(self.custom_buckets.keys())
        
        for bucket_name in all_buckets:
            if isinstance(bucket_name, Sector):
                bucket_name = bucket_name.value
            
            current_pct = 0
            if bucket_name in self.buckets:
                current_pct = self.buckets[bucket_name].percent_of_portfolio
            
            if current_pct < self.max_bucket_exposure_pct * 0.8:  # 80% of limit
                available.append(bucket_name)
        
        return available
    
    def rebalance_suggestions(self) -> List[Dict]:
        """Get suggestions for rebalancing overweight buckets."""
        suggestions = []
        
        for bucket_name, bucket in self.buckets.items():
            if bucket.percent_of_portfolio > self.max_bucket_exposure_pct:
                excess_pct = bucket.percent_of_portfolio - self.max_bucket_exposure_pct
                excess_value = excess_pct * self.portfolio_value
                
                suggestions.append({
                    "bucket": bucket_name,
                    "current_pct": bucket.percent_of_portfolio,
                    "target_pct": self.max_bucket_exposure_pct,
                    "reduce_by": excess_value,
                    "symbols": bucket.symbols
                })
        
        return suggestions
    
    def clear_all(self):
        """Clear all bucket tracking."""
        self.buckets.clear()
        logger.info("Correlation buckets cleared")
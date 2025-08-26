"""
Correlation calculation and tracking for portfolio risk management.

This module provides tools to calculate and monitor correlations between
positions to prevent concentrated risk exposure.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from robo_trader.logger import get_logger

logger = get_logger(__name__)


class CorrelationTracker:
    """
    Track and calculate correlations between portfolio holdings.
    
    Features:
    - Rolling correlation calculation
    - Correlation matrix caching
    - Sector correlation analysis
    - Real-time updates with new price data
    """
    
    def __init__(
        self,
        lookback_days: int = 60,
        min_observations: int = 30,
        correlation_threshold: float = 0.7,
        update_frequency_minutes: int = 15,
    ):
        """
        Initialize correlation tracker.
        
        Args:
            lookback_days: Days of history for correlation calculation
            min_observations: Minimum data points required
            correlation_threshold: Threshold for high correlation warning
            update_frequency_minutes: How often to recalculate
        """
        self.lookback_days = lookback_days
        self.min_observations = min_observations
        self.correlation_threshold = correlation_threshold
        self.update_frequency_minutes = update_frequency_minutes
        
        # Price history storage
        self.price_history: Dict[str, pd.Series] = {}
        self.returns_history: Dict[str, pd.Series] = {}
        
        # Correlation matrix cache
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.last_update: Optional[datetime] = None
        
        # Sector correlations
        self.sector_correlations: Dict[str, Dict[str, float]] = {}
        self.symbol_to_sector: Dict[str, str] = {}
    
    def add_price_data(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        sector: Optional[str] = None,
    ) -> None:
        """
        Add price data point for correlation calculation.
        
        Args:
            symbol: Trading symbol
            timestamp: Price timestamp
            price: Stock price
            sector: Sector classification
        """
        if symbol not in self.price_history:
            self.price_history[symbol] = pd.Series(dtype=float)
        
        # Add to price history
        self.price_history[symbol][timestamp] = price
        
        # Update sector mapping
        if sector:
            self.symbol_to_sector[symbol] = sector
        
        # Calculate returns if we have enough data
        if len(self.price_history[symbol]) >= 2:
            prices = self.price_history[symbol].sort_index()
            returns = prices.pct_change().dropna()
            self.returns_history[symbol] = returns
    
    def add_price_series(
        self,
        symbol: str,
        prices: pd.Series,
        sector: Optional[str] = None,
    ) -> None:
        """
        Add historical price series for a symbol.
        
        Args:
            symbol: Trading symbol
            prices: Series of prices with datetime index
            sector: Sector classification
        """
        # Store price history
        self.price_history[symbol] = prices.sort_index()
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        self.returns_history[symbol] = returns
        
        # Update sector mapping
        if sector:
            self.symbol_to_sector[symbol] = sector
        
        logger.debug(f"Added {len(prices)} price points for {symbol}")
    
    def calculate_correlation_matrix(
        self,
        symbols: Optional[List[str]] = None,
        force_update: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for specified symbols.
        
        Args:
            symbols: List of symbols (uses all if None)
            force_update: Force recalculation even if cache is fresh
            
        Returns:
            Correlation matrix DataFrame
        """
        # Check if update needed
        if not force_update and self.last_update:
            time_since_update = datetime.now() - self.last_update
            if time_since_update.total_seconds() < self.update_frequency_minutes * 60:
                if not symbols:
                    return self.correlation_matrix
                # Return subset if specific symbols requested
                available = [s for s in symbols if s in self.correlation_matrix.index]
                if available:
                    return self.correlation_matrix.loc[available, available]
        
        # Determine symbols to use
        if symbols is None:
            symbols = list(self.returns_history.keys())
        
        # Filter symbols with enough data
        valid_symbols = []
        for symbol in symbols:
            if symbol in self.returns_history:
                if len(self.returns_history[symbol]) >= self.min_observations:
                    valid_symbols.append(symbol)
                else:
                    logger.debug(f"Skipping {symbol}: insufficient data ({len(self.returns_history[symbol])} points)")
        
        if len(valid_symbols) < 2:
            logger.warning("Not enough symbols with sufficient data for correlation")
            return pd.DataFrame()
        
        # Prepare returns DataFrame
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        returns_data = {}
        
        for symbol in valid_symbols:
            returns = self.returns_history[symbol]
            # Filter to lookback period
            recent_returns = returns[returns.index >= cutoff_date]
            if len(recent_returns) >= self.min_observations:
                returns_data[symbol] = recent_returns
        
        if not returns_data:
            logger.warning("No symbols have enough recent data")
            return pd.DataFrame()
        
        # Align returns data
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr(method='pearson')
        
        # Handle NaN values
        correlation_matrix = correlation_matrix.fillna(0)
        
        # Update cache
        self.correlation_matrix = correlation_matrix
        self.last_update = datetime.now()
        
        logger.info(f"Updated correlation matrix for {len(valid_symbols)} symbols")
        
        return correlation_matrix
    
    def get_pairwise_correlation(
        self,
        symbol1: str,
        symbol2: str,
        force_update: bool = False,
    ) -> Optional[float]:
        """
        Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            force_update: Force recalculation
            
        Returns:
            Correlation coefficient or None if unavailable
        """
        if symbol1 == symbol2:
            return 1.0
        
        # Update correlation matrix if needed
        corr_matrix = self.calculate_correlation_matrix(
            symbols=[symbol1, symbol2],
            force_update=force_update
        )
        
        if corr_matrix.empty:
            return None
        
        if symbol1 in corr_matrix.index and symbol2 in corr_matrix.columns:
            return float(corr_matrix.loc[symbol1, symbol2])
        
        return None
    
    def get_correlation_dict(
        self,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get correlation dictionary for risk manager.
        
        Args:
            symbols: List of symbols to include
            
        Returns:
            Nested dictionary of correlations
        """
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return {}
        
        result = {}
        for symbol1 in corr_matrix.index:
            result[symbol1] = {}
            for symbol2 in corr_matrix.columns:
                if symbol1 != symbol2:
                    result[symbol1][symbol2] = float(corr_matrix.loc[symbol1, symbol2])
        
        return result
    
    def find_high_correlations(
        self,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, str, float]]:
        """
        Find symbol pairs with high correlation.
        
        Args:
            threshold: Correlation threshold (uses default if None)
            
        Returns:
            List of (symbol1, symbol2, correlation) tuples
        """
        if threshold is None:
            threshold = self.correlation_threshold
        
        corr_matrix = self.calculate_correlation_matrix()
        
        if corr_matrix.empty:
            return []
        
        high_correlations = []
        
        # Only check upper triangle to avoid duplicates
        for i, symbol1 in enumerate(corr_matrix.index):
            for j, symbol2 in enumerate(corr_matrix.columns):
                if j <= i:  # Skip diagonal and lower triangle
                    continue
                
                correlation = corr_matrix.loc[symbol1, symbol2]
                if abs(correlation) >= threshold:
                    high_correlations.append((symbol1, symbol2, float(correlation)))
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_correlations
    
    def calculate_sector_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate average correlations between sectors.
        
        Returns:
            Dictionary of sector-to-sector correlations
        """
        # Group symbols by sector
        sector_symbols: Dict[str, List[str]] = {}
        for symbol, sector in self.symbol_to_sector.items():
            if symbol in self.returns_history:
                if sector not in sector_symbols:
                    sector_symbols[sector] = []
                sector_symbols[sector].append(symbol)
        
        if len(sector_symbols) < 2:
            return {}
        
        # Calculate correlation matrix if needed
        all_symbols = [s for symbols in sector_symbols.values() for s in symbols]
        corr_matrix = self.calculate_correlation_matrix(all_symbols)
        
        if corr_matrix.empty:
            return {}
        
        # Calculate average correlation between sectors
        sector_corr = {}
        
        for sector1, symbols1 in sector_symbols.items():
            sector_corr[sector1] = {}
            
            for sector2, symbols2 in sector_symbols.items():
                if sector1 == sector2:
                    # Intra-sector correlation
                    correlations = []
                    for s1 in symbols1:
                        for s2 in symbols2:
                            if s1 != s2 and s1 in corr_matrix.index and s2 in corr_matrix.columns:
                                correlations.append(corr_matrix.loc[s1, s2])
                    
                    if correlations:
                        sector_corr[sector1][sector2] = float(np.mean(correlations))
                else:
                    # Inter-sector correlation
                    correlations = []
                    for s1 in symbols1:
                        for s2 in symbols2:
                            if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                                correlations.append(corr_matrix.loc[s1, s2])
                    
                    if correlations:
                        sector_corr[sector1][sector2] = float(np.mean(correlations))
        
        self.sector_correlations = sector_corr
        return sector_corr
    
    def get_portfolio_correlation_risk(
        self,
        positions: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate portfolio correlation risk score.
        
        Args:
            positions: Dictionary of symbol to position weight
            
        Returns:
            Tuple of (risk_score, contribution_by_symbol)
        """
        if len(positions) < 2:
            return 0.0, {}
        
        # Get correlation matrix for position symbols
        symbols = list(positions.keys())
        corr_matrix = self.calculate_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            return 0.0, {}
        
        # Calculate weighted correlation risk
        total_risk = 0.0
        contributions = {}
        
        for symbol1 in symbols:
            symbol_risk = 0.0
            if symbol1 in corr_matrix.index:
                for symbol2 in symbols:
                    if symbol2 != symbol1 and symbol2 in corr_matrix.columns:
                        correlation = corr_matrix.loc[symbol1, symbol2]
                        weight1 = positions[symbol1]
                        weight2 = positions[symbol2]
                        
                        # Risk contribution from this pair
                        pair_risk = abs(correlation * weight1 * weight2)
                        symbol_risk += pair_risk
                        total_risk += pair_risk / 2  # Divide by 2 to avoid double counting
            
            contributions[symbol1] = symbol_risk
        
        return total_risk, contributions
    
    def suggest_diversification(
        self,
        current_positions: List[str],
        candidate_symbols: List[str],
        max_suggestions: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Suggest symbols for diversification.
        
        Args:
            current_positions: Current portfolio symbols
            candidate_symbols: Potential symbols to add
            max_suggestions: Maximum suggestions to return
            
        Returns:
            List of (symbol, avg_correlation) tuples
        """
        if not current_positions or not candidate_symbols:
            return []
        
        # Calculate correlations
        all_symbols = list(set(current_positions + candidate_symbols))
        corr_matrix = self.calculate_correlation_matrix(all_symbols)
        
        if corr_matrix.empty:
            return []
        
        suggestions = []
        
        for candidate in candidate_symbols:
            if candidate in current_positions:
                continue
            
            if candidate not in corr_matrix.index:
                continue
            
            # Calculate average correlation with current positions
            correlations = []
            for position in current_positions:
                if position in corr_matrix.columns:
                    correlations.append(abs(corr_matrix.loc[candidate, position]))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                suggestions.append((candidate, float(avg_correlation)))
        
        # Sort by lowest correlation (best diversification)
        suggestions.sort(key=lambda x: x[1])
        
        return suggestions[:max_suggestions]
    
    def export_correlation_matrix(self, filepath: str) -> None:
        """
        Export correlation matrix to CSV.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.correlation_matrix.empty:
            self.calculate_correlation_matrix()
        
        if not self.correlation_matrix.empty:
            self.correlation_matrix.to_csv(filepath)
            logger.info(f"Exported correlation matrix to {filepath}")
    
    def get_correlation_summary(self) -> Dict:
        """
        Get summary statistics of correlations.
        
        Returns:
            Dictionary with correlation statistics
        """
        if self.correlation_matrix.empty:
            self.calculate_correlation_matrix()
        
        if self.correlation_matrix.empty:
            return {
                "mean_correlation": 0.0,
                "median_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_pairs": 0,
            }
        
        # Get upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool), k=1)
        correlations = self.correlation_matrix.where(mask).values.flatten()
        correlations = correlations[~np.isnan(correlations)]
        
        if len(correlations) == 0:
            return {
                "mean_correlation": 0.0,
                "median_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_pairs": 0,
            }
        
        high_corr_count = np.sum(np.abs(correlations) >= self.correlation_threshold)
        
        return {
            "mean_correlation": float(np.mean(correlations)),
            "median_correlation": float(np.median(correlations)),
            "max_correlation": float(np.max(correlations)),
            "min_correlation": float(np.min(correlations)),
            "high_correlation_pairs": int(high_corr_count),
            "total_pairs": len(correlations),
            "last_updated": self.last_update.isoformat() if self.last_update else None,
        }
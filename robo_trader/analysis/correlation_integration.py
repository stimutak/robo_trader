"""
Integration of correlation analysis with trading pipeline.

This module implements:
- Correlation-based position sizing
- Dynamic position limits based on correlations
- Integration with async runner
- Real-time correlation updates
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..correlation import CorrelationTracker
from ..logger import get_logger
from ..risk_manager import Position


class CorrelationBasedPositionSizer:
    """Position sizing with correlation awareness."""

    def __init__(
        self,
        correlation_tracker: CorrelationTracker,
        max_correlation: float = 0.7,
        correlation_penalty_factor: float = 0.5,
        max_correlated_exposure: float = 0.3,
    ):
        """
        Initialize correlation-based position sizer.

        Args:
            correlation_tracker: CorrelationTracker instance
            max_correlation: Maximum allowed correlation between positions
            correlation_penalty_factor: How much to reduce position size for correlated assets
            max_correlated_exposure: Maximum portfolio exposure to correlated positions
        """
        self.correlation_tracker = correlation_tracker
        self.max_correlation = max_correlation
        self.correlation_penalty_factor = correlation_penalty_factor
        self.max_correlated_exposure = max_correlated_exposure
        self.logger = get_logger("correlation.sizer")

        # Track current positions
        self.positions: Dict[str, Position] = {}

        # Performance metrics
        self.metrics = {"positions_reduced": 0, "positions_rejected": 0, "avg_correlation": []}

    async def calculate_position_size(
        self,
        symbol: str,
        base_size: int,
        current_positions: Dict[str, Position],
        portfolio_value: float,
    ) -> Tuple[int, str]:
        """
        Calculate position size considering correlations.

        Args:
            symbol: Symbol to size
            base_size: Base position size before correlation adjustment
            current_positions: Current portfolio positions
            portfolio_value: Total portfolio value

        Returns:
            Tuple of (adjusted_size, reason)
        """
        self.positions = current_positions

        # If no other positions, return base size
        if not current_positions or len(current_positions) == 1:
            return base_size, "No correlation adjustment needed"

        # Get correlations with existing positions
        position_correlations = []
        for pos_symbol, position in current_positions.items():
            if pos_symbol != symbol and position.quantity != 0:
                correlation = self.correlation_tracker.get_pairwise_correlation(symbol, pos_symbol)
                if correlation is not None:
                    position_correlations.append(
                        {
                            "symbol": pos_symbol,
                            "correlation": correlation,
                            "exposure": position.notional_value / portfolio_value,
                        }
                    )

        if not position_correlations:
            return base_size, "No correlation data available"

        # Calculate correlation-weighted exposure
        total_correlated_exposure = sum(
            pc["exposure"] * abs(pc["correlation"]) for pc in position_correlations
        )

        # Find highest correlation
        max_corr = max(abs(pc["correlation"]) for pc in position_correlations)

        # Store metrics
        avg_correlation = np.mean([abs(pc["correlation"]) for pc in position_correlations])
        self.metrics["avg_correlation"].append(avg_correlation)
        if len(self.metrics["avg_correlation"]) > 100:
            self.metrics["avg_correlation"] = self.metrics["avg_correlation"][-100:]

        # Apply correlation-based adjustments
        adjusted_size = base_size
        reasons = []

        # Reduce size if highly correlated
        if max_corr > self.max_correlation:
            reduction_factor = (
                1 - (max_corr - self.max_correlation) * self.correlation_penalty_factor
            )
            reduction_factor = max(0.2, reduction_factor)  # Minimum 20% of original size
            adjusted_size = int(base_size * reduction_factor)
            reasons.append(
                f"Reduced by {(1-reduction_factor)*100:.1f}% due to {max_corr:.2f} correlation"
            )
            self.metrics["positions_reduced"] += 1

        # Check total correlated exposure
        if total_correlated_exposure > self.max_correlated_exposure:
            # Further reduce position
            exposure_reduction = self.max_correlated_exposure / total_correlated_exposure
            adjusted_size = int(adjusted_size * exposure_reduction)
            reasons.append(f"Reduced for {total_correlated_exposure:.1%} correlated exposure")

        # Reject if correlation too high
        if max_corr > 0.95:
            self.metrics["positions_rejected"] += 1
            return 0, f"Rejected: {max_corr:.2f} correlation too high"

        reason = " | ".join(reasons) if reasons else "Correlation within limits"

        self.logger.info(
            f"Position sizing for {symbol}: base={base_size}, "
            f"adjusted={adjusted_size}, max_corr={max_corr:.2f}, "
            f"reason={reason}"
        )

        return adjusted_size, reason

    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Get correlation matrix for given symbols."""
        return self.correlation_tracker.calculate_correlation_matrix(symbols)

    def get_high_correlation_pairs(
        self, threshold: Optional[float] = None
    ) -> List[Tuple[str, str, float]]:
        """Get pairs of positions with high correlation."""
        if not self.positions:
            return []

        position_symbols = list(self.positions.keys())
        high_correlations = []

        for i, sym1 in enumerate(position_symbols):
            for j, sym2 in enumerate(position_symbols[i + 1 :], i + 1):
                correlation = self.correlation_tracker.get_pairwise_correlation(sym1, sym2)
                if correlation is not None:
                    threshold_value = threshold or self.max_correlation
                    if abs(correlation) >= threshold_value:
                        high_correlations.append((sym1, sym2, correlation))

        return sorted(high_correlations, key=lambda x: abs(x[2]), reverse=True)

    def calculate_portfolio_concentration(self) -> Dict[str, float]:
        """Calculate portfolio concentration metrics."""
        if not self.positions:
            return {
                "herfindahl_index": 0,
                "effective_n": 0,
                "max_position_weight": 0,
                "correlation_adjusted_concentration": 0,
            }

        # Calculate position weights
        total_value = sum(p.notional_value for p in self.positions.values())
        if total_value == 0:
            return {
                "herfindahl_index": 0,
                "effective_n": 0,
                "max_position_weight": 0,
                "correlation_adjusted_concentration": 0,
            }

        weights = {
            symbol: position.notional_value / total_value
            for symbol, position in self.positions.items()
        }

        # Herfindahl index
        herfindahl = sum(w**2 for w in weights.values())

        # Effective N (number of equally-weighted positions)
        effective_n = 1 / herfindahl if herfindahl > 0 else 0

        # Maximum position weight
        max_weight = max(weights.values()) if weights else 0

        # Correlation-adjusted concentration
        position_symbols = list(self.positions.keys())
        corr_matrix = self.correlation_tracker.calculate_correlation_matrix(position_symbols)

        if not corr_matrix.empty:
            # Calculate correlation-weighted concentration
            corr_concentration = 0
            for sym1 in position_symbols:
                for sym2 in position_symbols:
                    if sym1 in corr_matrix.index and sym2 in corr_matrix.columns:
                        w1 = weights.get(sym1, 0)
                        w2 = weights.get(sym2, 0)
                        corr = corr_matrix.loc[sym1, sym2]
                        corr_concentration += w1 * w2 * corr
        else:
            corr_concentration = herfindahl

        return {
            "herfindahl_index": herfindahl,
            "effective_n": effective_n,
            "max_position_weight": max_weight,
            "correlation_adjusted_concentration": corr_concentration,
        }

    def suggest_diversification(
        self, current_positions: Dict[str, Position], candidate_symbols: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Suggest symbols for diversification based on low correlation.

        Args:
            current_positions: Current portfolio positions
            candidate_symbols: List of potential symbols to add

        Returns:
            List of (symbol, avg_correlation) sorted by lowest correlation
        """
        if not current_positions:
            return [(symbol, 0.0) for symbol in candidate_symbols]

        position_symbols = list(current_positions.keys())
        suggestions = []

        for candidate in candidate_symbols:
            if candidate not in position_symbols:
                # Calculate average correlation with existing positions
                correlations = []
                for pos_symbol in position_symbols:
                    corr = self.correlation_tracker.get_pairwise_correlation(candidate, pos_symbol)
                    if corr is not None:
                        correlations.append(abs(corr))

                if correlations:
                    avg_correlation = np.mean(correlations)
                    suggestions.append((candidate, avg_correlation))

        # Sort by lowest average correlation
        suggestions.sort(key=lambda x: x[1])

        return suggestions

    def get_metrics(self) -> Dict[str, any]:
        """Get position sizing metrics."""
        return {
            "positions_reduced": self.metrics["positions_reduced"],
            "positions_rejected": self.metrics["positions_rejected"],
            "avg_correlation": (
                np.mean(self.metrics["avg_correlation"]) if self.metrics["avg_correlation"] else 0
            ),
            "concentration_metrics": self.calculate_portfolio_concentration(),
        }


class AsyncCorrelationManager:
    """Async wrapper for correlation tracking in the trading pipeline."""

    def __init__(
        self, correlation_tracker: CorrelationTracker, position_sizer: CorrelationBasedPositionSizer
    ):
        self.correlation_tracker = correlation_tracker
        self.position_sizer = position_sizer
        self.logger = get_logger("correlation.async")

        # Update task
        self.update_task: Optional[asyncio.Task] = None
        self.running = False

    async def start(self, update_interval: int = 300):
        """Start async correlation updates."""
        if self.running:
            return

        self.running = True
        self.update_task = asyncio.create_task(self._update_loop(update_interval))
        self.logger.info("Started async correlation manager")

    async def stop(self):
        """Stop async correlation updates."""
        self.running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped async correlation manager")

    async def _update_loop(self, interval: int):
        """Periodic correlation update loop."""
        while self.running:
            try:
                # Update correlations in background
                await asyncio.get_event_loop().run_in_executor(None, self._update_correlations)

                # Log high correlations
                high_corr = self.position_sizer.get_high_correlation_pairs()
                if high_corr:
                    self.logger.warning(f"High correlations detected: {high_corr[:3]}")

                await asyncio.sleep(interval)

            except Exception as e:
                self.logger.error(f"Correlation update error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    def _update_correlations(self):
        """Update correlation matrix (run in executor)."""
        try:
            # Force recalculation
            symbols = list(self.position_sizer.positions.keys())
            if symbols:
                self.correlation_tracker.calculate_correlation_matrix(
                    symbols=symbols, force_update=True
                )

                # Log concentration metrics
                concentration = self.position_sizer.calculate_portfolio_concentration()
                self.logger.info(
                    f"Portfolio concentration: HHI={concentration['herfindahl_index']:.3f}, "
                    f"Effective N={concentration['effective_n']:.1f}"
                )
        except Exception as e:
            self.logger.error(f"Failed to update correlations: {e}")

    async def get_adjusted_position_size(
        self,
        symbol: str,
        base_size: int,
        current_positions: Dict[str, Position],
        portfolio_value: float,
    ) -> Tuple[int, str]:
        """Get correlation-adjusted position size."""
        return await self.position_sizer.calculate_position_size(
            symbol=symbol,
            base_size=base_size,
            current_positions=current_positions,
            portfolio_value=portfolio_value,
        )

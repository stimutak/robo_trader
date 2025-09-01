"""Multi-Strategy Portfolio Manager for RoboTrader.

This module implements sophisticated portfolio management with:
- Dynamic capital allocation across strategies
- Risk budgeting and correlation-aware diversification
- Performance attribution and rebalancing
- Multi-strategy coordination
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

# Minimal imports to avoid circular dependencies
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..risk import Position, RiskManager
    from ..strategies.framework import Strategy

logger = structlog.get_logger(__name__)


class AllocationMethod(Enum):
    """Portfolio allocation methods."""
    
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"
    KELLY_OPTIMAL = "kelly_optimal"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyAllocation:
    """Allocation for a single strategy."""
    
    strategy_name: str
    target_weight: float
    current_weight: float
    allocated_capital: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    last_rebalance: Optional[datetime] = None


@dataclass
class PortfolioMetrics:
    """Portfolio-level performance metrics."""
    
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    correlation_matrix: Optional[pd.DataFrame] = None
    strategy_contributions: Dict[str, float] = field(default_factory=dict)


class MultiStrategyPortfolioManager:
    """Advanced portfolio manager for multiple trading strategies."""
    
    def __init__(
        self,
        config: Any,  # Config
        risk_manager: Any,  # RiskManager
        allocation_method: AllocationMethod = AllocationMethod.ADAPTIVE,
        rebalance_frequency: str = "daily",
        max_strategy_weight: float = 0.4,
        min_strategy_weight: float = 0.05,
    ):
        self.config = config
        self.risk_manager = risk_manager
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency
        self.max_strategy_weight = max_strategy_weight
        self.min_strategy_weight = min_strategy_weight
        
        # Strategy management
        self.strategies: Dict[str, Any] = {}  # Dict[str, Strategy]
        self.allocations: Dict[str, StrategyAllocation] = {}
        
        # Portfolio state
        self.total_capital: float = 0.0
        self.available_capital: float = 0.0
        self.portfolio_metrics = PortfolioMetrics()
        
        # Performance tracking
        self.returns_history: List[float] = []
        self.strategy_returns: Dict[str, List[float]] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        
        # Rebalancing
        self.last_rebalance: Optional[datetime] = None
        self.rebalance_threshold: float = 0.05  # 5% drift threshold
        
        self.logger = logger.bind(component="portfolio_manager")
    
    def register_strategy(
        self,
        strategy: Any,  # Strategy
        initial_weight: float = 0.0,
        risk_budget: float = 0.0,
    ) -> None:
        """Register a new strategy with the portfolio manager."""
        
        if strategy.name in self.strategies:
            self.logger.warning(f"Strategy {strategy.name} already registered")
            return
        
        self.strategies[strategy.name] = strategy
        self.strategy_returns[strategy.name] = []
        
        # Create allocation
        allocation = StrategyAllocation(
            strategy_name=strategy.name,
            target_weight=initial_weight,
            current_weight=0.0,
            allocated_capital=0.0,
        )
        
        self.allocations[strategy.name] = allocation
        
        self.logger.info(
            "Strategy registered",
            strategy=strategy.name,
            initial_weight=initial_weight,
            risk_budget=risk_budget,
        )
    
    def update_capital(self, total_capital: float) -> None:
        """Update total available capital."""
        self.total_capital = total_capital
        self.available_capital = total_capital
        
        # Recalculate allocations
        self._update_allocations()
    
    async def allocate_capital(self) -> Dict[str, float]:
        """Allocate capital across strategies based on allocation method."""
        
        if not self.strategies:
            return {}
        
        if self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
            weights = self._calculate_equal_weights()
        elif self.allocation_method == AllocationMethod.RISK_PARITY:
            weights = await self._calculate_risk_parity_weights()
        elif self.allocation_method == AllocationMethod.MEAN_VARIANCE:
            weights = await self._calculate_mean_variance_weights()
        elif self.allocation_method == AllocationMethod.KELLY_OPTIMAL:
            weights = await self._calculate_kelly_weights()
        else:  # ADAPTIVE
            weights = await self._calculate_adaptive_weights()
        
        # Apply constraints
        weights = self._apply_weight_constraints(weights)
        
        # Update allocations
        for strategy_name, weight in weights.items():
            if strategy_name in self.allocations:
                self.allocations[strategy_name].target_weight = weight
                self.allocations[strategy_name].allocated_capital = (
                    self.total_capital * weight
                )
        
        self.logger.info("Capital allocated", weights=weights)
        return weights
    
    def _calculate_equal_weights(self) -> Dict[str, float]:
        """Calculate equal weights for all strategies."""
        n_strategies = len(self.strategies)
        if n_strategies == 0:
            return {}
        
        weight = 1.0 / n_strategies
        return {name: weight for name in self.strategies.keys()}
    
    async def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calculate risk parity weights based on strategy volatilities."""
        
        if len(self.strategy_returns) < 2:
            return self._calculate_equal_weights()
        
        # Calculate volatilities
        volatilities = {}
        for strategy_name, returns in self.strategy_returns.items():
            if len(returns) >= 10:  # Need minimum history
                vol = np.std(returns) * np.sqrt(252)  # Annualized
                volatilities[strategy_name] = max(vol, 0.01)  # Minimum vol
        
        if not volatilities:
            return self._calculate_equal_weights()
        
        # Risk parity: weight inversely proportional to volatility
        inv_vols = {name: 1.0 / vol for name, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        
        weights = {name: inv_vol / total_inv_vol for name, inv_vol in inv_vols.items()}
        
        # Fill missing strategies with equal weight
        missing_strategies = set(self.strategies.keys()) - set(weights.keys())
        if missing_strategies:
            remaining_weight = 0.1  # Reserve 10% for new strategies
            equal_weight = remaining_weight / len(missing_strategies)
            
            # Scale down existing weights
            scale_factor = (1.0 - remaining_weight) / sum(weights.values())
            weights = {name: weight * scale_factor for name, weight in weights.items()}
            
            # Add missing strategies
            for strategy_name in missing_strategies:
                weights[strategy_name] = equal_weight
        
        return weights
    
    async def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance and risk."""
        
        if len(self.strategy_returns) < 2:
            return self._calculate_equal_weights()
        
        weights = {}
        performance_scores = {}
        
        # Calculate performance scores
        for strategy_name, returns in self.strategy_returns.items():
            if len(returns) >= 20:  # Need sufficient history
                recent_returns = returns[-20:]  # Last 20 periods
                
                # Calculate Sharpe ratio
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                sharpe = mean_return / max(std_return, 0.001) if std_return > 0 else 0
                
                # Calculate maximum drawdown
                cumulative = np.cumprod(1 + np.array(recent_returns))
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_dd = abs(np.min(drawdown))
                
                # Combined score (higher is better)
                score = sharpe * (1 - max_dd)
                performance_scores[strategy_name] = max(score, 0.1)  # Minimum score
        
        if not performance_scores:
            return self._calculate_equal_weights()
        
        # Convert scores to weights
        total_score = sum(performance_scores.values())
        weights = {
            name: score / total_score for name, score in performance_scores.items()
        }
        
        # Fill missing strategies
        missing_strategies = set(self.strategies.keys()) - set(weights.keys())
        if missing_strategies:
            remaining_weight = 0.1
            equal_weight = remaining_weight / len(missing_strategies)
            
            scale_factor = (1.0 - remaining_weight) / sum(weights.values())
            weights = {name: weight * scale_factor for name, weight in weights.items()}
            
            for strategy_name in missing_strategies:
                weights[strategy_name] = equal_weight
        
        return weights

    async def _calculate_mean_variance_weights(self) -> Dict[str, float]:
        """Mean-variance inspired weights using Sharpe-like scoring.

        Approximates mean-variance optimization by setting weights ∝ μ/σ
        over recent returns, with fallbacks to equal weight when insufficient data.
        """
        if not self.strategy_returns:
            return self._calculate_equal_weights()

        scores: Dict[str, float] = {}
        for name, ret in self.strategy_returns.items():
            if len(ret) < 20:
                continue
            r = np.array(ret[-60:], dtype=float)
            mu = float(np.mean(r))
            sigma = float(np.std(r))
            if sigma <= 1e-8:
                continue
            score = max(mu, 0.0) / sigma
            scores[name] = max(score, 0.0)

        if not scores:
            return self._calculate_equal_weights()

        total = sum(scores.values())
        weights = {name: scores.get(name, 0.0) / total for name in self.strategies.keys()}
        return weights

    async def _calculate_kelly_weights(self) -> Dict[str, float]:
        """Kelly-optimal inspired weights using mean/variance ratio.

        Uses w_i ∝ max(μ_i, 0)/σ_i^2 on recent returns as a practical, bounded
        approximation of Kelly sizing across strategies.
        """
        if not self.strategy_returns:
            return self._calculate_equal_weights()

        scores: Dict[str, float] = {}
        for name, ret in self.strategy_returns.items():
            if len(ret) < 20:
                continue
            r = np.array(ret[-60:], dtype=float)
            mu = float(np.mean(r))
            var = float(np.var(r))
            if var <= 1e-8:
                continue
            score = max(mu, 0.0) / var
            scores[name] = max(score, 0.0)

        if not scores:
            return self._calculate_equal_weights()

        total = sum(scores.values())
        weights = {name: scores.get(name, 0.0) / total for name in self.strategies.keys()}
        return weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply minimum and maximum weight constraints."""
        
        constrained_weights = {}
        
        for strategy_name, weight in weights.items():
            # Apply min/max constraints
            constrained_weight = max(
                self.min_strategy_weight,
                min(self.max_strategy_weight, weight)
            )
            constrained_weights[strategy_name] = constrained_weight
        
        # Normalize to sum to 1.0
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                name: weight / total_weight
                for name, weight in constrained_weights.items()
            }
        
        return constrained_weights
    
    def _update_allocations(self) -> None:
        """Update current allocations based on capital changes."""
        
        for allocation in self.allocations.values():
            allocation.allocated_capital = self.total_capital * allocation.target_weight
            # Current weight would be calculated from actual positions
            allocation.current_weight = allocation.target_weight  # Simplified
    
    async def should_rebalance(self) -> bool:
        """Check if portfolio should be rebalanced."""
        
        if self.last_rebalance is None:
            return True
        
        # Check time-based rebalancing
        if self.rebalance_frequency == "daily":
            time_threshold = timedelta(days=1)
        elif self.rebalance_frequency == "weekly":
            time_threshold = timedelta(weeks=1)
        else:  # monthly
            time_threshold = timedelta(days=30)
        
        if datetime.now() - self.last_rebalance > time_threshold:
            return True
        
        # Check drift-based rebalancing
        max_drift = 0.0
        for allocation in self.allocations.values():
            drift = abs(allocation.current_weight - allocation.target_weight)
            max_drift = max(max_drift, drift)
        
        return max_drift > self.rebalance_threshold
    
    async def rebalance(self) -> Dict[str, Any]:
        """Rebalance the portfolio to target allocations."""
        
        self.logger.info("Starting portfolio rebalance")
        
        # Recalculate target allocations
        new_weights = await self.allocate_capital()
        
        # Calculate rebalancing trades needed
        rebalance_trades = {}
        
        for strategy_name, allocation in self.allocations.items():
            target_capital = allocation.allocated_capital
            current_capital = allocation.current_weight * self.total_capital
            
            capital_diff = target_capital - current_capital
            
            if abs(capital_diff) > self.total_capital * 0.01:  # 1% threshold
                rebalance_trades[strategy_name] = {
                    "current_capital": current_capital,
                    "target_capital": target_capital,
                    "capital_change": capital_diff,
                    "action": "increase" if capital_diff > 0 else "decrease",
                }
        
        self.last_rebalance = datetime.now()
        
        self.logger.info(
            "Portfolio rebalanced",
            trades_needed=len(rebalance_trades),
            new_weights=new_weights,
        )
        
        return {
            "timestamp": self.last_rebalance,
            "new_weights": new_weights,
            "rebalance_trades": rebalance_trades,
        }
    
    def update_strategy_performance(
        self,
        strategy_name: str,
        return_pct: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update performance tracking for a strategy."""
        
        if strategy_name not in self.strategy_returns:
            self.strategy_returns[strategy_name] = []
        
        self.strategy_returns[strategy_name].append(return_pct)
        
        # Keep only recent history (e.g., last 252 periods)
        if len(self.strategy_returns[strategy_name]) > 252:
            self.strategy_returns[strategy_name] = self.strategy_returns[strategy_name][-252:]
        
        # Update allocation metrics
        if strategy_name in self.allocations:
            if metrics:
                self.allocations[strategy_name].performance_metrics.update(metrics)
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate and return current portfolio metrics."""
        
        if not self.returns_history:
            return self.portfolio_metrics
        
        returns = np.array(self.returns_history)
        
        # Calculate metrics
        total_return = np.prod(1 + returns) - 1
        sharpe_ratio = np.mean(returns) / max(np.std(returns), 0.001) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        volatility = np.std(returns) * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        
        # Strategy contributions
        strategy_contributions = {}
        for strategy_name, allocation in self.allocations.items():
            if strategy_name in self.strategy_returns and self.strategy_returns[strategy_name]:
                strategy_return = np.mean(self.strategy_returns[strategy_name][-20:])
                contribution = allocation.current_weight * strategy_return
                strategy_contributions[strategy_name] = contribution
        
        self.portfolio_metrics = PortfolioMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            var_95=var_95,
            strategy_contributions=strategy_contributions,
        )
        
        return self.portfolio_metrics
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocations."""
        
        summary = {
            "total_capital": self.total_capital,
            "allocation_method": self.allocation_method.value,
            "last_rebalance": self.last_rebalance,
            "strategies": {},
        }
        
        for strategy_name, allocation in self.allocations.items():
            summary["strategies"][strategy_name] = {
                "target_weight": allocation.target_weight,
                "current_weight": allocation.current_weight,
                "allocated_capital": allocation.allocated_capital,
                "performance_metrics": allocation.performance_metrics,
                "last_rebalance": allocation.last_rebalance,
            }
        
        return summary

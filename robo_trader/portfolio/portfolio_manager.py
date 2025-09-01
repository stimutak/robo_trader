"""Multi-Strategy Portfolio Manager.

Implements capital allocation across multiple strategies with support for
- Equal Weight, Risk Parity, and Adaptive methods
- Weight constraints and rebalancing
- Basic portfolio metrics and strategy performance tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, List, Any

import math
import numpy as np
import pandas as pd


class AllocationMethod(Enum):
    """Capital allocation method across strategies."""

    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    ADAPTIVE = "adaptive"


@dataclass
class StrategyAllocation:
    """Allocation state for a single strategy."""

    name: str
    target_weight: float = 0.0
    current_weight: float = 0.0
    allocated_capital: float = 0.0


@dataclass
class PortfolioMetrics:
    """Summary portfolio metrics."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    var_95: float = 0.0
    strategy_contributions: Dict[str, float] = field(default_factory=dict)


class MultiStrategyPortfolioManager:
    """Manages allocation across multiple strategies."""

    def __init__(
        self,
        config: Any,
        risk_manager: Any,
        allocation_method: AllocationMethod = AllocationMethod.EQUAL_WEIGHT,
        rebalance_frequency: str = "weekly",
        max_strategy_weight: float = 0.4,
        min_strategy_weight: float = 0.0,
    ) -> None:
        self.config = config
        self.risk_manager = risk_manager
        self.allocation_method = allocation_method
        self.rebalance_frequency = rebalance_frequency.lower()
        self.max_strategy_weight = float(max_strategy_weight)
        self.min_strategy_weight = float(min_strategy_weight)

        self.total_capital: float = 0.0
        self.last_rebalance: Optional[datetime] = None

        # Strategy state
        self.strategies: Dict[str, Any] = {}
        self.allocations: Dict[str, StrategyAllocation] = {}

        # Performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        self.returns_history: List[float] = []

    # ---- Registration & Capital -------------------------------------------------
    def register_strategy(self, strategy: Any, initial_weight: float = 0.0) -> None:
        """Register a strategy with an initial weight.

        Args:
            strategy: Strategy-like object with `name` attribute
            initial_weight: Starting portfolio weight for the strategy
        """
        name = getattr(strategy, "name", None)
        if not name:
            raise ValueError("Strategy must have a 'name' attribute")

        self.strategies[name] = strategy
        self.allocations[name] = StrategyAllocation(
            name=name, target_weight=float(initial_weight), current_weight=float(initial_weight)
        )
        self.strategy_performance.setdefault(name, {"returns": [], "metrics": {}})
        # Update allocated capital snapshot
        self.allocations[name].allocated_capital = self.total_capital * self.allocations[name].current_weight

    def update_capital(self, total_capital: float) -> None:
        """Update total capital and refresh allocated capital snapshots."""
        self.total_capital = float(total_capital)
        for alloc in self.allocations.values():
            alloc.allocated_capital = self.total_capital * alloc.current_weight

    # ---- Allocation Logic -------------------------------------------------------
    async def allocate_capital(self) -> Dict[str, float]:
        """Compute target weights for all registered strategies.

        Returns:
            Mapping of strategy name -> target weight (sums to 1.0)
        """
        if not self.allocations:
            return {}

        method = self.allocation_method
        names = list(self.allocations.keys())

        if method == AllocationMethod.EQUAL_WEIGHT:
            raw_weights = {n: 1.0 / len(names) for n in names}
        elif method == AllocationMethod.RISK_PARITY:
            raw_weights = self._risk_parity_weights(names)
        elif method == AllocationMethod.ADAPTIVE:
            raw_weights = self._adaptive_weights(names)
        else:
            raw_weights = {n: 1.0 / len(names) for n in names}

        weights = self._apply_weight_constraints(raw_weights)

        # Update target weights and allocated capital snapshot
        for name, w in weights.items():
            alloc = self.allocations[name]
            alloc.target_weight = w
            alloc.allocated_capital = self.total_capital * alloc.current_weight

        return weights

    def _risk_parity_weights(self, names: List[str]) -> Dict[str, float]:
        """Inverse-volatility weights as a simple risk-parity proxy."""
        vols: Dict[str, float] = {}
        for n in names:
            ret = np.array(self.strategy_performance.get(n, {}).get("returns", []), dtype=float)
            vol = float(np.std(ret)) if ret.size > 0 else 0.0
            # Fallback to a moderate vol to avoid division by zero
            vols[n] = vol if vol > 1e-6 else 0.10

        inv_vol = {n: 1.0 / v for n, v in vols.items()}
        total = sum(inv_vol.values())
        return {n: (v / total) for n, v in inv_vol.items()}

    def _adaptive_weights(self, names: List[str]) -> Dict[str, float]:
        """Adaptive weights using return/volatility scoring.

        Score = max(mean_return, 0) / (volatility + eps)
        """
        scores: Dict[str, float] = {}
        for n in names:
            ret = np.array(self.strategy_performance.get(n, {}).get("returns", []), dtype=float)
            mean_r = float(np.mean(ret)) if ret.size > 0 else 0.0
            vol = float(np.std(ret)) if ret.size > 0 else 0.0
            score = max(mean_r, 0.0) / (vol + 1e-6)
            # Guardrail: if no data, provide a small base score
            if ret.size == 0:
                score = 1.0
            scores[n] = score

        total = sum(scores.values())
        if total <= 0:
            # Fallback to equal weights
            return {n: 1.0 / len(names) for n in names}
        return {n: s / total for n, s in scores.items()}

    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamp weights to [min, max] and renormalize to sum to 1.0.

        Uses an iterative approach to enforce bounds while preserving proportions.
        """
        min_w = max(0.0, self.min_strategy_weight)
        max_w = min(1.0, self.max_strategy_weight)
        names = list(weights.keys())

        # Start with clamped values
        clamped = {n: float(np.clip(w, min_w, max_w)) for n, w in weights.items()}

        # If all at bounds, just normalize uniformly within bounds
        if sum(clamped.values()) == 0:
            uniform = 1.0 / len(names)
            clamped = {n: float(np.clip(uniform, min_w, max_w)) for n in names}

        # Iteratively adjust free weights
        fixed = {n for n, w in clamped.items() if (math.isclose(w, min_w) or math.isclose(w, max_w))}
        for _ in range(10):
            total_fixed = sum(clamped[n] for n in fixed)
            free = [n for n in names if n not in fixed]

            remaining = max(1e-9, 1.0 - total_fixed)
            if not free:
                break

            # Proportional redistribution among free weights using original proportions
            base = sum(weights[n] for n in free)
            if base <= 0:
                # If base zero, distribute uniformly among free
                for n in free:
                    clamped[n] = float(np.clip(remaining / len(free), min_w, max_w))
            else:
                for n in free:
                    target = remaining * (weights[n] / base)
                    clamped[n] = float(np.clip(target, min_w, max_w))

            # Update fixed set if any free weights hit bounds
            new_fixed = {n for n, w in clamped.items() if (math.isclose(w, min_w) or math.isclose(w, max_w))}
            if new_fixed == fixed:
                break
            fixed = new_fixed

        # Final normalization to ensure sum exactly 1.0 (with small epsilon safety)
        s = sum(clamped.values())
        if s > 0:
            clamped = {n: w / s for n, w in clamped.items()}
        return clamped

    # ---- Rebalancing ------------------------------------------------------------
    async def should_rebalance(self) -> bool:
        """Determine if rebalancing is due based on frequency."""
        if self.last_rebalance is None:
            return True

        now = datetime.now()
        delta: timedelta = now - self.last_rebalance
        freq = self.rebalance_frequency

        if freq == "daily":
            return delta.days >= 1
        if freq == "weekly":
            return delta.days >= 7
        if freq == "monthly":
            return delta.days >= 30

        # Default: conservative weekly if unknown value
        return delta.days >= 7

    async def rebalance(self) -> Dict[str, Any]:
        """Rebalance the portfolio to target weights.

        Returns:
            Dict with timestamp and new_weights mapping
        """
        new_weights = await self.allocate_capital()
        # Apply changes
        for name, w in new_weights.items():
            alloc = self.allocations[name]
            alloc.current_weight = w
            alloc.allocated_capital = self.total_capital * w

        self.last_rebalance = datetime.now()
        return {"timestamp": self.last_rebalance, "new_weights": dict(new_weights)}

    # ---- Performance Tracking ---------------------------------------------------
    def update_strategy_performance(self, strategy_name: str, return_pct: float, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Record performance for a strategy for use in allocation decisions."""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {"returns": [], "metrics": {}}
        self.strategy_performance[strategy_name]["returns"].append(float(return_pct))
        if metrics:
            # Merge/overwrite provided metrics
            self.strategy_performance[strategy_name]["metrics"].update(metrics)

    # ---- Reporting --------------------------------------------------------------
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Return a summary of current allocations and capital."""
        strategies_info: Dict[str, Any] = {}
        for name, alloc in self.allocations.items():
            strategies_info[name] = {
                "target_weight": alloc.target_weight,
                "current_weight": alloc.current_weight,
                "allocated_capital": alloc.allocated_capital,
            }

        return {
            "total_capital": self.total_capital,
            "allocation_method": self.allocation_method.value,
            "last_rebalance": self.last_rebalance,
            "strategies": strategies_info,
        }

    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Compute basic portfolio-level metrics from returns history."""
        if not self.returns_history:
            return PortfolioMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                var_95=0.0,
                strategy_contributions={},
            )

        r = pd.Series(self.returns_history, dtype=float)
        cumulative = (1 + r).cumprod()
        total_return = float(cumulative.iloc[-1] - 1.0)

        vol = float(r.std())
        ann_vol = vol * np.sqrt(252) if vol > 0 else 0.0
        mean_r = float(r.mean())
        sharpe = (mean_r / vol) * np.sqrt(252) if vol > 0 else 0.0

        rolling_max = cumulative.cummax()
        drawdowns = cumulative / rolling_max - 1.0
        max_dd = float(drawdowns.min()) if not drawdowns.empty else 0.0

        # Simple historical VaR at 95% confidence
        var_95 = float(-np.percentile(r.values, 5)) if len(r) > 0 else 0.0

        # Strategy contributions: average return times current weight
        contributions: Dict[str, float] = {}
        for name, perf in self.strategy_performance.items():
            s_ret = np.array(perf.get("returns", []), dtype=float)
            avg = float(np.mean(s_ret)) if s_ret.size > 0 else 0.0
            w = self.allocations.get(name, StrategyAllocation(name)).current_weight
            contributions[name] = avg * w

        return PortfolioMetrics(
            total_return=total_return,
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            volatility=float(ann_vol),
            var_95=float(var_95),
            strategy_contributions=contributions,
        )


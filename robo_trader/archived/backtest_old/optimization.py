"""Walk-forward optimization framework for strategy parameter tuning."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestEngine
from ..backtest.metrics import PerformanceMetrics
from ..data.pipeline import DataPipeline
from ..strategies.framework import Strategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizationWindow:
    """Defines a time window for walk-forward optimization."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int

    @property
    def train_days(self) -> int:
        """Calculate training period days."""
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        """Calculate test period days."""
        return (self.test_end - self.test_start).days


@dataclass
class ParameterGrid:
    """Defines parameter space for optimization."""

    parameters: Dict[str, List[Any]]

    def generate_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations."""
        keys = list(self.parameters.keys())
        values = [self.parameters[key] for key in keys]

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    @property
    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        total = 1
        for values in self.parameters.values():
            total *= len(values)
        return total


@dataclass
class OptimizationResult:
    """Result from a single optimization run."""

    window_id: int
    parameters: Dict[str, Any]
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    overfitting_score: float
    stability_score: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "window_id": self.window_id,
            "parameters": self.parameters,
            "train_sharpe": self.train_metrics.sharpe_ratio,
            "test_sharpe": self.test_metrics.sharpe_ratio,
            "train_returns": self.train_metrics.total_return,
            "test_returns": self.test_metrics.total_return,
            "overfitting_score": self.overfitting_score,
            "stability_score": self.stability_score,
        }


class WalkForwardOptimizer:
    """
    Implements walk-forward optimization for strategy parameters.

    This helps prevent overfitting by:
    1. Training on in-sample data
    2. Testing on out-of-sample data
    3. Rolling the window forward
    4. Measuring parameter stability
    """

    def __init__(
        self,
        strategy_class: type,
        data_pipeline: DataPipeline,
        backtest_engine: BacktestEngine,
    ):
        """
        Initialize optimizer.

        Args:
            strategy_class: Strategy class to optimize
            data_pipeline: Data source for backtesting
            backtest_engine: Engine for running backtests
        """
        self.strategy_class = strategy_class
        self.data_pipeline = data_pipeline
        self.backtest_engine = backtest_engine
        self.results: List[OptimizationResult] = []

    def create_windows(
        self,
        start_date: datetime,
        end_date: datetime,
        train_days: int = 252,  # 1 year
        test_days: int = 63,  # 3 months
        step_days: int = 21,  # 1 month
    ) -> List[OptimizationWindow]:
        """
        Create walk-forward windows.

        Args:
            start_date: Overall start date
            end_date: Overall end date
            train_days: Training period length
            test_days: Test period length
            step_days: Step size for rolling window

        Returns:
            List of optimization windows
        """
        windows = []
        window_id = 0

        current_date = start_date

        while current_date + timedelta(days=train_days + test_days) <= end_date:
            window = OptimizationWindow(
                train_start=current_date,
                train_end=current_date + timedelta(days=train_days),
                test_start=current_date + timedelta(days=train_days),
                test_end=current_date + timedelta(days=train_days + test_days),
                window_id=window_id,
            )
            windows.append(window)

            window_id += 1
            current_date += timedelta(days=step_days)

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def optimize_window(
        self,
        window: OptimizationWindow,
        param_grid: ParameterGrid,
        symbols: List[str],
        objective_func: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Optimize parameters for a single window.

        Args:
            window: Time window for optimization
            param_grid: Parameter combinations to test
            symbols: Symbols to trade
            objective_func: Custom objective function (default: Sharpe ratio)

        Returns:
            Best optimization result for this window
        """
        if objective_func is None:

            def default_objective(m):
                return m.sharpe_ratio

            objective_func = default_objective

        best_result = None
        best_score = float("-inf")

        # Test each parameter combination
        for params in param_grid.generate_combinations():
            # Run training backtest
            train_metrics = self._run_backtest(
                params=params,
                symbols=symbols,
                start_date=window.train_start,
                end_date=window.train_end,
            )

            # Skip if training failed
            if train_metrics is None:
                continue

            # Calculate objective score
            score = objective_func(train_metrics)

            if score > best_score:
                # Run test backtest with best params
                test_metrics = self._run_backtest(
                    params=params,
                    symbols=symbols,
                    start_date=window.test_start,
                    end_date=window.test_end,
                )

                if test_metrics is not None:
                    # Calculate overfitting score
                    overfitting = self._calculate_overfitting(train_metrics, test_metrics)

                    best_result = OptimizationResult(
                        window_id=window.window_id,
                        parameters=params,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        overfitting_score=overfitting,
                        stability_score=0.0,  # Will be calculated later
                    )
                    best_score = score

        return best_result

    def run_optimization(
        self,
        symbols: List[str],
        param_grid: ParameterGrid,
        start_date: datetime,
        end_date: datetime,
        train_days: int = 252,
        test_days: int = 63,
        step_days: int = 21,
        objective_func: Optional[Callable] = None,
    ) -> List[OptimizationResult]:
        """
        Run full walk-forward optimization.

        Args:
            symbols: Symbols to trade
            param_grid: Parameter space
            start_date: Overall start
            end_date: Overall end
            train_days: Training period
            test_days: Test period
            step_days: Window step size
            objective_func: Optimization objective

        Returns:
            List of optimization results
        """
        # Create windows
        windows = self.create_windows(
            start_date=start_date,
            end_date=end_date,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
        )

        logger.info(
            f"Starting optimization with {param_grid.total_combinations} "
            f"parameter combinations across {len(windows)} windows"
        )

        # Optimize each window
        results = []
        for i, window in enumerate(windows):
            logger.info(f"Optimizing window {i+1}/{len(windows)}")

            result = self.optimize_window(
                window=window,
                param_grid=param_grid,
                symbols=symbols,
                objective_func=objective_func,
            )

            if result:
                results.append(result)

        # Calculate stability scores
        self._calculate_stability_scores(results)

        self.results = results
        return results

    def _run_backtest(
        self,
        params: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[PerformanceMetrics]:
        """
        Run backtest with given parameters.

        Args:
            params: Strategy parameters
            symbols: Symbols to trade
            start_date: Backtest start
            end_date: Backtest end

        Returns:
            Performance metrics or None if failed
        """
        try:
            # Create strategy instance with parameters
            strategy = self.strategy_class(**params)

            # Run backtest
            results = self.backtest_engine.run(
                strategy=strategy,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
            )

            return results.metrics

        except Exception as e:
            logger.warning(f"Backtest failed with params {params}: {e}")
            return None

    def _calculate_overfitting(
        self, train_metrics: PerformanceMetrics, test_metrics: PerformanceMetrics
    ) -> float:
        """
        Calculate overfitting score.

        Lower is better. Score > 0.5 suggests overfitting.

        Args:
            train_metrics: In-sample performance
            test_metrics: Out-of-sample performance

        Returns:
            Overfitting score [0, 1]
        """
        # Compare Sharpe ratios
        sharpe_degradation = 0.0
        if train_metrics.sharpe_ratio > 0:
            sharpe_degradation = max(
                0, 1 - (test_metrics.sharpe_ratio / train_metrics.sharpe_ratio)
            )

        # Compare returns
        return_degradation = 0.0
        if train_metrics.total_return > 0:
            return_degradation = max(
                0, 1 - (test_metrics.total_return / train_metrics.total_return)
            )

        # Average degradation
        overfitting_score = (sharpe_degradation + return_degradation) / 2

        return min(1.0, max(0.0, overfitting_score))

    def _calculate_stability_scores(self, results: List[OptimizationResult]) -> None:
        """
        Calculate parameter stability across windows.

        Args:
            results: Optimization results to analyze
        """
        if len(results) < 2:
            return

        # Group by parameter set
        param_performance = {}

        for result in results:
            param_key = json.dumps(result.parameters, sort_keys=True)

            if param_key not in param_performance:
                param_performance[param_key] = []

            param_performance[param_key].append(result.test_metrics.sharpe_ratio)

        # Calculate stability for each result
        for result in results:
            param_key = json.dumps(result.parameters, sort_keys=True)
            sharpe_values = param_performance[param_key]

            if len(sharpe_values) > 1:
                # Stability = 1 - coefficient of variation
                mean_sharpe = np.mean(sharpe_values)
                std_sharpe = np.std(sharpe_values)

                if mean_sharpe != 0:
                    cv = std_sharpe / abs(mean_sharpe)
                    result.stability_score = max(0, 1 - cv)
                else:
                    result.stability_score = 0.0
            else:
                result.stability_score = 0.5  # Neutral if only one window

    def get_best_parameters(
        self, min_stability: float = 0.7, max_overfitting: float = 0.3
    ) -> Dict[str, Any]:
        """
        Get best parameters based on all windows.

        Args:
            min_stability: Minimum stability score
            max_overfitting: Maximum overfitting score

        Returns:
            Best parameter set
        """
        if not self.results:
            return {}

        # Filter by stability and overfitting
        valid_results = [
            r
            for r in self.results
            if r.stability_score >= min_stability and r.overfitting_score <= max_overfitting
        ]

        if not valid_results:
            logger.warning("No parameters meet stability/overfitting criteria")
            valid_results = self.results

        # Find best by test Sharpe
        best_result = max(valid_results, key=lambda r: r.test_metrics.sharpe_ratio)

        return best_result.parameters

    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to JSON.

        Args:
            filepath: Path to save results
        """
        results_data = [r.to_dict() for r in self.results]

        with open(filepath, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Saved {len(self.results)} results to {filepath}")

    def plot_results(self) -> None:
        """Plot optimization results (placeholder for visualization)."""
        # This would create plots showing:
        # 1. Parameter stability across windows
        # 2. Train vs test performance
        # 3. Overfitting scores
        # 4. Best parameters over time
        pass


class ParameterSweeper:
    """
    Tool for systematic parameter exploration.
    """

    @staticmethod
    def create_grid(
        base_params: Dict[str, Any], sweep_params: Dict[str, List[Any]]
    ) -> ParameterGrid:
        """
        Create parameter grid from base + sweep parameters.

        Args:
            base_params: Fixed parameters
            sweep_params: Parameters to sweep

        Returns:
            Parameter grid for optimization
        """
        # Combine base and sweep params
        all_params = {}

        # Add base params as single-item lists
        for key, value in base_params.items():
            all_params[key] = [value]

        # Add sweep params
        all_params.update(sweep_params)

        return ParameterGrid(parameters=all_params)

    @staticmethod
    def adaptive_grid(
        initial_params: Dict[str, Tuple[float, float]],
        iterations: int = 3,
        refinement_factor: float = 0.5,
    ) -> List[ParameterGrid]:
        """
        Create adaptive parameter grids that refine around best values.

        Args:
            initial_params: Initial parameter ranges (min, max)
            iterations: Number of refinement iterations
            refinement_factor: How much to narrow the search

        Returns:
            List of progressively refined grids
        """
        grids = []
        current_ranges = initial_params.copy()

        for i in range(iterations):
            # Create grid for this iteration
            grid_params = {}

            for param, (min_val, max_val) in current_ranges.items():
                # Create evenly spaced values
                if isinstance(min_val, int):
                    values = list(
                        range(
                            int(min_val),
                            int(max_val) + 1,
                            max(1, (int(max_val) - int(min_val)) // 4),
                        )
                    )
                else:
                    values = np.linspace(min_val, max_val, 5).tolist()

                grid_params[param] = values

            grids.append(ParameterGrid(parameters=grid_params))

            # Refine ranges for next iteration
            # (In practice, this would use results from previous iteration)
            for param in current_ranges:
                min_val, max_val = current_ranges[param]
                range_size = max_val - min_val
                center = (min_val + max_val) / 2
                new_range = range_size * refinement_factor

                current_ranges[param] = (center - new_range / 2, center + new_range / 2)

        return grids


class OverfittingDetector:
    """
    Detects and quantifies overfitting in strategy optimization.
    """

    @staticmethod
    def calculate_degradation(
        train_metrics: PerformanceMetrics, test_metrics: PerformanceMetrics
    ) -> Dict[str, float]:
        """
        Calculate performance degradation from train to test.

        Args:
            train_metrics: In-sample metrics
            test_metrics: Out-of-sample metrics

        Returns:
            Degradation scores for various metrics
        """
        degradation = {}

        # Sharpe ratio degradation
        if train_metrics.sharpe_ratio != 0:
            degradation["sharpe"] = 1 - (test_metrics.sharpe_ratio / train_metrics.sharpe_ratio)
        else:
            degradation["sharpe"] = 0.0

        # Return degradation
        if train_metrics.total_return != 0:
            degradation["return"] = 1 - (test_metrics.total_return / train_metrics.total_return)
        else:
            degradation["return"] = 0.0

        # Win rate degradation
        if train_metrics.win_rate != 0:
            degradation["win_rate"] = 1 - (test_metrics.win_rate / train_metrics.win_rate)
        else:
            degradation["win_rate"] = 0.0

        # Max drawdown increase (inverse degradation)
        if test_metrics.max_drawdown != 0:
            degradation["drawdown"] = (train_metrics.max_drawdown / test_metrics.max_drawdown) - 1
        else:
            degradation["drawdown"] = 0.0

        return degradation

    @staticmethod
    def complexity_penalty(n_parameters: int, n_samples: int, base_penalty: float = 0.01) -> float:
        """
        Calculate complexity penalty based on parameter count.

        Args:
            n_parameters: Number of parameters
            n_samples: Number of data samples
            base_penalty: Base penalty per parameter

        Returns:
            Complexity penalty score
        """
        # Penalty increases with parameter/sample ratio
        ratio = n_parameters / max(1, n_samples)

        # Exponential penalty for high complexity
        penalty = base_penalty * n_parameters * (1 + ratio) ** 2

        return min(1.0, penalty)

    @staticmethod
    def monte_carlo_test(
        strategy_returns: pd.Series,
        n_simulations: int = 1000,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Run Monte Carlo simulation to test if returns are random.

        Args:
            strategy_returns: Strategy returns series
            n_simulations: Number of simulations
            confidence_level: Confidence level for test

        Returns:
            Test statistics and p-value
        """
        actual_sharpe = strategy_returns.mean() / strategy_returns.std()

        # Generate random returns with same mean/std
        random_sharpes = []
        for _ in range(n_simulations):
            random_returns = np.random.normal(
                loc=strategy_returns.mean(),
                scale=strategy_returns.std(),
                size=len(strategy_returns),
            )
            random_sharpe = np.mean(random_returns) / np.std(random_returns)
            random_sharpes.append(random_sharpe)

        # Calculate p-value
        p_value = np.mean([s >= actual_sharpe for s in random_sharpes])

        # Calculate confidence bounds
        lower_bound = np.percentile(random_sharpes, (1 - confidence_level) * 100 / 2)
        upper_bound = np.percentile(random_sharpes, 100 - (1 - confidence_level) * 100 / 2)

        return {
            "actual_sharpe": actual_sharpe,
            "p_value": p_value,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "is_significant": p_value < (1 - confidence_level),
        }

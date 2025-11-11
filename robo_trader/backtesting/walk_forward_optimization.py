"""
Walk-forward optimization framework for robust strategy validation.
"""

import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Single window in walk-forward analysis."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    window_id: int


@dataclass
class OptimizationResult:
    """Result from optimization run."""

    parameters: Dict[str, Any]
    in_sample_performance: Dict[str, float]
    out_of_sample_performance: Dict[str, float]
    window: WalkForwardWindow
    overfitting_score: float


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy parameters.

    Implements:
    - Rolling window optimization
    - Out-of-sample validation
    - Overfitting detection
    - Parameter stability analysis
    - Anchored and rolling walk-forward
    """

    def __init__(
        self,
        strategy_class: type,
        parameter_space: Dict[str, List[Any]],
        objective_function: str = "sharpe_ratio",
        window_type: str = "rolling",
        train_periods: int = 252,  # Trading days
        test_periods: int = 63,  # Trading days
        step_periods: int = 21,  # Trading days
        min_trades: int = 30,
        parallel: bool = True,
        n_jobs: int = -1,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            strategy_class: Strategy class to optimize
            parameter_space: Dict of parameter names to list of values to test
            objective_function: Metric to optimize ('sharpe_ratio', 'returns', 'calmar')
            window_type: 'rolling' or 'anchored'
            train_periods: Number of periods for training
            test_periods: Number of periods for testing
            step_periods: Step size for moving windows
            min_trades: Minimum trades required for valid backtest
            parallel: Run optimizations in parallel
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.strategy_class = strategy_class
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.window_type = window_type
        self.train_periods = train_periods
        self.test_periods = test_periods
        self.step_periods = step_periods
        self.min_trades = min_trades
        self.parallel = parallel
        self.n_jobs = n_jobs if n_jobs > 0 else None

        self.optimization_results: List[OptimizationResult] = []
        self.best_parameters: Dict[str, Any] = {}

    def create_walk_forward_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows for optimization.

        Args:
            data: Historical price data

        Returns:
            List of WalkForwardWindow objects
        """
        windows = []
        dates = data.index.unique()

        if len(dates) < self.train_periods + self.test_periods:
            raise ValueError("Insufficient data for walk-forward analysis")

        if self.window_type == "rolling":
            # Rolling window - fixed size training window
            start_idx = 0
            window_id = 0

            while start_idx + self.train_periods + self.test_periods <= len(dates):
                train_start = dates[start_idx]
                train_end = dates[start_idx + self.train_periods - 1]
                test_start = dates[start_idx + self.train_periods]
                test_end = dates[
                    min(start_idx + self.train_periods + self.test_periods - 1, len(dates) - 1)
                ]

                windows.append(
                    WalkForwardWindow(
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        window_id=window_id,
                    )
                )

                start_idx += self.step_periods
                window_id += 1

        else:  # anchored
            # Anchored window - expanding training window
            train_start = dates[0]
            start_idx = self.train_periods
            window_id = 0

            while start_idx + self.test_periods <= len(dates):
                train_end = dates[start_idx - 1]
                test_start = dates[start_idx]
                test_end = dates[min(start_idx + self.test_periods - 1, len(dates) - 1)]

                windows.append(
                    WalkForwardWindow(
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        window_id=window_id,
                    )
                )

                start_idx += self.step_periods
                window_id += 1

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def optimize_window(
        self, window: WalkForwardWindow, data: pd.DataFrame, execution_simulator: Any
    ) -> OptimizationResult:
        """
        Optimize parameters for a single window.

        Args:
            window: Walk-forward window
            data: Historical price data
            execution_simulator: Execution simulator for realistic backtesting

        Returns:
            OptimizationResult for the window
        """
        # Split data into train and test
        train_data = data[window.train_start : window.train_end]
        test_data = data[window.test_start : window.test_end]

        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations()

        best_params = None
        best_score = -np.inf
        best_in_sample = {}

        # Test each parameter combination on training data
        for params in param_combinations:
            try:
                # Run backtest on training data
                in_sample_results = self._run_backtest(params, train_data, execution_simulator)

                if in_sample_results["num_trades"] < self.min_trades:
                    continue

                score = in_sample_results.get(self.objective_function, -np.inf)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_in_sample = in_sample_results

            except Exception as e:
                logger.warning(f"Failed to test parameters {params}: {e}")
                continue

        if best_params is None:
            logger.warning(f"No valid parameters found for window {window.window_id}")
            return None

        # Test best parameters on out-of-sample data
        out_sample_results = self._run_backtest(best_params, test_data, execution_simulator)

        # Calculate overfitting score
        overfitting_score = self._calculate_overfitting_score(best_in_sample, out_sample_results)

        return OptimizationResult(
            parameters=best_params,
            in_sample_performance=best_in_sample,
            out_of_sample_performance=out_sample_results,
            window=window,
            overfitting_score=overfitting_score,
        )

    def run_optimization(self, data: pd.DataFrame, execution_simulator: Any) -> Dict[str, Any]:
        """
        Run full walk-forward optimization.

        Args:
            data: Historical price data
            execution_simulator: Execution simulator

        Returns:
            Optimization results and analysis
        """
        # Create walk-forward windows
        windows = self.create_walk_forward_windows(data)

        # Run optimization for each window
        if self.parallel:
            results = self._run_parallel_optimization(windows, data, execution_simulator)
        else:
            results = []
            for window in windows:
                result = self.optimize_window(window, data, execution_simulator)
                if result:
                    results.append(result)
                    logger.info(f"Completed window {window.window_id}/{len(windows)}")

        self.optimization_results = results

        # Analyze results
        analysis = self._analyze_results()

        # Select best parameters
        self.best_parameters = self._select_best_parameters()

        return {
            "best_parameters": self.best_parameters,
            "results": results,
            "analysis": analysis,
            "windows": windows,
        }

    def _generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations to test."""
        import itertools

        keys = list(self.parameter_space.keys())
        values = [self.parameter_space[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _run_backtest(
        self, parameters: Dict[str, Any], data: pd.DataFrame, execution_simulator: Any
    ) -> Dict[str, float]:
        """
        Run backtest with given parameters.

        Returns:
            Performance metrics dictionary
        """
        # Import here to avoid circular dependency
        from .engine import BacktestEngine

        # Create strategy instance with parameters
        strategy = self.strategy_class(**parameters)

        # Create backtest engine
        engine = BacktestEngine(
            strategy=strategy,
            execution_simulator=execution_simulator,
            initial_capital=100000,
            commission=0.001,
        )

        # Run backtest
        results = engine.run(data)

        # Calculate metrics
        metrics = engine.calculate_metrics()

        return metrics

    def _calculate_overfitting_score(
        self, in_sample: Dict[str, float], out_sample: Dict[str, float]
    ) -> float:
        """
        Calculate overfitting score (0 = no overfitting, 1 = severe overfitting).
        """
        # Compare key metrics
        metrics_to_compare = ["sharpe_ratio", "returns", "max_drawdown"]

        scores = []
        for metric in metrics_to_compare:
            if metric in in_sample and metric in out_sample:
                in_val = in_sample[metric]
                out_val = out_sample[metric]

                if in_val != 0:
                    # Calculate degradation ratio
                    if metric == "max_drawdown":
                        # For drawdown, worse means more negative
                        degradation = max(0, (out_val - in_val) / abs(in_val))
                    else:
                        # For returns/sharpe, worse means lower
                        degradation = max(0, (in_val - out_val) / abs(in_val))

                    scores.append(min(1.0, degradation))

        return np.mean(scores) if scores else 0.5

    def _run_parallel_optimization(
        self, windows: List[WalkForwardWindow], data: pd.DataFrame, execution_simulator: Any
    ) -> List[OptimizationResult]:
        """Run optimization in parallel."""
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {
                executor.submit(self.optimize_window, window, data, execution_simulator): window
                for window in windows
            }

            for future in as_completed(futures):
                window = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(f"Completed window {window.window_id}")
                except Exception as e:
                    logger.error(f"Window {window.window_id} failed: {e}")

        return results

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results."""
        if not self.optimization_results:
            return {}

        # Parameter stability analysis
        param_history = {}
        for result in self.optimization_results:
            for param, value in result.parameters.items():
                if param not in param_history:
                    param_history[param] = []
                param_history[param].append(value)

        param_stability = {}
        for param, values in param_history.items():
            # Calculate coefficient of variation
            if len(set(values)) > 1:
                std = np.std(values) if isinstance(values[0], (int, float)) else 0
                mean = np.mean(values) if isinstance(values[0], (int, float)) else 0
                param_stability[param] = std / mean if mean != 0 else 0
            else:
                param_stability[param] = 0

        # Performance degradation analysis
        in_sample_scores = [
            r.in_sample_performance.get(self.objective_function, 0)
            for r in self.optimization_results
        ]
        out_sample_scores = [
            r.out_of_sample_performance.get(self.objective_function, 0)
            for r in self.optimization_results
        ]

        avg_degradation = np.mean(
            [(i - o) / i if i != 0 else 0 for i, o in zip(in_sample_scores, out_sample_scores)]
        )

        # Overfitting analysis
        overfitting_scores = [r.overfitting_score for r in self.optimization_results]

        return {
            "parameter_stability": param_stability,
            "avg_in_sample_score": np.mean(in_sample_scores),
            "avg_out_sample_score": np.mean(out_sample_scores),
            "performance_degradation": avg_degradation,
            "avg_overfitting_score": np.mean(overfitting_scores),
            "consistency_score": (
                1 - np.std(out_sample_scores) / np.mean(out_sample_scores)
                if np.mean(out_sample_scores) != 0
                else 0
            ),
        }

    def _select_best_parameters(self) -> Dict[str, Any]:
        """
        Select best parameters based on out-of-sample performance and stability.
        """
        if not self.optimization_results:
            return {}

        # Score each result
        scores = []
        for result in self.optimization_results:
            # Combine out-of-sample performance with overfitting penalty
            oos_score = result.out_of_sample_performance.get(self.objective_function, 0)
            penalty = result.overfitting_score
            combined_score = oos_score * (1 - penalty * 0.5)  # 50% penalty for overfitting
            scores.append(combined_score)

        # Get best result
        best_idx = np.argmax(scores)
        return self.optimization_results[best_idx].parameters

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Calculate parameter importance based on performance impact.
        """
        if not self.optimization_results:
            return {}

        importance = {}

        for param in self.parameter_space.keys():
            # Group results by parameter value
            value_performance = {}

            for result in self.optimization_results:
                value = result.parameters[param]
                score = result.out_of_sample_performance.get(self.objective_function, 0)

                if value not in value_performance:
                    value_performance[value] = []
                value_performance[value].append(score)

            # Calculate variance across different values
            mean_scores = [np.mean(scores) for scores in value_performance.values()]
            importance[param] = np.var(mean_scores)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def plot_results(self) -> None:
        """Plot optimization results (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt

            if not self.optimization_results:
                logger.warning("No results to plot")
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # In-sample vs out-of-sample performance
            in_sample = [
                r.in_sample_performance.get(self.objective_function, 0)
                for r in self.optimization_results
            ]
            out_sample = [
                r.out_of_sample_performance.get(self.objective_function, 0)
                for r in self.optimization_results
            ]

            axes[0, 0].scatter(in_sample, out_sample, alpha=0.6)
            axes[0, 0].plot(
                [min(in_sample), max(in_sample)], [min(in_sample), max(in_sample)], "r--"
            )
            axes[0, 0].set_xlabel("In-Sample Performance")
            axes[0, 0].set_ylabel("Out-of-Sample Performance")
            axes[0, 0].set_title("In-Sample vs Out-of-Sample")

            # Performance over time
            window_ids = [r.window.window_id for r in self.optimization_results]
            axes[0, 1].plot(window_ids, out_sample, "b-", label="Out-of-Sample")
            axes[0, 1].plot(window_ids, in_sample, "g--", label="In-Sample")
            axes[0, 1].set_xlabel("Window ID")
            axes[0, 1].set_ylabel("Performance")
            axes[0, 1].set_title("Performance Over Time")
            axes[0, 1].legend()

            # Overfitting scores
            overfitting = [r.overfitting_score for r in self.optimization_results]
            axes[1, 0].bar(window_ids, overfitting)
            axes[1, 0].set_xlabel("Window ID")
            axes[1, 0].set_ylabel("Overfitting Score")
            axes[1, 0].set_title("Overfitting Analysis")
            axes[1, 0].axhline(y=0.5, color="r", linestyle="--", label="Threshold")

            # Parameter importance
            importance = self.get_parameter_importance()
            if importance:
                axes[1, 1].bar(importance.keys(), importance.values())
                axes[1, 1].set_xlabel("Parameter")
                axes[1, 1].set_ylabel("Importance")
                axes[1, 1].set_title("Parameter Importance")
                axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")

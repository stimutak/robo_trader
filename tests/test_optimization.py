"""Tests for walk-forward optimization framework."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from robo_trader.backtest.metrics import PerformanceMetrics
from robo_trader.backtest.optimization import (
    OptimizationResult,
    OptimizationWindow,
    OverfittingDetector,
    ParameterGrid,
    ParameterSweeper,
    WalkForwardOptimizer,
)
from robo_trader.strategies.framework import Strategy


class TestOptimizationWindow(unittest.TestCase):
    """Test optimization window functionality."""

    def test_window_creation(self):
        """Test window creation and properties."""
        window = OptimizationWindow(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 1),
            test_start=datetime(2023, 6, 1),
            test_end=datetime(2023, 9, 1),
            window_id=1,
        )

        self.assertEqual(window.window_id, 1)
        self.assertEqual(window.train_days, 151)
        self.assertEqual(window.test_days, 92)


class TestParameterGrid(unittest.TestCase):
    """Test parameter grid generation."""

    def test_generate_combinations(self):
        """Test parameter combination generation."""
        grid = ParameterGrid(
            parameters={
                "fast_period": [10, 20],
                "slow_period": [50, 100],
                "threshold": [0.01],
            }
        )

        combinations = grid.generate_combinations()

        self.assertEqual(len(combinations), 4)
        self.assertEqual(grid.total_combinations, 4)

        # Check all combinations exist
        expected = [
            {"fast_period": 10, "slow_period": 50, "threshold": 0.01},
            {"fast_period": 10, "slow_period": 100, "threshold": 0.01},
            {"fast_period": 20, "slow_period": 50, "threshold": 0.01},
            {"fast_period": 20, "slow_period": 100, "threshold": 0.01},
        ]

        for exp in expected:
            self.assertIn(exp, combinations)


class TestWalkForwardOptimizer(unittest.TestCase):
    """Test walk-forward optimization."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy_class = Mock()
        self.data_pipeline = Mock()
        self.backtest_engine = Mock()

        self.optimizer = WalkForwardOptimizer(
            strategy_class=self.strategy_class,
            data_pipeline=self.data_pipeline,
            backtest_engine=self.backtest_engine,
        )

    def test_create_windows(self):
        """Test window creation for walk-forward analysis."""
        windows = self.optimizer.create_windows(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 1, 1),
            train_days=180,
            test_days=60,
            step_days=30,
        )

        # Should create multiple windows
        self.assertGreater(len(windows), 0)

        # Check first window
        first = windows[0]
        self.assertEqual(first.train_start, datetime(2023, 1, 1))
        self.assertEqual(first.train_days, 180)
        self.assertEqual(first.test_days, 60)

        # Check windows don't exceed end date
        last = windows[-1]
        self.assertLessEqual(last.test_end, datetime(2024, 1, 1))

    def test_calculate_overfitting(self):
        """Test overfitting score calculation."""
        train_metrics = PerformanceMetrics(
            total_return=0.20,
            sharpe_ratio=1.5,
            max_drawdown=-0.10,
            win_rate=0.60,
            profit_factor=1.8,
            total_trades=100,
        )

        test_metrics = PerformanceMetrics(
            total_return=0.10,  # 50% degradation
            sharpe_ratio=0.75,  # 50% degradation
            max_drawdown=-0.15,
            win_rate=0.55,
            profit_factor=1.4,
            total_trades=80,
        )

        score = self.optimizer._calculate_overfitting(train_metrics, test_metrics)

        # Should detect overfitting (degradation)
        self.assertAlmostEqual(score, 0.5, places=2)

    def test_optimize_window(self):
        """Test single window optimization."""
        # Mock backtest results
        mock_result = Mock()
        mock_result.metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            win_rate=0.58,
            profit_factor=1.6,
            total_trades=50,
        )

        self.backtest_engine.run.return_value = mock_result

        window = OptimizationWindow(
            train_start=datetime(2023, 1, 1),
            train_end=datetime(2023, 6, 1),
            test_start=datetime(2023, 6, 1),
            test_end=datetime(2023, 9, 1),
            window_id=0,
        )

        param_grid = ParameterGrid(parameters={"fast_period": [10], "slow_period": [50]})

        result = self.optimizer.optimize_window(
            window=window, param_grid=param_grid, symbols=["AAPL", "GOOGL"]
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.window_id, 0)
        self.assertIn("fast_period", result.parameters)

    def test_run_optimization(self):
        """Test full optimization run."""
        # Mock backtest results
        mock_result = Mock()
        mock_result.metrics = PerformanceMetrics(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            win_rate=0.58,
            profit_factor=1.6,
            total_trades=50,
        )

        self.backtest_engine.run.return_value = mock_result

        param_grid = ParameterGrid(parameters={"fast_period": [10, 20], "slow_period": [50]})

        results = self.optimizer.run_optimization(
            symbols=["AAPL"],
            param_grid=param_grid,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 1),
            train_days=180,
            test_days=60,
            step_days=30,
        )

        self.assertGreater(len(results), 0)

        # Check stability scores were calculated
        for result in results:
            self.assertGreaterEqual(result.stability_score, 0)
            self.assertLessEqual(result.stability_score, 1)

    def test_get_best_parameters(self):
        """Test parameter selection with filters."""
        # Create mock results
        results = [
            OptimizationResult(
                window_id=0,
                parameters={"fast": 10},
                train_metrics=Mock(sharpe_ratio=1.5),
                test_metrics=Mock(sharpe_ratio=1.2),
                overfitting_score=0.2,
                stability_score=0.8,
            ),
            OptimizationResult(
                window_id=1,
                parameters={"fast": 20},
                train_metrics=Mock(sharpe_ratio=1.3),
                test_metrics=Mock(sharpe_ratio=1.4),  # Better test performance
                overfitting_score=0.1,
                stability_score=0.9,
            ),
            OptimizationResult(
                window_id=2,
                parameters={"fast": 30},
                train_metrics=Mock(sharpe_ratio=2.0),
                test_metrics=Mock(sharpe_ratio=0.5),  # Overfitted
                overfitting_score=0.7,
                stability_score=0.3,
            ),
        ]

        self.optimizer.results = results

        # Get best with filters
        best = self.optimizer.get_best_parameters(min_stability=0.7, max_overfitting=0.3)

        # Should select second result (best test Sharpe with good stability)
        self.assertEqual(best["fast"], 20)


class TestParameterSweeper(unittest.TestCase):
    """Test parameter sweep utilities."""

    def test_create_grid(self):
        """Test grid creation from base and sweep params."""
        grid = ParameterSweeper.create_grid(
            base_params={"symbol": "AAPL", "timeframe": "1d"},
            sweep_params={"fast_period": [10, 20, 30], "slow_period": [50, 100]},
        )

        combinations = grid.generate_combinations()

        # Should have 6 combinations (3 * 2)
        self.assertEqual(len(combinations), 6)

        # All should have base params
        for combo in combinations:
            self.assertEqual(combo["symbol"], "AAPL")
            self.assertEqual(combo["timeframe"], "1d")

    def test_adaptive_grid(self):
        """Test adaptive grid refinement."""
        grids = ParameterSweeper.adaptive_grid(
            initial_params={"threshold": (0.0, 1.0), "period": (10, 100)},
            iterations=3,
            refinement_factor=0.5,
        )

        self.assertEqual(len(grids), 3)

        # Each iteration should have narrower ranges
        first_threshold = grids[0].parameters["threshold"]
        last_threshold = grids[2].parameters["threshold"]

        # Range should narrow
        first_range = max(first_threshold) - min(first_threshold)
        last_range = max(last_threshold) - min(last_threshold)
        self.assertLess(last_range, first_range)


class TestOverfittingDetector(unittest.TestCase):
    """Test overfitting detection methods."""

    def test_calculate_degradation(self):
        """Test performance degradation calculation."""
        train_metrics = PerformanceMetrics(
            total_return=0.30,
            sharpe_ratio=2.0,
            max_drawdown=-0.10,
            win_rate=0.65,
            profit_factor=2.0,
            total_trades=120,
        )

        test_metrics = PerformanceMetrics(
            total_return=0.15,  # 50% of train
            sharpe_ratio=1.0,  # 50% of train
            max_drawdown=-0.20,  # 2x worse
            win_rate=0.52,  # 80% of train
            profit_factor=1.5,
            total_trades=110,
        )

        degradation = OverfittingDetector.calculate_degradation(train_metrics, test_metrics)

        self.assertAlmostEqual(degradation["sharpe"], 0.5, places=2)
        self.assertAlmostEqual(degradation["return"], 0.5, places=2)
        self.assertAlmostEqual(degradation["win_rate"], 0.2, places=2)

    def test_complexity_penalty(self):
        """Test complexity penalty calculation."""
        # Low complexity
        penalty1 = OverfittingDetector.complexity_penalty(n_parameters=5, n_samples=1000)
        self.assertLess(penalty1, 0.1)

        # High complexity
        penalty2 = OverfittingDetector.complexity_penalty(n_parameters=50, n_samples=100)
        self.assertGreater(penalty2, 0.5)

        # More parameters = higher penalty
        self.assertGreater(penalty2, penalty1)

    def test_monte_carlo_test(self):
        """Test Monte Carlo significance testing."""
        # Create returns with clear pattern (should be significant)
        trend_returns = pd.Series(np.random.normal(0.001, 0.01, 252))  # Positive drift
        trend_returns = trend_returns.cumsum().pct_change().fillna(0)

        result = OverfittingDetector.monte_carlo_test(
            strategy_returns=trend_returns, n_simulations=100, confidence_level=0.95
        )

        self.assertIn("actual_sharpe", result)
        self.assertIn("p_value", result)
        self.assertIn("is_significant", result)

        # Random returns (should not be significant)
        random_returns = pd.Series(np.random.normal(0, 0.01, 252))  # No drift

        result2 = OverfittingDetector.monte_carlo_test(
            strategy_returns=random_returns, n_simulations=100, confidence_level=0.95
        )

        # Random should have higher p-value
        self.assertGreater(result2["p_value"], 0.05)


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for optimization framework."""

    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        # Create mock components
        strategy_class = Mock()
        data_pipeline = Mock()
        backtest_engine = Mock()

        # Mock strategy class to capture parameters
        def mock_strategy_init(**kwargs):
            strategy = Mock()
            for key, value in kwargs.items():
                setattr(strategy, key, value)
            return strategy

        strategy_class.side_effect = mock_strategy_init

        # Mock backtest results with varying performance
        def mock_backtest(*args, **kwargs):
            result = Mock()
            # Vary performance based on strategy instance
            strategy = kwargs.get("strategy")

            base_return = 0.1
            if hasattr(strategy, "fast_period") and strategy.fast_period == 20:
                base_return = 0.15  # Best parameter

            result.metrics = PerformanceMetrics(
                total_return=base_return,
                sharpe_ratio=base_return * 10,
                max_drawdown=-0.10,
                win_rate=0.55,
                profit_factor=1.5,
                total_trades=60,
            )
            return result

        backtest_engine.run.side_effect = mock_backtest

        # Run optimization
        optimizer = WalkForwardOptimizer(
            strategy_class=strategy_class,
            data_pipeline=data_pipeline,
            backtest_engine=backtest_engine,
        )

        param_grid = ParameterGrid(parameters={"fast_period": [10, 20, 30]})

        results = optimizer.run_optimization(
            symbols=["AAPL"],
            param_grid=param_grid,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),
            train_days=60,
            test_days=30,
            step_days=15,
        )

        # Should have results
        self.assertGreater(len(results), 0)

        # Best parameters should be fast_period=20
        best = optimizer.get_best_parameters()
        self.assertEqual(best.get("fast_period"), 20)


if __name__ == "__main__":
    unittest.main()

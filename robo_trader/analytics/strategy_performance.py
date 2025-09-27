"""Strategy-level performance analytics and attribution.

This module provides comprehensive strategy performance tracking,
attribution analysis, and comparison framework for ML-driven strategies.
"""

import asyncio
import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

from ..ml.model_selector import ModelSelector
from ..utils.market_time import get_market_time
from .performance import PerformanceAnalyzer

logger = structlog.get_logger(__name__)


class StrategyType(Enum):
    """Types of trading strategies."""

    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ML_ENSEMBLE = "ml_ensemble"
    CORRELATION_BASED = "correlation_based"
    HYBRID = "hybrid"


class PerformanceAttribution:
    """Performance attribution analysis for strategies."""

    def __init__(self, analyzer: Optional[PerformanceAnalyzer] = None):
        """Initialize attribution analyzer.

        Args:
            analyzer: Performance analyzer instance
        """
        self.analyzer = analyzer or PerformanceAnalyzer()

    def decompose_returns(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        benchmark_returns: pd.Series,
    ) -> Dict[str, float]:
        """Decompose strategy returns into various components.

        Args:
            strategy_returns: Strategy returns series
            factor_returns: Factor returns DataFrame
            benchmark_returns: Benchmark returns series

        Returns:
            Dictionary with return components
        """
        # Calculate alpha and beta
        beta = self.analyzer.beta(strategy_returns, benchmark_returns)
        alpha = self.analyzer.alpha(strategy_returns, benchmark_returns)

        # Market contribution
        market_return = benchmark_returns.mean() * 252
        market_contribution = beta * market_return

        # Factor contributions (if provided)
        factor_contributions = {}
        if not factor_returns.empty:
            for factor in factor_returns.columns:
                factor_beta = self.analyzer.beta(strategy_returns, factor_returns[factor])
                factor_contribution = factor_beta * factor_returns[factor].mean() * 252
                factor_contributions[f"factor_{factor}"] = factor_contribution

        # Residual (unexplained) return
        total_return = strategy_returns.mean() * 252
        explained_return = market_contribution + sum(factor_contributions.values())
        residual_return = total_return - explained_return

        return {
            "total_return": total_return,
            "alpha": alpha,
            "beta": beta,
            "market_contribution": market_contribution,
            "factor_contributions": factor_contributions,
            "residual_return": residual_return,
            "explained_ratio": explained_return / total_return if total_return != 0 else 0,
        }

    def symbol_attribution(self, positions: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-symbol performance attribution.

        Args:
            positions: DataFrame with position data
            returns: DataFrame with returns by symbol

        Returns:
            DataFrame with symbol-level attribution
        """
        attribution = []

        for symbol in returns.columns:
            if symbol not in positions.columns:
                continue

            symbol_returns = returns[symbol].dropna()
            position_weights = positions[symbol].fillna(0)

            # Calculate contribution
            contribution = (symbol_returns * position_weights).sum()

            # Calculate statistics
            avg_weight = position_weights.mean()
            total_return = symbol_returns.sum()

            attribution.append(
                {
                    "symbol": symbol,
                    "total_return": total_return,
                    "contribution": contribution,
                    "avg_weight": avg_weight,
                    "n_periods": len(symbol_returns),
                    "win_rate": (symbol_returns > 0).mean(),
                }
            )

        return pd.DataFrame(attribution).sort_values("contribution", ascending=False)

    def time_attribution(self, returns: pd.Series, frequency: str = "M") -> pd.DataFrame:
        """Analyze performance by time period.

        Args:
            returns: Returns series
            frequency: Time frequency (D, W, M, Q, Y)

        Returns:
            DataFrame with time-based attribution
        """
        # Resample returns
        period_returns = returns.resample(frequency).sum()

        # Calculate metrics per period
        periods = []
        for period, ret in period_returns.items():
            period_data = returns.loc[
                (returns.index >= period) & (returns.index < period + pd.DateOffset(months=1))
            ]

            periods.append(
                {
                    "period": period,
                    "return": ret,
                    "volatility": period_data.std() * np.sqrt(252),
                    "sharpe": self.analyzer.sharpe_ratio(period_data),
                    "max_drawdown": self.analyzer.maximum_drawdown(period_data)[0],
                    "n_trades": len(period_data),
                }
            )

        return pd.DataFrame(periods)


class StrategyPerformanceTracker:
    """Track and compare strategy performance."""

    def __init__(self, database: Optional[Any] = None, results_dir: Optional[Path] = None):
        """Initialize performance tracker.

        Args:
            database: Database connection
            results_dir: Directory to save results
        """
        self.database = database
        self.results_dir = results_dir or Path("performance_results")
        self.results_dir.mkdir(exist_ok=True)

        self.analyzer = PerformanceAnalyzer()
        self.attribution = PerformanceAttribution(self.analyzer)

        self.strategy_results: Dict[str, Dict] = {}
        self.comparison_cache: Dict[str, pd.DataFrame] = {}

    async def track_strategy_performance(
        self,
        strategy_name: str,
        strategy_type: StrategyType,
        returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
        trades: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Track performance for a strategy.

        Args:
            strategy_name: Name of the strategy
            strategy_type: Type of strategy
            returns: Strategy returns series
            positions: Position data
            trades: Trade data
            benchmark_returns: Benchmark returns for comparison
            metadata: Additional metadata

        Returns:
            Performance results dictionary
        """
        logger.info(f"Tracking performance for strategy: {strategy_name}")

        # Calculate base metrics
        performance = self.analyzer.performance_summary(returns, benchmark_returns)

        # Add strategy-specific metadata
        performance.update(
            {
                "strategy_name": strategy_name,
                "strategy_type": strategy_type.value,
                "start_date": returns.index[0].isoformat() if len(returns) > 0 else None,
                "end_date": returns.index[-1].isoformat() if len(returns) > 0 else None,
                "n_periods": len(returns),
            }
        )

        # Trade statistics
        if trades is not None and not trades.empty:
            trade_stats = self._calculate_trade_statistics(trades)
            performance["trade_statistics"] = trade_stats

        # Position statistics
        if positions is not None and not positions.empty:
            position_stats = self._calculate_position_statistics(positions)
            performance["position_statistics"] = position_stats

        # Rolling metrics
        if len(returns) > 60:
            rolling_metrics = self.analyzer.rolling_metrics(returns, window=60)
            performance["rolling_metrics"] = {
                "sharpe": (
                    rolling_metrics["rolling_sharpe"].dropna().tolist()[-10:]
                    if "rolling_sharpe" in rolling_metrics.columns
                    else []
                ),
                "volatility": (
                    rolling_metrics["rolling_volatility"].dropna().tolist()[-10:]
                    if "rolling_volatility" in rolling_metrics.columns
                    else []
                ),
                "drawdown": (
                    rolling_metrics["rolling_max_dd"].dropna().tolist()[-10:]
                    if "rolling_max_dd" in rolling_metrics.columns
                    else []
                ),
            }
        else:
            performance["rolling_metrics"] = {"sharpe": [], "volatility": [], "drawdown": []}

        # Add custom metadata
        if metadata:
            performance["metadata"] = metadata

        # Store results
        self.strategy_results[strategy_name] = performance

        # Save to database if available
        if self.database:
            await self._save_to_database(strategy_name, performance)

        # Save to file
        self._save_to_file(strategy_name, performance)

        return performance

    def compare_strategies(
        self, strategy_names: Optional[List[str]] = None, metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Compare multiple strategies.

        Args:
            strategy_names: List of strategies to compare (None for all)
            metrics: List of metrics to compare

        Returns:
            DataFrame with strategy comparison
        """
        if strategy_names is None:
            strategy_names = list(self.strategy_results.keys())

        if metrics is None:
            metrics = [
                "annual_return",
                "annual_volatility",
                "sharpe_ratio",
                "max_drawdown",
                "calmar_ratio",
                "win_rate",
            ]

        comparison = []
        for name in strategy_names:
            if name not in self.strategy_results:
                continue

            result = self.strategy_results[name]
            row = {"strategy": name}

            for metric in metrics:
                if metric in result:
                    row[metric] = result[metric]

            comparison.append(row)

        df = pd.DataFrame(comparison)

        # Add ranking
        for metric in metrics:
            if metric in df.columns:
                # Higher is better for most metrics
                ascending = metric in ["annual_volatility", "max_drawdown"]
                df[f"{metric}_rank"] = df[metric].rank(ascending=ascending, method="min")

        # Calculate overall score
        rank_cols = [col for col in df.columns if col.endswith("_rank")]
        if rank_cols:
            df["overall_score"] = df[rank_cols].mean(axis=1)
            df = df.sort_values("overall_score")

        # Cache result
        cache_key = f"{','.join(strategy_names)}_{','.join(metrics)}"
        self.comparison_cache[cache_key] = df

        return df

    def generate_report(self, strategy_name: str, include_charts: bool = False) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Args:
            strategy_name: Strategy to report on
            include_charts: Whether to include chart data

        Returns:
            Report dictionary
        """
        if strategy_name not in self.strategy_results:
            raise ValueError(f"Strategy {strategy_name} not found")

        performance = self.strategy_results[strategy_name]

        report = {
            "strategy": strategy_name,
            "generated_at": get_market_time().isoformat(),
            "summary": {
                "total_return": performance.get("total_return"),
                "annual_return": performance.get("annual_return"),
                "sharpe_ratio": performance.get("sharpe_ratio"),
                "max_drawdown": performance.get("max_drawdown"),
                "win_rate": performance.get("win_rate"),
            },
            "risk_metrics": {
                "volatility": performance.get("annual_volatility"),
                "var_95": performance.get("var_95"),
                "cvar_95": performance.get("cvar_95"),
                "beta": performance.get("beta"),
                "correlation": performance.get("correlation"),
            },
            "trade_analysis": performance.get("trade_statistics", {}),
            "position_analysis": performance.get("position_statistics", {}),
            "rolling_metrics": performance.get("rolling_metrics", {}),
        }

        # Add metadata
        if "metadata" in performance:
            report["metadata"] = performance["metadata"]

        # Add comparison if available
        comparison = self.compare_strategies()
        if not comparison.empty:
            strategy_rank = comparison[comparison["strategy"] == strategy_name]
            if not strategy_rank.empty:
                report["ranking"] = strategy_rank.to_dict("records")[0]

        return report

    def _calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-level statistics.

        Args:
            trades: DataFrame with trade data

        Returns:
            Trade statistics dictionary
        """
        stats = {
            "total_trades": len(trades),
            "avg_trade_size": trades["quantity"].mean() if "quantity" in trades else 0,
            "avg_holding_period": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "largest_win": 0,
            "largest_loss": 0,
        }

        if "pnl" in trades.columns:
            winning_trades = trades[trades["pnl"] > 0]
            losing_trades = trades[trades["pnl"] < 0]

            stats.update(
                {
                    "win_rate": len(winning_trades) / len(trades) if len(trades) > 0 else 0,
                    "avg_win": winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0,
                    "avg_loss": losing_trades["pnl"].mean() if len(losing_trades) > 0 else 0,
                    "largest_win": winning_trades["pnl"].max() if len(winning_trades) > 0 else 0,
                    "largest_loss": losing_trades["pnl"].min() if len(losing_trades) > 0 else 0,
                    "profit_factor": (
                        abs(winning_trades["pnl"].sum() / losing_trades["pnl"].sum())
                        if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0
                        else 0
                    ),
                }
            )

        if "holding_period" in trades.columns:
            stats["avg_holding_period"] = trades["holding_period"].mean()

        return stats

    def _calculate_position_statistics(self, positions: pd.DataFrame) -> Dict:
        """Calculate position-level statistics.

        Args:
            positions: DataFrame with position data

        Returns:
            Position statistics dictionary
        """
        stats = {
            "avg_positions": positions.count(axis=1).mean(),
            "max_positions": positions.count(axis=1).max(),
            "avg_concentration": 0,
            "max_concentration": 0,
            "turnover": 0,
        }

        # Calculate concentration (HHI)
        if not positions.empty:
            position_weights = positions.div(positions.sum(axis=1), axis=0)
            hhi = (position_weights**2).sum(axis=1)
            stats["avg_concentration"] = hhi.mean()
            stats["max_concentration"] = hhi.max()

            # Calculate turnover
            position_changes = positions.diff().abs().sum(axis=1)
            stats["turnover"] = position_changes.mean()

        return stats

    async def _save_to_database(self, strategy_name: str, performance: Dict):
        """Save performance results to database.

        Args:
            strategy_name: Strategy name
            performance: Performance results
        """
        if not self.database:
            return

        try:
            await self.database.save_performance_metrics(
                strategy=strategy_name, metrics=performance, timestamp=get_market_time()
            )
            logger.info(f"Saved performance for {strategy_name} to database")
        except Exception as e:
            logger.error(f"Failed to save performance to database: {e}")

    def _save_to_file(self, strategy_name: str, performance: Dict):
        """Save performance results to file.

        Args:
            strategy_name: Strategy name
            performance: Performance results
        """
        timestamp = get_market_time().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{strategy_name}_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(performance, f, indent=2, default=str)

        logger.info(f"Saved performance for {strategy_name} to {filename}")

    async def load_historical_performance(
        self,
        strategy_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Load historical performance data.

        Args:
            strategy_name: Strategy name
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            List of historical performance records
        """
        if self.database:
            return await self.database.get_performance_history(
                strategy=strategy_name, start_date=start_date, end_date=end_date
            )
        else:
            # Load from files
            pattern = f"{strategy_name}_*.json"
            files = list(self.results_dir.glob(pattern))

            results = []
            for file in files:
                with open(file, "r") as f:
                    data = json.load(f)
                    results.append(data)

            return results


class MLStrategyPerformance:
    """Performance tracking specifically for ML-driven strategies."""

    def __init__(
        self,
        model_selector: Optional[ModelSelector] = None,
        tracker: Optional[StrategyPerformanceTracker] = None,
    ):
        """Initialize ML strategy performance tracker.

        Args:
            model_selector: Model selector for ML strategies
            tracker: General performance tracker
        """
        self.model_selector = model_selector
        self.tracker = tracker or StrategyPerformanceTracker()

    async def track_model_performance(
        self,
        model_name: str,
        predictions: pd.Series,
        actual_returns: pd.Series,
        positions: Optional[pd.DataFrame] = None,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Track ML model performance.

        Args:
            model_name: Name of the ML model
            predictions: Model predictions
            actual_returns: Actual returns
            positions: Position data
            feature_importance: Feature importance scores

        Returns:
            Model performance metrics
        """
        # Align predictions and actual returns first
        common_index = predictions.index.intersection(actual_returns.index)
        predictions_aligned = predictions.loc[common_index]
        actual_returns_aligned = actual_returns.loc[common_index]

        # Calculate prediction accuracy
        if predictions_aligned.dtype == bool or predictions_aligned.dtype == int:
            # Classification
            accuracy = (predictions_aligned == (actual_returns_aligned > 0)).mean()
            precision = (
                ((predictions_aligned == 1) & (actual_returns_aligned > 0)).sum()
                / (predictions_aligned == 1).sum()
                if (predictions_aligned == 1).sum() > 0
                else 0
            )
            recall = (
                ((predictions_aligned == 1) & (actual_returns_aligned > 0)).sum()
                / (actual_returns_aligned > 0).sum()
                if (actual_returns_aligned > 0).sum() > 0
                else 0
            )

            ml_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                ),
            }
        else:
            # Regression
            mse = ((predictions_aligned - actual_returns_aligned) ** 2).mean()
            mae = (predictions_aligned - actual_returns_aligned).abs().mean()
            direction_accuracy = (
                np.sign(predictions_aligned) == np.sign(actual_returns_aligned)
            ).mean()

            ml_metrics = {
                "mse": mse,
                "mae": mae,
                "rmse": np.sqrt(mse),
                "direction_accuracy": direction_accuracy,
            }

        # Calculate strategy returns based on predictions
        if predictions_aligned.dtype == bool or predictions_aligned.dtype == int:
            strategy_returns = actual_returns_aligned * predictions_aligned
        else:
            strategy_returns = actual_returns_aligned * np.sign(predictions_aligned)

        # Track overall performance
        performance = await self.tracker.track_strategy_performance(
            strategy_name=f"ML_{model_name}",
            strategy_type=StrategyType.ML_ENSEMBLE,
            returns=strategy_returns,
            positions=positions,
            metadata={
                "model_type": model_name,
                "ml_metrics": ml_metrics,
                "feature_importance": feature_importance,
                "n_predictions": len(predictions),
            },
        )

        return performance

    def compare_models(self, model_names: List[str], metric: str = "sharpe_ratio") -> pd.DataFrame:
        """Compare multiple ML models.

        Args:
            model_names: List of model names
            metric: Metric to compare

        Returns:
            DataFrame with model comparison
        """
        ml_strategies = [f"ML_{name}" for name in model_names]
        comparison = self.tracker.compare_strategies(ml_strategies)

        # Add ML-specific metrics
        for strategy in ml_strategies:
            if strategy in self.tracker.strategy_results:
                ml_metrics = (
                    self.tracker.strategy_results[strategy]
                    .get("metadata", {})
                    .get("ml_metrics", {})
                )
                for key, value in ml_metrics.items():
                    comparison.loc[comparison["strategy"] == strategy, f"ml_{key}"] = value

        # Only sort if the metric exists
        if metric in comparison.columns:
            return comparison.sort_values(metric, ascending=False)
        else:
            return comparison

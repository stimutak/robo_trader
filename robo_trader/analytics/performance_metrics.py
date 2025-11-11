"""
Comprehensive performance metrics for strategy evaluation.
Standalone module for M4 that works without configuration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Container for all performance metrics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    best_trade: float
    worst_trade: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    recovery_factor: float
    ulcer_index: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Get summary string."""
        return f"""
Performance Summary:
-------------------
Total Return: {self.total_return:.2%}
Annualized Return: {self.annualized_return:.2%}
Volatility: {self.volatility:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.2f}
Total Trades: {self.total_trades}
        """


class StrategyPerformanceAnalyzer:
    """
    Comprehensive strategy performance analyzer.
    Calculates all standard performance metrics.
    """

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.risk_free_rate = risk_free_rate

    def calculate_metrics(
        self,
        returns: Optional[pd.Series] = None,
        equity_curve: Optional[pd.Series] = None,
        trades: Optional[List[Dict]] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Calculate all performance metrics.

        Args:
            returns: Daily returns series
            equity_curve: Equity curve series
            trades: List of trade dictionaries
            benchmark_returns: Benchmark returns for comparison

        Returns:
            Dictionary of performance metrics
        """
        metrics = {}

        # Calculate returns if not provided
        if returns is None and equity_curve is not None:
            returns = equity_curve.pct_change().dropna()

        if returns is not None:
            # Basic return metrics
            metrics.update(self._calculate_return_metrics(returns))

            # Risk metrics
            metrics.update(self._calculate_risk_metrics(returns))

            # Risk-adjusted metrics
            metrics.update(self._calculate_risk_adjusted_metrics(returns))

        if equity_curve is not None:
            # Drawdown metrics
            metrics.update(self._calculate_drawdown_metrics(equity_curve))

        if trades:
            # Trade statistics
            metrics.update(self._calculate_trade_metrics(trades))

        if benchmark_returns is not None and returns is not None:
            # Relative metrics
            metrics.update(self._calculate_relative_metrics(returns, benchmark_returns))

        return metrics

    def _calculate_return_metrics(self, returns: pd.Series) -> Dict:
        """Calculate return-based metrics."""
        total_return = (1 + returns).prod() - 1
        days = len(returns)
        years = days / 252

        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "mean_return": returns.mean(),
            "median_return": returns.median(),
            "best_day": returns.max(),
            "worst_day": returns.min(),
            "positive_days": (returns > 0).sum(),
            "negative_days": (returns < 0).sum(),
            "total_days": days,
        }

    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk metrics."""
        volatility = returns.std() * np.sqrt(252)
        downside_returns = returns[returns < 0]

        if len(downside_returns) > 0:
            downside_deviation = downside_returns.std() * np.sqrt(252)
        else:
            downside_deviation = 0

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns, 5)

        # Conditional Value at Risk
        cvar_95 = returns[returns <= var_95].mean()

        return {
            "volatility": volatility,
            "downside_deviation": downside_deviation,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "tail_ratio": (
                abs(returns.quantile(0.95) / returns.quantile(0.05))
                if returns.quantile(0.05) != 0
                else 0
            ),
        }

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        if volatility > 0:
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility
        else:
            sharpe_ratio = 0

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (
                (mean_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            )
        else:
            sortino_ratio = sharpe_ratio

        # Information Ratio (if we had tracking error)
        information_ratio = sharpe_ratio  # Simplified

        return {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "information_ratio": information_ratio,
        }

    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown metrics."""
        # Calculate drawdown series
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Maximum drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0

        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0

        # Calmar Ratio
        years = len(equity_curve) / 252
        if years > 0 and max_drawdown != 0:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            annualized_return = (1 + total_return) ** (1 / years) - 1
            calmar_ratio = annualized_return / abs(max_drawdown)
        else:
            calmar_ratio = 0

        # Recovery Factor
        if max_drawdown != 0:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            recovery_factor = total_return / abs(max_drawdown)
        else:
            recovery_factor = 0

        # Ulcer Index
        drawdown_squared = drawdown**2
        ulcer_index = np.sqrt(drawdown_squared.mean())

        return {
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": max_duration,
            "calmar_ratio": calmar_ratio,
            "recovery_factor": recovery_factor,
            "ulcer_index": ulcer_index,
            "current_drawdown": drawdown.iloc[-1] if len(drawdown) > 0 else 0,
        }

    def _calculate_trade_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate trade-based metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "average_win": 0,
                "average_loss": 0,
                "best_trade": 0,
                "worst_trade": 0,
            }

        # Extract P&L from trades
        pnls = [t.get("pnl", 0) for t in trades]
        returns = [t.get("return_pct", 0) for t in trades]

        # Win/Loss statistics
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        win_rate = len(winning_trades) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0
        )

        # Average win/loss
        average_win = np.mean(winning_trades) if winning_trades else 0
        average_loss = np.mean(losing_trades) if losing_trades else 0

        # Trade durations
        durations = [t.get("duration_days", 0) for t in trades if "duration_days" in t]
        avg_duration = np.mean(durations) if durations else 0

        # Expectancy
        expectancy = win_rate * average_win + (1 - win_rate) * average_loss

        # Kelly Criterion
        if average_loss != 0:
            win_loss_ratio = abs(average_win / average_loss)
            kelly_percentage = (
                (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
                if win_loss_ratio > 0
                else 0
            )
        else:
            kelly_percentage = win_rate

        return {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0,
            "avg_trade_duration": avg_duration,
            "expectancy": expectancy,
            "kelly_percentage": kelly_percentage,
            "payoff_ratio": abs(average_win / average_loss) if average_loss != 0 else 0,
        }

    def _calculate_relative_metrics(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict:
        """Calculate metrics relative to benchmark."""
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        # Excess returns
        excess_returns = aligned_returns - aligned_benchmark

        # Tracking error
        tracking_error = excess_returns.std() * np.sqrt(252)

        # Beta
        if aligned_benchmark.var() > 0:
            beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var()
        else:
            beta = 0

        # Alpha
        benchmark_annual_return = (1 + aligned_benchmark).prod() ** (
            252 / len(aligned_benchmark)
        ) - 1
        strategy_annual_return = (1 + aligned_returns).prod() ** (252 / len(aligned_returns)) - 1
        alpha = strategy_annual_return - (
            self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)
        )

        # Correlation
        correlation = aligned_returns.corr(aligned_benchmark)

        # Up/Down Capture
        up_periods = aligned_benchmark > 0
        down_periods = aligned_benchmark < 0

        if up_periods.any():
            up_capture = (1 + aligned_returns[up_periods]).prod() / (
                1 + aligned_benchmark[up_periods]
            ).prod()
        else:
            up_capture = 0

        if down_periods.any():
            down_capture = (1 + aligned_returns[down_periods]).prod() / (
                1 + aligned_benchmark[down_periods]
            ).prod()
        else:
            down_capture = 0

        return {
            "tracking_error": tracking_error,
            "beta": beta,
            "alpha": alpha,
            "correlation": correlation,
            "up_capture": up_capture,
            "down_capture": down_capture,
            "capture_ratio": up_capture / down_capture if down_capture != 0 else 0,
        }

    def create_performance_report(self, metrics: Dict) -> str:
        """Create a formatted performance report."""
        report = """
================================================================================
                           STRATEGY PERFORMANCE REPORT
================================================================================

RETURNS
-------
Total Return:           {total_return:>10.2%}
Annualized Return:      {annualized_return:>10.2%}
Volatility:            {volatility:>10.2%}

RISK-ADJUSTED METRICS
--------------------
Sharpe Ratio:          {sharpe_ratio:>10.2f}
Sortino Ratio:         {sortino_ratio:>10.2f}
Calmar Ratio:          {calmar_ratio:>10.2f}

DRAWDOWN
--------
Max Drawdown:          {max_drawdown:>10.2%}
Max DD Duration:       {max_drawdown_duration:>10.0f} days
Current Drawdown:      {current_drawdown:>10.2%}

TRADE STATISTICS
----------------
Total Trades:          {total_trades:>10.0f}
Win Rate:              {win_rate:>10.2%}
Profit Factor:         {profit_factor:>10.2f}
Average Win:           ${average_win:>10.2f}
Average Loss:          ${average_loss:>10.2f}
Best Trade:            ${best_trade:>10.2f}
Worst Trade:           ${worst_trade:>10.2f}

RISK METRICS
------------
VaR (95%):             {var_95:>10.2%}
CVaR (95%):            {cvar_95:>10.2%}
Skewness:              {skewness:>10.2f}
Kurtosis:              {kurtosis:>10.2f}

================================================================================
        """.format(
            **{k: metrics.get(k, 0) for k in metrics}
        )

        return report


class RiskMetrics:
    """Risk metrics calculator."""

    @staticmethod
    def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk."""
        var = RiskMetrics.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()

    @staticmethod
    def calculate_rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        return rolling_mean / rolling_std

    @staticmethod
    def calculate_rolling_drawdown(equity_curve: pd.Series) -> pd.Series:
        """Calculate rolling drawdown series."""
        running_max = equity_curve.expanding().max()
        return (equity_curve - running_max) / running_max

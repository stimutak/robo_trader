"""
Performance metrics calculation for backtesting.

This module provides comprehensive performance analytics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Drawdown analysis
- Trade statistics
- Win/loss metrics
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..logger import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Returns
    total_return: float = 0.0
    annualized_return: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    downside_deviation: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Trade counts
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Time metrics
    average_bars_in_trade: float = 0.0
    win_loss_ratio: float = 0.0

    # Statistical measures
    skewness: float = 0.0
    kurtosis: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0

    # Recovery metrics
    recovery_factor: float = 0.0
    payoff_ratio: float = 0.0


def calculate_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    initial_capital: float,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Calculate comprehensive performance metrics.

    Args:
        returns: Daily returns series
        equity_curve: Equity curve series
        trades: DataFrame of completed trades
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate for Sharpe
        periods_per_year: Trading periods per year

    Returns:
        Performance metrics object
    """
    metrics = PerformanceMetrics()

    if returns.empty:
        return metrics

    # Basic return metrics
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    metrics.total_return = total_return

    # Annualized return
    years = len(returns) / periods_per_year
    if years > 0:
        metrics.annualized_return = (1 + total_return) ** (1 / years) - 1

    # Volatility
    metrics.volatility = returns.std() * np.sqrt(periods_per_year)

    # Downside deviation (for Sortino)
    downside_returns = returns[returns < 0]
    if not downside_returns.empty:
        metrics.downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)

    # Drawdown analysis
    drawdown_series = calculate_drawdown_series(equity_curve)
    if not drawdown_series.empty:
        metrics.max_drawdown = drawdown_series.min()
        metrics.max_drawdown_duration = calculate_max_drawdown_duration(drawdown_series)

    # Sharpe ratio
    if metrics.volatility > 0:
        excess_return = metrics.annualized_return - risk_free_rate
        metrics.sharpe_ratio = excess_return / metrics.volatility

    # Sortino ratio
    if metrics.downside_deviation > 0:
        excess_return = metrics.annualized_return - risk_free_rate
        metrics.sortino_ratio = excess_return / metrics.downside_deviation

    # Calmar ratio
    if abs(metrics.max_drawdown) > 0:
        metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)

    # Trade statistics
    if not trades.empty:
        metrics.total_trades = len(trades)

        # Win/loss analysis
        winning = trades[trades["pnl"] > 0]
        losing = trades[trades["pnl"] < 0]

        metrics.winning_trades = len(winning)
        metrics.losing_trades = len(losing)

        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Average win/loss
        if not winning.empty:
            metrics.average_win = winning["pnl"].mean()
            metrics.largest_win = winning["pnl"].max()

        if not losing.empty:
            metrics.average_loss = abs(losing["pnl"].mean())
            metrics.largest_loss = abs(losing["pnl"].min())

        # Profit factor
        if not losing.empty and losing["pnl"].sum() != 0:
            gross_profit = winning["pnl"].sum() if not winning.empty else 0
            gross_loss = abs(losing["pnl"].sum())
            metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Expectancy
        if metrics.total_trades > 0:
            metrics.expectancy = trades["pnl"].sum() / metrics.total_trades

        # Win/loss ratio
        if metrics.average_loss > 0:
            metrics.win_loss_ratio = metrics.average_win / metrics.average_loss

        # Payoff ratio
        if metrics.losing_trades > 0:
            metrics.payoff_ratio = (metrics.average_win * metrics.winning_trades) / (
                metrics.average_loss * metrics.losing_trades
            )

        # Average time in trade
        if "bars_held" in trades.columns:
            metrics.average_bars_in_trade = trades["bars_held"].mean()

        # Consecutive wins/losses
        metrics.max_consecutive_wins = calculate_max_consecutive(trades["pnl"] > 0)
        metrics.max_consecutive_losses = calculate_max_consecutive(trades["pnl"] < 0)

    # Statistical measures
    if len(returns) > 3:
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)

    # Value at Risk (95%)
    if len(returns) > 20:
        metrics.var_95 = np.percentile(returns, 5)
        # Conditional VaR (Expected Shortfall)
        metrics.cvar_95 = returns[returns <= metrics.var_95].mean()

    # Recovery factor
    if metrics.max_drawdown < 0 and metrics.total_return > 0:
        metrics.recovery_factor = metrics.total_return / abs(metrics.max_drawdown)

    return metrics


def calculate_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series from equity curve.

    Args:
        equity_curve: Cumulative equity values

    Returns:
        Drawdown series (negative values)
    """
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown


def calculate_max_drawdown_duration(drawdown_series: pd.Series) -> int:
    """
    Calculate maximum drawdown duration in periods.

    Args:
        drawdown_series: Series of drawdown values

    Returns:
        Maximum duration of drawdown
    """
    if drawdown_series.empty:
        return 0

    # Find periods where we're in drawdown
    in_drawdown = drawdown_series < 0

    # Calculate consecutive periods in drawdown
    max_duration = 0
    current_duration = 0

    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    return max_duration


def calculate_max_consecutive(series: pd.Series) -> int:
    """
    Calculate maximum consecutive True values in boolean series.

    Args:
        series: Boolean series

    Returns:
        Maximum consecutive True values
    """
    if series.empty:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for value in series:
        if value:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def calculate_rolling_metrics(
    returns: pd.Series, window: int = 252, min_periods: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        returns: Returns series
        window: Rolling window size
        min_periods: Minimum periods required

    Returns:
        DataFrame with rolling metrics
    """
    rolling_metrics = pd.DataFrame(index=returns.index)

    # Rolling returns
    rolling_metrics["return"] = returns.rolling(window, min_periods=min_periods).sum()

    # Rolling volatility
    rolling_metrics["volatility"] = returns.rolling(
        window, min_periods=min_periods
    ).std() * np.sqrt(252)

    # Rolling Sharpe
    rolling_metrics["sharpe"] = (rolling_metrics["return"] - 0.02) / rolling_metrics["volatility"]

    # Rolling max drawdown
    cumsum = returns.cumsum()
    rolling_max = cumsum.rolling(window, min_periods=min_periods).max()
    rolling_metrics["drawdown"] = (cumsum - rolling_max) / rolling_max

    return rolling_metrics


def calculate_monthly_returns(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly return statistics.

    Args:
        returns: Daily returns series

    Returns:
        DataFrame with monthly statistics
    """
    monthly = returns.resample("M").agg(
        [
            ("return", "sum"),
            ("volatility", "std"),
            ("trades", "count"),
            ("best_day", "max"),
            ("worst_day", "min"),
        ]
    )

    monthly["volatility"] = monthly["volatility"] * np.sqrt(21)  # Monthly vol

    return monthly


def create_performance_report(metrics: PerformanceMetrics, trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Create detailed performance report.

    Args:
        metrics: Performance metrics
        trades: Trade history

    Returns:
        Performance report dictionary
    """
    report = {
        "summary": {
            "total_return": f"{metrics.total_return:.2%}",
            "annualized_return": f"{metrics.annualized_return:.2%}",
            "sharpe_ratio": f"{metrics.sharpe_ratio:.2f}",
            "max_drawdown": f"{metrics.max_drawdown:.2%}",
            "win_rate": f"{metrics.win_rate:.2%}",
        },
        "risk_metrics": {
            "volatility": f"{metrics.volatility:.2%}",
            "downside_deviation": f"{metrics.downside_deviation:.2%}",
            "sortino_ratio": f"{metrics.sortino_ratio:.2f}",
            "calmar_ratio": f"{metrics.calmar_ratio:.2f}",
            "var_95": f"{metrics.var_95:.2%}",
            "cvar_95": f"{metrics.cvar_95:.2%}",
        },
        "trade_statistics": {
            "total_trades": metrics.total_trades,
            "winning_trades": metrics.winning_trades,
            "losing_trades": metrics.losing_trades,
            "average_win": f"${metrics.average_win:.2f}",
            "average_loss": f"${metrics.average_loss:.2f}",
            "win_loss_ratio": f"{metrics.win_loss_ratio:.2f}",
            "profit_factor": f"{metrics.profit_factor:.2f}",
            "expectancy": f"${metrics.expectancy:.2f}",
        },
        "extremes": {
            "largest_win": f"${metrics.largest_win:.2f}",
            "largest_loss": f"${metrics.largest_loss:.2f}",
            "max_consecutive_wins": metrics.max_consecutive_wins,
            "max_consecutive_losses": metrics.max_consecutive_losses,
            "max_drawdown_duration": f"{metrics.max_drawdown_duration} periods",
        },
    }

    # Add trade distribution if available
    if not trades.empty and "pnl" in trades.columns:
        report["distribution"] = {
            "mean_pnl": f"${trades['pnl'].mean():.2f}",
            "median_pnl": f"${trades['pnl'].median():.2f}",
            "std_pnl": f"${trades['pnl'].std():.2f}",
            "skewness": f"{stats.skew(trades['pnl']):.2f}",
            "kurtosis": f"{stats.kurtosis(trades['pnl']):.2f}",
        }

    return report

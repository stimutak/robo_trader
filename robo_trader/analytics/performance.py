"""Performance analytics for RoboTrader.

This module provides comprehensive performance metrics including Sharpe ratio,
maximum drawdown, risk-adjusted returns, and advanced analytics.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import structlog
from scipy import stats

logger = structlog.get_logger(__name__)


class PerformanceAnalyzer:
    """Comprehensive performance analytics for trading strategies."""

    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize performance analyzer.

        Args:
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = risk_free_rate / 252  # Daily risk-free rate

    def calculate_returns(self, prices: pd.Series, method: str = "simple") -> pd.Series:
        """Calculate returns from price series.

        Args:
            prices: Price time series
            method: "simple" or "log" returns

        Returns:
            Returns series
        """
        if method == "simple":
            returns = prices.pct_change().dropna()
        elif method == "log":
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            raise ValueError(f"Unknown method: {method}")

        return returns

    def sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[float] = None, periods: int = 252
    ) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annual)
            periods: Periods per year for annualization

        Returns:
            Sharpe ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - (risk_free_rate / periods)

        std_dev = excess_returns.std()
        if std_dev < 1e-10:  # Use small tolerance for near-zero volatility
            return 0.0

        return np.sqrt(periods) * excess_returns.mean() / std_dev

    def sortino_ratio(
        self, returns: pd.Series, risk_free_rate: Optional[float] = None, periods: int = 252
    ) -> float:
        """Calculate Sortino ratio (downside deviation).

        Args:
            returns: Returns series
            risk_free_rate: Risk-free rate (annual)
            periods: Periods per year for annualization

        Returns:
            Sortino ratio
        """
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        excess_returns = returns - (risk_free_rate / periods)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float("inf") if excess_returns.mean() > 0 else 0.0

        downside_deviation = np.sqrt(periods) * downside_returns.std()

        return np.sqrt(periods) * excess_returns.mean() / downside_deviation

    def calmar_ratio(self, returns: pd.Series, periods: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown).

        Args:
            returns: Returns series
            periods: Periods per year for annualization

        Returns:
            Calmar ratio
        """
        annual_return = returns.mean() * periods
        max_dd = self.maximum_drawdown(returns)[0]

        if max_dd == 0:
            return float("inf") if annual_return > 0 else 0.0

        return annual_return / abs(max_dd)

    def maximum_drawdown(self, returns: pd.Series) -> Tuple[float, datetime, datetime, int]:
        """Calculate maximum drawdown and related metrics.

        Args:
            returns: Returns series

        Returns:
            Tuple of (max_drawdown, start_date, end_date, duration_days)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_end = drawdown.idxmin()

        # Find start of max drawdown
        max_dd_start = cumulative.loc[:max_dd_end].idxmax()

        # Calculate duration
        duration = (
            (max_dd_end - max_dd_start).days
            if hasattr(max_dd_end, "date")
            else len(returns.loc[max_dd_start:max_dd_end])
        )

        return max_dd, max_dd_start, max_dd_end, duration

    def value_at_risk(
        self, returns: pd.Series, confidence: float = 0.05, method: str = "historical"
    ) -> float:
        """Calculate Value at Risk (VaR).

        Args:
            returns: Returns series
            confidence: Confidence level (e.g., 0.05 for 5%)
            method: "historical", "parametric", or "cornish_fisher"

        Returns:
            VaR value
        """
        if method == "historical":
            return returns.quantile(confidence)

        elif method == "parametric":
            return stats.norm.ppf(confidence, returns.mean(), returns.std())

        elif method == "cornish_fisher":
            # Cornish-Fisher expansion for non-normal distributions
            skew = returns.skew()
            kurt = returns.kurtosis()

            z = stats.norm.ppf(confidence)
            cf_z = (
                z
                + (z**2 - 1) * skew / 6
                + (z**3 - 3 * z) * kurt / 24
                - (2 * z**3 - 5 * z) * (skew**2) / 36
            )

            return returns.mean() + cf_z * returns.std()

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def conditional_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Returns series
            confidence: Confidence level

        Returns:
            CVaR value
        """
        var = self.value_at_risk(returns, confidence)
        return returns[returns <= var].mean()

    def beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate beta relative to benchmark.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Beta coefficient
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        if len(aligned_returns) < 2:
            return 0.0

        covariance = aligned_returns.cov(aligned_benchmark)
        benchmark_variance = aligned_benchmark.var()

        if benchmark_variance == 0:
            return 0.0

        return covariance / benchmark_variance

    def alpha(self, returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252) -> float:
        """Calculate alpha (Jensen's alpha).

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods: Periods per year

        Returns:
            Alpha value (annualized)
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        if len(aligned_returns) < 2:
            return 0.0

        strategy_return = aligned_returns.mean() * periods
        benchmark_return = aligned_benchmark.mean() * periods
        beta_coeff = self.beta(aligned_returns, aligned_benchmark)

        return strategy_return - (
            self.risk_free_rate + beta_coeff * (benchmark_return - self.risk_free_rate)
        )

    def information_ratio(
        self, returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252
    ) -> float:
        """Calculate information ratio.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods: Periods per year

        Returns:
            Information ratio
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        if len(aligned_returns) < 2:
            return 0.0

        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std() * np.sqrt(periods)

        if tracking_error == 0:
            return 0.0

        return (active_returns.mean() * periods) / tracking_error

    def treynor_ratio(
        self, returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252
    ) -> float:
        """Calculate Treynor ratio.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            periods: Periods per year

        Returns:
            Treynor ratio
        """
        beta_coeff = self.beta(returns, benchmark_returns)

        if beta_coeff == 0:
            return 0.0

        excess_return = returns.mean() * periods - self.risk_free_rate
        return excess_return / beta_coeff

    def omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio.

        Args:
            returns: Returns series
            threshold: Threshold return level

        Returns:
            Omega ratio
        """
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns <= threshold]

        gains = returns_above.sum() if len(returns_above) > 0 else 0
        losses = returns_below.sum() if len(returns_below) > 0 else 0

        if losses == 0:
            return float("inf") if gains > 0 else 1.0

        return gains / losses

    def capture_ratios(
        self, returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """Calculate up and down capture ratios.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            Tuple of (up_capture, down_capture)
        """
        # Align series
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join="inner")

        if len(aligned_returns) < 2:
            return 0.0, 0.0

        # Up periods
        up_periods = aligned_benchmark > 0
        if up_periods.sum() > 0:
            up_capture = aligned_returns[up_periods].mean() / aligned_benchmark[up_periods].mean()
        else:
            up_capture = 0.0

        # Down periods
        down_periods = aligned_benchmark < 0
        if down_periods.sum() > 0:
            down_capture = (
                aligned_returns[down_periods].mean() / aligned_benchmark[down_periods].mean()
            )
        else:
            down_capture = 0.0

        return up_capture, down_capture

    def rolling_metrics(
        self, returns: pd.Series, window: int = 60, metrics: List[str] = None
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics.

        Args:
            returns: Returns series
            window: Rolling window size
            metrics: List of metrics to calculate

        Returns:
            DataFrame with rolling metrics
        """
        if metrics is None:
            metrics = ["sharpe", "volatility", "max_dd"]

        rolling_data = {}

        for metric in metrics:
            if metric == "sharpe":
                rolling_sharpe = []
                for i in range(len(returns) - window + 1):
                    subset = returns.iloc[i : i + window]
                    rolling_sharpe.append(self.sharpe_ratio(subset))
                rolling_data["rolling_sharpe"] = pd.Series(
                    rolling_sharpe, index=returns.index[window - 1 :]
                )

            elif metric == "volatility":
                rolling_data["rolling_volatility"] = returns.rolling(window).std() * np.sqrt(252)

            elif metric == "max_dd":
                rolling_dd = []
                for i in range(len(returns) - window + 1):
                    subset = returns.iloc[i : i + window]
                    rolling_dd.append(self.maximum_drawdown(subset)[0])
                rolling_data["rolling_max_dd"] = pd.Series(
                    rolling_dd, index=returns.index[window - 1 :]
                )

        return pd.DataFrame(rolling_data)

    def performance_summary(
        self, returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, periods: int = 252
    ) -> Dict:
        """Generate comprehensive performance summary.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            periods: Periods per year

        Returns:
            Dictionary with performance metrics
        """
        if returns.empty:
            return {}

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = returns.mean() * periods
        annual_volatility = returns.std() * np.sqrt(periods)

        # Risk metrics
        max_dd, dd_start, dd_end, dd_duration = self.maximum_drawdown(returns)
        var_95 = self.value_at_risk(returns, 0.05)
        cvar_95 = self.conditional_var(returns, 0.05)

        # Risk-adjusted metrics
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        calmar = self.calmar_ratio(returns)

        # Win/Loss statistics
        winning_periods = (returns > 0).sum()
        losing_periods = (returns < 0).sum()
        win_rate = winning_periods / len(returns) if len(returns) > 0 else 0

        avg_win = returns[returns > 0].mean() if winning_periods > 0 else 0
        avg_loss = returns[returns < 0].mean() if losing_periods > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        # Consistency metrics
        positive_months = (returns.resample("ME").sum() > 0).sum() if len(returns) > 30 else 0
        total_months = len(returns.resample("ME").sum()) if len(returns) > 30 else 1

        summary = {
            # Returns
            "total_return": round(total_return, 4),
            "annual_return": round(annual_return, 4),
            "annual_volatility": round(annual_volatility, 4),
            # Risk metrics
            "max_drawdown": round(max_dd, 4),
            "max_drawdown_duration": dd_duration,
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            # Risk-adjusted metrics
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            # Win/Loss statistics
            "win_rate": round(win_rate, 3),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "win_loss_ratio": round(win_loss_ratio, 2),
            # Consistency
            "positive_months": positive_months,
            "total_months": total_months,
            "monthly_win_rate": round(positive_months / total_months, 3) if total_months > 0 else 0,
            # Statistics
            "skewness": round(returns.skew(), 3),
            "kurtosis": round(returns.kurtosis(), 3),
            "n_observations": len(returns),
        }

        # Benchmark-relative metrics
        if benchmark_returns is not None:
            summary.update(
                {
                    "alpha": round(self.alpha(returns, benchmark_returns), 4),
                    "beta": round(self.beta(returns, benchmark_returns), 3),
                    "information_ratio": round(
                        self.information_ratio(returns, benchmark_returns), 3
                    ),
                    "treynor_ratio": round(self.treynor_ratio(returns, benchmark_returns), 4),
                }
            )

            up_capture, down_capture = self.capture_ratios(returns, benchmark_returns)
            summary.update(
                {"up_capture": round(up_capture, 3), "down_capture": round(down_capture, 3)}
            )

        return summary

    def monthly_returns_table(self, returns: pd.Series) -> pd.DataFrame:
        """Generate monthly returns table.

        Args:
            returns: Returns series

        Returns:
            DataFrame with monthly returns by year
        """
        if returns.empty:
            return pd.DataFrame()

        # Resample to monthly returns
        monthly_returns = returns.resample("ME").sum()

        # Create pivot table
        monthly_returns.index = pd.to_datetime(monthly_returns.index)

        table = (
            monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month])
            .sum()
            .unstack()
        )

        # Add column names
        table.columns = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(table.columns)]

        # Add annual returns
        annual_returns = returns.resample("YE").sum()
        annual_returns.index = annual_returns.index.year
        table["Annual"] = annual_returns.reindex(table.index, fill_value=0)

        return table.round(4)

    def drawdown_analysis(self, returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
        """Analyze top drawdown periods.

        Args:
            returns: Returns series
            top_n: Number of top drawdowns to analyze

        Returns:
            DataFrame with drawdown analysis
        """
        if returns.empty:
            return pd.DataFrame()

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None

        for date, dd in drawdown.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
                max_dd = dd
            elif dd < 0 and in_drawdown:
                # Continue drawdown
                if dd < max_dd:
                    max_dd = dd
            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                duration = (
                    (date - start_date).days
                    if hasattr(date, "date")
                    else len(returns.loc[start_date:date])
                )

                drawdown_periods.append(
                    {
                        "start": start_date,
                        "end": date,
                        "max_drawdown": max_dd,
                        "duration_days": duration,
                        "recovery_date": date,
                    }
                )

        # Handle ongoing drawdown
        if in_drawdown:
            duration = (
                (returns.index[-1] - start_date).days
                if hasattr(returns.index[-1], "date")
                else len(returns.loc[start_date:])
            )

            drawdown_periods.append(
                {
                    "start": start_date,
                    "end": returns.index[-1],
                    "max_drawdown": max_dd,
                    "duration_days": duration,
                    "recovery_date": None,
                }
            )

        if not drawdown_periods:
            return pd.DataFrame()

        # Convert to DataFrame and sort by magnitude
        dd_df = pd.DataFrame(drawdown_periods)
        dd_df = dd_df.sort_values("max_drawdown").head(top_n)

        return dd_df.round(4)

    def risk_return_scatter(
        self,
        returns_dict: Dict[str, pd.Series],
        benchmark_returns: Optional[pd.Series] = None,
        periods: int = 252,
    ) -> pd.DataFrame:
        """Create risk-return scatter data.

        Args:
            returns_dict: Dictionary of strategy returns
            benchmark_returns: Optional benchmark returns
            periods: Periods per year

        Returns:
            DataFrame with risk-return metrics
        """
        results = []

        for name, returns in returns_dict.items():
            if returns.empty:
                continue

            annual_return = returns.mean() * periods
            annual_volatility = returns.std() * np.sqrt(periods)
            sharpe = self.sharpe_ratio(returns)
            max_dd = self.maximum_drawdown(returns)[0]

            result = {
                "strategy": name,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
            }

            if benchmark_returns is not None:
                result["alpha"] = self.alpha(returns, benchmark_returns)
                result["beta"] = self.beta(returns, benchmark_returns)

            results.append(result)

        return pd.DataFrame(results).round(4)

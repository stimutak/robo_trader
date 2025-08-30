"""Risk metrics and analysis for RoboTrader.

This module provides advanced risk analytics including portfolio risk metrics,
stress testing, and risk decomposition.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog
from scipy import stats
from scipy.optimize import minimize

logger = structlog.get_logger(__name__)


class RiskAnalyzer:
    """Advanced risk analysis for trading strategies and portfolios."""

    def __init__(self, confidence_levels: List[float] = None):
        """Initialize risk analyzer.

        Args:
            confidence_levels: List of confidence levels for VaR calculations
        """
        self.confidence_levels = confidence_levels or [0.01, 0.05, 0.10]

    def portfolio_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: float = 0.05,
        method: str = "historical",
    ) -> float:
        """Calculate portfolio Value at Risk.

        Args:
            returns: Returns matrix (assets as columns)
            weights: Portfolio weights
            confidence: Confidence level
            method: "historical", "parametric", or "monte_carlo"

        Returns:
            Portfolio VaR
        """
        portfolio_returns = (returns * weights).sum(axis=1)

        if method == "historical":
            return portfolio_returns.quantile(confidence)

        elif method == "parametric":
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            return stats.norm.ppf(confidence, portfolio_mean, portfolio_std)

        elif method == "monte_carlo":
            return self._monte_carlo_var(returns, weights, confidence)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def _monte_carlo_var(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence: float,
        n_simulations: int = 10000,
    ) -> float:
        """Calculate VaR using Monte Carlo simulation.

        Args:
            returns: Returns matrix
            weights: Portfolio weights
            confidence: Confidence level
            n_simulations: Number of simulations

        Returns:
            Monte Carlo VaR
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov()
        mean_returns = returns.mean()

        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_simulations)

        # Calculate portfolio returns for each scenario
        portfolio_scenarios = (random_returns * weights).sum(axis=1)

        return np.percentile(portfolio_scenarios, confidence * 100)

    def component_var(
        self, returns: pd.DataFrame, weights: np.ndarray, confidence: float = 0.05
    ) -> pd.Series:
        """Calculate component VaR for each asset.

        Args:
            returns: Returns matrix
            weights: Portfolio weights
            confidence: Confidence level

        Returns:
            Component VaR for each asset
        """
        portfolio_var = self.portfolio_var(returns, weights, confidence)

        # Calculate marginal VaR for each asset
        marginal_vars = []

        for i in range(len(weights)):
            # Perturb weight slightly
            perturbed_weights = weights.copy()
            epsilon = 0.001
            perturbed_weights[i] += epsilon
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # Renormalize

            perturbed_var = self.portfolio_var(returns, perturbed_weights, confidence)
            marginal_var = (perturbed_var - portfolio_var) / epsilon
            marginal_vars.append(marginal_var)

        # Component VaR = Weight * Marginal VaR
        component_vars = weights * np.array(marginal_vars)

        return pd.Series(component_vars, index=returns.columns)

    def expected_shortfall(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR).

        Args:
            returns: Returns series
            confidence: Confidence level

        Returns:
            Expected Shortfall
        """
        var = returns.quantile(confidence)
        return returns[returns <= var].mean()

    def tail_ratio(self, returns: pd.Series, tail_percentile: float = 0.05) -> float:
        """Calculate tail ratio (right tail / left tail).

        Args:
            returns: Returns series
            tail_percentile: Percentile for tail definition

        Returns:
            Tail ratio
        """
        right_tail = returns.quantile(1 - tail_percentile)
        left_tail = returns.quantile(tail_percentile)

        right_tail_mean = returns[returns >= right_tail].mean()
        left_tail_mean = returns[returns <= left_tail].mean()

        if left_tail_mean == 0:
            return float("inf") if right_tail_mean > 0 else 1.0

        return abs(right_tail_mean / left_tail_mean)

    def risk_parity_weights(
        self, returns: pd.DataFrame, target_risk_contrib: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate risk parity portfolio weights.

        Args:
            returns: Returns matrix
            target_risk_contrib: Target risk contributions (equal if None)

        Returns:
            Risk parity weights
        """
        n_assets = len(returns.columns)

        if target_risk_contrib is None:
            target_risk_contrib = np.ones(n_assets) / n_assets

        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets

        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        ]
        bounds = [(0.01, 0.99) for _ in range(n_assets)]  # Allow small bounds

        def objective(weights):
            return self._risk_parity_objective(weights, returns.cov(), target_risk_contrib)

        result = minimize(
            objective, initial_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        if result.success:
            return result.x
        else:
            logger.warning("Risk parity optimization failed, returning equal weights")
            return initial_weights

    def _risk_parity_objective(
        self, weights: np.ndarray, cov_matrix: pd.DataFrame, target_risk_contrib: np.ndarray
    ) -> float:
        """Objective function for risk parity optimization.

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            target_risk_contrib: Target risk contributions

        Returns:
            Objective value
        """
        # Calculate risk contributions
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol

        # Normalize risk contributions
        risk_contrib = risk_contrib / np.sum(risk_contrib)

        # Minimize squared differences from target
        return np.sum((risk_contrib - target_risk_contrib) ** 2)

    def stress_test(
        self, returns: pd.DataFrame, weights: np.ndarray, scenarios: List[Dict]
    ) -> pd.DataFrame:
        """Perform stress testing on portfolio.

        Args:
            returns: Returns matrix
            weights: Portfolio weights
            scenarios: List of stress scenarios
                      e.g., [{"name": "2008 Crisis", "shocks": {"AAPL": -0.5, "SPY": -0.4}}]

        Returns:
            DataFrame with stress test results
        """
        results = []

        # Baseline portfolio statistics
        baseline_returns = (returns * weights).sum(axis=1)
        baseline_var = self.portfolio_var(returns, weights, 0.05)
        baseline_vol = baseline_returns.std() * np.sqrt(252)

        results.append(
            {
                "scenario": "Baseline",
                "portfolio_return": baseline_returns.mean() * 252,
                "portfolio_vol": baseline_vol,
                "var_95": baseline_var,
                "expected_shortfall": self.expected_shortfall(baseline_returns),
                "max_drawdown": self._calculate_max_drawdown(baseline_returns),
            }
        )

        # Stress scenarios
        for scenario in scenarios:
            stressed_returns = returns.copy()

            # Apply shocks
            for asset, shock in scenario.get("shocks", {}).items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] = stressed_returns[asset] + shock

            # Calculate stressed portfolio metrics
            stressed_portfolio_returns = (stressed_returns * weights).sum(axis=1)
            stressed_var = stressed_portfolio_returns.quantile(0.05)
            stressed_vol = stressed_portfolio_returns.std() * np.sqrt(252)

            results.append(
                {
                    "scenario": scenario["name"],
                    "portfolio_return": stressed_portfolio_returns.mean() * 252,
                    "portfolio_vol": stressed_vol,
                    "var_95": stressed_var,
                    "expected_shortfall": self.expected_shortfall(stressed_portfolio_returns),
                    "max_drawdown": self._calculate_max_drawdown(stressed_portfolio_returns),
                }
            )

        return pd.DataFrame(results)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def correlation_risk_analysis(self, returns: pd.DataFrame, threshold: float = 0.7) -> Dict:
        """Analyze correlation risk in portfolio.

        Args:
            returns: Returns matrix
            threshold: High correlation threshold

        Returns:
            Dictionary with correlation risk metrics
        """
        corr_matrix = returns.corr()

        # Find high correlation pairs
        high_corr_pairs = []
        for i, asset1 in enumerate(corr_matrix.columns):
            for j, asset2 in enumerate(corr_matrix.columns[i + 1 :], i + 1):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_corr_pairs.append(
                        {"asset1": asset1, "asset2": asset2, "correlation": corr}
                    )

        # Calculate portfolio concentration
        eigenvals = np.linalg.eigvals(corr_matrix.values)
        effective_assets = (np.sum(eigenvals) ** 2) / np.sum(eigenvals**2)
        concentration_ratio = len(returns.columns) / effective_assets

        # Average correlation
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        avg_correlation = corr_matrix.values[mask].mean()

        return {
            "high_correlation_pairs": high_corr_pairs,
            "n_high_corr_pairs": len(high_corr_pairs),
            "effective_assets": effective_assets,
            "concentration_ratio": concentration_ratio,
            "average_correlation": avg_correlation,
            "max_correlation": corr_matrix.values[mask].max(),
            "min_correlation": corr_matrix.values[mask].min(),
        }

    def risk_attribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict:
        """Perform risk attribution analysis.

        Args:
            returns: Returns matrix
            weights: Portfolio weights
            benchmark_returns: Optional benchmark returns

        Returns:
            Dictionary with risk attribution metrics
        """
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)

        # Calculate individual asset contributions to risk
        cov_matrix = returns.cov() * 252  # Annualized
        portfolio_variance = weights.T @ cov_matrix @ weights

        # Marginal contributions to risk
        marginal_contrib = (cov_matrix @ weights) / np.sqrt(portfolio_variance)

        # Component contributions (absolute)
        component_contrib = weights * marginal_contrib

        # Percentage contributions
        pct_contrib = component_contrib / component_contrib.sum()

        result = {
            "portfolio_volatility": portfolio_vol,
            "component_contributions": pd.Series(component_contrib, index=returns.columns),
            "percentage_contributions": pd.Series(pct_contrib, index=returns.columns),
            "marginal_contributions": pd.Series(marginal_contrib, index=returns.columns),
        }

        # Active risk vs benchmark
        if benchmark_returns is not None:
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)

            result.update(
                {
                    "tracking_error": tracking_error,
                    "information_ratio": (
                        (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
                    ),
                }
            )

        return result

    def factor_risk_model(
        self, returns: pd.DataFrame, factor_returns: pd.DataFrame, weights: np.ndarray
    ) -> Dict:
        """Simple factor risk model decomposition.

        Args:
            returns: Asset returns matrix
            factor_returns: Factor returns matrix
            weights: Portfolio weights

        Returns:
            Dictionary with factor risk decomposition
        """
        # Calculate factor loadings (betas) for each asset
        loadings = {}
        specific_risks = {}

        for asset in returns.columns:
            asset_returns = returns[asset].dropna()

            # Align with factor returns
            aligned_factors = factor_returns.reindex(asset_returns.index).dropna()
            aligned_asset = asset_returns.reindex(aligned_factors.index)

            if len(aligned_asset) < 20:  # Need minimum observations
                continue

            # Multiple regression: asset returns = alpha + beta * factors + error
            X = aligned_factors.values
            y = aligned_asset.values

            # Add constant term
            X = np.column_stack([np.ones(len(X)), X])

            try:
                # Ordinary least squares
                beta = np.linalg.lstsq(X, y, rcond=None)[0]

                # Predictions and residuals
                y_pred = X @ beta
                residuals = y - y_pred

                loadings[asset] = beta[1:]  # Exclude constant
                specific_risks[asset] = np.std(residuals) * np.sqrt(252)

            except np.linalg.LinAlgError:
                logger.warning(f"Failed to calculate factor loadings for {asset}")

        if not loadings:
            return {}

        # Portfolio factor exposures
        portfolio_loadings = np.zeros(len(factor_returns.columns))
        for i, asset in enumerate(returns.columns):
            if asset in loadings:
                portfolio_loadings += weights[i] * loadings[asset]

        # Factor covariance matrix
        factor_cov = factor_returns.cov() * 252

        # Portfolio factor risk
        factor_risk = np.sqrt(portfolio_loadings.T @ factor_cov @ portfolio_loadings)

        # Portfolio specific risk
        specific_risk = 0
        for i, asset in enumerate(returns.columns):
            if asset in specific_risks:
                specific_risk += (weights[i] ** 2) * (specific_risks[asset] ** 2)
        specific_risk = np.sqrt(specific_risk)

        return {
            "factor_loadings": pd.DataFrame(loadings).T,
            "specific_risks": pd.Series(specific_risks),
            "portfolio_factor_exposures": pd.Series(
                portfolio_loadings, index=factor_returns.columns
            ),
            "portfolio_factor_risk": factor_risk,
            "portfolio_specific_risk": specific_risk,
            "total_risk": np.sqrt(factor_risk**2 + specific_risk**2),
        }

    def tail_risk_measures(
        self, returns: pd.Series, confidence_levels: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """Calculate comprehensive tail risk measures.

        Args:
            returns: Returns series
            confidence_levels: List of confidence levels

        Returns:
            DataFrame with tail risk measures
        """
        if confidence_levels is None:
            confidence_levels = self.confidence_levels

        results = []

        for confidence in confidence_levels:
            var = returns.quantile(confidence)
            cvar = self.expected_shortfall(returns, confidence)

            results.append(
                {
                    "confidence_level": confidence,
                    "var": var,
                    "expected_shortfall": cvar,
                    "tail_expectation": cvar - var,
                    "tail_ratio": self.tail_ratio(returns, confidence),
                }
            )

        return pd.DataFrame(results)

    def liquidity_risk_metrics(
        self, returns: pd.DataFrame, volumes: pd.DataFrame, weights: np.ndarray
    ) -> Dict:
        """Calculate liquidity risk metrics.

        Args:
            returns: Returns matrix
            volumes: Volume matrix (same structure as returns)
            weights: Portfolio weights

        Returns:
            Dictionary with liquidity metrics
        """
        # Amihud illiquidity measure: |return| / volume
        amihud_illiq = (returns.abs() / volumes).replace([np.inf, -np.inf], np.nan)

        # Portfolio-weighted illiquidity
        portfolio_illiq = (amihud_illiq * weights).sum(axis=1).dropna()

        # Liquidity-adjusted VaR (increase VaR by illiquidity factor)
        portfolio_returns = (returns * weights).sum(axis=1)
        base_var = portfolio_returns.quantile(0.05)

        # Simple adjustment: increase VaR by average illiquidity
        avg_illiq_factor = 1 + portfolio_illiq.mean()
        liquidity_adjusted_var = base_var * avg_illiq_factor

        return {
            "amihud_illiquidity": amihud_illiq.mean(),
            "portfolio_illiquidity": portfolio_illiq.mean(),
            "base_var": base_var,
            "liquidity_adjusted_var": liquidity_adjusted_var,
            "liquidity_premium": liquidity_adjusted_var - base_var,
        }

    def regime_risk_analysis(
        self, returns: pd.DataFrame, regime_indicators: pd.Series, weights: np.ndarray
    ) -> Dict:
        """Analyze risk across different market regimes.

        Args:
            returns: Returns matrix
            regime_indicators: Series indicating market regime (0, 1, 2, etc.)
            weights: Portfolio weights

        Returns:
            Dictionary with regime-specific risk metrics
        """
        portfolio_returns = (returns * weights).sum(axis=1)

        # Align data
        aligned_returns = portfolio_returns.reindex(regime_indicators.index).dropna()
        aligned_regimes = regime_indicators.reindex(aligned_returns.index)

        regime_stats = {}

        for regime in aligned_regimes.unique():
            regime_returns = aligned_returns[aligned_regimes == regime]

            if len(regime_returns) < 10:  # Need minimum observations
                continue

            regime_stats[f"regime_{regime}"] = {
                "n_observations": len(regime_returns),
                "mean_return": regime_returns.mean() * 252,
                "volatility": regime_returns.std() * np.sqrt(252),
                "var_95": regime_returns.quantile(0.05),
                "expected_shortfall": self.expected_shortfall(regime_returns),
                "max_drawdown": self._calculate_max_drawdown(regime_returns),
                "skewness": regime_returns.skew(),
                "kurtosis": regime_returns.kurtosis(),
            }

        return regime_stats

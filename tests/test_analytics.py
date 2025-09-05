"""Tests for analytics modules."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from robo_trader.analytics.attribution import AttributionAnalyzer
from robo_trader.analytics.performance import PerformanceAnalyzer
from robo_trader.analytics.risk_metrics import RiskAnalyzer


@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")

    # Create realistic daily returns
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, len(dates)),  # ~0.125% daily mean, 1% daily vol
        index=dates,
        name="returns",
    )

    return returns


@pytest.fixture
def sample_portfolio_returns():
    """Create sample portfolio returns matrix."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")

    # Create 5 assets with different characteristics
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    returns_data = {}

    for asset in assets:
        returns_data[asset] = np.random.normal(0.0005, 0.015, len(dates))

    returns_df = pd.DataFrame(returns_data, index=dates)

    return returns_df


@pytest.fixture
def sample_benchmark():
    """Create sample benchmark returns."""
    np.random.seed(123)  # Different seed for benchmark
    dates = pd.date_range("2023-01-01", periods=252, freq="D")

    benchmark = pd.Series(
        np.random.normal(0.0004, 0.008, len(dates)),  # Lower vol benchmark
        index=dates,
        name="benchmark",
    )

    return benchmark


class TestPerformanceAnalyzer:
    """Test cases for PerformanceAnalyzer."""

    def test_initialization(self):
        """Test PerformanceAnalyzer initialization."""
        analyzer = PerformanceAnalyzer()
        assert analyzer.risk_free_rate == 0.02
        assert analyzer.daily_rf_rate == 0.02 / 252

        # Test custom risk-free rate
        analyzer_custom = PerformanceAnalyzer(risk_free_rate=0.03)
        assert analyzer_custom.risk_free_rate == 0.03

    def test_calculate_returns(self, sample_returns):
        """Test returns calculation."""
        analyzer = PerformanceAnalyzer()

        # Create price series from returns
        prices = (1 + sample_returns).cumprod() * 100

        # Test simple returns
        calc_returns = analyzer.calculate_returns(prices, method="simple")
        assert len(calc_returns) == len(prices) - 1
        assert not calc_returns.isna().any()

        # Test log returns
        log_returns = analyzer.calculate_returns(prices, method="log")
        assert len(log_returns) == len(prices) - 1
        assert not log_returns.isna().any()

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.calculate_returns(prices, method="invalid")

    def test_sharpe_ratio(self, sample_returns):
        """Test Sharpe ratio calculation."""
        analyzer = PerformanceAnalyzer()

        sharpe = analyzer.sharpe_ratio(sample_returns)
        assert isinstance(sharpe, float)
        assert not np.isnan(sharpe)

        # Test with zero volatility (should return 0)
        zero_vol_returns = pd.Series([0.01] * 100)
        sharpe_zero_vol = analyzer.sharpe_ratio(zero_vol_returns)
        assert sharpe_zero_vol == 0.0

    def test_sortino_ratio(self, sample_returns):
        """Test Sortino ratio calculation."""
        analyzer = PerformanceAnalyzer()

        sortino = analyzer.sortino_ratio(sample_returns)
        assert isinstance(sortino, float)
        assert not np.isnan(sortino)

        # Test with all positive returns
        positive_returns = pd.Series([0.01] * 100)
        sortino_positive = analyzer.sortino_ratio(positive_returns)
        assert sortino_positive == float("inf")

    def test_calmar_ratio(self, sample_returns):
        """Test Calmar ratio calculation."""
        analyzer = PerformanceAnalyzer()

        calmar = analyzer.calmar_ratio(sample_returns)
        assert isinstance(calmar, float)

        # Test with no drawdown (all positive returns)
        positive_returns = pd.Series([0.01] * 100)
        calmar_no_dd = analyzer.calmar_ratio(positive_returns)
        assert calmar_no_dd == float("inf")

    def test_maximum_drawdown(self, sample_returns):
        """Test maximum drawdown calculation."""
        analyzer = PerformanceAnalyzer()

        max_dd, start_date, end_date, duration = analyzer.maximum_drawdown(sample_returns)

        assert isinstance(max_dd, float)
        assert max_dd <= 0  # Drawdown should be negative
        assert isinstance(start_date, (pd.Timestamp, type(sample_returns.index[0])))
        assert isinstance(end_date, (pd.Timestamp, type(sample_returns.index[0])))
        assert isinstance(duration, int)
        assert duration >= 0

    def test_value_at_risk(self, sample_returns):
        """Test Value at Risk calculation."""
        analyzer = PerformanceAnalyzer()

        # Test historical VaR
        var_hist = analyzer.value_at_risk(sample_returns, method="historical")
        assert isinstance(var_hist, float)
        assert var_hist <= 0  # VaR should be negative

        # Test parametric VaR
        var_param = analyzer.value_at_risk(sample_returns, method="parametric")
        assert isinstance(var_param, float)

        # Test Cornish-Fisher VaR
        var_cf = analyzer.value_at_risk(sample_returns, method="cornish_fisher")
        assert isinstance(var_cf, float)

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown VaR method"):
            analyzer.value_at_risk(sample_returns, method="invalid")

    def test_conditional_var(self, sample_returns):
        """Test Conditional Value at Risk calculation."""
        analyzer = PerformanceAnalyzer()

        cvar = analyzer.conditional_var(sample_returns)
        assert isinstance(cvar, float)
        assert cvar <= 0  # CVaR should be negative

        # CVaR should be more negative than VaR
        var = analyzer.value_at_risk(sample_returns)
        assert cvar <= var

    def test_beta_alpha(self, sample_returns, sample_benchmark):
        """Test beta and alpha calculation."""
        analyzer = PerformanceAnalyzer()

        # Test beta
        beta = analyzer.beta(sample_returns, sample_benchmark)
        assert isinstance(beta, float)
        assert not np.isnan(beta)

        # Test alpha
        alpha = analyzer.alpha(sample_returns, sample_benchmark)
        assert isinstance(alpha, float)

        # Test with misaligned series
        short_benchmark = sample_benchmark.iloc[:-50]
        beta_short = analyzer.beta(sample_returns, short_benchmark)
        assert isinstance(beta_short, float)

    def test_information_ratio(self, sample_returns, sample_benchmark):
        """Test information ratio calculation."""
        analyzer = PerformanceAnalyzer()

        ir = analyzer.information_ratio(sample_returns, sample_benchmark)
        assert isinstance(ir, float)

        # Test with identical series (should be 0)
        ir_identical = analyzer.information_ratio(sample_returns, sample_returns)
        assert ir_identical == 0.0

    def test_performance_summary(self, sample_returns, sample_benchmark):
        """Test comprehensive performance summary."""
        analyzer = PerformanceAnalyzer()

        # Test without benchmark
        summary = analyzer.performance_summary(sample_returns)
        assert isinstance(summary, dict)
        assert "total_return" in summary
        assert "annual_return" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary

        # Test with benchmark
        summary_with_bench = analyzer.performance_summary(sample_returns, sample_benchmark)
        assert "alpha" in summary_with_bench
        assert "beta" in summary_with_bench
        assert "information_ratio" in summary_with_bench

        # Test empty series
        empty_returns = pd.Series(dtype=float)
        empty_summary = analyzer.performance_summary(empty_returns)
        assert empty_summary == {}

    def test_monthly_returns_table(self, sample_returns):
        """Test monthly returns table generation."""
        analyzer = PerformanceAnalyzer()

        table = analyzer.monthly_returns_table(sample_returns)
        assert isinstance(table, pd.DataFrame)

        if not table.empty:
            assert "Annual" in table.columns
            assert table.index.name is None  # Should be years

        # Test empty series
        empty_returns = pd.Series(dtype=float)
        empty_table = analyzer.monthly_returns_table(empty_returns)
        assert empty_table.empty

    def test_drawdown_analysis(self, sample_returns):
        """Test drawdown analysis."""
        analyzer = PerformanceAnalyzer()

        dd_analysis = analyzer.drawdown_analysis(sample_returns)
        assert isinstance(dd_analysis, pd.DataFrame)

        if not dd_analysis.empty:
            expected_columns = ["start", "end", "max_drawdown", "duration_days"]
            for col in expected_columns:
                assert col in dd_analysis.columns


class TestRiskAnalyzer:
    """Test cases for RiskAnalyzer."""

    def test_initialization(self):
        """Test RiskAnalyzer initialization."""
        analyzer = RiskAnalyzer()
        assert analyzer.confidence_levels == [0.01, 0.05, 0.10]

        custom_levels = [0.01, 0.05]
        analyzer_custom = RiskAnalyzer(confidence_levels=custom_levels)
        assert analyzer_custom.confidence_levels == custom_levels

    def test_portfolio_var(self, sample_portfolio_returns):
        """Test portfolio VaR calculation."""
        analyzer = RiskAnalyzer()
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        # Test historical VaR
        var_hist = analyzer.portfolio_var(sample_portfolio_returns, weights, method="historical")
        assert isinstance(var_hist, float)
        assert var_hist <= 0

        # Test parametric VaR
        var_param = analyzer.portfolio_var(sample_portfolio_returns, weights, method="parametric")
        assert isinstance(var_param, float)

        # Test Monte Carlo VaR
        var_mc = analyzer.portfolio_var(sample_portfolio_returns, weights, method="monte_carlo")
        assert isinstance(var_mc, float)

    def test_component_var(self, sample_portfolio_returns):
        """Test component VaR calculation."""
        analyzer = RiskAnalyzer()
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        component_vars = analyzer.component_var(sample_portfolio_returns, weights)

        assert isinstance(component_vars, pd.Series)
        assert len(component_vars) == len(weights)
        assert component_vars.index.equals(sample_portfolio_returns.columns)

    def test_risk_parity_weights(self, sample_portfolio_returns):
        """Test risk parity weights calculation."""
        analyzer = RiskAnalyzer()

        rp_weights = analyzer.risk_parity_weights(sample_portfolio_returns)

        assert isinstance(rp_weights, np.ndarray)
        assert len(rp_weights) == sample_portfolio_returns.shape[1]
        assert abs(rp_weights.sum() - 1.0) < 1e-6  # Weights should sum to 1
        assert all(w >= 0 for w in rp_weights)  # Weights should be non-negative

    def test_stress_test(self, sample_portfolio_returns):
        """Test stress testing."""
        analyzer = RiskAnalyzer()
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        scenarios = [{"name": "Market Crash", "shocks": {"AAPL": -0.2, "MSFT": -0.15}}]

        stress_results = analyzer.stress_test(sample_portfolio_returns, weights, scenarios)

        assert isinstance(stress_results, pd.DataFrame)
        assert len(stress_results) == 2  # Baseline + 1 scenario
        assert "scenario" in stress_results.columns
        assert "portfolio_return" in stress_results.columns
        assert "var_95" in stress_results.columns

    def test_correlation_risk_analysis(self, sample_portfolio_returns):
        """Test correlation risk analysis."""
        analyzer = RiskAnalyzer()

        corr_analysis = analyzer.correlation_risk_analysis(sample_portfolio_returns)

        assert isinstance(corr_analysis, dict)
        assert "high_correlation_pairs" in corr_analysis
        assert "effective_assets" in corr_analysis
        assert "concentration_ratio" in corr_analysis
        assert "average_correlation" in corr_analysis

        assert isinstance(corr_analysis["high_correlation_pairs"], list)
        assert isinstance(corr_analysis["effective_assets"], float)
        assert corr_analysis["effective_assets"] > 0


class TestAttributionAnalyzer:
    """Test cases for AttributionAnalyzer."""

    def test_initialization(self):
        """Test AttributionAnalyzer initialization."""
        analyzer = AttributionAnalyzer()
        assert analyzer is not None

    def test_factor_attribution(self, sample_returns):
        """Test factor attribution analysis."""
        analyzer = AttributionAnalyzer()

        # Create sample factor returns
        factor_returns = pd.DataFrame(
            {
                "Market": np.random.normal(0.0003, 0.01, len(sample_returns)),
                "Size": np.random.normal(0.0001, 0.005, len(sample_returns)),
                "Value": np.random.normal(0.0002, 0.006, len(sample_returns)),
            },
            index=sample_returns.index,
        )

        # Create sample factor loadings
        factor_loadings = pd.DataFrame(
            {
                "Market": [0.8] * len(sample_returns),
                "Size": [0.2] * len(sample_returns),
                "Value": [-0.1] * len(sample_returns),
            },
            index=sample_returns.index,
        )

        attribution = analyzer.factor_attribution(sample_returns, factor_returns, factor_loadings)

        assert isinstance(attribution, dict)
        assert "factor_contributions" in attribution
        assert "total_factor_return" in attribution
        assert "specific_return" in attribution
        assert "attribution_summary" in attribution

        assert isinstance(attribution["factor_contributions"], pd.DataFrame)
        assert len(attribution["factor_contributions"].columns) == 3

    def test_regime_attribution(self, sample_returns):
        """Test regime attribution analysis."""
        analyzer = AttributionAnalyzer()

        # Create sample regime indicator
        regime_indicator = pd.Series(
            np.random.choice([0, 1, 2], size=len(sample_returns)), index=sample_returns.index
        )

        regime_names = {0: "Bull", 1: "Bear", 2: "Sideways"}

        attribution = analyzer.regime_attribution(sample_returns, regime_indicator, regime_names)

        assert isinstance(attribution, dict)
        assert "regime_statistics" in attribution
        assert "regime_contributions" in attribution

        for regime_name in regime_names.values():
            if regime_name in attribution["regime_statistics"]:
                stats = attribution["regime_statistics"][regime_name]
                assert "n_periods" in stats
                assert "total_return" in stats
                assert "avg_return" in stats

    def test_generate_attribution_report(self, sample_returns, sample_benchmark):
        """Test attribution report generation."""
        analyzer = AttributionAnalyzer()

        # Create dummy attribution results
        attribution_results = [
            {
                "attribution_summary": {
                    "total_return": 0.1,
                    "factor_1_contribution": 0.08,
                    "factor_2_contribution": 0.02,
                }
            }
        ]

        report = analyzer.generate_attribution_report(
            sample_returns, sample_benchmark, attribution_results, "Test Report"
        )

        assert isinstance(report, dict)
        assert "report_name" in report
        assert "performance_summary" in report
        assert "attribution_analyses" in report
        assert "generated_at" in report

        perf_summary = report["performance_summary"]
        assert "total_portfolio_return" in perf_summary
        assert "tracking_error" in perf_summary
        assert "information_ratio" in perf_summary

"""Performance attribution analysis for RoboTrader.

This module provides comprehensive performance attribution analysis,
breaking down returns by various factors and sources.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class AttributionAnalyzer:
    """Performance attribution analysis for trading strategies."""
    
    def __init__(self):
        """Initialize attribution analyzer."""
        pass
    
    def holdings_based_attribution(
        self,
        portfolio_returns: pd.Series,
        holdings: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        asset_returns: pd.DataFrame
    ) -> Dict:
        """Holdings-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns time series
            holdings: Portfolio holdings over time (weights)
            benchmark_weights: Benchmark weights over time
            asset_returns: Individual asset returns
            
        Returns:
            Dictionary with attribution results
        """
        # Align all data
        common_index = portfolio_returns.index.intersection(
            holdings.index
        ).intersection(
            benchmark_weights.index
        ).intersection(
            asset_returns.index
        )
        
        portfolio_returns = portfolio_returns.reindex(common_index)
        holdings = holdings.reindex(common_index)
        benchmark_weights = benchmark_weights.reindex(common_index)
        asset_returns = asset_returns.reindex(common_index)
        
        # Calculate benchmark returns
        benchmark_returns = (benchmark_weights * asset_returns).sum(axis=1)
        
        # Active weights
        active_weights = holdings - benchmark_weights
        
        # Asset allocation effect: (w_p - w_b) * R_b
        allocation_effect = (active_weights.shift(1) * asset_returns).sum(axis=1)
        
        # Security selection effect: w_b * (R_p - R_b)
        excess_returns = asset_returns.subtract(benchmark_returns, axis=0)
        selection_effect = (benchmark_weights.shift(1) * excess_returns).sum(axis=1)
        
        # Interaction effect: (w_p - w_b) * (R_p - R_b)
        interaction_effect = (active_weights.shift(1) * excess_returns).sum(axis=1)
        
        # Total active return
        total_active_return = portfolio_returns - benchmark_returns
        
        return {
            "total_active_return": total_active_return,
            "allocation_effect": allocation_effect,
            "selection_effect": selection_effect,
            "interaction_effect": interaction_effect,
            "benchmark_returns": benchmark_returns,
            "attribution_summary": {
                "total_active_return": total_active_return.sum(),
                "allocation_contribution": allocation_effect.sum(),
                "selection_contribution": selection_effect.sum(),
                "interaction_contribution": interaction_effect.sum()
            }
        }
    
    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        factor_loadings: pd.DataFrame
    ) -> Dict:
        """Factor-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns matrix
            factor_loadings: Portfolio factor loadings over time
            
        Returns:
            Dictionary with factor attribution
        """
        # Align data
        common_index = portfolio_returns.index.intersection(
            factor_returns.index
        ).intersection(
            factor_loadings.index
        )
        
        portfolio_returns = portfolio_returns.reindex(common_index)
        factor_returns = factor_returns.reindex(common_index)
        factor_loadings = factor_loadings.reindex(common_index)
        
        # Factor contributions: beta * factor_return
        factor_contributions = factor_loadings.shift(1) * factor_returns
        
        # Total factor return
        total_factor_return = factor_contributions.sum(axis=1)
        
        # Specific (idiosyncratic) return
        specific_return = portfolio_returns - total_factor_return
        
        # Risk attribution - how much risk comes from each factor
        factor_vars = {}
        for factor in factor_returns.columns:
            factor_var = (factor_loadings[factor].shift(1) * factor_returns[factor]).var()
            factor_vars[factor] = factor_var
        
        specific_var = specific_return.var()
        total_var = portfolio_returns.var()
        
        return {
            "factor_contributions": factor_contributions,
            "total_factor_return": total_factor_return,
            "specific_return": specific_return,
            "factor_variances": pd.Series(factor_vars),
            "specific_variance": specific_var,
            "total_variance": total_var,
            "attribution_summary": {
                "total_return": portfolio_returns.sum(),
                **{f"factor_{col}_contribution": factor_contributions[col].sum() 
                   for col in factor_contributions.columns},
                "specific_contribution": specific_return.sum()
            }
        }
    
    def sector_attribution(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        sector_mappings: Dict[str, str],
        sector_returns: pd.DataFrame
    ) -> Dict:
        """Sector-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns
            portfolio_weights: Portfolio weights by asset
            sector_mappings: Mapping of assets to sectors
            sector_returns: Sector benchmark returns
            
        Returns:
            Dictionary with sector attribution
        """
        # Create sector weights from asset weights
        sector_weights = pd.DataFrame(index=portfolio_weights.index)
        
        for sector in set(sector_mappings.values()):
            sector_assets = [asset for asset, sect in sector_mappings.items() if sect == sector]
            sector_assets = [asset for asset in sector_assets if asset in portfolio_weights.columns]
            
            if sector_assets:
                sector_weights[sector] = portfolio_weights[sector_assets].sum(axis=1)
        
        # Align data
        common_index = portfolio_returns.index.intersection(
            sector_weights.index
        ).intersection(
            sector_returns.index
        )
        
        portfolio_returns = portfolio_returns.reindex(common_index)
        sector_weights = sector_weights.reindex(common_index)
        sector_returns = sector_returns.reindex(common_index)
        
        # Calculate sector contributions
        sector_contributions = sector_weights.shift(1) * sector_returns
        
        # Portfolio return from sectors
        portfolio_from_sectors = sector_contributions.sum(axis=1)
        
        # Stock selection effect (residual)
        stock_selection = portfolio_returns - portfolio_from_sectors
        
        return {
            "sector_contributions": sector_contributions,
            "portfolio_from_sectors": portfolio_from_sectors,
            "stock_selection_effect": stock_selection,
            "sector_weights": sector_weights,
            "attribution_summary": {
                "total_return": portfolio_returns.sum(),
                **{f"sector_{sector}_contribution": sector_contributions[sector].sum()
                   for sector in sector_contributions.columns},
                "stock_selection_contribution": stock_selection.sum()
            }
        }
    
    def regime_attribution(
        self,
        portfolio_returns: pd.Series,
        regime_indicator: pd.Series,
        regime_names: Optional[Dict[int, str]] = None
    ) -> Dict:
        """Regime-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns
            regime_indicator: Regime indicator (0, 1, 2, etc.)
            regime_names: Optional mapping of regime codes to names
            
        Returns:
            Dictionary with regime attribution
        """
        if regime_names is None:
            regime_names = {i: f"Regime_{i}" for i in regime_indicator.unique()}
        
        # Align data
        common_index = portfolio_returns.index.intersection(regime_indicator.index)
        portfolio_returns = portfolio_returns.reindex(common_index)
        regime_indicator = regime_indicator.reindex(common_index)
        
        # Calculate returns by regime
        regime_stats = {}
        regime_contributions = {}
        
        for regime_code, regime_name in regime_names.items():
            regime_mask = regime_indicator == regime_code
            regime_returns = portfolio_returns[regime_mask]
            
            if len(regime_returns) > 0:
                regime_stats[regime_name] = {
                    "n_periods": len(regime_returns),
                    "total_return": regime_returns.sum(),
                    "avg_return": regime_returns.mean(),
                    "volatility": regime_returns.std(),
                    "sharpe_ratio": regime_returns.mean() / regime_returns.std() if regime_returns.std() > 0 else 0,
                    "proportion_of_time": len(regime_returns) / len(portfolio_returns)
                }
                
                regime_contributions[regime_name] = regime_returns.sum()
        
        return {
            "regime_statistics": regime_stats,
            "regime_contributions": regime_contributions,
            "total_attribution": sum(regime_contributions.values())
        }
    
    def style_attribution(
        self,
        portfolio_returns: pd.Series,
        style_factors: pd.DataFrame,
        portfolio_style_exposures: pd.DataFrame
    ) -> Dict:
        """Style-based performance attribution.
        
        Args:
            portfolio_returns: Portfolio returns
            style_factors: Style factor returns (Value, Growth, Quality, etc.)
            portfolio_style_exposures: Portfolio exposures to style factors
            
        Returns:
            Dictionary with style attribution
        """
        # Align data
        common_index = portfolio_returns.index.intersection(
            style_factors.index
        ).intersection(
            portfolio_style_exposures.index
        )
        
        portfolio_returns = portfolio_returns.reindex(common_index)
        style_factors = style_factors.reindex(common_index)
        portfolio_style_exposures = portfolio_style_exposures.reindex(common_index)
        
        # Style contributions
        style_contributions = portfolio_style_exposures.shift(1) * style_factors
        
        # Total style return
        total_style_return = style_contributions.sum(axis=1)
        
        # Residual (alpha)
        alpha = portfolio_returns - total_style_return
        
        # Style risk contributions
        style_risks = {}
        for style in style_factors.columns:
            style_contribution_vol = (
                portfolio_style_exposures[style].shift(1) * style_factors[style]
            ).std()
            style_risks[style] = style_contribution_vol
        
        alpha_vol = alpha.std()
        
        return {
            "style_contributions": style_contributions,
            "total_style_return": total_style_return,
            "alpha": alpha,
            "style_risks": pd.Series(style_risks),
            "alpha_volatility": alpha_vol,
            "attribution_summary": {
                "total_return": portfolio_returns.sum(),
                **{f"style_{style}_contribution": style_contributions[style].sum()
                   for style in style_contributions.columns},
                "alpha_contribution": alpha.sum()
            }
        }
    
    def timing_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_beta: pd.Series
    ) -> Dict:
        """Market timing attribution analysis.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            portfolio_beta: Time-varying portfolio beta
            
        Returns:
            Dictionary with timing attribution
        """
        # Align data
        common_index = portfolio_returns.index.intersection(
            benchmark_returns.index
        ).intersection(
            portfolio_beta.index
        )
        
        portfolio_returns = portfolio_returns.reindex(common_index)
        benchmark_returns = benchmark_returns.reindex(common_index)
        portfolio_beta = portfolio_beta.reindex(common_index)
        
        # Market return (excess of risk-free rate assumed to be 0 for simplicity)
        market_excess_return = benchmark_returns
        
        # Beta timing: (beta - avg_beta) * market_return
        avg_beta = portfolio_beta.mean()
        beta_timing = (portfolio_beta.shift(1) - avg_beta) * market_excess_return
        
        # Selectivity: portfolio_return - beta * market_return
        selectivity = portfolio_returns - portfolio_beta.shift(1) * market_excess_return
        
        # Market return contribution
        market_contribution = avg_beta * market_excess_return
        
        return {
            "market_contribution": market_contribution,
            "selectivity": selectivity,
            "timing_contribution": beta_timing,
            "total_attribution": market_contribution + selectivity + beta_timing,
            "attribution_summary": {
                "total_return": portfolio_returns.sum(),
                "market_contribution": market_contribution.sum(),
                "selectivity_contribution": selectivity.sum(),
                "timing_contribution": beta_timing.sum()
            }
        }
    
    def multi_period_attribution(
        self,
        single_period_returns: pd.Series,
        benchmark_returns: pd.Series,
        attribution_components: pd.DataFrame
    ) -> Dict:
        """Multi-period attribution analysis with compounding effects.
        
        Args:
            single_period_returns: Single period portfolio returns
            benchmark_returns: Single period benchmark returns
            attribution_components: DataFrame with attribution components
            
        Returns:
            Dictionary with multi-period attribution
        """
        # Calculate cumulative returns
        portfolio_cumret = (1 + single_period_returns).cumprod() - 1
        benchmark_cumret = (1 + benchmark_returns).cumprod() - 1
        
        # Active return
        active_return = portfolio_cumret - benchmark_cumret
        
        # Compound attribution components
        compound_components = {}
        
        for component in attribution_components.columns:
            # Geometric linking of attribution
            component_series = attribution_components[component]
            
            # Simple compounding for attribution
            compound_components[component] = (1 + component_series).cumprod() - 1
        
        # Calculate attribution statistics
        total_portfolio_return = portfolio_cumret.iloc[-1]
        total_benchmark_return = benchmark_cumret.iloc[-1]
        total_active_return = active_return.iloc[-1]
        
        # Component contributions to total active return
        component_contributions = {}
        for component, series in compound_components.items():
            component_contributions[component] = series.iloc[-1]
        
        return {
            "portfolio_cumulative_return": portfolio_cumret,
            "benchmark_cumulative_return": benchmark_cumret,
            "active_cumulative_return": active_return,
            "compound_attribution_components": pd.DataFrame(compound_components),
            "summary": {
                "total_portfolio_return": total_portfolio_return,
                "total_benchmark_return": total_benchmark_return,
                "total_active_return": total_active_return,
                **component_contributions
            }
        }
    
    def transaction_cost_attribution(
        self,
        gross_returns: pd.Series,
        net_returns: pd.Series,
        transaction_costs: pd.Series
    ) -> Dict:
        """Attribution of transaction cost impact.
        
        Args:
            gross_returns: Returns before transaction costs
            net_returns: Returns after transaction costs
            transaction_costs: Transaction costs per period
            
        Returns:
            Dictionary with transaction cost attribution
        """
        # Align data
        common_index = gross_returns.index.intersection(
            net_returns.index
        ).intersection(
            transaction_costs.index
        )
        
        gross_returns = gross_returns.reindex(common_index)
        net_returns = net_returns.reindex(common_index)
        transaction_costs = transaction_costs.reindex(common_index)
        
        # Calculate cost impact
        cost_impact = gross_returns - net_returns
        
        # Cumulative impact
        cumulative_cost_impact = cost_impact.cumsum()
        
        # Cost as percentage of gross returns
        cost_ratio = cost_impact / gross_returns.abs()
        cost_ratio = cost_ratio.replace([np.inf, -np.inf], np.nan)
        
        return {
            "cost_impact": cost_impact,
            "cumulative_cost_impact": cumulative_cost_impact,
            "cost_ratio": cost_ratio,
            "summary": {
                "total_gross_return": gross_returns.sum(),
                "total_net_return": net_returns.sum(),
                "total_cost_impact": cost_impact.sum(),
                "avg_cost_ratio": cost_ratio.mean(),
                "cost_drag_annualized": cost_impact.mean() * 252
            }
        }
    
    def generate_attribution_report(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        attribution_results: List[Dict],
        report_name: str = "Performance Attribution Report"
    ) -> Dict:
        """Generate comprehensive attribution report.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            attribution_results: List of attribution analysis results
            report_name: Name for the report
            
        Returns:
            Dictionary with comprehensive report
        """
        # Basic performance metrics
        total_portfolio_return = (1 + portfolio_returns).prod() - 1
        total_benchmark_return = (1 + benchmark_returns).prod() - 1
        total_active_return = total_portfolio_return - total_benchmark_return
        
        portfolio_vol = portfolio_returns.std() * np.sqrt(252)
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Tracking error
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252)
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Compile all attribution results
        compiled_attributions = {}
        for i, result in enumerate(attribution_results):
            if "attribution_summary" in result:
                compiled_attributions[f"attribution_{i}"] = result["attribution_summary"]
        
        report = {
            "report_name": report_name,
            "period_start": portfolio_returns.index.min(),
            "period_end": portfolio_returns.index.max(),
            "performance_summary": {
                "total_portfolio_return": total_portfolio_return,
                "total_benchmark_return": total_benchmark_return,
                "total_active_return": total_active_return,
                "portfolio_volatility": portfolio_vol,
                "benchmark_volatility": benchmark_vol,
                "tracking_error": tracking_error,
                "information_ratio": information_ratio
            },
            "attribution_analyses": compiled_attributions,
            "generated_at": datetime.now().isoformat()
        }
        
        return report
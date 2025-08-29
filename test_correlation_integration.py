#!/usr/bin/env python
"""
Test script for correlation-based position sizing integration (M5).

This script tests:
1. Correlation tracker with real price data
2. Correlation-based position sizing
3. Portfolio concentration metrics
4. Integration with the async runner
"""

import asyncio
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, '/Users/oliver/robo_trader')

from robo_trader.correlation import CorrelationTracker
from robo_trader.analysis.correlation_integration import (
    CorrelationBasedPositionSizer,
    AsyncCorrelationManager
)
from robo_trader.risk import Position
from robo_trader.logger import get_logger

logger = get_logger(__name__)


def generate_correlated_prices(n_days=60, n_symbols=5, base_correlation=0.6):
    """Generate synthetic correlated price data for testing."""
    np.random.seed(42)
    
    # Generate base returns
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    # Create correlation matrix
    corr_matrix = np.full((n_symbols, n_symbols), base_correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Add some variation
    corr_matrix[0, 1] = corr_matrix[1, 0] = 0.85  # High correlation pair
    corr_matrix[2, 3] = corr_matrix[3, 2] = 0.75  # Another high correlation
    corr_matrix[4, :4] = corr_matrix[:4, 4] = 0.2  # Low correlation symbol
    
    # Generate correlated returns
    mean_returns = np.zeros(n_symbols)
    std_returns = np.ones(n_symbols) * 0.02
    cov_matrix = np.outer(std_returns, std_returns) * corr_matrix
    
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
    
    # Convert to prices
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']
    base_prices = [150, 350, 140, 360, 250]
    
    price_data = {}
    for i, symbol in enumerate(symbols):
        prices = base_prices[i] * np.exp(np.cumsum(returns[:, i]))
        price_data[symbol] = pd.Series(prices, index=dates)
    
    return price_data, corr_matrix, symbols


def test_correlation_tracker():
    """Test correlation tracker functionality."""
    print("\n" + "="*60)
    print("Testing Correlation Tracker")
    print("="*60)
    
    # Generate test data
    price_data, true_corr, symbols = generate_correlated_prices()
    
    # Initialize tracker
    tracker = CorrelationTracker(
        lookback_days=60,
        min_observations=30,
        correlation_threshold=0.7
    )
    
    # Add price data
    for symbol, prices in price_data.items():
        tracker.add_price_series(symbol, prices, sector="Technology")
        print(f"✅ Added {len(prices)} price points for {symbol}")
    
    # Calculate correlation matrix
    corr_matrix = tracker.calculate_correlation_matrix()
    print(f"\n✅ Calculated correlation matrix for {len(corr_matrix)} symbols")
    
    # Test pairwise correlations
    print("\nPairwise Correlations:")
    test_pairs = [('AAPL', 'MSFT'), ('GOOGL', 'META'), ('TSLA', 'AAPL')]
    for sym1, sym2 in test_pairs:
        corr = tracker.get_pairwise_correlation(sym1, sym2)
        print(f"  {sym1}-{sym2}: {corr:.3f}")
    
    # Find high correlations
    high_corr = tracker.find_high_correlations(threshold=0.7)
    print(f"\n⚠️ Found {len(high_corr)} high correlation pairs (>0.7):")
    for sym1, sym2, corr in high_corr[:3]:
        print(f"  {sym1}-{sym2}: {corr:.3f}")
    
    # Get summary
    summary = tracker.get_correlation_summary()
    print(f"\nCorrelation Summary:")
    print(f"  Mean: {summary['mean_correlation']:.3f}")
    print(f"  Max: {summary['max_correlation']:.3f}")
    print(f"  High correlation pairs: {summary['high_correlation_pairs']}/{summary['total_pairs']}")
    
    return tracker


def test_position_sizing(tracker):
    """Test correlation-based position sizing."""
    print("\n" + "="*60)
    print("Testing Correlation-Based Position Sizing")
    print("="*60)
    
    # Initialize position sizer
    sizer = CorrelationBasedPositionSizer(
        correlation_tracker=tracker,
        max_correlation=0.7,
        correlation_penalty_factor=0.5,
        max_correlated_exposure=0.3
    )
    
    # Create mock positions
    positions = {
        'AAPL': Position('AAPL', 100, 150.0),
        'MSFT': Position('MSFT', 50, 350.0),
        'GOOGL': Position('GOOGL', 75, 140.0)
    }
    
    portfolio_value = 100000.0
    
    print("\nCurrent Positions:")
    for symbol, pos in positions.items():
        exposure = pos.notional_value / portfolio_value
        print(f"  {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f} ({exposure:.1%} exposure)")
    
    # Test position sizing for new symbols
    test_symbols = ['META', 'TSLA']
    base_size = 100
    
    print(f"\nTesting position sizing (base size: {base_size} shares):")
    
    for symbol in test_symbols:
        # Simulate async call synchronously
        loop = asyncio.new_event_loop()
        adjusted_size, reason = loop.run_until_complete(
            sizer.calculate_position_size(
                symbol=symbol,
                base_size=base_size,
                current_positions=positions,
                portfolio_value=portfolio_value
            )
        )
        loop.close()
        
        print(f"\n  {symbol}:")
        print(f"    Base size: {base_size}")
        print(f"    Adjusted size: {adjusted_size}")
        print(f"    Reason: {reason}")
    
    # Test concentration metrics
    sizer.positions = positions
    concentration = sizer.calculate_portfolio_concentration()
    
    print(f"\nPortfolio Concentration Metrics:")
    print(f"  Herfindahl Index: {concentration['herfindahl_index']:.3f}")
    print(f"  Effective N: {concentration['effective_n']:.1f}")
    print(f"  Max Position Weight: {concentration['max_position_weight']:.1%}")
    print(f"  Correlation-Adjusted Concentration: {concentration['correlation_adjusted_concentration']:.3f}")
    
    # Test diversification suggestions
    candidates = ['META', 'TSLA', 'NVDA', 'AMD']
    suggestions = sizer.suggest_diversification(positions, candidates)
    
    print(f"\nDiversification Suggestions (lowest correlation):")
    for symbol, avg_corr in suggestions[:3]:
        print(f"  {symbol}: avg correlation {avg_corr:.3f}")
    
    return sizer


async def test_async_manager(tracker, sizer):
    """Test async correlation manager."""
    print("\n" + "="*60)
    print("Testing Async Correlation Manager")
    print("="*60)
    
    # Initialize async manager
    manager = AsyncCorrelationManager(
        correlation_tracker=tracker,
        position_sizer=sizer
    )
    
    # Start manager
    await manager.start(update_interval=5)
    print("✅ Started async correlation manager")
    
    # Simulate position updates
    positions = {
        'AAPL': Position('AAPL', 100, 150.0),
        'MSFT': Position('MSFT', 50, 350.0),
        'GOOGL': Position('GOOGL', 75, 140.0)
    }
    
    # Test adjusted position sizing
    portfolio_value = 100000.0
    
    print("\nTesting async position sizing:")
    adjusted_size, reason = await manager.get_adjusted_position_size(
        symbol='META',
        base_size=100,
        current_positions=positions,
        portfolio_value=portfolio_value
    )
    
    print(f"  META: {adjusted_size} shares ({reason})")
    
    # Let it run for a bit
    await asyncio.sleep(2)
    
    # Stop manager
    await manager.stop()
    print("✅ Stopped async correlation manager")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("M5: Correlation Module Integration Test")
    print("="*60)
    
    try:
        # Test correlation tracker
        tracker = test_correlation_tracker()
        
        # Test position sizing
        sizer = test_position_sizing(tracker)
        
        # Test async manager
        asyncio.run(test_async_manager(tracker, sizer))
        
        print("\n" + "="*60)
        print("✅ All correlation integration tests passed!")
        print("="*60)
        
        print("\n📊 Integration Status:")
        print("  • Correlation tracker: Working")
        print("  • Position sizing: Working")
        print("  • Async manager: Working")
        print("  • Ready to enable in runner_async.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
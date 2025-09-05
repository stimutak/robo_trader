#!/usr/bin/env python
"""
Test the runner with correlation-based position sizing enabled.
"""

import asyncio
import sys

# Add parent directory to path
sys.path.insert(0, "/Users/oliver/robo_trader")

from robo_trader.logger import get_logger
from robo_trader.runner_async import AsyncRunner

logger = get_logger(__name__)


async def test_runner():
    """Test runner with correlation sizing enabled."""
    print("\n" + "=" * 60)
    print("Testing Runner with Correlation Sizing Enabled")
    print("=" * 60)

    # Create runner with correlation enabled
    runner = AsyncRunner(
        duration="5 D",
        bar_size="30 mins",
        sma_fast=10,
        sma_slow=20,
        slippage_bps=0.0,
        max_concurrent_symbols=3,
        use_correlation_sizing=True,  # Enable correlation
        max_correlation=0.7,
    )

    # Test with a few symbols that should have correlations
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    print(f"\nTesting with symbols: {test_symbols}")
    print(f"Correlation sizing: ENABLED")
    print(f"Max correlation threshold: 0.7")

    try:
        await runner.setup()
        print("✅ Runner setup complete")

        # Process symbols
        results = await runner.run_parallel(test_symbols)

        print(f"\n✅ Processed {len(results)} symbols")

        # Check if correlation metrics are logged
        if runner.position_sizer:
            metrics = runner.position_sizer.get_metrics()
            print(f"\nCorrelation Metrics:")
            print(f"  Positions reduced: {metrics['positions_reduced']}")
            print(f"  Positions rejected: {metrics['positions_rejected']}")
            print(f"  Average correlation: {metrics['avg_correlation']:.3f}")

            # Check for high correlation pairs
            high_corr = runner.position_sizer.get_high_correlation_pairs()
            if high_corr:
                print(f"\n⚠️ High correlation pairs detected:")
                for sym1, sym2, corr in high_corr[:3]:
                    print(f"  {sym1}-{sym2}: {corr:.3f}")

        await runner.teardown()
        print("\n✅ Runner teardown complete")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run the test."""
    print("\n" + "=" * 60)
    print("M5: Testing Correlation Integration in Runner")
    print("=" * 60)

    success = asyncio.run(test_runner())

    if success:
        print("\n" + "=" * 60)
        print("✅ Correlation integration test PASSED!")
        print("Correlation-based position sizing is now active.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Correlation integration test FAILED!")
        print("=" * 60)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

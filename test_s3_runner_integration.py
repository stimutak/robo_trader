#!/usr/bin/env python3
"""Test S3: Multi-Strategy Portfolio Manager integration with runner_async."""

import asyncio
import sys
from datetime import datetime

from robo_trader.runner_async import AsyncRunner
from robo_trader.portfolio_manager.portfolio_manager import AllocationMethod


async def test_portfolio_manager_in_runner():
    """Test that portfolio manager works in the runner."""
    print("\n" + "=" * 60)
    print("Testing Portfolio Manager in AsyncRunner")
    print("=" * 60)
    
    # Create runner with portfolio manager enabled
    runner = AsyncRunner(
        duration=1,  # 1 second for testing
        slippage_bps=5.0,
        use_portfolio_manager=True,
        portfolio_allocation_method="adaptive"
    )
    
    # Initialize
    await runner.setup()
    
    # Check that portfolio manager is configured
    assert runner.use_portfolio_manager == True
    assert runner.portfolio_manager is not None
    assert runner.portfolio_manager.allocation_method == AllocationMethod.ADAPTIVE
    
    print("✅ Portfolio manager initialized in runner")
    print(f"✅ Allocation method: {runner.portfolio_manager.allocation_method.value}")
    print(f"✅ Strategies registered: {len(runner.portfolio_manager.strategies)}")
    
    # Check allocations
    summary = runner.portfolio_manager.get_allocation_summary()
    print("\n📊 Portfolio Allocation:")
    for strategy_name, details in summary['strategies'].items():
        print(f"   {strategy_name}:")
        print(f"      Target Weight: {details['target_weight']:.2%}")
        print(f"      Allocated Capital: ${details['allocated_capital']:,.2f}")
    
    # Cleanup
    await runner.cleanup()
    
    return True


async def test_allocation_methods():
    """Test different allocation methods in runner."""
    print("\n" + "=" * 60)
    print("Testing Different Allocation Methods")
    print("=" * 60)
    
    methods = ["equal_weight", "risk_parity", "adaptive", "kelly_optimal"]
    
    for method in methods:
        runner = AsyncRunner(
            duration=1,
            slippage_bps=5.0,
            use_portfolio_manager=True,
            portfolio_allocation_method=method
        )
        
        await runner.setup()
        
        print(f"\n📊 {method.upper()} Allocation:")
        summary = runner.portfolio_manager.get_allocation_summary()
        for strategy_name in summary['strategies']:
            weight = summary['strategies'][strategy_name]['target_weight']
            print(f"   {strategy_name}: {weight:.2%}")
        
        await runner.cleanup()
    
    return True


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("🚀 S3 Portfolio Manager - Runner Integration Tests")
    print("=" * 60)
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    tests = [
        ("Portfolio Manager in Runner", test_portfolio_manager_in_runner),
        ("Allocation Methods", test_allocation_methods),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Integration Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:30} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All integration tests passed!")
        print("\n📝 Usage:")
        print("   python -m robo_trader.runner_async --use-portfolio-manager")
        print("   python -m robo_trader.runner_async --use-portfolio-manager --portfolio-allocation-method risk_parity")
        print("\n✨ S3 Task (Multi-Strategy Portfolio Manager) is COMPLETE!")
    else:
        print(f"\n⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
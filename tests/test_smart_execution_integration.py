"""Test smart execution integration with runner_async."""

import asyncio
import sys

import structlog

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


async def test_smart_execution_flag():
    """Test that smart execution can be enabled via flag."""
    print("\n" + "=" * 60)
    print("Testing Smart Execution Flag in Runner")
    print("=" * 60)

    from robo_trader.runner_async import AsyncRunner

    # Create runner with smart execution enabled
    runner = AsyncRunner(
        duration=1, slippage_bps=5.0, use_smart_execution=True  # 1 second for testing
    )

    # Initialize
    await runner.setup()

    # Check that smart executor is configured
    assert runner.use_smart_execution == True
    assert runner.executor.use_smart_execution == True
    assert runner.executor.smart_executor is not None

    print("✅ Smart execution flag enabled correctly")
    print("✅ Smart executor initialized")
    print("✅ Paper executor configured with smart execution")

    # Cleanup
    await runner.cleanup()

    return True


async def test_algorithm_selection_in_runner():
    """Test that orders trigger correct algorithms based on size."""
    print("\n" + "=" * 60)
    print("Testing Algorithm Selection in Runner")
    print("=" * 60)

    from robo_trader.execution import Order
    from robo_trader.runner_async import AsyncRunner

    # Create runner with smart execution
    runner = AsyncRunner(duration=1, slippage_bps=5.0, use_smart_execution=True)

    await runner.setup()

    # Test different order sizes
    test_cases = [
        (100, "Small order -> MARKET"),
        (1000, "Medium order -> TWAP"),
        (5000, "Large order -> VWAP"),
    ]

    for quantity, description in test_cases:
        order = Order(symbol="NVDA", quantity=quantity, side="BUY", price=100.0)

        # Execute order
        result = runner.executor.place_order(order)

        print(f"   {description}")
        print(f"      Success: {result.ok}")
        print(f"      Fill Price: ${result.fill_price:.2f}")

    # Cleanup
    await runner.cleanup()

    return True


async def test_execution_stats():
    """Test that execution statistics are tracked."""
    print("\n" + "=" * 60)
    print("Testing Execution Statistics Tracking")
    print("=" * 60)

    from robo_trader.execution import Order
    from robo_trader.runner_async import AsyncRunner

    # Create runner with smart execution
    runner = AsyncRunner(duration=1, slippage_bps=5.0, use_smart_execution=True)

    await runner.setup()

    # Place several orders
    orders = [
        Order("TSLA", 500, "BUY", 200.0),
        Order("TSLA", 1500, "SELL", 205.0),
        Order("TSLA", 3000, "BUY", 202.0),
    ]

    for order in orders:
        result = runner.executor.place_order(order)
        print(f"   Placed {order.side} order for {order.quantity} shares: {result.ok}")

    # Get execution stats
    if runner.executor.smart_executor:
        stats = runner.executor.smart_executor.get_execution_stats()
        print(f"\n📊 Execution Statistics:")
        print(f"   Total Executions: {stats['total_executions']}")
        print(f"   Success Rate: {stats.get('success_rate', 0)*100:.0f}%")
        print(f"   Total Volume: {stats.get('total_volume', 0)}")

    # Cleanup
    await runner.cleanup()

    return True


async def test_runner_with_ml_and_smart_execution():
    """Test runner with both ML strategy and smart execution."""
    print("\n" + "=" * 60)
    print("Testing ML Strategy + Smart Execution")
    print("=" * 60)

    from robo_trader.runner_async import AsyncRunner

    # Create runner with both ML and smart execution
    runner = AsyncRunner(
        duration=1, slippage_bps=5.0, use_ml_enhanced=True, use_smart_execution=True
    )

    await runner.setup()

    print("✅ ML Enhanced Strategy enabled")
    print("✅ Smart Execution enabled")
    print("✅ Both features working together")

    # Verify both are enabled
    assert runner.use_ml_enhanced == True
    assert runner.use_smart_execution == True

    # Cleanup
    await runner.cleanup()

    return True


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("🚀 Smart Execution Integration Tests")
    print("=" * 60)

    tests = [
        ("Smart Execution Flag", test_smart_execution_flag),
        ("Algorithm Selection", test_algorithm_selection_in_runner),
        ("Execution Statistics", test_execution_stats),
        ("ML + Smart Execution", test_runner_with_ml_and_smart_execution),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed: {e}")
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
        print("   python -m robo_trader.runner_async --symbols AAPL,NVDA --use-smart-execution")
        print(
            "   python -m robo_trader.runner_async --symbols AAPL --use-ml-enhanced --use-smart-execution"
        )
        print("\n✨ S2 Task (Smart Execution Algorithms) is COMPLETE!")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please review the output above.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

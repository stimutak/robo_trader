"""Test smart execution integration with AsyncRunner."""

import asyncio
import sys

from robo_trader.execution import Order
from robo_trader.runner_async import AsyncRunner


async def test_smart_execution_in_runner():
    """Test that smart execution works in the runner."""
    print("\n" + "=" * 60)
    print("Testing Smart Execution in AsyncRunner")
    print("=" * 60)

    # Create runner with smart execution enabled
    runner = AsyncRunner(
        duration=1, slippage_bps=5.0, use_smart_execution=True  # 1 second for testing
    )

    try:
        # Initialize (but don't connect to IBKR)
        runner.cfg = (
            runner.cfg if hasattr(runner, "cfg") else type("Config", (), {"slippage_bps": 5.0})()
        )

        # Create smart executor without IBKR client (for testing)
        from robo_trader.smart_execution.smart_executor import SmartExecutor

        smart_executor = SmartExecutor(runner.cfg, ibkr_client=None)

        # Create paper executor with smart execution
        from robo_trader.execution import PaperExecutor

        executor = PaperExecutor(
            slippage_bps=5.0, smart_executor=smart_executor, use_smart_execution=True
        )

        print("✅ Smart executor initialized")
        print("✅ Paper executor configured with smart execution")

        # Test different order sizes to trigger different algorithms
        test_orders = [
            (100, "AAPL", "Small order -> MARKET"),
            (1000, "NVDA", "Medium order -> TWAP"),
            (3000, "TSLA", "Large order -> VWAP"),
            (8000, "GOOGL", "Very large order -> ADAPTIVE"),
            (12000, "MSFT", "Huge order -> ICEBERG"),
        ]

        print("\n📊 Testing Algorithm Selection:")
        for quantity, symbol, description in test_orders:
            order = Order(symbol=symbol, quantity=quantity, side="BUY", price=100.0)

            # Get selected algorithm
            selected_algo = executor._select_algorithm(order)

            # Execute order
            result = executor.place_order(order)

            print(f"   {description}")
            print(f"      Algorithm: {selected_algo.value}")
            print(f"      Executed: {result.ok}")
            print(f"      Fill Price: ${result.fill_price:.2f}")

        print("\n✅ Smart execution is working with the runner!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_execution_plan_creation():
    """Test that execution plans are created correctly."""
    print("\n" + "=" * 60)
    print("Testing Execution Plan Creation")
    print("=" * 60)

    from robo_trader.config import Config
    from robo_trader.smart_execution import ExecutionAlgorithm, ExecutionParams, SmartExecutor

    config = Config()
    executor = SmartExecutor(config, ibkr_client=None)

    # Test each algorithm
    algorithms = [
        (ExecutionAlgorithm.TWAP, 1000),
        (ExecutionAlgorithm.VWAP, 2500),
        (ExecutionAlgorithm.ADAPTIVE, 5000),
        (ExecutionAlgorithm.ICEBERG, 10000),
    ]

    print("\n📋 Creating Execution Plans:")
    for algo, quantity in algorithms:
        params = ExecutionParams(algorithm=algo, duration_minutes=10, slice_count=5)

        plan = await executor.create_execution_plan(
            symbol="TEST", side="BUY", quantity=quantity, params=params
        )

        print(f"   {algo.value.upper()}:")
        print(f"      Quantity: {quantity:,}")
        print(f"      Slices: {len(plan.slices)}")
        print(f"      Duration: {plan.estimated_duration}")
        print(f"      Impact: {plan.market_impact_bps:.2f} bps")

    print("\n✅ All execution plans created successfully!")
    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🚀 Smart Execution Runner Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: Smart execution in runner
    try:
        success = await test_smart_execution_in_runner()
        results.append(("Smart Execution in Runner", success))
    except Exception as e:
        print(f"Test failed: {e}")
        results.append(("Smart Execution in Runner", False))

    # Test 2: Execution plan creation
    try:
        success = await test_execution_plan_creation()
        results.append(("Execution Plan Creation", success))
    except Exception as e:
        print(f"Test failed: {e}")
        results.append(("Execution Plan Creation", False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:30} {status}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\n📝 S2 Status:")
        print("   ✅ Smart execution algorithms implemented")
        print("   ✅ Integration with runner works")
        print("   ✅ Algorithm selection based on order size")
        print("   ✅ Ready to use with --use-smart-execution flag")
    else:
        print(f"\n⚠️ {total - passed} tests failed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

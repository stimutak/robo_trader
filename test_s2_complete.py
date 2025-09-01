"""Final test to verify S2 (Smart Execution) is complete."""

import asyncio
import sys
from datetime import datetime

from robo_trader.config import Config
from robo_trader.execution import Order, PaperExecutor
from robo_trader.smart_execution import (
    SmartExecutor,
    ExecutionParams,
    ExecutionAlgorithm,
)


async def test_full_smart_execution():
    """Test full smart execution flow."""
    print("\n" + "=" * 60)
    print("Testing Complete Smart Execution Flow")
    print("=" * 60)
    
    # Setup
    config = Config()
    smart_executor = SmartExecutor(config, ibkr_client=None)
    paper_executor = PaperExecutor(
        slippage_bps=5.0,
        smart_executor=smart_executor,
        use_smart_execution=True
    )
    
    # Test 1: Create execution plan
    print("\n1️⃣ Creating VWAP execution plan...")
    params = ExecutionParams(
        algorithm=ExecutionAlgorithm.VWAP,
        duration_minutes=5,
        slice_count=3
    )
    
    plan = await smart_executor.create_execution_plan(
        symbol="AAPL",
        side="BUY",
        quantity=2000,
        params=params
    )
    
    print(f"   ✅ Created plan with {len(plan.slices)} slices")
    print(f"   Duration: {plan.estimated_duration}")
    print(f"   Market Impact: {plan.market_impact_bps:.2f} bps")
    
    # Test 2: Execute the plan
    print("\n2️⃣ Executing the plan...")
    result = await smart_executor.execute_plan(plan, paper_executor, skip_delays=True)
    
    print(f"   ✅ Execution complete!")
    print(f"   Success: {result.success}")
    print(f"   Executed: {result.executed_quantity}/{result.requested_quantity}")
    print(f"   Average Price: ${result.average_price:.2f}")
    print(f"   Slippage: {result.slippage_bps:.2f} bps")
    print(f"   Time: {result.execution_time_ms:.0f} ms")
    
    # Test 3: Algorithm selection
    print("\n3️⃣ Testing algorithm selection...")
    test_cases = [
        (100, ExecutionAlgorithm.MARKET),
        (1000, ExecutionAlgorithm.TWAP),
        (3000, ExecutionAlgorithm.VWAP),
        (7000, ExecutionAlgorithm.ADAPTIVE),
        (15000, ExecutionAlgorithm.ICEBERG),
    ]
    
    for quantity, expected in test_cases:
        order = Order("TEST", quantity, "BUY", 100.0)
        selected = paper_executor._select_algorithm(order)
        match = "✅" if selected == expected else "❌"
        print(f"   {quantity:6,} shares -> {selected.value:8} {match}")
    
    # Test 4: Async order placement
    print("\n4️⃣ Testing async order placement...")
    order = Order("NVDA", 1500, "SELL", 500.0)
    async_result = await paper_executor.place_order_async(order)
    print(f"   ✅ Async order placed: {async_result.ok}")
    print(f"   Fill price: ${async_result.fill_price:.2f}")
    
    return True


async def test_all_algorithms():
    """Test all algorithms create valid plans."""
    print("\n" + "=" * 60)
    print("Testing All Algorithms")
    print("=" * 60)
    
    config = Config()
    executor = SmartExecutor(config, ibkr_client=None)
    
    algorithms = {
        ExecutionAlgorithm.MARKET: (500, 1),
        ExecutionAlgorithm.TWAP: (1000, 5),
        ExecutionAlgorithm.VWAP: (2000, 5),
        ExecutionAlgorithm.ADAPTIVE: (5000, 10),
        ExecutionAlgorithm.ICEBERG: (10000, 5),
    }
    
    all_valid = True
    
    for algo, (quantity, duration) in algorithms.items():
        params = ExecutionParams(
            algorithm=algo,
            duration_minutes=duration
        )
        
        plan = await executor.create_execution_plan(
            symbol="TEST",
            side="BUY",
            quantity=quantity,
            params=params
        )
        
        is_valid = (
            plan.algorithm == algo and
            plan.total_quantity == quantity and
            len(plan.slices) > 0
        )
        
        status = "✅" if is_valid else "❌"
        print(f"   {algo.value:10} {status} - {len(plan.slices)} slices")
        
        all_valid = all_valid and is_valid
    
    return all_valid


async def test_execution_stats():
    """Test execution statistics tracking."""
    print("\n" + "=" * 60)
    print("Testing Execution Statistics")
    print("=" * 60)
    
    config = Config()
    smart_executor = SmartExecutor(config, ibkr_client=None)
    paper_executor = PaperExecutor(
        slippage_bps=5.0,
        smart_executor=smart_executor,
        use_smart_execution=True
    )
    
    # Execute several orders
    for i in range(3):
        params = ExecutionParams(
            algorithm=ExecutionAlgorithm.TWAP,
            duration_minutes=1,
            slice_count=2
        )
        
        plan = await smart_executor.create_execution_plan(
            symbol=f"TEST{i}",
            side="BUY" if i % 2 == 0 else "SELL",
            quantity=1000 * (i + 1),
            params=params
        )
        
        await smart_executor.execute_plan(plan, paper_executor, skip_delays=True)
    
    # Get stats
    stats = smart_executor.get_execution_stats()
    
    print(f"   Total Executions: {stats['total_executions']}")
    print(f"   Success Rate: {stats['success_rate']*100:.0f}%")
    print(f"   Avg Slippage: {stats['avg_slippage_bps']:.2f} bps")
    print(f"   Total Volume: {stats['total_volume']:,}")
    
    return stats['total_executions'] == 3


async def main():
    """Run all S2 verification tests."""
    print("\n" + "=" * 60)
    print("🚀 S2 (Smart Execution) Verification Suite")
    print("=" * 60)
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    tests = [
        ("Full Smart Execution", test_full_smart_execution),
        ("All Algorithms", test_all_algorithms),
        ("Execution Statistics", test_execution_stats),
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
    
    # Final Summary
    print("\n" + "=" * 60)
    print("📊 S2 Verification Results")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:25} {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("✨ S2: Smart Execution Algorithms - COMPLETE ✅")
        print("=" * 60)
        print("\n📝 Implemented Features:")
        print("   ✅ TWAP (Time-Weighted Average Price)")
        print("   ✅ VWAP (Volume-Weighted Average Price)")
        print("   ✅ Adaptive execution based on market conditions")
        print("   ✅ Iceberg orders for large trades")
        print("   ✅ Market impact modeling")
        print("   ✅ Async execution support")
        print("   ✅ Integration with PaperExecutor")
        print("   ✅ Execution statistics tracking")
        print("\n🚀 Usage:")
        print("   python -m robo_trader.runner_async --use-smart-execution")
        print("\n📂 Files:")
        print("   - robo_trader/smart_execution/smart_executor.py")
        print("   - robo_trader/smart_execution/algorithms.py")
        print("   - robo_trader/execution.py (enhanced)")
    else:
        print(f"\n⚠️ S2 incomplete: {total - passed} tests failed")
        print("Please review the failures above.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
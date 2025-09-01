"""Simple test to verify smart execution is working."""

import asyncio
from robo_trader.smart_execution import SmartExecutor, ExecutionParams, ExecutionAlgorithm
from robo_trader.config import Config

async def main():
    print("Testing Smart Execution...")
    
    # Create smart executor
    config = Config()
    executor = SmartExecutor(config)
    
    # Test each algorithm
    algorithms = [
        ExecutionAlgorithm.TWAP,
        ExecutionAlgorithm.VWAP, 
        ExecutionAlgorithm.ADAPTIVE,
        ExecutionAlgorithm.ICEBERG
    ]
    
    for algo in algorithms:
        params = ExecutionParams(algorithm=algo, duration_minutes=10)
        plan = await executor.create_execution_plan(
            symbol="AAPL",
            side="BUY", 
            quantity=1000,
            params=params
        )
        print(f"✅ {algo.value}: Created plan with {len(plan.slices)} slices")
    
    print("\n✅ All algorithms working!")

if __name__ == "__main__":
    asyncio.run(main())
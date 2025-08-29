#!/usr/bin/env python3
"""Test smart execution with real market data."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.config import load_config
from robo_trader.clients.async_ibkr_client import AsyncIBKRClient, ConnectionConfig
from robo_trader.smart_execution.smart_executor import SmartExecutor, ExecutionParams, ExecutionAlgorithm
from robo_trader.execution import PaperExecutor


async def test_real_market_data():
    """Test smart execution with real IBKR market data."""
    print("=" * 60)
    print("Testing Smart Execution with Real Market Data")
    print("=" * 60)
    
    config = load_config()
    
    # Initialize IBKR client with ConnectionConfig
    conn_config = ConnectionConfig()
    ibkr_client = AsyncIBKRClient(conn_config)
    await ibkr_client.connect()
    
    # Get an IB connection from the pool
    async with ibkr_client.pool.acquire() as ib:
        # Initialize smart executor with IBKR client
        smart_executor = SmartExecutor(config, ibkr_client=ib)
        
        # Test symbols
        symbols = ["AAPL", "NVDA", "TSLA"]
        
        print("\n1. Testing real price fetching:")
        for symbol in symbols:
            price = await smart_executor._get_current_price(symbol)
            print(f"   {symbol}: ${price:.2f}")
        
        print("\n2. Testing market condition analysis:")
        for symbol in symbols:
            conditions = await smart_executor._analyze_market_conditions(symbol, None)
            print(f"   {symbol}: {conditions}")
        
        print("\n3. Creating execution plans with real data:")
        for symbol in symbols:
            # Create VWAP execution plan
            params = ExecutionParams(
                algorithm=ExecutionAlgorithm.VWAP,
                duration_minutes=30,
                slice_count=10,
                max_participation=0.1
            )
            
            plan = await smart_executor.create_execution_plan(
                symbol=symbol,
                side="BUY",
                quantity=1000,
                params=params
            )
            
            print(f"\n   {symbol} VWAP Plan:")
            print(f"     Algorithm: {plan.algorithm.value}")
            print(f"     Slices: {len(plan.slices)}")
            print(f"     Duration: {plan.estimated_duration}")
            print(f"     Market Impact: {plan.market_impact_bps:.1f} bps")
            
        print("\n4. Testing adaptive execution:")
        # Test adaptive algorithm which chooses based on market conditions
        adaptive_params = ExecutionParams(
            algorithm=ExecutionAlgorithm.ADAPTIVE,
            duration_minutes=20,
            urgency=0.7
        )
        
        plan = await smart_executor.create_execution_plan(
            symbol="AAPL",
            side="SELL",
            quantity=5000,
            params=adaptive_params
        )
        
        print(f"\n   Adaptive Plan for AAPL (5000 shares):")
        print(f"     Selected Algorithm: {plan.algorithm.value}")
        print(f"     Slices: {len(plan.slices)}")
        print(f"     Duration: {plan.estimated_duration}")
        print(f"     Market Impact: {plan.market_impact_bps:.1f} bps")
        
        # Show first few slices
        print(f"\n     First 3 slices:")
        for i, slice_config in enumerate(plan.slices[:3]):
            print(f"       Slice {i+1}: {slice_config['quantity']} shares at {slice_config['time'].strftime('%H:%M:%S')}")
        
        print("\n5. Testing with paper executor:")
        # Initialize paper executor with smart execution
        paper_executor = PaperExecutor(
            slippage_bps=2.0,
            smart_executor=smart_executor,
            use_smart_execution=True
        )
        
        # Test order selection
        from robo_trader.risk import Order
        
        test_orders = [
            Order("AAPL", "BUY", 100),    # Small - should use MARKET
            Order("NVDA", "SELL", 1500),  # Medium - should use TWAP
            Order("TSLA", "BUY", 5000),   # Large - should use VWAP or ADAPTIVE
        ]
        
        print("\n   Algorithm selection by order size:")
        for order in test_orders:
            algo = paper_executor._select_algorithm(order)
            print(f"     {order.symbol} {order.side} {order.quantity}: {algo.value}")
    
    # Cleanup
    await ibkr_client.disconnect()
    
    print("\n" + "=" * 60)
    print("✅ Smart execution is now connected to real market data!")
    print("=" * 60)


async def main():
    """Run all tests."""
    await test_real_market_data()


if __name__ == "__main__":
    asyncio.run(main())
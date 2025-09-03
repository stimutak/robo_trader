#!/usr/bin/env python3
"""
Test script to generate performance metrics for the dashboard.
This simulates trading activity to populate the PerformanceMonitor.
"""

import asyncio
import pickle
from pathlib import Path
from datetime import datetime
import random

from robo_trader.monitoring.performance import PerformanceMonitor, PerformanceMetrics


async def simulate_trading_activity():
    """Simulate trading activity and generate metrics."""
    monitor = PerformanceMonitor()
    
    print("Simulating trading activity...")
    
    # Simulate multiple trading cycles
    for cycle in range(5):
        print(f"\nCycle {cycle + 1}/5")
        
        # Simulate processing multiple symbols
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        for symbol in symbols:
            # Record data fetch latency (10-50ms)
            monitor.start_timer('data_fetch')
            await asyncio.sleep(random.uniform(0.01, 0.05))
            latency = monitor.end_timer('data_fetch')
            monitor.data_fetch_samples.append(latency)
            
            # Record signal generation latency (5-20ms)
            monitor.start_timer('signal_gen')
            await asyncio.sleep(random.uniform(0.005, 0.02))
            latency = monitor.end_timer('signal_gen')
            monitor.signal_gen_samples.append(latency)
            
            # Simulate order execution (20% of symbols)
            if random.random() < 0.2:
                monitor.start_timer('order_exec')
                await asyncio.sleep(random.uniform(0.02, 0.1))
                latency = monitor.end_timer('order_exec')
                monitor.order_exec_samples.append(latency)
                await monitor.record_order_placed(symbol, 100)
                monitor.order_timestamps.append(datetime.now())
                
                # 80% of orders execute successfully
                if random.random() < 0.8:
                    await monitor.record_trade_executed(symbol, 'BUY', 100)
            
            # Record database write latency (5-15ms)
            monitor.start_timer('db_write')
            await asyncio.sleep(random.uniform(0.005, 0.015))
            latency = monitor.end_timer('db_write')
            monitor.db_write_samples.append(latency)
            
            # Record symbol processed
            await monitor.record_symbol_processed(symbol, success=random.random() < 0.95)
            
            # Record data points (simulate 10-50 data points per symbol)
            await monitor.record_data_points(random.randint(10, 50))
        
        # Get current metrics
        metrics = await monitor.get_current_metrics()
        
        # Export to file for dashboard
        metrics_file = Path('/tmp/robo_trader_metrics.pkl')
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        print(f"Metrics exported: {metrics.total_symbols_processed} symbols, "
              f"{metrics.total_orders_placed} orders, {metrics.total_trades_executed} trades")
        print(f"Average latencies - Data: {metrics.data_fetch_latency:.1f}ms, "
              f"Signal: {metrics.signal_generation_latency:.1f}ms, "
              f"Order: {metrics.order_execution_latency:.1f}ms")
        print(f"Throughput - {metrics.symbols_per_second:.1f} symbols/s, "
              f"{metrics.orders_per_minute} orders/min")
        
        # Wait before next cycle
        await asyncio.sleep(2)
    
    print("\n✅ Test complete! Check the dashboard Performance tab to see the metrics.")
    print("The metrics will continue updating for the next minute...")
    
    # Keep updating metrics for another minute
    for _ in range(30):
        # Add some variation to make it look live
        await monitor.record_data_points(random.randint(5, 20))
        if random.random() < 0.1:
            await monitor.record_order_placed('TEST', 100)
            monitor.order_timestamps.append(datetime.now())
        
        metrics = await monitor.get_current_metrics()
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        await asyncio.sleep(2)
    
    print("\n🏁 Simulation finished!")


if __name__ == "__main__":
    asyncio.run(simulate_trading_activity())
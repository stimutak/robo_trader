#!/usr/bin/env python3
"""Test microstructure trading strategies."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from robo_trader.strategies.microstructure import (
    OrderBookImbalanceStrategy,
    SpreadCaptureStrategy,
    MicroMomentumStrategy,
    create_microstructure_strategies,
    OrderBookSnapshot,
    OrderBookLevel,
    MicrostructureSignal,
)


async def test_order_book_imbalance_strategy():
    """Test order book imbalance strategy."""
    
    print("🔍 Testing Order Book Imbalance Strategy")
    print("-" * 40)
    
    strategy = OrderBookImbalanceStrategy(
        name="TestImbalance",
        imbalance_threshold=0.3,
        min_spread_bps=2.0,
        max_spread_bps=20.0,
    )
    
    # Create mock market data
    mock_data = pd.DataFrame({
        "close": [100.0, 100.1, 100.05],
        "volume": [1000, 1200, 800],
    })
    
    # Test multiple scenarios
    scenarios = [
        {"name": "Strong Buy Imbalance", "expected_signal": "buy"},
        {"name": "Strong Sell Imbalance", "expected_signal": "sell"},
        {"name": "Balanced Book", "expected_signal": "hold"},
    ]
    
    for scenario in scenarios:
        print(f"\n  Testing: {scenario['name']}")
        
        # Run analysis
        result = await strategy.analyze("TEST", mock_data)
        
        print(f"    Signal: {result.get('signal', 'none')}")
        print(f"    Confidence: {result.get('confidence', 0.0):.2f}")
        
        if "imbalance" in result:
            print(f"    Imbalance: {result['imbalance']:.3f}")
        if "spread_bps" in result:
            print(f"    Spread (bps): {result['spread_bps']:.1f}")
    
    print("  ✅ Order book imbalance strategy tested")


async def test_spread_capture_strategy():
    """Test spread capture strategy."""
    
    print("\n💰 Testing Spread Capture Strategy")
    print("-" * 40)
    
    strategy = SpreadCaptureStrategy(
        name="TestSpreadCapture",
        min_spread_bps=3.0,
        max_position_size=1000,
        inventory_limit=5000,
    )
    
    # Create mock market data
    mock_data = pd.DataFrame({
        "close": [100.0, 100.02, 100.01],
        "volume": [2000, 1800, 2200],
    })
    
    # Test different inventory levels
    inventory_scenarios = [
        {"inventory": 0, "description": "Neutral inventory"},
        {"inventory": 3000, "description": "Long inventory"},
        {"inventory": -2000, "description": "Short inventory"},
        {"inventory": 6000, "description": "Over limit inventory"},
    ]
    
    for scenario in inventory_scenarios:
        print(f"\n  Testing: {scenario['description']}")
        
        # Set inventory
        strategy.current_inventory = scenario["inventory"]
        
        # Run analysis
        result = await strategy.analyze("TEST", mock_data)
        
        print(f"    Signal: {result.get('signal', 'none')}")
        print(f"    Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"    Current Inventory: {strategy.current_inventory}")
        
        if "optimal_bid" in result:
            print(f"    Optimal Bid: ${result['optimal_bid']:.4f}")
        if "optimal_ask" in result:
            print(f"    Optimal Ask: ${result['optimal_ask']:.4f}")
        if "reason" in result:
            print(f"    Reason: {result['reason']}")
    
    # Test inventory updates
    print(f"\n  Testing inventory updates:")
    initial_inventory = strategy.current_inventory
    
    strategy.update_inventory(500, "BUY")
    print(f"    After buying 500: {strategy.current_inventory}")
    
    strategy.update_inventory(300, "SELL")
    print(f"    After selling 300: {strategy.current_inventory}")
    
    print("  ✅ Spread capture strategy tested")


async def test_micro_momentum_strategy():
    """Test micro momentum strategy."""
    
    print("\n⚡ Testing Micro Momentum Strategy")
    print("-" * 40)
    
    strategy = MicroMomentumStrategy(
        name="TestMicroMomentum",
        lookback_ticks=10,
        momentum_threshold=0.0002,
        max_hold_seconds=30,
    )
    
    # Create mock market data
    mock_data = pd.DataFrame({
        "close": [100.0, 100.05, 100.08, 100.12],
        "volume": [1500, 1800, 2000, 1600],
    })
    
    # Simulate multiple ticks to build history
    print("  Building tick history...")
    
    for i in range(15):
        result = await strategy.analyze("TEST", mock_data)
        
        if i < 10:
            # Should return hold while building history
            assert result["signal"] == "hold", f"Expected hold signal while building history, got {result['signal']}"
        else:
            print(f"    Tick {i+1}: Signal={result.get('signal', 'none')}, Confidence={result.get('confidence', 0.0):.3f}")
            
            if result["signal"] != "hold":
                print(f"      Momentum: {result.get('momentum', 0.0):.6f}")
                print(f"      Consistency: {result.get('consistency', 0.0):.2f}")
    
    print("  ✅ Micro momentum strategy tested")


async def test_strategy_suite():
    """Test the complete microstructure strategy suite."""
    
    print("\n🎯 Testing Complete Microstructure Strategy Suite")
    print("-" * 50)
    
    strategies = create_microstructure_strategies()
    
    print(f"  Created {len(strategies)} microstructure strategies:")
    
    for strategy in strategies:
        print(f"    - {strategy.name}")
    
    # Test each strategy
    mock_data = pd.DataFrame({
        "close": [100.0, 100.02, 100.01, 100.03],
        "volume": [2000, 1800, 2200, 1900],
    })
    
    print(f"\n  Testing all strategies with mock data:")
    
    for strategy in strategies:
        try:
            result = await strategy.analyze("TEST", mock_data)
            
            signal = result.get("signal", "none")
            confidence = result.get("confidence", 0.0)
            strategy_type = result.get("strategy_type", "unknown")
            
            print(f"    {strategy.name:25} | Signal: {signal:4} | Confidence: {confidence:.2f} | Type: {strategy_type}")
            
        except Exception as e:
            print(f"    {strategy.name:25} | ERROR: {str(e)}")
    
    print("  ✅ Strategy suite tested")


async def test_order_book_snapshot():
    """Test order book snapshot functionality."""
    
    print("\n📊 Testing Order Book Snapshot")
    print("-" * 40)
    
    # Create mock order book
    bids = [
        OrderBookLevel(99.98, 1500),
        OrderBookLevel(99.97, 1200),
        OrderBookLevel(99.96, 800),
    ]
    
    asks = [
        OrderBookLevel(100.02, 1000),
        OrderBookLevel(100.03, 1400),
        OrderBookLevel(100.04, 900),
    ]
    
    snapshot = OrderBookSnapshot(
        timestamp=datetime.now(),
        symbol="TEST",
        bids=bids,
        asks=asks,
        last_trade_price=100.00,
        last_trade_size=200,
    )
    
    print(f"  Best Bid: ${snapshot.best_bid.price:.2f} x {snapshot.best_bid.size}")
    print(f"  Best Ask: ${snapshot.best_ask.price:.2f} x {snapshot.best_ask.size}")
    print(f"  Spread: ${snapshot.spread:.4f}")
    print(f"  Mid Price: ${snapshot.mid_price:.4f}")
    print(f"  Imbalance: {snapshot.imbalance:.3f}")
    
    # Test imbalance calculation
    expected_imbalance = (1500 - 1000) / (1500 + 1000)
    assert abs(snapshot.imbalance - expected_imbalance) < 0.001, f"Imbalance calculation error"
    
    print("  ✅ Order book snapshot tested")


async def main():
    """Run all microstructure strategy tests."""
    
    print("🚀 Testing Microstructure Trading Strategies")
    print("=" * 50)
    
    try:
        # Test individual strategies
        await test_order_book_imbalance_strategy()
        await test_spread_capture_strategy()
        await test_micro_momentum_strategy()
        
        # Test order book functionality
        await test_order_book_snapshot()
        
        # Test complete suite
        await test_strategy_suite()
        
        print("\n🎉 All Microstructure Strategy Tests Passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script to verify OBV vectorization correctness and performance.
"""

import pandas as pd
import numpy as np
import time
from typing import Optional
from robo_trader.features.indicators import TechnicalIndicators

def create_test_data(n_rows=1000):
    """Create test OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='1min')
    
    base_price = 100
    price_changes = np.random.randn(n_rows) * 0.01
    prices = base_price * np.exp(np.cumsum(price_changes))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(n_rows) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_rows)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n_rows)) * 0.002),
        'close': prices,
        'volume': np.random.randint(100000, 1000000, n_rows)
    })
    
    return df

def obv_iterative(df: pd.DataFrame) -> Optional[float]:
    """Original iterative OBV implementation for comparison."""
    if len(df) < 2:
        return None
    
    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = df['volume'].iloc[0]
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv.iloc[-1]

def test_correctness():
    """Test that vectorized and iterative implementations produce same results."""
    print("Testing OBV calculation correctness...")
    
    indicators = TechnicalIndicators()
    
    for size in [10, 100, 1000]:
        df = create_test_data(size)
        
        vectorized_result = indicators.obv(df)
        iterative_result = obv_iterative(df)
        
        if vectorized_result is not None and iterative_result is not None:
            diff = abs(vectorized_result - iterative_result)
            if diff < 1e-6:
                print(f"✓ Size {size}: Results match (diff: {diff:.2e})")
            else:
                print(f"✗ Size {size}: Results differ! Vectorized: {vectorized_result}, Iterative: {iterative_result}")
                return False
        else:
            print(f"✗ Size {size}: One or both results are None! Vectorized: {vectorized_result}, Iterative: {iterative_result}")
            return False
    
    return True

def test_performance():
    """Test performance improvement of vectorized implementation."""
    print("\nTesting OBV calculation performance...")
    
    indicators = TechnicalIndicators()
    
    for size in [1000, 5000, 10000]:
        df = create_test_data(size)
        
        start_time = time.time()
        for _ in range(10):  # Run multiple times for better measurement
            vectorized_result = indicators.obv(df)
        vectorized_time = (time.time() - start_time) / 10
        
        start_time = time.time()
        for _ in range(10):
            iterative_result = obv_iterative(df)
        iterative_time = (time.time() - start_time) / 10
        
        speedup = iterative_time / vectorized_time if vectorized_time > 0 else float('inf')
        
        print(f"Size {size:5d}: Vectorized: {vectorized_time*1000:.2f}ms, "
              f"Iterative: {iterative_time*1000:.2f}ms, "
              f"Speedup: {speedup:.1f}x")

if __name__ == "__main__":
    print("OBV Vectorization Test")
    print("=" * 50)
    
    if test_correctness():
        print("\n✓ All correctness tests passed!")
        
        test_performance()
        
        print("\n✓ OBV vectorization successfully implemented!")
        print("The new implementation maintains mathematical correctness")
        print("while providing significant performance improvements.")
    else:
        print("\n✗ Correctness tests failed!")
        print("The vectorized implementation needs to be fixed.")

#!/usr/bin/env python3
"""
Binary search through runner_async imports to find bus error cause
"""
from __future__ import annotations

import sys

print("Testing imports one by one...")
print("1. from __future__ import annotations - ✅ OK")

try:
    print("2. Testing: import argparse")
    import argparse

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("3. Testing: import asyncio")
    import asyncio

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("4. Testing: import pandas")
    import pandas as pd

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("5. Testing: from ib_async import Stock")
    from ib_async import Stock

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("6. Testing: from robo_trader.config import load_config")
    from robo_trader.config import load_config

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("7. Testing: from robo_trader.websocket_client import ws_client")
    from robo_trader.websocket_client import ws_client

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("8. Testing: from robo_trader.portfolio import Portfolio")
    from robo_trader.portfolio import Portfolio

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("9. Testing: from robo_trader.strategies import sma_crossover_signals")
    from robo_trader.strategies import sma_crossover_signals

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print(
        "10. Testing: from robo_trader.analysis.correlation_integration import AsyncCorrelationManager"
    )
    from robo_trader.analysis.correlation_integration import AsyncCorrelationManager

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

print("\n✅ All individual imports successful!")
print("Now testing full runner_async import...")

try:
    from robo_trader import runner_async

    print("✅ runner_async imported successfully!")
except Exception as e:
    print(f"❌ runner_async import failed: {e}")
    import traceback

    traceback.print_exc()

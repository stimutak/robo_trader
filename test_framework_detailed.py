#!/usr/bin/env python3
"""
Detailed test of framework.py to find exact bus error location
"""
from __future__ import annotations

import sys

print("Testing framework.py imports step by step...\n")

# Test 1: Basic imports
try:
    print("1. Testing: from abc import ABC, abstractmethod")
    from abc import ABC, abstractmethod

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("2. Testing: from dataclasses import dataclass, field")
    from dataclasses import dataclass, field

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("3. Testing: from datetime import datetime")
    from datetime import datetime

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("4. Testing: from enum import Enum")
    from enum import Enum

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("5. Testing: from typing import Any, Dict, List, Optional, Tuple")
    from typing import Any, Dict, List, Optional, Tuple

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("6. Testing: import numpy as np")
    import numpy as np

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("7. Testing: import pandas as pd")
    import pandas as pd

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("8. Testing: from robo_trader.features.engine import FeatureSet")
    from robo_trader.features.engine import FeatureSet

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("9. Testing: from robo_trader.logger import get_logger")
    from robo_trader.logger import get_logger

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("10. Testing: logger = get_logger(__name__)")
    logger = get_logger(__name__)
    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

print("\n✅ All individual imports successful!")
print("\nNow testing class definitions from framework.py...\n")

# Test creating the enums and dataclasses
try:
    print("11. Testing: SignalType enum definition")

    class SignalType(Enum):
        """Types of trading signals."""

        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"
        CLOSE = "close"
        SCALE_IN = "scale_in"
        SCALE_OUT = "scale_out"

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

try:
    print("12. Testing: Signal dataclass definition")

    @dataclass
    class Signal:
        """Trading signal with metadata."""

        timestamp: datetime
        symbol: str
        signal_type: SignalType
        strength: float
        quantity: Optional[int] = None
        entry_price: Optional[float] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        risk_reward_ratio: Optional[float] = None
        expected_value: Optional[float] = None

    print("   ✅ OK")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

print("\n✅ All class definitions successful!")
print("\nNow testing full framework.py import...\n")

try:
    print("13. Testing: from robo_trader.strategies.framework import Strategy")
    from robo_trader.strategies.framework import Strategy

    print("   ✅ OK - FRAMEWORK IMPORTED SUCCESSFULLY!")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)

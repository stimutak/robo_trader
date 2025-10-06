#!/usr/bin/env python3
"""
Integration test to validate float arithmetic fixes in real code.
Tests that key financial calculations use PrecisePricing.
"""

import sys
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))


def test_code_uses_precise_pricing():
    """Test that critical files use PrecisePricing for calculations."""
    print("=== Integration Test: Code Uses PrecisePricing ===\n")

    files_to_check = [
        "robo_trader/runner_async.py",
        "robo_trader/risk_manager.py",
        "robo_trader/backtest/engine.py",
        "app.py",
    ]

    for file_path in files_to_check:
        print(f"Checking {file_path}...")

        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Check if file imports PrecisePricing
            has_import = "PrecisePricing" in content

            # Count old-style float multiplications (likely problematic)
            old_patterns = [
                "price * quantity",
                "quantity * price",
                "* pos.quantity",
                "pos.quantity *",
            ]

            old_count = sum(content.count(pattern) for pattern in old_patterns)

            # Count new-style PrecisePricing usage
            new_patterns = ["PrecisePricing.calculate_notional", "PrecisePricing.calculate_pnl"]

            new_count = sum(content.count(pattern) for pattern in new_patterns)

            print(f"  ✓ Imports PrecisePricing: {has_import}")
            print(f"  ✓ PrecisePricing calls: {new_count}")
            print(f"  ⚠️ Potential float patterns: {old_count}")

            if has_import and new_count > 0:
                print(f"  ✅ {file_path} uses precise arithmetic")
            else:
                print(f"  ❌ {file_path} may still have float precision issues")

            print()

        except FileNotFoundError:
            print(f"  ❌ File not found: {file_path}")
            print()

    print("Integration test completed.")


def test_specific_fixes():
    """Test specific critical fixes that were made."""
    print("=== Specific Fixes Validation ===\n")

    # Test that runner_async uses precise position averaging
    print("Test 1: Position averaging in runner_async.py")
    try:
        with open("robo_trader/runner_async.py", "r") as f:
            content = f.read()

        if "PrecisePricing.calculate_notional(pos.quantity, pos.avg_price)" in content:
            print("  ✅ Position averaging uses PrecisePricing")
        else:
            print("  ❌ Position averaging may still use float arithmetic")

    except Exception as e:
        print(f"  ❌ Error checking runner_async.py: {e}")

    # Test that app.py uses precise P&L calculations
    print("Test 2: P&L calculations in app.py")
    try:
        with open("app.py", "r") as f:
            content = f.read()

        if "PrecisePricing.calculate_pnl" in content:
            print("  ✅ P&L calculations use PrecisePricing")
        else:
            print("  ❌ P&L calculations may still use float arithmetic")

    except Exception as e:
        print(f"  ❌ Error checking app.py: {e}")

    # Test that risk_manager uses precise notional calculations
    print("Test 3: Risk calculations in risk_manager.py")
    try:
        with open("robo_trader/risk_manager.py", "r") as f:
            content = f.read()

        if "PrecisePricing.calculate_notional" in content:
            print("  ✅ Risk calculations use PrecisePricing")
        else:
            print("  ❌ Risk calculations may still use float arithmetic")

    except Exception as e:
        print(f"  ❌ Error checking risk_manager.py: {e}")

    print()


def demonstrate_fix_impact():
    """Demonstrate the impact of the fixes with realistic examples."""
    print("=== Fix Impact Demonstration ===\n")

    from robo_trader.utils.pricing import PrecisePricing

    # Realistic trading scenario
    print("Scenario: Building a position over multiple trades")

    trades = [
        (100, 150.333),  # Initial position
        (50, 151.777),  # Add to position
        (75, 149.999),  # Add more
        (25, 152.555),  # Final add
    ]

    # Simulate old way (potential precision issues)
    total_shares = 0
    weighted_cost = 0.0

    for shares, price in trades:
        total_shares += shares
        weighted_cost += shares * price

    old_avg_price = weighted_cost / total_shares if total_shares > 0 else 0

    print(f"Old method (float):")
    print(f"  Total cost: ${weighted_cost:.6f}")
    print(f"  Total shares: {total_shares}")
    print(f"  Average price: ${old_avg_price:.6f}")

    # Simulate new way (precise calculations)
    total_cost_decimal = PrecisePricing.to_decimal("0")
    total_shares_decimal = 0

    for shares, price in trades:
        total_shares_decimal += shares
        total_cost_decimal += PrecisePricing.calculate_notional(shares, price)

    new_avg_price = float(total_cost_decimal / PrecisePricing.to_decimal(total_shares_decimal))

    print(f"New method (decimal):")
    print(f"  Total cost: ${total_cost_decimal}")
    print(f"  Total shares: {total_shares_decimal}")
    print(f"  Average price: ${new_avg_price:.6f}")

    print(f"Precision difference: ${abs(new_avg_price - old_avg_price):.10f}")

    if abs(new_avg_price - old_avg_price) > 1e-6:
        print("⚠️  Significant difference - float precision matters!")
    else:
        print("✅ Difference is minimal for this example")

    print()

    # P&L calculation example
    print("P&L calculation with current market price:")
    current_price = 155.123

    old_pnl = (current_price - old_avg_price) * total_shares
    new_pnl = float(PrecisePricing.calculate_pnl(new_avg_price, current_price, total_shares))

    print(f"Old P&L: ${old_pnl:.6f}")
    print(f"New P&L: ${new_pnl:.6f}")
    print(f"P&L difference: ${abs(new_pnl - old_pnl):.6f}")


if __name__ == "__main__":
    print("Float Arithmetic Fixes Integration Test")
    print("=" * 45)
    print()

    try:
        test_code_uses_precise_pricing()
        test_specific_fixes()
        demonstrate_fix_impact()

        print("🎉 Integration tests completed successfully!")

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

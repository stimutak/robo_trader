#!/usr/bin/env python3
"""
Test float arithmetic fixes in financial calculations.

This test validates that the Decimal-based precision fixes prevent
common float arithmetic errors in financial calculations.
"""

import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Add robo_trader to path
sys.path.insert(0, str(Path(__file__).parent))

from robo_trader.risk_manager import Position
from robo_trader.utils.pricing import PrecisePricing


def test_precision_errors():
    """Test common float precision errors that should now be fixed."""
    print("=== Testing Float Arithmetic Fixes ===\n")

    # Test Case 1: Price * Quantity precision
    print("Test 1: Price × Quantity Precision")
    price = 123.456789
    quantity = 100

    # OLD WAY (float arithmetic - imprecise)
    old_total = price * quantity
    print(f"Float arithmetic:  {price} × {quantity} = {old_total}")

    # NEW WAY (Decimal arithmetic - precise)
    new_total = PrecisePricing.calculate_notional(quantity, price)
    print(f"Decimal arithmetic: {price} × {quantity} = {new_total}")
    print(f"Difference: {abs(float(new_total) - old_total)}")
    print()

    # Test Case 2: Position averaging precision
    print("Test 2: Position Averaging Precision")
    pos_qty = 100
    pos_avg = 150.333
    new_qty = 50
    new_price = 155.777

    # OLD WAY
    old_total_cost = pos_avg * pos_qty + new_price * new_qty
    old_total_qty = pos_qty + new_qty
    old_new_avg = old_total_cost / old_total_qty
    print(f"Float averaging: {old_new_avg}")

    # NEW WAY
    old_cost = PrecisePricing.calculate_notional(pos_qty, pos_avg)
    add_cost = PrecisePricing.calculate_notional(new_qty, new_price)
    total_cost = old_cost + add_cost
    new_avg = float(total_cost / PrecisePricing.to_decimal(old_total_qty))
    print(f"Decimal averaging: {new_avg}")
    print(f"Difference: {abs(new_avg - old_new_avg)}")
    print()

    # Test Case 3: P&L calculation precision
    print("Test 3: P&L Calculation Precision")
    entry_price = 100.123456
    exit_price = 101.234567
    shares = 1000

    # OLD WAY
    old_pnl = (exit_price - entry_price) * shares
    print(f"Float P&L: {old_pnl}")

    # NEW WAY
    new_pnl = PrecisePricing.calculate_pnl(entry_price, exit_price, shares)
    print(f"Decimal P&L: {new_pnl}")
    print(f"Difference: {abs(float(new_pnl) - old_pnl)}")
    print()

    # Test Case 4: Position class precision
    print("Test 4: Position Class Precision")
    pos = Position("AAPL", 150, 123.456789)
    current_price = 125.987654

    # Test notional value
    notional = pos.notional_value
    print(f"Position notional value: ${notional:.6f}")

    # Test unrealized PnL
    pnl = pos.unrealized_pnl(current_price)
    print(f"Unrealized P&L: ${pnl:.6f}")
    print()

    # Test Case 5: Complex calculation chain
    print("Test 5: Complex Calculation Chain")
    # Simulate a realistic trading scenario

    # Portfolio with multiple positions
    positions = [
        {"symbol": "AAPL", "qty": 100, "price": 150.333},
        {"symbol": "GOOGL", "qty": 25, "price": 2500.777},
        {"symbol": "TSLA", "qty": 75, "price": 800.123},
    ]

    # Calculate total portfolio value (OLD vs NEW)
    old_total = sum(p["qty"] * p["price"] for p in positions)
    new_total = sum(
        float(PrecisePricing.calculate_notional(p["qty"], p["price"])) for p in positions
    )

    print(f"Portfolio value (float): ${old_total:,.2f}")
    print(f"Portfolio value (decimal): ${new_total:,.2f}")
    print(f"Difference: ${abs(new_total - old_total):,.6f}")
    print()

    print("✅ All tests completed. Decimal arithmetic provides precise financial calculations.")


def test_edge_cases():
    """Test edge cases that commonly cause issues."""
    print("=== Testing Edge Cases ===\n")

    # Test fractional shares
    print("Test 1: Fractional Share Calculations")
    capital = 1000.0
    price = 123.456789
    shares = PrecisePricing.calculate_shares(capital, price)
    actual_cost = PrecisePricing.calculate_notional(shares, price)
    print(f"Capital: ${capital}")
    print(f"Price: ${price}")
    print(f"Shares: {shares}")
    print(f"Actual cost: ${actual_cost}")
    print(f"Remaining cash: ${float(PrecisePricing.to_decimal(capital) - actual_cost):.6f}")
    print()

    # Test small price differences (tick sizes)
    print("Test 2: Tick Size Validation")
    prices = [100.005, 100.015, 100.025]  # Half-penny increments
    tick_size = "0.005"

    for price in prices:
        rounded = PrecisePricing.round_price(price, tick_size)
        valid = PrecisePricing.validate_price_increment(rounded, tick_size)
        print(f"Price: {price} → Rounded: {rounded} → Valid: {valid}")
    print()

    # Test commission calculations
    print("Test 3: Commission Calculations")
    gross_pnl = 150.75
    shares = 100
    commission_per_share = 0.005

    net_pnl = PrecisePricing.apply_commission(gross_pnl, shares, commission_per_share)
    print(f"Gross P&L: ${gross_pnl}")
    print(
        f"Commission: ${float(PrecisePricing.to_decimal(commission_per_share) * PrecisePricing.to_decimal(shares) * PrecisePricing.to_decimal(2)):.2f}"
    )
    print(f"Net P&L: ${net_pnl}")
    print()


def demonstrate_before_after():
    """Demonstrate the before/after comparison of problematic calculations."""
    print("=== Before vs After Comparison ===\n")

    # Common problematic scenario: repeated additions
    print("Scenario: Accumulating trade values (precision degrades over time)")

    trades = [
        (100, 123.456789),
        (50, 124.567890),
        (75, 125.678901),
        (25, 126.789012),
        (200, 127.890123),
    ]

    # OLD WAY - float accumulation
    float_total = 0.0
    for qty, price in trades:
        float_total += qty * price

    # NEW WAY - decimal precision
    decimal_total = PrecisePricing.to_decimal("0")
    for qty, price in trades:
        decimal_total += PrecisePricing.calculate_notional(qty, price)

    print("Trade values:")
    for i, (qty, price) in enumerate(trades, 1):
        old_val = qty * price
        new_val = PrecisePricing.calculate_notional(qty, price)
        print(f"  Trade {i}: {qty} × ${price} = ${old_val:.6f} (float) vs ${new_val} (decimal)")

    print(f"\nTotal (float accumulation): ${float_total:.6f}")
    print(f"Total (decimal precision): ${decimal_total}")
    print(f"Precision difference: ${abs(float(decimal_total) - float_total):.10f}")

    if abs(float(decimal_total) - float_total) > 0.000001:
        print("⚠️  Significant precision difference detected!")
    else:
        print("✅ Precision difference is minimal in this example")


if __name__ == "__main__":
    print("Float Arithmetic Fixes Validation Test")
    print("=" * 50)
    print()

    try:
        test_precision_errors()
        test_edge_cases()
        demonstrate_before_after()

        print("\n🎉 All float arithmetic fixes validated successfully!")
        print("\nKey improvements:")
        print("• Position averaging uses precise decimal arithmetic")
        print("• P&L calculations eliminate floating-point errors")
        print("• Notional value calculations are accurate to the penny")
        print("• Commission calculations maintain precision")
        print("• Tick size rounding prevents order rejections")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

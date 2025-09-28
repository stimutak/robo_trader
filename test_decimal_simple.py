#!/usr/bin/env python3
"""
Simple test for Decimal precision implementation - no external dependencies.
"""

import sys
from decimal import Decimal


def test_float_precision_bug():
    """Demonstrate the original float precision bug vs Decimal fix."""
    print("=== Float Precision Bug vs Decimal Fix ===")

    # Common trading scenario that causes precision errors
    price = 123.456789
    quantity = 0.1

    # Float calculation (problematic)
    total_float = price * quantity
    print(f"Float: {price} * {quantity} = {total_float}")
    print(f"Float repr: {total_float!r}")

    # Decimal calculation (precise)
    price_d = Decimal('123.456789')
    quantity_d = Decimal('0.1')
    total_decimal = price_d * quantity_d
    print(f"Decimal: {price_d} * {quantity_d} = {total_decimal}")

    # Show precision difference
    difference = abs(float(total_decimal) - total_float)
    print(f"Precision difference: {difference}")

    # More realistic trading scenario
    print("\n--- Realistic Trading Scenario ---")
    shares = 100
    entry_price = 123.456789
    exit_price = 125.987654

    # Float P&L (imprecise)
    pnl_float = (exit_price - entry_price) * shares
    print(f"Float P&L: ({exit_price} - {entry_price}) * {shares} = {pnl_float}")
    print(f"Float P&L repr: {pnl_float!r}")

    # Decimal P&L (precise)
    entry_d = Decimal('123.456789')
    exit_d = Decimal('125.987654')
    shares_d = Decimal('100')
    pnl_decimal = (exit_d - entry_d) * shares_d
    print(f"Decimal P&L: ({exit_d} - {entry_d}) * {shares_d} = {pnl_decimal}")

    pnl_difference = abs(float(pnl_decimal) - pnl_float)
    print(f"P&L precision difference: {pnl_difference}")

    print("✅ Precision differences demonstrated\n")


def test_pricing_utilities():
    """Test the PrecisePricing utilities."""
    print("=== Testing PrecisePricing Utilities ===")

    # Import here to test the module
    try:
        from robo_trader.utils.pricing import PrecisePricing
    except ImportError as e:
        print(f"❌ Could not import PrecisePricing: {e}")
        return False

    # Test to_decimal conversion
    test_values = [123.45, "123.45", Decimal('123.45'), 100]
    for val in test_values:
        result = PrecisePricing.to_decimal(val)
        print(f"to_decimal({val!r}) = {result}")
        assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"

    # Test price rounding
    price = PrecisePricing.round_price("123.456789", "0.01")
    expected = Decimal("123.46")
    assert price == expected, f"Expected {expected}, got {price}"
    print(f"✅ Price rounding: 123.456789 -> {price}")

    # Test share calculation
    shares = PrecisePricing.calculate_shares(10000, 123.45)
    expected_shares = 81  # int(10000 / 123.45)
    assert shares == expected_shares, f"Expected {expected_shares}, got {shares}"
    print(f"✅ Share calculation: $10000 / $123.45 = {shares} shares")

    # Test notional calculation
    notional = PrecisePricing.calculate_notional(100, "123.45")
    expected_notional = Decimal("12345.00")
    assert notional == expected_notional, f"Expected {expected_notional}, got {notional}"
    print(f"✅ Notional: 100 shares * $123.45 = ${notional}")

    # Test P&L calculation
    pnl = PrecisePricing.calculate_pnl("100.00", "105.50", 100)
    expected_pnl = Decimal("550.00")
    assert pnl == expected_pnl, f"Expected {expected_pnl}, got {pnl}"
    print(f"✅ P&L: ($105.50 - $100.00) * 100 = ${pnl}")

    print("✅ All PrecisePricing utilities working correctly\n")
    return True


def test_portfolio_basic():
    """Test basic Portfolio functionality with Decimals."""
    print("=== Testing Portfolio Decimal Implementation ===")

    try:
        from robo_trader.portfolio import Portfolio
    except ImportError as e:
        print(f"❌ Could not import Portfolio: {e}")
        return False

    # Create portfolio
    portfolio = Portfolio(10000.0)

    # Verify Decimal types
    assert isinstance(portfolio.cash, Decimal), f"Cash should be Decimal, got {type(portfolio.cash)}"
    assert isinstance(portfolio.realized_pnl, Decimal), f"Realized P&L should be Decimal, got {type(portfolio.realized_pnl)}"
    print(f"✅ Portfolio created with cash: ${portfolio.cash}")

    # Test buy order
    portfolio.update_fill("TEST", "BUY", 100, 123.456789)

    position = portfolio.positions.get("TEST")
    assert position is not None, "Position should be created"
    assert isinstance(position.avg_price, Decimal), f"Position price should be Decimal, got {type(position.avg_price)}"
    print(f"✅ Buy order: 100 shares @ ${position.avg_price}")

    # Test cash deduction
    expected_cash = Decimal('10000.0') - (Decimal('123.456789') * Decimal('100'))
    assert portfolio.cash == expected_cash, f"Expected {expected_cash}, got {portfolio.cash}"
    print(f"✅ Cash after buy: ${portfolio.cash}")

    # Test sell order
    portfolio.update_fill("TEST", "SELL", 50, 125.987654)

    # Check realized P&L is Decimal
    assert isinstance(portfolio.realized_pnl, Decimal), "Realized P&L should be Decimal"
    print(f"✅ Realized P&L after sell: ${portfolio.realized_pnl}")

    # Test unrealized P&L
    market_prices = {"TEST": 130.123456}
    unrealized = portfolio.compute_unrealized(market_prices)
    assert isinstance(unrealized, Decimal), "Unrealized P&L should be Decimal"
    print(f"✅ Unrealized P&L: ${unrealized}")

    # Test equity
    equity = portfolio.equity(market_prices)
    assert isinstance(equity, Decimal), "Equity should be Decimal"
    print(f"✅ Total equity: ${equity}")

    print("✅ Portfolio Decimal implementation working correctly\n")
    return True


def main():
    """Run all simple decimal tests."""
    print("Simple Decimal Precision Test")
    print("=" * 50)

    success = True

    try:
        test_float_precision_bug()

        if not test_pricing_utilities():
            success = False

        if not test_portfolio_basic():
            success = False

        if success:
            print("=" * 50)
            print("🎉 ALL TESTS PASSED!")
            print("✅ Decimal precision implementation working")
            return 0
        else:
            print("❌ Some tests failed")
            return 1

    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
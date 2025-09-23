#!/usr/bin/env python3
"""
Test the dashboard fixes for trading status and market hours.
"""

import json
from datetime import datetime

import pytz
import requests


def test_market_hours():
    """Test market hours API endpoint."""
    print("=" * 50)
    print("TESTING MARKET HOURS FIX")
    print("=" * 50)

    try:
        # Test the market hours module directly
        from robo_trader.market_hours import get_market_session, is_market_open

        current_et = datetime.now(pytz.timezone("US/Eastern"))
        is_open = is_market_open()
        session = get_market_session()

        print(f"Current time (ET): {current_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Is market open: {is_open}")
        print(f"Market session: {session}")

        # Expected behavior at different times
        hour = current_et.hour
        minute = current_et.minute
        time_minutes = hour * 60 + minute

        if 9 * 60 + 30 <= time_minutes < 16 * 60 + 30:  # 9:30 AM - 4:30 PM
            expected_open = True
            expected_session = "regular"
        elif 16 * 60 + 30 <= time_minutes < 20 * 60:  # 4:30 PM - 8:00 PM
            expected_open = False
            expected_session = "after-hours"
        elif 4 * 60 <= time_minutes < 9 * 60 + 30:  # 4:00 AM - 9:30 AM
            expected_open = False
            expected_session = "pre-market"
        else:
            expected_open = False
            expected_session = "closed"

        print(f"Expected open: {expected_open}")
        print(f"Expected session: {expected_session}")

        if is_open == expected_open and session == expected_session:
            print("✅ Market hours logic is CORRECT!")
            return True
        else:
            print("❌ Market hours logic is INCORRECT!")
            return False

    except Exception as e:
        print(f"❌ Market hours test failed: {e}")
        return False


def test_trading_status():
    """Test trading status API endpoint."""
    print("\n" + "=" * 50)
    print("TESTING TRADING STATUS FIX")
    print("=" * 50)

    try:
        # Test the status API endpoint
        response = requests.get("http://localhost:5555/api/status", timeout=5)

        if response.status_code == 200:
            data = response.json()
            trading_status = data.get("trading_status", {})
            is_trading = trading_status.get("is_trading", False)

            print(f"API Response Status: {response.status_code}")
            print(f"Is trading: {is_trading}")
            print(f"Connected: {trading_status.get('connected', False)}")
            print(f"Mode: {trading_status.get('mode', 'unknown')}")

            # Since we know no trading process is running, it should be False
            if not is_trading:
                print("✅ Trading status is CORRECT (stopped)!")
                return True
            else:
                print("❌ Trading status is INCORRECT (should be stopped)!")
                return False

        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Dashboard not running - start with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Trading status test failed: {e}")
        return False


def test_market_status_api():
    """Test market status API endpoint."""
    print("\n" + "=" * 50)
    print("TESTING MARKET STATUS API")
    print("=" * 50)

    try:
        response = requests.get("http://localhost:5555/api/market/status", timeout=5)

        if response.status_code == 200:
            data = response.json()

            print(f"API Response: {json.dumps(data, indent=2)}")

            is_open = data.get("is_open", False)
            session = data.get("session", "unknown")
            status_text = data.get("status_text", "unknown")

            print(f"Market open: {is_open}")
            print(f"Session: {session}")
            print(f"Status text: {status_text}")

            # This should match our direct test
            from robo_trader.market_hours import get_market_session, is_market_open

            expected_open = is_market_open()
            expected_session = get_market_session()

            if is_open == expected_open and session == expected_session:
                print("✅ Market status API is CORRECT!")
                return True
            else:
                print("❌ Market status API doesn't match direct test!")
                return False

        else:
            print(f"❌ API request failed with status {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Dashboard not running - start with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Market status API test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing dashboard fixes...")

    # Test market hours logic
    market_hours_ok = test_market_hours()

    # Test trading status
    trading_status_ok = test_trading_status()

    # Test market status API
    market_api_ok = test_market_status_api()

    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Market hours logic: {'✅ PASS' if market_hours_ok else '❌ FAIL'}")
    print(f"Trading status: {'✅ PASS' if trading_status_ok else '❌ FAIL'}")
    print(f"Market status API: {'✅ PASS' if market_api_ok else '❌ FAIL'}")

    if market_hours_ok and trading_status_ok and market_api_ok:
        print("\n🎉 ALL TESTS PASSED! Dashboard fixes are working correctly.")
        return True
    else:
        print("\n❌ Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

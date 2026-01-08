"""
Market hours utilities for checking if the stock market is open.
"""

from datetime import date, datetime, time, timedelta
from typing import Optional, Tuple

import pytz

# Early close days (1:00 PM ET) - month, day tuples
# Note: These vary by year - check NYSE calendar for specifics
EARLY_CLOSE_DAYS = [
    (7, 3),  # Day before Independence Day (if July 4 is not weekend)
    (12, 24),  # Christmas Eve (when it falls on a weekday)
    # Note: Dec 31, 2025 is NOT an early close day per NYSE calendar
    # Note: Black Friday (day after Thanksgiving) is dynamic, handled separately
]

# Full market holidays - month, day tuples (fixed dates only)
MARKET_HOLIDAYS = [
    (1, 1),  # New Year's Day
    (7, 4),  # Independence Day
    (12, 25),  # Christmas Day
    # Note: Other holidays (MLK, Presidents, Memorial, Labor, Thanksgiving) are dynamic
]


def _is_early_close_day(dt: datetime) -> bool:
    """Check if the given date is an early close day (1:00 PM ET close)."""
    month_day = (dt.month, dt.day)

    # Check fixed early close days
    if month_day in EARLY_CLOSE_DAYS:
        return True

    # Check for Black Friday (4th Friday of November)
    if dt.month == 11 and dt.weekday() == 4:  # Friday in November
        # Count Fridays in November up to this date
        first_day = dt.replace(day=1)
        friday_count = sum(1 for d in range(1, dt.day + 1) if dt.replace(day=d).weekday() == 4)
        if friday_count == 4:  # 4th Friday
            return True

    return False


def _is_market_holiday(dt: datetime) -> bool:
    """Check if the given date is a market holiday."""
    month_day = (dt.month, dt.day)

    # Check fixed holidays
    if month_day in MARKET_HOLIDAYS:
        return True

    # TODO: Add dynamic holidays (MLK, Presidents, Memorial, Labor, Thanksgiving)
    # For now, just handle the fixed ones

    return False


def _get_market_close_time(dt: datetime) -> time:
    """Get the market close time for a given date."""
    if _is_early_close_day(dt):
        return time(13, 0)  # 1:00 PM ET on early close days
    return time(16, 0)  # 4:00 PM ET normally


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """
    Check if the US stock market is currently open for regular trading.

    Args:
        dt: Datetime to check (defaults to current time)

    Returns:
        True if market is open, False otherwise
    """
    if dt is None:
        dt = datetime.now(pytz.timezone("US/Eastern"))
    elif dt.tzinfo is None:
        # Assume naive datetime is in Eastern time
        dt = pytz.timezone("US/Eastern").localize(dt)
    else:
        # Convert to Eastern time
        dt = dt.astimezone(pytz.timezone("US/Eastern"))

    # Check if it's a weekday (Monday=0, Sunday=6)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check for market holidays
    if _is_market_holiday(dt):
        return False

    # Check time (9:30 AM - close time, which varies by day)
    market_open = time(9, 30)
    market_close = _get_market_close_time(dt)
    current_time = dt.time()

    return market_open <= current_time < market_close


def is_extended_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if we're in extended trading hours (pre-market or after-hours).

    Pre-market: 4:00 AM - 9:30 AM ET
    After-hours: 4:00 PM - 8:00 PM ET

    Args:
        dt: Datetime to check (defaults to current time)

    Returns:
        True if in extended hours, False otherwise
    """
    if dt is None:
        dt = datetime.now(pytz.timezone("US/Eastern"))
    elif dt.tzinfo is None:
        dt = pytz.timezone("US/Eastern").localize(dt)
    else:
        dt = dt.astimezone(pytz.timezone("US/Eastern"))

    # Check if it's a weekday
    if dt.weekday() >= 5:  # Weekend
        return False

    # Check for market holidays
    if _is_market_holiday(dt):
        return False

    current_time = dt.time()

    # Pre-market: 4:00 AM - 9:30 AM
    pre_market_start = time(4, 0)
    pre_market_end = time(9, 30)

    # After-hours: starts when regular hours end (varies by day)
    after_hours_start = _get_market_close_time(dt)
    after_hours_end = time(20, 0)

    return (
        pre_market_start <= current_time < pre_market_end
        or after_hours_start <= current_time < after_hours_end
    )


def get_market_session(dt: Optional[datetime] = None) -> str:
    """
    Get the current market session.

    Args:
        dt: Datetime to check (defaults to current time)

    Returns:
        One of: "regular", "pre-market", "after-hours", "closed"
    """
    if dt is None:
        dt = datetime.now(pytz.timezone("US/Eastern"))
    elif dt.tzinfo is None:
        dt = pytz.timezone("US/Eastern").localize(dt)
    else:
        dt = dt.astimezone(pytz.timezone("US/Eastern"))

    # Check weekend
    if dt.weekday() >= 5:
        return "closed"

    # Check for market holidays
    if _is_market_holiday(dt):
        return "closed"

    current_time = dt.time()

    # Define market hours (close time varies by day)
    pre_market_start = time(4, 0)
    regular_start = time(9, 30)
    regular_end = _get_market_close_time(dt)  # 1:00 PM on early close days, 4:00 PM otherwise
    after_hours_end = time(20, 0)

    if pre_market_start <= current_time < regular_start:
        return "pre-market"
    elif regular_start <= current_time < regular_end:
        return "regular"
    elif regular_end <= current_time < after_hours_end:
        return "after-hours"
    else:
        return "closed"


def get_next_market_open() -> datetime:
    """
    Get the next market open time.

    Returns:
        DateTime of next market open (9:30 AM ET on next trading day)
    """
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    # Start with today at 9:30 AM
    next_open = now.replace(hour=9, minute=30, second=0, microsecond=0)

    # If it's already past 9:30 AM today, move to next day
    if now.time() >= time(9, 30):
        next_open = next_open + timedelta(days=1)

    # Skip weekends by advancing day-by-day
    while next_open.weekday() >= 5:  # Saturday = 5, Sunday = 6
        next_open = next_open + timedelta(days=1)

    return next_open


def seconds_until_market_open() -> int:
    """
    Get seconds until next market open.

    Returns:
        Number of seconds until market opens (0 if market is open)
    """
    if is_market_open():
        return 0

    next_open = get_next_market_open()
    eastern = pytz.timezone("US/Eastern")
    now = datetime.now(eastern)

    delta = next_open - now
    return int(delta.total_seconds())

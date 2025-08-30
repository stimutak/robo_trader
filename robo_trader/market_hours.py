"""
Market hours utilities for checking if the stock market is open.
"""

from datetime import datetime, time
from typing import Optional, Tuple

import pytz


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

    # Check time (9:30 AM - 4:00 PM Eastern)
    market_open = time(9, 30)
    market_close = time(16, 0)
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

    current_time = dt.time()

    # Pre-market: 4:00 AM - 9:30 AM
    pre_market_start = time(4, 0)
    pre_market_end = time(9, 30)

    # After-hours: 4:00 PM - 8:00 PM
    after_hours_start = time(16, 0)
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

    current_time = dt.time()

    # Define market hours
    pre_market_start = time(4, 0)
    regular_start = time(9, 30)
    regular_end = time(16, 0)
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

    # If it's already past 9:30 AM today, move to tomorrow
    if now.time() >= time(9, 30):
        next_open = next_open.replace(day=next_open.day + 1)

    # Skip weekends
    while next_open.weekday() >= 5:  # Saturday = 5, Sunday = 6
        next_open = next_open.replace(day=next_open.day + 1)

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

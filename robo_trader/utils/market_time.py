"""
Market Time Utilities - Fix for Critical Bug #7: Timezone Confusion

Provides timezone-aware datetime utilities for trading operations.
Ensures all market operations use correct Eastern Time (ET) timezone.
"""

from datetime import datetime, time, timedelta
from typing import Optional

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    import pytz  # Fallback for older Python

    class ZoneInfo:
        def __init__(self, zone_name):
            self.zone = pytz.timezone(zone_name)

        def __call__(self):
            return self.zone


# Market timezone (Eastern Time)
MARKET_TZ = ZoneInfo("America/New_York")

# Market hours
MARKET_OPEN_TIME = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET
PRE_MARKET_OPEN = time(4, 0)  # 4:00 AM ET
AFTER_HOURS_CLOSE = time(20, 0)  # 8:00 PM ET


def get_market_time() -> datetime:
    """Get current market time (Eastern Time)."""
    if hasattr(MARKET_TZ, "zone"):
        # pytz fallback
        return datetime.now(MARKET_TZ.zone)
    else:
        # zoneinfo
        return datetime.now(MARKET_TZ)


def get_market_date() -> str:
    """Get current market date as YYYY-MM-DD string."""
    return get_market_time().strftime("%Y-%m-%d")


def is_market_open(check_time: Optional[datetime] = None) -> bool:
    """Check if market is open at given time (or current time)."""
    if check_time is None:
        check_time = get_market_time()

    # Ensure we have timezone-aware datetime
    if check_time.tzinfo is None:
        if hasattr(MARKET_TZ, "zone"):
            check_time = MARKET_TZ.zone.localize(check_time)
        else:
            check_time = check_time.replace(tzinfo=MARKET_TZ)

    # Convert to market timezone if needed
    if hasattr(MARKET_TZ, "zone"):
        market_time = check_time.astimezone(MARKET_TZ.zone)
    else:
        market_time = check_time.astimezone(MARKET_TZ)

    # Check if weekday (0=Monday, 6=Sunday)
    if market_time.weekday() >= 5:  # Saturday or Sunday
        return False

    # Check market hours
    current_time = market_time.time()
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME


def is_extended_hours(check_time: Optional[datetime] = None) -> bool:
    """Check if in extended hours (pre-market or after-hours)."""
    if check_time is None:
        check_time = get_market_time()

    # Ensure timezone-aware
    if check_time.tzinfo is None:
        if hasattr(MARKET_TZ, "zone"):
            check_time = MARKET_TZ.zone.localize(check_time)
        else:
            check_time = check_time.replace(tzinfo=MARKET_TZ)

    # Convert to market timezone
    if hasattr(MARKET_TZ, "zone"):
        market_time = check_time.astimezone(MARKET_TZ.zone)
    else:
        market_time = check_time.astimezone(MARKET_TZ)

    # Check if weekday
    if market_time.weekday() >= 5:
        return False

    current_time = market_time.time()

    # Pre-market: 4:00 AM - 9:30 AM ET
    pre_market = PRE_MARKET_OPEN <= current_time < MARKET_OPEN_TIME

    # After-hours: 4:00 PM - 8:00 PM ET
    after_hours = MARKET_CLOSE_TIME < current_time <= AFTER_HOURS_CLOSE

    return pre_market or after_hours


def is_near_close(minutes_before: int = 30, check_time: Optional[datetime] = None) -> bool:
    """Check if near market close."""
    if check_time is None:
        check_time = get_market_time()

    # Ensure timezone-aware
    if check_time.tzinfo is None:
        if hasattr(MARKET_TZ, "zone"):
            check_time = MARKET_TZ.zone.localize(check_time)
        else:
            check_time = check_time.replace(tzinfo=MARKET_TZ)

    # Convert to market timezone
    if hasattr(MARKET_TZ, "zone"):
        market_time = check_time.astimezone(MARKET_TZ.zone)
    else:
        market_time = check_time.astimezone(MARKET_TZ)

    if not is_market_open(market_time):
        return False

    # Calculate time until close
    close_time = market_time.replace(
        hour=MARKET_CLOSE_TIME.hour, minute=MARKET_CLOSE_TIME.minute, second=0, microsecond=0
    )

    time_to_close = (close_time - market_time).total_seconds() / 60
    return 0 <= time_to_close <= minutes_before


def seconds_until_market_open(check_time: Optional[datetime] = None) -> int:
    """Get seconds until next market open."""
    if check_time is None:
        check_time = get_market_time()

    # Ensure timezone-aware
    if check_time.tzinfo is None:
        if hasattr(MARKET_TZ, "zone"):
            check_time = MARKET_TZ.zone.localize(check_time)
        else:
            check_time = check_time.replace(tzinfo=MARKET_TZ)

    # Convert to market timezone
    if hasattr(MARKET_TZ, "zone"):
        market_time = check_time.astimezone(MARKET_TZ.zone)
    else:
        market_time = check_time.astimezone(MARKET_TZ)

    if is_market_open(market_time):
        return 0

    # Calculate next market open starting from today at open time
    next_open = market_time.replace(
        hour=MARKET_OPEN_TIME.hour, minute=MARKET_OPEN_TIME.minute, second=0, microsecond=0
    )

    # Advance to the next valid trading day if today's open has passed or it's a weekend
    if market_time.time() >= MARKET_OPEN_TIME or market_time.weekday() >= 5:
        next_open += timedelta(days=1)

    # Skip weekends safely using timedelta arithmetic
    while next_open.weekday() >= 5:
        next_open += timedelta(days=1)

    return int((next_open - market_time).total_seconds())


def format_market_time(dt: Optional[datetime] = None, include_tz: bool = True) -> str:
    """Format datetime for market time display."""
    if dt is None:
        dt = get_market_time()

    # Ensure timezone-aware
    if dt.tzinfo is None:
        if hasattr(MARKET_TZ, "zone"):
            dt = MARKET_TZ.zone.localize(dt)
        else:
            dt = dt.replace(tzinfo=MARKET_TZ)

    if include_tz:
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def to_market_time(dt: datetime) -> datetime:
    """Convert any datetime to market timezone."""
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)

    if hasattr(MARKET_TZ, "zone"):
        return dt.astimezone(MARKET_TZ.zone)
    else:
        return dt.astimezone(MARKET_TZ)


# Replacement for datetime.now() - use this instead!
now = get_market_time

# Export commonly used functions
__all__ = [
    "get_market_time",
    "get_market_date",
    "is_market_open",
    "is_extended_hours",
    "is_near_close",
    "seconds_until_market_open",
    "format_market_time",
    "to_market_time",
    "now",
    "MARKET_TZ",
    "MARKET_OPEN_TIME",
    "MARKET_CLOSE_TIME",
]

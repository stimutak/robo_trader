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
]


def _get_nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """
    Get the nth occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)
        n: Which occurrence (1=first, 2=second, etc.)

    Returns:
        Date of the nth weekday in the month
    """
    first_day = date(year, month, 1)
    # Find first occurrence of the weekday
    days_until_weekday = (weekday - first_day.weekday()) % 7
    first_occurrence = first_day + timedelta(days=days_until_weekday)
    # Add (n-1) weeks to get nth occurrence
    return first_occurrence + timedelta(weeks=n - 1)


def _get_last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """
    Get the last occurrence of a weekday in a month.

    Args:
        year: Year
        month: Month (1-12)
        weekday: Day of week (0=Monday, 6=Sunday)

    Returns:
        Date of the last weekday in the month
    """
    # Start from last day of month
    if month == 12:
        last_day = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)

    # Find the last occurrence of the weekday
    days_since_weekday = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=days_since_weekday)


def _get_easter_sunday(year: int) -> date:
    """
    Calculate Easter Sunday using the Anonymous Gregorian algorithm.

    Args:
        year: Year

    Returns:
        Date of Easter Sunday
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    el = (32 + 2 * e + 2 * i - h - k) % 7  # 'el' instead of 'l' to avoid E741
    m = (a + 11 * h + 22 * el) // 451
    month = (h + el - 7 * m + 114) // 31
    day = ((h + el - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _get_good_friday(year: int) -> date:
    """Get Good Friday (2 days before Easter Sunday)."""
    easter = _get_easter_sunday(year)
    return easter - timedelta(days=2)


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
    """Check if the given date is a market holiday.

    NYSE holidays:
    - New Year's Day (Jan 1, or observed)
    - MLK Day (3rd Monday of January)
    - Presidents Day (3rd Monday of February)
    - Good Friday (Friday before Easter)
    - Memorial Day (last Monday of May)
    - Juneteenth (June 19, or observed) - added in 2021
    - Independence Day (July 4, or observed)
    - Labor Day (1st Monday of September)
    - Thanksgiving (4th Thursday of November)
    - Christmas (Dec 25, or observed)
    """
    check_date = dt.date() if isinstance(dt, datetime) else dt
    year = check_date.year
    month_day = (check_date.month, check_date.day)

    # Check fixed holidays (with weekend observation rules)
    # New Year's Day
    new_years = date(year, 1, 1)
    if new_years.weekday() == 6:  # Sunday -> observe Monday
        new_years = date(year, 1, 2)
    elif new_years.weekday() == 5:  # Saturday -> observe Friday (prev year)
        new_years = date(year - 1, 12, 31)
    if check_date == new_years:
        return True

    # Independence Day (July 4)
    july_4 = date(year, 7, 4)
    if july_4.weekday() == 6:  # Sunday -> observe Monday
        july_4 = date(year, 7, 5)
    elif july_4.weekday() == 5:  # Saturday -> observe Friday
        july_4 = date(year, 7, 3)
    if check_date == july_4:
        return True

    # Juneteenth (June 19) - NYSE holiday since 2021
    if year >= 2021:
        juneteenth = date(year, 6, 19)
        if juneteenth.weekday() == 6:  # Sunday -> observe Monday
            juneteenth = date(year, 6, 20)
        elif juneteenth.weekday() == 5:  # Saturday -> observe Friday
            juneteenth = date(year, 6, 18)
        if check_date == juneteenth:
            return True

    # Christmas (Dec 25)
    christmas = date(year, 12, 25)
    if christmas.weekday() == 6:  # Sunday -> observe Monday
        christmas = date(year, 12, 26)
    elif christmas.weekday() == 5:  # Saturday -> observe Friday
        christmas = date(year, 12, 24)
    if check_date == christmas:
        return True

    # Dynamic holidays
    # MLK Day - 3rd Monday of January
    mlk_day = _get_nth_weekday_of_month(year, 1, 0, 3)  # Monday=0
    if check_date == mlk_day:
        return True

    # Presidents Day - 3rd Monday of February
    presidents_day = _get_nth_weekday_of_month(year, 2, 0, 3)
    if check_date == presidents_day:
        return True

    # Good Friday - Friday before Easter
    good_friday = _get_good_friday(year)
    if check_date == good_friday:
        return True

    # Memorial Day - Last Monday of May
    memorial_day = _get_last_weekday_of_month(year, 5, 0)
    if check_date == memorial_day:
        return True

    # Labor Day - 1st Monday of September
    labor_day = _get_nth_weekday_of_month(year, 9, 0, 1)
    if check_date == labor_day:
        return True

    # Thanksgiving - 4th Thursday of November
    thanksgiving = _get_nth_weekday_of_month(year, 11, 3, 4)  # Thursday=3
    if check_date == thanksgiving:
        return True

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

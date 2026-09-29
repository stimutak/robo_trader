"""Published NYSE session boundaries used to prove complete liquidity days."""

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from robo_trader import market_hours

ET = ZoneInfo("America/New_York")


@pytest.mark.parametrize(
    "day,expected",
    [
        (date(2024, 11, 22), True),
        (date(2024, 11, 29), False),
        (date(2025, 1, 9), False),
        (date(2021, 6, 18), True),
        (date(2022, 6, 20), False),
        (date(2027, 12, 31), True),
    ],
)
def test_published_calendar_regressions(day, expected):
    at = datetime(day.year, day.month, day.day, 14, tzinfo=ET)
    assert market_hours.is_market_open(at) is expected


@pytest.mark.parametrize(
    "day,opened,closed",
    [
        (date(2026, 3, 6), "2026-03-06T14:30:00+00:00", "2026-03-06T21:00:00+00:00"),
        (date(2026, 3, 9), "2026-03-09T13:30:00+00:00", "2026-03-09T20:00:00+00:00"),
        (date(2024, 11, 29), "2024-11-29T14:30:00+00:00", "2024-11-29T18:00:00+00:00"),
    ],
)
def test_regular_session_bounds_include_dst_and_early_close(day, opened, closed):
    assert market_hours.regular_session_bounds(day) == (
        datetime.fromisoformat(opened),
        datetime.fromisoformat(closed),
    )


@pytest.mark.parametrize("day", [date(2025, 1, 9), date(2026, 7, 3), date(2026, 9, 13)])
def test_closed_days_have_no_regular_session(day):
    assert market_hours.regular_session_bounds(day) is None


@pytest.mark.parametrize("day", [date(2023, 12, 31), date(2029, 1, 1), datetime.now(timezone.utc)])
def test_risk_calendar_rejects_unverified_years_and_non_dates(day):
    with pytest.raises(ValueError):
        market_hours.regular_session_bounds(day)


# Independent dates transcribed from the NYSE notices linked in
# docs/market-calendar.md, including the separate Carter closure notice.
@pytest.mark.parametrize(
    "year,holidays,early",
    [
        (2024, "01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25", "07-03 11-29 12-24"),
        (
            2025,
            "01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
            "07-03 11-28 12-24",
        ),
        (2026, "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25", "11-27 12-24"),
        (2027, "01-01 01-18 02-15 03-26 05-31 06-18 07-05 09-06 11-25 12-24", "11-26"),
        (2028, "01-17 02-21 04-14 05-29 06-19 07-04 09-04 11-23 12-25", "07-03 11-24"),
    ],
)
def test_every_date_matches_published_annual_schedule(year, holidays, early):
    from datetime import timedelta

    closed = set(holidays.split())
    shortened = set(early.split())
    day = date(year, 1, 1)
    while day.year == year:
        bounds = market_hours.regular_session_bounds(day)
        key = day.strftime("%m-%d")
        if day.weekday() >= 5 or key in closed:
            assert bounds is None, day
        else:
            start, end = (value.astimezone(ET) for value in bounds)
            assert (start.hour, start.minute) == (9, 30), day
            assert (end.hour, end.minute) == (13 if key in shortened else 16, 0), day
        day += timedelta(days=1)

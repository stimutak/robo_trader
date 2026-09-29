"""Complete regular sessions cannot be inferred from partial bar coverage."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from robo_trader.market_hours import regular_session_bounds
from robo_trader.market_data_contract import CanonicalBar, CanonicalBarBatch, MarketSession
from robo_trader.risk.completed_sessions import completed_regular_sessions
from tests.risk.test_canonical_correlation import batch as base_batch


def history(days, now):
    original = base_batch()
    contract = replace(original.contract, timeframe="30 mins", retrieval_time=now, broker_time=now)
    bars = []
    for day in days:
        start, end = regular_session_bounds(day)
        while start < end:
            bars.append(
                CanonicalBar(contract, start, *(Decimal("100"),) * 4, 10, MarketSession.REGULAR)
            )
            start += timedelta(minutes=30)
    return CanonicalBarBatch(contract, tuple(bars))


@pytest.mark.parametrize(
    "days,now",
    [
        ((date(2026, 3, 6),), datetime(2026, 3, 9, 15, tzinfo=timezone.utc)),
        ((date(2024, 11, 27), date(2024, 11, 29)), datetime(2024, 12, 2, 15, tzinfo=timezone.utc)),
        ((date(2025, 1, 8),), datetime(2025, 1, 10, 15, tzinfo=timezone.utc)),
    ],
)
def test_latest_complete_sessions_follow_holidays_and_dst(days, now):
    result = completed_regular_sessions(history(days, now), session_count=len(days), now=now)
    assert result.session_dates == days
    assert result.latest_close == regular_session_bounds(days[-1])[1]
    assert len(result.bars[-1]) == (7 if days[-1] == date(2024, 11, 29) else 13)


@pytest.mark.parametrize("failure", ["opening", "middle", "closing", "duplicate", "whole_day"])
def test_missing_or_duplicate_coverage_fails(failure):
    now = datetime(2026, 3, 10, 15, tzinfo=timezone.utc)
    batch = history((date(2026, 3, 6), date(2026, 3, 9)), now)
    rows = batch.bars
    if failure == "opening":
        rows = rows[1:]
    elif failure == "middle":
        rows = rows[:5] + rows[6:]
    elif failure == "closing":
        rows = rows[:-1]
    elif failure == "duplicate":
        rows = rows[:1] + rows
    else:
        rows = rows[13:]
    with pytest.raises(ValueError):
        completed_regular_sessions(replace(batch, bars=rows), session_count=2, now=now)


def test_newly_closed_session_requires_new_source_coverage():
    now = datetime(2026, 3, 9, 20, tzinfo=timezone.utc)
    batch = history((date(2026, 3, 6),), now - timedelta(seconds=1))
    with pytest.raises(ValueError):
        completed_regular_sessions(batch, session_count=1, now=now)


def test_partial_current_day_is_not_a_completed_session():
    now = datetime(2026, 3, 9, 15, tzinfo=timezone.utc)
    batch = history((date(2026, 3, 6), date(2026, 3, 9)), now)
    batch = replace(batch, bars=batch.bars[:15])
    result = completed_regular_sessions(batch, session_count=1, now=now)
    assert result.session_dates == (date(2026, 3, 6),)


def test_stale_retrieval_is_not_refreshed_by_recent_session_dates():
    now = datetime(2026, 3, 9, 15, tzinfo=timezone.utc)
    batch = history((date(2026, 3, 6),), now - timedelta(hours=2))
    with pytest.raises(ValueError):
        completed_regular_sessions(batch, session_count=1, now=now)

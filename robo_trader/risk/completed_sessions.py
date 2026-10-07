"""Prove scheduled regular-session bar coverage without inferring volume units.

This is a structural read, not producer authentication or liquidity approval.
Consumers must bind the original batch to the current broker transport and
separately verify the reported volume unit before calculating dollar liquidity.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..market_data_contract import (
    CanonicalBar,
    CanonicalBarBatch,
    HistoricalBarContract,
    MarketSession,
    MarketSessionPolicy,
    bar_interval_seconds,
    market_data_max_age_seconds,
)
from ..market_hours import RISK_CALENDAR_VERSION, regular_session_bounds


@dataclass(frozen=True, slots=True)
class CompletedRegularSessions:
    calendar_version: str
    session_dates: tuple[date, ...]
    bars: tuple[tuple[CanonicalBar, ...], ...]
    latest_close: datetime
    observed_at: datetime


def completed_regular_sessions(batch, *, session_count, now) -> CompletedRegularSessions:
    """Require every interval in the latest requested number of completed sessions.

    Missing days and opening/closing bars fail; a partially completed current
    session never substitutes for an earlier full day. Only timeframes that
    divide every selected regular session exactly are supported.
    """
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("session coverage clock must be aware")
    if type(session_count) is not int or not 1 <= session_count <= 60:
        raise ValueError("session coverage count must be 1..60")
    if (
        type(batch) is not CanonicalBarBatch
        or type(batch.contract) is not HistoricalBarContract
        or type(batch.bars) is not tuple
    ):
        raise ValueError("session coverage requires exact canonical bars")
    batch.__post_init__()
    contract = batch.contract
    contract.__post_init__()
    if contract.session_policy is not MarketSessionPolicy.REGULAR_ONLY:
        raise ValueError("session coverage requires regular-only source data")
    interval = timedelta(seconds=bar_interval_seconds(contract.timeframe))
    max_age = timedelta(seconds=market_data_max_age_seconds(int(interval.total_seconds())))
    if not timedelta(0) <= now - contract.retrieval_time <= max_age:
        raise ValueError("session coverage retrieval is stale")
    eastern = ZoneInfo("America/New_York")
    day = now.astimezone(eastern).date()
    sessions = []
    while len(sessions) < session_count:
        bounds = regular_session_bounds(day)
        if bounds is not None and bounds[1] <= now:
            sessions.append((day, bounds))
        day -= timedelta(days=1)
    sessions.reverse()
    required_dates = {day for day, _ in sessions}
    by_day = {day: [] for day in required_dates}
    previous = None
    for bar in batch.bars:
        if type(bar) is not CanonicalBar:
            raise ValueError("session coverage bar is malformed")
        bar.__post_init__()
        if bar.timestamp > now or (previous is not None and bar.timestamp <= previous):
            raise ValueError("session coverage timestamps are invalid")
        previous = bar.timestamp
        if bar.session is not MarketSession.REGULAR:
            raise ValueError("session coverage contains nonregular bars")
        bar_day = bar.timestamp.astimezone(eastern).date()
        if bar_day in by_day:
            by_day[bar_day].append(bar)
    groups = []
    for day, (opened, closed) in sessions:
        if closed > min(contract.retrieval_time, contract.broker_time):
            raise ValueError("session coverage source predates a completed session")
        count, remainder = divmod(closed - opened, interval)
        if remainder:
            raise ValueError("session coverage interval does not divide the session")
        expected = tuple(opened + i * interval for i in range(count))
        rows = tuple(by_day[day])
        if tuple(bar.timestamp for bar in rows) != expected:
            raise ValueError("session coverage has missing, extra or misaligned bars")
        groups.append(rows)
    return CompletedRegularSessions(
        RISK_CALENDAR_VERSION,
        tuple(day for day, _ in sessions),
        tuple(groups),
        sessions[-1][1][1],
        contract.retrieval_time,
    )

"""Non-authorizing exact correlation statistics from canonical intraday bars.

Callers own transport provenance, expected account coverage and window policy.
This module never mints risk evidence or treats a constructed batch as authority.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
from math import isqrt
from zoneinfo import ZoneInfo

from ..market_data_contract import (
    CanonicalBar,
    CanonicalBarBatch,
    HistoricalBarContract,
    bar_interval_seconds,
    market_data_max_age_seconds,
)


@dataclass(frozen=True, slots=True)
class CanonicalCorrelation:
    max_absolute_correlation: Decimal
    compared_symbols: tuple[str, ...]
    return_count: int
    source_data_version: str
    observed_at: datetime
    window_end: datetime


def _upper_correlation(left, right):
    count = len(left)
    covariance = count * sum(a * b for a, b in zip(left, right)) - sum(left) * sum(right)
    left_variance = count * sum(a * a for a in left) - sum(left) ** 2
    right_variance = count * sum(b * b for b in right) - sum(right) ** 2
    if left_variance <= 0 or right_variance <= 0:
        raise ValueError("correlation requires nonconstant returns")
    squared = covariance**2 / (left_variance * right_variance)
    scale = 10**18
    numerator, denominator = squared.numerator * scale**2, squared.denominator
    ticks = isqrt(numerator // denominator)
    if ticks * ticks * denominator < numerator:
        ticks += 1
    return Decimal((0, tuple(int(c) for c in str(ticks)), -18))


def canonical_correlation(
    *, candidate, held, expected_contracts, transport_generation, return_count, now
) -> CanonicalCorrelation:
    """Compare the last explicitly requested aligned, completed intraday returns.

    Overnight and session-boundary returns are excluded. Missing in-session bars,
    partial coverage and differing windows fail; no intersection or zero fill is
    used. Pearson's absolute coefficient is rounded upward to 18 decimal places.
    """
    if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("correlation clock must be aware")
    if type(return_count) is not int or not 2 <= return_count <= 256:
        raise ValueError("correlation return count must be 2..256")
    if type(held) is not tuple or type(expected_contracts) is not tuple:
        raise ValueError("correlation coverage must be immutable")
    expected = {}
    for identity in expected_contracts:
        if (
            type(identity) is not tuple
            or len(identity) != 3
            or type(identity[0]) is not str
            or type(identity[1]) is not int
            or identity[1] <= 0
            or type(identity[2]) is not str
            or identity[0] in expected
        ):
            raise ValueError("correlation contract coverage is ambiguous")
        expected[identity[0]] = identity[1:]
    batches = (candidate,) + held
    actual = set()
    windows = []
    fingerprints = []
    policy = None
    oldest_observation = now
    eastern = ZoneInfo("America/New_York")
    for batch in batches:
        if (
            type(batch) is not CanonicalBarBatch
            or type(batch.contract) is not HistoricalBarContract
        ):
            raise ValueError("correlation requires exact canonical batches")
        batch.__post_init__()
        contract = batch.contract
        contract.__post_init__()
        if (
            contract.symbol in actual
            or expected.get(contract.symbol) != (contract.con_id, contract.primary_exchange)
            or contract.transport_generation != transport_generation
        ):
            raise ValueError("correlation contract or generation mismatch")
        actual.add(contract.symbol)
        current_policy = (contract.timeframe, contract.session_policy, contract.adjustment_state)
        if policy is not None and policy != current_policy:
            raise ValueError("correlation bar policies disagree")
        policy = current_policy
        interval = timedelta(seconds=bar_interval_seconds(contract.timeframe))
        max_age = timedelta(seconds=market_data_max_age_seconds(int(interval.total_seconds())))
        if not timedelta(0) <= now - contract.retrieval_time <= max_age:
            raise ValueError("correlation retrieval is stale")
        oldest_observation = min(oldest_observation, contract.retrieval_time)
        completed = []
        previous = None
        for bar in batch.bars:
            if type(bar) is not CanonicalBar:
                raise ValueError("correlation bar type is invalid")
            bar.__post_init__()
            if bar.timestamp > now or (
                previous is not None and bar.timestamp <= previous.timestamp
            ):
                raise ValueError("correlation timestamps are invalid")
            if previous is not None:
                same_session = (
                    previous.session == bar.session
                    and previous.timestamp.astimezone(eastern).date()
                    == bar.timestamp.astimezone(eastern).date()
                )
                if same_session and bar.timestamp - previous.timestamp != interval:
                    raise ValueError("correlation in-session window has gaps")
                if same_session and bar.timestamp + interval <= min(
                    now, contract.retrieval_time, contract.broker_time
                ):
                    completed.append(
                        (
                            previous.timestamp,
                            bar.timestamp,
                            Fraction(bar.close) / Fraction(previous.close) - 1,
                        )
                    )
            previous = bar
        if len(completed) < return_count:
            raise ValueError("correlation window is incomplete")
        window = tuple(completed[-return_count:])
        if not timedelta(0) <= now - window[-1][1] <= max_age:
            raise ValueError("correlation completed bars are stale")
        windows.append(window)
        fingerprints.append(
            (
                contract.symbol,
                contract.con_id,
                contract.primary_exchange,
                contract.timeframe,
                contract.session_policy.value,
                contract.adjustment_state.value,
                contract.transport_generation,
                contract.retrieval_time.isoformat(),
                contract.broker_time.isoformat(),
                tuple(
                    (start.isoformat(), end.isoformat(), str(value)) for start, end, value in window
                ),
            )
        )
    if actual != set(expected):
        raise ValueError("correlation account coverage is incomplete")
    timestamps = tuple((a, b) for a, b, _ in windows[0])
    if any(tuple((a, b) for a, b, _ in window) != timestamps for window in windows[1:]):
        raise ValueError("correlation windows are not aligned")
    candidate_returns = tuple(value for _, _, value in windows[0])
    maximum = Decimal("0")
    for window in windows[1:]:
        maximum = max(
            maximum, _upper_correlation(candidate_returns, tuple(value for _, _, value in window))
        )
    version = hashlib.sha256(
        json.dumps(sorted(fingerprints), separators=(",", ":")).encode()
    ).hexdigest()
    return CanonicalCorrelation(
        maximum,
        tuple(sorted(actual - {candidate.contract.symbol})),
        return_count,
        "corr-" + version,
        oldest_observation,
        windows[0][-1][1],
    )

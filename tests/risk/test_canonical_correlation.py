"""Exact correlation uses matched completed canonical return windows."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from robo_trader.market_data_contract import canonicalize_historical_bars
from robo_trader.risk.canonical_correlation import canonical_correlation

NOW = datetime(2026, 7, 23, 15, 5, tzinfo=timezone.utc)


def batch(symbol="AAPL", con_id=1, prices=(100, 110, 99, 108), offset=0):
    return canonicalize_historical_bars(
        symbol=symbol,
        records=[
            dict(
                date=NOW - timedelta(minutes=len(prices) - i + offset),
                open=str(p),
                high=str(p),
                low=str(p),
                close=str(p),
                volume=100,
            )
            for i, p in enumerate(prices)
        ],
        lineage=SimpleNamespace(
            symbol=symbol,
            con_id=con_id,
            exchange="SMART",
            primary_exchange="NASDAQ",
            retrieval_timestamp=NOW,
            broker_timestamp=NOW,
            transport_generation="generation-1",
        ),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=NOW,
    )


def calculate(candidate=None, held=None, **kwargs):
    return canonical_correlation(
        candidate=candidate or batch(),
        held=held if held is not None else (batch("MSFT", 2, (200, 220, 198, 216)),),
        expected_contracts=(("AAPL", 1, "NASDAQ"), ("MSFT", 2, "NASDAQ")),
        transport_generation="generation-1",
        return_count=3,
        now=NOW,
        **kwargs,
    )


def test_exact_identical_returns_and_low_decimal_context():
    with localcontext() as ctx:
        ctx.prec = 2
        result = calculate()
    assert result.max_absolute_correlation == Decimal("1")
    assert result.return_count == 3
    assert result.compared_symbols == ("MSFT",)


def test_nonperfect_result_is_a_conservative_bound():
    result = calculate(held=(batch("MSFT", 2, (100, 111, 97, 109)),))
    assert Decimal("0.99") < result.max_absolute_correlation < Decimal("1")
    assert result.source_data_version != calculate().source_data_version


@pytest.mark.parametrize(
    "held",
    [
        (),
        (batch("MSFT", 2, (100, 100, 100, 100)),),
        (batch("MSFT", 2, offset=1),),
        (batch("MSFT", 99),),
        (batch("MSFT", 2), batch("MSFT", 2)),
    ],
)
def test_missing_flat_unaligned_or_ambiguous_data_fails(held):
    with pytest.raises(ValueError):
        calculate(held=held)


def test_empty_account_requires_explicit_empty_expected_coverage():
    result = canonical_correlation(
        candidate=batch(),
        held=(),
        expected_contracts=(("AAPL", 1, "NASDAQ"),),
        transport_generation="generation-1",
        return_count=3,
        now=NOW,
    )
    assert result.max_absolute_correlation == 0


def test_mutated_or_stale_batches_fail():
    candidate = batch()
    object.__setattr__(candidate.bars[0], "close", Decimal("0"))
    with pytest.raises(ValueError):
        calculate(candidate=candidate)
    with pytest.raises(ValueError):
        canonical_correlation(
            candidate=batch(),
            held=(),
            expected_contracts=(("AAPL", 1, "NASDAQ"),),
            transport_generation="generation-1",
            return_count=3,
            now=NOW + timedelta(minutes=10),
        )


def test_coefficient_rounding_is_upward_and_absolute():
    from fractions import Fraction
    from robo_trader.risk.canonical_correlation import _upper_correlation

    left = tuple(map(Fraction, (1, 0, -1)))
    right = tuple(map(Fraction, (1, -1, 0)))
    assert _upper_correlation(left, right) == Decimal("0.5")
    assert _upper_correlation(left, tuple(-x for x in left)) == 1
    right = tuple(map(Fraction, (2, -1, 0)))
    result = _upper_correlation(left, right)
    # Exact squared coefficient = 3/7. The returned lattice point is the
    # smallest 18-place Decimal at or above its irrational square root.
    assert Fraction(result) ** 2 >= Fraction(3, 7)
    assert (Fraction(result) - Fraction(1, 10**18)) ** 2 < Fraction(3, 7)


@pytest.mark.parametrize("change", ["generation", "policy", "gaps", "duplicate", "forming"])
def test_mismatched_or_incomplete_window_is_rejected(change):
    from dataclasses import replace

    other = batch("MSFT", 2)
    if change == "generation":
        object.__setattr__(other.contract, "transport_generation", "replacement")
    elif change == "policy":
        from robo_trader.market_data_contract import AdjustmentState

        object.__setattr__(other.contract, "adjustment_state", AdjustmentState.ADJUSTED)
    elif change == "gaps":
        object.__setattr__(other, "bars", other.bars[:1] + other.bars[2:])
    elif change == "duplicate":
        object.__setattr__(other, "bars", other.bars[:1] + other.bars)
    else:
        # A still-forming latest bar cannot be used to meet the required count.
        object.__setattr__(
            other,
            "bars",
            tuple(replace(b, timestamp=b.timestamp + timedelta(minutes=1)) for b in other.bars),
        )
    with pytest.raises(ValueError):
        calculate(held=(other,))


def test_session_boundary_return_is_excluded():
    from dataclasses import replace
    from robo_trader.market_data_contract import MarketSession, MarketSessionPolicy

    candidate = batch()
    for item in (candidate,):
        object.__setattr__(item.contract, "session_policy", MarketSessionPolicy.EXTENDED)
        object.__setattr__(item.contract, "use_rth", False)
        object.__setattr__(
            item, "bars", (replace(item.bars[0], session=MarketSession.PRE_MARKET),) + item.bars[1:]
        )
    with pytest.raises(ValueError, match="incomplete"):
        canonical_correlation(
            candidate=candidate,
            held=(),
            expected_contracts=(("AAPL", 1, "NASDAQ"),),
            transport_generation="generation-1",
            return_count=3,
            now=NOW,
        )

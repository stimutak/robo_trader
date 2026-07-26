"""Producer ownership and gateway-strength protective quote revalidation."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from robo_trader.protective_quote_evidence import (
    ProtectiveQuoteEvidence,
    ProtectiveQuoteSource,
    ProtectiveQuoteValidationError,
    assert_current_authoritative_protective_quote,
    assert_producer_owned_protective_quote,
)
from robo_trader.stop_loss_monitor import StopLossMonitor

NOW = datetime(2026, 7, 26, 15, 0, tzinfo=timezone.utc)


def _monitor() -> StopLossMonitor:
    monitor = StopLossMonitor(
        execute_reduction=AsyncMock(),
        risk_manager=MagicMock(),
        portfolio_id="portfolio-a",
    )
    monitor._utcnow = MagicMock(return_value=NOW)
    monitor._monotonic = MagicMock(return_value=100.0)
    return monitor


async def _live_quote(monitor: StopLossMonitor) -> ProtectiveQuoteEvidence:
    accepted = await monitor.update_price(
        "AAPL",
        187.25,
        source_timestamp=NOW,
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="generation-1",
        source_event_id="ticker-42",
    )
    assert accepted is True
    quote = monitor.get_protective_quote_evidence("AAPL")
    assert quote is not None
    return quote


def _strict(monitor: StopLossMonitor, quote: ProtectiveQuoteEvidence):
    return assert_current_authoritative_protective_quote(
        quote,
        producer=monitor,
        expected_portfolio_id="portfolio-a",
        expected_symbol="AAPL",
        expected_con_id=265598,
        expected_transport_generation="generation-1",
    )


@pytest.mark.asyncio
async def test_live_quote_is_exact_immutable_registered_and_current() -> None:
    monitor = _monitor()
    quote = await _live_quote(monitor)

    assert quote.price == Decimal("187.25")
    assert type(quote.price) is Decimal
    assert quote.source_timestamp is NOW
    assert quote.receipt_monotonic == 100.0
    assert quote.receipt_order == 1
    assert quote.source is ProtectiveQuoteSource.LIVE_BROKER
    assert quote.con_id == 265598
    assert quote.transport_generation == "generation-1"
    assert quote.source_event_id == "ticker-42"
    assert quote.quote_id.startswith("quote:v1:")
    assert len(quote.fingerprint) == 64
    assert quote.quote_id == f"quote:v1:{quote.fingerprint}"
    assert (
        hashlib.sha256(quote.canonical_payload().encode("utf-8")).hexdigest() == quote.fingerprint
    )
    assert assert_producer_owned_protective_quote(quote, producer=monitor) is quote
    assert _strict(monitor, quote) is quote
    with pytest.raises(FrozenInstanceError):
        quote.price = Decimal("1")  # type: ignore[misc]


@pytest.mark.asyncio
async def test_legacy_quote_is_representable_but_never_gateway_authoritative() -> None:
    monitor = _monitor()
    assert await monitor.update_price("AAPL", 187.25, source_timestamp=NOW)
    quote = monitor.get_protective_quote_evidence("AAPL")
    assert quote is not None
    assert quote.source is ProtectiveQuoteSource.LEGACY_CALLBACK
    assert quote.con_id is None
    assert quote.transport_generation is None
    assert assert_producer_owned_protective_quote(quote, producer=monitor) is quote
    with pytest.raises(ProtectiveQuoteValidationError, match="not gateway-authoritative"):
        _strict(monitor, quote)


@pytest.mark.asyncio
async def test_live_source_without_contract_lineage_is_rejected_before_cache_mutation() -> None:
    monitor = _monitor()
    assert not await monitor.update_price(
        "AAPL",
        187.25,
        source_timestamp=NOW,
        source=ProtectiveQuoteSource.LIVE_BROKER,
    )
    assert monitor.last_prices == {}
    assert monitor._protective_quote_evidence == {}


@pytest.mark.asyncio
async def test_copy_and_post_production_mutation_are_not_producer_owned() -> None:
    monitor = _monitor()
    quote = await _live_quote(monitor)
    copied = replace(quote)
    with pytest.raises(ProtectiveQuoteValidationError, match="not producer-owned"):
        assert_producer_owned_protective_quote(copied, producer=monitor)

    object.__setattr__(quote, "price", Decimal("188.00"))
    with pytest.raises(ProtectiveQuoteValidationError, match="changed after production"):
        assert_producer_owned_protective_quote(quote, producer=monitor)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"expected_portfolio_id": "portfolio-b"}, "portfolio"),
        ({"expected_symbol": "MSFT"}, "symbol"),
        ({"expected_con_id": 123}, "contract"),
        ({"expected_transport_generation": "generation-2"}, "transport generation"),
    ],
)
async def test_strict_revalidation_rejects_expected_identity_mismatch(
    overrides: dict[str, object],
    message: str,
) -> None:
    monitor = _monitor()
    quote = await _live_quote(monitor)
    expected = {
        "expected_portfolio_id": "portfolio-a",
        "expected_symbol": "AAPL",
        "expected_con_id": 265598,
        "expected_transport_generation": "generation-1",
    }
    expected.update(overrides)
    with pytest.raises(ProtectiveQuoteValidationError, match=message):
        assert_current_authoritative_protective_quote(
            quote,
            producer=monitor,
            **expected,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        lambda monitor: monitor.last_prices.__setitem__("AAPL", 188.0),
        lambda monitor: monitor.price_event_times.__setitem__("AAPL", NOW - timedelta(seconds=1)),
        lambda monitor: monitor.price_receipt_monotonic.__setitem__("AAPL", 99.0),
        lambda monitor: monitor.price_receipt_orders.__setitem__("AAPL", 2),
        lambda monitor: monitor._protective_quote_evidence.pop("AAPL"),
    ],
)
async def test_mutable_compatibility_cache_cannot_invalidate_latched_evidence(mutation) -> None:
    monitor = _monitor()
    quote = await _live_quote(monitor)
    mutation(monitor)
    assert _strict(monitor, quote) is quote


@pytest.mark.asyncio
async def test_newer_live_quote_does_not_invalidate_fresh_latched_crossing() -> None:
    monitor = _monitor()
    latched = await _live_quote(monitor)
    monitor._utcnow = MagicMock(return_value=NOW + timedelta(seconds=1))
    monitor._monotonic = MagicMock(return_value=101.0)
    assert await monitor.update_price(
        "AAPL",
        186.75,
        source_timestamp=NOW + timedelta(seconds=1),
        source=ProtectiveQuoteSource.LIVE_BROKER,
        con_id=265598,
        transport_generation="generation-1",
        source_event_id="ticker-43",
    )
    current = monitor.get_protective_quote_evidence("AAPL")
    assert current is not None and current is not latched
    assert _strict(monitor, latched) is latched


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wall_now", "monotonic_now", "message"),
    [
        (NOW - timedelta(microseconds=1), 100.0, "backwards|future"),
        (NOW, 99.999, "backwards|future"),
        (NOW + timedelta(seconds=11), 100.0, "stale"),
        (NOW, 111.0, "stale"),
    ],
)
async def test_strict_revalidation_rejects_future_rollback_and_staleness(
    wall_now: datetime,
    monotonic_now: float,
    message: str,
) -> None:
    monitor = _monitor()
    quote = await _live_quote(monitor)
    monitor._utcnow = MagicMock(return_value=wall_now)
    monitor._monotonic = MagicMock(return_value=monotonic_now)
    with pytest.raises(ProtectiveQuoteValidationError, match=message):
        _strict(monitor, quote)

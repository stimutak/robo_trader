from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from robo_trader.runner.data_fetcher import DataFetcher
from tests.canonical_batch_test_support import bind_test_canonical_batch
from tests.test_fractional_volume_contract import batch
from tests.test_fractional_volume_storage import database  # noqa: F401


@pytest.mark.asyncio
async def test_fetcher_stores_owned_fractional_batch_and_projects_numeric_frame(
    database, monkeypatch
):
    source = batch("10.0000000000000001")
    holder = SimpleNamespace()
    bind_test_canonical_batch(holder, source)
    monkeypatch.setattr(holder.ib, "get_canonical_historical_bars", AsyncMock(return_value=source))
    monkeypatch.setattr(
        holder.ib, "get_historical_bars", AsyncMock(side_effect=AssertionError("raw path"))
    )
    monkeypatch.setattr("robo_trader.runner.data_fetcher.is_market_open", lambda: True)
    monitor = Mock()
    monitor.end_timer.return_value = 0
    fetcher = DataFetcher(holder.ib, database, monitor)
    frame = await fetcher.fetch_and_store("AAPL")
    assert frame is not None
    assert frame["volume"].dtype.kind == "f"
    assert frame.attrs["canonical_bar_batch"] is source
    saved = await database.get_latest_market_data("AAPL")
    assert saved[0]["volume"] == "10.0000000000000001"
    assert saved[0]["schema_version"] == 2

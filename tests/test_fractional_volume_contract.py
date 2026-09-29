from dataclasses import replace
from decimal import Decimal, localcontext
import json

import pytest

from robo_trader.market_data_contract import (
    MarketDataContractError,
    canonicalize_historical_bars,
    validate_canonical_storage_row,
)
from tests.test_pr3_market_data_contract import _lineage, _record


def batch(volume, version=2):
    return canonicalize_historical_bars(
        symbol="AAPL",
        records=[_record("2026-07-23T15:01:00+00:00", volume=volume)],
        lineage=_lineage(),
        bar_size="1 min",
        use_rth=True,
        what_to_show="TRADES",
        now=_lineage().retrieval_timestamp,
        schema_version=version,
    )


@pytest.mark.parametrize(
    "raw", ["1e127", "10.0000000000000001", "1000000000000000001.25", "0", "0.0000000000000000001"]
)
def test_v2_volume_survives_json_and_storage_validation(raw):
    with localcontext() as context:
        context.prec = 6
        result = batch(raw)
        assert type(result.bars[0].volume) is Decimal
        assert result.bars[0].volume == Decimal(raw)
        row = json.loads(json.dumps(result.storage_rows()[0]))
        assert type(row["volume"]) is str
        assert row["volume_unit"] == "unknown"
        validated = validate_canonical_storage_row(row)
        assert Decimal(validated["volume"]) == Decimal(raw)
        assert result.to_frame().attrs["canonical_bar_batch"] is result
        assert result.to_frame()["volume"].dtype.kind == "f"


@pytest.mark.parametrize("value", [True, "NaN", "Infinity", "-0.25", 0.25])
def test_v2_rejects_invalid_or_already_rounded_volume(value):
    with pytest.raises(MarketDataContractError):
        batch(value)


def test_v1_retains_integer_representation_and_rejects_fraction():
    result = batch(10, version=1)
    assert type(result.bars[0].volume) is int
    assert "volume_unit" not in result.storage_rows()[0]
    validate_canonical_storage_row(result.storage_rows()[0])
    with pytest.raises(MarketDataContractError):
        batch("10.25", version=1)


def test_v2_storage_requires_exact_text_and_explicit_unknown_unit():
    row = batch("10.25").storage_rows()[0]
    for change in ({"volume": 10.25}, {"volume_unit": "shares"}, {"schema_version": 3}):
        with pytest.raises(MarketDataContractError):
            validate_canonical_storage_row({**row, **change})
    row.pop("volume_unit")
    with pytest.raises(MarketDataContractError):
        validate_canonical_storage_row(row)


def test_v2_cannot_relabel_v1_integer_object():
    result = batch(10, version=1)
    with pytest.raises(MarketDataContractError):
        replace(result.bars[0], contract=replace(result.contract, schema_version=2))


def test_v2_representation_limit_is_closed_under_serialization():
    with pytest.raises(MarketDataContractError, match="representation limits"):
        batch("1e128")

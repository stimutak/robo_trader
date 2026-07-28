"""Deterministic content-identity tests for offline backtest inputs."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import robo_trader.backtesting.provenance as provenance
from robo_trader.backtesting.provenance import (
    ContentDigest,
    canonical_json_bytes,
    digest_dataframe,
    digest_file,
    digest_file_set,
    digest_json,
    hashed_dataframe_input,
)


def test_canonical_json_is_order_independent_and_type_preserving() -> None:
    timezone = ZoneInfo("America/New_York")
    first = {
        "decimal": Decimal("1.2300"),
        "float": 1.25,
        "integer": 1,
        "time": datetime(2026, 1, 5, 9, 30, tzinfo=timezone),
    }
    second = dict(reversed(list(first.items())))

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert digest_json(first) == digest_json(second)
    assert b'"$float"' in canonical_json_bytes(first)
    assert b'"$int"' in canonical_json_bytes(first)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), Decimal("NaN")])
def test_canonical_hash_rejects_nonfinite_values(value) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        digest_json({"bad": value})


def test_canonical_hash_rejects_naive_datetimes_and_arbitrary_objects() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        digest_json({"time": datetime(2026, 1, 5)})
    with pytest.raises(TypeError, match="unsupported"):
        digest_json({"object": object()})


def test_dataframe_digest_covers_values_schema_labels_and_timezone() -> None:
    index = pd.date_range("2026-01-05 09:30", periods=2, freq="1h", tz="America/New_York")
    frame = pd.DataFrame(
        {"close": [10.0, 11.0], "volume": np.array([100, 200], dtype=np.int64)},
        index=index,
    )

    original = digest_dataframe(frame)
    assert original == digest_dataframe(frame.copy(deep=True))
    assert original != digest_dataframe(frame.assign(close=[10.0, 12.0]))
    assert original != digest_dataframe(frame.rename(columns={"close": "price"}))
    assert hashed_dataframe_input("bars-v1", frame).digest == original


def test_dataframe_digest_rejects_ambiguous_or_nonfinite_data() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0]}, index=[0, 0])
    with pytest.raises(ValueError, match="unique"):
        digest_dataframe(frame)
    with pytest.raises(ValueError, match="non-finite"):
        digest_dataframe(pd.DataFrame({"close": [float("nan")]}, index=[0]))


def test_file_and_file_set_hashes_cover_bytes_and_relative_names(tmp_path: Path) -> None:
    first = tmp_path / "a.py"
    second = tmp_path / "nested" / "b.json"
    second.parent.mkdir()
    first.write_bytes(b"print('a')\n")
    second.write_bytes(b'{"b":1}\n')

    single = digest_file(first)
    assert single.byte_length == len(first.read_bytes())
    assert digest_file_set([first, second], tmp_path) == digest_file_set([second, first], tmp_path)
    renamed = tmp_path / "renamed.py"
    renamed.write_bytes(first.read_bytes())
    assert digest_file_set([first], tmp_path) != digest_file_set([renamed], tmp_path)


def test_file_hash_rejects_symlinks_and_file_set_escape(tmp_path: Path) -> None:
    regular = tmp_path / "regular"
    regular.write_bytes(b"content")
    link = tmp_path / "link"
    link.symlink_to(regular)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"outside")
    try:
        with pytest.raises(ValueError, match="non-symlink"):
            digest_file(link)
        with pytest.raises(ValueError, match="escapes"):
            digest_file_set([outside], tmp_path)
    finally:
        outside.unlink()


def test_file_hash_rejects_symlink_swap_between_check_and_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    outside = tmp_path / "outside"
    candidate.write_bytes(b"reviewed")
    outside.write_bytes(b"different")
    real_open = provenance.os.open

    def swap_then_open(path, flags):
        candidate.unlink()
        candidate.symlink_to(outside)
        return real_open(path, flags)

    monkeypatch.setattr(provenance.os, "open", swap_then_open)

    with pytest.raises(ValueError, match="without following links"):
        digest_file(candidate)


def test_content_digest_rejects_malformed_identity() -> None:
    with pytest.raises(ValueError, match="lowercase"):
        ContentDigest("sha256", "A" * 64, 1)
    with pytest.raises(ValueError, match="only sha256"):
        ContentDigest("sha1", "a" * 64, 1)

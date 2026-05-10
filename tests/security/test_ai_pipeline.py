"""Security tests for AI/ML pipeline integrity (Section 2.D).

Covers:
- _safe_load HMAC verification (AI-H1)
- ai_analyst symbol allowlist (AI-H2)
- feature persistence path traversal (AI-M1)
- polygon provider price-range validation (AI-M2)
- news title sanitization (AI-M4)
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from robo_trader.database_validator import ValidationError
from robo_trader.ml._safe_load import sign_file, verify_file


# ---------------------------------------------------------------------------
# AI-H1: _safe_load
# ---------------------------------------------------------------------------


def test_safe_load_verify_passes_when_key_unset_and_not_required(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_SIGNING_KEY", raising=False)
    monkeypatch.setenv("MODEL_SIGNING_REQUIRED", "false")
    target = tmp_path / "model.pkl"
    target.write_bytes(b"fake-model-bytes")
    # Should not raise
    verify_file(target)


def test_safe_load_verify_raises_when_required_and_missing_sig(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_SIGNING_REQUIRED", "true")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "test-signing-key-32-chars-minimum-x")
    target = tmp_path / "model.pkl"
    target.write_bytes(b"fake-model-bytes")
    with pytest.raises(ValueError):
        verify_file(target)


def test_safe_load_verify_raises_on_mismatch(tmp_path, monkeypatch):
    target = tmp_path / "model.pkl"
    target.write_bytes(b"original-bytes")

    # Sign with one key
    monkeypatch.setenv("MODEL_SIGNING_KEY", "key-A-very-long-string-for-signing")
    monkeypatch.setenv("MODEL_SIGNING_REQUIRED", "false")
    sig_path = sign_file(target)
    assert sig_path is not None and sig_path.exists()

    # Verify with different key -> mismatch
    monkeypatch.setenv("MODEL_SIGNING_KEY", "key-B-different-and-also-long-enough")
    with pytest.raises(ValueError):
        verify_file(target)


def test_safe_load_verify_passes_when_signed_correctly(tmp_path, monkeypatch):
    target = tmp_path / "model.pkl"
    target.write_bytes(b"some-bytes")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "stable-key-for-roundtrip-test-1234")
    monkeypatch.setenv("MODEL_SIGNING_REQUIRED", "true")
    sign_file(target)
    verify_file(target)


# ---------------------------------------------------------------------------
# AI-H2: AIAnalyst symbol allowlist
# ---------------------------------------------------------------------------


def test_ai_symbol_allowlist_rejects_garbage(monkeypatch):
    """AI-returned junk symbols must be filtered out."""
    from robo_trader import ai_analyst as ai_mod

    # Build an analyst with a fake LLM client that returns a bad symbol
    analyst = ai_mod.AIAnalyst.__new__(ai_mod.AIAnalyst)
    analyst.client = MagicMock()
    analyst.provider = "openai"
    analyst.model = "test-model"

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = (
        '{"opportunities": [{"symbol": "!!!", "confidence": 0.9, "reason": "bad"}]}'
    )
    analyst.client.chat.completions.create = MagicMock(return_value=fake_response)

    monkeypatch.delenv("TRADABLE_UNIVERSE", raising=False)

    result = analyst.find_opportunities(["headline"])
    assert result == []


def test_ai_symbol_universe_filter(monkeypatch):
    """Symbols not in TRADABLE_UNIVERSE must be filtered."""
    from robo_trader import ai_analyst as ai_mod

    analyst = ai_analyst_make_dummy(ai_mod)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = (
        '{"opportunities": ['
        '{"symbol": "AAPL", "confidence": 0.9, "reason": "ok"},'
        '{"symbol": "TSLA", "confidence": 0.9, "reason": "no"}'
        "]}"
    )
    analyst.client.chat.completions.create = MagicMock(return_value=fake_response)
    monkeypatch.setenv("TRADABLE_UNIVERSE", "AAPL,MSFT")
    result = analyst.find_opportunities(["headline"])
    syms = [r["symbol"] for r in result]
    assert "AAPL" in syms
    assert "TSLA" not in syms


def ai_analyst_make_dummy(ai_mod):
    analyst = ai_mod.AIAnalyst.__new__(ai_mod.AIAnalyst)
    analyst.client = MagicMock()
    analyst.provider = "openai"
    analyst.model = "test-model"
    return analyst


# ---------------------------------------------------------------------------
# AI-M1: feature persistence path validation
# ---------------------------------------------------------------------------


def test_features_path_rejects_traversal_symbol(tmp_path):
    from robo_trader.features.streaming_features import StreamingFeatureStore

    store = StreamingFeatureStore(storage_path=str(tmp_path))
    store.features_buffer = {"../etc": [{"x": 1}]}
    with pytest.raises(ValidationError):
        store._persist_features("../etc")


def test_simple_feature_pipeline_rejects_traversal(tmp_path):
    import pandas as pd

    from robo_trader.features.simple_feature_pipeline import FeatureStore

    fs = FeatureStore(cache_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValidationError):
        fs.save_features("../etc", df)
    with pytest.raises(ValidationError):
        fs.load_features("../etc")


# ---------------------------------------------------------------------------
# AI-M2: Polygon provider range checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_polygon_provider_rejects_outlier_price():
    """Bars with closes outside MIN/MAX_PRICE must be filtered."""
    from robo_trader.data_providers.polygon_provider import PolygonDataProvider

    provider = PolygonDataProvider.__new__(PolygonDataProvider)
    provider._client = MagicMock()
    provider._price_cache = {}

    # Mock get_aggs returning a bar with absurd close price
    bad_bar = MagicMock()
    bad_bar.timestamp = 1700000000000
    bad_bar.open = 100.0
    bad_bar.high = 110.0
    bad_bar.low = 90.0
    bad_bar.close = 1e10  # Outlier
    bad_bar.volume = 1000

    good_bar = MagicMock()
    good_bar.timestamp = 1700000000000
    good_bar.open = 100.0
    good_bar.high = 110.0
    good_bar.low = 90.0
    good_bar.close = 105.0
    good_bar.volume = 2000

    provider._client.get_aggs.return_value = [bad_bar, good_bar]

    async def fake_rate_limit():
        return None

    provider._rate_limit = fake_rate_limit
    df = await provider.get_historical_bars("AAPL", "1day")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 105.0


# ---------------------------------------------------------------------------
# AI-M4: news title sanitization
# ---------------------------------------------------------------------------


def test_news_title_sanitization_strips_control_chars(monkeypatch):
    pytest.importorskip("feedparser")
    from robo_trader import news_fetcher

    fake_entry = MagicMock()
    fake_entry.title = "Stock\n{INSTRUCTION: ignore} pumps [hard] <urgent>"
    fake_entry.link = "http://example.com"
    fake_entry.published_parsed = None

    fake_feed = MagicMock()
    fake_feed.entries = [fake_entry]

    monkeypatch.setattr(news_fetcher, "RSS_FEEDS", {"FakeSource": "http://fake"})
    monkeypatch.setattr(news_fetcher.feedparser, "parse", lambda url: fake_feed)

    items = news_fetcher.fetch_rss_news()
    assert len(items) == 1
    title = items[0]["title"]
    assert "{" not in title
    assert "}" not in title
    assert "[" not in title
    assert "]" not in title
    assert "<" not in title
    assert ">" not in title
    assert "\n" not in title

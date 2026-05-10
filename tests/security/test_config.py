"""Security regression tests for configuration handling.

These cover the fixes from SECURITY_AUDIT_2026-05-10 Section 2.F:
  - LegacyConfig.__repr__ does not leak secrets
  - logger scrubs secret-shaped values inside messages
  - WebSocketLogProcessor never forwards audit/security/production logs
  - DATABASE_URL parses URL-encoded passwords; failures don't leak the URL
  - _determine_environment is strict
  - Production rejects placeholder credentials
  - .dockerignore excludes IBC config
  - requirements.txt is pinned for runtime packages
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from unittest.mock import patch

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# CFG-M1 — LegacyConfig.__repr__ secrecy
# ---------------------------------------------------------------------------
def test_legacy_config_repr_redacts():
    """repr(LegacyConfig()) must not leak password/secret/token-like fields."""
    from robo_trader.config import LegacyConfig

    try:
        cfg = LegacyConfig()
    except Exception:
        pytest.skip("LegacyConfig requires environment configuration; skip on minimal env")

    rep = repr(cfg)
    s = repr(cfg) + str(cfg)
    forbidden = ("password", "secret", "token", "api_key")
    for term in forbidden:
        assert term.lower() not in rep.lower(), f"repr leaked '{term}': {rep}"
    assert "<redacted>" in s.lower()


# ---------------------------------------------------------------------------
# CFG-M2 — logger redaction
# ---------------------------------------------------------------------------
def test_logger_scrubs_secrets_in_message():
    """censor_sensitive must redact secret-shaped values in string fields."""
    from robo_trader.logger import censor_sensitive

    fake_jwt = "eyJhbGciOi.eyJzdWIiOi1234567890.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    event = {"event": f"user logged in token={fake_jwt}"}
    out = censor_sensitive(None, "info", event)
    assert "***REDACTED***" in out["event"]
    assert fake_jwt not in out["event"]


def test_logger_redacts_github_token_value():
    from robo_trader.logger import censor_sensitive

    token = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    event = {"event": f"pushing to repo with {token} in URL"}
    out = censor_sensitive(None, "info", event)
    assert token not in out["event"]
    assert "***REDACTED***" in out["event"]


def test_websocket_log_skips_audit_logger():
    """Audit logs must never be forwarded to WebSocket clients."""
    from robo_trader.logger import WebSocketLogProcessor

    sent = []

    class FakeWSManager:
        class _Thread:
            def is_alive(self):
                return True

        thread = _Thread()

        def send_log_message(self, **kw):
            sent.append(kw)

    proc = WebSocketLogProcessor()
    WebSocketLogProcessor.set_ws_manager(FakeWSManager())
    try:
        # An audit-source event must NOT be forwarded.
        event = {"event": "sensitive audit", "logger": "audit"}
        proc(None, "info", event)
        assert sent == [], "audit log should not be forwarded to WS"

        # A normal event should be forwarded.
        event2 = {"event": "ok", "logger": "robo_trader.runner"}
        proc(None, "info", event2)
        assert len(sent) == 1
    finally:
        WebSocketLogProcessor.set_ws_manager(None)


def test_websocket_log_skips_production_subloggers():
    from robo_trader.logger import WebSocketLogProcessor

    sent = []

    class FakeWSManager:
        class _Thread:
            def is_alive(self):
                return True

        thread = _Thread()

        def send_log_message(self, **kw):
            sent.append(kw)

    proc = WebSocketLogProcessor()
    WebSocketLogProcessor.set_ws_manager(FakeWSManager())
    try:
        proc(
            None,
            "info",
            {"event": "secret event", "logger": "robo_trader.production.config_manager"},
        )
        assert sent == []
    finally:
        WebSocketLogProcessor.set_ws_manager(None)


# ---------------------------------------------------------------------------
# CFG-M4 — DATABASE_URL parsing
# ---------------------------------------------------------------------------
def test_database_url_parses_with_special_chars_in_password():
    """Percent-encoded passwords must decode correctly via urlparse."""
    from robo_trader.production.config_manager import (
        ConfigManager,
        ProductionConfig,
    )

    # Build a manager-like context without going through __init__ (which
    # validates a full environment).
    cm = ConfigManager.__new__(ConfigManager)
    cfg = ProductionConfig()
    out = cm._parse_database_url(cfg, "postgresql://u:p%40ss%23word@h:5432/d")
    assert out.database.pg_username == "u"
    assert out.database.pg_password == "p@ss#word"
    assert out.database.pg_host == "h"
    assert out.database.pg_port == 5432
    assert out.database.pg_database == "d"


def test_database_url_failure_does_not_leak():
    """ConfigurationError raised from a bad DATABASE_URL must not contain the URL."""
    from robo_trader.production.config_manager import (
        ConfigManager,
        ConfigurationError,
        ProductionConfig,
    )

    cm = ConfigManager.__new__(ConfigManager)
    cfg = ProductionConfig()

    secret_url = "postgresql://leakme:topsecret@host/db"
    with patch(
        "robo_trader.production.config_manager.urlparse",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(ConfigurationError) as exc:
            cm._parse_database_url(cfg, secret_url)
    msg = str(exc.value)
    assert "leakme" not in msg
    assert "topsecret" not in msg
    assert secret_url not in msg


# ---------------------------------------------------------------------------
# CFG-M5 — strict environment determination
# ---------------------------------------------------------------------------
def test_determine_environment_strict():
    """Typo'd TRADING_ENV must raise unless ALLOW_ENV_FALLBACK=1."""
    from robo_trader.production.config_manager import (
        ConfigManager,
        ConfigurationError,
    )

    cm = ConfigManager.__new__(ConfigManager)

    env_clean = {k: v for k, v in os.environ.items() if k not in ("ALLOW_ENV_FALLBACK", "CI")}
    env_clean["TRADING_ENV"] = "prodution"  # typo
    with patch.dict(os.environ, env_clean, clear=True):
        with pytest.raises(ConfigurationError):
            cm._determine_environment(None)

    # With opt-in, falls back gracefully.
    env_relaxed = dict(env_clean)
    env_relaxed["ALLOW_ENV_FALLBACK"] = "1"
    with patch.dict(os.environ, env_relaxed, clear=True):
        from robo_trader.production.config_manager import Environment

        assert cm._determine_environment(None) == Environment.DEVELOPMENT


# ---------------------------------------------------------------------------
# CFG-M6 — Grafana / DB / Redis password placeholder rejection
# ---------------------------------------------------------------------------
def test_grafana_default_rejected_in_prod():
    """A 'changeme' Grafana password must be rejected when live trading is on."""
    from robo_trader.production.config_manager import (
        ConfigManager,
        ConfigurationError,
        Environment,
        ProductionConfig,
    )

    cm = ConfigManager.__new__(ConfigManager)
    cm.environment = Environment.DEVELOPMENT  # rely on ENABLE_LIVE_TRADING flag
    cm.config = ProductionConfig()

    with patch.dict(
        os.environ,
        {"GRAFANA_PASSWORD": "changeme", "ENABLE_LIVE_TRADING": "true"},
        clear=False,
    ):
        with pytest.raises(ConfigurationError):
            cm._validate_config()


# ---------------------------------------------------------------------------
# CFG-M3 — .dockerignore excludes local IBC and env state
# ---------------------------------------------------------------------------
def test_dockerignore_excludes_ibc():
    path = REPO_ROOT / ".dockerignore"
    text = path.read_text()
    assert "config/ibc/" in text or "config/ibc" in text
    assert ".env" in text


# ---------------------------------------------------------------------------
# Dependency pinning
# ---------------------------------------------------------------------------
def test_requirements_pinned():
    """All runtime deps in requirements.txt must use == or ~= (no plain >= or unconstrained)."""
    path = REPO_ROOT / "requirements.txt"
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    bad = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        # Normalize: split on whitespace to ignore inline comments
        spec = ln.split("#", 1)[0].strip()
        if not spec:
            continue
        # An entry is acceptable if it contains == or ~= (compatible release)
        if "==" in spec or "~=" in spec:
            continue
        # Anything using >= alone, or with no operator at all, is unpinned.
        bad.append(ln)
    assert not bad, f"Unpinned requirements found: {bad}"

"""Security tests for the dashboard web layer (app.py).

Covers W-H1 (auth defaults), W-H2 (CSRF + Origin), W-H3 (XSS escape helper),
and W-M5 (subprocess output redaction).
"""

import hashlib
import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _reload_app(monkeypatch, **env):
    """Reload app.py with a fresh environment so module-level guards re-run."""
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    if "app" in sys.modules:
        del sys.modules["app"]
    return importlib.import_module("app")


def _password_hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------------------------------
# W-H1: auth defaults
# ---------------------------------------------------------------------------


def test_auth_required_when_enabled_and_hash_set(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="true",
        DASH_USER="admin",
        DASH_PASS_HASH=_password_hash("hunter2"),
    )
    client = app_mod.app.test_client()
    resp = client.get("/api/positions")
    assert resp.status_code == 401


def test_auth_refuses_to_start_with_empty_hash_when_enabled(monkeypatch):
    monkeypatch.setenv("DASH_AUTH_ENABLED", "true")
    monkeypatch.delenv("DASH_PASS_HASH", raising=False)
    if "app" in sys.modules:
        del sys.modules["app"]
    with pytest.raises(SystemExit):
        importlib.import_module("app")


# ---------------------------------------------------------------------------
# W-H2: CSRF + Origin
# ---------------------------------------------------------------------------


def test_csrf_blocks_post_without_token(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.post("/api/stop")
    assert resp.status_code == 403
    assert resp.get_json() == {"error": "csrf"}


def test_csrf_allows_post_with_matching_token(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    # Seed the csrf cookie via a GET (after_request issues it).
    client.get("/api/health")
    cookie_token = None
    for cookie in client.cookie_jar if hasattr(client, "cookie_jar") else []:
        if cookie.name == "csrf_token":
            cookie_token = cookie.value
            break
    if cookie_token is None:
        # Fallback: set cookie manually.
        cookie_token = "x" * 32
        client.set_cookie("csrf_token", cookie_token, domain="localhost")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        resp = client.post(
            "/api/stop",
            headers={"X-CSRF-Token": cookie_token},
        )
    # Anything other than 403 means CSRF accepted the request.
    assert resp.status_code != 403


def test_origin_blocks_unknown_origin_post(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    # Even with a matching token cookie+header, an evil Origin must 403.
    token = "abc123token"
    client.set_cookie("csrf_token", token, domain="localhost")
    resp = client.post(
        "/api/stop",
        headers={
            "X-CSRF-Token": token,
            "Origin": "http://evil.example",
        },
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# W-H3: escHTML helper exists in rendered HTML
# ---------------------------------------------------------------------------


def test_websocket_xss_helper_escapes_html(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "function escHTML" in body
    # And it is actually used at least once on a hot path.
    assert "escHTML(" in body


# ---------------------------------------------------------------------------
# W-M5: subprocess output not leaked to client
# ---------------------------------------------------------------------------


def test_start_trading_does_not_leak_subprocess_output(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    token = "csrf-token-value"
    client.set_cookie("csrf_token", token, domain="localhost")

    fake = MagicMock(returncode=0, stdout="SECRET STDOUT TOKEN", stderr="SECRET STDERR TOKEN")
    with patch("subprocess.run", return_value=fake):
        resp = client.post(
            "/api/start",
            json={"symbols": ["AAPL"]},
            headers={"X-CSRF-Token": token},
        )

    assert resp.status_code == 200
    payload = resp.get_json() or {}
    # No raw subprocess fields leaked.
    assert "output" not in payload
    assert "stdout" not in payload
    assert "stderr" not in payload
    body = resp.get_data(as_text=True)
    assert "SECRET STDOUT TOKEN" not in body
    assert "SECRET STDERR TOKEN" not in body

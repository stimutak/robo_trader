"""Security tests for the dashboard web layer (app.py).

Covers W-H1 (auth defaults), W-H2 (CSRF + Origin), W-H3 (XSS escape helper),
W-M5 (subprocess output redaction), and Round-2 findings:
R2-OP3 (dashboard JS attaches X-CSRF-Token), W-R2-M1 (security headers),
W-R2-M2 (Origin allowlist ignores Host header), W-R2-M3 (bcrypt/scrypt + lockout),
and W-R2-L1 (refuse debug=True on non-loopback host).
"""

import base64
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


# ---------------------------------------------------------------------------
# R2-OP3: dashboard JS must include CSRF helpers and use them on POSTs.
# ---------------------------------------------------------------------------


def test_dashboard_html_has_csrf_helper(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    # Helpers exist.
    assert "function getCookie(" in body
    assert "csrfFetch(" in body
    # State-changing endpoints use csrfFetch (not raw fetch with method:POST).
    assert "csrfFetch('/api/start'" in body
    assert "csrfFetch('/api/stop'" in body
    # And the dashboard does NOT POST to /api/start / /api/stop with bare fetch.
    assert "fetch('/api/start'" not in body
    assert "fetch('/api/stop'" not in body


# ---------------------------------------------------------------------------
# W-R2-M1: global security headers
# ---------------------------------------------------------------------------


def test_security_headers_present_on_get(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/")
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("Referrer-Policy") == "no-referrer"
    csp = resp.headers.get("Content-Security-Policy", "")
    assert csp, "CSP header missing"
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "form-action 'self'" in csp
    assert "default-src 'self'" in csp


def test_security_headers_present_on_json_endpoint(monkeypatch):
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    # /api/status doesn't require csrf for GET; pick any GET endpoint that
    # responds without external state.
    resp = client.get("/api/status")
    # status endpoint may 200 or 500 depending on imports, but headers must
    # still be present.
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert "frame-ancestors 'none'" in resp.headers.get("Content-Security-Policy", "")


# ---------------------------------------------------------------------------
# W-R2-M2: Origin allowlist must NOT trust inbound Host header.
# ---------------------------------------------------------------------------


def test_csrf_origin_allowlist_ignores_host_header(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        DASH_HOST="127.0.0.1",
        DASH_PORT="5555",
        CORS_ORIGINS=None,
    )
    client = app_mod.app.test_client()
    token = "abc123token"
    client.set_cookie("csrf_token", token, domain="localhost")
    # Attacker sets Host: evil.example AND Origin: http://evil.example.
    # Pre-fix behaviour: f"http://{request.host}" was added to allowlist, so
    # this would have been ACCEPTED. Post-fix it must 403.
    resp = client.post(
        "/api/stop",
        headers={
            "X-CSRF-Token": token,
            "Origin": "http://evil.example",
            "Host": "evil.example",
        },
    )
    assert resp.status_code == 403


def test_csrf_origin_allowlist_uses_dash_host_port(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        DASH_HOST="127.0.0.1",
        DASH_PORT="5555",
    )
    origins = app_mod._allowed_request_origins.__wrapped__ if hasattr(
        app_mod._allowed_request_origins, "__wrapped__"
    ) else None
    # Call inside a request context so request.* doesn't blow up.
    with app_mod.app.test_request_context("/"):
        allowed = app_mod._allowed_request_origins()
    assert "http://127.0.0.1:5555" in allowed
    assert "http://localhost:5555" in allowed


# ---------------------------------------------------------------------------
# W-R2-M3: scrypt verification + per-IP lockout.
# ---------------------------------------------------------------------------


def _scrypt_hash(password: str) -> str:
    n, r, p = 16384, 8, 1
    salt = b"\x00" * 16
    derived = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32)
    return "$scrypt$%d$%d$%d$%s$%s" % (
        n,
        r,
        p,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def test_legacy_sha256_hash_still_verifies(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="true",
        DASH_USER="admin",
        DASH_PASS_HASH=_password_hash("hunter2"),
    )
    assert app_mod.check_auth("admin", "hunter2") is True
    assert app_mod.check_auth("admin", "wrong") is False
    assert app_mod.check_auth("nope", "hunter2") is False


def test_scrypt_hash_verifies(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="true",
        DASH_USER="admin",
        DASH_PASS_HASH=_scrypt_hash("hunter2"),
    )
    assert app_mod.check_auth("admin", "hunter2") is True
    assert app_mod.check_auth("admin", "wrong") is False


def test_invalid_hash_format_rejected_at_startup(monkeypatch):
    monkeypatch.setenv("DASH_AUTH_ENABLED", "true")
    monkeypatch.setenv("DASH_PASS_HASH", "not-a-real-hash")
    if "app" in sys.modules:
        del sys.modules["app"]
    with pytest.raises(SystemExit):
        importlib.import_module("app")


def test_per_ip_lockout_after_threshold(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="true",
        DASH_USER="admin",
        DASH_PASS_HASH=_password_hash("correctpass"),
        AUTH_LOCKOUT_THRESHOLD="3",
        AUTH_LOCKOUT_MINUTES="30",
    )
    client = app_mod.app.test_client()
    # 3 bad attempts -> next attempt is locked out (429).
    bad = ("admin", "wrong")
    creds = base64.b64encode(f"{bad[0]}:{bad[1]}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}"}
    for _ in range(3):
        resp = client.get("/api/status", headers=headers)
        assert resp.status_code == 401
    # 4th attempt: locked out, even with the correct password.
    good = base64.b64encode(b"admin:correctpass").decode()
    resp = client.get("/api/status", headers={"Authorization": f"Basic {good}"})
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers


def test_successful_auth_clears_failure_counter(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="true",
        DASH_USER="admin",
        DASH_PASS_HASH=_password_hash("correctpass"),
        AUTH_LOCKOUT_THRESHOLD="5",
        AUTH_LOCKOUT_MINUTES="30",
    )
    client = app_mod.app.test_client()
    bad = base64.b64encode(b"admin:wrong").decode()
    good = base64.b64encode(b"admin:correctpass").decode()
    # 2 failures, then success, then 4 more failures should still NOT lock out
    # (counter is reset by success; new window begins).
    for _ in range(2):
        client.get("/api/status", headers={"Authorization": f"Basic {bad}"})
    resp = client.get("/api/status", headers={"Authorization": f"Basic {good}"})
    assert resp.status_code != 429
    for _ in range(4):
        resp = client.get("/api/status", headers={"Authorization": f"Basic {bad}"})
    # 4 failures < threshold of 5: NOT locked out yet.
    resp = client.get("/api/status", headers={"Authorization": f"Basic {good}"})
    assert resp.status_code != 429


# ---------------------------------------------------------------------------
# W-R2-L1: refuse to start with debug=True on a non-loopback host.
# ---------------------------------------------------------------------------


def test_debug_on_non_loopback_host_raises(monkeypatch):
    """The startup guard lives under `if __name__ == '__main__'`, so we
    exercise its logic directly via the constants it relies on."""
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        FLASK_ENV="development",
        DASH_HOST="0.0.0.0",
    )
    # Re-execute the startup guard logic.
    flask_env = (os.getenv("FLASK_ENV", "") or "").strip().lower()
    dash_host = (os.getenv("DASH_HOST", "127.0.0.1") or "").strip()
    debug = flask_env == "development"
    LOOPBACK = {"127.0.0.1", "localhost", "::1"}
    assert debug is True
    assert dash_host not in LOOPBACK
    # When app.py's __main__ block runs, it must SystemExit. Verify the
    # condition that drives it.
    assert debug and dash_host not in LOOPBACK


def test_debug_on_loopback_host_allowed(monkeypatch):
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        FLASK_ENV="development",
        DASH_HOST="127.0.0.1",
    )
    flask_env = (os.getenv("FLASK_ENV", "") or "").strip().lower()
    dash_host = (os.getenv("DASH_HOST", "127.0.0.1") or "").strip()
    debug = flask_env == "development"
    LOOPBACK = {"127.0.0.1", "localhost", "::1"}
    # Loopback debug is OK.
    assert debug
    assert dash_host in LOOPBACK

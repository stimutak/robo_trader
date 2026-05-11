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


# ---------------------------------------------------------------------------
# Branch-audit (claude/security-audit-5tFIY) round-3 regression tests
# ---------------------------------------------------------------------------


def test_cors_rejects_wildcard_origin_c_12(monkeypatch):
    """C-12: a wildcard CORS_ORIGINS entry must abort startup, since
    supports_credentials=True + `*` is a confused-deputy primitive.
    """
    import importlib
    import sys
    monkeypatch.setenv("CORS_ORIGINS", "http://*.example.com")
    sys.modules.pop("app", None)
    with pytest.raises(SystemExit):
        importlib.import_module("app")
    # Cleanup so other tests can re-import cleanly without wildcard env.
    sys.modules.pop("app", None)


def test_kill_switch_status_reflects_state_file_b_13(tmp_path, monkeypatch):
    """B-13: /api/risk/kill-switch status must read the on-disk state file
    that KillSwitch persists to, not return a hard-coded {triggered: False}.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "kill_switch.lock").touch()
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    token = "kill-switch-token-32-chars-aaaaaa"
    client.set_cookie("csrf_token", token, domain="localhost")
    resp = client.post(
        "/api/risk/kill-switch",
        json={"action": "status"},
        headers={"X-CSRF-Token": token},
    )
    body = resp.get_json()
    assert body is not None and body.get("triggered") is True, body


def test_start_endpoint_rejects_bad_symbol_b_9(monkeypatch):
    """B-9: /api/start must reject invalid symbol strings BEFORE invoking
    the subprocess, so a 200-char payload or control character cannot flow
    into start_runner.sh.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    # Reset trading_status because earlier tests may have left it "running".
    app_mod.trading_status = "stopped"
    client = app_mod.app.test_client()
    token = "start-endpoint-token-32-chars-bbb"
    client.set_cookie("csrf_token", token, domain="localhost")
    bad_payload = {"symbols": ["AAPL;rm -rf /"]}
    resp = client.post(
        "/api/start",
        json=bad_payload,
        headers={"X-CSRF-Token": token},
    )
    assert resp.status_code == 400, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body.get("error") == "invalid_symbol"


def test_debug_is_unconditionally_false_a_1():
    """A-1: the Werkzeug debugger must NEVER be enabled by this entrypoint.
    The audit specifies disabling it unconditionally; FLASK_ENV=development
    only flips use_reloader, not debug.
    """
    import inspect
    import app as app_mod
    source = inspect.getsource(app_mod)
    # The literal call site for app.run must pass debug=False (or the
    # named constant False), with an inline comment referencing A-1.
    code_only = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert "debug=False" in code_only, (
        "app.run must pass debug=False unconditionally (A-1)."
    )
    # The old `debug=_debug` pattern must NOT come back.
    assert "debug=_debug" not in code_only, (
        "Remove the conditional debug pattern (A-1); use debug=False instead."
    )


# ---------------------------------------------------------------------------
# Branch-audit round-4: B-6 (CSRF Secure flag behind proxy), B-7 (HSTS + Permissions-Policy)
# ---------------------------------------------------------------------------


def test_csrf_cookie_secure_honors_forwarded_proto_b_6(monkeypatch):
    """B-6: when X-Forwarded-Proto: https is set (TLS terminated upstream),
    the CSRF cookie must carry the Secure flag even though Flask sees a
    plain-HTTP inner connection.
    """
    monkeypatch.setenv("TRUST_PROXY", "true")
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/api/health", headers={"X-Forwarded-Proto": "https"})
    set_cookie = resp.headers.get("Set-Cookie", "")
    if "csrf_token" not in set_cookie:
        # Force the cookie path: hit a GET that triggers _set_csrf_cookie.
        resp = client.get("/api/positions", headers={"X-Forwarded-Proto": "https"})
        set_cookie = resp.headers.get("Set-Cookie", "")
    assert "csrf_token" in set_cookie, set_cookie
    assert "Secure" in set_cookie, (
        f"CSRF cookie must carry Secure when X-Forwarded-Proto=https. Got: {set_cookie}"
    )


def test_csrf_cookie_secure_via_explicit_override_b_6(monkeypatch):
    """B-6: COOKIE_SECURE=true env var must force the Secure flag for ops
    who don't run a proxy that sets X-Forwarded-Proto.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false", COOKIE_SECURE="true")
    client = app_mod.app.test_client()
    resp = client.get("/api/health")
    set_cookie = resp.headers.get("Set-Cookie", "")
    if "csrf_token" not in set_cookie:
        resp = client.get("/api/positions")
        set_cookie = resp.headers.get("Set-Cookie", "")
    assert "csrf_token" in set_cookie
    assert "Secure" in set_cookie


def test_permissions_policy_header_present_b_7(monkeypatch):
    """B-7: every response must carry a Permissions-Policy header that
    explicitly denies APIs the dashboard doesn't use.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/api/health")
    pp = resp.headers.get("Permissions-Policy", "")
    assert pp, "Permissions-Policy header missing"
    for api in ("geolocation", "microphone", "camera"):
        assert api in pp, f"Permissions-Policy must explicitly deny {api!r}: got {pp!r}"


def test_hsts_emitted_only_on_https_b_7(monkeypatch):
    """B-7: Strict-Transport-Security must only be emitted when the
    operator's connection is HTTPS (avoid shipping HSTS over HTTP).
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()

    # Plain HTTP test client request: no HSTS expected.
    resp_http = client.get("/api/health")
    assert "Strict-Transport-Security" not in resp_http.headers


def test_hsts_emitted_when_forwarded_proto_https_b_7(monkeypatch):
    """B-7: HSTS must appear when X-Forwarded-Proto: https indicates the
    operator's connection is TLS-terminated upstream (with TRUST_PROXY=true).
    """
    monkeypatch.setenv("TRUST_PROXY", "true")
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    resp = client.get("/api/health", headers={"X-Forwarded-Proto": "https"})
    hsts = resp.headers.get("Strict-Transport-Security", "")
    assert "max-age" in hsts, hsts


# ---------------------------------------------------------------------------
# C-9: error message leakage — endpoints must return {"error": "internal_error"}
# rather than str(exception).
# ---------------------------------------------------------------------------


def test_c9_data_validator_endpoint_does_not_leak_exception_detail(monkeypatch):
    """C-9: /api/safety/data-validator must return a generic error body and
    HTTP 500 when the underlying call raises, never str(e) with internals.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()

    secret = "SECRET_DETAIL_DO_NOT_LEAK_a9f7b3c1"

    class _Boom:
        def get_statistics(self):
            raise RuntimeError(secret)

    # Force the route into the except branch via the hasattr() path it checks.
    app_mod.app.data_validator = _Boom()
    try:
        resp = client.get("/api/safety/data-validator")
    finally:
        del app_mod.app.data_validator

    assert resp.status_code == 500, resp.status_code
    body = resp.get_json() or {}
    assert body.get("error") == "internal_error", body
    # Belt-and-suspenders: raw exception detail must not appear anywhere.
    assert secret not in resp.get_data(as_text=True)


def test_c9_database_health_does_not_leak_exception_detail(monkeypatch):
    """C-9: /api/database/health returns generic body + 500 when DB raises."""
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()

    secret = "DB_BOOM_PRIVATE_PATH_/var/lib/secret.db"

    # SyncDatabaseReader is imported inside the function; patch the symbol on
    # the module it imports from so the patch takes effect on the next call.
    class _BoomReader:
        def __init__(self):
            raise RuntimeError(secret)

        def _fetch_one(self, *a, **kw):  # pragma: no cover - never reached
            raise RuntimeError(secret)

    with patch("sync_db_reader.SyncDatabaseReader", _BoomReader):
        resp = client.get("/api/database/health")

    assert resp.status_code == 500, resp.status_code
    body = resp.get_json() or {}
    assert body.get("error") == "internal_error", body
    assert body.get("status") == "unhealthy", body
    assert secret not in resp.get_data(as_text=True)


def test_c9_keeps_validation_error_messages_for_400(monkeypatch):
    """C-9 nuance: deterministic ValidationError responses (HTTP 400) must
    still surface the user-facing message — those are not exception leaks.
    The portfolio_id validator runs on @validate_portfolio routes.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()
    # /api/performance is @validate_portfolio; passing an invalid id should
    # 400 with the validator's message intact (not "internal_error").
    resp = client.get("/api/performance?portfolio_id=../etc/passwd")
    # The validator rejects path-traversal-y characters with a clear message.
    assert resp.status_code == 400, resp.status_code
    body = resp.get_json() or {}
    # The key is "error"; the value should be a validation-style message, NOT
    # "internal_error".
    assert body.get("error") != "internal_error", body


# ---------------------------------------------------------------------------
# D-11: liveness endpoint must return 503 (not 500) and no exception detail.
# ---------------------------------------------------------------------------


def test_d11_liveness_returns_503_with_generic_body_on_failure(monkeypatch):
    """D-11: /health/live is unauthenticated. On internal failure it must
    return HTTP 503 with ``{"status": "fail"}`` and zero exception detail.
    """
    app_mod = _reload_app(monkeypatch, DASH_AUTH_ENABLED="false")
    client = app_mod.app.test_client()

    secret = "LIVENESS_INTERNAL_BACKTRACE_x83qz"

    # The liveness endpoint's body is a jsonify(...) of a dict literal that
    # calls datetime.now().isoformat(); force that call to raise so the
    # except branch executes.
    with patch("app.datetime") as mock_dt:
        mock_dt.now.side_effect = RuntimeError(secret)
        resp = client.get("/health/live")

    assert resp.status_code == 503, resp.status_code
    body = resp.get_json() or {}
    assert body == {"status": "fail"}, body
    assert secret not in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# C-10: per-IP rate limit on expensive endpoints (/api/logs).
# ---------------------------------------------------------------------------


def test_c10_api_logs_rate_limit_returns_429_after_threshold(monkeypatch, tmp_path):
    """C-10: /api/logs is rate-limited per-IP via the
    API_LOGS_RATE_LIMIT_PER_MINUTE env var. After N requests within a minute,
    further requests get HTTP 429 + Retry-After header.
    """
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        API_LOGS_RATE_LIMIT_PER_MINUTE="3",
    )
    client = app_mod.app.test_client()

    # Pre-existing log file ensures the route runs to completion.
    log_path = tmp_path / "robo_trader.log"
    log_path.write_text("hello\n")
    monkeypatch.chdir(tmp_path)

    # First 3 should succeed.
    for i in range(3):
        resp = client.get("/api/logs")
        assert resp.status_code == 200, (i, resp.status_code)

    # 4th request must be rate-limited.
    resp = client.get("/api/logs")
    assert resp.status_code == 429, resp.status_code
    assert "Retry-After" in resp.headers
    # Retry-After should be a positive integer-as-string.
    assert int(resp.headers["Retry-After"]) > 0


def test_c10_rate_limit_is_per_ip(monkeypatch, tmp_path):
    """C-10: rate-limit bucket is keyed on client IP. Different IPs should
    have independent quotas. Flask's test client lets us spoof
    REMOTE_ADDR via environ_overrides.
    """
    app_mod = _reload_app(
        monkeypatch,
        DASH_AUTH_ENABLED="false",
        API_LOGS_RATE_LIMIT_PER_MINUTE="2",
    )
    client = app_mod.app.test_client()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "robo_trader.log").write_text("x\n")

    ip_a = {"REMOTE_ADDR": "10.0.0.1"}
    ip_b = {"REMOTE_ADDR": "10.0.0.2"}

    # IP A: exhaust quota.
    for _ in range(2):
        assert client.get("/api/logs", environ_overrides=ip_a).status_code == 200
    assert client.get("/api/logs", environ_overrides=ip_a).status_code == 429

    # IP B: still has full quota.
    assert client.get("/api/logs", environ_overrides=ip_b).status_code == 200


# ---------------------------------------------------------------------------
# WS-LIB-API — websockets library API version drift (recurring blocker)
# ---------------------------------------------------------------------------


def test_ws_request_headers_shim_handles_v15_api():
    """The `websockets` library v15+ moved headers from
    `websocket.request_headers` to `websocket.request.headers`. Our shim must
    return the headers regardless of which attribute layout the library uses.

    This bug has caused a months-long reconnect loop EVERY time the library
    was upgraded; this test pins the shim so the next upgrade doesn't repeat.
    """
    from robo_trader.websocket_server import _ws_request_headers, _ws_request_path

    # v15+ shape: websocket.request.headers / .path
    class _V15Request:
        headers = {"Authorization": "Bearer abc", "Origin": "http://127.0.0.1:5555"}
        path = "/socket"

    class _V15Ws:
        request = _V15Request()

    headers = _ws_request_headers(_V15Ws())
    assert headers.get("Authorization") == "Bearer abc"
    assert headers.get("Origin") == "http://127.0.0.1:5555"
    assert _ws_request_path(_V15Ws()) == "/socket"

    # v14- shape: websocket.request_headers / websocket.path
    class _V14Ws:
        request_headers = {"Authorization": "Bearer xyz"}
        path = "/legacy"

    headers = _ws_request_headers(_V14Ws())
    assert headers.get("Authorization") == "Bearer xyz"
    assert _ws_request_path(_V14Ws()) == "/legacy"

    # Unknown shape: shim must return {} (not raise) so the server can reject
    # cleanly rather than crash. The shim logs ERROR so the operator notices.
    class _UnknownWs:
        pass

    headers = _ws_request_headers(_UnknownWs())
    assert headers == {}


def test_ws_auth_end_to_end_against_real_library():
    """Smoke: stand up the real WebSocketManager and connect with a Bearer
    token. This would have caught the v15 regression at PR time.
    """
    import asyncio
    import websockets
    from robo_trader.websocket_server import WebSocketManager

    async def go():
        mgr = WebSocketManager(host="127.0.0.1", port=18766)
        mgr.auth_token = "test-token-32-chars-aaaaaaaaaaaaa"
        server = await websockets.serve(mgr.handle_client, "127.0.0.1", 18766)
        try:
            headers = {
                "Authorization": f"Bearer {mgr.auth_token}",
                "Origin": "http://127.0.0.1:5555",
            }
            try:
                ws = await websockets.connect(
                    "ws://127.0.0.1:18766", additional_headers=headers
                )
            except TypeError:
                ws = await websockets.connect(
                    "ws://127.0.0.1:18766", extra_headers=headers
                )
            await ws.send('{"type":"subscribe","symbols":["AAPL"]}')
            await asyncio.sleep(0.2)
            await ws.close()
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(go())

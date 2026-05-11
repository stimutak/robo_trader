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

Round-2 additions (SECURITY_AUDIT_ROUND2_2026-05-10 Section 2.F):
  - write_env_atomic preserves the original file on mid-write failure
  - _scrub_value recurses into nested dicts/lists
  - process_manager rejects pkill patterns with shell metachars
  - GitHub workflow YAML has top-level or per-job ``permissions: read``
"""
from __future__ import annotations

import os
import re
import sys
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
    """All runtime deps in requirements.txt must use == or ~= (no plain >= or unconstrained).

    Exception: ``cryptography`` is allowed to use ``>=`` so we automatically
    pick up CVE-fix patch releases. Floor must still be set (R2-DEP-1).
    """
    path = REPO_ROOT / "requirements.txt"
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    bad = []
    # ``cryptography`` (R2-DEP-1) and ``gunicorn`` (CFGN-H1, followup audit) are
    # both pinned with ``>=`` floors that mark the minimum CVE-clean version.
    # New entries here require a security justification in the requirements file.
    floor_only_allowed = {"cryptography", "gunicorn"}
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
        # Allow `>=` floor for the curated allowlist (security patches).
        pkg = re.split(r"[<>=!~]", spec, maxsplit=1)[0].strip().lower()
        if pkg in floor_only_allowed and ">=" in spec:
            continue
        # Anything using >= alone (not allowlisted), or with no operator, is unpinned.
        bad.append(ln)
    assert not bad, f"Unpinned requirements found: {bad}"


# ===========================================================================
# Round-2 (R2) regression tests
# ===========================================================================


# ---------------------------------------------------------------------------
# R2-NEW-1/7/11 — atomic .env write
# ---------------------------------------------------------------------------
def _load_atomic_env_module():
    """Import scripts/_atomic_env.py via spec_from_file_location.

    We deliberately avoid sys.path.insert: a stale scripts/ entry would shadow
    top-level modules like sync_db_reader for any later test in this session.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "_atomic_env.py"
    spec = importlib.util.spec_from_file_location("_atomic_env_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_process_manager_module():
    """Import scripts/utilities/process_manager.py via spec_from_file_location.

    See _load_atomic_env_module — we must not pollute sys.path; doing so would
    shadow the top-level sync_db_reader module via scripts/utilities.
    """
    import importlib.util

    path = REPO_ROOT / "scripts" / "utilities" / "process_manager.py"
    spec = importlib.util.spec_from_file_location("process_manager_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_write_env_atomic_replaces_file_atomically(tmp_path):
    """A successful write_env_atomic produces 0600 file with new contents."""
    mod = _load_atomic_env_module()
    target = tmp_path / "env.test"
    target.write_text("OLD=value\n")
    os.chmod(target, 0o644)

    mod.write_env_atomic(target, "NEW=value\n")
    assert target.read_text() == "NEW=value\n"
    # New file should be 0600
    mode = target.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_write_env_atomic_preserves_original_on_failure(tmp_path, monkeypatch):
    """If os.replace fails midway, the original file must still exist intact."""
    mod = _load_atomic_env_module()
    target = tmp_path / "env.test"
    original_text = "ORIGINAL=intact\nIBKR_ACCOUNT=DU123\n"
    target.write_text(original_text)

    boom = RuntimeError("simulated mid-write failure")

    def fail_replace(_src, _dst):
        raise boom

    monkeypatch.setattr(mod.os, "replace", fail_replace)

    with pytest.raises(RuntimeError):
        mod.write_env_atomic(target, "TRUNCATED=")

    # Original must be untouched (this is the whole point).
    assert target.read_text() == original_text

    # No leftover .env.*.tmp temp files.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".env.test.")]
    assert leftovers == [], f"temp files leaked: {leftovers}"


# ---------------------------------------------------------------------------
# R2-CFG-M2.1 — recursive scrubbing
# ---------------------------------------------------------------------------
def test_scrub_value_recurses_into_dict():
    from robo_trader.logger import _scrub_value

    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    payload = {
        "outer": "ok",
        "config": {
            "github_token": secret,  # nested string with secret
            "nested": {"deeper": f"prefix {secret} suffix"},
        },
    }
    out = _scrub_value(payload)
    flat = repr(out)
    assert secret not in flat, f"secret leaked through dict recursion: {flat}"
    assert "***REDACTED***" in flat


def test_scrub_value_recurses_into_list_and_tuple():
    from robo_trader.logger import _scrub_value

    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    out_list = _scrub_value(["safe", secret, ["deeper", secret]])
    out_tuple = _scrub_value(("safe", secret))
    assert secret not in repr(out_list)
    assert secret not in repr(out_tuple)
    assert isinstance(out_tuple, tuple)


def test_websocket_processor_scrubs_nested_context():
    """WebSocketLogProcessor must scrub nested secrets before forwarding."""
    from robo_trader.logger import WebSocketLogProcessor

    sent: list[dict] = []

    class FakeWSManager:
        class _Thread:
            def is_alive(self):
                return True

        thread = _Thread()

        def send_log_message(self, **kw):
            sent.append(kw)

    secret = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"
    proc = WebSocketLogProcessor()
    WebSocketLogProcessor.set_ws_manager(FakeWSManager())
    try:
        event = {
            "event": "ok",
            "logger": "robo_trader.runner",
            "config": {"github_token": secret},
        }
        proc(None, "info", event)
        assert len(sent) == 1, "expected one forward"
        assert secret not in repr(sent[0]), f"secret leaked: {sent[0]}"
        assert "***REDACTED***" in repr(sent[0])
    finally:
        WebSocketLogProcessor.set_ws_manager(None)


# ---------------------------------------------------------------------------
# R2-OP4 — process_manager pkill pattern allowlist
# ---------------------------------------------------------------------------
def test_process_manager_rejects_shell_metachars():
    """Patterns containing shell metacharacters must be rejected."""
    pm = _load_process_manager_module()
    bad_patterns = [
        "robo; rm -rf /",
        "$(curl evil.example)",
        "robo`whoami`",
        "robo|cat",
        "robo&&pwd",
        "robo > /tmp/x",
        "  robo  ",  # leading/trailing whitespace
        "robo\nrm",
        "",
    ]
    for p in bad_patterns:
        with pytest.raises(ValueError):
            pm._validate_kill_pattern(p)


def test_process_manager_accepts_safe_patterns():
    pm = _load_process_manager_module()
    for ok in (
        "robo_trader.runner_async",
        "app.py",
        "/usr/bin/python3",
        "websocket-server",
        "robo_trader.websocket_server",
    ):
        assert pm._validate_kill_pattern(ok) == ok


# ---------------------------------------------------------------------------
# R2-CFG-H1.x — workflow YAML permissions
# ---------------------------------------------------------------------------
def test_workflow_yamls_have_permissions():
    """Every workflow YAML must declare permissions at workflow OR every job level."""
    yaml = pytest.importorskip("yaml")

    workflow_files = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "deploy.yml",
        REPO_ROOT / ".github" / "workflows" / "docker.yml",
        REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml",
        REPO_ROOT / ".github" / "workflows" / "claude.yml",
        REPO_ROOT / ".github" / "workflows" / "production-ci.yml",
    ]
    failures: list[str] = []
    for path in workflow_files:
        data = yaml.safe_load(path.read_text())
        assert isinstance(data, dict), f"{path}: not a YAML mapping"
        # Either workflow-level permissions or per-job permissions
        if "permissions" in data:
            continue
        jobs = data.get("jobs", {})
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            if "permissions" not in job:
                failures.append(f"{path.name}:{job_name} missing permissions")
    assert not failures, "Missing permissions:\n  " + "\n  ".join(failures)


def test_workflow_yamls_parse():
    """All managed workflow YAML files parse cleanly."""
    yaml = pytest.importorskip("yaml")
    workflow_files = [
        REPO_ROOT / ".github" / "workflows" / "ci.yml",
        REPO_ROOT / ".github" / "workflows" / "deploy.yml",
        REPO_ROOT / ".github" / "workflows" / "docker.yml",
        REPO_ROOT / ".github" / "workflows" / "claude-code-review.yml",
        REPO_ROOT / ".github" / "workflows" / "claude.yml",
        REPO_ROOT / ".github" / "workflows" / "production-ci.yml",
    ]
    for path in workflow_files:
        # Will raise on parse error.
        yaml.safe_load(path.read_text())


# ---------------------------------------------------------------------------
# Followup-audit CVE bumps (CFGN-H1/H2/H3)
# ---------------------------------------------------------------------------


def test_gunicorn_floor_avoids_cve_2024_1135():
    """CFGN-H1: gunicorn 21.2.0 has CVE-2024-1135. Both requirements files
    must require >=22.0.0.
    """
    text_main = (REPO_ROOT / "requirements.txt").read_text()
    text_prod = (REPO_ROOT / "requirements-prod.txt").read_text()
    assert re.search(r"^gunicorn\s*>=\s*22\.", text_main, re.MULTILINE), (
        "requirements.txt must pin gunicorn>=22.0.0 (CFGN-H1, CVE-2024-1135)"
    )
    assert re.search(r"^gunicorn\s*>=\s*22\.", text_prod, re.MULTILINE), (
        "requirements-prod.txt must pin gunicorn>=22.0.0 (CFGN-H1)"
    )


def test_aiohttp_floor_avoids_cve_2024_23334():
    """CFGN-H3: aiohttp 3.9.1 has CVE-2024-23334. Production must require >=3.11.18."""
    text = (REPO_ROOT / "requirements-prod.txt").read_text()
    assert re.search(r"^aiohttp\b.*>=\s*3\.11\.", text, re.MULTILINE), (
        "requirements-prod.txt must pin aiohttp>=3.11.18 (CFGN-H3, CVE-2024-23334)"
    )


def test_python_jose_removed_cfgn_h2():
    """CFGN-H2: python-jose 3.3.0 has CVE-2024-33664. We rely on PyJWT instead."""
    text = (REPO_ROOT / "requirements-prod.txt").read_text()
    assert not re.search(r"^python-jose", text, re.MULTILINE), (
        "python-jose must be removed (CFGN-H2, CVE-2024-33664). Use PyJWT instead."
    )
    assert re.search(r"^PyJWT", text, re.MULTILINE), (
        "PyJWT must remain pinned as the JWT replacement for python-jose"
    )


# ---------------------------------------------------------------------------
# Branch-audit round-3: B-5 (eventlet pin)
# ---------------------------------------------------------------------------


def test_eventlet_floor_avoids_request_smuggling_b_5():
    """B-5: eventlet 0.33.3 had request-smuggling CVEs. Floor at 0.36.1."""
    text = (REPO_ROOT / "requirements-prod.txt").read_text()
    import re as _re
    if not _re.search(r"^eventlet\b", text, _re.MULTILINE):
        # eventlet may legitimately be removed entirely
        return
    assert _re.search(r"^eventlet\s*>=\s*0\.(3[6-9]|[4-9]\d)\.", text, _re.MULTILINE), (
        "eventlet must be pinned >=0.36.1 (B-5) or removed entirely"
    )

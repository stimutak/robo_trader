"""
Security tests for IBKR / subprocess / Gateway surface.

Covers fixes from SECURITY_AUDIT_2026-05-10.md Section 2.E:
    - IB-H1: Gateway read-only enforcement (config + startup self-check)
    - IB-M1: VIRTUAL_ENV restriction in subprocess client
    - IB-L1: mktemp usage in force_gateway_reconnect.sh
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# IB-H1 — Gateway read-only enforcement
# ---------------------------------------------------------------------------


def _read_template_text() -> str:
    template = PROJECT_ROOT / "config" / "ibc" / "config.ini.template"
    assert template.exists(), f"Template not found at {template}"
    return template.read_text()


def test_ibc_template_is_readonly():
    """The shipped IBC template must default to read-only API access.

    Audit reference: IB-H1.
    """
    text = _read_template_text()

    # Match exact line, ignoring surrounding whitespace.
    lines = [line.strip() for line in text.splitlines()]

    assert "ReadOnlyApi=yes" in lines, (
        "config/ibc/config.ini.template must contain 'ReadOnlyApi=yes' "
        "to enforce Gateway-side read-only safety (IB-H1)."
    )

    # The audit explicitly calls out AllowBlindTrading=yes as unsafe.
    assert "AllowBlindTrading=yes" not in lines, (
        "config/ibc/config.ini.template must not enable blind trading."
    )
    assert "AllowBlindTrading=no" in lines, (
        "config/ibc/config.ini.template must explicitly set "
        "'AllowBlindTrading=no'."
    )


def test_start_trader_checks_readonly():
    """START_TRADER.sh must contain an explicit grep for ReadOnlyApi=yes
    that aborts startup if the active config has been weakened.

    Audit reference: IB-H1 (startup self-check).

    NOTE: We intentionally don't actually invoke START_TRADER.sh here —
    it would attempt to connect to Gateway. We assert the safety check
    is present in the script source. A full end-to-end integration test
    is left as TODO until a sandboxed gateway harness is available.
    """
    script_path = PROJECT_ROOT / "START_TRADER.sh"
    assert script_path.exists(), f"START_TRADER.sh not found at {script_path}"

    text = script_path.read_text()
    assert "ReadOnlyApi=yes" in text, (
        "START_TRADER.sh must reference 'ReadOnlyApi=yes' for the safety check."
    )
    # The check should be a grep + a non-zero exit when the requirement isn't met.
    assert "grep -q '^ReadOnlyApi=yes'" in text, (
        "START_TRADER.sh must use a grep to verify the active IBC config."
    )
    assert "exit 4" in text, (
        "START_TRADER.sh must exit non-zero (audit prescribes 4) when the "
        "ReadOnlyApi check fails."
    )


def test_start_gateway_script_checks_readonly():
    """scripts/start_gateway.sh must refuse to launch if config lacks
    ReadOnlyApi=yes. Audit reference: IB-H1."""
    script_path = PROJECT_ROOT / "scripts" / "start_gateway.sh"
    assert script_path.exists()

    text = script_path.read_text()
    assert "grep -q '^ReadOnlyApi=yes'" in text, (
        "scripts/start_gateway.sh must verify ReadOnlyApi=yes in the active config."
    )
    assert "exit 3" in text, (
        "scripts/start_gateway.sh must exit non-zero on failed safety check."
    )


# ---------------------------------------------------------------------------
# IB-L2 — IBC config permissions / umask
# ---------------------------------------------------------------------------


def test_start_gateway_chmod_and_umask():
    """scripts/start_gateway.sh must restrict permissions on a freshly
    created config.ini and set a restrictive umask (audit IB-L2)."""
    script_path = PROJECT_ROOT / "scripts" / "start_gateway.sh"
    text = script_path.read_text()

    assert "umask 077" in text, (
        "scripts/start_gateway.sh must set 'umask 077' to keep credential "
        "files group/other-readable-proof."
    )
    assert 'chmod 600 "$IBC_INI"' in text, (
        "scripts/start_gateway.sh must chmod 600 the freshly-created "
        "IBC config (it contains plaintext IBKR credentials)."
    )


# ---------------------------------------------------------------------------
# IB-M1 — subprocess client must not honor an external VIRTUAL_ENV
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_client_ignores_external_venv(monkeypatch, tmp_path):
    """If VIRTUAL_ENV points outside the project root, the subprocess
    client must refuse to use it. Audit reference: IB-M1.

    We can't actually start the worker subprocess here (the env may not
    have ib_async / Gateway), so we monkeypatch subprocess.Popen to
    capture the python_exe arg before any real process is spawned, then
    short-circuit by raising.
    """
    # Make sure pytest-asyncio is available; if not, skip cleanly.
    pytest.importorskip("pytest_asyncio")

    from robo_trader.clients import subprocess_ibkr_client as mod

    # Pretend an attacker has set VIRTUAL_ENV to an external path.
    evil_venv = tmp_path / "evil_venv"
    (evil_venv / "bin").mkdir(parents=True)
    evil_python = evil_venv / "bin" / "python3"
    evil_python.write_text("#!/bin/sh\nexit 0\n")
    evil_python.chmod(0o755)
    monkeypatch.setenv("VIRTUAL_ENV", str(evil_venv))

    captured = {}

    class _BoomPopen:
        def __init__(self, args, *a, **kw):
            captured["args"] = args
            # Halt the test before any real subprocess work.
            raise RuntimeError("stop here for test")

    monkeypatch.setattr(mod.subprocess, "Popen", _BoomPopen)

    client = mod.SubprocessIBKRClient()
    with pytest.raises(RuntimeError, match="stop here for test"):
        await client.start()

    args = captured.get("args")
    assert args, "subprocess.Popen was never called"
    python_exe = args[0]
    # The resolved interpreter must NOT be the attacker-controlled one.
    assert python_exe != str(evil_python), (
        "subprocess_ibkr_client honored an external VIRTUAL_ENV "
        f"({python_exe}); IB-M1 requires that be ignored."
    )
    assert str(evil_venv) not in python_exe, (
        f"Resolved python_exe '{python_exe}' is inside the attacker venv "
        f"'{evil_venv}'."
    )


def test_subprocess_client_source_has_relative_to_check():
    """Belt-and-suspenders: the source must contain the relative_to() guard
    that anchors VIRTUAL_ENV to the project root (audit IB-M1)."""
    src_path = (
        PROJECT_ROOT / "robo_trader" / "clients" / "subprocess_ibkr_client.py"
    )
    text = src_path.read_text()
    assert ".relative_to(project_root)" in text, (
        "subprocess_ibkr_client.py must constrain VIRTUAL_ENV to the project "
        "root via Path.relative_to()."
    )
    assert "outside project root" in text, (
        "subprocess_ibkr_client.py must log a clear warning when VIRTUAL_ENV "
        "is rejected for being outside the project root."
    )


# ---------------------------------------------------------------------------
# IB-L1 — temp file handling in force_gateway_reconnect.sh
# ---------------------------------------------------------------------------


def test_force_gateway_reconnect_uses_mktemp():
    """force_gateway_reconnect.sh must create its temp script via mktemp,
    not at a predictable /tmp/test_gateway_accept.py path (audit IB-L1)."""
    script_path = PROJECT_ROOT / "force_gateway_reconnect.sh"
    assert script_path.exists()

    text = script_path.read_text()

    # Must call mktemp.
    assert "mktemp" in text, (
        "force_gateway_reconnect.sh must use mktemp instead of a hardcoded path."
    )

    # The old hardcoded path must not be present as a write target. The
    # comment-only mention in cleanup is fine, but we should not see
    # `cat > /tmp/test_gateway_accept.py` or `python3 /tmp/test_gateway_accept.py`.
    assert "cat > /tmp/test_gateway_accept.py" not in text, (
        "force_gateway_reconnect.sh must not write to a predictable path."
    )
    assert "python3 /tmp/test_gateway_accept.py" not in text, (
        "force_gateway_reconnect.sh must not exec from a predictable path."
    )

    # And it should clean up via trap.
    assert "trap" in text and 'rm -f "$TMPFILE"' in text, (
        "force_gateway_reconnect.sh must clean up its tempfile via trap."
    )

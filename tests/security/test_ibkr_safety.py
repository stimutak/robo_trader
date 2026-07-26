"""
Security tests for IBKR / subprocess / Gateway surface.

Covers fixes from SECURITY_AUDIT_2026-05-10.md Section 2.E
and SECURITY_AUDIT_ROUND2_2026-05-10.md Section 2.E:
    - IB-H1: Gateway read-only enforcement (config + startup self-check)
    - IB-M1: VIRTUAL_ENV restriction in subprocess client
    - IB-L1: mktemp usage in force_gateway_reconnect.sh
    - NEW-IB-H1.1: ReadOnlyApi grep regex must be anchored + case-insensitive
    - NEW-IB-M1.1: Interpreter path must be resolved + allowlisted
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
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
    assert (
        "AllowBlindTrading=yes" not in lines
    ), "config/ibc/config.ini.template must not enable blind trading."
    assert "AllowBlindTrading=no" in lines, (
        "config/ibc/config.ini.template must explicitly set " "'AllowBlindTrading=no'."
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
    assert (
        "ReadOnlyApi=yes" in text
    ), "START_TRADER.sh must reference 'ReadOnlyApi=yes' for the safety check."
    assert "validate_ibc_safety_config" in text and "readonlyapi" in text.lower(), (
        "START_TRADER.sh must normalize and cardinality-check ReadOnlyApi " "(NEW-IB-H1.1)."
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
    is_quarantined = "DISABLED:" in text and "exit 2" in text
    # NEW-IB-H1.1: anchored, case-insensitive grep.
    has_readonly_guard = "grep -Eqi" in text and "readonlyapi" in text.lower()
    assert is_quarantined or has_readonly_guard, (
        "scripts/start_gateway.sh must be inert or verify ReadOnlyApi=yes via "
        "anchored, case-insensitive grep (NEW-IB-H1.1)."
    )
    assert ("exit 2" in text) if is_quarantined else ("exit 3" in text)


# ---------------------------------------------------------------------------
# IB-L2 — IBC config permissions / umask
# ---------------------------------------------------------------------------


def test_start_gateway_chmod_and_umask():
    """scripts/start_gateway.sh must restrict permissions on a freshly
    created config.ini and set a restrictive umask (audit IB-L2)."""
    script_path = PROJECT_ROOT / "scripts" / "start_gateway.sh"
    text = script_path.read_text()
    is_quarantined = "DISABLED:" in text and "exit 2" in text

    assert is_quarantined or "umask 077" in text, (
        "scripts/start_gateway.sh must set 'umask 077' to keep credential "
        "files group/other-readable-proof."
    )
    assert is_quarantined or 'chmod 600 "$IBC_INI"' in text, (
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

    # On CI the test interpreter may resolve outside the production allowlist
    # (e.g. /opt/hostedtoolcache on GitHub Actions). That allowlist is NOT what
    # this test exercises — it verifies the external VIRTUAL_ENV is ignored — so
    # trust the real interpreter's prefix to reach the Popen capture point. The
    # attacker venv lives under tmp_path and remains outside the allowlist, so
    # the assertions below still prove it was rejected.
    monkeypatch.setattr(
        mod,
        "_INTERPRETER_PREFIX_ALLOWLIST",
        mod._INTERPRETER_PREFIX_ALLOWLIST + (Path(sys.executable).resolve().parent,),
    )

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
        f"Resolved python_exe '{python_exe}' is inside the attacker venv " f"'{evil_venv}'."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_runtime_environment", "expected_synthetic_environment"),
    [("test", "test"), ("production", None)],
)
async def test_subprocess_worker_launch_isolated_from_cwd_pythonpath_and_secrets(
    monkeypatch,
    tmp_path,
    worker_runtime_environment,
    expected_synthetic_environment,
):
    """The child executes one verified project script with no parent secrets."""
    pytest.importorskip("pytest_asyncio")

    from robo_trader.clients import subprocess_ibkr_client as mod

    evil_cwd = tmp_path / "attacker"
    evil_package = evil_cwd / "robo_trader" / "clients"
    evil_package.mkdir(parents=True)
    (evil_cwd / "robo_trader" / "__init__.py").write_text("", encoding="utf-8")
    (evil_cwd / "robo_trader" / "clients" / "__init__.py").write_text("", encoding="utf-8")
    (evil_package / "ibkr_subprocess_worker.py").write_text(
        "raise RuntimeError('hijacked')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(evil_cwd)
    monkeypatch.setenv("PYTHONPATH", str(evil_cwd))
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", "dashboard-secret")
    monkeypatch.setenv("IBKR_PASSWORD", "broker-secret")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "model-secret")
    monkeypatch.setenv("DATABASE_URL", "database-secret")
    # These ambient values must never decide the isolated worker's policy.
    monkeypatch.setenv("ENVIRONMENT", "dev")
    monkeypatch.setenv("RT_TEST_MODE", "1")

    captured = {}

    def capture_popen(args, *popen_args, **kwargs):
        captured.update(args=args, popen_args=popen_args, kwargs=kwargs)
        raise RuntimeError("captured isolated worker launch")

    monkeypatch.setattr(mod.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(
        mod,
        "_INTERPRETER_PREFIX_ALLOWLIST",
        mod._INTERPRETER_PREFIX_ALLOWLIST + (Path(sys.executable).resolve().parent,),
    )

    client = mod.SubprocessIBKRClient(
        worker_runtime_environment=worker_runtime_environment,
    )
    with pytest.raises(RuntimeError, match="captured isolated worker launch"):
        await client.start()

    args = captured["args"]
    kwargs = captured["kwargs"]
    project_root = mod._find_project_root(Path(mod.__file__)).resolve()
    worker_script = (
        project_root / "robo_trader" / "clients" / "ibkr_subprocess_worker.py"
    ).resolve()

    assert args[1:4] == ["-I", "-c", mod._WORKER_BOOTSTRAP]
    assert args[4:] == [str(project_root), str(worker_script)]
    assert kwargs["cwd"] == str(project_root)
    assert str(evil_cwd) not in args

    child_env = kwargs["env"]
    assert "PYTHONPATH" not in child_env
    assert "VIRTUAL_ENV" not in child_env
    assert "DASHBOARD_PASSWORD_HASH" not in child_env
    assert "IBKR_PASSWORD" not in child_env
    assert "MODEL_SIGNING_KEY" not in child_env
    assert "DATABASE_URL" not in child_env
    assert "ENVIRONMENT" not in child_env
    assert "RT_TEST_MODE" not in child_env
    assert set(child_env) <= (
        mod._WORKER_ENV_ALLOWLIST
        | {
            "PYTHONIOENCODING",
            "PYTHONSAFEPATH",
            "PYTHONUNBUFFERED",
            "ROBOTRADER_WORKER_GENERATION_ID",
            mod._WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT,
        }
    )
    assert child_env["PYTHONSAFEPATH"] == "1"
    assert child_env["ROBOTRADER_WORKER_GENERATION_ID"]
    if expected_synthetic_environment is None:
        assert mod._WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT not in child_env
    else:
        assert (
            child_env[mod._WORKER_SYNTHETIC_ACCOUNT_ENVIRONMENT] == expected_synthetic_environment
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_runtime_environment", "synthetic_account_permitted"),
    [("test", True), ("production", False)],
)
async def test_real_worker_child_attests_isolated_policy_without_gateway(
    monkeypatch,
    tmp_path,
    worker_runtime_environment,
    synthetic_account_permitted,
):
    """Exercise the real isolated child handshake and its effective policy."""

    from robo_trader.clients import subprocess_ibkr_client as mod

    evil_cwd = tmp_path / "attacker"
    evil_cwd.mkdir()
    monkeypatch.chdir(evil_cwd)
    monkeypatch.setenv("PYTHONPATH", str(evil_cwd))
    monkeypatch.setenv("VIRTUAL_ENV", str(evil_cwd / "venv"))
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", "dashboard-secret")
    monkeypatch.setenv("IBKR_PASSWORD", "broker-secret")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "model-secret")
    monkeypatch.setenv("DATABASE_URL", "database-secret")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("RT_TEST_MODE", "1")
    monkeypatch.setattr(
        mod,
        "_INTERPRETER_PREFIX_ALLOWLIST",
        mod._INTERPRETER_PREFIX_ALLOWLIST + (Path(sys.executable).resolve().parent,),
    )

    client = mod.SubprocessIBKRClient(
        worker_runtime_environment=worker_runtime_environment,
    )
    try:
        await asyncio.wait_for(client.start(), timeout=10.0)
        policy_response = await asyncio.wait_for(
            client._execute_command({"command": "health"}, timeout=5.0),
            timeout=10.0,
        )
    finally:
        await asyncio.wait_for(client.stop(), timeout=10.0)

    assert client.process is None
    assert policy_response["worker_policy"] == {
        "reserved_synthetic_account_permitted": synthetic_account_permitted,
        "forbidden_ambient_keys_present": [],
    }


@pytest.mark.asyncio
async def test_zombie_check_uses_absolute_lsof_with_secret_free_environment(
    monkeypatch,
):
    from robo_trader.clients import subprocess_ibkr_client as mod

    trusted_lsof = Path("/usr/sbin/lsof")
    monkeypatch.setattr(mod, "_trusted_lsof_executable", lambda: trusted_lsof)
    monkeypatch.setenv("PATH", "/attacker-controlled")
    monkeypatch.setenv("IBKR_PASSWORD", "broker-secret")
    monkeypatch.setenv("DASHBOARD_PASSWORD_HASH", "dashboard-secret")
    monkeypatch.setenv("MODEL_SIGNING_KEY", "model-secret")

    captured = {}

    def fake_run(args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(mod.subprocess, "run", fake_run)

    client = mod.SubprocessIBKRClient()
    assert await client._check_zombie_connections(4002) == (0, "No zombies detected")

    assert captured["args"][0] == str(trusted_lsof)
    assert Path(captured["args"][0]).is_absolute()
    assert captured["kwargs"]["cwd"] == "/"
    assert captured["kwargs"]["close_fds"] is True
    child_env = captured["kwargs"]["env"]
    assert set(child_env) <= mod._WORKER_ENV_ALLOWLIST
    assert "PATH" not in child_env
    assert "IBKR_PASSWORD" not in child_env
    assert "DASHBOARD_PASSWORD_HASH" not in child_env
    assert "MODEL_SIGNING_KEY" not in child_env


@pytest.mark.asyncio
async def test_zombie_check_never_falls_back_to_path_lookup(monkeypatch, tmp_path):
    from robo_trader.clients import subprocess_ibkr_client as mod

    attacker_lsof = tmp_path / "lsof"
    attacker_lsof.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    attacker_lsof.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(
        mod,
        "_LSOF_EXECUTABLE_CANDIDATES",
        (tmp_path / "missing-approved-lsof",),
    )

    def unexpected_run(*args, **kwargs):
        raise AssertionError("PATH-resolved lsof must never execute")

    monkeypatch.setattr(mod.subprocess, "run", unexpected_run)

    client = mod.SubprocessIBKRClient()
    with pytest.raises(mod.IBKRError, match="Trusted absolute lsof executable is unavailable"):
        await client._check_zombie_connections(4002)


@pytest.mark.asyncio
async def test_zombie_check_fails_closed_without_exposing_lsof_stderr(monkeypatch):
    from robo_trader.clients import subprocess_ibkr_client as mod

    raw_secret = "DU1234567 broker-secret"
    monkeypatch.setattr(
        mod,
        "_trusted_lsof_executable",
        lambda: Path("/usr/sbin/lsof"),
    )
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args,
            1,
            stdout="",
            stderr=raw_secret,
        ),
    )

    client = mod.SubprocessIBKRClient()
    with pytest.raises(mod.IBKRError, match="Trusted lsof zombie check failed") as exc_info:
        await client._check_zombie_connections(4002)

    assert raw_secret not in str(exc_info.value)


def test_subprocess_client_source_has_relative_to_check():
    """Belt-and-suspenders: the source must contain the relative_to() guard
    that anchors VIRTUAL_ENV to the project root (audit IB-M1)."""
    src_path = PROJECT_ROOT / "robo_trader" / "clients" / "subprocess_ibkr_client.py"
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
    or remain fully quarantined (audit IB-L1 / PR-01)."""
    script_path = PROJECT_ROOT / "force_gateway_reconnect.sh"
    assert script_path.exists()

    text = script_path.read_text()

    is_quarantined = "DISABLED:" in text and "exit 2" in text
    assert is_quarantined or "mktemp" in text, (
        "force_gateway_reconnect.sh must be inert or use mktemp instead of " "a hardcoded path."
    )

    # The old hardcoded path must not be present as a write target. The
    # comment-only mention in cleanup is fine, but we should not see
    # `cat > /tmp/test_gateway_accept.py` or `python3 /tmp/test_gateway_accept.py`.
    assert (
        "cat > /tmp/test_gateway_accept.py" not in text
    ), "force_gateway_reconnect.sh must not write to a predictable path."
    assert (
        "python3 /tmp/test_gateway_accept.py" not in text
    ), "force_gateway_reconnect.sh must not exec from a predictable path."

    # An active implementation must clean up via trap. The quarantined stub
    # creates no temporary file.
    assert is_quarantined or (
        "trap" in text and 'rm -f "$TMPFILE"' in text
    ), "force_gateway_reconnect.sh must be inert or clean up its tempfile via trap."


# ---------------------------------------------------------------------------
# NEW-IB-H1.1 — ReadOnlyApi grep regex regression tests
# ---------------------------------------------------------------------------

# The exact bash grep used in START_TRADER.sh and scripts/start_gateway.sh.
# Keep this in sync with both scripts and with gateway_manager._READONLY_API_RE.
_BASH_RO_REGEX = r"^[[:space:]]*readonlyapi[[:space:]]*=[[:space:]]*yes[[:space:]]*$"


def _bash_grep_accepts(line: str) -> bool:
    """Run the actual bash grep used by the scripts against `line`.

    Returns True if grep matches (i.e. the safety check accepts the line).
    Uses stdin to feed `line` verbatim so embedded tabs survive intact.
    """
    cmd = f"grep -Eqi '{_BASH_RO_REGEX}'"
    result = subprocess.run(
        ["bash", "-c", cmd],
        input=(line + "\n").encode("utf-8"),
        capture_output=True,
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "line",
    [
        "ReadOnlyApi=yes",
        "readonlyapi=yes",
        "READONLYAPI=YES",
        "ReadOnlyApi=Yes",
        "ReadOnlyApi = yes",
        "  ReadOnlyApi=yes  ",
        "\tReadOnlyApi=yes",
    ],
)
def test_bash_grep_accepts_valid_readonly(line: str):
    """The anchored, case-insensitive grep must accept these forms — IBC honors
    all of them and Round 1's grep falsely rejected the lowercase / mixed-case
    variants (NEW-IB-H1.1)."""
    assert _bash_grep_accepts(line), f"bash grep should accept: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "ReadOnlyApi=yesno",  # The Round-1 bypass — must NOT match.
        "ReadOnlyApi=no",
        "ReadOnlyApi=",
        "ReadOnlyApi=true",
        "# ReadOnlyApi=yes",  # commented out
        "x ReadOnlyApi=yes",  # something before the key
        "ReadOnlyApi=yes extra",
        "AllowReadOnlyApi=yes",  # different key with same suffix
    ],
)
def test_bash_grep_rejects_invalid(line: str):
    """The grep must NOT accept lines that aren't a clean ReadOnlyApi=yes
    assignment. The Round-1 'ReadOnlyApi=yesno' bypass is the headline case."""
    assert not _bash_grep_accepts(line), f"bash grep should reject: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "ReadOnlyApi=yes",
        "readonlyapi=yes",
        "READONLYAPI=YES",
        "ReadOnlyApi=Yes",
        "ReadOnlyApi = yes",
        "  ReadOnlyApi=yes  ",
    ],
)
def test_python_regex_accepts_valid_readonly(line: str):
    """gateway_manager._READONLY_API_RE must accept the same set of valid
    forms as the bash grep (NEW-IB-H1.1)."""
    # Import lazily — the module imports a few heavy stdlib modules but no
    # third-party deps.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gateway_manager",
        PROJECT_ROOT / "scripts" / "gateway_manager.py",
    )
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)  # type: ignore[union-attr]

    assert gm._READONLY_API_RE.search(line + "\n"), f"Python regex should accept: {line!r}"


@pytest.mark.parametrize(
    "line",
    [
        "ReadOnlyApi=yesno",
        "ReadOnlyApi=no",
        "ReadOnlyApi=",
        "# ReadOnlyApi=yes",
        "AllowReadOnlyApi=yes",
        "ReadOnlyApi=yes extra",
    ],
)
def test_python_regex_rejects_invalid(line: str):
    """gateway_manager._READONLY_API_RE must reject the same set of invalid
    forms as the bash grep (NEW-IB-H1.1)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gateway_manager",
        PROJECT_ROOT / "scripts" / "gateway_manager.py",
    )
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)  # type: ignore[union-attr]

    assert not gm._READONLY_API_RE.search(line + "\n"), f"Python regex should reject: {line!r}"


def test_python_regex_finds_anywhere_in_multiline_config():
    """Realistic config file: many key=value lines, the readonly entry is one
    of them. The regex (with MULTILINE) must find it without being confused
    by surrounding lines."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gateway_manager",
        PROJECT_ROOT / "scripts" / "gateway_manager.py",
    )
    gm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gm)  # type: ignore[union-attr]

    config = (
        "# IBC config\n"
        "IbLoginId=foo\n"
        "TradingMode=paper\n"
        "ReadOnlyApi=yes\n"
        "AllowBlindTrading=no\n"
    )
    assert gm._READONLY_API_RE.search(config) is not None


# ---------------------------------------------------------------------------
# NEW-IB-M1.1 — interpreter resolve + allowlist
# ---------------------------------------------------------------------------


def test_subprocess_client_has_interpreter_allowlist():
    """The source must contain the realpath + allowlist enforcement for the
    worker interpreter (NEW-IB-M1.1)."""
    src_path = PROJECT_ROOT / "robo_trader" / "clients" / "subprocess_ibkr_client.py"
    text = src_path.read_text()
    assert (
        "_is_interpreter_path_safe" in text
    ), "subprocess_ibkr_client.py must define the interpreter allowlist helper."
    assert (
        "/Library/Frameworks/Python.framework" in text
    ), "Allowlist must include the macOS framework Python prefix."
    assert "/opt/homebrew" in text, "Allowlist must include Homebrew."
    assert (
        "_find_project_root" in text
    ), "subprocess_ibkr_client.py must use marker-file probing for project root."


def test_is_interpreter_path_safe_accepts_allowlist(tmp_path):
    """Realpaths under each trusted prefix must be accepted."""
    from robo_trader.clients.subprocess_ibkr_client import (
        _is_interpreter_path_safe,
    )

    project_root = tmp_path / "proj"
    project_root.mkdir()

    # Project-root-relative — accepted.
    venv_py = project_root / ".venv" / "bin" / "python3"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text("")
    assert _is_interpreter_path_safe(venv_py.resolve(), project_root)

    # System prefixes — accepted (these directories typically exist).
    for ok in [
        Path("/usr/bin/python3"),
        Path("/usr/local/bin/python3"),
        Path("/opt/homebrew/bin/python3"),
        Path("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"),
    ]:
        assert _is_interpreter_path_safe(ok, project_root), f"{ok} should be in the allowlist"


def test_is_interpreter_path_safe_rejects_outside_allowlist(tmp_path):
    """Realpaths outside the project root and outside trusted system prefixes
    must be rejected (NEW-IB-M1.1)."""
    from robo_trader.clients.subprocess_ibkr_client import (
        _is_interpreter_path_safe,
    )

    project_root = tmp_path / "proj"
    project_root.mkdir()

    # /tmp/evil — not under project_root, not under any allowlisted prefix.
    evil = tmp_path / "evil_venv" / "bin" / "python3"
    evil.parent.mkdir(parents=True)
    evil.write_text("")

    assert not _is_interpreter_path_safe(evil.resolve(), project_root)

    # ~ (user home) is not in the allowlist either.
    home_py = Path.home() / ".local" / "bin" / "python3"
    # Whether or not it exists, it shouldn't be in the allowlist.
    assert not _is_interpreter_path_safe(home_py, project_root)


@pytest.mark.asyncio
async def test_subprocess_client_rejects_symlinked_external_venv(monkeypatch, tmp_path):
    """An attacker who creates a venv inside the project root that symlinks
    its `bin/python3` to an external interpreter must be rejected by the
    realpath check (NEW-IB-M1.1).

    We can't trick `_find_project_root` cheaply — it's anchored to the real
    project tree — but we CAN test the helper directly, plus the wider Popen
    capture pattern used in the existing IB-M1 test verifies that the
    runtime guard fires.
    """
    pytest.importorskip("pytest_asyncio")

    from robo_trader.clients import subprocess_ibkr_client as mod

    # Create a venv structure inside a fake project root.
    fake_project_root = tmp_path / "proj"
    fake_project_root.mkdir()
    # Marker so _find_project_root would stop here.
    (fake_project_root / "pyproject.toml").write_text("")

    fake_venv = fake_project_root / ".venv"
    (fake_venv / "bin").mkdir(parents=True)
    inner_py = fake_venv / "bin" / "python3"

    # Outside-allowlist target.
    outside_py = tmp_path / "outside" / "python3"
    outside_py.parent.mkdir(parents=True)
    outside_py.write_text("#!/bin/sh\nexit 0\n")
    outside_py.chmod(0o755)

    # Symlink the venv python to the outside binary.
    inner_py.symlink_to(outside_py)

    # The realpath check must reject this.
    resolved = inner_py.resolve()
    assert resolved == outside_py.resolve()
    assert not mod._is_interpreter_path_safe(resolved, fake_project_root), (
        "Realpath of the symlink lands outside the allowlist; client must " "refuse to exec it."
    )


# ---------------------------------------------------------------------------
# Followup-audit findings (SECURITY_AUDIT_2026-05-10_FOLLOWUP.md section 2.E)
# ---------------------------------------------------------------------------


def test_start_runner_sh_enforces_readonly_ibn_m3():
    """IBN-M3: the legacy dashboard launcher must not bypass read-only safety.

    PR-01 quarantines this alternate path entirely.  If it is ever restored,
    the original anchored ReadOnlyApi validation remains mandatory.
    """
    import pathlib

    text = pathlib.Path("scripts/start_runner.sh").read_text()
    is_quarantined = "DISABLED:" in text and "exit 2" in text and "pkill" not in text
    has_readonly_guard = "grep -Eqi" in text and "readonlyapi" in text.lower()
    assert is_quarantined or has_readonly_guard, (
        "start_runner.sh must be inert or include the same ReadOnlyApi guard "
        "as the authoritative launcher"
    )


def test_gateway_manager_start_refuses_without_readonly_ibn_h1():
    """IBN-H1: gateway_manager.py:start_gateway must refuse to start the
    Gateway if the IBC config does not set ReadOnlyApi=yes. The check is
    structural — we assert the regex is referenced in start_gateway and that
    it returns False on the missing-readonly path.
    """
    import inspect

    import scripts.gateway_manager as gm

    source = inspect.getsource(gm.start_gateway)
    assert (
        "_ibc_safety_file_error" in source
    ), "start_gateway must validate the complete IBC safety contract before launching Gateway"
    # It's the same regex used by show_status, so cross-check it works.
    assert gm._READONLY_API_RE.search("ReadOnlyApi=yes") is not None
    assert gm._READONLY_API_RE.search("ReadOnlyApi=no") is None


@pytest.mark.parametrize(
    ("config_text", "accepted"),
    [
        ("ReadOnlyApi=yes\nTradingMode=paper\n", True),
        (" readonlyapi = YES \n tradingmode = PAPER \n", True),
        ("TradingMode=paper\n", False),
        ("ReadOnlyApi=yes\n", False),
        ("ReadOnlyApi=no\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=live\n", False),
        ("ReadOnlyApi=yes\nReadOnlyApi=yes\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nReadOnlyApi=no\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=paper\nTradingMode=paper\n", False),
        ("ReadOnlyApi=yes\nTradingMode=paper\nTradingMode=live\n", False),
    ],
)
def test_gateway_manager_requires_unambiguous_ibc_safety_settings(config_text, accepted):
    import scripts.gateway_manager as gm

    assert (gm._ibc_safety_config_error(config_text) is None) is accepted


def test_gateway_manager_restart_validates_before_stopping_gateway(monkeypatch, tmp_path):
    import scripts.gateway_manager as gm

    unsafe_config = tmp_path / "config.ini"
    unsafe_config.write_text(
        "ReadOnlyApi=yes\nReadOnlyApi=no\nTradingMode=paper\n",
        encoding="utf-8",
    )
    stopped = False

    def record_stop():
        nonlocal stopped
        stopped = True
        return True

    monkeypatch.setattr(gm, "IBC_CONFIG", unsafe_config)
    monkeypatch.setattr(gm, "stop_gateway", record_stop)

    assert gm.restart_gateway("paper") is False
    assert stopped is False


def test_gateway_manager_start_validates_before_running_gateway_shortcut(monkeypatch, tmp_path):
    import scripts.gateway_manager as gm

    unsafe_config = tmp_path / "config.ini"
    unsafe_config.write_text(
        "ReadOnlyApi=yes\nTradingMode=paper\nTradingMode=live\n",
        encoding="utf-8",
    )
    process_checked = False

    def record_process_check():
        nonlocal process_checked
        process_checked = True
        return True

    monkeypatch.setattr(gm, "IBC_CONFIG", unsafe_config)
    monkeypatch.setattr(gm, "is_gateway_running", record_process_check)

    assert gm.start_gateway("paper") is False
    assert process_checked is False


def test_gateway_manager_start_uses_env_allowlist_ibn_h2():
    """IBN-H2: gateway_manager.start_gateway must build env from a minimal
    allowlist instead of os.environ.copy() (which propagates dashboard auth
    hashes, model-signing keys, etc. into the IBC subprocess that doesn't
    need them).
    """
    import inspect

    import scripts.gateway_manager as gm

    source = inspect.getsource(gm.start_gateway)
    assert "_IBC_ENV_ALLOWLIST" in source, (
        "start_gateway must build env from an allowlist (IBN-H2). "
        "Without an allowlist, parent-process secrets propagate to IBC."
    )
    # Strip comments and inspect non-comment lines only — the call
    # os.environ.copy() must not be used to construct env, even though the
    # symbol may appear in an explanatory comment.
    code_only = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert (
        "os.environ.copy()" not in code_only
    ), "start_gateway code must not call os.environ.copy() (IBN-H2)."


@pytest.mark.parametrize(
    "argv",
    [
        ["gateway_manager.py", "start", "--paper"],
        ["gateway_manager.py", "stop"],
        ["gateway_manager.py", "restart", "--paper"],
        ["gateway_manager.py", "restart", "--help"],
        ["gateway_manager.py", "clear-zombies"],
    ],
)
def test_gateway_manager_operator_lifecycle_commands_fail_closed(monkeypatch, capsys, argv):
    """Operators must enter every Gateway lifecycle through START_TRADER."""
    import scripts.gateway_manager as gm

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.delenv(gm._INTERNAL_LIFECYCLE_ENV, raising=False)
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(
        gm,
        "start_gateway",
        lambda *args, **kwargs: pytest.fail("direct lifecycle command executed"),
    )
    monkeypatch.setattr(
        gm,
        "stop_gateway",
        lambda *args, **kwargs: pytest.fail("direct lifecycle command executed"),
    )
    monkeypatch.setattr(
        gm,
        "restart_gateway",
        lambda *args, **kwargs: pytest.fail("direct lifecycle command executed"),
    )
    monkeypatch.setattr(
        gm,
        "get_zombie_connections",
        lambda *args, **kwargs: pytest.fail("direct lifecycle command executed"),
    )

    assert gm.main() == 2
    captured = capsys.readouterr()
    assert "./START_TRADER.sh" in captured.err
    assert "Refusing direct Gateway lifecycle command" in captured.err


def test_gateway_manager_status_remains_operator_available(monkeypatch):
    """The read-only status command remains callable without an internal marker."""
    import scripts.gateway_manager as gm

    called = False

    def record_status():
        nonlocal called
        called = True

    monkeypatch.setattr(sys, "argv", ["gateway_manager.py", "status"])
    monkeypatch.delenv(gm._INTERNAL_LIFECYCLE_ENV, raising=False)
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(gm, "show_status", record_status)

    assert gm.main() == 0
    assert called is True


def test_gateway_manager_internal_recovery_marker_allows_restart(monkeypatch):
    """Persistent recovery may retain the existing internal restart function."""
    import scripts.gateway_manager as gm

    called = False

    def record_restart(mode):
        nonlocal called
        called = True
        assert mode == "paper"
        return True

    monkeypatch.setattr(sys, "argv", ["gateway_manager.py", "restart", "--paper"])
    monkeypatch.setenv(gm._INTERNAL_LIFECYCLE_ENV, gm._INTERNAL_LIFECYCLE_VALUE)
    monkeypatch.setattr(gm, "_parent_is_runner", lambda: True)
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    lifecycle_events = []

    class FakeLifecycleLock:
        def __init__(self, path):
            lifecycle_events.append(("created", path))

        def acquire(self):
            lifecycle_events.append(("acquired", None))
            return True

        def release(self):
            lifecycle_events.append(("released", None))

    monkeypatch.setattr(gm, "RuntimeLifecycleLock", FakeLifecycleLock)
    monkeypatch.setattr(gm, "restart_gateway", record_restart)

    assert gm.main() == 0
    assert called is True
    assert [event[0] for event in lifecycle_events] == ["created", "acquired", "released"]


def test_gateway_manager_internal_recovery_rejects_held_lifecycle_lock(monkeypatch, capsys):
    """A launcher-owned lock blocks an otherwise-authorized recovery child."""
    import scripts.gateway_manager as gm

    class HeldLifecycleLock:
        def __init__(self, path):
            self.path = path

        def acquire(self):
            return False

        def release(self):
            pytest.fail("an unacquired lock must not be released")

    monkeypatch.setattr(sys, "argv", ["gateway_manager.py", "restart", "--paper"])
    monkeypatch.setenv(gm._INTERNAL_LIFECYCLE_ENV, gm._INTERNAL_LIFECYCLE_VALUE)
    monkeypatch.setattr(gm, "_parent_is_runner", lambda: True)
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(gm, "RuntimeLifecycleLock", HeldLifecycleLock)
    monkeypatch.setattr(
        gm,
        "restart_gateway",
        lambda *args, **kwargs: pytest.fail("concurrent recovery executed"),
    )

    assert gm.main() == 75
    assert "Refusing concurrent Gateway recovery" in capsys.readouterr().err


def test_gateway_manager_marker_without_runner_parent_is_rejected(monkeypatch, capsys):
    """An operator cannot unlock lifecycle commands by setting the marker."""
    import scripts.gateway_manager as gm

    monkeypatch.setattr(sys, "argv", ["gateway_manager.py", "restart", "--paper"])
    monkeypatch.setenv(gm._INTERNAL_LIFECYCLE_ENV, gm._INTERNAL_LIFECYCLE_VALUE)
    monkeypatch.setattr(gm, "_parent_is_runner", lambda: False)
    monkeypatch.setattr(gm, "PLATFORM", "Darwin")
    monkeypatch.setattr(
        gm,
        "restart_gateway",
        lambda *args, **kwargs: pytest.fail("direct lifecycle command executed"),
    )

    assert gm.main() == 2
    assert "Refusing direct Gateway lifecycle command" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/usr/bin/python3 -m robo_trader.runner_async --symbols AAPL", True),
        ("/usr/bin/python3 /repo/robo_trader/runner_async.py", True),
        ("/bin/zsh -l", False),
        ("/usr/bin/python3 scripts/gateway_manager.py restart --paper", False),
    ],
)
def test_gateway_manager_parent_runner_check(monkeypatch, command, expected):
    """Internal lifecycle authorization is tied to the actual parent command."""
    import scripts.gateway_manager as gm

    monkeypatch.setattr(
        gm.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, command, ""),
    )

    assert gm._parent_is_runner() is expected


def test_gateway_manager_public_help_hides_lifecycle_commands(monkeypatch, capsys):
    """The public CLI advertises status only."""
    import scripts.gateway_manager as gm

    monkeypatch.setattr(sys, "argv", ["gateway_manager.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        gm.main()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "{status}" in help_text
    assert "{start" not in help_text
    assert "clear-zombies" not in help_text


def test_gateway_manager_lifecycle_lock_matches_authoritative_launcher():
    """Shell startup and runner recovery use the same atomic lock module."""
    import scripts.gateway_manager as gm

    launcher = (PROJECT_ROOT / "START_TRADER.sh").read_text()

    assert '"$SCRIPT_DIR/robo_trader/runtime_lifecycle_lock.py"' in launcher
    assert 'mkdir "$STARTUP_LOCK' not in launcher
    assert gm._RUNTIME_LIFECYCLE_LOCK_PATH == Path(f"/tmp/robotrader-runtime-{os.getuid()}.lock")


def test_robust_connection_sets_internal_marker_for_gateway_recovery(monkeypatch):
    """The runner's persistent recovery explicitly identifies its child."""
    import robo_trader.utils.robust_connection as rc

    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    monkeypatch.setattr(rc.os.path, "exists", lambda path: True)
    monkeypatch.setattr(rc.subprocess, "run", fake_run)

    assert rc.restart_gateway_for_zombies(port=4002, timeout=1) is False
    assert "restart" in captured["argv"]
    assert captured["env"][rc._INTERNAL_GATEWAY_RECOVERY_ENV] == rc._INTERNAL_GATEWAY_RECOVERY_VALUE


# ---------------------------------------------------------------------------
# Branch-audit (claude/security-audit-5tFIY) round-3
# ---------------------------------------------------------------------------


def test_robust_connection_refuses_cert_none_for_non_loopback_b_11():
    """B-11: the SSL transport path in robust_connection must refuse
    CERT_NONE to non-loopback hosts unless IBKR_GATEWAY_CAFILE is set.
    """
    import inspect

    import robo_trader.utils.robust_connection as rc

    source = inspect.getsource(rc)
    # The CERT_NONE branch must check for loopback OR cafile.
    assert (
        "IBKR_GATEWAY_CAFILE" in source
    ), "robust_connection must read IBKR_GATEWAY_CAFILE for cert pinning (B-11)"
    code_only = "\n".join(line.split("#", 1)[0] for line in source.splitlines())
    assert (
        "loopback" in code_only.lower() or "127.0.0.1" in code_only
    ), "robust_connection must refuse CERT_NONE on non-loopback hosts (B-11)"

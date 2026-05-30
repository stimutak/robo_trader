"""
Operations / ops-hardening regression tests.

These guard against the failure mode that took the trader down overnight
on 2026-05-11: the launchd watchdog was not loaded on the dev machine,
and nothing in the setup docs told the operator to load it.

Each test pins one piece of the install/recovery contract so a future
refactor cannot silently regress it.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PLIST_PATH = SCRIPTS_DIR / "com.robotrader.watchdog.plist"
WATCHDOG_SH = SCRIPTS_DIR / "watchdog.sh"
INSTALL_SH = SCRIPTS_DIR / "install_watchdog.sh"
START_TRADER_SH = PROJECT_ROOT / "START_TRADER.sh"
DEV_SETUP_MD = PROJECT_ROOT / "DEV_SETUP.md"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"


# ---------------------------------------------------------------------------
# Plist contract
# ---------------------------------------------------------------------------


def _load_plist() -> dict:
    assert PLIST_PATH.exists(), f"missing plist: {PLIST_PATH}"
    with PLIST_PATH.open("rb") as fh:
        return plistlib.load(fh)


def test_watchdog_plist_paths_exist():
    """ProgramArguments[0], StandardOut, StandardError must resolve to real,
    writable locations. Otherwise launchd silently fails."""
    plist = _load_plist()

    program_args = plist.get("ProgramArguments")
    assert program_args, "ProgramArguments missing or empty"
    program = Path(program_args[0])
    assert program.exists(), f"watchdog script not found at {program}"
    # Must be executable for launchd to invoke it.
    assert os.access(program, os.X_OK), f"{program} not executable"

    for key in ("StandardOutPath", "StandardErrorPath"):
        out_path = Path(plist[key])
        # The log file itself doesn't have to exist yet, but its directory
        # must exist and be writable — otherwise launchd can't redirect.
        assert out_path.parent.exists(), f"{key} parent missing: {out_path.parent}"
        assert os.access(out_path.parent, os.W_OK), f"{key} parent not writable: {out_path.parent}"

    wd = Path(plist.get("WorkingDirectory", ""))
    assert wd.exists(), f"WorkingDirectory missing: {wd}"

    # Audit hardening from Round-2: agent must be pinned to the GUI session
    # of a real user, not run as root or in a background session.
    assert plist.get("UserName"), "UserName must be set on the agent"
    assert (
        plist.get("LimitLoadToSessionType") == "Aqua"
    ), "LimitLoadToSessionType must be 'Aqua' so the agent runs in the GUI session"


def test_watchdog_plist_lints_ok():
    """`plutil -lint` must pass; otherwise launchctl refuses to load it."""
    result = subprocess.run(
        ["plutil", "-lint", str(PLIST_PATH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"plutil -lint failed: {result.stdout} {result.stderr}"


# ---------------------------------------------------------------------------
# Install script contract
# ---------------------------------------------------------------------------


def test_install_watchdog_script_exists_and_is_executable():
    assert INSTALL_SH.exists(), f"missing installer: {INSTALL_SH}"
    assert os.access(INSTALL_SH, os.X_OK), "install_watchdog.sh must be executable"


def test_install_watchdog_script_bash_syntax():
    """`bash -n` must pass; otherwise running the installer crashes the operator."""
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_install_watchdog_script_is_idempotent(tmp_path):
    """Run the installer twice into a fake LaunchAgents dir; both runs must
    exit 0. SKIP_LAUNCHCTL=1 skips the actual launchctl load (which would
    require user-session context and side-effect the real system)."""
    fake_la = tmp_path / "LaunchAgents"
    env = os.environ.copy()
    env["LAUNCH_AGENTS_DIR"] = str(fake_la)
    env["SKIP_LAUNCHCTL"] = "1"

    for run_no in (1, 2):
        result = subprocess.run(
            ["bash", str(INSTALL_SH)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert result.returncode == 0, (
            f"install_watchdog.sh run #{run_no} failed: "
            f"stdout={result.stdout!r} stderr={result.stderr!r}"
        )
        copied = fake_la / "com.robotrader.watchdog.plist"
        assert copied.exists(), f"plist not copied to fake LaunchAgents on run #{run_no}"


# ---------------------------------------------------------------------------
# Docs contract — operators must be TOLD to install the watchdog
# ---------------------------------------------------------------------------


def test_dev_setup_mentions_watchdog_install():
    """DEV_SETUP.md must point operators at the install script. If this
    regresses, fresh machines will repeat the 2026-05-11 outage."""
    assert DEV_SETUP_MD.exists(), "DEV_SETUP.md missing"
    content = DEV_SETUP_MD.read_text()
    assert (
        "install_watchdog.sh" in content
    ), "DEV_SETUP.md must reference scripts/install_watchdog.sh"
    # Also assert it's marked as required, not optional.
    lowered = content.lower()
    assert (
        "required" in lowered or "not optional" in lowered
    ), "DEV_SETUP.md must mark the watchdog install as required"


def test_claude_md_mentions_watchdog_install():
    """CLAUDE.md must tell future Claude sessions about the install step."""
    assert CLAUDE_MD.exists(), "CLAUDE.md missing"
    content = CLAUDE_MD.read_text()
    assert "install_watchdog.sh" in content, "CLAUDE.md must reference scripts/install_watchdog.sh"


# ---------------------------------------------------------------------------
# START_TRADER.sh must warn loudly if the watchdog isn't loaded
# ---------------------------------------------------------------------------


def test_start_trader_warns_if_watchdog_not_loaded():
    assert START_TRADER_SH.exists(), "START_TRADER.sh missing"
    src = START_TRADER_SH.read_text()
    assert (
        "launchctl list" in src
    ), "START_TRADER.sh must call `launchctl list` to detect a missing watchdog"
    assert "robotrader" in src, "START_TRADER.sh launchctl check must grep for 'robotrader'"
    assert (
        "WARNING" in src
    ), "START_TRADER.sh must print a WARNING block when the watchdog is missing"
    assert (
        "install_watchdog.sh" in src
    ), "START_TRADER.sh WARNING must reference scripts/install_watchdog.sh"


# ---------------------------------------------------------------------------
# watchdog.sh shell syntax — guards against the bug fixed on 2026-05-12
# (arithmetic expansion with quoted variables failed silently in a crash loop)
# ---------------------------------------------------------------------------


def test_watchdog_sh_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", str(WATCHDOG_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"watchdog.sh bash -n failed: {result.stderr}"

"""Tests for ``_enforce_preflight_or_exit`` (Q11.6 server-side enforcement).

The runner refuses to start unless ``data/.preflight_last_ok`` exists and was
written within the last ``max_age_seconds`` (default 300s). The flag file is
written by ``scripts/preflight_check.py`` on every clean exit (PASS or
``--force`` bypass) per Q11.4. The escape hatch ``PREFLIGHT_ENFORCEMENT=off``
short-circuits the check entirely.

Exit codes (distinct from preflight CLI's 1/2/3):
- 4 = flag file missing entirely
- 5 = flag file present but stale
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from robo_trader.runner_async import _enforce_preflight_or_exit


def _make_flag(project_root: Path, age_seconds: float = 0.0) -> Path:
    """Create ``project_root/data/.preflight_last_ok`` with the given age."""
    (project_root / "data").mkdir(parents=True, exist_ok=True)
    path = project_root / "data" / ".preflight_last_ok"
    path.write_text("2026-05-24T12:00:00-04:00\n", encoding="utf-8")
    if age_seconds > 0:
        target = time.time() - age_seconds
        os.utime(path, (target, target))
    return path


# --------------------------------------------------------------------------- #
# 1. Missing flag → exit 4
# --------------------------------------------------------------------------- #


def test_missing_flag_exits_with_code_4(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # No flag file at all.
    with pytest.raises(SystemExit) as exc_info:
        _enforce_preflight_or_exit(project_root=tmp_path)

    assert exc_info.value.code == 4
    err = capsys.readouterr().err
    assert "PREFLIGHT NOT RUN" in err
    assert "runner refuses to start" in err
    # Helpful guidance for the operator.
    assert "START_TRADER.sh" in err
    assert "PREFLIGHT_ENFORCEMENT=off" in err
    # Should show the expected path so they know where it would go.
    assert str(tmp_path / "data" / ".preflight_last_ok") in err


# --------------------------------------------------------------------------- #
# 2. Fresh flag → returns cleanly
# --------------------------------------------------------------------------- #


def test_fresh_flag_returns_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_flag(tmp_path, age_seconds=0)

    # Should NOT raise SystemExit.
    _enforce_preflight_or_exit(project_root=tmp_path)

    # No warning / error printed for the happy path.
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


# --------------------------------------------------------------------------- #
# 3. Stale flag → exit 5
# --------------------------------------------------------------------------- #


def test_stale_flag_exits_with_code_5(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 10 minutes old, default threshold is 5 minutes.
    _make_flag(tmp_path, age_seconds=600)

    with pytest.raises(SystemExit) as exc_info:
        _enforce_preflight_or_exit(project_root=tmp_path)

    assert exc_info.value.code == 5
    err = capsys.readouterr().err
    assert "PREFLIGHT STALE" in err
    # Should report how many minutes old and what the threshold is.
    assert "10 minutes ago" in err
    assert "threshold 5 min" in err
    assert "PREFLIGHT_ENFORCEMENT=off" in err


# --------------------------------------------------------------------------- #
# 4. Flag exactly inside the threshold → returns cleanly
# --------------------------------------------------------------------------- #


def test_flag_just_inside_threshold_returns_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1 second inside the 300s default → fresh.
    _make_flag(tmp_path, age_seconds=299)

    _enforce_preflight_or_exit(project_root=tmp_path)

    captured = capsys.readouterr()
    assert captured.err == ""


# --------------------------------------------------------------------------- #
# 5. PREFLIGHT_ENFORCEMENT=off → bypasses with warning, even when flag missing
# --------------------------------------------------------------------------- #


def test_env_bypass_skips_check_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PREFLIGHT_ENFORCEMENT", "off")

    # No flag file — would normally exit 4. The bypass must override.
    _enforce_preflight_or_exit(project_root=tmp_path)

    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "PREFLIGHT_ENFORCEMENT=off" in err
    assert "bypassed" in err


def test_env_bypass_case_insensitive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Be lenient — accept "OFF", "Off", etc.
    monkeypatch.setenv("PREFLIGHT_ENFORCEMENT", "OFF")

    _enforce_preflight_or_exit(project_root=tmp_path)

    err = capsys.readouterr().err
    assert "WARNING" in err


def test_env_value_other_than_off_does_not_bypass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Any value other than "off" must still enforce. Guards against typos
    # like PREFLIGHT_ENFORCEMENT=false or =0 silently disabling the gate.
    monkeypatch.setenv("PREFLIGHT_ENFORCEMENT", "false")

    with pytest.raises(SystemExit) as exc_info:
        _enforce_preflight_or_exit(project_root=tmp_path)

    assert exc_info.value.code == 4


# --------------------------------------------------------------------------- #
# 6. Custom max_age_seconds respected
# --------------------------------------------------------------------------- #


def test_custom_max_age_seconds_tighter_threshold(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 2 minutes old. Default 300s would pass; max_age=60 should fail.
    _make_flag(tmp_path, age_seconds=120)

    with pytest.raises(SystemExit) as exc_info:
        _enforce_preflight_or_exit(project_root=tmp_path, max_age_seconds=60)

    assert exc_info.value.code == 5
    err = capsys.readouterr().err
    assert "threshold 1 min" in err


def test_custom_max_age_seconds_looser_threshold(tmp_path: Path) -> None:
    # 10 minutes old. Default 300s would fail; max_age=1200 should pass.
    _make_flag(tmp_path, age_seconds=600)

    _enforce_preflight_or_exit(project_root=tmp_path, max_age_seconds=1200)


# --------------------------------------------------------------------------- #
# 7. Custom project_root respected (the parameter is actually used)
# --------------------------------------------------------------------------- #


def test_custom_project_root_used_for_flag_lookup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Build flag under a non-default project root and verify the check
    # reads from that location (not from cwd or the runner's own location).
    other_root = tmp_path / "some_other_project"
    other_root.mkdir()
    _make_flag(other_root, age_seconds=0)

    # Passing other_root: should find the flag.
    _enforce_preflight_or_exit(project_root=other_root)
    assert capsys.readouterr().err == ""

    # Passing tmp_path (where no flag exists): should fail with code 4
    # AND the error message should reference tmp_path's data dir, not
    # the runner's default location — proving the parameter is honored.
    with pytest.raises(SystemExit) as exc_info:
        _enforce_preflight_or_exit(project_root=tmp_path)
    assert exc_info.value.code == 4
    err = capsys.readouterr().err
    assert str(tmp_path / "data" / ".preflight_last_ok") in err

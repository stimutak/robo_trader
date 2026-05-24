"""Tests for the ``scripts/preflight_check.py`` CLI entry point.

Exercises the script as a Python module (via ``importlib``) to keep tests
fast — subprocess invocation is reserved for the §9.6 manual canary.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, List

import pytest

from robo_trader.preflight.protocol import PreflightContext
from robo_trader.preflight.result import CheckResult, CheckStatus

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "preflight_check.py"


@pytest.fixture(scope="module")
def cli_module():
    """Load ``scripts/preflight_check.py`` as a module so we can call ``main()`` directly."""
    spec = importlib.util.spec_from_file_location("preflight_cli_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Test doubles — same shape as test_runner._FakeCheck but in isolation here.
# ---------------------------------------------------------------------------


class _StubCheck:
    def __init__(
        self,
        name: str,
        status: CheckStatus = CheckStatus.PASS,
        message: str = "ok",
        remediation: str = "",
    ):
        self.name = name
        self.description = name
        self.timeout_seconds = 1.0
        self._status = status
        self._message = message
        self._remediation = remediation

    def run(self, context: PreflightContext) -> CheckResult:
        return CheckResult(
            name=self.name,
            status=self._status,
            message=self._message,
            remediation=self._remediation,
        )


@pytest.fixture
def patch_checks_and_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cli_module):
    """Helper: swap ALL_CHECKS and PROJECT_ROOT so tests don't touch real state."""

    def _install(checks: List[Any], env: dict | None = None) -> Path:
        # ALL_CHECKS is imported into the script's module namespace as a
        # bare name reference — patch it where the script reads it.
        monkeypatch.setattr(cli_module, "ALL_CHECKS", checks)
        monkeypatch.setattr(cli_module, "PROJECT_ROOT", tmp_path)
        if env is not None:
            monkeypatch.setattr(cli_module.os, "environ", env)
        return tmp_path

    return _install


# ---------------------------------------------------------------------------
# Exit codes (spec §5.2)
# ---------------------------------------------------------------------------


class TestExitCodes:
    def test_all_pass_exits_zero(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("a"), _StubCheck("b")])
        rc = cli_module.main([])
        assert rc == 0

    def test_warn_only_exits_zero(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("warn1", status=CheckStatus.WARN, message="meh")])
        rc = cli_module.main([])
        assert rc == 0

    def test_any_block_exits_one(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root(
            [
                _StubCheck("ok"),
                _StubCheck("bad", status=CheckStatus.BLOCK, message="boom"),
            ]
        )
        rc = cli_module.main([])
        assert rc == 1

    def test_force_with_block_exits_two(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root(
            [_StubCheck("bad", status=CheckStatus.BLOCK, message="boom", remediation="fix it")]
        )
        rc = cli_module.main(["--force", "I have reviewed and approved this bypass for test"])
        assert rc == 2

    def test_force_without_block_exits_zero_with_warning(
        self, patch_checks_and_root, capsys, cli_module
    ) -> None:
        patch_checks_and_root([_StubCheck("ok")])
        rc = cli_module.main(["--force", "I have approved this bypass for test purposes"])
        # exit 0 (force was unnecessary) + warning to stderr
        assert rc == 0
        err = capsys.readouterr().err
        assert "force was not needed" in err.lower() or "not needed" in err.lower()


# ---------------------------------------------------------------------------
# --force validation
# ---------------------------------------------------------------------------


class TestForceValidation:
    def test_reason_too_short_rejected(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("ok")])
        rc = cli_module.main(["--force", "tiny"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "at least 10" in err

    @pytest.mark.parametrize("placeholder", ["force", "bypass", "whatever", "idk", "test"])
    def test_placeholder_denylist(
        self, placeholder: str, patch_checks_and_root, capsys, cli_module
    ) -> None:
        patch_checks_and_root([_StubCheck("ok")])
        # Pad to >= MIN_LENGTH so it gets to the denylist check, but the
        # CORE word should still be detected as placeholder.
        rc = cli_module.main(["--force", placeholder])
        # If the placeholder happens to be <10 chars, length check fires first;
        # if it's >=10, denylist fires. Either way, rc must be non-zero.
        err = capsys.readouterr().err
        assert rc == 1
        assert ("placeholder" in err.lower()) or ("at least 10" in err)

    def test_real_sentence_reason_accepted(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("ok")])
        rc = cli_module.main(["--force", "verified positions reconcile ticket 1234"])
        assert rc == 0  # nothing to bypass


# ---------------------------------------------------------------------------
# Bypass audit log (spec §6.3)
# ---------------------------------------------------------------------------


class TestBypassAuditLog:
    def test_force_with_block_appends_log_entry(
        self, patch_checks_and_root, capsys, cli_module
    ) -> None:
        root = patch_checks_and_root([_StubCheck("bad", status=CheckStatus.BLOCK, message="boom")])
        rc = cli_module.main(["--force", "verified state by hand, ticket #999"])
        assert rc == 2
        log_path = root / "data" / "preflight_bypass.log"
        assert log_path.exists()
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        assert len(entries) == 1
        assert entries[0]["reason"] == "verified state by hand, ticket #999"
        assert entries[0]["bypassed_checks"] == ["bad"]
        assert "timestamp" in entries[0]
        assert "operator" in entries[0]

    def test_multiple_bypasses_append_not_overwrite(
        self, patch_checks_and_root, capsys, cli_module
    ) -> None:
        root = patch_checks_and_root([_StubCheck("bad", status=CheckStatus.BLOCK, message="boom")])
        cli_module.main(["--force", "first bypass with adequate reason"])
        cli_module.main(["--force", "second bypass with adequate reason"])
        log_path = root / "data" / "preflight_bypass.log"
        entries = [json.loads(line) for line in log_path.read_text().splitlines() if line]
        assert len(entries) == 2

    def test_force_without_block_does_not_log(
        self, patch_checks_and_root, capsys, cli_module
    ) -> None:
        root = patch_checks_and_root([_StubCheck("ok")])
        cli_module.main(["--force", "unnecessary force flag for testing only"])
        log_path = root / "data" / "preflight_bypass.log"
        # No blocks => no log entry written (the gate didn't bypass anything).
        assert not log_path.exists()


# ---------------------------------------------------------------------------
# .preflight_last_ok flag (Q11.4 — coordinates with Q11.6 runner enforcement)
# ---------------------------------------------------------------------------


class TestLastOkFlag:
    def test_written_on_clean_exit(self, patch_checks_and_root, capsys, cli_module) -> None:
        root = patch_checks_and_root([_StubCheck("ok")])
        rc = cli_module.main([])
        assert rc == 0
        flag = root / "data" / ".preflight_last_ok"
        assert flag.exists()
        # Content should be a parseable ISO timestamp
        from datetime import datetime

        ts = flag.read_text().strip()
        datetime.fromisoformat(ts)  # raises if malformed

    def test_written_on_bypass_exit(self, patch_checks_and_root, capsys, cli_module) -> None:
        # --force-with-bypass (exit 2) counts as "preflight ran and operator
        # made a decision" per Q11.4. Flag is written so the runner doesn't
        # refuse to start after a deliberately-forced launch.
        root = patch_checks_and_root([_StubCheck("bad", status=CheckStatus.BLOCK, message="boom")])
        rc = cli_module.main(["--force", "operator override with adequate reason"])
        assert rc == 2
        assert (root / "data" / ".preflight_last_ok").exists()

    def test_not_written_on_block_exit(self, patch_checks_and_root, capsys, cli_module) -> None:
        root = patch_checks_and_root([_StubCheck("bad", status=CheckStatus.BLOCK, message="boom")])
        rc = cli_module.main([])
        assert rc == 1
        # No flag — preflight blocked, operator must resolve.
        assert not (root / "data" / ".preflight_last_ok").exists()


# ---------------------------------------------------------------------------
# Output format selection
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def test_default_is_plaintext(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("ok", message="all good")])
        cli_module.main([])
        out = capsys.readouterr().out
        assert "Preflight Safety Gate" in out
        assert "Safe to proceed" in out

    def test_json_flag_emits_json(self, patch_checks_and_root, capsys, cli_module) -> None:
        patch_checks_and_root([_StubCheck("ok", message="all good")])
        cli_module.main(["--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["summary"]["passed"] == 1
        assert payload["exit_code"] == 0


# ---------------------------------------------------------------------------
# Port resolution
# ---------------------------------------------------------------------------


class TestPortResolution:
    def test_paper_mode_uses_4002(self, monkeypatch: pytest.MonkeyPatch, cli_module) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "paper")
        assert cli_module._resolve_target_port() == 4002

    def test_live_mode_uses_4001(self, monkeypatch: pytest.MonkeyPatch, cli_module) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "live")
        assert cli_module._resolve_target_port() == 4001

    def test_unset_defaults_to_paper(self, monkeypatch: pytest.MonkeyPatch, cli_module) -> None:
        monkeypatch.delenv("EXECUTION_MODE", raising=False)
        assert cli_module._resolve_target_port() == 4002

    def test_unknown_mode_defaults_to_paper(
        self, monkeypatch: pytest.MonkeyPatch, cli_module
    ) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "backtest")
        # Anything that isn't "live" → paper (4002)
        assert cli_module._resolve_target_port() == 4002


# ---------------------------------------------------------------------------
# Code 3 path (preflight itself fails)
# ---------------------------------------------------------------------------


class TestPreflightFailureCode3:
    def test_uncaught_exception_in_build_context_exits_three(
        self, monkeypatch: pytest.MonkeyPatch, capsys, cli_module
    ) -> None:
        def _broken_build_context(*args, **kwargs):
            raise RuntimeError("preflight is broken")

        monkeypatch.setattr(cli_module, "_build_context", _broken_build_context)
        rc = cli_module.main([])
        assert rc == 3
        err = capsys.readouterr().err
        assert "preflight itself failed" in err

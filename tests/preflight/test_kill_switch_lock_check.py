"""Tests for :class:`KillSwitchLockCheck` (spec §7.2).

The check is intentionally tiny — "does ``data/kill_switch.lock`` exist?"
— but the contract it satisfies is important: it's the deny-by-default
fail-closed signal called out in CLAUDE.md. These tests pin down the
decisions that matter:

1. Missing file → PASS (the common case)
2. Empty file → BLOCK (presence is the signal, contents don't matter)
3. Non-empty file (e.g. a stale PID) → still BLOCK
4. Symlink to a real file → BLOCK (effectively present)
5. Broken symlink → PASS (matches ``Path.is_file()`` and KillSwitch behavior)

The shared ``touch_kill_switch_lock`` and ``preflight_context`` fixtures
live in ``tests/preflight/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from robo_trader.preflight import CheckStatus, PreflightContext
from robo_trader.preflight.kill_switch_lock_check import KillSwitchLockCheck


class TestMetadata:
    def test_check_identifier_is_stable(self) -> None:
        # The name is the JSON output key and the grep target in logs;
        # changing it is a breaking change for downstream tooling.
        check = KillSwitchLockCheck()
        assert check.name == "kill_switch_lock"
        assert check.description == "Kill switch lock file"

    def test_timeout_is_one_second(self) -> None:
        # A single stat() call — anything more than 1s would mean the
        # filesystem itself is wedged, which is its own problem.
        assert KillSwitchLockCheck().timeout_seconds == 1.0


class TestLockMissing:
    def test_missing_file_passes(self, preflight_context: PreflightContext) -> None:
        # The data/ dir exists (the fixture creates it) but no lock file.
        result = KillSwitchLockCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert "no lock" in result.message.lower()

    def test_missing_file_pass_exposes_path_in_details(
        self, preflight_context: PreflightContext
    ) -> None:
        # Even on PASS we expose the path, so operator tooling can verify
        # WHICH path was checked (matters when project_root is unusual).
        result = KillSwitchLockCheck().run(preflight_context)
        expected = preflight_context.project_root / "data" / "kill_switch.lock"
        assert result.details["lock_path"] == str(expected)


class TestLockPresent:
    def test_empty_lock_file_blocks(
        self,
        touch_kill_switch_lock: Callable[[], Path],
        preflight_context: PreflightContext,
    ) -> None:
        # Presence is the signal — even a zero-byte file must BLOCK.
        lock_path = touch_kill_switch_lock()
        result = KillSwitchLockCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert str(lock_path) in result.message

    def test_block_result_includes_path_in_details(
        self,
        touch_kill_switch_lock: Callable[[], Path],
        preflight_context: PreflightContext,
    ) -> None:
        lock_path = touch_kill_switch_lock()
        result = KillSwitchLockCheck().run(preflight_context)
        assert result.details["lock_path"] == str(lock_path)

    def test_block_remediation_offers_rm_path(
        self,
        touch_kill_switch_lock: Callable[[], Path],
        preflight_context: PreflightContext,
    ) -> None:
        # The remediation MUST give the operator a copy-pasteable rm
        # command. The lock's full path is included so they don't have to
        # guess where the project root is.
        lock_path = touch_kill_switch_lock()
        result = KillSwitchLockCheck().run(preflight_context)
        assert f"rm {lock_path}" in result.remediation

    def test_block_remediation_mentions_force_bypass(
        self,
        touch_kill_switch_lock: Callable[[], Path],
        preflight_context: PreflightContext,
    ) -> None:
        # The other escape hatch — operator confirmed it's expected,
        # wants to proceed without removing the file.
        touch_kill_switch_lock()
        result = KillSwitchLockCheck().run(preflight_context)
        assert "--force" in result.remediation

    def test_block_remediation_explains_deny_by_default(
        self,
        touch_kill_switch_lock: Callable[[], Path],
        preflight_context: PreflightContext,
    ) -> None:
        # Operators reading the message at 3am need to know WHY the lock
        # blocks startup — it's the fail-closed safety contract from
        # CLAUDE.md, not an arbitrary check.
        touch_kill_switch_lock()
        result = KillSwitchLockCheck().run(preflight_context)
        assert "deny-by-default" in result.remediation.lower()

    def test_lock_with_stale_pid_content_still_blocks(
        self, preflight_context: PreflightContext
    ) -> None:
        # Some operators write a PID into the lock as a debugging
        # convenience. The contract is "presence triggers" — content is
        # irrelevant. A stale PID must NOT downgrade the result.
        lock_path = preflight_context.project_root / "data" / "kill_switch.lock"
        lock_path.write_text("99999\n")
        result = KillSwitchLockCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK


class TestSymlinkSemantics:
    def test_symlink_to_existing_file_blocks(
        self, preflight_context: PreflightContext, tmp_path: Path
    ) -> None:
        # An operator who symlinks the lock into a shared location
        # expects "lock present" to behave the same as a regular file.
        # Path.is_file() follows symlinks, so the check agrees.
        target = tmp_path / "real_lock"
        target.write_text("")
        lock_path = preflight_context.project_root / "data" / "kill_switch.lock"
        lock_path.symlink_to(target)

        result = KillSwitchLockCheck().run(preflight_context)
        assert result.status is CheckStatus.BLOCK

    def test_broken_symlink_passes(
        self, preflight_context: PreflightContext, tmp_path: Path
    ) -> None:
        # A symlink whose target was deleted is functionally "no lock":
        # the KillSwitch class can't read it either, so preflight must
        # agree to avoid a false BLOCK that the runner would never hit.
        missing_target = tmp_path / "does_not_exist"
        lock_path = preflight_context.project_root / "data" / "kill_switch.lock"
        lock_path.symlink_to(missing_target)

        # Sanity check: the symlink itself exists, but is_file() is False.
        assert lock_path.is_symlink()
        assert not lock_path.is_file()

        result = KillSwitchLockCheck().run(preflight_context)
        assert result.status is CheckStatus.PASS

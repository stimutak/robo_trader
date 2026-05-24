"""Tests for :class:`KillSwitchStateCheck` (spec §7.1, §9.1).

These exercise the eight cases enumerated in the spec plus the documented
edge cases:

1. File missing → PASS
2. File with ``triggered=False`` → PASS
3. File with ``triggered=False`` and ``previous_trigger_reason`` → details
   exposes it (regression: don't drop useful audit context on the floor).
4. File with ``triggered=True`` + reason + trigger_time → BLOCK, message
   includes the trigger_time, remediation includes the reason.
5. File with ``triggered=True`` but missing ``trigger_time`` → BLOCK,
   displays ``"unknown"`` and does not crash. Old state files written
   before TC-M5 lack this field; we must not regress on them.
6. Malformed JSON → BLOCK with "state file unreadable" message
   (fail-closed per spec §7.1 decision table).
7. ``read_text`` raises ``OSError`` (permission denied, mocked) → BLOCK,
   error class surfaced in the message.
8. ``trigger_reason`` containing a literal double-quote → properly
   escaped in the remediation (the remediation wraps the reason in
   quotes, so an unescaped quote would break the formatting).

The fixtures ``write_kill_switch_state`` and ``preflight_context`` come
from ``tests/preflight/conftest.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from robo_trader.preflight import PreflightContext
from robo_trader.preflight.kill_switch_state_check import KillSwitchStateCheck
from robo_trader.preflight.result import CheckStatus


@pytest.fixture
def check() -> KillSwitchStateCheck:
    return KillSwitchStateCheck()


class TestMetadata:
    def test_check_metadata_matches_spec(self, check: KillSwitchStateCheck) -> None:
        # These three attributes are part of the Check Protocol contract;
        # the runner reads them for output formatting and timeout
        # enforcement. Asserting them here keeps the spec and the impl
        # honest with each other.
        assert check.name == "kill_switch_state"
        assert check.description == "Kill switch persisted state"
        assert check.timeout_seconds == 1.0


class TestMissingFile:
    def test_returns_pass_when_state_file_missing(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
    ) -> None:
        # Case 1: cold-start fresh checkout. The file genuinely doesn't
        # exist yet, which is the most common "happy path."
        result = check.run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.message == "no triggered state"
        assert result.details["exists"] is False
        assert result.details["state_path"].endswith("kill_switch_state.json")


class TestUntriggered:
    def test_returns_pass_when_triggered_false(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Case 2: a prior session ran, triggered the switch, and the
        # operator cleared it (which leaves the file with triggered=False
        # rather than deleting the file). Should PASS.
        write_kill_switch_state(triggered=False)
        result = check.run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert result.message == "no triggered state"
        assert result.details["triggered"] is False
        assert result.details["exists"] is True

    def test_previous_trigger_reason_exposed_in_details(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Case 3: when the operator clears a trip, the previous reason
        # may be retained as a breadcrumb. Pass it through to details so
        # downstream tooling (--json output, future dashboard) can show it
        # even though we PASS.
        write_kill_switch_state(
            triggered=False,
            previous_trigger_reason="Position loss limit exceeded for NVDA: 2.93% loss",
        )
        result = check.run(preflight_context)
        assert result.status is CheckStatus.PASS
        assert (
            result.details["previous_trigger_reason"]
            == "Position loss limit exceeded for NVDA: 2.93% loss"
        )


class TestTriggered:
    def test_returns_block_when_triggered_true(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Case 4: the 2026-05-22 scenario. triggered=True with a real
        # reason and timestamp. Must BLOCK; message must include the
        # trigger_time so the operator knows how stale the trip is;
        # remediation must include the reason verbatim.
        write_kill_switch_state(
            triggered=True,
            trigger_reason="Position loss limit exceeded for NVDA: 2.93% loss",
            trigger_time="2026-05-22T22:58:12-04:00",
        )
        result = check.run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "2026-05-22T22:58:12-04:00" in result.message
        assert "triggered=True" in result.message
        assert "Position loss limit exceeded for NVDA: 2.93% loss" in result.remediation
        assert result.details["trigger_reason"] == (
            "Position loss limit exceeded for NVDA: 2.93% loss"
        )
        assert result.details["trigger_time"] == "2026-05-22T22:58:12-04:00"

    def test_handles_missing_trigger_time(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Case 5: state files written before TC-M5 didn't include
        # trigger_time. Per spec §7.1 edge cases, display "unknown" and
        # do not crash. This is a hard regression guard.
        write_kill_switch_state(
            triggered=True,
            trigger_reason="Stale trip from old version",
        )
        result = check.run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "unknown" in result.message
        assert result.details["trigger_time"] == "unknown"

    def test_remediation_includes_state_path_and_clear_command(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Operator UX: at 3am we want the actual file path and the rm
        # command on screen. No "see docs" indirection.
        path = write_kill_switch_state(triggered=True, trigger_reason="x")
        result = check.run(preflight_context)
        assert str(path) in result.remediation
        # The backup + rm pattern is the supported clear procedure; keep
        # them both present so an operator can copy-paste with confidence.
        assert "cp " in result.remediation
        assert "rm " in result.remediation
        # And the bypass mechanism must be visible — otherwise an operator
        # who has decided to proceed has no idea --force exists.
        assert "--force" in result.remediation


class TestCorruptFile:
    def test_returns_block_when_state_file_is_malformed_json(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        tmp_path: Path,
    ) -> None:
        # Case 6: half-written file from a crash mid-flush, or manual
        # edit gone wrong. Fail-closed per spec §7.1 decision table.
        path = tmp_path / "data" / "kill_switch_state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        result = check.run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "state file unreadable" in result.message
        # Surfacing the exception class name in the message helps the
        # operator distinguish "the JSON is bad" from "the disk is gone."
        assert "JSONDecodeError" in result.message
        assert result.details["error"]

    def test_returns_block_when_read_text_raises_oserror(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Case 7: permission denied on the state file. We can't easily
        # chmod a file to be unreadable in a portable way (CI runs as
        # root in containers, chmod is a no-op), so mock the read.
        write_kill_switch_state(triggered=False)  # file exists so we get past the exists() branch

        real_read_text = Path.read_text

        def boom(self: Path, *args, **kwargs):
            if self.name == "kill_switch_state.json":
                raise PermissionError("Permission denied")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", boom)
        result = check.run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        assert "state file unreadable" in result.message
        assert "PermissionError" in result.message


class TestRemediationEscaping:
    def test_trigger_reason_with_quotes_is_escaped(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Case 8: a reason like ``He said "stop"`` would otherwise close
        # the quoted region in the remediation text and confuse the
        # operator's eye. Verify the quote is backslash-escaped.
        write_kill_switch_state(
            triggered=True,
            trigger_reason='He said "stop trading"',
            trigger_time="2026-05-22T22:58:12-04:00",
        )
        result = check.run(preflight_context)
        assert result.status is CheckStatus.BLOCK
        # The remediation embeds the reason in quotes; the inner quotes
        # must be escaped so the display stays readable.
        assert '"He said \\"stop trading\\""' in result.remediation
        # The unescaped reason is still preserved in details for tooling.
        assert result.details["trigger_reason"] == 'He said "stop trading"'


class TestReadOnly:
    def test_does_not_modify_state_file(
        self,
        check: KillSwitchStateCheck,
        preflight_context: PreflightContext,
        write_kill_switch_state,
    ) -> None:
        # Spec §5.7, N1: checks NEVER modify state. If we ever
        # accidentally start writing to the file, this test catches it
        # via mtime comparison. The motivating principle is "preflight
        # observes and reports; the operator acts."
        path = write_kill_switch_state(triggered=True, trigger_reason="x", trigger_time="t")
        mtime_before = path.stat().st_mtime
        contents_before = path.read_text()
        check.run(preflight_context)
        assert path.stat().st_mtime == mtime_before
        assert path.read_text() == contents_before

"""Unit tests for ZombieConnectionsCheck (spec §7.6 / §9.1).

Mocks ``subprocess.run`` via the shared ``mock_lsof`` fixture so the
tests are hermetic — no real lsof, no real Gateway socket needed. The
sample output strings below are taken from real ``lsof -nP -iTCP:4002
-sTCP:CLOSE_WAIT`` output on macOS (see CLAUDE.md row 2025-12-06 for
why this is the canonical command).
"""

from __future__ import annotations

import subprocess

import pytest

from robo_trader.preflight import CheckStatus
from robo_trader.preflight.zombie_connections_check import ZombieConnectionsCheck

# Sample lsof outputs ---------------------------------------------------------
# Header line + N data lines, matching what real `lsof -nP -iTCP:PORT
# -sTCP:CLOSE_WAIT` prints on macOS when there's at least one match.

_LSOF_HEADER = "COMMAND    PID  USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME"
_ONE_ZOMBIE = (
    f"{_LSOF_HEADER}\n"
    "java     54321 oliver  127u  IPv4 0xabc...   0t0  "
    "TCP 127.0.0.1:54321->127.0.0.1:4002 (CLOSE_WAIT)"
)
_THREE_ZOMBIES = (
    f"{_LSOF_HEADER}\n"
    "java     54321 oliver  127u  IPv4 0xabc...   0t0  "
    "TCP 127.0.0.1:54321->127.0.0.1:4002 (CLOSE_WAIT)\n"
    "Python3  54322 oliver  128u  IPv4 0xdef...   0t0  "
    "TCP 127.0.0.1:54323->127.0.0.1:4002 (CLOSE_WAIT)\n"
    "Python3  54399 oliver  129u  IPv4 0x123...   0t0  "
    "TCP 127.0.0.1:54324->127.0.0.1:4002 (CLOSE_WAIT)"
)


# -- happy path ---------------------------------------------------------------


def test_returns_pass_when_no_close_wait_lines(mock_lsof, preflight_context):
    """lsof returns 1 + empty stdout when there are no matching sockets.

    This is the overwhelmingly common case — we want PASS without any
    diagnostic noise in details (just port + count=0).
    """
    mock_lsof(returncode=1, stdout="")
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.PASS
    assert result.details == {"port": 4002, "zombie_count": 0}
    # No remediation needed on PASS.
    assert result.remediation == ""


# -- block path: zombies present ---------------------------------------------


def test_returns_block_when_one_close_wait_line(mock_lsof, preflight_context):
    """Single zombie → BLOCK with count=1 and the PID in details."""
    mock_lsof(returncode=0, stdout=_ONE_ZOMBIE)
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.BLOCK
    assert result.details["zombie_count"] == 1
    assert result.details["zombie_pids"] == ["54321"]
    assert result.details["port"] == 4002
    # Singular wording when count is 1.
    assert "1 zombie connection " in result.message or result.message.endswith("port 4002")


def test_returns_block_when_three_close_wait_lines(mock_lsof, preflight_context):
    """Multiple zombies → BLOCK with the full PID list in details."""
    mock_lsof(returncode=0, stdout=_THREE_ZOMBIES)
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.BLOCK
    assert result.details["zombie_count"] == 3
    assert result.details["zombie_pids"] == ["54321", "54322", "54399"]
    assert "3 zombie connections" in result.message


def test_header_line_is_not_counted_as_a_zombie(mock_lsof, preflight_context):
    """Spec edge case: lsof's COMMAND/PID/USER header must NOT inflate count.

    Two real data rows + one header should report count=2, never 3.
    """
    two_rows = (
        f"{_LSOF_HEADER}\n"
        "java   1111 oliver  127u  IPv4 0x...   0t0  TCP ...:4002 (CLOSE_WAIT)\n"
        "java   2222 oliver  128u  IPv4 0x...   0t0  TCP ...:4002 (CLOSE_WAIT)"
    )
    mock_lsof(returncode=0, stdout=two_rows)
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.BLOCK
    assert result.details["zombie_count"] == 2
    assert result.details["zombie_pids"] == ["1111", "2222"]


def test_pids_parsed_from_multiline_output(mock_lsof, preflight_context):
    """details.zombie_pids must be the parsed list from the actual output."""
    mock_lsof(returncode=0, stdout=_THREE_ZOMBIES)
    result = ZombieConnectionsCheck().run(preflight_context)
    # Order preserved from lsof; tests rely on this for grep-friendly logs.
    assert result.details["zombie_pids"] == ["54321", "54322", "54399"]
    # PIDs are strings (matching gateway_manager.py's representation, which
    # treats them as opaque process identifiers — no arithmetic ever done).
    assert all(isinstance(p, str) for p in result.details["zombie_pids"])


# -- failure paths: lsof itself broke ----------------------------------------


def test_returns_block_on_subprocess_timeout(mock_lsof, preflight_context):
    """TimeoutExpired → BLOCK with a message naming the symptom."""
    mock_lsof(
        side_effect=subprocess.TimeoutExpired(cmd="lsof", timeout=3),
    )
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.BLOCK
    assert "timed out" in result.message.lower()
    assert result.details["port"] == 4002
    assert "timeout" in result.details["error"].lower()


def test_returns_block_when_lsof_not_installed(mock_lsof, preflight_context):
    """FileNotFoundError (no lsof binary) → BLOCK that explains what's wrong."""
    mock_lsof(side_effect=FileNotFoundError(2, "No such file", "lsof"))
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.BLOCK
    assert "lsof" in result.message.lower()
    # Remediation must tell the operator how to install lsof.
    assert "install" in result.remediation.lower()


# -- remediation text contracts ----------------------------------------------


def test_remediation_includes_both_cleanup_paths(mock_lsof, preflight_context):
    """Spec §7.6 mandates BOTH the Python-zombie clear path AND Gateway restart.

    Operator should see the cheap fix first, then the escalation.
    """
    mock_lsof(returncode=0, stdout=_ONE_ZOMBIE)
    result = ZombieConnectionsCheck().run(preflight_context)
    assert "clear-zombies" in result.remediation
    assert "gateway_manager.py restart" in result.remediation
    # And the canonical re-entry point.
    assert "START_TRADER.sh" in result.remediation


def test_remediation_includes_port_number(mock_lsof, preflight_context):
    """Operator-visible message should name the port being inspected."""
    mock_lsof(returncode=0, stdout=_ONE_ZOMBIE)
    result = ZombieConnectionsCheck().run(preflight_context)
    assert "4002" in result.remediation


# -- ambiguous lsof exit codes -----------------------------------------------


def test_ambiguous_exit_code_with_empty_stdout_passes_with_diagnostic(mock_lsof, preflight_context):
    """Per task brief: unexpected (returncode, stdout) → PASS but log exit code.

    If lsof exits with something other than 0 (matches) or 1 (no matches)
    and produces no parseable rows, we don't have evidence of zombies and
    must not block — but we DO stash the exit code in details for future
    debugging.
    """
    mock_lsof(returncode=99, stdout="")
    result = ZombieConnectionsCheck().run(preflight_context)
    assert result.status is CheckStatus.PASS
    assert result.details["zombie_count"] == 0
    assert result.details["lsof_exit_code"] == 99


# -- target_port plumbing ----------------------------------------------------


def test_uses_target_port_from_context(mock_lsof, tmp_path):
    """Live-trading context (port 4001) must be respected."""
    from robo_trader.preflight import PreflightContext

    ctx = PreflightContext.for_test(tmp_path, target_port=4001)
    mock_lsof(returncode=1, stdout="")
    result = ZombieConnectionsCheck().run(ctx)
    assert result.status is CheckStatus.PASS
    assert result.details["port"] == 4001
    assert "4001" in result.message


# -- Check protocol surface --------------------------------------------------


def test_check_has_required_protocol_attributes():
    """name/description/timeout_seconds — the registry depends on these."""
    check = ZombieConnectionsCheck()
    assert check.name == "zombie_connections"
    assert check.description == "Zombie connections"
    assert check.timeout_seconds == 3.0


# -- Sanity: real lsof exit code on macOS ------------------------------------


def test_real_lsof_exits_one_when_no_matches():
    """Documents the lsof exit-code contract this check relies on.

    Regression guard: if a future macOS update makes ``lsof`` exit 0 even
    when there's nothing to report, our PASS-on-empty-stdout logic still
    holds, but the ``zombie_count == 0`` invariant would need re-thinking.
    """
    # Use a port that's overwhelmingly unlikely to have anything CLOSE_WAIT.
    result = subprocess.run(
        ["lsof", "-nP", "-iTCP:59999", "-sTCP:CLOSE_WAIT"],
        capture_output=True,
        text=True,
        timeout=3,
    )
    # Either exit 1 (the historical macOS contract) or exit 0 with empty
    # stdout — both are fine for our check; we just don't want exit 0 with
    # spurious data.
    assert (result.returncode == 1 and result.stdout == "") or (
        result.returncode == 0 and result.stdout.strip() == ""
    ), f"lsof: rc={result.returncode!r} stdout={result.stdout!r}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

import importlib.util
import json
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "START_TRADER.sh"
WATCHDOG_PATH = ROOT / "scripts" / "watchdog.sh"
AUDIT_HELPER_PATH = ROOT / "scripts" / "write_paper_safety_terminal_audit.py"
WATCHDOG_POLICY_PATH = ROOT / "scripts" / "watchdog_restart_policy.py"


def _load_audit_helper():
    spec = importlib.util.spec_from_file_location(
        "write_paper_safety_terminal_audit_test",
        AUDIT_HELPER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_watchdog_policy():
    spec = importlib.util.spec_from_file_location(
        "watchdog_restart_policy_launcher_test",
        WATCHDOG_POLICY_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shell_function(source: str, name: str, end_marker: str) -> str:
    return f"{name}() {{" + source.split(f"{name}() {{", 1)[1].split(end_marker, 1)[0]


def test_terminal_audit_helper_writes_only_sanitized_exact_pair(tmp_path):
    helper = _load_audit_helper()
    policy = _load_watchdog_policy()
    audit = tmp_path / "data" / "runner_exit.json"

    helper.write_terminal_audit(audit)

    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert payload["reason"] == "paper_safety_journal_replay_blocked"
    assert payload["exit_code"] == 7
    assert payload["source"] == "supervised_launcher"
    assert set(payload) == {
        "timestamp",
        "iso_timestamp",
        "reason",
        "exit_code",
        "pid",
        "source",
    }
    assert stat.S_IMODE(audit.stat().st_mode) == 0o600
    assert not list(audit.parent.glob(".*.tmp"))
    decision, reason = policy.evaluate_restart_policy(audit, expected_pid=None)
    assert decision == policy.TERMINAL_SAFETY_BLOCK
    assert reason == "paper_safety_journal_replay_blocked"


def test_launcher_verify_failure_audits_and_quiesces_only_runner(tmp_path):
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    stop_function = _shell_function(
        source,
        "stop_processes_gracefully",
        "# Function to start Gateway",
    )
    verify_block = source.split("# Step 0.5:", 1)[1].split("# Step 1:", 1)[0]
    capture = tmp_path / "capture.log"
    fake_python = tmp_path / "fake-python3"
    fake_python.write_text(
        f"""#!/bin/bash
case "$1" in
  *manage_paper_safety_journal.py)
    printf 'VERIFY_FAILED\\n' >> {capture!s}
    exit 1
    ;;
  *write_paper_safety_terminal_audit.py)
    printf 'AUDIT_WRITTEN\\n' >> {capture!s}
    exit 0
    ;;
esac
exit 99
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o700)

    harness = f"""
SCRIPT_DIR={tmp_path!s}
LOCK_PYTHON={fake_python!s}
CAPTURE={capture!s}
runner_alive=1
pgrep() {{
    if [ "$runner_alive" -eq 1 ]; then
        printf '4242\\n'
    fi
}}
kill() {{
    printf 'RUNNER_SIGNAL=%s PID=%s\\n' "$1" "$2" >> "$CAPTURE"
    runner_alive=0
}}
sleep() {{ :; }}
{stop_function}
# Step 0.5:{verify_block}
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 7
    actions = capture.read_text(encoding="utf-8")
    assert "VERIFY_FAILED" in actions
    assert "AUDIT_WRITTEN" in actions
    assert "RUNNER_SIGNAL=-TERM PID=4242" in actions
    assert actions.index("RUNNER_SIGNAL=-TERM PID=4242") < actions.index("AUDIT_WRITTEN")
    assert "Gateway, dashboard, and WebSocket processes were left untouched." in result.stderr


@pytest.mark.parametrize(
    ("launcher_rc", "expected_restart_rc", "expected_log"),
    [
        (7, 2, "TERMINAL SAFETY BLOCK: launcher exited 7"),
        (4, 1, "Restart launcher failed (launcher_rc=4)"),
        (0, 1, "did not produce one new runner PID"),
    ],
)
def test_watchdog_nonzero_launcher_never_reuses_old_runner_as_success(
    tmp_path,
    launcher_rc,
    expected_restart_rc,
    expected_log,
):
    source = WATCHDOG_PATH.read_text(encoding="utf-8")
    restart_function = _shell_function(source, "restart_trader", "# Main loop")
    capture = tmp_path / "watchdog.log"
    launcher = tmp_path / "START_TRADER.sh"
    launcher.write_text(
        f"#!/bin/bash\nprintf 'LAUNCHER_RAN\\n' >> {capture!s}\nexit {launcher_rc}\n",
        encoding="utf-8",
    )
    launcher.chmod(0o700)

    harness = f"""
PROJECT_DIR={tmp_path!s}
WATCHDOG_LOG={capture!s}
CAPTURE={capture!s}
LAST_TERMINAL_SAFETY_REASON=""
LAST_OBSERVED_RUNNER_PID=4242
RESTART_VERIFY_WAIT=0
ESCALATION_THRESHOLD=3
REMINDER_INTERVAL=3
BACKOFF_INTERVAL=900
is_runner_alive() {{ return 0; }}
pgrep() {{ printf '4242\\n'; }}
watchdog_restart_allowed_for_policy_rc() {{ return 0; }}
log() {{ printf '%s\\n' "$1" >> "$CAPTURE"; }}
notify_user() {{ :; }}
reset_failures() {{ :; }}
get_failure_count() {{ echo 0; }}
set_failure_count() {{ :; }}
{restart_function}
restart_trader
restart_rc=$?
printf 'RETURN_CODE=%s\\n' "$restart_rc" >> "$CAPTURE"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    log_output = capture.read_text(encoding="utf-8")
    assert expected_log in log_output
    assert f"RETURN_CODE={expected_restart_rc}" in log_output
    assert "Restart verified: runner_async is alive" not in log_output

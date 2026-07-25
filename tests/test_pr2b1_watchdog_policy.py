import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "scripts" / "watchdog_restart_policy.py"
SPEC = importlib.util.spec_from_file_location("watchdog_restart_policy_pr2b1", POLICY_PATH)
assert SPEC is not None and SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


def _write_audit(path: Path, *, reason: str, exit_code: int, **extra) -> None:
    path.write_text(
        json.dumps(
            {
                "reason": reason,
                "exit_code": exit_code,
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_paper_safety_journal_replay_block_is_an_exact_terminal_exit(tmp_path):
    audit = tmp_path / "runner_exit.json"
    _write_audit(
        audit,
        reason="paper_safety_journal_replay_blocked",
        exit_code=7,
    )

    decision, reason = policy.evaluate_restart_policy(
        audit,
        expected_pid=None,
        now=10_000.0,
    )

    assert (
        "paper_safety_journal_replay_blocked",
        7,
    ) in policy.TERMINAL_EXITS
    assert decision == policy.TERMINAL_SAFETY_BLOCK
    assert reason == "paper_safety_journal_replay_blocked"


@pytest.mark.parametrize(
    ("reason", "exit_code"),
    [
        ("paper_safety_journal_replay_blocked", 6),
        ("paper_safety_journal_replay_blocked", 8),
        ("paper_safety_journal_replay_failed", 7),
    ],
)
def test_near_miss_replay_exit_pairs_fail_closed_as_unknown(tmp_path, reason, exit_code):
    audit = tmp_path / "runner_exit.json"
    _write_audit(audit, reason=reason, exit_code=exit_code)

    decision, classification = policy.evaluate_restart_policy(
        audit,
        expected_pid=1234,
        now=10_000.0,
    )

    assert decision == policy.POLICY_EVIDENCE_INVALID
    assert classification == "exit_audit_unknown_pair"

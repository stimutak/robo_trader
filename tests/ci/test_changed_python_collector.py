"""Behavioral tests for the fail-closed changed-Python collector."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts" / "ci" / "collect_changed_python.sh"
ZERO_SHA = "0" * 40


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "CI Test")
    return repo


def _collect(repo: Path, **environment: str) -> list[str]:
    output = repo / "changed-python.zlist"
    env = os.environ.copy()
    env.update(
        {
            "CHANGED_PYTHON_OUTPUT": str(output),
            "DEFAULT_BRANCH": "main",
            **environment,
        }
    )
    subprocess.run(
        ["bash", str(COLLECTOR)],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return [os.fsdecode(path) for path in output.read_bytes().split(b"\0") if path]


def test_pull_request_collects_all_branch_commits_and_preserves_names(
    git_repo: Path,
) -> None:
    (git_repo / "base.py").write_text("BASE = True\n", encoding="utf-8")
    base_sha = _commit(git_repo, "base")
    _git(git_repo, "checkout", "-b", "feature")

    for filename in ("first.py", "-leading.py", "space name.py", "line\nbreak.py"):
        (git_repo / filename).write_text("VALUE = 1\n", encoding="utf-8")
    _commit(git_repo, "first feature commit")
    (git_repo / "second.py").write_text("VALUE = 2\n", encoding="utf-8")
    head_sha = _commit(git_repo, "second feature commit")

    changed = _collect(
        git_repo,
        GITHUB_EVENT_NAME="pull_request",
        GITHUB_SHA=head_sha,
        PR_BASE_SHA=base_sha,
        PR_BASE_REF="main",
    )

    assert set(changed) == {
        "-leading.py",
        "first.py",
        "line\nbreak.py",
        "second.py",
        "space name.py",
    }


def test_push_uses_valid_before_sha(git_repo: Path) -> None:
    (git_repo / "base.py").write_text("BASE = True\n", encoding="utf-8")
    before_sha = _commit(git_repo, "base")
    (git_repo / "new.py").write_text("NEW = True\n", encoding="utf-8")
    head_sha = _commit(git_repo, "push")

    changed = _collect(
        git_repo,
        GITHUB_EVENT_NAME="push",
        GITHUB_SHA=head_sha,
        PUSH_BASE_SHA=before_sha,
    )

    assert changed == ["new.py"]


def test_all_zero_push_base_with_main_at_head_checks_complete_tree(
    git_repo: Path,
) -> None:
    (git_repo / "base.py").write_text("BASE = True\n", encoding="utf-8")
    head_sha = _commit(git_repo, "base")
    _git(git_repo, "update-ref", "refs/remotes/origin/main", head_sha)

    changed = _collect(
        git_repo,
        GITHUB_EVENT_NAME="push",
        GITHUB_SHA=head_sha,
        PUSH_BASE_SHA=ZERO_SHA,
    )

    assert changed == ["base.py"]


def test_valid_base_with_only_non_python_changes_is_honestly_empty(
    git_repo: Path,
) -> None:
    (git_repo / "base.py").write_text("BASE = True\n", encoding="utf-8")
    before_sha = _commit(git_repo, "base")
    (git_repo / "README.md").write_text("documentation\n", encoding="utf-8")
    head_sha = _commit(git_repo, "docs")

    changed = _collect(
        git_repo,
        GITHUB_EVENT_NAME="push",
        GITHUB_SHA=head_sha,
        PUSH_BASE_SHA=before_sha,
    )

    assert changed == []
